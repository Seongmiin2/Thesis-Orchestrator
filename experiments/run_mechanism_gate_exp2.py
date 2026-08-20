from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, output_dim: int = 41):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence):
        _, hidden = self.gru(sequence)
        return self.output(hidden[-1])


def variants() -> list[str]:
    return ["original", "permute_time", "keep_last_1", "keep_last_5", "keep_last_10"] + [f"occlude_XMV_{i:02d}" for i in range(1, 12)]


def perturb(x: np.ndarray, name: str) -> np.ndarray:
    if name == "original":
        return x
    out = x.copy()
    if name == "permute_time":
        out[:, :, 41:] = out[:, ::-1, 41:]
    elif name.startswith("keep_last_"):
        k = int(name.rsplit("_", 1)[1])
        out[:, :-k, 41:] = 0.0
    elif name.startswith("occlude_XMV_"):
        j = int(name.rsplit("_", 1)[1]) - 1
        out[:, :, 41 + j] = 0.0
    else:
        raise ValueError(name)
    return out


def checkpoint(root: Path, cfg: dict, seed: int) -> Path:
    key = "checkpoints_42_44" if seed <= 44 else "checkpoints_45_46"
    return (root / cfg["paths"][key] / f"reinartz_f1_seed_{seed}.pt").resolve()


def score_runs(model, features, run_ids, mean, std, variant, start, end, window, device, batch_size):
    rows = []
    with torch.no_grad():
        for run in run_ids:
            a = np.asarray(features[int(run)], dtype=np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(a, (window, 52))[: 2000 - window, 0]
            lo, hi = start - 1 - window, end - window
            x = (np.array(windows[lo:hi], copy=True) - mean) / std
            x = perturb(x, variant)
            y = (a[start - 1:end - 1, :41] - mean[:41]) / std[:41]
            scores = []
            for offset in range(0, len(x), batch_size):
                xb = torch.from_numpy(x[offset:offset + batch_size]).to(device)
                yb = torch.from_numpy(y[offset:offset + batch_size]).to(device)
                scores.append(torch.mean(torch.abs(model(xb) - yb), dim=1).cpu().numpy())
            rows.append(np.concatenate(scores))
    return np.stack(rows)


def first_alarm(score: np.ndarray, threshold: float, onset_offset: int, consecutive: int) -> int:
    hit = score[onset_offset:] > threshold
    found = np.flatnonzero(np.convolve(hit.astype(np.int8), np.ones(consecutive, dtype=np.int8), mode="valid") == consecutive)
    return int(found[0]) if len(found) else int(len(score) - onset_offset + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mechanism_gate_exp2.yaml")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--max-runs", type=int, default=None, help="Smoke-test only; never use for final inference.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    seeds, conditions = args.seeds or cfg["seeds"], args.variants or variants()
    out = (root / cfg["paths"]["output"]).resolve(); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    features = np.load((root / cfg["paths"]["cache"] / "features.npy").resolve(), mmap_mode="r")
    split = pd.read_csv((root / cfg["paths"]["split"]).resolve())
    start, end, onset = cfg["evaluation_sample_start"], cfg["evaluation_sample_end"], cfg["fault_onset"]
    validation = split[split.split == "validation"].sort_values("run_index")
    test = split[split.split == "test"].sort_values(["fault_id", "run_index"])
    if args.max_runs:
        validation = validation.groupby("fault_id").head(args.max_runs)
        test = test.groupby("fault_id").head(args.max_runs)
    all_rows, run_rows = [], []
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    for seed in seeds:
        cp = torch.load(checkpoint(root, cfg, seed), map_location="cpu", weights_only=False)
        mean, std = np.asarray(cp["mean"], np.float32), np.asarray(cp["std"], np.float32)
        model = GRUForecaster(52, int(cp["config"]["hidden_dim"]), int(cp["config"]["layers"]))
        model.load_state_dict(cp["state_dict"]); model.to(device).eval()
        for condition in conditions:
            logging.info("seed=%s condition=%s", seed, condition)
            val_scores = score_runs(model, features, validation.run_index, mean, std, condition, start, onset - 1, cfg["window"], device, cfg["batch_size"])
            threshold = float(np.percentile(val_scores, cfg["threshold_percentile"]))
            for fault, group in test.groupby("fault_id", sort=True):
                scores = score_runs(model, features, group.run_index, mean, std, condition, start, end, cfg["window"], device, cfg["batch_size"])
                labels = np.tile(np.arange(start, end + 1) >= onset, len(group))
                flat = scores.reshape(-1)
                pre = scores[:, : onset - start]
                delays = [first_alarm(s, threshold, onset - start, cfg["alarm_consecutive"]) for s in scores]
                detected = [d <= end - onset for d in delays]
                row = {"seed": seed, "condition": condition, "fault_id": int(fault), "runs": len(group), "threshold": threshold,
                       "auroc": roc_auc_score(labels, flat), "auprc": average_precision_score(labels, flat),
                       "pre_fpr": float(np.mean(pre > threshold)), "detected_ratio": float(np.mean(detected)),
                       "censored_delay_mean": float(np.mean(delays))}
                all_rows.append(row)
                for meta, score, delay in zip(group.itertuples(), scores, delays):
                    run_rows.append({"seed": seed, "condition": condition, "fault_id": int(fault), "run_index": int(meta.run_index),
                                     "pre_fpr": float(np.mean(score[: onset - start] > threshold)), "alarm_delay_censored": delay})
        pd.DataFrame(all_rows).to_csv(out / "MECHANISM_GATE_EXP2_RESULTS.partial.csv", index=False)
        pd.DataFrame(run_rows).to_csv(out / "MECHANISM_GATE_EXP2_RUNS.partial.csv", index=False)
    pd.DataFrame(all_rows).to_csv(out / "MECHANISM_GATE_EXP2_RESULTS.csv", index=False)
    pd.DataFrame(run_rows).to_csv(out / "MECHANISM_GATE_EXP2_RUNS.csv", index=False)
    manifest = {"status": "SMOKE_ONLY" if args.max_runs else "COMPLETE", "device": str(device), "seeds": seeds, "conditions": conditions, "max_runs": args.max_runs}
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
