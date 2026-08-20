from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import run_mechanism_gate_exp2 as base


def fit_normal_imputer(features: np.ndarray, train_runs: np.ndarray, mean: np.ndarray, std: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    # Predict current standardized XMV from current XMEAS and previous XMV, using pre-onset normal data only.
    xtx = np.zeros((53, 53), dtype=np.float64); xty = np.zeros((53, 11), dtype=np.float64)
    for run in train_runs:
        a = np.asarray(features[int(run), :599], dtype=np.float64)
        z = (a - mean) / std
        x = np.c_[z[1:, :41], z[:-1, 41:], np.ones(len(z) - 1)]
        y = z[1:, 41:]
        xtx += x.T @ x; xty += x.T @ y
    penalty = np.eye(53) * ridge; penalty[-1, -1] = 0
    return np.linalg.solve(xtx + penalty, xty).astype(np.float32)


def score_runs(model, features, run_ids, mean, std, weights, channel, start, end, window, device, batch_size):
    rows = []
    with torch.no_grad():
        for run in run_ids:
            a = np.asarray(features[int(run)], dtype=np.float32)
            z = (a - mean) / std
            predictors = np.c_[z[1:, :41], z[:-1, 41:], np.ones(len(z) - 1, dtype=np.float32)]
            predicted = predictors @ weights
            windows = np.lib.stride_tricks.sliding_window_view(a, (window, 52))[: 2000 - window, 0]
            lo, hi = start - 1 - window, end - window
            x = (np.array(windows[lo:hi], copy=True) - mean) / std
            if channel is not None:
                # Window raw indices for target t are [t-window, t); predicted index is raw index-1.
                replacement = np.lib.stride_tricks.sliding_window_view(predicted[:, channel], window)[lo:hi]
                x[:, :, 41 + channel] = replacement
            y = z[start - 1:end, :41]
            scores = []
            for offset in range(0, len(x), batch_size):
                xb = torch.from_numpy(x[offset:offset + batch_size]).to(device)
                yb = torch.from_numpy(y[offset:offset + batch_size]).to(device)
                scores.append(torch.mean(torch.abs(model(xb) - yb), dim=1).cpu().numpy())
            rows.append(np.concatenate(scores))
    return np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/mechanism_gate_exp2.yaml")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    out = root / "outputs" / "conditional_chum_g1"; out.mkdir(parents=True, exist_ok=True)
    features = np.load((root / cfg["paths"]["cache"] / "features.npy").resolve(), mmap_mode="r")
    split = pd.read_csv((root / cfg["paths"]["split"]).resolve())
    train = split[split.split == "train"].run_index.to_numpy(); validation = split[split.split == "validation"].sort_values("run_index")
    test = split[split.split == "test"].sort_values(["fault_id", "run_index"])
    seed0 = torch.load(base.checkpoint(root, cfg, 42), map_location="cpu", weights_only=False)
    mean, std = np.asarray(seed0["mean"], np.float32), np.asarray(seed0["std"], np.float32)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("fitting normal-only conditional imputer on %d runs", len(train))
    weights = fit_normal_imputer(features, train, mean, std); np.save(out / "normal_xmv_imputer_weights.npy", weights)
    start, end, onset = cfg["evaluation_sample_start"], cfg["evaluation_sample_end"], cfg["fault_onset"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    result_rows, run_rows = [], []
    conditions = [("original", None)] + [(f"conditional_XMV_{i+1:02d}", i) for i in range(11)]
    for seed in cfg["seeds"]:
        cp = torch.load(base.checkpoint(root, cfg, seed), map_location="cpu", weights_only=False)
        model = base.GRUForecaster(52, int(cp["config"]["hidden_dim"]), int(cp["config"]["layers"]))
        model.load_state_dict(cp["state_dict"]); model.to(device).eval()
        for condition, channel in conditions:
            logging.info("seed=%d condition=%s", seed, condition)
            val = score_runs(model, features, validation.run_index, mean, std, weights, channel, start, onset - 1, cfg["window"], device, cfg["batch_size"])
            threshold = float(np.percentile(val, cfg["threshold_percentile"]))
            for fault, group in test.groupby("fault_id", sort=True):
                scores = score_runs(model, features, group.run_index, mean, std, weights, channel, start, end, cfg["window"], device, cfg["batch_size"])
                labels = np.tile(np.arange(start, end + 1) >= onset, len(group)); flat = scores.reshape(-1)
                auroc, auprc = base.roc_auc_score(labels, flat), base.average_precision_score(labels, flat)
                delays = [base.first_alarm(s, threshold, onset - start, cfg["alarm_consecutive"]) for s in scores]
                result_rows.append({"seed": seed, "condition": condition, "fault_id": int(fault), "runs": len(group), "threshold": threshold,
                                    "auroc": auroc, "auprc": auprc, "pre_fpr": float(np.mean(scores[:, :onset-start] > threshold)),
                                    "detected_ratio": float(np.mean(np.asarray(delays) <= end-onset)), "censored_delay_mean": float(np.mean(delays))})
                for meta, score, delay in zip(group.itertuples(), scores, delays):
                    run_rows.append({"seed": seed, "condition": condition, "fault_id": int(fault), "run_index": int(meta.run_index),
                                     "pre_fpr": float(np.mean(score[:onset-start] > threshold)), "alarm_delay_censored": delay})
        pd.DataFrame(result_rows).to_csv(out / "CONDITIONAL_CHUM_RESULTS.partial.csv", index=False)
        pd.DataFrame(run_rows).to_csv(out / "CONDITIONAL_CHUM_RUNS.partial.csv", index=False)
    pd.DataFrame(result_rows).to_csv(out / "CONDITIONAL_CHUM_RESULTS.csv", index=False)
    pd.DataFrame(run_rows).to_csv(out / "CONDITIONAL_CHUM_RUNS.csv", index=False)
    (out / "RUN_MANIFEST.json").write_text(json.dumps({"status":"COMPLETE", "seeds":cfg["seeds"], "conditions":[x[0] for x in conditions], "device":str(device)}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
