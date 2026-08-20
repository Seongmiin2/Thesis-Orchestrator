from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from captum.attr import IntegratedGradients

from run_architecture_gate_g2 import TCNForecaster, TransformerForecaster


ROOT = Path(__file__).resolve().parents[1]
MODEL_CLASSES = {"tcn": TCNForecaster, "transformer": TransformerForecaster}


class AnomalyScore(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(self.model(x) - target), dim=1)


def build_examples(
    features: np.ndarray,
    run_ids: np.ndarray,
    target_samples: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for run in run_ids:
        array = np.asarray(features[int(run)], dtype=np.float32)
        standardized = (array - mean) / std
        for sample in target_samples:
            target_index = int(sample) - 1
            if target_index - window < 0 or target_index >= len(array):
                raise ValueError(f"Target sample {sample} is outside run {run}")
            x_rows.append(standardized[target_index - window : target_index])
            y_rows.append(standardized[target_index, :41])
    return np.stack(x_rows).astype(np.float32), np.stack(y_rows).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/integrated_gradients_baseline.yaml")
    parser.add_argument("--architectures", nargs="+", choices=sorted(MODEL_CLASSES))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--faults", nargs="+", type=int)
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    architectures = args.architectures or cfg["architectures"]
    seeds = args.seeds or cfg["seeds"]
    faults = args.faults or cfg["faults"]
    output = (ROOT / cfg["paths"]["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "INTEGRATED_GRADIENTS.log", encoding="utf-8"),
        ],
    )

    features = np.load((ROOT / cfg["paths"]["cache"] / "features.npy").resolve(), mmap_mode="r")
    split = pd.read_csv((ROOT / cfg["paths"]["split"]).resolve())
    test = split.loc[split.split == "test"].sort_values(["fault_id", "run_index"])
    scaler = pd.read_csv((ROOT / cfg["paths"]["scaler"]).resolve())
    mean = scaler["mean"].to_numpy(np.float32)
    std = scaler["std"].to_numpy(np.float32)
    if len(mean) != 52 or np.any(std <= 0):
        raise ValueError("Expected a valid 52-channel scaler")
    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    partial_path = output / "INTEGRATED_GRADIENTS.partial.csv"
    existing = pd.read_csv(partial_path) if partial_path.exists() else pd.DataFrame()
    completed: set[tuple[str, int, int]] = set()
    if len(existing):
        for key, group in existing.groupby(["architecture", "seed", "fault_id"]):
            if len(group) == 11 and group.channel.nunique() == 11:
                completed.add((str(key[0]), int(key[1]), int(key[2])))
        existing = existing.loc[
            [
                (str(row.architecture), int(row.seed), int(row.fault_id)) in completed
                for row in existing.itertuples(index=False)
            ]
        ]
    rows = existing.to_dict("records") if len(existing) else []
    for architecture in architectures:
        for seed in seeds:
            checkpoint = (
                ROOT
                / cfg["paths"]["checkpoints"]
                / architecture
                / f"reinartz_f1_seed_{seed}.pt"
            ).resolve()
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not np.array_equal(np.asarray(payload["mean"], np.float32), mean):
                raise ValueError(f"Scaler mean mismatch in {checkpoint}")
            if not np.array_equal(np.asarray(payload["std"], np.float32), std):
                raise ValueError(f"Scaler std mismatch in {checkpoint}")
            cp_config = payload["config"]
            model = MODEL_CLASSES[architecture](
                52, int(cp_config["hidden_dim"]), int(cp_config["layers"])
            )
            model.load_state_dict(payload["state_dict"])
            model.to(device).eval()
            score_model = AnomalyScore(model)
            integrated_gradients = IntegratedGradients(score_model)
            for fault in faults:
                key = (architecture, int(seed), int(fault))
                if key in completed:
                    logging.info("skip completed architecture=%s seed=%d fault=%d", *key)
                    continue
                run_ids = test.loc[test.fault_id == fault, "run_index"].to_numpy()
                if len(run_ids) != 20:
                    raise ValueError(f"Fault {fault} has {len(run_ids)} test runs, expected 20")
                x, y = build_examples(
                    features,
                    run_ids,
                    [int(sample) for sample in cfg["target_samples"]],
                    mean,
                    std,
                    int(cfg["window"]),
                )
                xb = torch.from_numpy(x).to(device)
                yb = torch.from_numpy(y).to(device)
                logging.info(
                    "architecture=%s seed=%d fault=%d examples=%d",
                    architecture,
                    seed,
                    fault,
                    len(x),
                )
                attribution, convergence_delta = integrated_gradients.attribute(
                    xb,
                    baselines=torch.zeros_like(xb),
                    additional_forward_args=(yb,),
                    n_steps=int(cfg["integration_steps"]),
                    method="gausslegendre",
                    internal_batch_size=int(cfg["internal_batch_size"]),
                    return_convergence_delta=True,
                )
                absolute = attribution.detach().abs().mean(dim=(0, 1)).cpu().numpy()
                xmv = absolute[41:]
                normalized = xmv / xmv.sum() if xmv.sum() > 0 else np.zeros_like(xmv)
                ranks = pd.Series(-normalized).rank(method="min").to_numpy(dtype=int)
                original_score = score_model(xb, yb).detach().mean().cpu().item()
                convergence = convergence_delta.detach().abs().mean().cpu().item()
                for channel in range(1, 12):
                    rows.append(
                        {
                            "architecture": architecture,
                            "seed": int(seed),
                            "fault_id": int(fault),
                            "channel": channel,
                            "examples": len(x),
                            "target_samples": ",".join(map(str, cfg["target_samples"])),
                            "integration_steps": int(cfg["integration_steps"]),
                            "mean_abs_ig": float(xmv[channel - 1]),
                            "normalized_xmv_ig": float(normalized[channel - 1]),
                            "xmv_rank": int(ranks[channel - 1]),
                            "mean_anomaly_score": float(original_score),
                            "mean_abs_convergence_delta": float(convergence),
                        }
                    )
                completed.add(key)
                pd.DataFrame(rows).to_csv(partial_path, index=False)

    result = pd.DataFrame(rows).sort_values(["architecture", "seed", "fault_id", "channel"])
    result.to_csv(output / "INTEGRATED_GRADIENTS.csv", index=False)
    expected_tasks = len(architectures) * len(seeds) * len(faults)
    actual_tasks = result[["architecture", "seed", "fault_id"]].drop_duplicates().shape[0]
    manifest = {
        "status": "COMPLETE" if actual_tasks == expected_tasks else "INCOMPLETE",
        "architectures": architectures,
        "seeds": seeds,
        "faults": faults,
        "target_samples": cfg["target_samples"],
        "integration_steps": cfg["integration_steps"],
        "baseline": "zero in training-standardized space (normal mean)",
        "tasks": actual_tasks,
        "expected_tasks": expected_tasks,
        "rows": len(result),
    }
    (output / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
