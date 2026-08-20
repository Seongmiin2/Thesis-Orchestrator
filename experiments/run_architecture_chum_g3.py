from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from conditional_imputer import predict_leave_one_channel_out, sample_residual_blocks
from run_architecture_gate_g2 import TCNForecaster, TransformerForecaster
from run_mechanism_gate_exp2 import first_alarm


ROOT = Path(__file__).resolve().parents[1]
MODEL_CLASSES = {
    "tcn": TCNForecaster,
    "transformer": TransformerForecaster,
}


def load_config(path: str) -> tuple[dict, Path]:
    config_path = (ROOT / path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return config, config_path


def resolve(path: str) -> Path:
    return (ROOT / path).resolve()


def checkpoint_path(config: dict, architecture: str, seed: int) -> Path:
    return resolve(config["paths"]["checkpoints"]) / architecture / f"reinartz_f1_seed_{seed}.pt"


def task_name(mode: str, channel: int | None) -> str:
    return "original" if mode == "original" else f"{mode}_XMV_{channel:02d}"


def build_tasks(config: dict) -> list[tuple[str, int | None]]:
    tasks: list[tuple[str, int | None]] = [("original", None)]
    for mode in config["modes"]:
        tasks.extend((str(mode), int(channel)) for channel in config["channels"])
    return tasks


def make_windows(
    array: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    loo_weights: np.ndarray | None,
    residual_bank: np.ndarray | None,
    mode: str,
    channel: int | None,
    start: int,
    end: int,
    window: int,
    rng: np.random.Generator | None = None,
    residual_block_length: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    standardized = (array - mean) / std
    windows = np.lib.stride_tricks.sliding_window_view(array, (window, 52))[
        : len(array) - window, 0
    ]
    lo, hi = start - 1 - window, end - window
    x = (np.array(windows[lo:hi], copy=True) - mean) / std

    if mode == "conditional":
        if channel is None:
            raise ValueError("Conditional replacement requires a channel")
        predictors = np.c_[
            standardized[1:, :41],
            standardized[:-1, 41:],
            np.ones(len(standardized) - 1, dtype=np.float32),
        ]
        predicted = predictors @ weights
        # Raw history index r is predicted at index r - 1 in `predicted`.
        replacement = np.lib.stride_tricks.sliding_window_view(
            predicted[:, channel - 1], window
        )[lo - 1 : hi - 1]
        x[:, :, 40 + channel] = replacement
    elif mode == "loo_sample":
        if channel is None or loo_weights is None or residual_bank is None or rng is None:
            raise ValueError("LOO sampling requires a channel, weights, residual bank, and RNG")
        predicted = predict_leave_one_channel_out(standardized, loo_weights)
        predicted[:, channel - 1] += sample_residual_blocks(
            residual_bank,
            channel - 1,
            len(predicted),
            rng,
            residual_block_length,
        )
        replacement = np.lib.stride_tricks.sliding_window_view(
            predicted[:, channel - 1], window
        )[lo - 1 : hi - 1]
        x[:, :, 40 + channel] = replacement
    elif mode == "zero":
        if channel is None:
            raise ValueError("Zero replacement requires a channel")
        x[:, :, 40 + channel] = 0.0
    elif mode != "original":
        raise ValueError(f"Unknown perturbation mode: {mode}")

    y = standardized[start - 1 : end, :41]
    if len(x) != len(y):
        raise AssertionError(f"Window/target mismatch: {len(x)} != {len(y)}")
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


def score_runs(
    model: torch.nn.Module,
    features: np.ndarray,
    run_ids: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    loo_weights: np.ndarray | None,
    residual_bank: np.ndarray | None,
    mode: str,
    channel: int | None,
    start: int,
    end: int,
    window: int,
    device: torch.device,
    batch_size: int,
    run_batch_size: int,
    conditional_sample_draws: int,
    residual_block_length: int,
) -> np.ndarray:
    score_blocks: list[np.ndarray] = []
    with torch.inference_mode():
        for run_offset in range(0, len(run_ids), run_batch_size):
            run_block = run_ids[run_offset : run_offset + run_batch_size]
            x_rows: list[np.ndarray] = []
            y_rows: list[np.ndarray] = []
            for run in run_block:
                array = np.asarray(features[int(run)], dtype=np.float32)
                draws = conditional_sample_draws if mode == "loo_sample" else 1
                for draw in range(draws):
                    rng = np.random.default_rng(
                        20260821 + int(run) * 1000 + int(channel or 0) * 10 + draw
                    )
                    x, y = make_windows(
                        array,
                        mean,
                        std,
                        weights,
                        loo_weights,
                        residual_bank,
                        mode,
                        channel,
                        start,
                        end,
                        window,
                        rng,
                        residual_block_length,
                    )
                    x_rows.append(x)
                    y_rows.append(y)
            x_block = np.concatenate(x_rows)
            y_block = np.concatenate(y_rows)
            block_scores: list[np.ndarray] = []
            for offset in range(0, len(x_block), batch_size):
                xb = torch.from_numpy(x_block[offset : offset + batch_size]).to(device)
                yb = torch.from_numpy(y_block[offset : offset + batch_size]).to(device)
                residual = model(xb) - yb
                block_scores.append(torch.mean(torch.abs(residual), dim=1).cpu().numpy())
            draws = conditional_sample_draws if mode == "loo_sample" else 1
            scores = np.concatenate(block_scores).reshape(len(run_block), draws, -1).mean(axis=1)
            score_blocks.append(scores)
    return np.concatenate(score_blocks, axis=0)


def metric_rows(
    architecture: str,
    seed: int,
    condition: str,
    mode: str,
    channel: int | None,
    threshold: float,
    scores: np.ndarray,
    test: pd.DataFrame,
    start: int,
    end: int,
    onset: int,
    consecutive: int,
) -> tuple[list[dict], list[dict]]:
    fault_rows: list[dict] = []
    run_rows: list[dict] = []
    labels_one = (np.arange(start, end + 1) >= onset).astype(np.int8)
    for fault, positions in test.groupby("fault_id", sort=True).indices.items():
        fault_scores = scores[np.asarray(positions)]
        flat_labels = np.tile(labels_one, len(fault_scores))
        flat_scores = fault_scores.reshape(-1)
        delays = [
            first_alarm(score, threshold, onset - start, consecutive)
            for score in fault_scores
        ]
        fault_rows.append(
            {
                "architecture": architecture,
                "seed": seed,
                "condition": condition,
                "mode": mode,
                "channel": channel,
                "fault_id": int(fault),
                "runs": len(fault_scores),
                "threshold": threshold,
                "auroc": roc_auc_score(flat_labels, flat_scores),
                "auprc": average_precision_score(flat_labels, flat_scores),
                "pre_fpr": float(np.mean(fault_scores[:, : onset - start] > threshold)),
                "detected_ratio": float(np.mean(np.asarray(delays) <= end - onset)),
                "censored_delay_mean": float(np.mean(delays)),
            }
        )
        group = test.iloc[np.asarray(positions)]
        for meta, score, delay in zip(group.itertuples(), fault_scores, delays):
            run_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "condition": condition,
                    "mode": mode,
                    "channel": channel,
                    "fault_id": int(fault),
                    "run_index": int(meta.run_index),
                    "auroc": roc_auc_score(labels_one, score),
                    "auprc": average_precision_score(labels_one, score),
                    "pre_fpr": float(np.mean(score[: onset - start] > threshold)),
                    "alarm_delay_censored": delay,
                }
            )
    return fault_rows, run_rows


def completed_tasks(
    fault_rows: pd.DataFrame,
    run_rows: pd.DataFrame,
    expected_fault_rows: int,
    expected_run_rows: int,
) -> set[tuple[str, int, str]]:
    if fault_rows.empty or run_rows.empty:
        return set()
    keys = ["architecture", "seed", "condition"]
    required_fault_columns = set(keys + ["fault_id"])
    required_run_columns = set(keys + ["run_index"])
    if not required_fault_columns.issubset(fault_rows) or not required_run_columns.issubset(run_rows):
        raise ValueError("Partial result schema does not match the G3 runner")
    fault_counts = fault_rows.groupby(keys).agg(rows=("fault_id", "size"), faults=("fault_id", "nunique"))
    run_counts = run_rows.groupby(keys).agg(rows=("run_index", "size"), runs=("run_index", "nunique"))
    return {
        (str(architecture), int(seed), str(condition))
        for (architecture, seed, condition), counts in fault_counts.iterrows()
        if counts["rows"] == expected_fault_rows
        and counts["faults"] == expected_fault_rows
        and (architecture, seed, condition) in run_counts.index
        and run_counts.loc[(architecture, seed, condition), "rows"] == expected_run_rows
        and run_counts.loc[(architecture, seed, condition), "runs"] == expected_run_rows
    }


def retain_completed_rows(
    frame: pd.DataFrame, done: set[tuple[str, int, str]]
) -> pd.DataFrame:
    if frame.empty or not done:
        return frame.iloc[0:0].copy()
    keep = [
        (str(row.architecture), int(row.seed), str(row.condition)) in done
        for row in frame[["architecture", "seed", "condition"]].itertuples(index=False)
    ]
    return frame.loc[keep].copy()


def save_partial(output: Path, fault_rows: list[dict], run_rows: list[dict]) -> None:
    pd.DataFrame(fault_rows).to_csv(output / "G3_FAULT_RESULTS.partial.csv", index=False)
    pd.DataFrame(run_rows).to_csv(output / "G3_RUN_RESULTS.partial.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/architecture_chum_g3.yaml")
    parser.add_argument("--architectures", nargs="+", choices=sorted(MODEL_CLASSES))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--modes", nargs="+", choices=["conditional", "loo_sample", "zero"])
    parser.add_argument("--channels", nargs="+", type=int)
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    starting_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    declared_config = json.loads(json.dumps(config))
    if args.architectures:
        config["architectures"] = args.architectures
    if args.seeds:
        config["seeds"] = args.seeds
    if args.modes:
        config["modes"] = args.modes
    if args.channels:
        config["channels"] = args.channels

    output = resolve(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "G3_RUN.log", encoding="utf-8"),
        ],
    )

    features = np.load(resolve(config["paths"]["cache"]) / "features.npy", mmap_mode="r")
    split = pd.read_csv(resolve(config["paths"]["split"]))
    validation = split[split.split == "validation"].sort_values("run_index").reset_index(drop=True)
    test = split[split.split == "test"].sort_values(["fault_id", "run_index"]).reset_index(drop=True)
    scaler = pd.read_csv(resolve(config["paths"]["scaler"]))
    mean = scaler["mean"].to_numpy(np.float32)
    std = scaler["std"].to_numpy(np.float32)
    weights = np.load(resolve(config["paths"]["imputer_weights"]))
    loo_weights = (
        np.load(resolve(config["paths"]["loo_imputer_weights"]))
        if "loo_sample" in config["modes"]
        else None
    )
    residual_bank = (
        np.load(resolve(config["paths"]["loo_residual_bank"]), mmap_mode="r")
        if "loo_sample" in config["modes"]
        else None
    )

    if features.ndim != 3 or features.shape[2] != 52:
        raise ValueError(f"Expected features with shape [runs, samples, 52], got {features.shape}")
    if len(mean) != 52 or len(std) != 52 or np.any(std <= 0):
        raise ValueError("Scaler must contain 52 features with strictly positive standard deviations")
    if weights.shape != (53, 11):
        raise ValueError(f"Expected imputer weights with shape (53, 11), got {weights.shape}")
    if loo_weights is not None and loo_weights.shape != (94, 11):
        raise ValueError(f"Expected LOO imputer weights with shape (94, 11), got {loo_weights.shape}")
    if residual_bank is not None and (
        residual_bank.ndim != 3 or residual_bank.shape[1:] != (598, 11)
    ):
        raise ValueError(
            "Expected LOO residual bank with shape [training runs, 598, 11], "
            f"got {residual_bank.shape}"
        )
    if int(config.get("conditional_sample_draws", 1)) < 1:
        raise ValueError("conditional_sample_draws must be at least one")
    if test.run_index.nunique() != len(test) or validation.run_index.nunique() != len(validation):
        raise ValueError("Split manifest contains duplicate run indices")
    expected_fault_rows = int(test.fault_id.nunique())
    expected_run_rows = len(test)

    partial_fault_path = output / "G3_FAULT_RESULTS.partial.csv"
    partial_run_path = output / "G3_RUN_RESULTS.partial.csv"
    existing_faults = pd.read_csv(partial_fault_path) if partial_fault_path.exists() else pd.DataFrame()
    existing_runs = pd.read_csv(partial_run_path) if partial_run_path.exists() else pd.DataFrame()
    done = completed_tasks(
        existing_faults, existing_runs, expected_fault_rows, expected_run_rows
    )
    # A process can be interrupted between the two partial-file writes. Drop every
    # incomplete task before resuming so it cannot be duplicated indefinitely.
    existing_faults = retain_completed_rows(existing_faults, done)
    existing_runs = retain_completed_rows(existing_runs, done)
    fault_rows = existing_faults.to_dict("records")
    run_rows = existing_runs.to_dict("records")

    start = int(config["evaluation_sample_start"])
    end = int(config["evaluation_sample_end"])
    onset = int(config["fault_onset"])
    device_name = str(config["device"])
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    tasks = build_tasks(config)
    requested_keys = {
        (str(architecture), int(seed), task_name(mode, channel))
        for architecture in config["architectures"]
        for seed in config["seeds"]
        for mode, channel in tasks
    }
    full_tasks = build_tasks(declared_config)
    full_config_keys = {
        (str(architecture), int(seed), task_name(mode, channel))
        for architecture in declared_config["architectures"]
        for seed in declared_config["seeds"]
        for mode, channel in full_tasks
    }
    total = len(requested_keys)
    completed = len(done & requested_keys)

    for architecture in config["architectures"]:
        for seed in config["seeds"]:
            cp_path = checkpoint_path(config, str(architecture), int(seed))
            payload = torch.load(cp_path, map_location="cpu", weights_only=False)
            cp_mean = np.asarray(payload["mean"], dtype=np.float32)
            cp_std = np.asarray(payload["std"], dtype=np.float32)
            if not (np.allclose(mean, cp_mean) and np.allclose(std, cp_std)):
                raise ValueError(f"Scaler mismatch in {cp_path}")
            cp_config = payload["config"]
            model = MODEL_CLASSES[str(architecture)](
                52, int(cp_config["hidden_dim"]), int(cp_config["layers"])
            )
            model.load_state_dict(payload["state_dict"])
            model.to(device).eval()

            for mode, channel in tasks:
                condition = task_name(mode, channel)
                key = (str(architecture), int(seed), condition)
                if key in done:
                    logging.info("skip completed %s seed=%s condition=%s", architecture, seed, condition)
                    continue
                completed += 1
                logging.info(
                    "task %d/%d architecture=%s seed=%s condition=%s",
                    completed,
                    total,
                    architecture,
                    seed,
                    condition,
                )
                validation_scores = score_runs(
                    model,
                    features,
                    validation.run_index.to_numpy(),
                    mean,
                    std,
                    weights,
                    loo_weights,
                    residual_bank,
                    mode,
                    channel,
                    start,
                    onset - 1,
                    int(config["window"]),
                    device,
                    int(config["batch_size"]),
                    int(config["run_batch_size"]),
                    int(config["conditional_sample_draws"]),
                    int(config["residual_block_length"]),
                )
                threshold = float(
                    np.percentile(validation_scores, float(config["threshold_percentile"]))
                )
                test_scores = score_runs(
                    model,
                    features,
                    test.run_index.to_numpy(),
                    mean,
                    std,
                    weights,
                    loo_weights,
                    residual_bank,
                    mode,
                    channel,
                    start,
                    end,
                    int(config["window"]),
                    device,
                    int(config["batch_size"]),
                    int(config["run_batch_size"]),
                    int(config["conditional_sample_draws"]),
                    int(config["residual_block_length"]),
                )
                task_faults, task_runs = metric_rows(
                    str(architecture),
                    int(seed),
                    condition,
                    mode,
                    channel,
                    threshold,
                    test_scores,
                    test,
                    start,
                    end,
                    onset,
                    int(config["alarm_consecutive"]),
                )
                fault_rows.extend(task_faults)
                run_rows.extend(task_runs)
                done.add(key)
                save_partial(output, fault_rows, run_rows)

    fault_frame = pd.DataFrame(fault_rows).sort_values(
        ["architecture", "seed", "mode", "channel", "fault_id"], na_position="first"
    )
    run_frame = pd.DataFrame(run_rows).sort_values(
        ["architecture", "seed", "mode", "channel", "fault_id", "run_index"],
        na_position="first",
    )
    fault_frame.to_csv(output / "G3_FAULT_RESULTS.csv", index=False)
    run_frame.to_csv(output / "G3_RUN_RESULTS.csv", index=False)
    final_done = completed_tasks(
        fault_frame, run_frame, expected_fault_rows, expected_run_rows
    )
    ending_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config_changed_during_run = ending_config_hash != starting_config_hash
    manifest = {
        "status": (
            "CONFIG_CHANGED_DURING_RUN"
            if config_changed_during_run
            else (
                "COMPLETE_FULL_CONFIG"
                if full_config_keys.issubset(final_done)
                else "COMPLETE_REQUESTED_SUBSET"
            )
        ),
        "config": str(config_path),
        "config_sha256": starting_config_hash,
        "config_sha256_at_end": ending_config_hash,
        "config_changed_during_run": config_changed_during_run,
        "architectures": config["architectures"],
        "seeds": config["seeds"],
        "modes": config["modes"],
        "channels": config["channels"],
        "device": str(device),
        "fault_rows": len(fault_frame),
        "run_rows": len(run_frame),
        "requested_tasks": len(requested_keys),
        "completed_requested_tasks": len(requested_keys & final_done),
        "full_config_tasks": len(full_config_keys),
        "completed_full_config_tasks": len(full_config_keys & final_done),
        "expected_fault_rows_per_task": expected_fault_rows,
        "expected_run_rows_per_task": expected_run_rows,
    }
    (output / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
