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

from hai_conditional_imputer import (
    predict_leave_one_channel_out,
    replace_control_channel,
)
from run_hai_external_validation import (
    GRUForecaster,
    LABEL_COLUMNS,
    TestEpisode,
    apply_alarm_rule,
    etapr_metrics,
    event_metrics,
    load_data,
)


ROOT = Path(__file__).resolve().parents[1]


def score_sequences_fast(
    model: torch.nn.Module,
    sequences: list[np.ndarray],
    input_indices: np.ndarray,
    output_indices: np.ndarray,
    window: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    use_amp = device.type == "cuda"
    results: list[np.ndarray] = []
    with torch.inference_mode():
        for sequence in sequences:
            windows = np.lib.stride_tricks.sliding_window_view(
                sequence, (window, sequence.shape[1])
            )[: len(sequence) - window, 0]
            blocks: list[np.ndarray] = []
            for offset in range(0, len(windows), batch_size):
                end = min(offset + batch_size, len(windows))
                x = np.ascontiguousarray(windows[offset:end, :, input_indices])
                y = np.ascontiguousarray(
                    sequence[window + offset : window + end, output_indices]
                )
                xb = torch.from_numpy(x).to(device, non_blocking=True)
                yb = torch.from_numpy(y).to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    scores = torch.mean(torch.abs(model(xb) - yb), dim=1)
                blocks.append(scores.float().cpu().numpy())
            results.append(np.concatenate(blocks))
    return results


def perturbed_scores(
    model: torch.nn.Module,
    sequences: list[np.ndarray],
    predictions: list[np.ndarray],
    sensor_count: int,
    channel: int,
    mode: str,
    residual_bank: np.ndarray,
    residual_segments: np.ndarray,
    draws: int,
    block_length: int,
    window: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    task_draws = draws if mode == "loo_sample" else 1
    accumulated: list[np.ndarray] | None = None
    for draw in range(task_draws):
        replaced: list[np.ndarray] = []
        for sequence_index, (sequence, predicted) in enumerate(
            zip(sequences, predictions)
        ):
            rng = np.random.default_rng(
                20260821 + channel * 10_000 + sequence_index * 100 + draw
            )
            replaced.append(
                replace_control_channel(
                    sequence,
                    predicted,
                    sensor_count,
                    channel,
                    mode,
                    residual_bank,
                    residual_segments,
                    rng,
                    block_length,
                )
            )
        scores = score_sequences_fast(
            model,
            replaced,
            np.arange(replaced[0].shape[1], dtype=np.int64),
            np.arange(sensor_count, dtype=np.int64),
            window,
            batch_size,
            device,
        )
        if accumulated is None:
            accumulated = [np.asarray(score, dtype=np.float64) for score in scores]
        else:
            for current, score in zip(accumulated, scores):
                current += score
    if accumulated is None:
        raise RuntimeError("No perturbation scores were produced")
    return [(score / task_draws).astype(np.float32) for score in accumulated]


def evaluate_scores(
    validation_scores: list[np.ndarray],
    test_scores: list[np.ndarray],
    test_episodes: list[TestEpisode],
    attack_targets: pd.DataFrame,
    feature: str,
    cfg: dict,
    etapr_repository: Path,
) -> tuple[dict, list[dict]]:
    window = int(cfg["window"])
    threshold = float(
        np.percentile(
            np.concatenate(validation_scores), float(cfg["threshold_percentile"])
        )
    )
    predictions: list[np.ndarray] = []
    aligned_scores: list[np.ndarray] = []
    for episode, scores in zip(test_episodes, test_scores):
        aligned = np.full(len(episode.features), np.nan, dtype=np.float32)
        aligned[window:] = scores
        prediction = np.zeros(len(episode.features), dtype=bool)
        prediction[window:] = apply_alarm_rule(
            scores, threshold, int(cfg["alarm_consecutive"])
        )
        aligned_scores.append(aligned)
        predictions.append(prediction)

    labels = [episode.labels[:, 0].astype(bool) for episode in test_episodes]
    valid_labels = np.concatenate([label[window:] for label in labels])
    valid_scores = np.concatenate([score[window:] for score in aligned_scores])
    valid_predictions = np.concatenate(
        [prediction[window:] for prediction in predictions]
    )
    events = 0
    detected_total = 0.0
    delay_total = 0.0
    for label, prediction in zip(labels, predictions):
        count, detected_ratio, delay = event_metrics(label, prediction)
        events += count
        if count:
            detected_total += detected_ratio * count
            delay_total += delay * count
    official = etapr_metrics(
        labels,
        predictions,
        etapr_repository,
        float(cfg.get("etapr_theta_p", 0.7)),
        float(cfg.get("etapr_theta_r", 0.1)),
        float(cfg.get("etapr_delta", 0.0)),
    )
    metric = {
        "threshold": threshold,
        "rows": len(valid_labels),
        "attack_rows": int(valid_labels.sum()),
        "events": events,
        "auroc": float(roc_auc_score(valid_labels, valid_scores)),
        "auprc": float(average_precision_score(valid_labels, valid_scores)),
        "fpr": float(np.mean(valid_predictions[~valid_labels])),
        "event_detected_ratio": detected_total / events,
        "censored_delay_mean": delay_total / events,
        **official,
    }

    event_rows: list[dict] = []
    for episode, scores, prediction in zip(
        test_episodes, aligned_scores, predictions
    ):
        label = episode.labels[:, 0].astype(np.int8)
        padded = np.r_[0, label, 0]
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1) - 1
        metadata = attack_targets.loc[attack_targets.file == episode.name].set_index(
            "event_index"
        )
        if len(metadata) != len(starts):
            raise ValueError(f"Attack-target count mismatch in {episode.name}")
        for event_index, (start, end) in enumerate(zip(starts, ends), start=1):
            meta = metadata.loc[event_index]
            hits = np.flatnonzero(prediction[start : end + 1])
            targets = set(str(meta.target_points).split(";"))
            mean_score = float(np.nanmean(scores[start : end + 1]))
            max_score = float(np.nanmax(scores[start : end + 1]))
            event_rows.append(
                {
                    "file": episode.name,
                    "event_index": event_index,
                    "global_event": int(meta.global_event),
                    "attack_id": meta.id,
                    "target_points": meta.target_points,
                    "target_class": meta.target_class,
                    "targeted_channel": feature in targets,
                    "duration_seconds": int(end - start + 1),
                    "detected": bool(len(hits)),
                    "alarm_delay_censored": int(hits[0])
                    if len(hits)
                    else int(end - start + 1),
                    "mean_event_score": mean_score,
                    "max_event_score": max_score,
                    "normalized_mean_event_score": mean_score / threshold,
                    "normalized_max_event_score": max_score / threshold,
                }
            )
    return metric, event_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hai_conditional_chum.yaml")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--modes", nargs="+", choices=["loo_sample", "zero"])
    parser.add_argument("--channels", nargs="+", type=int)
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = (ROOT / cfg["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    imputer_manifest = json.loads(
        (output / "HAI_IMPUTER_MANIFEST.json").read_text(encoding="utf-8")
    )
    if imputer_manifest["assessment"] != "PASS_TO_HAI_CONDITIONAL_CHUM":
        raise RuntimeError("HAI conditional imputer quality gate has not passed")
    role_manifest = json.loads(
        (ROOT / cfg["roles_manifest"]).read_text(encoding="utf-8")
    )
    sensors = role_manifest["f0_active_columns"]
    features = role_manifest["f1_active_columns"]
    controls = features[len(sensors) :]
    reliable = set(imputer_manifest["reliable_channels"])
    requested_channels = args.channels or (
        list(range(1, len(controls) + 1))
        if cfg["channels"] == "all"
        else [int(channel) for channel in cfg["channels"]]
    )
    if any(channel < 1 or channel > len(controls) for channel in requested_channels):
        raise ValueError("Requested HAI control channel is out of range")
    seeds = args.seeds or [int(seed) for seed in cfg["seeds"]]
    modes = args.modes or [str(mode) for mode in cfg["modes"]]

    specification = {
        "config_sha256": config_hash,
        "seeds": seeds,
        "modes": modes,
        "channels": requested_channels,
        "features": features,
        "reliable_channels": sorted(reliable),
    }
    specification_path = output / "HAI_CONDITIONAL_RUN_SPEC.json"
    if specification_path.exists():
        existing_specification = json.loads(
            specification_path.read_text(encoding="utf-8")
        )
        if existing_specification != specification:
            raise RuntimeError("Run specification changed while partial results exist")
    else:
        specification_path.write_text(
            json.dumps(specification, indent=2), encoding="utf-8"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(output / "HAI_CONDITIONAL_CHUM.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    train, validation, test_episodes, mean, std = load_data(
        (ROOT / cfg["source"]).resolve(),
        (ROOT / cfg["prepared"]).resolve(),
        features,
        float(cfg["train_fraction"]),
    )
    del train
    external_output = (ROOT / cfg["external_output"]).resolve()
    scaler = pd.read_csv(external_output / "HAI_EXTERNAL_SCALER.csv")
    if scaler.feature.tolist() != features:
        raise ValueError("External scaler feature order does not match role manifest")
    if not np.allclose(scaler["mean"], mean) or not np.allclose(scaler["std"], std):
        raise ValueError("External scaler statistics do not match current data load")
    attack_targets = pd.read_csv((ROOT / cfg["attack_targets"]).resolve())
    residual_bank = np.load(output / "HAI_LOO_RESIDUAL_BANK.npy", mmap_mode="r")
    residual_segments = np.load(output / "HAI_LOO_RESIDUAL_SEGMENTS.npy")
    weights = np.load(output / "HAI_LOO_IMPUTER_WEIGHTS.npy")
    expected_shape = (len(sensors) * 2 + len(controls) + 1, len(controls))
    if weights.shape != expected_shape:
        raise ValueError(f"Unexpected imputer weight shape: {weights.shape}")
    validation_predictions = [
        predict_leave_one_channel_out(sequence, len(sensors), weights)
        for sequence in validation
    ]
    test_predictions = [
        predict_leave_one_channel_out(episode.features, len(sensors), weights)
        for episode in test_episodes
    ]
    device = torch.device(
        cfg["device"] if cfg["device"] == "cpu" or torch.cuda.is_available() else "cpu"
    )

    metric_path = output / "HAI_CONDITIONAL_METRICS.partial.csv"
    event_path = output / "HAI_CONDITIONAL_EVENTS.partial.csv"
    metrics = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
    events = pd.read_csv(event_path) if event_path.exists() else pd.DataFrame()
    completed: set[tuple[int, str, int]] = set()
    if len(metrics) and len(events):
        for task, group in events.groupby(["seed", "mode", "channel"]):
            metric_group = metrics.loc[
                (metrics.seed == task[0])
                & (metrics["mode"] == task[1])
                & (metrics.channel == task[2])
            ]
            if len(group) == len(attack_targets) and len(metric_group) == 1:
                completed.add((int(task[0]), str(task[1]), int(task[2])))
    if completed:
        metrics = metrics.loc[
            [
                (int(row.seed), str(row.mode), int(row.channel)) in completed
                for row in metrics.itertuples(index=False)
            ]
        ]
        events = events.loc[
            [
                (int(row.seed), str(row.mode), int(row.channel)) in completed
                for row in events.itertuples(index=False)
            ]
        ]
    else:
        metrics = pd.DataFrame()
        events = pd.DataFrame()
    metric_rows = metrics.to_dict("records")
    event_rows = events.to_dict("records")
    external_metrics = pd.read_csv(external_output / "HAI_EXTERNAL_METRICS.csv")

    for seed in seeds:
        checkpoint_path = external_output / "checkpoints" / f"hai_f1_seed_{seed}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint["input_columns"] != features
            or checkpoint["target_columns"] != sensors
            or not np.allclose(checkpoint["mean"], mean)
            or not np.allclose(checkpoint["std"], std)
        ):
            raise ValueError(f"Checkpoint metadata mismatch for seed {seed}")
        model = GRUForecaster(
            len(features),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["layers"]),
            len(sensors),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        tasks = [("original", 0)] + [
            (mode, channel) for mode in modes for channel in requested_channels
        ]
        for mode, channel_number in tasks:
            task = (int(seed), mode, int(channel_number))
            if task in completed:
                logging.info(
                    "skip completed seed=%d mode=%s channel=%d",
                    seed,
                    mode,
                    channel_number,
                )
                continue
            feature = "original" if mode == "original" else controls[channel_number - 1]
            logging.info(
                "seed=%d mode=%s channel=%d feature=%s reliable=%s",
                seed,
                mode,
                channel_number,
                feature,
                feature in reliable if mode != "original" else True,
            )
            if mode == "original":
                validation_scores = score_sequences_fast(
                    model,
                    validation,
                    np.arange(len(features), dtype=np.int64),
                    np.arange(len(sensors), dtype=np.int64),
                    int(cfg["window"]),
                    int(cfg["batch_size"]),
                    device,
                )
                test_scores = score_sequences_fast(
                    model,
                    [episode.features for episode in test_episodes],
                    np.arange(len(features), dtype=np.int64),
                    np.arange(len(sensors), dtype=np.int64),
                    int(cfg["window"]),
                    int(cfg["batch_size"]),
                    device,
                )
            else:
                validation_scores = perturbed_scores(
                    model,
                    validation,
                    validation_predictions,
                    len(sensors),
                    channel_number - 1,
                    mode,
                    residual_bank,
                    residual_segments,
                    int(cfg["conditional_sample_draws"]),
                    int(cfg["residual_block_length"]),
                    int(cfg["window"]),
                    int(cfg["batch_size"]),
                    device,
                )
                test_scores = perturbed_scores(
                    model,
                    [episode.features for episode in test_episodes],
                    test_predictions,
                    len(sensors),
                    channel_number - 1,
                    mode,
                    residual_bank,
                    residual_segments,
                    int(cfg["conditional_sample_draws"]),
                    int(cfg["residual_block_length"]),
                    int(cfg["window"]),
                    int(cfg["batch_size"]),
                    device,
                )
            metric, task_events = evaluate_scores(
                validation_scores,
                test_scores,
                test_episodes,
                attack_targets,
                feature,
                cfg,
                (ROOT / cfg["etapr_repository"]).resolve(),
            )
            if mode == "original":
                reference = external_metrics.loc[
                    (external_metrics.seed == seed)
                    & (external_metrics.variant == "F1")
                    & (external_metrics.label == "attack")
                ].iloc[0]
                for key in ["threshold", "auroc", "auprc", "fpr", "etaf1"]:
                    if not np.isclose(metric[key], float(reference[key]), atol=1e-6):
                        raise RuntimeError(
                            f"Original-score parity failed for seed {seed}, metric {key}"
                        )
            reliable_channel = mode == "original" or feature in reliable
            metric_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "channel": channel_number,
                    "feature": feature,
                    "reliable_imputer": reliable_channel,
                    **metric,
                }
            )
            event_rows.extend(
                {
                    "seed": seed,
                    "mode": mode,
                    "channel": channel_number,
                    "feature": feature,
                    "reliable_imputer": reliable_channel,
                    **row,
                }
                for row in task_events
            )
            pd.DataFrame(metric_rows).to_csv(metric_path, index=False)
            pd.DataFrame(event_rows).to_csv(event_path, index=False)

    metric_frame = pd.DataFrame(metric_rows)
    event_frame = pd.DataFrame(event_rows)
    expected_tasks = len(seeds) * (1 + len(modes) * len(requested_channels))
    if (
        len(metric_frame) != expected_tasks
        or len(event_frame) != expected_tasks * len(attack_targets)
        or metric_frame.duplicated(["seed", "mode", "channel"]).any()
        or event_frame.duplicated(
            ["seed", "mode", "channel", "global_event"]
        ).any()
    ):
        raise RuntimeError("HAI conditional result tables are incomplete or duplicated")
    metric_frame.to_csv(output / "HAI_CONDITIONAL_METRICS.csv", index=False)
    event_frame.to_csv(output / "HAI_CONDITIONAL_EVENTS.csv", index=False)
    manifest = {
        "status": "COMPLETE",
        "device": str(device),
        "config_sha256": config_hash,
        "seeds": seeds,
        "modes": modes,
        "channels": requested_channels,
        "reliable_channel_count": len(reliable),
        "reliable_channels": sorted(reliable),
        "tasks": expected_tasks,
        "metric_rows": len(metric_frame),
        "event_rows": len(event_frame),
        "attack_events": len(attack_targets),
        "conditional_sample_draws": int(cfg["conditional_sample_draws"]),
        "residual_block_length": int(cfg["residual_block_length"]),
        "external_run": str(external_output),
    }
    (output / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
