from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from run_hai_external_validation import (
    GRUForecaster,
    apply_alarm_rule,
    event_metrics,
    load_data,
)


ROOT = Path(__file__).resolve().parents[1]


def residual_sequences(
    model: torch.nn.Module,
    sequences: list[np.ndarray],
    input_indices: np.ndarray,
    output_indices: np.ndarray,
    window: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for sequence in sequences:
            blocks: list[np.ndarray] = []
            for start in range(window, len(sequence), batch_size):
                targets = np.arange(start, min(start + batch_size, len(sequence)))
                x = np.stack(
                    [sequence[target - window : target, input_indices] for target in targets]
                )
                y = sequence[targets[:, None], output_indices]
                prediction = model(torch.from_numpy(x).to(device)).cpu().numpy()
                blocks.append(np.abs(prediction - y))
            rows.append(np.concatenate(blocks))
    return rows


def evaluate_policy(
    scores: list[np.ndarray],
    labels: list[np.ndarray],
    window: int,
    consecutive: int,
    thresholds: list[float],
    evaluation_starts: list[int],
) -> dict:
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    event_count = 0
    detected_sum = 0.0
    delay_sum = 0.0
    for score, label, threshold, evaluation_start in zip(
        scores, labels, thresholds, evaluation_starts
    ):
        aligned_prediction = np.zeros(len(label), dtype=bool)
        aligned_prediction[window:] = apply_alarm_rule(score, threshold, consecutive)
        start = max(window, evaluation_start)
        all_scores.append(score[start - window :])
        all_labels.append(label[start:])
        all_predictions.append(aligned_prediction[start:])
        events, detected, delay = event_metrics(label[start:], aligned_prediction[start:])
        event_count += events
        if events:
            detected_sum += detected * events
            delay_sum += delay * events
    flat_scores = np.concatenate(all_scores)
    flat_labels = np.concatenate(all_labels).astype(bool)
    flat_predictions = np.concatenate(all_predictions)
    return {
        "rows": len(flat_labels),
        "attack_rows": int(flat_labels.sum()),
        "auroc": roc_auc_score(flat_labels, flat_scores),
        "auprc": average_precision_score(flat_labels, flat_scores),
        "fpr": float(np.mean(flat_predictions[~flat_labels])),
        "event_detected_ratio": detected_sum / event_count,
        "censored_delay_mean": delay_sum / event_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hai_external_validation.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-seconds", type=int, default=3600)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    prepared = (ROOT / cfg["prepared"]).resolve()
    output = (ROOT / cfg["output"]).resolve()
    roles = json.loads((ROOT / cfg["roles_manifest"]).read_text(encoding="utf-8"))
    sensor_columns = roles["f0_active_columns"]
    f1_columns = roles["f1_active_columns"]
    train, validation, tests, _, _ = load_data(
        (ROOT / cfg["source"]).resolve(),
        prepared,
        f1_columns,
        float(cfg["train_fraction"]),
    )
    del train
    checkpoint = torch.load(
        output / "checkpoints" / f"hai_f0_seed_{args.seed}.pt",
        map_location="cpu",
        weights_only=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUForecaster(
        len(sensor_columns),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["layers"]),
        len(sensor_columns),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    indices = np.arange(len(sensor_columns), dtype=np.int64)
    validation_residuals = residual_sequences(
        model,
        validation,
        indices,
        indices,
        int(cfg["window"]),
        int(cfg["batch_size"]),
        device,
    )
    test_residuals = residual_sequences(
        model,
        [episode.features for episode in tests],
        indices,
        indices,
        int(cfg["window"]),
        int(cfg["batch_size"]),
        device,
    )
    validation_matrix = np.concatenate(validation_residuals)
    labels = [episode.labels[:, 0].astype(bool) for episode in tests]

    calibration_scale = np.quantile(validation_matrix, 0.95, axis=0)
    calibration_scale = np.maximum(calibration_scale, 1e-6)
    policies = {
        "raw_mean": (
            [residual.mean(axis=1) for residual in validation_residuals],
            [residual.mean(axis=1) for residual in test_residuals],
        ),
        "channel_calibrated_mean": (
            [(residual / calibration_scale).mean(axis=1) for residual in validation_residuals],
            [(residual / calibration_scale).mean(axis=1) for residual in test_residuals],
        ),
        "channel_calibrated_median": (
            [np.median(residual / calibration_scale, axis=1) for residual in validation_residuals],
            [np.median(residual / calibration_scale, axis=1) for residual in test_residuals],
        ),
    }
    metric_rows: list[dict] = []
    score_rows: list[dict] = []
    for policy, (validation_scores, test_scores) in policies.items():
        global_threshold = float(
            np.percentile(
                np.concatenate(validation_scores), float(cfg["threshold_percentile"])
            )
        )
        metric_rows.append(
            {
                "policy": policy,
                "calibration": "train_validation",
                "threshold": global_threshold,
                **evaluate_policy(
                    test_scores,
                    labels,
                    int(cfg["window"]),
                    int(cfg["alarm_consecutive"]),
                    [global_threshold] * len(tests),
                    [int(cfg["window"])] * len(tests),
                ),
            }
        )
        warmup_thresholds: list[float] = []
        for episode, score, label in zip(tests, test_scores, labels):
            warmup = score[: args.warmup_seconds - int(cfg["window"])]
            if label[: args.warmup_seconds].any():
                raise ValueError(f"Warm-up contains an attack in {episode.name}")
            threshold = float(np.percentile(warmup, float(cfg["threshold_percentile"])))
            warmup_thresholds.append(threshold)
            normal = score[~label[int(cfg["window"]) :]]
            attack = score[label[int(cfg["window"]) :]]
            score_rows.append(
                {
                    "policy": policy,
                    "file": episode.name,
                    "validation_threshold": global_threshold,
                    "warmup_threshold": threshold,
                    "normal_median": float(np.median(normal)),
                    "normal_q995": float(np.quantile(normal, 0.995)),
                    "attack_median": float(np.median(attack)),
                }
            )
        metric_rows.append(
            {
                "policy": policy,
                "calibration": f"fixed_{args.warmup_seconds}s_episode_warmup",
                "threshold": np.nan,
                **evaluate_policy(
                    test_scores,
                    labels,
                    int(cfg["window"]),
                    int(cfg["alarm_consecutive"]),
                    warmup_thresholds,
                    [args.warmup_seconds] * len(tests),
                ),
            }
        )

    normal_test = np.concatenate(
        [
            residual[~label[int(cfg["window"]) :]]
            for residual, label in zip(test_residuals, labels)
        ]
    )
    channel_shift = pd.DataFrame(
        {
            "point": sensor_columns,
            "validation_median_abs_error": np.median(validation_matrix, axis=0),
            "validation_q995_abs_error": np.quantile(validation_matrix, 0.995, axis=0),
            "test_normal_median_abs_error": np.median(normal_test, axis=0),
            "test_normal_q995_abs_error": np.quantile(normal_test, 0.995, axis=0),
        }
    )
    channel_shift["median_error_ratio"] = (
        channel_shift.test_normal_median_abs_error
        / np.maximum(channel_shift.validation_median_abs_error, 1e-9)
    )
    metrics = pd.DataFrame(metric_rows)
    scores = pd.DataFrame(score_rows)
    metrics.to_csv(output / "HAI_SHIFT_POLICY_DIAGNOSTIC.csv", index=False)
    scores.to_csv(output / "HAI_SHIFT_SCORE_DIAGNOSTIC.csv", index=False)
    channel_shift.sort_values("median_error_ratio", ascending=False).to_csv(
        output / "HAI_SHIFT_CHANNEL_DIAGNOSTIC.csv", index=False
    )
    report = [
        "# HAI Train-Test Shift Diagnostic",
        "",
        "This diagnostic was triggered after the first full seed produced test-normal FPR 1.0 under a validation-only threshold. It is post-observation and cannot be described as preregistered.",
        "",
        metrics.round(4).to_markdown(index=False),
        "",
        "## Largest Channel Shifts",
        "",
        channel_shift.nlargest(10, "median_error_ratio").round(4).to_markdown(index=False),
        "",
        f"A fixed {args.warmup_seconds}-second episode warm-up is label-free but uses target-domain observations. It is acceptable only as an explicitly adaptive deployment protocol, not as pure zero-shot external validation.",
    ]
    (output / "HAI_SHIFT_DIAGNOSTIC_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
