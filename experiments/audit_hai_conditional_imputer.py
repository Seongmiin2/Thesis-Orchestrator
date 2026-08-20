from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wasserstein_distance

from hai_conditional_imputer import (
    build_residual_bank,
    fit_leave_one_channel_out,
    predict_leave_one_channel_out,
    sample_residual_blocks,
)
from run_hai_external_validation import load_data


ROOT = Path(__file__).resolve().parents[1]


def lag1(values: list[np.ndarray]) -> float:
    left = np.concatenate([value[:-1] for value in values])
    right = np.concatenate([value[1:] for value in values])
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hai_conditional_chum.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    output = (ROOT / cfg["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    role_manifest = json.loads(
        (ROOT / cfg["roles_manifest"]).read_text(encoding="utf-8")
    )
    sensors = role_manifest["f0_active_columns"]
    features = role_manifest["f1_active_columns"]
    if features[: len(sensors)] != sensors:
        raise ValueError("Role manifest does not preserve sensor-first feature order")
    train, validation, _, _, _ = load_data(
        (ROOT / cfg["source"]).resolve(),
        (ROOT / cfg["prepared"]).resolve(),
        features,
        float(cfg["train_fraction"]),
    )
    sensor_count = len(sensors)
    control_names = features[sensor_count:]
    weights = fit_leave_one_channel_out(
        train, sensor_count, float(cfg["ridge"])
    )
    residual_bank, residual_segments = build_residual_bank(
        train, sensor_count, weights
    )
    np.save(output / "HAI_LOO_IMPUTER_WEIGHTS.npy", weights)
    np.save(output / "HAI_LOO_RESIDUAL_BANK.npy", residual_bank)
    np.save(output / "HAI_LOO_RESIDUAL_SEGMENTS.npy", residual_segments)

    predicted_sequences = [
        predict_leave_one_channel_out(sequence, sensor_count, weights)
        for sequence in validation
    ]
    actual_all = np.concatenate(
        [sequence[1:, sensor_count:] for sequence in validation]
    )
    predicted_all = np.concatenate(predicted_sequences)
    train_controls = np.concatenate(
        [sequence[:, sensor_count:] for sequence in train]
    )
    quality = cfg["imputer_quality"]
    rows: list[dict] = []
    for channel, name in enumerate(control_names):
        sampled_sequences: list[np.ndarray] = []
        for sequence_index, predicted in enumerate(predicted_sequences):
            rng = np.random.default_rng(20260821 + channel * 100 + sequence_index)
            residual = sample_residual_blocks(
                residual_bank,
                residual_segments,
                channel,
                len(predicted),
                rng,
                int(cfg["residual_block_length"]),
            )
            sampled_sequences.append(predicted[:, channel] + residual)
        sampled = np.concatenate(sampled_sequences)
        actual = actual_all[:, channel]
        predicted = predicted_all[:, channel]
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        r2 = (
            float(1.0 - np.sum((actual - predicted) ** 2) / denominator)
            if denominator > 1e-12
            else float("nan")
        )
        persistence = np.concatenate(
            [sequence[:-1, sensor_count + channel] for sequence in validation]
        )
        persistence_r2 = (
            float(1.0 - np.sum((actual - persistence) ** 2) / denominator)
            if denominator > 1e-12
            else float("nan")
        )
        actual_sd = float(np.std(actual))
        sampled_mean_shift_sd = (
            abs(float(sampled.mean() - actual.mean())) / actual_sd
            if actual_sd > 1e-12
            else float("inf")
        )
        sampled_sd_ratio = (
            float(np.std(sampled) / actual_sd) if actual_sd > 1e-12 else float("nan")
        )
        actual_lag1 = lag1(
            [sequence[1:, sensor_count + channel] for sequence in validation]
        )
        sampled_lag1 = lag1(sampled_sequences)
        lag1_error = (
            abs(sampled_lag1 - actual_lag1)
            if np.isfinite(actual_lag1) and np.isfinite(sampled_lag1)
            else float("inf")
        )
        lower = float(train_controls[:, channel].min())
        upper = float(train_controls[:, channel].max())
        outside_fraction = float(np.mean((sampled < lower) | (sampled > upper)))
        wasserstein_sd = (
            float(wasserstein_distance(actual, sampled) / actual_sd)
            if actual_sd > 1e-12
            else float("inf")
        )
        reliable = bool(
            np.isfinite(r2)
            and r2 >= float(quality["minimum_r2"])
            and float(quality["minimum_sampled_sd_ratio"])
            <= sampled_sd_ratio
            <= float(quality["maximum_sampled_sd_ratio"])
            and sampled_mean_shift_sd
            <= float(quality["maximum_absolute_mean_shift_sd"])
            and lag1_error <= float(quality["maximum_lag1_error"])
            and outside_fraction
            <= float(quality["maximum_outside_train_range_fraction"])
        )
        rows.append(
            {
                "channel": channel + 1,
                "feature": name,
                "r2": r2,
                "persistence_r2": persistence_r2,
                "sampled_mean_shift_sd": sampled_mean_shift_sd,
                "sampled_sd_ratio": sampled_sd_ratio,
                "actual_lag1": actual_lag1,
                "sampled_lag1": sampled_lag1,
                "lag1_error": lag1_error,
                "outside_train_range_fraction": outside_fraction,
                "wasserstein_sd": wasserstein_sd,
                "reliable": reliable,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "HAI_IMPUTER_QUALITY.csv", index=False)
    reliable = frame.loc[frame.reliable, "feature"].tolist()
    assessment = (
        "PASS_TO_HAI_CONDITIONAL_CHUM"
        if len(reliable) >= int(quality["minimum_reliable_channels"])
        else "BLOCKED_IMPUTER_QUALITY"
    )
    manifest = {
        "status": "COMPLETE",
        "assessment": assessment,
        "sensor_count": sensor_count,
        "control_count": len(control_names),
        "train_rows": int(sum(len(sequence) for sequence in train)),
        "validation_rows": int(sum(len(sequence) for sequence in validation)),
        "predictor_dimension": int(weights.shape[0]),
        "residual_rows": int(len(residual_bank)),
        "reliable_channels": reliable,
        "reliable_channel_count": len(reliable),
        "quality_thresholds": quality,
    }
    (output / "HAI_IMPUTER_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(4)
    report = [
        "# HAI Conditional Imputer Quality Audit",
        "",
        "## Decision",
        "",
        f"**{assessment}**: {len(reliable)}/{len(control_names)} active control channels passed every preregistered quality criterion.",
        "",
        "The primary replacement excludes the target channel's own previous value and restores normal residual variation with a length-20 block bootstrap.",
        "",
        "## Channel Results",
        "",
        display.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "Passing this gate means the sampled normal-validation trajectories meet the stated predictive and distributional checks. It does not make them physical interventions or prove that discrete commands are semantically valid at every sampled instant.",
    ]
    (output / "HAI_IMPUTER_QUALITY_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
