from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp, wasserstein_distance

from conditional_imputer import (
    build_residual_bank,
    fit_leave_one_channel_out,
    predict_leave_one_channel_out,
    sample_residual_blocks,
)


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    return (ROOT / path).resolve()


def fit_legacy_imputer(
    features: np.ndarray,
    train_runs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ridge: float = 1e-2,
) -> np.ndarray:
    xtx = np.zeros((53, 53), dtype=np.float64)
    xty = np.zeros((53, 11), dtype=np.float64)
    for run in train_runs:
        array = np.asarray(features[int(run), :599], dtype=np.float64)
        standardized = (array - mean) / std
        x = np.c_[
            standardized[1:, :41],
            standardized[:-1, 41:],
            np.ones(len(standardized) - 1),
        ]
        y = standardized[1:, 41:]
        xtx += x.T @ x
        xty += x.T @ y
    penalty = np.eye(53) * ridge
    penalty[-1, -1] = 0.0
    return np.linalg.solve(xtx + penalty, xty).astype(np.float32)


def lag_one_correlation(values: np.ndarray) -> float:
    left = values[:, :-1].reshape(-1)
    right = values[:, 1:].reshape(-1)
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def quality_rows(
    method: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    persistence: np.ndarray,
    reference_observed: np.ndarray,
    observed_sequences: np.ndarray,
    predicted_sequences: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for channel in range(11):
        y = observed[:, channel]
        y_hat = predicted[:, channel]
        reference = reference_observed[:, channel]
        baseline = persistence[:, channel]
        residual = y - y_hat
        sse = float(np.sum(residual**2))
        sst = float(np.sum((y - y.mean()) ** 2))
        rows.append(
            {
                "method": method,
                "channel": channel + 1,
                "n_validation_points": len(y),
                "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "zero_baseline_rmse": float(np.sqrt(np.mean(y**2))),
                "persistence_rmse": float(np.sqrt(np.mean((y - baseline) ** 2))),
                "observed_mean": float(reference.mean()),
                "predicted_mean": float(y_hat.mean()),
                "observed_std": float(reference.std()),
                "predicted_std": float(y_hat.std()),
                "std_ratio": float(y_hat.std() / reference.std())
                if reference.std() > 0
                else float("nan"),
                "residual_std": float(residual.std()),
                "pearson": float(np.corrcoef(y, y_hat)[0, 1])
                if y.std() > 0 and y_hat.std() > 0
                else float("nan"),
                "wasserstein": float(wasserstein_distance(reference, y_hat)),
                "ks_statistic": float(ks_2samp(reference, y_hat).statistic),
                "observed_lag1": lag_one_correlation(observed_sequences[:, :, channel]),
                "predicted_lag1": lag_one_correlation(predicted_sequences[:, :, channel]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/architecture_chum_g3.yaml")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))

    output = resolve(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    features = np.load(resolve(config["paths"]["cache"]) / "features.npy", mmap_mode="r")
    split = pd.read_csv(resolve(config["paths"]["split"]))
    scaler = pd.read_csv(resolve(config["paths"]["scaler"]))
    mean = scaler["mean"].to_numpy(np.float32)
    std = scaler["std"].to_numpy(np.float32)
    train_runs = split.loc[split.split == "train", "run_index"].to_numpy()
    validation_runs = split.loc[split.split == "validation", "run_index"].to_numpy()

    fitted_legacy = fit_legacy_imputer(features, train_runs, mean, std)
    saved_legacy_path = resolve(config["paths"]["imputer_weights"])
    saved_legacy = np.load(saved_legacy_path)
    max_weight_difference = float(np.max(np.abs(fitted_legacy - saved_legacy)))

    loo_weights = fit_leave_one_channel_out(features, train_runs, mean, std)
    residual_bank = build_residual_bank(features, train_runs, mean, std, loo_weights)
    loo_weights_path = output / "normal_xmv_loo_imputer_weights.npy"
    residual_bank_path = output / "normal_xmv_loo_residual_bank.npy"
    np.save(loo_weights_path, loo_weights)
    np.save(residual_bank_path, residual_bank)

    observed_rows: list[np.ndarray] = []
    persistence_rows: list[np.ndarray] = []
    legacy_rows: list[np.ndarray] = []
    loo_mean_rows: list[np.ndarray] = []
    sampled_observed_rows: list[np.ndarray] = []
    sampled_persistence_rows: list[np.ndarray] = []
    loo_sample_rows: list[np.ndarray] = []
    observed_sequences: list[np.ndarray] = []
    legacy_sequences: list[np.ndarray] = []
    loo_mean_sequences: list[np.ndarray] = []
    sampled_observed_sequences: list[np.ndarray] = []
    loo_sample_sequences: list[np.ndarray] = []
    rng = np.random.default_rng(20260821)
    sample_draws = 3
    block_length = int(config["window"])

    for run in validation_runs:
        array = np.asarray(features[int(run), :599], dtype=np.float32)
        standardized = (array - mean) / std
        legacy_predictors = np.c_[
            standardized[1:, :41],
            standardized[:-1, 41:],
            np.ones(len(standardized) - 1, dtype=np.float32),
        ]
        observed = standardized[1:, 41:]
        persistence = standardized[:-1, 41:]
        legacy = legacy_predictors @ saved_legacy
        loo_mean = predict_leave_one_channel_out(standardized, loo_weights)

        observed_rows.append(observed)
        persistence_rows.append(persistence)
        legacy_rows.append(legacy)
        loo_mean_rows.append(loo_mean)
        observed_sequences.append(observed)
        legacy_sequences.append(legacy)
        loo_mean_sequences.append(loo_mean)
        for _ in range(sample_draws):
            sampled = loo_mean.copy()
            for channel in range(11):
                sampled[:, channel] += sample_residual_blocks(
                    residual_bank,
                    channel,
                    len(sampled),
                    rng,
                    block_length,
                )
            sampled_observed_rows.append(observed)
            sampled_persistence_rows.append(persistence)
            loo_sample_rows.append(sampled)
            sampled_observed_sequences.append(observed)
            loo_sample_sequences.append(sampled)

    observed = np.concatenate(observed_rows)
    persistence = np.concatenate(persistence_rows)
    legacy = np.concatenate(legacy_rows)
    loo_mean = np.concatenate(loo_mean_rows)
    sampled_observed = np.concatenate(sampled_observed_rows)
    sampled_persistence = np.concatenate(sampled_persistence_rows)
    loo_sample = np.concatenate(loo_sample_rows)
    observed_sequence = np.stack(observed_sequences)

    rows = quality_rows(
        "legacy_mean",
        observed,
        legacy,
        persistence,
        observed,
        observed_sequence,
        np.stack(legacy_sequences),
    )
    rows.extend(
        quality_rows(
            "loo_mean",
            observed,
            loo_mean,
            persistence,
            observed,
            observed_sequence,
            np.stack(loo_mean_sequences),
        )
    )
    rows.extend(
        quality_rows(
            "loo_residual_sample",
            sampled_observed,
            loo_sample,
            sampled_persistence,
            observed,
            np.stack(sampled_observed_sequences),
            np.stack(loo_sample_sequences),
        )
    )
    result = pd.DataFrame(rows)
    result.to_csv(output / "IMPUTER_VALIDATION.csv", index=False)

    legacy_result = result[result.method == "legacy_mean"]
    loo_result = result[result.method == "loo_mean"]
    sample_result = result[result.method == "loo_residual_sample"]
    constant_channels = legacy_result.loc[
        legacy_result.observed_std <= 1e-12, "channel"
    ].astype(int).tolist()
    weak_channels = legacy_result.loc[
        (legacy_result.observed_std > 1e-12) & (legacy_result.r2 < 0.5), "channel"
    ].astype(int).tolist()
    collapsed_channels = legacy_result.loc[
        (legacy_result.observed_std > 1e-12) & (legacy_result.std_ratio < 0.75),
        "channel",
    ].astype(int).tolist()
    sampled_collapsed_channels = sample_result.loc[
        (sample_result.observed_std > 1e-12) & (sample_result.std_ratio < 0.75),
        "channel",
    ].astype(int).tolist()
    report = [
        "# Conditional Imputer Validation",
        "",
        "## Scope",
        "",
        f"The imputers were fit on {len(train_runs)} training runs and evaluated on {len(validation_runs)} held-out validation runs using only pre-onset normal samples.",
        "",
        "## Reproducibility Check",
        "",
        f"Maximum absolute difference between refit and saved legacy weights: `{max_weight_difference:.3e}`.",
        "",
        "## Target-History Leakage Guard",
        "",
        "The leave-one-channel-out model excludes the target XMV's own previous value. It uses current and previous XMEAS plus previous values of the other ten XMV channels. The stochastic version adds 20-sample residual blocks drawn only from normal training runs.",
        "",
        "## Aggregate Quality",
        "",
        f"Legacy mean R2: `{legacy_result.r2.mean():.4f}`; leave-one-channel-out mean R2: `{loo_result.r2.mean():.4f}`.",
        f"Mean absolute standard-deviation-ratio error: legacy `{np.nanmean(np.abs(legacy_result.std_ratio - 1)):.4f}`, residual-sampled LOO `{np.nanmean(np.abs(sample_result.std_ratio - 1)):.4f}`.",
        f"Constant channels: `{constant_channels}`.",
        f"Legacy channels with R2 below 0.5: `{weak_channels}`.",
        f"Legacy channels with standard-deviation ratio below 0.75: `{collapsed_channels}`.",
        f"Residual-sampled LOO channels still below 0.75: `{sampled_collapsed_channels}`.",
        "",
        "## Interpretation Boundary",
        "",
        "The residual sampler is a normal-data block bootstrap, not a physical intervention or an exact conditional generative model. Agreement between legacy mean, leave-one-channel-out sampling, zero occlusion, and architecture families is reported as sensitivity evidence rather than causal proof.",
        "",
        "## Channel Results",
        "",
        result.round(4).to_markdown(index=False),
    ]
    (output / "IMPUTER_VALIDATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (output / "IMPUTER_VALIDATION_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "train_runs": len(train_runs),
                "validation_runs": len(validation_runs),
                "max_abs_refit_weight_difference": max_weight_difference,
                "constant_channels": constant_channels,
                "legacy_weak_r2_channels": weak_channels,
                "legacy_variance_collapsed_channels": collapsed_channels,
                "sampled_variance_collapsed_channels": sampled_collapsed_channels,
                "loo_weights_path": str(loo_weights_path),
                "residual_bank_path": str(residual_bank_path),
                "sample_draws": sample_draws,
                "residual_block_length": block_length,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
