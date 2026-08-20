from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]


def parse_channel(condition: pd.Series) -> pd.Series:
    return condition.str.extract(r"XMV_(\d+)$", expand=False).astype(int)


def attach_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["architecture", "seed", "fault_id"]
    original = frame.loc[frame["mode"] == "original", keys + ["auroc", "auprc", "pre_fpr"]]
    original = original.rename(
        columns={
            "auroc": "original_auroc",
            "auprc": "original_auprc",
            "pre_fpr": "original_pre_fpr",
        }
    )
    perturbed = frame.loc[frame["mode"] != "original"].copy()
    merged = perturbed.merge(original, on=keys, how="left", validate="many_to_one")
    merged["delta_auroc"] = merged.original_auroc - merged.auroc
    merged["delta_auprc"] = merged.original_auprc - merged.auprc
    merged["delta_pre_fpr"] = merged.pre_fpr - merged.original_pre_fpr
    return merged


def load_gru_fault_results(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    specifications = [
        (root / "outputs/conditional_chum_g1/CONDITIONAL_CHUM_RESULTS.csv", "conditional", "conditional_XMV_"),
        (root / "outputs/mechanism_gate_exp2/MECHANISM_GATE_EXP2_RESULTS.csv", "zero", "occlude_XMV_"),
    ]
    for path, mode, prefix in specifications:
        source = pd.read_csv(path)
        original = source[source.condition == "original"].copy()
        original["architecture"] = "gru"
        original["mode"] = "original"
        original["channel"] = np.nan
        perturbed = source[source.condition.str.startswith(prefix)].copy()
        perturbed["architecture"] = "gru"
        perturbed["mode"] = mode
        perturbed["channel"] = parse_channel(perturbed.condition)
        frames.extend([original, perturbed])
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(["architecture", "seed", "condition", "fault_id", "mode"])


def hierarchical_run_intervals(run_frame: pd.DataFrame, repeats: int = 2000) -> pd.DataFrame:
    keys = ["architecture", "seed", "fault_id", "run_index"]
    original = run_frame.loc[run_frame["mode"] == "original", keys + ["auroc", "auprc"]]
    original = original.rename(columns={"auroc": "original_auroc", "auprc": "original_auprc"})
    paired = run_frame.loc[run_frame["mode"] != "original"].merge(
        original, on=keys, how="left", validate="many_to_one"
    )
    paired["run_delta_auroc"] = paired.original_auroc - paired.auroc
    paired["run_delta_auprc"] = paired.original_auprc - paired.auprc
    rng = np.random.default_rng(20260821)
    rows: list[dict] = []
    group_keys = ["architecture", "mode", "fault_id", "channel"]
    for group_key, group in paired.groupby(group_keys, sort=True):
        by_seed = [
            seed_frame.sort_values("run_index")
            for _, seed_frame in group.groupby("seed", sort=True)
        ]
        run_counts = {len(seed_frame) for seed_frame in by_seed}
        if len(run_counts) != 1:
            raise ValueError(f"Unequal run counts in bootstrap cell {group_key}: {run_counts}")
        auroc_values = np.stack(
            [seed_frame.run_delta_auroc.to_numpy(float) for seed_frame in by_seed]
        )
        auprc_values = np.stack(
            [seed_frame.run_delta_auprc.to_numpy(float) for seed_frame in by_seed]
        )
        seed_draws = rng.integers(0, len(by_seed), size=(repeats, len(by_seed)))
        run_draws = rng.integers(
            0, auroc_values.shape[1], size=(repeats, len(by_seed), auroc_values.shape[1])
        )
        auroc_boot = auroc_values[seed_draws[:, :, None], run_draws].mean(axis=(1, 2))
        auprc_boot = auprc_values[seed_draws[:, :, None], run_draws].mean(axis=(1, 2))
        architecture, mode, fault, channel = group_key
        rows.append(
            {
                "architecture": architecture,
                "mode": mode,
                "fault_id": int(fault),
                "channel": int(channel),
                "run_delta_auroc_mean": float(group.run_delta_auroc.mean()),
                "run_delta_auroc_ci_low": float(np.quantile(auroc_boot, 0.025)),
                "run_delta_auroc_ci_high": float(np.quantile(auroc_boot, 0.975)),
                "run_delta_auprc_mean": float(group.run_delta_auprc.mean()),
                "run_delta_auprc_ci_low": float(np.quantile(auprc_boot, 0.025)),
                "run_delta_auprc_ci_high": float(np.quantile(auprc_boot, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def rank_agreement(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for mode in sorted(summary["mode"].unique()):
        selected = summary[summary["mode"] == mode]
        for fault in sorted(selected.fault_id.unique()):
            fault_frame = selected[selected.fault_id == fault]
            pivot = fault_frame.pivot(index="channel", columns="architecture", values="mean_delta_auroc")
            for left, right in combinations(sorted(pivot.columns), 2):
                pair = pivot[[left, right]].dropna()
                correlation = spearmanr(pair[left], pair[right]).statistic if len(pair) >= 3 else np.nan
                rows.append(
                    {
                        "mode": mode,
                        "fault_id": int(fault),
                        "architecture_left": left,
                        "architecture_right": right,
                        "channels": len(pair),
                        "spearman": correlation,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/architecture_chum_g3")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()

    run_manifest = json.loads((output / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "COMPLETE_FULL_CONFIG":
        raise RuntimeError(
            "G3 analysis requires a complete full-config run; "
            f"found {run_manifest.get('status')!r}"
        )

    new_faults = pd.read_csv(output / "G3_FAULT_RESULTS.csv")
    gru_faults = load_gru_fault_results(ROOT)
    all_faults = pd.concat([new_faults, gru_faults], ignore_index=True, sort=False)
    cells = attach_deltas(all_faults)
    cells.to_csv(output / "G3_CELL_RESULTS.csv", index=False)

    summary = (
        cells.groupby(["architecture", "mode", "fault_id", "channel"], as_index=False)
        .agg(
            mean_delta_auroc=("delta_auroc", "mean"),
            median_delta_auroc=("delta_auroc", "median"),
            min_delta_auroc=("delta_auroc", "min"),
            mean_delta_auprc=("delta_auprc", "mean"),
            positive_auroc_seeds=("delta_auroc", lambda values: int(np.sum(values > 0))),
            max_abs_pre_fpr_shift=("delta_pre_fpr", lambda values: float(np.max(np.abs(values)))),
        )
    )
    summary["stable_material_raw"] = (
        (summary.mean_delta_auroc >= 0.02)
        & (summary.positive_auroc_seeds >= 4)
        & (summary.max_abs_pre_fpr_shift <= 0.005)
    )

    imputer = pd.read_csv(output / "IMPUTER_VALIDATION.csv")
    legacy_quality = imputer[imputer.method == "legacy_mean"].copy()
    legacy_quality["mode"] = "conditional"
    legacy_quality["imputer_reliable"] = (
        (legacy_quality.observed_std > 1e-12)
        & (legacy_quality.r2 >= 0.5)
        & (legacy_quality.std_ratio >= 0.75)
    )
    sampled_quality = imputer[imputer.method == "loo_residual_sample"].copy()
    sampled_quality["mode"] = "loo_sample"
    sampled_quality["imputer_reliable"] = (
        (sampled_quality.observed_std > 1e-12)
        & (sampled_quality.std_ratio.between(0.75, 1.25))
        & (sampled_quality.wasserstein <= 0.1)
        & (sampled_quality.ks_statistic <= 0.1)
        & ((sampled_quality.observed_lag1 - sampled_quality.predicted_lag1).abs() <= 0.2)
    )
    imputer_quality = pd.concat([legacy_quality, sampled_quality], ignore_index=True)[
        [
            "mode",
            "method",
            "channel",
            "r2",
            "std_ratio",
            "wasserstein",
            "ks_statistic",
            "observed_lag1",
            "predicted_lag1",
            "observed_std",
            "imputer_reliable",
        ]
    ]
    summary = summary.merge(
        imputer_quality, on=["mode", "channel"], how="left", validate="many_to_one"
    )
    conditional_modes = summary["mode"].isin(["conditional", "loo_sample"])
    summary["stable_material"] = summary.stable_material_raw & (
        ~conditional_modes | summary.imputer_reliable.fillna(False)
    )

    run_frame = pd.read_csv(output / "G3_RUN_RESULTS.csv")
    intervals = hierarchical_run_intervals(run_frame, repeats=args.bootstrap_repeats)
    intervals.to_csv(output / "G3_HIERARCHICAL_RUN_CI.csv", index=False)
    summary = summary.merge(
        intervals,
        on=["architecture", "mode", "fault_id", "channel"],
        how="left",
        validate="one_to_one",
    )
    summary["run_ci_excludes_zero"] = summary.run_delta_auroc_ci_low > 0
    summary["stable_material"] = (
        summary.stable_material & summary.run_ci_excludes_zero.fillna(False)
    )
    summary.to_csv(output / "G3_CELL_SUMMARY.csv", index=False)

    consensus_rows: list[dict] = []
    for (mode, fault, channel), group in summary.groupby(
        ["mode", "fault_id", "channel"], sort=True
    ):
        raw_architectures = sorted(
            group.loc[group.stable_material_raw, "architecture"].tolist()
        )
        stable_architectures = sorted(group.loc[group.stable_material, "architecture"].tolist())
        consensus_rows.append(
            {
                "mode": mode,
                "fault_id": int(fault),
                "channel": int(channel),
                "stable_architecture_count": len(stable_architectures),
                "stable_architectures": ",".join(stable_architectures),
                "raw_stable_architecture_count": len(raw_architectures),
                "raw_stable_architectures": ",".join(raw_architectures),
                "excluded_after_quality_or_ci": ",".join(
                    sorted(set(raw_architectures) - set(stable_architectures))
                ),
                "two_architecture_consensus": len(stable_architectures) >= 2,
                "three_architecture_consensus": len(stable_architectures) == 3,
            }
        )
    consensus = pd.DataFrame(consensus_rows)
    consensus.to_csv(output / "G3_ARCHITECTURE_CONSENSUS.csv", index=False)

    agreement = rank_agreement(summary)
    agreement.to_csv(output / "G3_RANK_AGREEMENT.csv", index=False)
    fpr_summary = (
        cells.assign(fpr_exception=np.abs(cells.delta_pre_fpr) > 0.005)
        .groupby(["architecture", "mode"], as_index=False)
        .agg(cells=("fpr_exception", "size"), fpr_exceptions=("fpr_exception", "sum"))
    )
    fpr_summary.to_csv(output / "G3_FPR_ROBUSTNESS.csv", index=False)

    primary_mode = "loo_sample" if "loo_sample" in summary["mode"].unique() else "conditional"
    primary_fault_ids = [4, 19, 25, 26]
    consensus_pass = consensus[
        (consensus["mode"] == primary_mode)
        & consensus.fault_id.isin(primary_fault_ids)
        & consensus.two_architecture_consensus
    ]
    primary = summary[
        (summary["mode"] == primary_mode) & summary.fault_id.isin(primary_fault_ids)
    ].sort_values(["architecture", "fault_id", "mean_delta_auroc"], ascending=[True, True, False])
    primary_top = primary.groupby(["architecture", "fault_id"], as_index=False).head(1)
    consensus_faults = sorted(consensus_pass.fault_id.unique().astype(int).tolist())
    decision = "PASS" if len(consensus_faults) >= 3 else "MODIFY"
    report = [
        "# Architecture Conditional CHUM G3 Report",
        "",
        "## Decision",
        "",
        f"**{decision}**: primary mode `{primary_mode}` has two-architecture consensus on `{len(consensus_faults)}` of the four locked primary faults: `{consensus_faults}`. The pass rule requires at least three distinct primary faults.",
        "",
        "## Primary Faults",
        "",
        primary_top[
            [
                "architecture",
                "fault_id",
                "channel",
                "mean_delta_auroc",
                "mean_delta_auprc",
                "positive_auroc_seeds",
                "max_abs_pre_fpr_shift",
                "imputer_reliable",
                "run_ci_excludes_zero",
                "stable_material",
                "run_delta_auroc_ci_low",
                "run_delta_auroc_ci_high",
            ]
        ].round(4).to_markdown(index=False),
        "",
        "## Consensus Cells",
        "",
        consensus_pass.to_markdown(index=False) if len(consensus_pass) else "No cell passed.",
        "",
        "## Conditional Imputer Reliability Gate",
        "",
        f"Reliable `{primary_mode}` channels under held-out distribution and lag checks: `{sorted(imputer_quality.loc[(imputer_quality['mode'] == primary_mode) & imputer_quality.imputer_reliable, 'channel'].astype(int).tolist())}`.",
        f"Excluded `{primary_mode}` channels: `{sorted(imputer_quality.loc[(imputer_quality['mode'] == primary_mode) & ~imputer_quality.imputer_reliable, 'channel'].astype(int).tolist())}`.",
        "Raw material cells on excluded channels are retained in the CSV for diagnosis but cannot establish conditional-attribution consensus.",
        "",
        "## FPR Robustness",
        "",
        fpr_summary.to_markdown(index=False),
        "",
        "## Architecture Rank Agreement",
        "",
        f"Median `{primary_mode}` channel Spearman correlation: `{agreement.loc[agreement['mode'] == primary_mode, 'spearman'].median():.4f}`.",
        "",
        "## Statistical Boundary",
        "",
        "The run-level intervals use paired hierarchical bootstrap resampling of model seeds and test runs. They quantify stability on this fixed TEP test distribution; they do not create independent datasets or support causal controller claims. Five seeds still cannot attain a two-sided exact seed sign-flip p-value below 0.05.",
    ]
    (output / "ARCHITECTURE_CONDITIONAL_CHUM_G3_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (output / "G3_ANALYSIS_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "decision": decision,
                "primary_mode": primary_mode,
                "consensus_cells": len(consensus_pass),
                "consensus_primary_faults": consensus_faults,
                "required_primary_faults": 3,
                "bootstrap_repeats": args.bootstrap_repeats,
                "architectures": sorted(summary.architecture.unique().tolist()),
                "reliable_imputer_channels": sorted(
                    imputer_quality.loc[
                        (imputer_quality["mode"] == primary_mode)
                        & imputer_quality.imputer_reliable,
                        "channel",
                    ]
                    .astype(int)
                    .tolist()
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
