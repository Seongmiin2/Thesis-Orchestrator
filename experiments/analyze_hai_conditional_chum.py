from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hai_conditional_chum.yaml")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = (ROOT / cfg["output"]).resolve()
    run_manifest = json.loads(
        (output / "RUN_MANIFEST.json").read_text(encoding="utf-8")
    )
    if run_manifest.get("status") != "COMPLETE":
        raise RuntimeError("HAI conditional CHUM run is incomplete")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if run_manifest.get("config_sha256") != config_hash:
        raise RuntimeError("Analysis config does not match the completed run")

    metrics = pd.read_csv(output / "HAI_CONDITIONAL_METRICS.csv")
    events = pd.read_csv(output / "HAI_CONDITIONAL_EVENTS.csv")
    expected_tasks = int(run_manifest["tasks"])
    expected_events = expected_tasks * int(run_manifest["attack_events"])
    if (
        len(metrics) != expected_tasks
        or len(events) != expected_events
        or metrics.duplicated(["seed", "mode", "channel"]).any()
        or events.duplicated(
            ["seed", "mode", "channel", "global_event"]
        ).any()
    ):
        raise ValueError("HAI conditional result tables are incomplete or duplicated")

    original_metrics = metrics.loc[
        metrics["mode"] == "original",
        [
            "seed",
            "threshold",
            "auroc",
            "auprc",
            "etaf1",
            "fpr",
            "event_detected_ratio",
            "censored_delay_mean",
        ],
    ].rename(columns=lambda name: name if name == "seed" else f"original_{name}")
    metric_deltas = metrics.loc[metrics["mode"] != "original"].merge(
        original_metrics, on="seed", how="left", validate="many_to_one"
    )
    for metric in [
        "auroc",
        "auprc",
        "etaf1",
        "event_detected_ratio",
    ]:
        metric_deltas[f"delta_{metric}"] = (
            metric_deltas[f"original_{metric}"] - metric_deltas[metric]
        )
    metric_deltas["delta_fpr"] = (
        metric_deltas.fpr - metric_deltas.original_fpr
    )
    metric_deltas["delta_censored_delay"] = (
        metric_deltas.censored_delay_mean
        - metric_deltas.original_censored_delay_mean
    )
    metric_deltas.to_csv(output / "HAI_CONDITIONAL_METRIC_DELTAS.csv", index=False)

    original_events = events.loc[
        events["mode"] == "original",
        [
            "seed",
            "global_event",
            "normalized_mean_event_score",
            "normalized_max_event_score",
            "detected",
            "alarm_delay_censored",
        ],
    ].rename(
        columns={
            "normalized_mean_event_score": "original_normalized_mean_event_score",
            "normalized_max_event_score": "original_normalized_max_event_score",
            "detected": "original_detected",
            "alarm_delay_censored": "original_alarm_delay_censored",
        }
    )
    event_deltas = events.loc[events["mode"] != "original"].merge(
        original_events,
        on=["seed", "global_event"],
        how="left",
        validate="many_to_one",
    )
    event_deltas["delta_normalized_mean_event_score"] = (
        event_deltas.original_normalized_mean_event_score
        - event_deltas.normalized_mean_event_score
    )
    event_deltas["delta_normalized_max_event_score"] = (
        event_deltas.original_normalized_max_event_score
        - event_deltas.normalized_max_event_score
    )
    event_deltas["delta_detected"] = (
        event_deltas.original_detected.astype(int)
        - event_deltas.detected.astype(int)
    )
    event_deltas["delta_alarm_delay"] = (
        event_deltas.alarm_delay_censored
        - event_deltas.original_alarm_delay_censored
    )
    event_deltas = event_deltas.merge(
        metric_deltas[["seed", "mode", "channel", "delta_fpr"]],
        on=["seed", "mode", "channel"],
        how="left",
        validate="many_to_one",
    )
    event_deltas.to_csv(output / "HAI_CONDITIONAL_EVENT_DELTAS.csv", index=False)

    decision_cfg = cfg["decision"]
    minimum_delta = float(decision_cfg["minimum_normalized_event_score_delta"])
    required_positive = int(decision_cfg["required_positive_seeds"])
    fpr_tolerance = float(decision_cfg["maximum_absolute_fpr_shift"])
    targeted = event_deltas.loc[event_deltas.targeted_channel].copy()
    summary = (
        targeted.groupby(
            [
                "mode",
                "channel",
                "feature",
                "global_event",
                "attack_id",
                "target_class",
                "reliable_imputer",
            ],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            mean_delta_normalized_score=(
                "delta_normalized_mean_event_score",
                "mean",
            ),
            min_delta_normalized_score=(
                "delta_normalized_mean_event_score",
                "min",
            ),
            max_delta_normalized_score=(
                "delta_normalized_mean_event_score",
                "max",
            ),
            positive_seeds=(
                "delta_normalized_mean_event_score",
                lambda values: int(np.sum(values > 0)),
            ),
            mean_detection_drop=("delta_detected", "mean"),
            mean_alarm_delay_increase=("delta_alarm_delay", "mean"),
            max_abs_fpr_shift=(
                "delta_fpr",
                lambda values: float(np.max(np.abs(values))),
            ),
        )
    )
    summary["stable_targeted_cell"] = (
        summary.reliable_imputer.astype(bool)
        & (summary.seeds == len(run_manifest["seeds"]))
        & (summary.mean_delta_normalized_score >= minimum_delta)
        & (summary.positive_seeds >= required_positive)
        & (summary.max_abs_fpr_shift <= fpr_tolerance)
    )
    summary.to_csv(output / "HAI_CONDITIONAL_TARGETED_SUMMARY.csv", index=False)

    reliable_metric_deltas = metric_deltas.loc[
        metric_deltas.reliable_imputer.astype(bool)
    ].copy()
    reliable_metric_deltas["fpr_exception"] = (
        reliable_metric_deltas.delta_fpr.abs() > fpr_tolerance
    )
    fpr_robustness = (
        reliable_metric_deltas.groupby("mode", as_index=False)
        .agg(
            seed_channel_cells=("fpr_exception", "size"),
            fpr_exceptions=("fpr_exception", "sum"),
            max_abs_fpr_shift=(
                "delta_fpr", lambda values: float(np.max(np.abs(values)))
            ),
        )
        .sort_values("mode")
    )
    fpr_robustness.to_csv(output / "HAI_CONDITIONAL_FPR_ROBUSTNESS.csv", index=False)

    stable_loo = summary.loc[
        (summary["mode"] == "loo_sample") & summary.stable_targeted_cell
    ].sort_values("mean_delta_normalized_score", ascending=False)
    fpr_counts = fpr_robustness.set_index("mode").fpr_exceptions.to_dict()
    required_cells = int(decision_cfg["minimum_targeted_stable_cells"])
    enough_cells = len(stable_loo) >= required_cells
    fpr_comparison = int(fpr_counts.get("loo_sample", 0)) <= int(
        fpr_counts.get("zero", 0)
    )
    support = enough_cells and fpr_comparison
    decision = (
        "EXTERNAL_CHANNEL_SUPPORT"
        if support
        else "EXTERNAL_CHANNEL_NONCONFIRMATION"
    )

    display_columns = [
        "global_event",
        "attack_id",
        "target_class",
        "feature",
        "mean_delta_normalized_score",
        "min_delta_normalized_score",
        "positive_seeds",
        "mean_detection_drop",
        "mean_alarm_delay_increase",
        "max_abs_fpr_shift",
    ]
    top_targeted = summary.loc[
        (summary["mode"] == "loo_sample")
        & summary.reliable_imputer.astype(bool)
    ].sort_values("mean_delta_normalized_score", ascending=False)
    targeted_reliable_channels = sorted(top_targeted.feature.unique().tolist())
    report = [
        "# HAI 21.03 Conditional CHUM Report",
        "",
        "## Decision",
        "",
        f"**{decision}**: `{len(stable_loo)}` directly targeted event-channel cells passed the locked conditional criteria; `{required_cells}` were required.",
        f"Conditional replacement produced `{int(fpr_counts.get('loo_sample', 0))}` FPR exceptions across reliable seed-channel cells versus `{int(fpr_counts.get('zero', 0))}` for the zero-replacement diagnostic.",
        "",
        "## Passing Targeted Cells",
        "",
        stable_loo[display_columns].round(4).to_markdown(index=False)
        if len(stable_loo)
        else "No targeted conditional cell passed every locked criterion.",
        "",
        "## Strongest Quality-Gated Targeted Cells",
        "",
        top_targeted[display_columns].head(20).round(4).to_markdown(index=False),
        "",
        "## FPR Robustness",
        "",
        fpr_robustness.round(6).to_markdown(index=False),
        "",
        "## Imputer Gate",
        "",
        f"`{run_manifest['reliable_channel_count']}` of 28 active control channels passed the held-out predictive, variance, mean, lag, and range checks: `{run_manifest['reliable_channels']}`.",
        f"Only `{len(targeted_reliable_channels)}` quality-gated channels are directly named by at least one audited HAI attack and therefore enter targeted cell decisions: `{targeted_reliable_channels}`.",
        "",
        "## Interpretation Boundary",
        "",
        "Positive score deltas mean that replacing the directly attacked control history reduced the threshold-normalized attack score. The comparison uses fixed completed F1 checkpoints and recalibrates each threshold on equally perturbed validation-normal data.",
        "These are conditional predictive-information results, not physical interventions, controller root-cause labels, or causal plant-mechanism estimates. Because every HAI event directly attacks at least one control-history point, TEP remains the primary evidence for faults without direct control manipulation.",
        "Only three seeds are available, so seed summaries are descriptive and the all-seed direction rule is used as a stability filter rather than a significance test.",
    ]
    (output / "HAI_CONDITIONAL_CHUM_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (output / "HAI_CONDITIONAL_ANALYSIS_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "decision": decision,
                "stable_targeted_cells": len(stable_loo),
                "required_targeted_cells": required_cells,
                "loo_fpr_exceptions": int(fpr_counts.get("loo_sample", 0)),
                "zero_fpr_exceptions": int(fpr_counts.get("zero", 0)),
                "fpr_comparison_pass": bool(fpr_comparison),
                "minimum_normalized_event_score_delta": minimum_delta,
                "required_positive_seeds": required_positive,
                "maximum_absolute_fpr_shift": fpr_tolerance,
                "reliable_channel_count": int(
                    run_manifest["reliable_channel_count"]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
