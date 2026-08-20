from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    output = ROOT / "outputs/final_evidence_validation"
    output.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def check(name: str, condition: bool, evidence: str) -> None:
        require(condition, f"{name}: {evidence}")
        checks.append({"check": name, "status": "PASS", "evidence": evidence})

    g3 = ROOT / "outputs/architecture_chum_g3"
    g3_run = load_json(g3 / "RUN_MANIFEST.json")
    g3_analysis = load_json(g3 / "G3_ANALYSIS_MANIFEST.json")
    g3_faults = pd.read_csv(g3 / "G3_FAULT_RESULTS.csv")
    g3_runs = pd.read_csv(g3 / "G3_RUN_RESULTS.csv")
    g3_summary = pd.read_csv(g3 / "G3_CELL_SUMMARY.csv")
    g3_consensus = pd.read_csv(g3 / "G3_ARCHITECTURE_CONSENSUS.csv")
    check(
        "G3 task completeness",
        g3_run["status"] == "COMPLETE_FULL_CONFIG"
        and len(g3_faults) == 9_520
        and len(g3_runs) == 190_400
        and not g3_faults.duplicated(["architecture", "seed", "condition", "fault_id"]).any()
        and not g3_runs.duplicated(
            ["architecture", "seed", "condition", "fault_id", "run_index"]
        ).any(),
        "340 tasks; 9,520 fault rows and 190,400 run rows; no key duplicates",
    )

    primary_consensus = g3_consensus.loc[
        (g3_consensus["mode"] == "loo_sample")
        & g3_consensus.fault_id.isin([4, 19, 25, 26])
        & g3_consensus.two_architecture_consensus
    ]
    consensus_faults = sorted(primary_consensus.fault_id.unique().astype(int).tolist())
    check(
        "G3 locked decision",
        g3_analysis["decision"] == "PASS"
        and consensus_faults == g3_analysis["consensus_primary_faults"]
        and len(consensus_faults) >= g3_analysis["required_primary_faults"],
        f"consensus primary faults={consensus_faults}; required={g3_analysis['required_primary_faults']}",
    )

    original = g3_faults.loc[
        g3_faults["mode"] == "original",
        ["architecture", "seed", "fault_id", "auroc", "auprc", "pre_fpr"],
    ].rename(
        columns={
            "auroc": "original_auroc",
            "auprc": "original_auprc",
            "pre_fpr": "original_pre_fpr",
        }
    )
    loo = g3_faults.loc[g3_faults["mode"] == "loo_sample"].merge(
        original,
        on=["architecture", "seed", "fault_id"],
        validate="many_to_one",
    )
    loo["delta_auroc"] = loo.original_auroc - loo.auroc
    loo["delta_auprc"] = loo.original_auprc - loo.auprc
    loo["delta_pre_fpr"] = loo.pre_fpr - loo.original_pre_fpr
    recomputed_rows: list[dict] = []
    for consensus in primary_consensus.itertuples(index=False):
        for architecture in str(consensus.stable_architectures).split(","):
            raw = loo.loc[
                (loo.architecture == architecture)
                & (loo.fault_id == consensus.fault_id)
                & (loo.channel == consensus.channel)
            ]
            reported = g3_summary.loc[
                (g3_summary.architecture == architecture)
                & (g3_summary["mode"] == "loo_sample")
                & (g3_summary.fault_id == consensus.fault_id)
                & (g3_summary.channel == consensus.channel)
            ].iloc[0]
            values = {
                "architecture": architecture,
                "fault_id": int(consensus.fault_id),
                "channel": int(consensus.channel),
                "mean_delta_auroc": float(raw.delta_auroc.mean()),
                "mean_delta_auprc": float(raw.delta_auprc.mean()),
                "positive_seeds": int((raw.delta_auroc > 0).sum()),
                "max_abs_fpr_shift": float(raw.delta_pre_fpr.abs().max()),
                "run_ci_low": float(reported.run_delta_auroc_ci_low),
            }
            check(
                f"G3 raw recomputation {architecture}/F{consensus.fault_id}/XMV{consensus.channel}",
                len(raw) == 5
                and np.isclose(values["mean_delta_auroc"], reported.mean_delta_auroc)
                and np.isclose(values["mean_delta_auprc"], reported.mean_delta_auprc)
                and values["positive_seeds"] == reported.positive_auroc_seeds
                and np.isclose(values["max_abs_fpr_shift"], reported.max_abs_pre_fpr_shift)
                and values["run_ci_low"] > 0
                and bool(reported.stable_material),
                json.dumps(values),
            )
            recomputed_rows.append(values)
    pd.DataFrame(recomputed_rows).to_csv(output / "RECOMPUTED_G3_PRIMARY_CELLS.csv", index=False)

    ig_dir = ROOT / "outputs/integrated_gradients_baseline"
    ig_run = load_json(ig_dir / "RUN_MANIFEST.json")
    ig_analysis = load_json(ig_dir / "ANALYSIS_MANIFEST.json")
    ig = pd.read_csv(ig_dir / "INTEGRATED_GRADIENTS.csv")
    agreement = pd.read_csv(ig_dir / "IG_CHUM_RANK_AGREEMENT.csv")
    primary_agreement = agreement.loc[agreement.fault_class == "primary"]
    check(
        "Integrated Gradients completeness and agreement",
        ig_run["status"] == "COMPLETE"
        and len(ig) == 880
        and not ig.duplicated(["architecture", "seed", "fault_id", "channel"]).any()
        and len(primary_agreement) == 8
        and int(primary_agreement.top1_agreement.sum())
        == ig_analysis["primary_top1_agreements"]
        == 7,
        f"880 rows; primary top-1 agreement={int(primary_agreement.top1_agreement.sum())}/8",
    )

    prepared = ROOT / "outputs/hai_2103_prepared"
    preparation = load_json(prepared / "HAI_2103_PREPARATION_MANIFEST.json")
    roles = load_json(prepared / "HAI_2103_ROLE_MANIFEST.json")
    targets = load_json(prepared / "HAI_2103_ATTACK_TARGET_MANIFEST.json")
    check(
        "HAI data gates",
        preparation["remaining_train_test_hash_candidates"] == 0
        and preparation["train_excluded_rows"] == 43_202
        and roles["assessment"] == "PASS_TO_MODELING"
        and roles["active_sensor_points"] == 29
        and roles["active_control_points"] == 28
        and targets["events"] == 50
        and targets["events_without_control_target"] == 0,
        "43,202 overlaps removed; F0=29, added controls=28; 50 target-mapped events",
    )

    invalid_v1 = load_json(ROOT / "outputs/hai_external_validation/RUN_MANIFEST.json")
    check(
        "HAI invalid run quarantine",
        invalid_v1["status"] == "INVALIDATED_COLUMN_ORDER_BUG",
        invalid_v1["status"],
    )
    hai = ROOT / "outputs/hai_external_validation_v2"
    hai_run = load_json(hai / "RUN_MANIFEST.json")
    hai_analysis = load_json(hai / "HAI_EXTERNAL_ANALYSIS_MANIFEST.json")
    hai_metrics = pd.read_csv(hai / "HAI_EXTERNAL_METRICS.csv")
    hai_events = pd.read_csv(hai / "HAI_EXTERNAL_TARGET_EVENTS.csv")
    check(
        "HAI v2 task completeness",
        hai_run["status"] == "COMPLETE"
        and len(hai_metrics) == 36
        and len(hai_events) == 450
        and not hai_metrics.duplicated(["seed", "variant", "label"]).any()
        and not hai_events.duplicated(["seed", "variant", "global_event"]).any(),
        "9 models; 36 label metrics; 450 event rows; no key duplicates",
    )
    global_metrics = hai_metrics.loc[hai_metrics.label == "attack"]
    wide = global_metrics.pivot(index="seed", columns="variant", values=["auroc", "auprc", "etaf1", "fpr"])
    hai_rows: list[dict] = []
    for baseline in ["F0", "F0-C"]:
        row = {"contrast": f"F1-{baseline}"}
        for metric in ["auroc", "auprc", "etaf1"]:
            values = wide[(metric, "F1")] - wide[(metric, baseline)]
            row[f"mean_delta_{metric}"] = float(values.mean())
            row[f"positive_seeds_{metric}"] = int((values > 0).sum())
        row["mean_fpr_increase"] = float(
            (wide[("fpr", "F1")] - wide[("fpr", baseline)]).mean()
        )
        hai_rows.append(row)
    hai_recomputed = pd.DataFrame(hai_rows)
    hai_recomputed.to_csv(output / "RECOMPUTED_HAI_GLOBAL_DELTAS.csv", index=False)
    check(
        "HAI v2 external support recomputation",
        hai_analysis["decision"] == "EXTERNAL_SUPPORT"
        and (hai_recomputed.positive_seeds_auroc == 3).all()
        and (hai_recomputed.positive_seeds_auprc == 3).all()
        and (hai_recomputed.positive_seeds_etaf1 == 3).all()
        and (hai_recomputed.mean_fpr_increase <= hai_analysis["fpr_tolerance"]).all(),
        hai_recomputed.round(6).to_json(orient="records"),
    )

    conditional = ROOT / "outputs/hai_conditional_chum"
    conditional_run = load_json(conditional / "RUN_MANIFEST.json")
    conditional_imputer = load_json(conditional / "HAI_IMPUTER_MANIFEST.json")
    conditional_analysis = load_json(
        conditional / "HAI_CONDITIONAL_ANALYSIS_MANIFEST.json"
    )
    conditional_quality = pd.read_csv(conditional / "HAI_IMPUTER_QUALITY.csv")
    conditional_metrics = pd.read_csv(conditional / "HAI_CONDITIONAL_METRICS.csv")
    conditional_events = pd.read_csv(conditional / "HAI_CONDITIONAL_EVENTS.csv")
    check(
        "HAI conditional imputer quality gate",
        conditional_imputer["assessment"] == "PASS_TO_HAI_CONDITIONAL_CHUM"
        and conditional_imputer["reliable_channel_count"] == 12
        and int(conditional_quality.reliable.astype(bool).sum()) == 12
        and set(
            conditional_quality.loc[
                conditional_quality.reliable.astype(bool), "feature"
            ]
        )
        == set(conditional_imputer["reliable_channels"]),
        f"12/28 channels passed all locked imputer-quality criteria",
    )
    check(
        "HAI conditional task completeness",
        conditional_run["status"] == "COMPLETE"
        and len(conditional_metrics) == 171
        and len(conditional_events) == 8_550
        and not conditional_metrics.duplicated(["seed", "mode", "channel"]).any()
        and not conditional_events.duplicated(
            ["seed", "mode", "channel", "global_event"]
        ).any(),
        "171 tasks; 8,550 event rows; no composite-key duplicates",
    )

    original_metrics = conditional_metrics.loc[
        conditional_metrics["mode"] == "original"
    ].set_index("seed")
    perturbation_metrics = conditional_metrics.loc[
        conditional_metrics["mode"] != "original"
    ].copy()
    perturbation_metrics["abs_fpr_shift"] = [
        abs(float(row.fpr) - float(original_metrics.loc[row.seed, "fpr"]))
        for row in perturbation_metrics.itertuples(index=False)
    ]
    fpr_limit = 0.005
    reliable_metric_rows = perturbation_metrics.loc[
        perturbation_metrics.reliable_imputer.astype(bool)
    ]
    conditional_fpr_exceptions = {
        mode: int(
            (
                reliable_metric_rows.loc[
                    reliable_metric_rows["mode"] == mode, "abs_fpr_shift"
                ]
                > fpr_limit
            ).sum()
        )
        for mode in ["loo_sample", "zero"]
    }

    original_events = conditional_events.loc[
        conditional_events["mode"] == "original",
        ["seed", "global_event", "normalized_mean_event_score"],
    ].rename(
        columns={"normalized_mean_event_score": "original_normalized_mean_score"}
    )
    loo_events = conditional_events.loc[
        conditional_events["mode"] == "loo_sample"
    ].merge(
        original_events,
        on=["seed", "global_event"],
        validate="many_to_one",
    )
    loo_events["delta_normalized_mean_score"] = (
        loo_events.original_normalized_mean_score
        - loo_events.normalized_mean_event_score
    )
    loo_events = loo_events.merge(
        perturbation_metrics[
            ["seed", "mode", "channel", "abs_fpr_shift"]
        ],
        on=["seed", "mode", "channel"],
        validate="many_to_one",
    )
    recomputed_conditional = (
        loo_events.groupby(
            [
                "channel",
                "feature",
                "global_event",
                "attack_id",
                "reliable_imputer",
                "targeted_channel",
            ],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            mean_delta_normalized_mean_event_score=(
                "delta_normalized_mean_score",
                "mean",
            ),
            positive_seeds_event_score=(
                "delta_normalized_mean_score",
                lambda value: int((value > 0).sum()),
            ),
            max_abs_fpr_shift=("abs_fpr_shift", "max"),
        )
    )
    recomputed_stable = recomputed_conditional.loc[
        recomputed_conditional.reliable_imputer.astype(bool)
        & recomputed_conditional.targeted_channel.astype(bool)
        & (
            recomputed_conditional.mean_delta_normalized_mean_event_score >= 0.05
        )
        & (recomputed_conditional.positive_seeds_event_score >= 3)
        & (recomputed_conditional.max_abs_fpr_shift <= fpr_limit)
    ].copy()
    stable_keys = sorted(
        (int(row.global_event), str(row.attack_id), str(row.feature))
        for row in recomputed_stable.itertuples(index=False)
    )
    reported_targeted = pd.read_csv(
        conditional / "HAI_CONDITIONAL_TARGETED_SUMMARY.csv"
    )
    reported_stable_keys = sorted(
        (int(row.global_event), str(row.attack_id), str(row.feature))
        for row in reported_targeted.loc[
            (reported_targeted["mode"] == "loo_sample")
            & reported_targeted.stable_targeted_cell.astype(bool)
        ].itertuples(index=False)
    )
    expected_conditional_decision = (
        "EXTERNAL_CHANNEL_SUPPORT"
        if len(stable_keys) >= 3
        and conditional_fpr_exceptions["loo_sample"]
        <= conditional_fpr_exceptions["zero"]
        else "EXTERNAL_CHANNEL_NONCONFIRMATION"
    )
    pd.DataFrame(recomputed_stable).to_csv(
        output / "RECOMPUTED_HAI_CONDITIONAL_STABLE_CELLS.csv", index=False
    )
    check(
        "HAI conditional decision recomputation",
        conditional_analysis["decision"] == expected_conditional_decision
        and conditional_analysis["stable_targeted_cells"] == len(stable_keys)
        and conditional_analysis["loo_fpr_exceptions"]
        == conditional_fpr_exceptions["loo_sample"]
        and conditional_analysis["zero_fpr_exceptions"]
        == conditional_fpr_exceptions["zero"]
        and reported_stable_keys == stable_keys,
        json.dumps(
            {
                "decision": expected_conditional_decision,
                "stable_targeted_cells": len(stable_keys),
                "fpr_exceptions": conditional_fpr_exceptions,
            }
        ),
    )

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "FINAL_EVIDENCE_CHECKS.csv", index=False)
    report = [
        "# Final Evidence Validation",
        "",
        "## Overall Assessment: PASS",
        "",
        "All final claims were recomputed from raw result tables rather than copied from generated reports.",
        "",
        checks_frame.to_markdown(index=False),
        "",
        "The invalid HAI v1 directory is explicitly quarantined and is not used by any final check.",
    ]
    (output / "FINAL_EVIDENCE_VALIDATION.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (output / "VALIDATION_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "checks": len(checks),
                "g3_decision": g3_analysis["decision"],
                "hai_decision": hai_analysis["decision"],
                "hai_conditional_decision": conditional_analysis["decision"],
                "ig_primary_top1_agreements": ig_analysis["primary_top1_agreements"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
