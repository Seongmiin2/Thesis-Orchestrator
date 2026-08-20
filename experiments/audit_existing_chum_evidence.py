from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/evidence_audit"


def architecture_gate() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(ROOT / "outputs/architecture_gate_g2_full/G2_FAULT_RESULTS.csv")
    expected_rows = 2 * 5 * 3 * 28
    if len(source) != expected_rows:
        raise ValueError(f"G2 row count mismatch: {len(source)} != {expected_rows}")
    keys = ["architecture", "seed", "fault_id"]
    wide = source.pivot(index=keys, columns="variant", values=["auroc", "auprc"]).reset_index()
    wide.columns = ["_".join(value for value in column if value) for column in wide.columns]
    for metric in ("auroc", "auprc"):
        wide[f"delta_{metric}_f0"] = wide[f"{metric}_F1"] - wide[f"{metric}_F0"]
        wide[f"delta_{metric}_f0c"] = wide[f"{metric}_F1"] - wide[f"{metric}_F0-C"]
    rows: list[dict] = []
    for (architecture, fault), group in wide.groupby(["architecture", "fault_id"], sort=True):
        record = {
            "architecture": architecture,
            "fault_id": int(fault),
            "mean_delta_auroc_f0": float(group.delta_auroc_f0.mean()),
            "mean_delta_auroc_f0c": float(group.delta_auroc_f0c.mean()),
            "mean_delta_auprc_f0": float(group.delta_auprc_f0.mean()),
            "mean_delta_auprc_f0c": float(group.delta_auprc_f0c.mean()),
            "positive_auroc_f0_seeds": int(np.sum(group.delta_auroc_f0 > 0)),
            "positive_auroc_f0c_seeds": int(np.sum(group.delta_auroc_f0c > 0)),
        }
        record["gain"] = (
            record["mean_delta_auroc_f0"] >= 0.02
            and record["mean_delta_auroc_f0c"] >= 0.02
            and record["mean_delta_auprc_f0"] >= 0.01
            and record["mean_delta_auprc_f0c"] >= 0.01
            and record["positive_auroc_f0_seeds"] >= 4
            and record["positive_auroc_f0c_seeds"] >= 4
        )
        rows.append(record)
    return pd.DataFrame(rows), wide


def perturbation_cells(path: Path, prefix: str, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(path)
    expected_rows = 5 * 28 * source.condition.nunique()
    if len(source) != expected_rows:
        raise ValueError(f"{mode} row count mismatch: {len(source)} != {expected_rows}")
    original = source[source.condition == "original"].rename(
        columns={"auroc": "original_auroc", "auprc": "original_auprc", "pre_fpr": "original_pre_fpr"}
    )
    selected = source[source.condition.str.startswith(prefix)].copy()
    selected["channel"] = selected.condition.str.extract(r"XMV_(\d+)$", expand=False).astype(int)
    merged = selected.merge(
        original[["seed", "fault_id", "original_auroc", "original_auprc", "original_pre_fpr"]],
        on=["seed", "fault_id"],
        how="left",
        validate="many_to_one",
    )
    merged["delta_auroc"] = merged.original_auroc - merged.auroc
    merged["delta_auprc"] = merged.original_auprc - merged.auprc
    merged["delta_pre_fpr"] = merged.pre_fpr - merged.original_pre_fpr
    merged["mode"] = mode
    summary = (
        merged.groupby(["mode", "fault_id", "channel"], as_index=False)
        .agg(
            mean_delta_auroc=("delta_auroc", "mean"),
            mean_delta_auprc=("delta_auprc", "mean"),
            positive_seeds=("delta_auroc", lambda values: int(np.sum(values > 0))),
            max_abs_pre_fpr_shift=("delta_pre_fpr", lambda values: float(np.max(np.abs(values)))),
        )
    )
    summary["stable_material"] = (
        (summary.mean_delta_auroc >= 0.02)
        & (summary.positive_seeds >= 4)
        & (summary.max_abs_pre_fpr_shift <= 0.005)
    )
    return summary, merged


def split_audit() -> dict:
    split = pd.read_csv(ROOT / "outputs/final_gate_exp1/artifacts/reinartz_split_manifest.csv")
    counts = split.split.value_counts().to_dict()
    per_fault = split.groupby(["fault_id", "split"]).size().unstack(fill_value=0)
    return {
        "rows": len(split),
        "unique_run_index": int(split.run_index.nunique()),
        "unique_run_id": int(split.run_id.nunique()),
        "split_counts": {key: int(value) for key, value in counts.items()},
        "max_splits_per_run": int(split.groupby("run_index").split.nunique().max()),
        "faults": int(split.fault_id.nunique()),
        "per_fault_counts_identical": bool((per_fault.nunique(axis=0) == 1).all()),
        "per_fault_counts": {
            column: int(per_fault[column].iloc[0]) for column in per_fault.columns
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    architecture, seed_rows = architecture_gate()
    conditional, conditional_seed_rows = perturbation_cells(
        ROOT / "outputs/conditional_chum_g1/CONDITIONAL_CHUM_RESULTS.csv",
        "conditional_XMV_",
        "conditional",
    )
    zero, zero_seed_rows = perturbation_cells(
        ROOT / "outputs/mechanism_gate_exp2/MECHANISM_GATE_EXP2_RESULTS.csv",
        "occlude_XMV_",
        "zero",
    )
    split = split_audit()

    architecture.to_csv(OUTPUT / "AUDITED_G2_GAIN.csv", index=False)
    seed_rows.to_csv(OUTPUT / "AUDITED_G2_SEED_DELTAS.csv", index=False)
    pd.concat([conditional, zero], ignore_index=True).to_csv(
        OUTPUT / "AUDITED_ATTRIBUTION_CELLS.csv", index=False
    )
    pd.concat([conditional_seed_rows, zero_seed_rows], ignore_index=True).to_csv(
        OUTPUT / "AUDITED_ATTRIBUTION_SEED_DELTAS.csv", index=False
    )

    gain_sets = {
        architecture_name: architecture.loc[
            (architecture.architecture == architecture_name) & architecture.gain, "fault_id"
        ].astype(int).tolist()
        for architecture_name in sorted(architecture.architecture.unique())
    }
    conditional_material = conditional[conditional.stable_material].sort_values(
        "mean_delta_auroc", ascending=False
    )
    zero_fpr_exceptions = int(np.sum(np.abs(zero_seed_rows.delta_pre_fpr) > 0.005))
    conditional_fpr_exceptions = int(
        np.sum(np.abs(conditional_seed_rows.delta_pre_fpr) > 0.005)
    )
    issues: list[str] = []
    if gain_sets.get("tcn") != [4, 7, 19, 23, 24, 25, 26]:
        issues.append("TCN GAIN set differs from the report")
    if gain_sets.get("transformer") != [4, 7, 19, 23, 24, 25, 26]:
        issues.append("Transformer GAIN set differs from the report")
    if split["rows"] != 2800 or split["max_splits_per_run"] != 1:
        issues.append("Split integrity failed")
    assessment = "READY_TO_EXTEND" if not issues else "NEEDS_REVISION"
    report = [
        "# Existing CHUM Evidence Audit",
        "",
        f"## Overall Assessment: {assessment}",
        "",
        "This audit independently recomputed the headline G1/G2 decisions from saved CSV artifacts.",
        "",
        "## Split Integrity",
        "",
        f"Rows/unique runs: `{split['rows']}/{split['unique_run_index']}`; split counts: `{split['split_counts']}`; per-fault counts: `{split['per_fault_counts']}`; maximum splits per run: `{split['max_splits_per_run']}`.",
        "",
        "## Architecture Gate",
        "",
        f"Recomputed GAIN sets: `{gain_sets}`.",
        "",
        "## Conditional Attribution",
        "",
        f"Stable material cells under mean AUROC loss >= 0.02, >=4/5 positive seeds, and max absolute FPR shift <= 0.005: `{len(conditional_material)}`.",
        "",
        conditional_material[
            ["fault_id", "channel", "mean_delta_auroc", "mean_delta_auprc", "positive_seeds", "max_abs_pre_fpr_shift"]
        ].round(4).to_markdown(index=False),
        "",
        "## FPR Comparison",
        "",
        f"Seed-level fault-channel cells exceeding absolute pre-FPR shift 0.005: zero occlusion `{zero_fpr_exceptions}`, conditional replacement `{conditional_fpr_exceptions}`.",
        "",
        "## Issues",
        "",
        "\n".join(f"- {issue}" for issue in issues) if issues else "No headline-result discrepancy was found.",
        "",
        "## Required Caveats",
        "",
        "- Five training seeds are repeated model fits on one fixed dataset, not five independent datasets.",
        "- Conditional replacement estimates predictive utility, not causal controller effects or physical root cause.",
        "- The existing architecture result is event-level; individual channel consensus requires G3.",
    ]
    (OUTPUT / "EXISTING_CHUM_EVIDENCE_AUDIT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (OUTPUT / "AUDIT_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "assessment": assessment,
                "gain_sets": gain_sets,
                "conditional_material_cells": len(conditional_material),
                "zero_fpr_exceptions": zero_fpr_exceptions,
                "conditional_fpr_exceptions": conditional_fpr_exceptions,
                "split": split,
                "issues": issues,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
