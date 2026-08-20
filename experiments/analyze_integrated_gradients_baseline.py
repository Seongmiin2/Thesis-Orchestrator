from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_FAULTS = {4, 19, 25, 26}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/integrated_gradients_baseline")
    parser.add_argument("--g3", default="outputs/architecture_chum_g3")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    g3 = (ROOT / args.g3).resolve()

    manifest = json.loads((output / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("Integrated Gradients run is incomplete")
    g3_manifest = json.loads((g3 / "G3_ANALYSIS_MANIFEST.json").read_text(encoding="utf-8"))
    if g3_manifest.get("status") != "COMPLETE":
        raise RuntimeError("G3 analysis is incomplete")

    ig = pd.read_csv(output / "INTEGRATED_GRADIENTS.csv")
    chum = pd.read_csv(g3 / "G3_CELL_SUMMARY.csv")
    chum = chum.loc[
        (chum["mode"] == g3_manifest["primary_mode"])
        & chum.architecture.isin(manifest["architectures"])
        & chum.fault_id.isin(manifest["faults"])
    ].copy()
    ig_summary = (
        ig.groupby(["architecture", "fault_id", "channel"], as_index=False)
        .agg(
            mean_normalized_xmv_ig=("normalized_xmv_ig", "mean"),
            sd_normalized_xmv_ig=("normalized_xmv_ig", "std"),
            mean_abs_ig=("mean_abs_ig", "mean"),
            mean_abs_convergence_delta=("mean_abs_convergence_delta", "mean"),
        )
    )
    merged = ig_summary.merge(
        chum[
            [
                "architecture",
                "fault_id",
                "channel",
                "mean_delta_auroc",
                "mean_delta_auprc",
                "stable_material",
            ]
        ],
        on=["architecture", "fault_id", "channel"],
        how="inner",
        validate="one_to_one",
    )
    merged["ig_rank"] = merged.groupby(["architecture", "fault_id"])[
        "mean_normalized_xmv_ig"
    ].rank(ascending=False, method="min")
    merged["chum_rank"] = merged.groupby(["architecture", "fault_id"])[
        "mean_delta_auroc"
    ].rank(ascending=False, method="min")
    merged.to_csv(output / "IG_CHUM_CHANNEL_COMPARISON.csv", index=False)

    agreement_rows: list[dict] = []
    for (architecture, fault), group in merged.groupby(["architecture", "fault_id"], sort=True):
        ig_top = int(group.sort_values("ig_rank").channel.iloc[0])
        chum_top = int(group.sort_values("chum_rank").channel.iloc[0])
        ig_top3 = set(group.nsmallest(3, "ig_rank").channel.astype(int))
        chum_top3 = set(group.nsmallest(3, "chum_rank").channel.astype(int))
        agreement_rows.append(
            {
                "architecture": architecture,
                "fault_id": int(fault),
                "fault_class": "primary" if int(fault) in PRIMARY_FAULTS else "negative_or_exploratory",
                "spearman": float(
                    spearmanr(group.mean_normalized_xmv_ig, group.mean_delta_auroc).statistic
                ),
                "ig_top_channel": ig_top,
                "chum_top_channel": chum_top,
                "top1_agreement": ig_top == chum_top,
                "top3_overlap": len(ig_top3 & chum_top3),
                "material_chum_cells": int(group.stable_material.sum()),
                "mean_abs_convergence_delta": float(group.mean_abs_convergence_delta.mean()),
            }
        )
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(output / "IG_CHUM_RANK_AGREEMENT.csv", index=False)
    class_summary = (
        agreement.groupby("fault_class", as_index=False)
        .agg(
            cells=("spearman", "size"),
            median_spearman=("spearman", "median"),
            mean_spearman=("spearman", "mean"),
            top1_agreements=("top1_agreement", "sum"),
            mean_top3_overlap=("top3_overlap", "mean"),
        )
    )
    class_summary.to_csv(output / "IG_CHUM_CLASS_SUMMARY.csv", index=False)

    primary = agreement[agreement.fault_class == "primary"]
    report = [
        "# Integrated Gradients Attribution Baseline",
        "",
        "## Scope",
        "",
        "Integrated Gradients attributes the instantaneous forecasting-error score from a normal-mean baseline. CHUM measures the change in fault-detection performance after replacing one control channel. Agreement is corroboration; disagreement is expected because the estimands differ.",
        "",
        "## Rank Agreement",
        "",
        agreement.round(4).to_markdown(index=False),
        "",
        "## Fault-Class Summary",
        "",
        class_summary.round(4).to_markdown(index=False),
        "",
        "## Interpretation Boundary",
        "",
        f"Across the `{len(primary)}` architecture-by-primary-fault cells, exact top-channel agreement occurs in `{int(primary.top1_agreement.sum())}` and the median top-three overlap is `{primary.top3_overlap.median():.1f}` of 3.",
        "Integrated Gradients is baseline-dependent and describes local score sensitivity, not conditional channel necessity. CHUM remains the primary utility measure; this analysis checks whether a standard gradient attribution method tells a compatible or materially different story.",
    ]
    (output / "INTEGRATED_GRADIENTS_BASELINE_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (output / "ANALYSIS_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "primary_mode": g3_manifest["primary_mode"],
                "architecture_fault_cells": len(agreement),
                "primary_cells": len(primary),
                "primary_top1_agreements": int(primary.top1_agreement.sum()),
                "primary_median_spearman": float(primary.spearman.median()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
