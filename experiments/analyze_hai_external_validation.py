from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ["auroc", "auprc", "etap", "etar", "etaf1", "fpr", "event_detected_ratio", "censored_delay_mean"]


def paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot(index=["seed", "label"], columns="variant", values=METRICS)
    rows: list[dict] = []
    for seed, label in wide.index:
        for baseline in ["F0", "F0-C"]:
            row = {"seed": int(seed), "label": label, "contrast": f"F1-{baseline}"}
            for metric in METRICS:
                # Positive delay delta means F1 detects earlier; positive FPR delta means F1 is safer.
                if metric in {"censored_delay_mean", "fpr"}:
                    row[f"delta_{metric}"] = wide.loc[(seed, label), (metric, baseline)] - wide.loc[(seed, label), (metric, "F1")]
                else:
                    row[f"delta_{metric}"] = wide.loc[(seed, label), (metric, "F1")] - wide.loc[(seed, label), (metric, baseline)]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_deltas(deltas: pd.DataFrame, repeats: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260821)
    rows: list[dict] = []
    for (label, contrast), group in deltas.groupby(["label", "contrast"], sort=True):
        row = {"label": label, "contrast": contrast, "seeds": len(group)}
        for metric in METRICS:
            column = f"delta_{metric}"
            values = group[column].to_numpy(float)
            draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
            row[f"mean_delta_{metric}"] = float(values.mean())
            row[f"positive_seeds_{metric}"] = int(np.sum(values > 0))
            row[f"ci_low_{metric}"] = float(np.quantile(draws, 0.025))
            row[f"ci_high_{metric}"] = float(np.quantile(draws, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/hai_external_validation_v2")
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()

    manifest = json.loads((output / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("HAI external-validation run is incomplete")
    metrics = pd.read_csv(output / "HAI_EXTERNAL_METRICS.csv")
    target_events = pd.read_csv(output / "HAI_EXTERNAL_TARGET_EVENTS.csv")
    expected = len(manifest["seeds"]) * len(manifest["variants"]) * 4
    if len(metrics) != expected or metrics.duplicated(["seed", "variant", "label"]).any():
        raise ValueError("HAI metric table is incomplete or duplicated")
    if set(metrics.variant) != {"F0", "F1", "F0-C"}:
        raise ValueError("F0, F1, and F0-C are all required")

    aggregate = (
        metrics.groupby(["variant", "label"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            parameters=("parameters", "mean"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            etap_mean=("etap", "mean"),
            etar_mean=("etar", "mean"),
            etaf1_mean=("etaf1", "mean"),
            etaf1_sd=("etaf1", "std"),
            fpr_mean=("fpr", "mean"),
            detected_mean=("event_detected_ratio", "mean"),
            delay_mean=("censored_delay_mean", "mean"),
        )
    )
    deltas = paired_deltas(metrics)
    summary = summarize_deltas(deltas, args.bootstrap_repeats)
    aggregate.to_csv(output / "HAI_EXTERNAL_AGGREGATE.csv", index=False)
    deltas.to_csv(output / "HAI_EXTERNAL_SEED_DELTAS.csv", index=False)
    summary.to_csv(output / "HAI_EXTERNAL_DELTA_SUMMARY.csv", index=False)

    target_seed = (
        target_events.groupby(["seed", "variant", "target_class"], as_index=False)
        .agg(
            events=("global_event", "nunique"),
            detected_ratio=("detected", "mean"),
            censored_delay_mean=("alarm_delay_censored", "mean"),
        )
    )
    target_summary = (
        target_seed.groupby(["variant", "target_class"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            events=("events", "max"),
            detected_ratio_mean=("detected_ratio", "mean"),
            detected_ratio_sd=("detected_ratio", "std"),
            censored_delay_mean=("censored_delay_mean", "mean"),
        )
    )
    target_wide = target_seed.pivot(
        index=["seed", "target_class"],
        columns="variant",
        values=["detected_ratio", "censored_delay_mean"],
    )
    target_delta_rows: list[dict] = []
    for seed, target_class in target_wide.index:
        for baseline in ["F0", "F0-C"]:
            target_delta_rows.append(
                {
                    "seed": int(seed),
                    "target_class": target_class,
                    "contrast": f"F1-{baseline}",
                    "delta_detected_ratio": float(
                        target_wide.loc[(seed, target_class), ("detected_ratio", "F1")]
                        - target_wide.loc[(seed, target_class), ("detected_ratio", baseline)]
                    ),
                    "delta_censored_delay_mean": float(
                        target_wide.loc[(seed, target_class), ("censored_delay_mean", baseline)]
                        - target_wide.loc[(seed, target_class), ("censored_delay_mean", "F1")]
                    ),
                }
            )
    target_deltas = pd.DataFrame(target_delta_rows)
    target_delta_summary = (
        target_deltas.groupby(["target_class", "contrast"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_delta_detected_ratio=("delta_detected_ratio", "mean"),
            positive_seeds_detected_ratio=("delta_detected_ratio", lambda value: int((value > 0).sum())),
            mean_delta_censored_delay=("delta_censored_delay_mean", "mean"),
            positive_seeds_delay=("delta_censored_delay_mean", lambda value: int((value > 0).sum())),
        )
    )
    target_seed.to_csv(output / "HAI_EXTERNAL_TARGET_SEED_METRICS.csv", index=False)
    target_summary.to_csv(output / "HAI_EXTERNAL_TARGET_SUMMARY.csv", index=False)
    target_deltas.to_csv(output / "HAI_EXTERNAL_TARGET_SEED_DELTAS.csv", index=False)
    target_delta_summary.to_csv(output / "HAI_EXTERNAL_TARGET_DELTA_SUMMARY.csv", index=False)

    global_summary = summary[summary.label == "attack"].set_index("contrast")
    global_metrics = metrics.loc[metrics.label == "attack"]
    finite_columns = ["threshold", "auroc", "auprc", "etaf1", "fpr"]
    finite_metrics = np.isfinite(global_metrics[finite_columns].to_numpy(float)).all()
    fpr_saturated = bool(
        (global_metrics.groupby("variant")["fpr"].mean() >= 0.99).all()
    )
    evaluation_valid = bool(finite_metrics and not fpr_saturated)
    f0 = global_summary.loc["F1-F0"]
    f0c = global_summary.loc["F1-F0-C"]
    fpr_tolerance = 0.005
    support = (
        f0.mean_delta_etaf1 > 0
        and f0.positive_seeds_etaf1 >= 2
        and f0c.mean_delta_etaf1 > 0
        and f0c.positive_seeds_etaf1 >= 2
        and f0.mean_delta_fpr >= -fpr_tolerance
        and f0c.mean_delta_fpr >= -fpr_tolerance
    )
    if not evaluation_valid:
        decision = "EXTERNAL_EVALUATION_INVALID"
    else:
        decision = "EXTERNAL_SUPPORT" if support else "EXTERNAL_NONCONFIRMATION"
    global_table = aggregate[aggregate.label == "attack"][
        [
            "variant",
            "parameters",
            "auroc_mean",
            "auprc_mean",
            "etap_mean",
            "etar_mean",
            "etaf1_mean",
            "fpr_mean",
            "detected_mean",
            "delay_mean",
        ]
    ].round(4)
    process_table = aggregate[aggregate.label != "attack"][
        ["variant", "label", "auroc_mean", "auprc_mean", "etaf1_mean", "fpr_mean", "detected_mean"]
    ].round(4)
    delta_table = summary[
        [
            "label",
            "contrast",
            "mean_delta_auroc",
            "positive_seeds_auroc",
            "mean_delta_auprc",
            "positive_seeds_auprc",
            "mean_delta_etaf1",
            "positive_seeds_etaf1",
            "mean_delta_fpr",
            "mean_delta_censored_delay_mean",
        ]
    ].round(4)
    parameter_gap = abs(manifest["f1_parameter_target"] - int(
        metrics.loc[metrics.variant == "F0-C", "parameters"].iloc[0]
    )) / manifest["f1_parameter_target"]
    report = [
        "# HAI 21.03 External Validation Report",
        "",
        "## Decision",
        "",
        f"**{decision}** under the descriptive rule locked before the complete v2 result analysis. This is an external dataset check, not an independent causal proof.",
        "" if evaluation_valid else "The global metrics failed the finite-value or false-positive saturation quality gate, so no external-support conclusion is permitted.",
        "",
        "## Global Attack Label",
        "",
        global_table.to_markdown(index=False),
        "",
        "## Paired Seed Differences",
        "",
        "Positive AUROC/AUPRC/eTaF1 values favor F1. Positive FPR values mean F1 has fewer false alarms; positive delay values mean F1 is earlier.",
        "",
        delta_table.to_markdown(index=False),
        "",
        "## Process Labels",
        "",
        process_table.to_markdown(index=False),
        "Process-specific FPR treats attacks assigned only to another process as negatives, so it is not directly comparable to global-label FPR.",
        "",
        "## Attack-Target Subgroups",
        "",
        "All 50 HAI 21.03 events directly target at least one control-history point: 31 target only control points and 19 jointly target control and sensor points.",
        "",
        target_summary.round(4).to_markdown(index=False),
        "",
        "Positive detection and delay deltas favor F1.",
        "",
        target_delta_summary.round(4).to_markdown(index=False),
        "",
        "## Design Controls",
        "",
        f"F0 and F1 predict the same 29 next-step sensor/model targets. F1 adds 28 nonconstant control-history inputs. F0-C uses F0 inputs with a widened hidden state; its parameter gap from F1 is `{parameter_gap:.2%}`.",
        f"Training used `{manifest['train_rows_after_overlap_exclusion']:,}` retained official-normal rows before chronological splitting. Exact train-test duplicates were excluded before scaling and window construction.",
        "The official eTaPR implementation was used with fixed validation-calibrated thresholds. CSV files remain separate episodes during windowing and are separated by a normal gap for range evaluation.",
        "",
        "## Statistical Boundary",
        "",
        "Only three seeds are available, so the bootstrap intervals are descriptive sensitivity summaries and no seed-level significance claim is made. Process labels are partial annotations; the global attack label is primary. Medium-confidence HAI 21.03 alias mappings remain a documented external-validity limitation.",
        "Because every event directly manipulates a control-history point, HAI cannot validate the stronger claim that control history helps when no control channel is attacked. The TEP fault experiments remain primary for that claim.",
    ]
    (output / "HAI_EXTERNAL_VALIDATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (output / "HAI_EXTERNAL_ANALYSIS_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "decision": decision,
                "evaluation_valid": evaluation_valid,
                "fpr_saturated": fpr_saturated,
                "bootstrap_repeats": args.bootstrap_repeats,
                "fpr_tolerance": fpr_tolerance,
                "parameter_gap_fraction": parameter_gap,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
