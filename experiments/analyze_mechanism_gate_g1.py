from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


GAIN = {3, 4, 7, 19, 24, 25, 26}


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    means = [abs(np.mean(values * np.asarray(signs))) for signs in itertools.product((-1, 1), repeat=len(values))]
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p); ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted); out[order] = np.minimum(adjusted, 1.0)
    return out


def hierarchical_ci(frame: pd.DataFrame, draws: int, rng: np.random.Generator) -> tuple[float, float, float]:
    by_seed = {seed: group.delta.to_numpy(float) for seed, group in frame.groupby("seed")}
    seeds = np.asarray(sorted(by_seed))
    estimates = np.empty(draws)
    for i in range(draws):
        selected = rng.choice(seeds, len(seeds), replace=True)
        values = []
        for seed in selected:
            runs = by_seed[int(seed)]
            values.append(rng.choice(runs, len(runs), replace=True).mean())
        estimates[i] = np.mean(values)
    return float(np.mean([v.mean() for v in by_seed.values()])), float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/mechanism_gate_exp2")
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; source = root / args.input
    result = pd.read_csv(source / "MECHANISM_GATE_EXP2_RESULTS.csv")
    runs = pd.read_csv(source / "MECHANISM_GATE_EXP2_RUNS.csv")

    base = result[result.condition == "original"][["seed", "fault_id", "auroc", "auprc"]].rename(columns={"auroc": "base_auroc", "auprc": "base_auprc"})
    effects = result[result.condition != "original"].merge(base, on=["seed", "fault_id"])
    effects["auroc_loss"] = effects.base_auroc - effects.auroc
    effects["auprc_loss"] = effects.base_auprc - effects.auprc
    rows = []
    for (condition, fault), group in effects.groupby(["condition", "fault_id"]):
        loss = group.auroc_loss.to_numpy()
        rows.append({"condition": condition, "fault_id": int(fault), "group": "GAIN" if int(fault) in GAIN else "NO_GAIN",
                     "auroc_loss_mean": loss.mean(), "auroc_loss_sd": loss.std(ddof=1), "positive_seeds": int((loss > 0).sum()),
                     "signflip_p": exact_signflip_p(loss)})
    summary = pd.DataFrame(rows)
    summary["bh_q_global"] = bh(summary.signflip_p.to_numpy())
    summary["bh_q_within_fault"] = summary.groupby("fault_id").signflip_p.transform(lambda x: bh(x.to_numpy()))

    base_run = runs[runs.condition == "original"][["seed", "fault_id", "run_index", "pre_fpr"]].rename(columns={"pre_fpr": "base_pre_fpr"})
    paired = runs[runs.condition != "original"].merge(base_run, on=["seed", "fault_id", "run_index"])
    paired["delta"] = paired.pre_fpr - paired.base_pre_fpr
    rng = np.random.default_rng(20260820); fpr_rows = []
    for (condition, fault), group in paired.groupby(["condition", "fault_id"]):
        mean, low, high = hierarchical_ci(group, args.draws, rng)
        fpr_rows.append({"condition": condition, "fault_id": int(fault), "fpr_delta_mean": mean, "fpr_delta_ci_low": low,
                         "fpr_delta_ci_high": high, "ci_outside_tolerance": bool(low > .005 or high < -.005),
                         "ci_fully_inside_tolerance": bool(low >= -.005 and high <= .005)})
    fpr = pd.DataFrame(fpr_rows)
    combined = summary.merge(fpr, on=["condition", "fault_id"])
    combined.to_csv(source / "G1_STATISTICAL_SUMMARY.csv", index=False)
    stable = combined[(combined.group == "GAIN") & (combined.auroc_loss_mean >= .02) & (combined.positive_seeds >= 4)]
    decision = {
        "stable_gain_faults_before_multiplicity": sorted(stable.fault_id.unique().tolist()),
        "globally_fdr_significant_cells": int((combined.bh_q_global < .05).sum()),
        "within_fault_fdr_significant_cells": int((combined.bh_q_within_fault < .05).sum()),
        "fpr_cells_ci_outside_tolerance": int(combined.ci_outside_tolerance.sum()),
        "fpr_cells_ci_fully_inside_tolerance": int(combined.ci_fully_inside_tolerance.sum()),
        "note": "With five seeds, an exact two-sided sign-flip test has minimum attainable p=0.0625; seed-direction stability and hierarchical run uncertainty must be reported rather than misrepresented as p<0.05 evidence."
    }
    (source / "G1_STATISTICAL_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
