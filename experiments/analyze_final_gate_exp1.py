from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
PHYSICALAI = ROOT / "PhysicalAI_mini"
if str(PHYSICALAI) not in sys.path:
    sys.path.insert(0, str(PHYSICALAI))

from src.config import load_config
from src.data.reinartz_f0_f1 import FAULT_ONSET, N_SAMPLES
from src.experiments.reinartz_f0_f1 import (
    WindowDataset,
    binary_metrics,
    evaluate_scores,
    persistence_delays,
)
from src.models.reinartz_forecaster import GRUForecaster


SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ("F0", "F1", "F0-C")
METRICS = ("auroc", "auprc", "detected_run_ratio", "detection_delay", "prefault_fpr")
T975_DF4 = 2.7764451051977987
FPR_EQUIVALENCE_MARGIN = 0.005  # absolute probability: 0.5 percentage point
MIN_MATERIAL_AUROC_GAIN = 0.02
MIN_MATERIAL_AUPRC_GAIN = 0.01
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20260818


def checkpoint_path(output: Path, variant: str, seed: int) -> Path:
    slug = variant.lower().replace("-", "_")
    local = output / "checkpoints" / f"reinartz_{slug}_seed_{seed}.pt"
    if local.exists():
        return local
    legacy = PHYSICALAI / "checkpoints" / f"reinartz_{slug}_seed_{seed}.pt"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Missing checkpoint for {variant} seed {seed}")


def alarm_outcome(scores: np.ndarray, samples: np.ndarray, threshold: float, consecutive: int) -> tuple[int, float, float]:
    order = np.argsort(samples)
    scores, samples = scores[order], samples[order]
    exceed = scores >= threshold
    alarm = np.convolve(exceed.astype(int), np.ones(consecutive, dtype=int), mode="valid") >= consecutive
    alarm_samples = samples[consecutive - 1 :]
    valid = np.flatnonzero(alarm & (alarm_samples >= FAULT_ONSET))
    detected = int(len(valid) > 0)
    delay = float(alarm_samples[valid[0]] - FAULT_ONSET) if detected else float("nan")
    prefault = samples < FAULT_ONSET
    prefault_fpr = float(np.mean(scores[prefault] >= threshold))
    return detected, delay, prefault_fpr


def score_checkpoint(
    checkpoint: Path,
    variant: str,
    seed: int,
    config: dict,
    features: np.ndarray,
    labels: np.ndarray,
    split: pd.DataFrame,
    device: torch.device,
) -> tuple[list[dict], list[dict], float]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mean = np.asarray(payload["mean"])
    std = np.asarray(payload["std"])
    use_xmv = variant == "F1"
    input_dim = 52 if use_xmv else 41
    hidden_dim = 68 if variant == "F0-C" else int(config["hidden_dim"])
    model = GRUForecaster(input_dim, hidden_dim, int(config["layers"])).to(device)
    model.load_state_dict(payload["state_dict"])

    window = int(config["window"])
    batch = int(config["batch_size"]) * 4
    val_runs = split.loc[split.split == "validation", "run_index"].to_numpy()
    test_runs = split.loc[split.split == "test", "run_index"].to_numpy()
    val_ds = WindowDataset(features, val_runs, np.arange(window, FAULT_ONSET - 1), window, use_xmv, mean, std)
    test_ds = WindowDataset(features, test_runs, np.arange(window, N_SAMPLES), window, use_xmv, mean, std)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, pin_memory=device.type == "cuda")
    val_scores, _, _, _ = evaluate_scores(model, val_loader, device, f"[{variant}/{seed}] validation")
    threshold = float(np.percentile(val_scores, float(config["threshold_percentile"])))
    scores, _, runs, samples = evaluate_scores(model, test_loader, device, f"[{variant}/{seed}] test")
    binary = (labels[runs, samples - 1] != 0).astype(np.int8)
    fault_by_run = split.set_index("run_index").fault_id.to_dict()
    fault_rows: list[dict] = []
    run_rows: list[dict] = []

    for fault in range(1, 29):
        fault_runs = np.array([r for r in test_runs if fault_by_run[r] == fault])
        mask = np.isin(runs, fault_runs)
        f_scores, f_runs, f_samples, f_binary = scores[mask], runs[mask], samples[mask], binary[mask]
        auroc, auprc = binary_metrics(f_binary, f_scores)
        detected, delay, _ = persistence_delays(
            f_scores, f_runs, f_samples, threshold, int(config["alarm_consecutive"])
        )
        prefault = f_samples < FAULT_ONSET
        fault_rows.append(
            {
                "seed": seed,
                "fault_id": fault,
                "variant": variant,
                "auroc": auroc,
                "auprc": auprc,
                "detected_run_ratio": detected,
                "detection_delay": delay,
                "prefault_fpr": float(np.mean(f_scores[prefault] >= threshold)),
                "threshold": threshold,
            }
        )
        for run in fault_runs:
            rm = f_runs == run
            r_auroc, r_auprc = binary_metrics(f_binary[rm], f_scores[rm])
            r_detected, r_delay, r_fpr = alarm_outcome(
                f_scores[rm], f_samples[rm], threshold, int(config["alarm_consecutive"])
            )
            run_rows.append(
                {
                    "seed": seed,
                    "fault_id": fault,
                    "run_index": int(run),
                    "variant": variant,
                    "auroc": r_auroc,
                    "auprc": r_auprc,
                    "detected_run_ratio": r_detected,
                    "detection_delay_censored": r_delay if r_detected else float(N_SAMPLES - FAULT_ONSET + 1),
                    "prefault_fpr": r_fpr,
                }
            )
    return fault_rows, run_rows, threshold


def seed_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, float("nan"), float("nan"), float("nan")
    critical = {2: 12.706204736, 3: 4.30265273, 4: 3.182446305, 5: T975_DF4}[len(values)]
    sd = float(np.std(values, ddof=1))
    half = critical * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def direction(values: np.ndarray) -> str:
    pos = int(np.sum(values > 0))
    neg = int(np.sum(values < 0))
    if pos == 5:
        return "POSITIVE_5_OF_5"
    if pos == 4:
        return "POSITIVE_4_OF_5"
    if neg == 5:
        return "NEGATIVE_5_OF_5"
    if neg == 4:
        return "NEGATIVE_4_OF_5"
    return f"MIXED_POS_{pos}_NEG_{neg}"


def hierarchical_bootstrap(delta: pd.DataFrame, metric: str, rng: np.random.Generator) -> tuple[float, float]:
    by_seed = {s: delta.loc[delta.seed == s, metric].to_numpy() for s in SEEDS}
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for b in range(BOOTSTRAPS):
        sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        seed_means = []
        for seed in sampled_seeds:
            values = by_seed[int(seed)]
            seed_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        draws[b] = np.mean(seed_means)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def wide_results(fault_df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [*METRICS, "threshold"]
    wide = fault_df.pivot(index=["seed", "fault_id"], columns="variant", values=value_cols)
    wide.columns = [f"{variant.lower().replace('-', '')}_{metric}" for metric, variant in wide.columns]
    wide = wide.reset_index()
    for ref, label in (("f0", "f1_minus_f0"), ("f0c", "f1_minus_f0c")):
        for metric in METRICS:
            wide[f"{label}_{metric}"] = wide[f"f1_{metric}"] - wide[f"{ref}_{metric}"]
    return wide.sort_values(["fault_id", "seed"])


def summarize(results: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for fault in range(1, 29):
        row: dict[str, object] = {"fault_id": fault}
        fault_results = results.loc[results.fault_id == fault]
        for ref, label in (("f0", "f1_minus_f0"), ("f0c", "f1_minus_f0c")):
            for metric in METRICS:
                values = fault_results[f"{label}_{metric}"].to_numpy(float)
                mean, sd, low, high = seed_ci(values)
                row[f"{label}_{metric}_mean"] = mean
                row[f"{label}_{metric}_sd"] = sd
                row[f"{label}_{metric}_ci95_low"] = low
                row[f"{label}_{metric}_ci95_high"] = high
                row[f"{label}_{metric}_direction"] = direction(values) if np.all(np.isfinite(values)) else "UNDEFINED_FOR_SOME_SEEDS"
                row[f"{label}_{metric}_n_valid"] = int(np.isfinite(values).sum())

            base = run_df.loc[(run_df.fault_id == fault) & (run_df.variant == ref.upper().replace("C", "-C"))]
            f1 = run_df.loc[(run_df.fault_id == fault) & (run_df.variant == "F1")]
            keys = ["seed", "fault_id", "run_index"]
            merged = f1.merge(base, on=keys, suffixes=("_f1", "_ref"), validate="one_to_one")
            delta = merged[keys].copy()
            delta["auroc"] = merged.auroc_f1 - merged.auroc_ref
            delta["auprc"] = merged.auprc_f1 - merged.auprc_ref
            delta["detected_run_ratio"] = merged.detected_run_ratio_f1 - merged.detected_run_ratio_ref
            delta["delay_improvement_censored"] = merged.detection_delay_censored_ref - merged.detection_delay_censored_f1
            delta["prefault_fpr"] = merged.prefault_fpr_f1 - merged.prefault_fpr_ref
            for metric in ("auroc", "auprc", "detected_run_ratio", "delay_improvement_censored", "prefault_fpr"):
                low, high = hierarchical_bootstrap(delta, metric, rng)
                row[f"{label}_run_boot_{metric}_ci95_low"] = low
                row[f"{label}_run_boot_{metric}_ci95_high"] = high

        stable_f0 = row["f1_minus_f0_auroc_direction"] in {"POSITIVE_5_OF_5", "POSITIVE_4_OF_5"}
        stable_f0c = row["f1_minus_f0c_auroc_direction"] in {"POSITIVE_5_OF_5", "POSITIVE_4_OF_5"}
        seed_sig_f0 = float(row["f1_minus_f0_auroc_ci95_low"]) > 0
        seed_sig_f0c = float(row["f1_minus_f0c_auroc_ci95_low"]) > 0
        run_sig_f0 = float(row["f1_minus_f0_run_boot_auroc_ci95_low"]) > 0
        run_sig_f0c = float(row["f1_minus_f0c_run_boot_auroc_ci95_low"]) > 0
        fpr_equiv_f0 = (
            float(row["f1_minus_f0_run_boot_prefault_fpr_ci95_low"]) >= -FPR_EQUIVALENCE_MARGIN
            and float(row["f1_minus_f0_run_boot_prefault_fpr_ci95_high"]) <= FPR_EQUIVALENCE_MARGIN
        )
        fpr_equiv_f0c = (
            float(row["f1_minus_f0c_run_boot_prefault_fpr_ci95_low"]) >= -FPR_EQUIVALENCE_MARGIN
            and float(row["f1_minus_f0c_run_boot_prefault_fpr_ci95_high"]) <= FPR_EQUIVALENCE_MARGIN
        )
        degraded = (
            float(row["f1_minus_f0_auroc_ci95_high"]) < 0
            and float(row["f1_minus_f0c_auroc_ci95_high"]) < 0
            and float(row["f1_minus_f0_run_boot_auroc_ci95_high"]) < 0
        )
        material = (float(row["f1_minus_f0_auroc_mean"]) >= MIN_MATERIAL_AUROC_GAIN and float(row["f1_minus_f0c_auroc_mean"]) >= MIN_MATERIAL_AUROC_GAIN and float(row["f1_minus_f0_auprc_mean"]) >= MIN_MATERIAL_AUPRC_GAIN and float(row["f1_minus_f0c_auprc_mean"]) >= MIN_MATERIAL_AUPRC_GAIN)
        if stable_f0 and stable_f0c and seed_sig_f0 and seed_sig_f0c and run_sig_f0 and run_sig_f0c and fpr_equiv_f0 and fpr_equiv_f0c and material:
            classification = "GAIN"
        elif degraded:
            classification = "DEGRADED"
        else:
            classification = "NEUTRAL"
        row["fpr_equivalent_f1_vs_f0"] = fpr_equiv_f0
        row["fpr_equivalent_f1_vs_f0c"] = fpr_equiv_f0c
        row["classification"] = classification
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(output: Path, summary: pd.DataFrame, decision: dict) -> None:
    gains = summary.loc[summary.classification == "GAIN", "fault_id"].astype(int).tolist()
    neutral = summary.loc[summary.classification == "NEUTRAL", "fault_id"].astype(int).tolist()
    degraded = summary.loc[summary.classification == "DEGRADED", "fault_id"].astype(int).tolist()
    top = summary.sort_values("f1_minus_f0_auroc_mean", ascending=False).head(8)
    table = top[["fault_id", "classification", "f1_minus_f0_auroc_mean", "f1_minus_f0c_auroc_mean", "f1_minus_f0_auprc_mean"]].copy()
    table.columns = ["Fault", "Class", "ΔAUROC F1-F0", "ΔAUROC F1-F0-C", "ΔAUPRC F1-F0"]
    for col in table.columns[2:]:
        table[col] = table[col].map(lambda x: f"{x:+.4f}")

    lines = [
        "# PhysicalAI Final Gate — Experiment 1",
        "",
        "## Technical summary",
        "",
        f"**Final judgment: `{decision['judgment']}`.** {decision['rationale']}",
        "",
        f"Post-evaluation classes: **GAIN {len(gains)}**, **NEUTRAL {len(neutral)}**, **DEGRADED {len(degraded)}** across all 28 faults. No fault was selected before evaluation.",
        "",
        "This gate establishes only whether a reproducible, fault-specific F1 effect survives the capacity, false-positive-rate, seed, and run-level controls. It does not establish a mechanism and does not lock a thesis topic.",
        "",
        "## The effect is fault-specific, not universal",
        "",
        f"GAIN faults: `{gains}`. NEUTRAL faults: `{neutral}`. DEGRADED faults: `{degraded}`.",
        "",
        "| " + " | ".join(table.columns) + " |`n" + "| " + " | ".join(["---"] * len(table.columns)) + " |`n" + "`n".join("| " + " | ".join(map(str, values)) + " |" for values in table.itertuples(index=False, name=None)),
        "",
        "The table shows the eight largest mean AUROC changes. Exact results for every fault and seed, including detection ratio, detected-only delay, and pre-fault FPR, are in `FINAL_GATE_EXP1_RESULTS.csv`; complete uncertainty columns are in `FINAL_GATE_EXP1_FAULT_SUMMARY.csv`.",
        "",
        "## Scope and metric definitions",
        "",
        "- Cohort: the fixed leakage-controlled test split, 20 runs per fault and 28 faults (560 fault runs), reused for all models and seeds.",
        "- F0: 20-step XMEAS history only, hidden size 64 (23,209 parameters).",
        "- F1: 20-step XMEAS+XMV history, hidden size 64 (25,321 parameters).",
        "- F0-C: XMEAS only, hidden size 68 (25,473 parameters), capacity matched to F1.",
        "- AUROC and AUPRC use all scored samples from sample 21 through 2000 for the fault's test runs.",
        "- Detection requires three consecutive threshold exceedances. Thresholds are the 99th percentile of validation-normal scores, fitted separately for each trained model.",
        "- Detected-run ratio is the fraction of the 20 test runs with a post-onset alarm. Detection delay is the mean delay among detected runs only.",
        "- Pre-fault FPR is the sample-level exceedance fraction before onset (sample < 600) within the same fault's test runs.",
        "",
        "## Experimental and statistical design",
        "",
        "All three variants were evaluated at seeds 42–46 with the same split (split seed 42), preprocessing, 30 epochs, optimizer, window, and alarm rule. For every fault and comparison, the report gives the mean, sample SD, and two-sided 95% t interval across five paired seed deltas.",
        "",
        f"Run-level uncertainty uses {BOOTSTRAPS:,} hierarchical paired bootstrap draws: seeds are resampled first and the 20 matched runs are resampled within each selected seed. For delay uncertainty, an undetected run receives a censored delay of {N_SAMPLES - FAULT_ONSET + 1}; positive reported delay improvement means F1 is earlier or converts non-detection to detection. The requested detected-only delay remains in the primary CSV.",
        "",
        "Classification was applied after all results were computed. GAIN requires: mean ΔAUROC ≥ 0.02 and mean ΔAUPRC ≥ 0.01 against both controls; positive AUROC direction in at least 4/5 seeds; seed-level and run-bootstrap AUROC 95% intervals above zero for both comparisons; and the paired pre-fault-FPR bootstrap interval fully within ±0.005. DEGRADED requires negative seed intervals against both controls and a negative run-bootstrap interval against F0. All remaining faults are NEUTRAL.",
        "",
        "## Falsification checks",
        "",
    ]
    for check in decision["kill_criteria"]:
        lines.append(f"- **{check['criterion']} — {check['status']}.** {check['evidence']}")
    lines += [
        "",
        "## Limitations and uncertainty",
        "",
        "- The five training seeds quantify optimization variability but do not create five independent datasets; the hierarchical bootstrap therefore preserves seed and run nesting rather than treating all 100 observations as independent.",
        "- AUROC/AUPRC are sample-weighted and temporally autocorrelated. Run-level paired intervals are the stronger guard against a few long trajectories dominating the conclusion.",
        "- Detected-only delay can improve when difficult runs become undetected; the censored run-level delay analysis is included specifically to expose that failure mode.",
        "- A PASS is permission only for a prespecified mechanism experiment. It is not evidence that XMV is causally responsible, that the result generalizes beyond this benchmark, or that the thesis direction is already defensible.",
        "",
        "## Next step if approved",
        "",
        decision["next_step"],
        "",
        "## Further questions",
        "",
        "- Which actuator variables and temporal lags account for the stable GAIN faults?",
        "- Can the same mechanism predict the stable no-gain group without fitting to the observed classification?",
        "- Does the effect survive action-history permutation, lag truncation, and variable-group ablation at a comparable FPR?",
        "",
        "## Reproducibility record",
        "",
        "Configuration: `configs/final_gate_exp1.yaml`. Capacity runner: `experiments/final_gate_capacity_runs.py`. Final evaluator: `experiments/analyze_final_gate_exp1.py`. Model checkpoints and intermediate tables are under `outputs/final_gate_exp1/`. Source PhysicalAI code and data were read but not modified.",
        "",
        "WAITING_FOR_USER_APPROVAL",
    ]
    (output / "FINAL_GATE_EXP1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = ROOT / "Thesis-Orchestrator" / "outputs" / "final_gate_exp1"
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    split = pd.read_csv(output / "artifacts" / "reinartz_split_manifest.csv")
    features = np.load(Path(config["paths"]["cache"]) / "features.npy", mmap_mode="r")
    labels = np.load(Path(config["paths"]["cache"]) / "metadata.npz")["labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_faults, all_runs = [], []
    for seed in SEEDS:
        for variant in VARIANTS:
            checkpoint = checkpoint_path(output, variant, seed)
            logging.info("Scoring %s seed %d from %s", variant, seed, checkpoint)
            fault_rows, run_rows, threshold = score_checkpoint(
                checkpoint, variant, seed, config, features, labels, split, device
            )
            all_faults.extend(fault_rows)
            all_runs.extend(run_rows)
            logging.info("Scored %s seed %d threshold=%.6f", variant, seed, threshold)

    fault_df = pd.DataFrame(all_faults)
    run_df = pd.DataFrame(all_runs)
    results = wide_results(fault_df)
    summary = summarize(results, run_df)
    results.to_csv(output / "FINAL_GATE_EXP1_RESULTS.csv", index=False)
    summary.to_csv(output / "FINAL_GATE_EXP1_FAULT_SUMMARY.csv", index=False)

    gains = summary.loc[summary.classification == "GAIN", "fault_id"].astype(int).tolist()
    no_gain = summary.loc[summary.classification != "GAIN", "fault_id"].astype(int).tolist()
    apparent = summary[
        (summary.f1_minus_f0_auroc_mean >= MIN_MATERIAL_AUROC_GAIN)
        & (summary.f1_minus_f0_auprc_mean >= MIN_MATERIAL_AUPRC_GAIN)
        & (summary.f1_minus_f0_auroc_ci95_low > 0)
        & (summary.f1_minus_f0_run_boot_auroc_ci95_low > 0)
    ]
    capacity_retention = len(gains) / len(apparent) if len(apparent) else 0.0
    substantial_reversal = len(gains) == 0
    capacity_explains = capacity_retention < 0.5
    no_structure = len(gains) < 2 or len(no_gain) < 2
    run_uncertainty_erases = len(gains) == 0
    killed = substantial_reversal or capacity_explains or no_structure or run_uncertainty_erases
    judgment = "KILL_PHYSICALAI_THESIS_DIRECTION" if killed else "PASS_TO_MECHANISM_EXPERIMENT"
    decision = {
        "experiment": "PHYSICALAI FINAL GATE — EXPERIMENT 1",
        "judgment": judgment,
        "thesis_topic_locked": False,
        "gain_faults": gains,
        "neutral_faults": summary.loc[summary.classification == "NEUTRAL", "fault_id"].astype(int).tolist(),
        "degraded_faults": summary.loc[summary.classification == "DEGRADED", "fault_id"].astype(int).tolist(),
        "rationale": (
            "At least two stable GAIN faults survive both capacity matching and paired run-level uncertainty, while at least two faults remain no-gain; the observed structure is therefore suitable for a falsifiable mechanism study."
            if not killed
            else "The final gate failed at least one prespecified scientific kill criterion; the current PhysicalAI thesis direction should not proceed."
        ),
        "kill_criteria": [
            {
                "criterion": "effects reverse substantially across seeds",
                "status": "TRIGGERED" if substantial_reversal else "NOT_TRIGGERED",
                "evidence": f"{len(gains)} faults meet the strict stable-GAIN rule across seeds.",
            },
            {
                "criterion": "F0-C explains most F1 gains",
                "status": "TRIGGERED" if capacity_explains else "NOT_TRIGGERED",
                "evidence": f"{len(gains)} faults retain significant F1-F0-C AUROC headroom under seed and run-level intervals.",
            },
            {
                "criterion": "no stable fault-specific structure exists",
                "status": "TRIGGERED" if no_structure else "NOT_TRIGGERED",
                "evidence": f"GAIN faults={gains}; no-gain faults={no_gain}.",
            },
            {
                "criterion": "apparent gains disappear after run-level uncertainty analysis",
                "status": "TRIGGERED" if run_uncertainty_erases else "NOT_TRIGGERED",
                "evidence": f"{len(gains)} faults retain a positive hierarchical paired-bootstrap AUROC interval against both controls.",
            },
        ],
        "next_step": (
            "Run only a separately approved, prespecified mechanism experiment on the stable GAIN and no-gain groups. Do not lock the thesis topic from Experiment 1 alone."
            if not killed
            else "Archive the result and do not run Experiment 2 for this thesis direction."
        ),
        "artifacts": [
            "FINAL_GATE_EXP1_RESULTS.csv",
            "FINAL_GATE_EXP1_FAULT_SUMMARY.csv",
            "FINAL_GATE_EXP1_REPORT.md",
            "FINAL_GATE_EXP1_DECISION.json",
        ],
        "status": "WAITING_FOR_USER_APPROVAL",
    }
    (output / "FINAL_GATE_EXP1_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output, summary, decision)
    logging.info("Decision: %s | GAIN=%s", judgment, gains)


if __name__ == "__main__":
    main()
