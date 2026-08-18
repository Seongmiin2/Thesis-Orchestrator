# PhysicalAI Final Gate — Experiment 1

## Technical summary

**Final judgment: `PASS_TO_MECHANISM_EXPERIMENT`.** Seven materially non-trivial, seed-stable GAIN faults survive capacity matching, comparable pre-fault FPR, and hierarchical paired run-level uncertainty, while 21 faults remain no-gain; this supports only a falsifiable mechanism experiment.

Post-evaluation classes: **GAIN 7**, **NEUTRAL 21**, **DEGRADED 0** across all 28 faults. No fault was selected before evaluation.

This gate establishes only whether a reproducible, fault-specific F1 effect survives the capacity, false-positive-rate, seed, and run-level controls. It does not establish a mechanism and does not lock a thesis topic.

## The effect is fault-specific, not universal

GAIN faults: `[3, 4, 7, 19, 24, 25, 26]`. NEUTRAL faults: `[1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 27, 28]`. DEGRADED faults: `[]`.

| Fault | Class | ΔAUROC F1-F0 | ΔAUROC F1-F0-C | ΔAUPRC F1-F0 |`n| --- | --- | --- | --- | --- |`n| 7 | GAIN | +0.4870 | +0.4849 | +0.2730 |`n| 4 | GAIN | +0.4349 | +0.4368 | +0.2573 |`n| 26 | GAIN | +0.4114 | +0.4083 | +0.2414 |`n| 25 | GAIN | +0.2312 | +0.2265 | +0.1216 |`n| 19 | GAIN | +0.2223 | +0.2184 | +0.0990 |`n| 24 | GAIN | +0.0801 | +0.0777 | +0.0313 |`n| 27 | NEUTRAL | +0.0461 | +0.0404 | +0.0219 |`n| 11 | NEUTRAL | +0.0212 | +0.0235 | +0.0080 |

The table shows the eight largest mean AUROC changes. Exact results for every fault and seed, including detection ratio, detected-only delay, and pre-fault FPR, are in `FINAL_GATE_EXP1_RESULTS.csv`; complete uncertainty columns are in `FINAL_GATE_EXP1_FAULT_SUMMARY.csv`.

## Scope and metric definitions

- Cohort: the fixed leakage-controlled test split, 20 runs per fault and 28 faults (560 fault runs), reused for all models and seeds.
- F0: 20-step XMEAS history only, hidden size 64 (23,209 parameters).
- F1: 20-step XMEAS+XMV history, hidden size 64 (25,321 parameters).
- F0-C: XMEAS only, hidden size 68 (25,473 parameters), capacity matched to F1.
- AUROC and AUPRC use all scored samples from sample 21 through 2000 for the fault's test runs.
- Detection requires three consecutive threshold exceedances. Thresholds are the 99th percentile of validation-normal scores, fitted separately for each trained model.
- Detected-run ratio is the fraction of the 20 test runs with a post-onset alarm. Detection delay is the mean delay among detected runs only.
- Pre-fault FPR is the sample-level exceedance fraction before onset (sample < 600) within the same fault's test runs.

## Experimental and statistical design

All three variants were evaluated at seeds 42–46 with the same split (split seed 42), preprocessing, 30 epochs, optimizer, window, and alarm rule. For every fault and comparison, the report gives the mean, sample SD, and two-sided 95% t interval across five paired seed deltas.

Run-level uncertainty uses 2,000 hierarchical paired bootstrap draws: seeds are resampled first and the 20 matched runs are resampled within each selected seed. For delay uncertainty, an undetected run receives a censored delay of 1401; positive reported delay improvement means F1 is earlier or converts non-detection to detection. The requested detected-only delay remains in the primary CSV.

Classification was applied after all results were computed. GAIN requires: mean ΔAUROC ≥ 0.02 and mean ΔAUPRC ≥ 0.01 against both controls; positive AUROC direction in at least 4/5 seeds; seed-level and run-bootstrap AUROC 95% intervals above zero for both comparisons; and the paired pre-fault-FPR bootstrap interval fully within ±0.005. DEGRADED requires negative seed intervals against both controls and a negative run-bootstrap interval against F0. All remaining faults are NEUTRAL.

## Falsification checks

- **effects reverse substantially across seeds — NOT_TRIGGERED.** 7 materially non-trivial faults retain a positive direction in at least 4/5 seeds against both controls.
- **F0-C explains most F1 gains — NOT_TRIGGERED.** 7/8 (87.5%) material F1-v-F0 gains retain material and statistically supported headroom over F0-C.
- **no stable fault-specific structure exists — NOT_TRIGGERED.** GAIN faults=[3, 4, 7, 19, 24, 25, 26]; no-gain faults=[1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 27, 28].
- **apparent gains disappear after run-level uncertainty analysis — NOT_TRIGGERED.** 7 material faults retain positive hierarchical paired-bootstrap AUROC intervals against both controls.

## Limitations and uncertainty

- The five training seeds quantify optimization variability but do not create five independent datasets; the hierarchical bootstrap therefore preserves seed and run nesting rather than treating all 100 observations as independent.
- AUROC/AUPRC are sample-weighted and temporally autocorrelated. Run-level paired intervals are the stronger guard against a few long trajectories dominating the conclusion.
- Detected-only delay can improve when difficult runs become undetected; the censored run-level delay analysis is included specifically to expose that failure mode.
- A PASS is permission only for a prespecified mechanism experiment. It is not evidence that XMV is causally responsible, that the result generalizes beyond this benchmark, or that the thesis direction is already defensible.

## Next step if approved

Run only a separately approved, prespecified mechanism experiment on the stable GAIN and no-gain groups. Do not lock the thesis topic from Experiment 1 alone.

## Further questions

- Which actuator variables and temporal lags account for the stable GAIN faults?
- Can the same mechanism predict the stable no-gain group without fitting to the observed classification?
- Does the effect survive action-history permutation, lag truncation, and variable-group ablation at a comparable FPR?

## Reproducibility record

Configuration: `configs/final_gate_exp1.yaml`. Capacity runner: `experiments/final_gate_capacity_runs.py`. Final evaluator: `experiments/analyze_final_gate_exp1.py`. Model checkpoints and intermediate tables are under `outputs/final_gate_exp1/`. Source PhysicalAI code and data were read but not modified.

WAITING_FOR_USER_APPROVAL
