# Mechanism Gate Experiment 2

## Decision

**`MODIFY — STRONG STRUCTURE, FPR ROBUSTNESS FOLLOW-UP REQUIRED`.**

The frozen five-seed, no-training perturbation experiment found strong fault-specific variable and temporal structure. Six of seven Experiment-1 GAIN faults met the materiality and seed-direction rule. However, nine fault/seed/condition rows exceeded the preregistered absolute test pre-FPR difference limit of 0.005. The experiment therefore does not yet pass directly to a locked thesis claim.

## Frozen design executed

- F1 checkpoints: seeds 42–46; no retraining.
- Test cohort: all 28 faults and all 20 test runs per fault.
- Evaluation interval: samples 400–900; fault onset at 600.
- Conditions: original, XMV time reversal, retain-last 1/5/10 XMV positions, and individual mean occlusion of XMV01–XMV11.
- Threshold: recalibrated per seed and condition at the validation-normal 99th percentile.
- Outputs: 2,240 fault-level rows and 44,800 run-level rows.

## Main finding

Material, direction-stable effects occurred in faults 4, 7, 19, 24, 25, and 26. Fault 3 did not reach the 0.02 materiality threshold for an individual perturbation.

| Fault | Strongest condition | Mean AUROC loss | Positive seeds |
| ---: | --- | ---: | ---: |
| 4 | occlude XMV10 | 0.4198 | 5/5 |
| 7 | occlude XMV04 | 0.4158 | 5/5 |
| 19 | occlude XMV07 | 0.0763 | 5/5 |
| 24 | occlude XMV01 | 0.0662 | 5/5 |
| 25 | occlude XMV02 | 0.2282 | 5/5 |
| 26 | occlude XMV04 | 0.3676 | 5/5 |

Fault 19 also lost 0.0711 AUROC under XMV08 occlusion. The mappings align closely with the earlier exploratory shift audit, but this experiment measures predictive reliance rather than physical cause.

The mean of each fault's maximum single-variable occlusion loss was 0.2259 in the frozen GAIN group and 0.0082 in the no-gain group. Median values were 0.2282 and 0.0051, respectively.

## Temporal evidence

Mean AUROC loss by group:

| Condition | GAIN | No-gain |
| --- | ---: | ---: |
| retain last 1 | 0.0737 | 0.0026 |
| retain last 5 | 0.0348 | 0.0024 |
| retain last 10 | 0.0195 | 0.0014 |
| reverse XMV time | 0.0109 | 0.0015 |

The lag-truncation gradient supports dependence on more than the latest XMV sample for some GAIN faults. Time reversal was material for faults 4 (0.0294) and 19 (0.0206), both in 5/5 seeds.

## FPR exception and interpretation

Condition-level mean test pre-FPR changes were close to zero (range -0.00072 to +0.00034). Nevertheless, nine individual rows exceeded an absolute change of 0.005. The largest was fault 10, seed 45, under XMV02 occlusion: pre-FPR 0.02075 versus 0.00975 original (difference 0.011). Violations were concentrated in XMV01/XMV02 occlusion for no-gain faults 10, 17, and 23.

This does not explain the large GAIN-fault AUROC losses, but it violates the preregistered universal guardrail and requires a paired run-level uncertainty analysis. Do not weaken the guardrail after seeing the result.

## Claim boundary

Supported: specific past XMV channels and temporal positions carry reproducible incremental predictive information for a subset of faults.

Not supported: XMV causes the fault, controller action causes improved detection, the mapped XMV has a verified physical mechanism, or the result generalizes beyond this Reinartz TEP distribution.

## Required follow-up

1. Apply hierarchical paired run bootstrap and Benjamini–Hochberg correction to AUROC-loss and FPR-difference tests.
2. Diagnose fault-10 XMV01/XMV02 pre-fault distribution shift without tuning thresholds on test data.
3. Report constant channels XMV05 and XMV09 as negative controls (both produced zero change).
4. If the variable effects survive and the FPR exception is bounded honestly, write the thesis around fault-specific predictive information in manipulated-variable histories, not causal control effects.

## Reproducibility

- Preregistration: `outputs/methodology/MECHANISM_GATE_EXP2_PREREGISTRATION.md`
- Configuration: `configs/mechanism_gate_exp2.yaml`
- Executed runner: `experiments/run_mechanism_gate_exp2_fixed.py`
- Fault-level results: `MECHANISM_GATE_EXP2_RESULTS.csv`
- Run-level results: `MECHANISM_GATE_EXP2_RUNS.csv`
- Manifest: `RUN_MANIFEST.json`
