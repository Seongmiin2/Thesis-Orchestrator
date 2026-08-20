# Architecture Gate G2 Report

## Decision

**PASS — EVENT-LEVEL ARCHITECTURE CONSENSUS.**

The control-history gain is not specific to the original GRU. Under the frozen split, validation calibration, five seeds, and capacity controls, both TCN and compact Transformer identified the same seven GAIN faults: 4, 7, 19, 23, 24, 25, and 26.

The four faults that survived the stricter normal-conditional CHUM replacement in G1 (4, 19, 25, 26) all retained material F1 gains in GRU, TCN, and Transformer. This supports an architecture-robust event-level information effect. It does not yet establish architecture consensus for individual XMV channel attribution.

## Execution

- Architectures: two-layer TCN and two-layer compact Transformer.
- Models: F0 XMEAS-only, F1 XMEAS+XMV, F0-C capacity-matched XMEAS-only.
- Seeds: 42–46; ten epochs each; 30 trained and fully evaluated models.
- Cohort: original leakage-controlled split; all 28 faults and 20 test runs/fault.
- Thresholds: model-specific validation-normal 99th percentile.
- TCN parameters: F0 8,425; F1 9,481; F0-C 9,526.
- Transformer parameters: F0/F0-C 20,489; F1 20,841 (1.7% difference).

## Stable GAIN faults

GAIN requires mean AUROC improvement >=0.02 and mean AUPRC improvement >=0.01 against both F0 and F0-C, with positive AUROC direction in at least 4/5 seeds for both comparisons.

| Fault | TCN ΔAUROC vs F0-C | Transformer ΔAUROC vs F0-C | Direction |
| ---: | ---: | ---: | ---: |
| 4 | 0.1663 | 0.1087 | 5/5 both |
| 7 | 0.4880 | 0.4845 | 5/5 both |
| 19 | 0.1783 | 0.1970 | 5/5 both |
| 23 | 0.0300 | 0.0234 | 5/5 both |
| 24 | 0.1091 | 0.0877 | 5/5 both |
| 25 | 0.2916 | 0.2776 | 5/5 both |
| 26 | 0.3868 | 0.4154 | 5/5 both |

Fault 23 is new relative to the strict GRU Experiment-1 classification and must be reported as architecture-dependent/near-threshold, not added retrospectively to the original GRU GAIN group.

## Relation to conditional attribution

Normal-only conditional XMV replacement removed the prior zero-occlusion FPR exceptions: maximum absolute test pre-FPR change was 0.002, with zero rows above 0.005. It retained material single-channel effects for:

- fault 4 — conditional replacement of XMV10: mean AUROC loss 0.4191, 5/5 seeds;
- fault 19 — XMV08: 0.0289, 5/5;
- fault 25 — XMV02: 0.0217, 5/5;
- fault 26 — XMV04: 0.0840, 5/5.

These four events have architecture-robust *overall* control-history gains. The channel identities above are currently validated only in frozen GRU checkpoints. Applying conditional CHUM to TCN and Transformer is required before claiming architecture-robust channel attribution.

## Statistical limitation

With five seeds, a two-sided exact sign-flip test has minimum attainable p=0.0625. Therefore seed-level Benjamini–Hochberg q<0.05 cannot be achieved regardless of perfect 5/5 direction agreement. The thesis must report effect sizes, all seed values, run-level hierarchical intervals, and this resolution limit. More seeds are required only if formal seed-level p<0.05 is made a mandatory claim.

## Claim now supported

On this Reinartz TEP distribution, past XMV history supplies strongly fault-specific predictive information that survives model-capacity control and recurs across GRU, TCN, and compact Transformer forecasters.

## Claims not yet supported

- architecture-robust attribution to the same individual XMV channels;
- physical or controller causality;
- transfer to a real or hardware-in-the-loop control system;
- superiority of CHUM over all existing attribution methods.

## Next gate

1. Apply conditional CHUM to TCN and Transformer checkpoints for faults 4, 19, 25, and 26 plus all-fault negative controls.
2. Replicate event-level heterogeneity on HAI; use its point-role metadata and official time-aware metric.
3. Use Integrated Gradients or a classical contribution method as an attribution baseline.

## Reproducibility

- Frozen protocol: `outputs/methodology/ARCHITECTURE_GATE_G2_PREREGISTRATION.md`
- Configuration: `configs/architecture_gate_g2_frozen.yaml`
- Runner: `experiments/run_architecture_gate_g2.py`
- Metrics: `G2_METRICS.csv`
- Fault results: `G2_FAULT_RESULTS.csv`
- Summary: `G2_FAULT_SUMMARY.csv`
