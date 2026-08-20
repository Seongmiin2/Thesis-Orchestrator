# Mechanism Gate Experiment 2 — Preregistration

Status: `AUTHORIZED / FROZEN BEFORE PERTURBED TEST EVALUATION`

## Question and claim boundary

Which available manipulated-variable histories and temporal positions carry the incremental predictive information associated with the seed-stable, fault-specific F1 detection gains? This is an information-structure experiment, not a causal intervention study. XMV may contain controller responses to a fault.

## Frozen cohorts

- Stable GAIN faults from Experiment 1: 3, 4, 7, 19, 24, 25, 26.
- No-gain comparator: every other fault (21 faults), retained to avoid cherry-picking.
- Frozen models: F1 checkpoints at seeds 42–46. No retraining or test-dependent model selection.
- Frozen test window: samples 400–900; onset is sample 600. This targets pre-onset calibration and the first 300 fault samples.

## Perturbations

All XMV replacements use the checkpoint's training mean (zero after scaling).

1. `permute_time`: reverse the 20 XMV time positions while leaving XMEAS unchanged.
2. `keep_last_{1,5,10}`: retain only the most recent k XMV positions and mean-occlude earlier positions.
3. `occlude_XMV_01..11`: mean-occlude one complete XMV history at a time.

Each checkpoint/perturbation receives a separately calibrated threshold: the 99th percentile score on samples 400–599 of all validation runs. This preserves the matched false-alarm principle.

## Outcomes

- Primary: fault-wise AUROC loss relative to unperturbed F1 in samples 400–900.
- Secondary: AUPRC loss, detected-run ratio, censored alarm delay (301 for no alarm), and pre-onset FPR.
- Variable importance: AUROC loss under single-XMV occlusion, summarized across five paired seeds.
- Temporal evidence: AUROC loss under reversal and lag truncation.

## Decision rule

`PASS_TO_THESIS_DRAFT` requires all of the following:

1. At least four of seven frozen GAIN faults lose at least 0.02 mean AUROC under one temporal perturbation or one XMV occlusion.
2. The responsible perturbation has the same loss direction in at least four of five seeds.
3. The mean maximum single-variable occlusion loss is larger in GAIN than no-gain faults.
4. Perturbed pre-onset FPR remains within 0.005 absolute of the original condition after condition-specific calibration.

Otherwise the result is `MODIFY` if stable descriptive structure exists but group discrimination fails, and `KILL` if neither stable temporal nor variable structure survives.

Multiplicity-adjusted inference and hierarchical run bootstrap are required before a final thesis claim. This gate may authorize drafting, but it cannot establish physical or controller causality.
