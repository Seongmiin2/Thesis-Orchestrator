# Project Handover

This handover distinguishes `CONFIRMED FACT`, `INTERPRETATION`, and `HYPOTHESIS`. The source projects were inspected read-only on 2026-08-18.

## A. PhysicalAI_mini

### Purpose and data

- **CONFIRMED FACT:** The project studies industrial CPS anomaly detection and recovery, with the current research focus on control-history-associated state variation and residuals.
- **CONFIRMED FACT:** The Reinartz TEP distribution contains 2,800 runs, 28 faults, 2,000 samples per run, 41 XMEAS variables and 11 available XMV variables. Fault onset is sample 600. XMV12 and operating mode are absent.
- **CONFIRMED FACT:** TEP is simulated process data and XMV represents manipulated-variable history in a closed loop, not independent causal intervention.

### Methodology

- **CONFIRMED FACT:** F0 uses the previous 20 samples of 41 XMEAS variables to predict the next XMEAS vector with a one-layer GRU.
- **CONFIRMED FACT:** F1 adds 11 XMV histories to the same forecasting setup. Seed-42 F0/F1 parameter counts are 23,209/25,321.
- **CONFIRMED FACT:** Splits are run-level and fault-stratified; the scaler is fit only on normal training samples; windows do not cross runs; thresholds use validation pre-fault scores; alarms require three consecutive exceedances.

### Completed experiments and results

- **CONFIRMED FACT:** Seed-42 pooled F0→F1 results were AUROC 0.75077→0.81965, AUPRC 0.89950→0.93123, detected-run ratio 0.64821→0.68036, delay 51.87→24.58 samples, at essentially equal pre-fault sample FPR (~0.01).
- **CONFIRMED FACT:** Normal forecasting changed only marginally (MAE 0.49748→0.49684; RMSE 0.73607→0.73379).
- **CONFIRMED FACT:** Directional detection improvement repeated across seeds 42–44. A one-run matched-capacity F0-C check stayed near F0 and did not reproduce F1.
- **CONFIRMED FACT:** Effects are heterogeneous; mechanism auditing emphasizes faults 19, 24, 25 and 26 and classifies the evidence as `MIXED_MECHANISM`.
- **INTERPRETATION:** XMV history appears more useful for residual separability in some faults than for general normal forecasting.
- **HYPOTHESIS:** Operating/control context may determine when a historically similar episode is applicable to a current episode.

### Risks and required work

- Causal-control claims are unsupported; dataset provenance and missing XMV12/mode limit interpretation.
- The mathematical novelty gate is `INCONCLUSIVE`; prior work overlaps with input-conditioned prediction residuals.
- F0/F1 experiments are closed in the reference project. The documented next action is a preregistered, no-training, all-fault mechanism diagnostic; Mercer 2002 and Ji 2024 full-text verification remains outstanding.

## B. FAVE-RAG

### Purpose and methodology

- **CONFIRMED FACT:** The project targets evidence that is relevant and often true but invalid for the current procedure due to units, variable binding, applicability conditions, physical constraints or corrupted steps.
- **CONFIRMED FACT:** Benchmark fields include valid, invalid and contested evidence plus expected arbitration. Baselines include LLM-only, Vanilla RAG, CRAG-style evaluation, DeMo-style reasoning and FAVE arbitration.
- **CONFIRMED FACT:** The seed benchmark has 10 hand-crafted items. Forty real-dataset-derived rows are annotation candidates, not evaluated gold benchmark items.

### Results and limitations

- **CONFIRMED FACT:** The README records a real 10-item OpenAI pilot: mixed accuracy 0.70 for LLM-only, Vanilla RAG and FAVE-RAG; 0.80 for DeMo-style; FAVE conflict-detection F1 0.842.
- **CONFIRMED FACT:** This does not show a FAVE robustness gain. The repository also contains mock/example CSVs that must not be confused with real evidence.
- **CONFIRMED FACT:** Required next work includes a larger 40–50 item benchmark, closed-book filtering, stronger inapplicable distractors, varying evidence ratios, a factuality baseline and double labeling.
- **INTERPRETATION:** The strongest reusable contribution is the separation of relevance/factuality from contextual applicability and the explicit valid/invalid/contested evidence ledger.

## C. Possible bridge

- **HYPOTHESIS:** `Relevance != Validity` may transfer to industrial time series as `Similarity != Applicability`.
- **HYPOTHESIS:** A retrieved historical window could be close in sensor space yet unsuitable because its control regime, operating context, sensor–action relationship or fault mechanism differs.
- This bridge is not a confirmed research direction. It needs literature validation, a formal applicability definition, controlled invalid-reference construction, and downstream evaluation before it can support a thesis claim.

