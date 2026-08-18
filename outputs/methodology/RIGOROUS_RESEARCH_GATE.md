# Rigorous Scientific Thesis Gate

> Decision status: **WAITING_FOR_USER_APPROVAL**  
> Evidence cutoff: 2026-08-18  
> Decision basis: scientific strength first; 4–6 week intensive execution second.

## Technical summary

Candidate A is scientifically insufficient as an XMEAS-versus-XMEAS+XMV comparison. It remains viable only if upgraded into a mechanism-level, statistically reliable account of *why* control history changes detection for particular faults, variables, and lags.

Candidate B has the stronger potential contribution, but its novelty is not established by the slogan `Similarity != Applicability`. Case-based reasoning already distinguishes similarity from adaptability, and recent time-series methods already retrieve historical examples. Candidate B earns a scientific GO only if applicability is defined independently of the downstream prediction, labeled objectively, and shown to add value beyond strong context-aware retrieval and adaptation baselines.

An emergent combined direction is scientifically coherent: **applicability-verified historical evidence for control-aware early fault detection**. Candidate A would provide preliminary heterogeneity evidence and the control/lag ablations inside Candidate B, not a separate topic. This direction is not locked.

## 1. Recent prior-work matrix

| Work | Year | Area | What it already covers | Direct threat / remaining gap | Verification |
|---|---:|---|---|---|---|
| Hsu, Frusque & Fink, *A Comparison of Residual-based Methods on Fault Detection* | 2023 | Input-output residual FD | Conditions sensor prediction on operating inputs; sensor-wise residual interpretation and delay | Strong threat to A if A is only input-conditioned residual detection; does not test a separate retrieval-applicability verifier | Primary arXiv abstract inspected |
| Jiang et al., nonlinear FD with modeling errors | 2022/2023 | Data-driven residual FD | Nonlinear input-output residual generator and quantile thresholding | Threatens generic residual/threshold novelty; A needs mechanism and controlled comparative inference | IEEE record inspected |
| Continuous Test-time Domain Adaptation for Efficient Fault Detection | 2024 | Changing operating conditions | Adapts FD under evolving operating conditions | Threatens claims that handling operating context is itself novel | Primary arXiv record identified; full method audit required |
| TimeRAG | 2024/2025 | TS retrieval forecasting | DTW retrieval of similar historical sequences for LLM forecasting | Direct similarity-retrieval baseline for B | IEEE/arXiv records inspected |
| RAF, *Retrieval Augmented Time Series Forecasting* | 2024/2025 | TS foundation models | Retrieves related examples and integrates them into forecasts; includes mechanistic studies | Strong direct retrieval baseline; B must isolate post-retrieval verification | arXiv/OpenReview records inspected |
| TimeRAF | 2024/2025 | Learned TS retrieval | End-to-end learned retriever and channel prompting | Threatens novelty based on learnable/contextual retrieval alone | arXiv metadata identified; full text required |
| RATFM | 2025 | Retrieval-augmented TS anomaly detection | Uses retrieved normal target-domain examples for anomaly detection adaptation | Direct downstream anomaly-detection baseline for B | Primary arXiv abstract inspected |
| Forecast2Anomaly | 2025 | Retrieval-augmented anomaly prediction | Retrieves historical horizons and conditions anomaly prediction across 16 datasets | Very strong overlap threat; exact contamination handling and retrieval constraints require full-text audit | Primary arXiv abstract inspected |
| CRAG | 2024 | Corrective RAG | Retrieval quality evaluator with corrective actions | Baseline for post-retrieval filtering, but targets document relevance/quality rather than industrial case transfer validity | Primary arXiv record inspected |
| ASTUTE RAG | 2024/2025 | Imperfect retrieval/conflict | Source-aware consolidation of internal and external evidence under conflict | Strong robustness baseline concept; does not itself establish industrial time-series applicability labels | ACL/arXiv records inspected |
| Karray et al., adaptation-guided industrial diagnostic CBR | 2013 | Case adaptation | Explicitly states the most similar case need not be most adaptable; retrieves using similarity and adaptation measures | Major conceptual novelty threat to B; B must differ via measurable temporal sensor-control-mechanism validity and controlled robustness | Publisher abstract inspected |

**Novelty conclusion:** neither A nor B currently passes a novelty gate. A overlaps input-conditioned residual monitoring. B overlaps adaptation-guided CBR and rapidly emerging retrieval-augmented time-series models. The plausible gap is a *separately supervised or rule-auditable post-retrieval applicability verifier* for industrial temporal evidence, evaluated with sensor-similar hard negatives and early-detection outcomes. This remains a hypothesis until full-text comparison is complete.

## 2. Formal problem formulation

Let a query window be

```text
q_t = (X[t-L+1:t], U[t-L+1:t], r_t, metadata_t)
```

and a historical case be

```text
c_i = (X_i, U_i, regime_i, mechanism_i, outcome_i, provenance_i).
```

A retriever returns `R_k(q)` by a similarity score `s(q,c)`. Applicability must not be defined as similarity or as “the downstream model predicted correctly.” Define an independently auditable eligibility vector:

```text
v(q,c) = [
  regime_compatible,
  control_response_compatible,
  temporal_lag_compatible,
  variable_support_compatible,
  mechanism_transfer_compatible,
  provenance_acceptable
].
```

For a preregistered policy `P`, gold applicability is:

```text
A*(q,c) = P(v(q,c)) in {APPLICABLE, INAPPLICABLE, CONTESTED}.
```

The learned or rule-based verifier estimates `g(q,c) -> A`. It operates **after retrieval**. The testable distinction is:

```text
s(q,c) is high  does not imply  A*(q,c) = APPLICABLE.
```

The downstream detector `h(q, accepted_cases)` must be evaluated at a matched pre-fault false-alarm rate. The verifier contributes only if it improves robustness or detection outcomes beyond retrievers that already use regime, controls, and learned covariates.

## 3. Objective case and ground-truth construction

1. Split by run before building the case bank; never retrieve from the query run or a duplicated trajectory.
2. Build query/case pairs only after freezing train/validation/test and normal-only preprocessing.
3. Construct `APPLICABLE` cases from independently known simulator/run metadata and preregistered compatibility predicates.
4. Construct hard `INAPPLICABLE` cases by matching sensor similarity distributions while violating exactly one controlled factor: regime, XMV response direction/magnitude, lag, variable support, or known fault mechanism.
5. Use `CONTESTED` when metadata cannot determine transfer eligibility; exclude these from primary binary claims and report them separately.
6. Prevent circularity: fault labels and post-onset outcomes may define gold evaluation strata, but cannot be exposed to the online retriever/verifier unless the operational scenario genuinely provides them.
7. Double-label a stratified subset using a written protocol; report agreement and adjudication. Simulator-derived labels and human labels must be reported separately.

## 4. Candidate hypotheses

### A hypotheses

- **A-H1:** At matched pre-fault FPR, XMV history changes detection performance heterogeneously across fault mechanisms rather than uniformly.
- **A-H2:** The faults with reproducible F1 gains have identifiable variable-level XMV contributions and lead/lag structure that precede sensor residual separation.
- **A-H3:** Detection gains can occur without material average prediction-error gains because XMV conditioning changes the *distribution and ranking* of fault residuals, not global normal MSE.
- **A-H0 kill condition:** multi-seed effects lack stable fault/variable/lag structure after multiplicity and capacity controls.

### B hypotheses

- **B-H1:** Under matched sensor similarity, objectively inapplicable cases cause larger detection degradation than applicable cases.
- **B-H2:** A post-retrieval applicability verifier reduces contamination-induced AUROC/AUPRC loss and delay inflation relative to similarity-only and context-aware retrieval.
- **B-H3:** The verifier retains incremental value after the retriever already uses regime and control covariates.
- **B-H4:** Each applicability component has a distinct failure signature; no single metadata filter explains the complete gain.
- **B-H0 kill condition:** applicability cannot be labeled without downstream outcomes, or verifier gains vanish against adaptation-/context-aware baselines.

### Single coherent combined hypothesis

> Historical evidence improves early fault detection only when sensor-control-mechanism compatibility is verified; explicit post-retrieval applicability verification therefore improves robustness to sensor-similar but transfer-invalid cases at a controlled false-alarm rate.

Candidate A supplies evidence for choosing compatibility factors and becomes a mechanism/ablation layer inside this question.

## 5. Experimental design

### Phase G0 — novelty and construct gate

- Complete equation/algorithm-level full-text comparison for all high-overlap works.
- Freeze applicability schema, observable inputs, label policy, and non-circularity audit.
- Produce at least 100–200 pilot pairs spanning all applicability conflict types.
- **Pass:** two blinded annotators or deterministic simulator metadata can reproduce labels with acceptable agreement, and high-similarity inapplicable pairs actually exist.

### Phase G1 — Candidate A mechanism audit

- Reuse frozen F0/F1 checkpoints for no-training fault-, run-, variable-, and lag-level analysis.
- Repeat only preregistered necessary training across at least five model seeds if existing checkpoints are insufficient.
- Decompose residual mean, tail, rank separation, persistence crossing, and delay.
- Use permutation/occlusion or grouped feature ablation on XMV histories; validate against correlated-variable artifacts.
- Output a stable map `fault × XMV × lag × residual channel`.

### Phase G2 — retrieval contamination benchmark

- Query units: pre-onset and early post-onset windows from held-out runs.
- Case bank: training runs only, with strict run and provenance isolation.
- Contamination ratios: 0%, 25%, 50%, 75%, 100% among top-k evidence.
- Conflict types: regime, control response, lag, variable support, mechanism, mixed.
- Match similarity-score distributions between applicable and inapplicable cases.
- Evaluate clean context and mixed context separately.

### Phase G3 — applicability verifier

- Start with an interpretable rule/logistic verifier over preregistered compatibility features.
- Add a learned verifier only if it improves held-out calibration and cross-regime generalization.
- Keep retriever and verifier separately scoreable: retrieval recall/precision, applicability precision/recall/F1/calibration, then downstream detection.

### Phase G4 — downstream detection

- Compare no retrieval, contaminated retrieval, oracle applicability, and predicted applicability.
- Lock false-alarm calibration on validation normal data.
- Primary endpoints: per-run AUROC/AUPRC, detected-run ratio, time-to-detection with censoring, and pre-fault FPR.
- Evaluate whether oracle applicability has useful headroom before attributing failure to the learned verifier.

## 6. Required baselines

| Layer | Required baseline |
|---|---|
| Detection | F0; F1; capacity-matched F0-C; simple residual threshold; competitive reconstruction/prediction baseline |
| Retrieval-free | Detector without historical evidence |
| Similarity | Euclidean/cosine on standardized windows; DTW; learned embedding kNN |
| Context-aware retrieval | Regime filter; XMV/control-conditioned distance; joint XMEAS+XMV embedding |
| Adaptation-aware | CBR-style adaptation cost or eligibility-weighted retrieval |
| Time-series RAG | Reproducible analogue of TimeRAG/RAF/RATFM appropriate to available compute |
| Post-retrieval filter | relevance/quality classifier; CRAG-style ternary evaluator |
| Applicability | oracle applicability; rule-based verifier; learned verifier |
| Robustness bounds | clean applicable-only context; random contamination; similarity-matched hard contamination |

## 7. Ablation plan

- Remove regime compatibility.
- Remove control-response compatibility.
- Remove temporal-lag compatibility.
- Remove variable-support compatibility.
- Remove mechanism compatibility.
- XMEAS-only versus XMEAS+XMV retrieval features.
- Retriever-only versus verifier-only gating versus joint score.
- Hard reject versus soft weighting versus contested-case abstention.
- Top-k and contamination ratio sensitivity.
- Rule-based versus learned verifier.
- Oracle applicability gap decomposition: retrieval miss, verifier error, downstream integration error.
- Fault-wise leave-one-mechanism-out generalization.

## 8. Robustness and reproducibility

- Minimum five training seeds for any newly trained neural component; preserve identical data splits.
- Run-level and fault-level reporting; never rely only on pooled samples.
- Independent second dataset or a paired pure-Python TEP simulator study if labels/metadata permit; otherwise explicitly limit external validity.
- Leakage tests for run identity, overlap, scaling, threshold tuning, future XMV, onset alignment, and retrieval duplicates.
- Stress tests for missing XMV, noisy control signals, regime metadata errors, case-bank imbalance, and unseen fault mechanisms.
- Freeze configs, code commit, environment, split manifest, case-bank manifest, label protocol, random seeds, and raw-to-result lineage.

## 9. Statistical validation

- Predeclare one primary endpoint and a small family of secondary endpoints.
- Use run-level stratified bootstrap confidence intervals; resample runs, not overlapping windows.
- Compare paired detectors using run-level paired bootstrap/permutation tests; McNemar for paired detected/not-detected outcomes where applicable.
- Analyze delay as time-to-event with undetected runs censored; supplement medians and restricted mean time-to-detection rather than averaging detected runs only.
- Use hierarchical or mixed-effects models for repeated run/fault/seed structure when assumptions are defensible.
- Report effect sizes and confidence intervals, not p-values alone.
- Control multiple fault/variable/lag comparisons with Benjamini-Hochberg FDR; label exploratory analyses explicitly.
- Calibrate and compare at matched validation-derived pre-fault FPR. Report calibration error/Brier score for verifier probabilities.
- Conduct sensitivity analyses across top-k, contamination ratios, thresholds, and label-policy variants.

## 10. Expected failure modes

| Failure | Diagnostic | Consequence |
|---|---|---|
| Applicability is circular | Labels change when downstream model changes | Kill B formulation |
| Verifier is only a regime filter | Regime-only baseline matches it | Downgrade novelty; reformulate or kill |
| Hard negatives are unrealistic | Easy separation or metadata shortcut | Rebuild benchmark; no robustness claim |
| Retrieval has no oracle headroom | Oracle applicability fails to help | Kill retrieval contribution |
| XMV effect is controller reaction only | Gains start after sensor evidence or lack stable lead | Retain association language; reject causal mechanism |
| Pooled delay artifact | Effect disappears with censored run-level analysis | Retract early-detection claim |
| Fault-specific overfitting | Leave-one-mechanism-out collapse | Limit scope or kill general claim |
| Dataset identity leakage | Randomized identifiers preserve performance | Invalidate experiment and repair split |
| Similarity and applicability collapse | High correlation and no discordant pairs | B has no identifiable research variable |

## 11. Implementation workload: 4–6 intensive weeks

| Week | Primary deliverable | Exit criterion |
|---:|---|---|
| 1 | Full-text prior-work audit; construct and label protocol; split/case-bank freeze | Novelty and non-circularity gate pass |
| 2 | Candidate A no-training mechanism audit; baseline retrievals; hard-negative generator | Reproducible discordant similar/inapplicable pairs |
| 3 | Contamination benchmark; context/adaptation baselines; oracle study | Oracle headroom demonstrated |
| 4 | Rule and learned verifier; downstream integration; main multi-seed runs | Primary endpoint complete at matched FPR |
| 5 | Ablations, robustness, second-data/simulator validation, failure analysis | Major alternative explanations tested |
| 6 | Statistical synthesis, reproducibility rerun, figures/tables, claim audit | Clean-room rerun and claim-evidence table pass |

Workload is substantial but technically plausible if G0 and oracle-headroom gates pass in the first three weeks. Failure at either gate should stop implementation rather than produce a weak thesis.

## 12. Scientific judgment

| Candidate | Judgment | Scientific rationale |
|---|---|---|
| A as simple F0/F1 comparison | **KILL** | Insufficient novelty and mechanism; existing evidence is preliminary only |
| A as mechanism-level control-information study | **MODIFY** | Defensible only if stable variable/lag/fault structure survives statistics and alternative explanations |
| B as covariate-aware retrieval | **KILL** | Methodologically equivalent to strong context-aware retrieval/CBR precedents |
| B as independently evaluated post-retrieval applicability verification | **MODIFY / CONDITIONAL GO** | Strong potential if non-circular labels, discordant pairs, oracle headroom, and incremental value are demonstrated |
| Emergent combined direction C | **MODIFY — HIGHEST POTENTIAL** | One coherent question; A becomes evidence and ablation within B. Still depends on B's construct and novelty gates |

### Recommendation

Do not choose A because it is easier. Do not choose B because it sounds more novel. Prioritize a one-week G0 scientific gate for the emergent combined question, followed immediately by an oracle-applicability study. If either non-circular ground truth or oracle downstream headroom fails, kill B/C and evaluate A only as a mechanism-level thesis. No topic is locked.

## Sources to verify at full-text level before topic approval

- Hsu et al. 2023: https://arxiv.org/abs/2309.02274
- Jiang et al. 2022/2023: https://doi.org/10.1109/TCYB.2022.3163301
- Continuous test-time adaptation: https://arxiv.org/abs/2406.06607
- TimeRAG: https://arxiv.org/abs/2412.16643
- RAF: https://arxiv.org/abs/2411.08249
- RATFM: https://arxiv.org/abs/2506.02081
- Forecast2Anomaly: https://arxiv.org/abs/2511.03149
- CRAG: https://arxiv.org/abs/2401.15884
- ASTUTE RAG: https://aclanthology.org/2025.acl-long.1476/
- Karray et al. 2013: https://doi.org/10.1016/j.engappai.2013.05.001

