# G0 Scientific Gate Report

> **Status: WAITING_FOR_USER_APPROVAL**  
> Topic lock: `null`  
> Research-question lock: `null`  
> Gate executed: 2026-08-18

## Technical decision

**Candidate B/C is KILLED for the current Reinartz-based formulation.** This is a scientific, not scheduling, decision. Two independent gate failures occurred:

1. The available Reinartz distribution cannot support the proposed full applicability variable objectively: operating mode, XMV12, verified fault-mechanism mapping, and known future control are unavailable.
2. A non-circular, observable past-XMV consistency proxy produced no downstream oracle headroom over sensor-only retrieval: AUROC `-0.0051`, AUPRC `-0.00003`, detected-run ratio `+0.0000`, and detected-run median delay reduction `0`.

The proxy is narrower than the intended applicability construct and must not be presented as a universal disproof of post-retrieval verification. It is sufficient to reject Candidate B/C **on the current dataset and operationalization**, because the required gold variable cannot be completed and the strongest executable proxy provides no headroom.

Candidate C is therefore **not re-proposed**. Candidate A remains `MODIFY` as a possible mechanism-level research direction; it is not locked.

## Evidence classification

### Confirmed evidence

- RAG4CTS explicitly states that visually similar target histories can be invalid references when driving covariates differ.
- RAG4CTS implements regime-aware raw historical storage, history-decay and known-future-control weighting, mutual-information covariate weighting, weighted shape retrieval, weighted state-precision retrieval, and dynamic context selection.
- Reinartz lacks operating mode, XMV12, verified mechanism labels, and known future controls.
- The G0 proxy experiment used 5,376 training-bank cases, 1,344 validation queries, and 1,680 test queries from 120 held-out runs.
- Among sensor-nearest top-5 cases, 35.74% of validation-normal cases and 78.63% of selected test cases violated the fixed past-XMV consistency proxy.
- Oracle proxy filtering did not improve AUROC, AUPRC, detected-run ratio, or median detected-run delay over sensor-only retrieval.

### Interpretation

- Discordant sensor/control pairs exist, but their existence is not novel after RAG4CTS.
- The very high test discordance partly reflects fault-induced control-response distribution shift; it is not evidence of mechanism-level transfer invalidity.
- Joint sensor+control retrieval improved ranking metrics more than post-filtering, which is consistent with RAG4CTS's premise that control alignment belongs inside a strong retriever.
- Candidate B's proposed verifier currently collapses into a control-consistency filter rather than establishing an additional applicability variable.

### Unverified hypotheses

- A separately evaluated verifier might still add value on a dataset with independently known regimes, mechanisms, future controls, and case-transfer outcomes.
- Candidate A's variable/lag findings might support a different mechanism-level thesis after preregistration and statistical validation.
- Neither hypothesis is a current thesis claim.

## 1. RAG4CTS algorithm/equation-level comparison

RAG4CTS defines a hierarchical raw-regime knowledge base

```text
B = {(M_i, X_i)} for i=1..N                                      Eq. (1)
```

where `M_i` is a physical metadata path and `X_i` is a complete raw operational regime. This already covers physics-informed regime organization.

Its point-weight matrix uses observable history, known future covariates, and masked unknown targets:

```text
W_point[t,v] = lambda^(L_hist-t)   for historical points
               1                   for future driving covariates
               0                   for unknown future targets        Eq. (2)
```

Covariates are weighted by mutual information with the target:

```text
I(X_v;Y) = sum p(x,y) log[p(x,y)/(p(x)p(y))]                         Eq. (3)
w_cov[v] = I(X_v;Y) / max_k I(X_k;Y)                                Eq. (4)
W = W_point elementwise-multiplied by w_cov                          Eq. (5)
```

Retrieval then applies weighted cosine shape alignment and weighted Matrix Profile/state distance:

```text
S_shape(Q,C_i;W)                                                     Eq. (6)
D_mp(Q,C_i;W) = sqrt(sum_t sum_v W[t,v](Q[t,v]-C_i[t,v])^2)          Eq. (7)
C_final = Top-K ordered by D_mp                                     Eq. (8)
```

Finally, it chooses context length using the known future of the top-1 retrieved case and applies that configuration to the user query:

```text
k* = argmin_k Loss(F(Concat(retrieved contexts, top-1 case)),Y_top1) Eq. (9)
Yhat = F(Concat(selected historical contexts, user query))           Eq. (10)
```

### Consequence for novelty

The following are **not available novelty claims**:

- control-aware retrieval;
- covariate-aware retrieval;
- regime-aware retrieval;
- physics-informed historical regime retrieval;
- future-control alignment;
- the observation that visual similarity can return invalid references.

The only remaining candidate distinction was a separately evaluated **post-retrieval** applicability variable with independent labels and downstream incremental value. G0 did not establish it.

## 2. High-overlap comparison matrix

| Work | Verified mechanism | Overlap with proposed B/C | Residual possible distinction | Status |
|---|---|---|---|---|
| RAG4CTS, Liang et al. 2026 | Eq. 1–10: physical hierarchy, future-control and MI covariate weighting, weighted two-stage retrieval, agentic context selection | Directly covers control/regime/physics-aware retrieval and invalid visual matches | Only an independently labeled post-retrieval verifier | Full text inspected |
| TS-RAG, 2025 | Embedding retriever returns context/future pairs; Adaptive Retrieval Mixer learns integration weights | Covers learned retrieval and adaptive evidence integration | Separate applicability supervision/abstention | Full text inspected |
| RAF, 2024/2025 | Retrieves related historical examples for frozen/fine-tuned TSFMs | Covers retrieval-augmented forecasting and mechanistic retrieval analysis | Industrial transfer-validity labels | Algorithm-level record inspected; equation audit incomplete |
| RATFM, 2025 | Retrieved normal-domain examples condition forecast-based anomaly detection; moving-average score smoothing | Direct anomaly-detection overlap; documents failures when similar inputs have different futures | Explicit industrial applicability label and contamination test | Full text inspected |
| Adaptation-guided industrial CBR, Karray et al. 2013 | Similarity and adaptation are separate retrieval measures | Conceptually anticipates `similarity != applicability` | Temporal sensor-control-mechanism operationalization | Publisher abstract inspected; lawful full equation audit incomplete |
| Hsu et al. 2023 | Input-output predictors condition expected sensor values on operating conditions | Strong overlap with Candidate A residual logic | Fault-variable-lag mechanism inference | Full text/primary record previously audited |
| Chen 2016 | Input-conditioned static and dynamic residuals | Direct overlap with Candidate A `e_t` | Two-predictor difference does not itself establish identification | Full text previously audited in PhysicalAI |

**Gate result:** the required full-text matrix is substantially complete for the highest recent retrieval threats, especially RAG4CTS, TS-RAG, and RATFM. It remains incomplete for adaptation-guided CBR equations and some older residual papers. Missing coverage increases, rather than reduces, novelty risk.

## 3. Non-circular applicability definition audit

The intended definition was:

```text
A*(q,c) = P(regime compatibility,
            control-response compatibility,
            temporal-lag compatibility,
            variable-support compatibility,
            mechanism-transfer compatibility,
            provenance acceptability)
```

To be non-circular, `A*` cannot depend on whether the downstream detector happened to predict correctly. It must be determined from information available independently of the evaluated model.

### Dataset audit

| Required variable | Reinartz availability | Consequence |
|---|---|---|
| Past XMEAS | Available | Sensor similarity observable |
| Past XMV | 11 variables available | Past control consistency observable, incomplete |
| XMV12 | Missing | Control representation incomplete |
| Operating mode/regime | Missing | Regime compatibility not labelable |
| Verified fault mechanism | Missing | Mechanism-transfer compatibility not labelable |
| Known future control | Not available in the online setup | Cannot reproduce RAG4CTS Eq. 2 alignment |
| Independent case-transfer outcome | Missing | Full applicability cannot be adjudicated |

**Gate result: FAIL.** Only a past-XMV consistency proxy is objectively executable. Calling it full applicability would be construct overclaiming.

## 4. Observable proxy and pair construction

The executed proxy used only past information:

```text
Sensor representation  = mean + last value + window slope of 41 XMEAS
Control representation = mean + last value + window slope of 11 XMV
```

The case bank used normal windows from training runs only. Query runs were held out. No query could retrieve itself. A sensor-nearest pair was proxy-applicable when its squared standardized control-summary distance was below a fixed threshold derived from 20,000 training-bank pairs. Neither downstream error, test labels, future observations, nor fault identity entered this rule.

This construction is computationally reproducible but semantically limited: it measures control consistency, not mechanism-level transfer validity.

## 5. Discordant-pair existence and frequency

| Population | Sensor top-5 proxy-inapplicable | Shortlist proxy-applicable |
|---|---:|---:|
| Validation normal queries | 35.74% | 57.80% |
| Selected test queries | 78.63% | 18.99% |

**Confirmed:** high sensor similarity does not guarantee past-XMV consistency in this dataset.

**Not confirmed:** these pairs are invalid evidence in the stronger mechanism-transfer sense. RAG4CTS already predicts this kind of control mismatch and addresses it within retrieval.

## 6. Labeling reproducibility

- Computational reproduction: deterministic code, fixed seed 42, fixed threshold, fixed split and case bank.
- Human/inter-annotator reproduction: **not evaluable**, because the dataset lacks regime and mechanism metadata required by the intended label protocol.
- Construct validity: **not established**. Exact code reproducibility does not convert a proxy into scientific ground truth.

**Gate result: FAIL.**

## 7. Oracle proxy downstream headroom

All thresholds were calibrated at the validation-normal 99th percentile. Test metrics cover six previously emphasized faults (4, 7, 19, 24, 25, 26), 120 runs, and early windows through sample 700.

| Method | AUROC | AUPRC | Test pre-fault FPR | Detected-run ratio | Median delay among detected |
|---|---:|---:|---:|---:|---:|
| Sensor-only retrieval | 0.5823 | 0.8397 | 0.0111 | 0.0083 | 70 samples |
| Joint sensor+control retrieval | 0.6198 | 0.8581 | 0.0222 | 0.0333 | 35 samples |
| Oracle proxy post-filter | 0.5772 | 0.8397 | 0.0139 | 0.0083 | 70 samples |

Oracle-proxy change versus sensor-only:

```text
AUROC                    -0.00512
AUPRC                    -0.00003
Detected-run ratio       +0.00000
Median delay reduction    0 samples
```

The post-filter provides no headroom. Joint control-aware retrieval performs better on ranking metrics, though its held-out test FPR is higher despite validation calibration and its absolute detected-run ratio remains very low. This is preliminary diagnostic evidence, not a production baseline claim.

**Gate result: FAIL.** Under the user's preregistered decision rule, Candidate B/C must be killed when oracle applicability produces no meaningful detection or delay headroom.

## 8. Scientific judgment

| Candidate | Judgment | Evidence-based reason |
|---|---|---|
| A: simple feature comparison | KILL | No sufficient contribution |
| A: mechanism-level control-information analysis | MODIFY / NOT LOCKED | Remains possible but requires preregistered variable/lag/fault statistics and failure analysis |
| B: control/regime/covariate-aware retrieval | KILL | Directly overlapped by RAG4CTS |
| B: post-retrieval applicability verifier on Reinartz | KILL | Full objective labels unavailable; executable oracle proxy has no headroom |
| C: Applicability-Verified Historical Evidence | KILL FOR CURRENT FORMULATION | Depends on B's failed construct and oracle gates |

## 9. Reproducibility artifacts

- Experiment: `experiments/g0_oracle_applicability.py`
- JSON result: `outputs/methodology/g0/g0_oracle_proxy_results.json`
- Metrics CSV: `outputs/methodology/g0/g0_oracle_proxy_metrics.csv`
- PhysicalAI source data: read-only Reinartz memmap, split manifest, and documented scaler provenance
- Random seed: 42
- Reference projects modified: no

## 10. Next decision requiring approval

No topic is locked. The next scientifically defensible choice is whether to:

1. authorize a revised Candidate A mechanism-level gate; or
2. source a different dataset with objective regimes, mechanisms, and known future controls before reconsidering post-retrieval applicability under a materially different formulation; or
3. stop both directions and reopen topic selection.

The system terminates at `WAITING_FOR_USER_APPROVAL`.

## Primary sources

- RAG4CTS: https://arxiv.org/pdf/2603.04951
- TS-RAG: https://proceedings.neurips.cc/paper_files/paper/2025/file/eed25c037bc08afcbefab6f7a6b700e0-Paper-Conference.pdf
- RAF: https://arxiv.org/abs/2411.08249
- RATFM: https://arxiv.org/abs/2506.02081
- Adaptation-guided industrial CBR: https://doi.org/10.1016/j.engappai.2013.05.001
- Residual comparison: https://arxiv.org/abs/2309.02274

