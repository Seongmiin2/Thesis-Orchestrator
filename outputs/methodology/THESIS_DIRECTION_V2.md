# Thesis Direction V2 — Control-History Utility Across Industrial Time-Series Models

Status: `PROPOSED — APPROVAL AND NOVELTY GATE REQUIRED`

## Working title

**When Does Control History Help? Architecture-Robust Utility Attribution for Industrial Time-Series Anomaly Detection**

Korean working title:

**산업 시계열 이상 탐지에서 제어 이력의 조건부 유용성: 모델 구조에 강건한 변수·시간 기여도 분석**

## One-sentence thesis

Past actuator/manipulated-variable histories do not improve anomaly detection uniformly; their incremental utility is event-, channel-, lag-, and architecture-dependent, and can be measured reproducibly using distribution-respecting conditional perturbations under matched false-alarm control.

## Minimum defensible novelty

The thesis does not claim a new GRU, Transformer, or universally superior detector. It proposes and evaluates a reusable audit protocol, provisionally named **Control-History Utility Mapping (CHUM)**.

CHUM produces a map

`event × control channel × temporal block × architecture → incremental detection utility`

with four requirements:

1. matched sensor-only, sensor+control, and capacity-matched sensor-only models;
2. condition-specific validation calibration at a fixed false-alarm target;
3. distribution-respecting conditional replacement of control histories, alongside zero occlusion as a diagnostic only;
4. stability reporting across seeds, runs, and architecture families.

The novelty is the controlled estimation and cross-architecture validation of *when* control history is useful, not merely whether adding all controls improves a pooled score.

## Research questions

- **RQ1 — Existence:** Is incremental control-history utility heterogeneous across faults/attacks rather than a uniform consequence of added parameters?
- **RQ2 — Structure:** Is the utility concentrated in reproducible actuator channels and temporal blocks?
- **RQ3 — Architecture robustness:** Which utility patterns persist across GRU, TCN, and a compact Transformer under matched capacity tiers?
- **RQ4 — Dataset transfer:** Do the same qualitative patterns appear in simulated process faults and physical/hardware-in-the-loop control-system attacks?
- **RQ5 — Reliability:** Does conditional replacement yield more stable and lower-FPR attribution than zero occlusion or unconditional permutation?

## Hypotheses

- **H1:** Sensor+control models have significant gains only for a subset of events after capacity and FPR matching.
- **H2:** For gain events, maximum single-control conditional-replacement loss is larger than for no-gain events.
- **H3:** At least one channel/time-block utility pattern retains direction in two of three architectures.
- **H4:** Zero occlusion produces more pre-fault score shift than conditional replacement.
- **H5:** Event-level utility rankings are more reproducible across seeds than raw gradient/attention attribution rankings.

## Dataset portfolio

### Primary: Reinartz extended Tennessee Eastman Process

- Already local and processed: 2,800 runs, 28 faults, 2,000 samples/run, 5.6 million rows.
- 41 XMEAS sensors and 11 available XMV channels.
- Strengths: repeated runs, known onset, fault-wise evaluation, clean train/validation/test separation.
- Limits: simulated, XMV12 and operating mode absent, physical mechanism labels incomplete.
- Role: full mechanism discovery and statistical validation.

### External validation A: SWaT

- Physical water-treatment testbed; approximately 0.9 million one-second records.
- 26 sensors and 25/26 actuator variables depending on release/schema; documented attacks.
- Strengths: real physical testbed and explicit sensor/actuator roles.
- Limits: attacks are not equivalent to process faults; access request and terms are required; one long trajectory requires event-block rather than i.i.d. sample inference.
- Role: test whether control-history utility and corruption sensitivity recur in a physical ICS.

### External validation B: HAI 21.03 or 22.04

- Hardware-in-the-loop turbine, boiler, and water-treatment control system.
- Public versions contain long normal and attack scenarios with coupled process/control points.
- Strengths: open repository, multiple operating situations, competition baselines, time-aware metric guidance.
- Limits: point-role mapping must be audited; attacks and faults must not be pooled semantically.
- Role: replication on a second physical/HIL control environment.

### Optional mechanism validation: COSTEP/full TEP simulator

- Rich access to internal, manipulated, and actuator signals would improve physical interpretation.
- Role: one small paired intervention experiment only if the executable environment and license are available.
- This is optional and must not block the thesis.

## Shared data schema

Every dataset is converted to:

`dataset, episode_id, timestamp, sensor_vector, control_vector, event_id, event_type, onset, split, provenance`

Dataset-specific semantics remain separate. A TEP process fault and a SWaT cyberattack are never treated as the same label. Cross-dataset claims concern the *utility phenomenon and audit reliability*, not identical mechanisms.

## Models

Three deliberately modest architecture families:

1. **GRU forecaster** — current reproducible analysis anchor.
2. **TCN forecaster** — non-recurrent local/multi-scale temporal baseline.
3. **Compact Transformer forecaster** — attention-based robustness check.

For each family:

- sensor-only;
- sensor+control;
- capacity-matched sensor-only;
- approximately 25K and 100K parameter tiers where feasible.

The thesis does not reward a larger model for winning. Its question is whether the estimated control-utility structure survives architecture and capacity changes.

## CHUM method

### Step 1 — Establish event-level incremental utility

For event `e`, architecture `m`, and seed `s`:

`Delta(e,m,s) = Metric(sensor+control) - Metric(capacity-matched sensor-only)`

Use AUROC/AUPRC plus event-wise detection ratio and censored delay at a separately calibrated validation threshold.

### Step 2 — Distribution-respecting conditional replacement

Train a small control-history imputer on normal training data only:

`p(U_history | X_history, previous U)`

For channel or temporal block `b`, replace its observed history with multiple samples/conditional means from the imputer. The CHUM utility is the paired deterioration relative to the unmodified sensor+control model.

Zero occlusion, reversal, and unconditional permutation remain baselines. They are not the primary attribution because they can create impossible control trajectories.

### Step 3 — Stability filter

A utility cell is called stable only if:

- mean AUROC loss is at least 0.02;
- direction agrees in at least 4/5 seeds;
- hierarchical paired run/event bootstrap interval excludes zero;
- Benjamini–Hochberg adjusted q-value is below 0.05;
- absolute pre-event FPR shift is within the prespecified tolerance or is explicitly classified as an attribution failure.

### Step 4 — Architecture consensus

Report both architecture-specific maps and a consensus map. A consensus cell requires stable support from at least two architecture families. Attention weights alone are never accepted as evidence of utility.

## Baselines

- sensor-only and capacity-matched sensor-only;
- all-sensor+control model;
- zero occlusion;
- unconditional within-run time permutation;
- gradient or Integrated Gradients attribution;
- attention weights for the Transformer, labeled as descriptive;
- classical dynamic/CVA contribution analysis where implementable;
- model-family baselines: GRU, TCN, compact Transformer.

## Evaluation and statistics

- fixed run/episode-level splits; no overlapping windows across splits;
- five seeds on TEP, at least three seeds on external datasets;
- validation-only threshold selection;
- hierarchical paired bootstrap: seed → episode/run;
- Benjamini–Hochberg FDR across event/channel/block cells;
- event-level metrics; sample-level pooled metrics are secondary;
- censored delay assigns the full evaluation horizon plus one to missed events;
- dataset-specific official/time-aware metrics reported for SWaT/HAI where required.

## Contributions that may be claimed if gates pass

1. A reproducible, FPR-controlled protocol for control-history utility attribution in industrial anomaly detectors.
2. Evidence that incremental control information is strongly event-, channel-, and lag-specific rather than a generic benefit of model size.
3. An architecture-consensus analysis separating stable information structure from model-specific attribution.
4. Cross-domain validation spanning simulated faults and physical/HIL control-system attacks.

## Claims that are prohibited

- The selected actuator caused the fault.
- Controller actions causally improve detection.
- CHUM identifies root cause without intervention or verified mechanism metadata.
- A result on TEP automatically transfers to real plants.
- A larger Transformer is intrinsically a fairer or better scientific test.

## Decision gates

### G1 — Complete current TEP statistics

PASS if the existing six-fault structure survives paired hierarchical uncertainty and FDR, and conditional replacement reduces the identified zero-occlusion FPR exceptions.

### G2 — Architecture robustness

PASS if at least four TEP GAIN faults remain material in two of GRU/TCN/Transformer and at least three channel mappings have consensus support. Otherwise narrow the claim to architecture-dependent utility.

### G3 — External replication

PASS if SWaT or HAI shows statistically heterogeneous event-level control utility and conditional replacement improves attribution reliability. Exact channel identities need not transfer.

### Final GO

Proceed to the final thesis only if G1 and G2 pass. G3 upgrades generality but is not allowed to invalidate an otherwise rigorous TEP-scoped thesis when access or event semantics prevent a fair comparison.

## Minimum viable thesis and stronger version

**Minimum viable:** TEP + GRU/TCN/Transformer + CHUM conditional replacement + full statistics.

**Stronger:** add one of SWaT or HAI as external replication.

**Over-scoped and rejected:** inventing a large new Transformer, training on every anomaly benchmark, or claiming causal controller mechanisms without intervention data.

## Proposed chapter structure

1. Introduction: control history is available but its utility is not uniform.
2. Related work: residual monitoring, controller-aware monitoring, deep industrial anomaly detection, attribution reliability.
3. Problem formulation and CHUM.
4. Datasets and role/provenance audit.
5. Controlled architecture and capacity experiments.
6. Variable/time utility maps and statistical validation.
7. External replication and failure cases.
8. Limitations, claim boundaries, and conclusion.

## Immediate execution order

1. Finish paired bootstrap/FDR and conditional-replacement prototype on the frozen TEP checkpoints.
2. Implement capacity-matched TCN and compact Transformer using the existing split/evaluator.
3. Run one-seed smoke tests, freeze hyperparameters on validation, then execute full seeds.
4. Audit HAI schemas first because it is openly obtainable; request SWaT access in parallel.
5. Lock the final title only after G2.
