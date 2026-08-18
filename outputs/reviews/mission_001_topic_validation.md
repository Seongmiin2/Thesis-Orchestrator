# THESIS TOPIC VALIDATION REPORT

> Status: WAITING_FOR_USER_APPROVAL

## Candidate A: Control-Aware Predictive Anomaly Detection

### Strengths
- Reuses completed leakage-controlled F0/F1 infrastructure and verified artifacts.
- Has repeat directional evidence across three model seeds and one capacity control.
- Fits a master's scope if claims stay incremental and non-causal.

### Weaknesses
- Normal prediction gain is inconsistent and detection gains are fault-specific.
- Current mathematical distinction gate is inconclusive and F0/F1 experiments are closed.

### Confirmed Evidence
- F1 improved pooled AUROC, AUPRC and delay at Seed 42 with similar pre-fault FPR.
- Directional detection improvement repeated for seeds 42–44.
- A one-run capacity-matched F0-C did not reproduce F1 performance.

### Open Questions
- Which XMV/fault mechanisms explain benefit without implying causal control effects?
- Does a preregistered run-level mechanism test falsify the observed association?

### Novelty Risks
- Input-conditioned prediction residuals have strong prior-art overlap.
- Adding XMV to a GRU is not methodological novelty by itself.

### Required Experiments
- Complete the preregistered no-training all-fault mechanism diagnostic.
- Complete equation-level comparison with the highest-overlap lawful full texts.

**Verdict: MODIFY**

## Candidate B: FAVE-TS / Applicability-Aware Retrieval for Industrial Time Series

### Strengths
- Transfers FAVE's explicit applicability distinction to a high-value Physical AI setting.
- Offers richer methodological space for schemas, arbitration, conflict types and ablations.

### Weaknesses
- No industrial applicability benchmark or experiment exists yet.
- Requires a defensible retrieval task and gold applicability labels.

### Confirmed Evidence
- FAVE-RAG implements valid/invalid/contested evidence and conflict detection in telecom math.
- PhysicalAI artifacts show strongly fault-specific benefit from XMV context.

### Open Questions
- Can applicability be operationalized independently of downstream prediction correctness?
- Is this distinct from conditional similarity, case-based reasoning and regime-aware retrieval?

### Novelty Risks
- FAVE-TS may be only a domain rename without a new formal criterion or benchmark.
- Regime-aware and context-aware time-series retrieval may already cover the core idea.

### Required Experiments
- Literature novelty review using verified sources.
- Define valid, invalid and contested historical references with annotator protocol.
- Compare similarity-only retrieval with applicability-aware arbitration on downstream diagnosis.
- Ablate control regime, operating context, sensor-action relationship and fault mechanism.

**Verdict: MODIFY**

## Comparison

- Safer topic: Candidate A, because it reuses verified experiments, but only after narrowing the claim and clearing the novelty gate.
- Stronger methodological potential: Candidate B, because it admits explicit validity schemas and arbitration, but it begins with much less evidence.
- Best use of current work: A uses PhysicalAI most directly; B combines concepts from both projects.
- Higher novelty risk: B, because domain transfer may collapse into existing regime-aware retrieval or metadata filtering.
- Master's realism: A is currently more realistic; B should remain a staged feasibility hypothesis until literature and benchmark gates pass.

## Recommendation

Do not lock either topic. Provisionally prioritize a narrowed Candidate A as the lower-risk thesis path, while running a small, time-boxed literature-and-benchmark feasibility gate for Candidate B. Promote B only if applicability can be defined non-circularly, prior-work overlap is manageable, and a controlled benchmark can be built.

This recommendation is PROPOSED only. No thesis topic, research question, main hypothesis, contribution, or main claim has been locked.

## Provenance warning

Agent deliberation in this report is MOCK. Confirmed evidence entries are copied from the read-only project handover; mock output is not stored as evidence.
