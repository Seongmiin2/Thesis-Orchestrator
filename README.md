# Thesis Orchestrator

A small, mock-first research orchestration MVP for evaluating two master's-thesis directions grounded in the read-only `PhysicalAI_mini` and `FAVE-RAG` projects.

It does **not** write a thesis automatically, fabricate citations, execute experiments, call a paid API, or lock a research topic. Its job is to preserve evidence provenance, structure competing arguments, and stop at a human approval gate.

## Architecture

```text
Mission + Research State
          |
          v
 Research Orchestrator
   |       |       |
 Literature  Methodology  Reviewer
   \_______ shared Provider _______/
          |
  report + state + decision log
          |
 WAITING_FOR_USER_APPROVAL
```

- `ResearchOrchestrator`: loads mission/state, delegates bounded tasks, aggregates disagreements, logs the proposed decision, and enforces the approval gate.
- `LiteratureAgent`: generates literature-verification questions and novelty risks. In mock mode it invents no papers or citations.
- `MethodologyAgent`: emits a falsifiable design with variables, datasets, baselines, ablations, metrics, experiments and alternative explanations.
- `ReviewerAgent`: independently attacks novelty, leakage, causal claims, weak baselines, evaluation and reproducibility.
- `Provider`: isolates agent logic from `MockProvider`, future local inference, and a deliberately disabled OpenAI adapter.

## Run

From this directory with Python 3.11+:

```powershell
python -m pip install -e ".[dev]"
$env:LLM_BACKEND = "mock"
python main.py
pytest -q
```

Generated experiment artifacts are intentionally not retained in the current tree. The professor-facing report is `outputs/PROFESSOR_BRIEF_KO.md`, and provenance/reproduction guidance is in `outputs/EVIDENCE_AND_REPRODUCTION_INDEX.md`. The state must end in `WAITING_FOR_USER_APPROVAL`, with both lock fields set to `null`.

## Research experiments

The frozen TEP/CHUM experiments use a separate optional environment because PyTorch is not required by the mock-first orchestrator itself.

```powershell
python -m venv ..\.venv-research
..\.venv-research\Scripts\python.exe -m pip install -e ".[research]"
..\.venv-research\Scripts\python.exe experiments\audit_conditional_imputer.py
..\.venv-research\Scripts\python.exe experiments\run_architecture_chum_g3.py
..\.venv-research\Scripts\python.exe experiments\analyze_architecture_chum_g3.py
..\.venv-research\Scripts\python.exe experiments\run_integrated_gradients_baseline.py
..\.venv-research\Scripts\python.exe experiments\analyze_integrated_gradients_baseline.py
..\.venv-research\Scripts\python.exe experiments\prepare_hai_2103.py
..\.venv-research\Scripts\python.exe experiments\validate_hai_2103_roles.py
..\.venv-research\Scripts\python.exe experiments\validate_hai_2103_attack_targets.py
..\.venv-research\Scripts\python.exe experiments\run_hai_external_validation.py
..\.venv-research\Scripts\python.exe experiments\analyze_hai_external_validation.py
..\.venv-research\Scripts\python.exe experiments\audit_hai_conditional_imputer.py
..\.venv-research\Scripts\python.exe experiments\run_hai_conditional_chum.py
..\.venv-research\Scripts\python.exe experiments\analyze_hai_conditional_chum.py
..\.venv-research\Scripts\python.exe experiments\validate_final_evidence.py
..\.venv-research\Scripts\python.exe experiments\build_professor_report_v2.py
```

`run_architecture_chum_g3.py` writes resumable partial CSVs after every architecture/seed/condition task. Its analysis command refuses to run until every task in `configs/architecture_chum_g3.yaml` is complete.

HAI commands expect the official `icsdataset/hai` repository at `../HAI` and the official `saurf4ng/eTaPR` repository at `../eTaPR`. The first HAI run was invalidated by a feature-column-order bug and must not be cited. Only the corrected v2 and conditional-CHUM evidence summarized in the retained report is usable.

The current tree retains exactly two curated output documents: `outputs/PROFESSOR_BRIEF_KO.md` and `outputs/EVIDENCE_AND_REPRODUCTION_INDEX.md`. Removed raw artifacts remain recoverable from Git commit `bc6166f`.

The HAI conditional runner must reuse only corrected v2 F1 checkpoints, recalibrate each perturbation on validation-normal data, and write resumable partial CSVs after every seed/mode/channel task. The locked protocol and accepted evidence are summarized in the retained report and reproduction index.

## Evidence rules

- `CONFIRMED_FACT`: traceable to inspected code, documentation, CSV, or logs.
- `INTERPRETATION`: a bounded reading of confirmed findings.
- `HYPOTHESIS`: not yet experimentally or bibliographically verified.
- Every mock agent response carries `provenance: MOCK` and is never appended to `EVIDENCE_LEDGER.jsonl`.
- The reference projects are never modified. A test hashes their files before and after a mission.

## Provider policy

`LLM_BACKEND=mock` is the default. `local` and `openai` are interface placeholders only. The OpenAI adapter raises immediately and neither requests an API key nor sends network traffic.

## Human approval gate

Explicit approval is required before changing the thesis topic, research question, main hypothesis, contribution, experiment-retirement decision, research direction, or paper claim. Mission 001 only recommends; it never locks.
