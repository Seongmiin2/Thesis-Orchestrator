from __future__ import annotations

from .base import GenerationRequest, Provider


class MockProvider(Provider):
    """Deterministic plumbing provider. Its output is never research evidence."""

    def generate(self, request: GenerationRequest) -> dict:
        candidate = request.context["candidate"]
        role = request.agent
        common = {"agent": role, "candidate_id": candidate["id"], "provenance": "MOCK"}
        if role == "literature":
            return common | {
                "questions": candidate["literature_questions"],
                "novelty_risks": candidate["novelty_risks"],
                "evidence_records": [],
                "warning": "MOCK planning output; no papers or citations were invented.",
            }
        if role == "methodology":
            return common | {"design": candidate["methodology"]}
        if role == "reviewer":
            return common | {
                "criticisms": candidate["reviewer_risks"],
                "verdict": candidate["verdict"],
                "independent": True,
            }
        raise ValueError(f"Unsupported mock agent: {role}")

