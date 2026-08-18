from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .literature import LiteratureAgent
from .methodology import MethodologyAgent
from .reviewer import ReviewerAgent


class ResearchOrchestrator:
    def __init__(self, root: Path, provider):
        self.root = root
        self.agents = [LiteratureAgent(provider), MethodologyAgent(provider), ReviewerAgent(provider)]

    def run(self, mission: dict) -> Path:
        state_path = self.root / "state" / "RESEARCH_STATE.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        results = []
        for candidate in mission["candidates"]:
            outputs = {agent.name: agent.run(mission, candidate) for agent in self.agents}
            results.append((candidate, outputs))

        report = self._report(mission, results)
        output = self.root / "outputs" / "reviews" / "mission_001_topic_validation.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")

        # Human gate: candidates and recommendation may be recorded; no topic is locked.
        state["candidate_topics"] = [c["title"] for c, _ in results]
        state["pending_human_approval"] = True
        state["status"] = "WAITING_FOR_USER_APPROVAL"
        state["locked_topic"] = None
        state["locked_research_question"] = None
        state["last_mission"] = mission["id"]
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._log(mission, output)
        return output

    def _log(self, mission: dict, output: Path) -> None:
        log = self.root / "state" / "DECISION_LOG.md"
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = (
            f"| {stamp} | {mission['id']} | PROPOSED | Mock agents completed; no topic locked | "
            f"Human approval required | Review `{output.relative_to(self.root)}` |\n"
        )
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line)

    @staticmethod
    def _report(mission: dict, results: list[tuple[dict, dict]]) -> str:
        lines = ["# THESIS TOPIC VALIDATION REPORT", "", "> Status: WAITING_FOR_USER_APPROVAL", ""]
        for candidate, outputs in results:
            lines += [f"## Candidate {candidate['id']}: {candidate['title']}", ""]
            for heading, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses"),
                                 ("Confirmed Evidence", "confirmed_evidence"), ("Open Questions", "open_questions"),
                                 ("Novelty Risks", "novelty_risks"), ("Required Experiments", "required_experiments")):
                lines.append(f"### {heading}")
                lines.extend(f"- {item}" for item in candidate[key])
                lines.append("")
            lines += [f"**Verdict: {outputs['reviewer']['verdict']}**", ""]
        lines += ["## Comparison", ""] + [f"- {x}" for x in mission["comparison"]]
        lines += ["", "## Recommendation", "", mission["recommendation"], "",
                  "This recommendation is PROPOSED only. No thesis topic, research question, main hypothesis, contribution, or main claim has been locked.", "",
                  "## Provenance warning", "", "Agent deliberation in this report is MOCK. Confirmed evidence entries are copied from the read-only project handover; mock output is not stored as evidence.", ""]
        return "\n".join(lines)

