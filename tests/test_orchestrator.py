import hashlib
import json
from pathlib import Path

import yaml

from agents import ResearchOrchestrator
from providers.mock import MockProvider


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
REFERENCES = [WORKSPACE / "PhysicalAI_mini", WORKSPACE / "FAVE-RAG"]


def _mission():
    return yaml.safe_load((ROOT / "missions" / "mission_001_topic_validation.yaml").read_text(encoding="utf-8"))


def _manifest(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*") if p.is_file() and ".git" not in p.parts
    }


def test_reference_projects_are_not_modified(tmp_path):
    before = [_manifest(path) for path in REFERENCES]
    ResearchOrchestrator(ROOT, MockProvider()).run(_mission())
    after = [_manifest(path) for path in REFERENCES]
    assert before == after


def test_mock_mission_runs_to_report():
    output = ResearchOrchestrator(ROOT, MockProvider()).run(_mission())
    text = output.read_text(encoding="utf-8")
    assert "THESIS TOPIC VALIDATION REPORT" in text
    assert "Candidate A" in text and "Candidate B" in text
    assert "WAITING_FOR_USER_APPROVAL" in text


def test_agent_outputs_follow_schema():
    mission = _mission()
    provider = MockProvider()
    for candidate in mission["candidates"]:
        for agent in ("literature", "methodology", "reviewer"):
            from providers.base import GenerationRequest
            result = provider.generate(GenerationRequest(agent, mission, {"candidate": candidate}))
            assert result["agent"] == agent
            assert result["candidate_id"] == candidate["id"]
            assert result["provenance"] == "MOCK"


def test_topic_is_not_auto_locked():
    ResearchOrchestrator(ROOT, MockProvider()).run(_mission())
    state = json.loads((ROOT / "state" / "RESEARCH_STATE.json").read_text(encoding="utf-8"))
    assert state["locked_topic"] is None
    assert state["locked_research_question"] is None
    assert state["pending_human_approval"] is True
    assert state["status"] == "WAITING_FOR_USER_APPROVAL"


def test_major_decision_is_logged():
    before = (ROOT / "state" / "DECISION_LOG.md").read_text(encoding="utf-8")
    ResearchOrchestrator(ROOT, MockProvider()).run(_mission())
    after = (ROOT / "state" / "DECISION_LOG.md").read_text(encoding="utf-8")
    assert len(after) > len(before)
    assert "mission_001_topic_validation" in after


def test_mock_is_not_written_to_evidence_ledger():
    ledger = ROOT / "state" / "EVIDENCE_LEDGER.jsonl"
    before = ledger.read_bytes()
    ResearchOrchestrator(ROOT, MockProvider()).run(_mission())
    assert ledger.read_bytes() == before
    for line in before.decode("utf-8").splitlines():
        assert json.loads(line)["mock"] is False

