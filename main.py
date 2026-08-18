from pathlib import Path

import yaml

from agents import ResearchOrchestrator
from providers import get_provider


def main() -> None:
    root = Path(__file__).resolve().parent
    mission_path = root / "missions" / "mission_001_topic_validation.yaml"
    mission = yaml.safe_load(mission_path.read_text(encoding="utf-8"))
    output = ResearchOrchestrator(root, get_provider()).run(mission)
    print(f"Mission complete: {output}")
    print("Status: WAITING_FOR_USER_APPROVAL")


if __name__ == "__main__":
    main()
