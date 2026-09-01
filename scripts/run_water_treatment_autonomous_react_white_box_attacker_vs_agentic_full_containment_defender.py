#!/usr/bin/env python3
"""Run the treatment autonomous-ReAct white-box attacker baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from ics_agent.orchestrator import ExperimentOrchestrator


CONFIG_PATH = (
    SCRIPT_REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_autonomous_react_white_box_attacker_vs_agentic_full_containment_defender.yaml"
)


def main() -> int:
    summary = ExperimentOrchestrator(testbed_id="water_treatment").run(CONFIG_PATH)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
