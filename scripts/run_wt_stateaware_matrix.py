#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from ics_agent.experiments.stateaware_oi_matrix import main_for


CONFIG_NAME = "water_treatment_stateaware_oi_matrix_vs_agentic_full_containment_defender.yaml"


if __name__ == "__main__":
    raise SystemExit(
        main_for(
            default_config=CONFIG_NAME,
            testbed_id="water_treatment",
            description=(
                "Run the six-scenario Water Treatment state-aware O/I matrix "
                "against the agentic full-containment defender."
            ),
        )
    )
