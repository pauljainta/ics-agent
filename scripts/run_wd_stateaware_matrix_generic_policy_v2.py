#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from ics_agent.experiments.stateaware_oi_matrix import (  # noqa: E402
    WD_EXPECTED_CONDITIONS,
    main_for,
)


CONFIG_NAME = (
    "water_distribution_stateaware_oi_matrix_generic_policy_v2_vs_"
    "agentic_full_containment_defender.yaml"
)


if __name__ == "__main__":
    raise SystemExit(
        main_for(
            default_config=CONFIG_NAME,
            testbed_id="water_distribution",
            expected_conditions=WD_EXPECTED_CONDITIONS,
            description=(
                "Run the 12-run Water Distribution state-aware O/I matrix with "
                "the shared generic-policy-v2 attacker scaffold against the "
                "agentic full-containment defender."
            ),
        )
    )
