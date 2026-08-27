#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from ics_agent.experiments.model_variant_runner import main_for  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        main_for(["--model", "qwen3_14b", "--testbed", "water_distribution"])
    )
