#!/usr/bin/env python3
"""Launch the fixed N3 Qwen3.8-27B profile against OT-Realistic V2."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.scadaver_sphere.n3_v2 import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
