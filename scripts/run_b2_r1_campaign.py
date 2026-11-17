#!/usr/bin/env python3
"""Access-VM entry point for the B2-R1 C0-C4 campaign runner."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.scadaver_sphere.b2_r1_campaign import main


if __name__ == "__main__":
    raise SystemExit(main())
