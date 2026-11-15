"""Run exactly one E1-B episode using the shared E1-B runner."""

from __future__ import annotations

import sys

from experiments.scadaver_sphere.e1b_batch import main


if __name__ == "__main__":
    main([*sys.argv[1:], "--episodes", "1"])
