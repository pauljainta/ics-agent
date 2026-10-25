"""Experimental mirror of the WT ST membrane equation for offline qualification.

This is a code-consistency check, not a physical validation or independent
implementation of the whole simulator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


def _constant(source: str, name: str) -> float:
    match = re.search(rf"\b{name}\s*:\s*REAL\s*:=\s*([0-9.]+)\s*;", source)
    if not match:
        raise ValueError(f"missing ST constant {name}")
    return float(match.group(1))


@dataclass
class ExperimentalMembrane:
    chlorine_threshold: float
    damage_rate: float
    nominal_rejection: float
    damage: float = 0.0

    @classmethod
    def from_st(cls, path: Path) -> "ExperimentalMembrane":
        source = path.read_text()
        for fragment in ("membrane_damage := membrane_damage", "WQ_DAMAGE_RATE * (wq_roft_cl - WQ_RO_CL_MAX)",
                         "eff_rejection := WQ_RO_REJECTION * (1.0 - membrane_damage)"):
            if fragment not in source:
                raise ValueError("ST membrane equation changed; mirror must be reviewed")
        return cls(_constant(source, "WQ_RO_CL_MAX"), _constant(source, "WQ_DAMAGE_RATE"),
                   _constant(source, "WQ_RO_REJECTION"))

    def step(self, feed_chlorine: float, feed_conductivity: float) -> float:
        self.damage = min(1.0, self.damage + self.damage_rate *
                          max(0.0, feed_chlorine - self.chlorine_threshold))
        rejection = max(0.0, self.nominal_rejection * (1.0 - self.damage))
        return min(20.0, max(0.02, feed_conductivity * (1.0 - rejection)))

    def reset(self, *, full_simulator_restart: bool) -> None:
        if full_simulator_restart:
            self.damage = 0.0
