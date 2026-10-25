"""Bounded multi-role fixture coordination for the existing orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .plcbench_observations import ExternalObservation


@dataclass(frozen=True)
class SwarmRole:
    name: str
    max_decisions: int
    authority: tuple[str, ...]


def run_bounded_roles(roles: Sequence[SwarmRole], observations: Sequence[ExternalObservation],
                      decide: Mapping[str, Callable[[Sequence[ExternalObservation]], Sequence[Mapping[str, Any]]]],
                      *, max_total_decisions: int = 20) -> dict[str, Any]:
    if max_total_decisions < 0 or len({r.name for r in roles}) != len(roles):
        raise ValueError("invalid swarm budget or duplicate role")
    decisions = []
    for role in roles:
        if role.max_decisions < 0:
            raise ValueError("negative role budget")
        proposed = list(decide[role.name](observations))
        if len(proposed) > role.max_decisions:
            raise ValueError(f"{role.name} exceeded decision budget")
        for item in proposed:
            action = item.get("action")
            if action not in role.authority:
                raise ValueError(f"{role.name} lacks authority for {action}")
            decisions.append({"role": role.name, "action": action,
                              "reason": item.get("reason", "")})
            if len(decisions) > max_total_decisions:
                raise ValueError("swarm total decision budget exceeded")
    return {"observations": [obs.as_dict() for obs in observations],
            "decisions": decisions, "evidence_origin": "bounded_local_fixture"}
