"""Fixed-disturbance operator-response fixtures, with independent score axes.

These are offline contract/evidence exercises. The caller supplies a response
function; no fixture outcome is a measured plant or human result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class OperatorFixture:
    fixture_id: str
    observations: tuple[Mapping[str, Any], ...]
    allowed_interventions: tuple[str, ...]
    plant_responses: Mapping[str, Mapping[str, Any]]
    safe_envelope: Mapping[str, tuple[float, float]]
    service_target: Mapping[str, tuple[float, float]]


def _within(values: Mapping[str, Any], bounds: Mapping[str, tuple[float, float]]) -> bool:
    return all(key in values and lo <= float(values[key]) <= hi
               for key, (lo, hi) in bounds.items())


def run_operator_fixture(fixture: OperatorFixture, policy_name: str,
                         decide: Callable[[Sequence[Mapping[str, Any]], Sequence[str]], str],
                         *, authority: Sequence[str] | None = None) -> dict[str, Any]:
    permitted = tuple(authority if authority is not None else fixture.allowed_interventions)
    if set(permitted) - set(fixture.allowed_interventions):
        raise ValueError("policy authority exceeds fixture authority")
    observation = [dict(v) for v in fixture.observations]
    decision = decide(observation, permitted)
    if decision not in permitted:
        raise ValueError(f"policy {policy_name} selected unauthorized intervention {decision}")
    response = dict(fixture.plant_responses[decision])
    trace = [
        {"seq": 1, "type": "observation", "payload": observation},
        {"seq": 2, "type": "decision", "policy": policy_name, "payload": decision},
        {"seq": 3, "type": "intervention", "payload": decision, "authorized": True},
        {"seq": 4, "type": "plant_response", "payload": response, "origin": "offline_fixture"},
    ]
    return {"fixture_id": fixture.fixture_id, "policy": policy_name,
            "authority": list(permitted), "trace": trace,
            "trace_sha256": hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
            "safety_score": int(_within(response, fixture.safe_envelope)),
            "service_score": int(_within(response, fixture.service_target)),
            "evidence_origin": "offline_fixture"}


def verify_chain(result: Mapping[str, Any]) -> bool:
    trace = result["trace"]
    return ([step["type"] for step in trace] ==
            ["observation", "decision", "intervention", "plant_response"] and
            trace[1]["payload"] == trace[2]["payload"] and
            trace[2]["authorized"] is True and
            trace[3]["origin"] == "offline_fixture")


def fixed_reference(_observations: Sequence[Mapping[str, Any]], authority: Sequence[str]) -> str:
    return "restore" if "restore" in authority else "observe"


def fixed_deterministic(observations: Sequence[Mapping[str, Any]], authority: Sequence[str]) -> str:
    alarm = any(bool(item.get("alarm")) for item in observations)
    return "restore" if alarm and "restore" in authority else "observe"


def observation_authority_factorial(observation_modes: Sequence[str],
                                    authority_modes: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    """Separate declared cells; interventions must be explicit per authority level."""
    if len(set(observation_modes)) != len(observation_modes):
        raise ValueError("duplicate observation mode")
    return [{"cell_id": f"obs-{obs}__authority-{name}", "observation_mode": obs,
             "authority_mode": name, "allowed_interventions": list(actions)}
            for obs in observation_modes for name, actions in authority_modes.items()]


def adaptive_local_fixture(seeds: Sequence[int], policy: Callable[[int, int], str],
                           score: Callable[[int, str], float], rounds: int = 3) -> list[dict[str, Any]]:
    """Bounded local wargame; seed/round history is separate from fixed studies."""
    if rounds < 1 or rounds > 20:
        raise ValueError("rounds must be 1..20")
    history = []
    for seed in seeds:
        for round_no in range(1, rounds + 1):
            action = policy(seed, round_no)
            history.append({"seed": seed, "round": round_no, "action": action,
                            "fixture_score": score(seed, action),
                            "evidence_origin": "adaptive_local_fixture"})
    return history
