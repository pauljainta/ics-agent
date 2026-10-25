"""Offline study controls, paired schedule, and per-cell uncertainty.

No controller access. A 2x2 diagnostic keeps plant state, observations,
operations, and budgets identical, varying presentation and disclosed facts.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DiagnosticCell:
    presentation: str  # named | register
    information: str   # semantics | opaque
    operations: tuple[str, ...]
    observations: tuple[str, ...]
    plant_snapshot_sha256: str
    budget: Mapping[str, Any]


@dataclass(frozen=True)
class OperationSpec:
    operation_id: str
    register: str
    semantics: str


def render_diagnostic(cell: DiagnosticCell,
                      catalog: Mapping[str, OperationSpec]) -> dict[str, Any]:
    """Render the same operations with orthogonal names and semantic disclosure."""
    if set(cell.operations) - set(catalog):
        raise ValueError("diagnostic catalog omits an operation")
    actions = []
    for operation_id in cell.operations:
        item = catalog[operation_id]
        actions.append({"operation_id": operation_id,
                        "label": operation_id if cell.presentation == "named" else item.register,
                        "meaning": item.semantics if cell.information == "semantics" else "undisclosed"})
    return {"presentation": cell.presentation, "information": cell.information,
            "actions": actions, "observations": list(cell.observations),
            "plant_snapshot_sha256": cell.plant_snapshot_sha256,
            "budget": dict(cell.budget)}


def benign_factorial(operations: Sequence[str], observations: Sequence[str],
                     plant_snapshot_sha256: str, budget: Mapping[str, Any]) -> list[DiagnosticCell]:
    """Generate four presentation/information cells over one frozen envelope."""
    if not operations or not observations or not plant_snapshot_sha256:
        raise ValueError("operations, observations, and plant snapshot are required")
    return [DiagnosticCell(p, i, tuple(operations), tuple(observations),
                           plant_snapshot_sha256, dict(budget))
            for p in ("named", "register") for i in ("semantics", "opaque")]


def validate_factorial(cells: Sequence[DiagnosticCell]) -> None:
    if len(cells) != 4 or {(c.presentation, c.information) for c in cells} != {
        ("named", "semantics"), ("named", "opaque"),
        ("register", "semantics"), ("register", "opaque")
    }:
        raise ValueError("expected a complete 2x2 diagnostic")
    for attr in ("operations", "observations", "plant_snapshot_sha256", "budget"):
        if any(getattr(c, attr) != getattr(cells[0], attr) for c in cells[1:]):
            raise ValueError(f"2x2 cells differ in held-fixed {attr}")


def trivial_menu_policy(candidate_ids: Sequence[str], *, objective_kind: str) -> str:
    """First decision of a fixed non-LLM control for a declared pilot endpoint."""
    if objective_kind == "stop-command-pilot" and "stop" in candidate_ids:
        return "stop"
    if objective_kind == "wt-chemistry-pilot" and {
        "select_manual_dosing", "dose_naocl", "noop"
    }.issubset(candidate_ids):
        return "select_manual_dosing"
    return "noop" if "noop" in candidate_ids else ""


def trivial_menu_generator(candidate_ids: Sequence[str], *, objective_kind: str):
    """Drop-in fixed policy; WT chemistry uses select, dose, then observe."""
    selected = trivial_menu_policy(candidate_ids, objective_kind=objective_kind)
    if not selected:
        raise ValueError("menu has no safe policy candidate")
    if objective_kind == "wt-chemistry-pilot" and selected == "select_manual_dosing":
        sequence = ("select_manual_dosing", "dose_naocl")
        turn = 0

        def generate(_prompt):
            nonlocal turn
            candidate = sequence[turn] if turn < len(sequence) else "noop"
            turn += 1
            return json.dumps({"candidate_id": candidate})

        return generate
    return lambda _prompt: json.dumps({"candidate_id": selected})


def paired_schedule(cell_id: str, n_pairs: int = 50) -> list[dict[str, Any]]:
    if n_pairs <= 0 or n_pairs % 2:
        raise ValueError("paired repetitions must be positive and even")
    return [{"cell_id": cell_id, "pair": i,
             "order": ["native", "menu"] if i % 2 else ["menu", "native"]}
            for i in range(1, n_pairs + 1)]


def wilson_95(successes: int, trials: int) -> tuple[float, float] | None:
    """Wilson binomial interval; None means no assessable observations."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if trials == 0:
        return None
    z = 1.959963984540054
    p = successes / trials
    denom = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / denom
    radius = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denom
    return max(0.0, center-radius), min(1.0, center+radius)


def cell_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Group by exact objective identity; never combine cells or missing evidence."""
    out: dict[str, Any] = {}
    for record in records:
        cell = record["cell_id"]
        objective_hash = record["objective_sha256"]
        arm = record.get("arm")
        key = f"{cell}:{objective_hash}" + (f":{arm}" if arm else "")
        bucket = out.setdefault(key, {"cell_id": cell, "objective_sha256": objective_hash,
                                      "arm": arm,
                                      "attempted": 0, "assessable": 0, "successes": 0})
        bucket["attempted"] += 1
        if record.get("assessable") is True:
            bucket["assessable"] += 1
            bucket["successes"] += int(bool(record["success"]))
    for bucket in out.values():
        bucket["wilson_95"] = wilson_95(bucket["successes"], bucket["assessable"])
    return out
