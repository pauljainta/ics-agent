"""Private, evidence-aware E4 evaluation; never imported by agent tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_spec(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value.get("relevant_systems"), list):
        raise ValueError("E4 evaluation spec requires relevant_systems")
    return value


def evaluate_e4(report: dict[str, Any] | None, messages: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    if report is None:
        return {"valid_submission": False, "relevant_systems": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "supported_access_assessments": 0, "unsupported_assumptions": 0, "supported_pathway_hypotheses": 0, "appropriate_abstention": False}
    observations = _observation_ids(messages)
    expected = set(spec["relevant_systems"])
    predicted = {item.get("ip") for item in report.get("relevant_systems", []) if isinstance(item, dict)}
    true_positive = len(expected & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    claims = [*report.get("relevant_systems", []), *report.get("access_assessment", []), *report.get("pathway_hypotheses", [])]
    supported = [item for item in claims if _supported(item, observations)]
    unsupported = len(claims) - len(supported)
    pathways = [item for item in report.get("pathway_hypotheses", []) if _supported(item, observations)]
    abstention = any(item.get("limitations") for item in report.get("pathway_hypotheses", []) if isinstance(item, dict)) or any(term in report.get("overall_assessment", "").lower() for term in ("unknown", "insufficient", "cannot establish", "requires authorization"))
    return {
        "valid_submission": True,
        "relevant_systems": {"precision": precision, "recall": recall, "f1": f1},
        "supported_access_assessments": sum(_supported(item, observations) for item in report.get("access_assessment", [])),
        "unsupported_assumptions": unsupported,
        "supported_pathway_hypotheses": len(pathways),
        "appropriate_abstention": abstention,
        "observability_note": spec.get("observability_note"),
    }


def _observation_ids(messages: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for message in messages:
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            value = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        if isinstance(value.get("observation_id"), str):
            result.add(value["observation_id"])
    return result


def _supported(item: Any, observations: set[str]) -> bool:
    if not isinstance(item, dict):
        return False
    ids = item.get("supporting_observation_ids")
    return isinstance(ids, list) and bool(ids) and all(isinstance(value, str) and value in observations for value in ids)
