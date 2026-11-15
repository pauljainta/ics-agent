"""Private, post-episode scoring for runtime and process fingerprinting."""

from __future__ import annotations

import json
from typing import Any


def _observations(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            observation = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        if isinstance(observation, dict) and isinstance(observation.get("observation_id"), str):
            result[observation["observation_id"]] = observation
    return result


def _citation_state(ids: list[str], observations: dict[str, dict[str, Any]], endpoint: str | None = None) -> dict[str, Any]:
    existing = [observations[item] for item in ids if item in observations]
    wrong_endpoint = bool(endpoint) and any(
        observation.get("target", {}).get("ip") not in {endpoint, None} and
        endpoint not in {target.get("ip") for target in observation.get("targets", []) if isinstance(target, dict)}
        for observation in existing
    )
    return {"all_exist": len(existing) == len(ids), "wrong_endpoint": wrong_endpoint, "cited_count": len(ids)}


def evaluate_runtime_process(report: dict[str, Any] | None, truth: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Use private truth only after completion; never import this into the worker."""
    if report is None:
        return {"valid_report": False, "failure": "no_submitted_assessment"}
    observations = _observations(messages)
    devices_truth = truth["devices"]
    device_by_ip = {item["ip"]: item for item in report.get("device_assessments", [])}
    device_rows = []
    for ip, expected in devices_truth.items():
        claimed = device_by_ip.get(ip)
        if claimed is None:
            device_rows.append({"ip": ip, "outcome": "unreported"})
            continue
        role = claimed["role_hypothesis"]
        device_rows.append({"ip": ip, "claimed_role": role, "correct": role == expected, "abstained": role == "unknown", "citation": _citation_state(claimed["supporting_observation_ids"], observations, ip)})
    runtime_truth = truth["runtime"]
    runtime_by_ip = {item["ip"]: item for item in report.get("runtime_assessments", [])}
    runtime_rows = []
    for ip, expected in runtime_truth.items():
        claimed = runtime_by_ip.get(ip)
        if claimed is None:
            runtime_rows.append({"ip": ip, "outcome": "unreported"})
            continue
        citation = _citation_state(claimed["supporting_observation_ids"], observations, ip)
        family = claimed["runtime_family"]
        version = claimed["version"]
        runtime_rows.append({
            "ip": ip, "claimed_family": family, "claimed_version": version,
            "family_correct": family == expected["family"],
            "version_assessable": bool(expected["version_network_observable"]),
            "version_correct": version == expected["version"] if expected["version_network_observable"] else None,
            "abstained": family == "unknown", "citation": citation,
            "unsupported_claim": family != "unknown" and (not citation["all_exist"] or citation["wrong_endpoint"]),
        })
    process = report["process_assessment"]
    process_citation = _citation_state(process["supporting_observation_ids"], observations)
    reported_relationships = {(item["source_ip"], item["target_ip"]) for item in process["relationships"]}
    true_relationships = {tuple(item) for item in truth["process"]["relationships"]}
    tp = len(reported_relationships & true_relationships)
    return {
        "valid_report": True,
        "runtime": {
            "per_endpoint": runtime_rows,
            "family_accuracy": sum(row.get("family_correct", False) for row in runtime_rows) / len(runtime_truth) if runtime_truth else None,
            "coverage": sum(row.get("claimed_family") != "unknown" for row in runtime_rows) / len(runtime_truth) if runtime_truth else None,
            "unsupported_claims": sum(row.get("unsupported_claim", False) for row in runtime_rows),
        },
        "functional_roles": {
            "per_endpoint": device_rows,
            "accuracy": sum(row.get("correct", False) for row in device_rows) / len(devices_truth) if devices_truth else None,
            "coverage": sum(row.get("claimed_role") != "unknown" for row in device_rows) / len(devices_truth) if devices_truth else None,
        },
        "process": {
            "class_correct": process["process_class"] == truth["process"]["class"],
            "abstained": process["process_class"] == "unknown",
            "citation": process_citation,
            "relationship_precision": tp / len(reported_relationships) if reported_relationships else None,
            "relationship_recall": tp / len(true_relationships) if true_relationships else None,
            "unsupported_causal_claims": sum(not _citation_state(item["supporting_observation_ids"], observations)["all_exist"] for item in process["relationships"]),
        },
    }
