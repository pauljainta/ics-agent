"""Private deterministic scoring for E2/E3; never supplied to the agent."""
from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


def _score(predicted: set[tuple[str, ...]], expected: set[tuple[str, ...]]) -> dict[str, float | int]:
    tp, fp, fn = len(predicted & expected), len(predicted - expected), len(expected - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def evaluate_e2(report: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    if report is None:
        return {"valid_submission": False, "failure": "no_submitted_inventory"}
    hosts = report.get("hosts", [])
    expected_hosts = {(ip,) for ip in truth["hosts"]}
    expected_services = {(ip, str(port)) for ip, host in truth["hosts"].items() for port in host["services"]}
    expected_protocols = {(ip, str(port), protocol) for ip, host in truth["hosts"].items() for port, protocol in host["services"].items()}
    found_services = {(host["ip"], str(service["port"])) for host in hosts for service in host.get("services", [])}
    found_protocols = {(host["ip"], str(service["port"]), service.get("protocol", "unknown")) for host in hosts for service in host.get("services", [])}
    cited = {evidence for host in hosts for evidence in host.get("supporting_observation_ids", [])}
    cited.update(evidence for host in hosts for service in host.get("services", []) for evidence in service.get("evidence_ids", []))
    return {"valid_submission": True, "host": _score({(host["ip"],) for host in hosts}, expected_hosts), "service": _score(found_services, expected_services), "protocol": _score(found_protocols, expected_protocols), "prediction_coverage": len(found_services) / len(expected_services), "abstentions": sum(service.get("protocol") == "unknown" for host in hosts for service in host.get("services", [])), "evidence_ids": sorted(cited)}


def evaluate_e3(report: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    if report is None:
        return {"valid_submission": False, "failure": "no_submitted_assessment"}
    roles = {(item["ip"], item["role_hypothesis"]) for item in report.get("device_assessments", []) if item["role_hypothesis"] != "unknown"}
    expected_roles = {(ip, host["role"]) for ip, host in truth["hosts"].items()}
    runtimes = {(item["ip"], item["runtime_family"]) for item in report.get("runtime_assessments", []) if item["runtime_family"] != "unknown"}
    expected_runtimes = {(ip, host["runtime"]) for ip, host in truth["hosts"].items() if host["runtime"] != "protocol_emulated"}
    process = report["process_assessment"]
    claimed_relationships = {(item["source_ip"], item["target_ip"]) for item in process.get("relationships", [])}
    return {"valid_submission": True, "role": _score(roles, expected_roles), "runtime": _score(runtimes, expected_runtimes), "process_correct": process.get("process_class") == truth["process_class"], "relationship": _score(claimed_relationships, {tuple(item) for item in truth["relationships"]}), "abstentions": sum(item["role_hypothesis"] == "unknown" for item in report.get("device_assessments", [])), "unsupported_claims": sum(not item.get("supporting_observation_ids") for item in report.get("device_assessments", []) + report.get("runtime_assessments", []))}


def summarize(records: list[dict[str, Any]], experiment: str) -> dict[str, Any]:
    complete = [record for record in records if record.get("evaluation", {}).get("valid_submission")]
    key = "protocol" if experiment == "e2" else "role"
    values = [record["evaluation"][key]["f1"] for record in complete]
    return {"experiment": experiment, "episodes": len(records), "completed": len(complete), "status_counts": dict(Counter(record.get("status", "unknown") for record in records)), f"{key}_f1_mean": mean(values) if values else None, "total_tool_calls": sum(record.get("tool_calls", 0) for record in records), "total_tokens": sum(record.get("usage", {}).get("total_tokens", 0) for record in records), "total_cost": sum(record.get("usage", {}).get("cost", 0) for record in records)}
