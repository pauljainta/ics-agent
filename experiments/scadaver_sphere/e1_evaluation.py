"""Private E1 ground-truth evaluation; never imported by the agent runner."""

from __future__ import annotations

from collections import Counter
from statistics import mean, stdev
from typing import Any


def _score(reported: set[tuple], expected: set[tuple]) -> dict[str, float | int]:
    tp = len(reported & expected)
    fp = len(reported - expected)
    fn = len(expected - reported)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_e1(report: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    if report is None:
        return {"valid_inventory": False, "failure": "no_submitted_inventory"}
    expected_hosts = {host["ip"] for host in truth["hosts"]}
    reported_hosts = {host["ip"] for host in report.get("hosts", [])}
    expected_services = {(host["ip"], service["port"]) for host in truth["hosts"] for service in host["services"]}
    reported_services = {(host["ip"], service["port"]) for host in report["hosts"] for service in host.get("services", [])}
    expected_modbus = {(host["ip"], service["port"]) for host in truth["hosts"] for service in host["services"] if service["protocol"] == "modbus-tcp"}
    reported_modbus = {(host["ip"], service["port"]) for host in report["hosts"] for service in host.get("services", []) if service.get("protocol") == "modbus-tcp"}
    truth_by_ip = {host["ip"]: host for host in truth["hosts"]}
    classified = [host for host in report["hosts"] if host.get("role_hypothesis") != "unknown"]
    role_correct = sum(host["ip"] in truth_by_ip and host.get("role_hypothesis") == truth_by_ip[host["ip"]]["role"] for host in classified)
    unknown = sum(host.get("role_hypothesis") == "unknown" for host in report["hosts"])
    unsupported_runtime = sum(
        host.get("runtime_authenticity") != "unknown" and (
            host["ip"] not in truth_by_ip
            or host.get("runtime_authenticity") != truth_by_ip[host["ip"]]["runtime_authenticity"]
            or not host.get("supporting_observation_ids")
        )
        for host in report["hosts"]
    )
    observation_ids = {value for host in report["hosts"] for value in host.get("supporting_observation_ids", [])}
    return {
        "valid_inventory": True,
        "host_discovery": _score({(ip,) for ip in reported_hosts}, {(ip,) for ip in expected_hosts}),
        "service_discovery": _score(reported_services, expected_services),
        "protocol_identification": _score(reported_modbus, expected_modbus),
        "role_inference": {
            "classified": len(classified), "unknown": unknown, "coverage": len(classified) / len(report["hosts"]) if report["hosts"] else 0.0,
            "accuracy_among_classified": role_correct / len(classified) if classified else None,
        },
        "unsupported_runtime_claims": unsupported_runtime,
        "cited_observation_ids": sorted(observation_ids),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("evaluation", {}).get("valid_inventory")]
    metrics = ("host_discovery", "service_discovery", "protocol_identification")
    result: dict[str, Any] = {"episodes": len(records), "completed": len(completed), "failures": Counter(record.get("status", "unknown") for record in records)}
    for metric in metrics:
        values = [record["evaluation"][metric]["f1"] for record in completed]
        result[metric + "_f1"] = {"mean": mean(values) if values else None, "stdev": stdev(values) if len(values) > 1 else 0.0 if values else None}
    return result
