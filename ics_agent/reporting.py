from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .utils import dump_yaml


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown_summary(path: Path, bundle: dict[str, Any]) -> None:
    scenario = bundle["scenario"]
    metrics = bundle.get("metrics", {})
    observer_detection = metrics.get("observer_detection", {})
    defender_detection = metrics.get("defender_detection", {})
    defender_intervention = metrics.get("defender_intervention", {})
    lines = [
        f"# {scenario['scenario_id']}",
        "",
        f"- Condition: `{bundle['condition_name']}`",
        f"- Status: `{bundle['status']}`",
        f"- Passed: `{bundle['passed']}`",
        f"- Samples: `{len(bundle.get('observations', []))}`",
        f"- Actions: `{len(bundle.get('executed_actions', []))}`",
        f"- Alerts: `{len(bundle.get('alerts', []))}`",
        f"- Blue-Team TTD: `{metrics.get('time_to_detection_sec')}`",
        f"- Observer TTD: `{observer_detection.get('time_to_detection_sec')}`",
        f"- Defender TTD: `{defender_detection.get('time_to_detection_sec')}`",
        f"- Defender TTM: `{defender_intervention.get('time_to_mitigation_sec')}`",
        f"- Blue-Team TPR: `{metrics.get('true_positive_rate')}`",
        f"- Blue-Team FPR: `{metrics.get('false_positive_rate')}`",
        f"- Safety Violations: `{metrics.get('safety_violation_count')}`",
        "",
        "## Intent",
        scenario["metadata"]["intent"],
        "",
        "## Expected Outcomes",
        *[f"- {item}" for item in scenario["metadata"].get("expected_outcomes", [])],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_run_bundle(run_dir: Path, bundle: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "summary.json", bundle["summary"])
    write_json(run_dir / "metrics.json", bundle["metrics"])
    write_json(run_dir / "actions.json", bundle.get("executed_actions", []))
    write_json(run_dir / "decisions.json", bundle.get("decisions", []))
    write_json(run_dir / "alerts.json", bundle.get("alerts", []))
    write_json(run_dir / "checks.json", bundle.get("expectation_checks", []))
    write_json(run_dir / "system_events.json", bundle.get("system_events", []))
    write_json(run_dir / "offline_analysis.json", bundle.get("offline_analyses", []))
    write_json(run_dir / "errors.json", bundle.get("errors", []))
    write_csv(run_dir / "observations.csv", bundle.get("observations", []))
    dump_yaml(run_dir / "config_snapshot.yaml", bundle["config_snapshot"])
    dump_yaml(run_dir / "scenario_resolved.yaml", bundle["scenario"])
    write_markdown_summary(run_dir / "summary.md", bundle)
