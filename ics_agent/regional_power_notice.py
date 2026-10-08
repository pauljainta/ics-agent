"""Replay a supplied regional power notice; never infer a physical feeder."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from math import isfinite
from pathlib import Path
from typing import Any

from .utils import load_yaml


def read_pinned_fixture(source_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Read only the exact composition fixture named by a CTF notice plan."""
    root = source_root.resolve()
    source = plan["source"]
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != source["revision"]:
        raise ValueError("source checkout differs from pinned composition revision")
    relative = Path(source["fixture"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("fixture path must remain within the source checkout")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("fixture path escapes the source checkout")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != source["fixture_sha256"]:
        raise ValueError("composition fixture differs from pinned source bytes")
    return json.loads(contents)


def _time(value: Any) -> float:
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError("notice times must be finite nonnegative numbers")
    return result


def _source_window(fixture: dict[str, Any], scenario: str, start: float) -> dict[str, Any]:
    windows = fixture["scenarios"][scenario]["power_schedule"]
    matches = [item for item in windows if _time(item["from_s"]) == start]
    if len(matches) != 1:
        raise ValueError("notice must identify one source power window")
    window = matches[0]
    if _time(window["valid_until_s"]) <= start:
        raise ValueError("source power window has no validity interval")
    quality = window["quality"]
    value = window.get("available_kw")
    if quality not in {"supplied", "unknown"} or (value is None) != (quality == "unknown"):
        raise ValueError("power value and quality must distinguish supplied from unknown")
    if value is not None and (not isfinite(float(value)) or float(value) < 0):
        raise ValueError("supplied power must be finite and nonnegative")
    return window


def _window_at(fixture: dict[str, Any], scenario: str, at: float) -> dict[str, Any]:
    matches = [
        item for item in fixture["scenarios"][scenario]["power_schedule"]
        if _time(item["from_s"]) <= at < _time(item["valid_until_s"])
    ]
    if len(matches) != 1:
        raise ValueError("decision time must belong to one source power window")
    return matches[0]


def replay(plan: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != "ctf-regional-power-notice":
        raise ValueError("unknown regional notice plan schema")
    if plan.get("mode") != "synthetic_authored":
        raise ValueError("regional notice replay must be synthetic_authored")
    if fixture.get("schema") != "sphere.composition-fixture/1":
        raise ValueError("unknown power-water composition fixture schema")
    source = plan["source"]
    if source.get("evidence_class") != "supplied" or source.get("physical_feeder_claim") is not False:
        raise ValueError("site power is a supplied scenario input, not a hydro feeder")
    scenario = source["scenario"]
    if scenario not in fixture["scenarios"]:
        raise ValueError("composition scenario is absent from the pinned fixture")

    results = []
    ids = set()
    for case in plan["cases"]:
        case_id = str(case["id"])
        if case_id in ids:
            raise ValueError("regional notice case IDs must be unique")
        ids.add(case_id)
        start = _time(case["source_window_from_sec"])
        window = _source_window(fixture, scenario, start)
        request_at = _time(case["request_at_sec"])
        acknowledge_at = _time(case["acknowledge_at_sec"])
        notice_at = _time(case["notice_at_sec"])
        decision_at = _time(case["decision_at_sec"])
        if not request_at <= acknowledge_at <= notice_at or decision_at < request_at:
            raise ValueError(f"{case_id}: invalid handoff chronology")
        choice = case["choice"]
        if choice not in {"recommend_operation", "hold_operation"}:
            raise ValueError(f"{case_id}: unknown operator recommendation")
        if not str(case["basis"]).strip():
            raise ValueError(f"{case_id}: operator recommendation needs a basis")

        if notice_at > decision_at:
            notice_status = "not_received"
        elif not start <= decision_at < _time(window["valid_until_s"]):
            notice_status = "outside_validity"
        elif window["quality"] == "unknown":
            notice_status = "unknown_supplied_input"
        else:
            notice_status = "current_supplied_notice"

        events = [
            {"event_type": "requested", "actor": plan["roles"]["requester"],
             "time_sec": request_at},
        ]
        if acknowledge_at <= decision_at:
            events.append({"event_type": "acknowledged", "actor": plan["roles"]["responder"],
                           "time_sec": acknowledge_at})
        if notice_at <= decision_at:
            events.append({
                "event_type": "notice_received", "actor": plan["roles"]["responder"],
                "time_sec": notice_at, "evidence_class": "supplied",
                "service": source["service"], "available_kw": window["available_kw"],
                "quality": window["quality"], "valid_from_sec": start,
                "valid_until_sec": window["valid_until_s"],
            })
        events.append({
            "event_type": "recommendation", "actor": plan["roles"]["requester"],
            "time_sec": decision_at, "choice": choice, "basis": case["basis"],
            "notice_status": notice_status, "actuation": False,
        })
        results.append({
            "case_id": case_id, "notice_status": notice_status,
            "staff_source_window": window,
            "staff_decision_window": _window_at(fixture, scenario, decision_at),
            "events": events,
        })

    plan_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema": "ctf-regional-power-notice-replay",
        "mode": "synthetic_authored",
        "source_revision": source["revision"],
        "source_fixture_sha256": source["fixture_sha256"],
        "source_scenario": scenario,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "physical_feeder_claim": False,
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sphere_usecases", type=Path)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = load_yaml(args.plan)
    print(json.dumps(replay(plan, read_pinned_fixture(args.sphere_usecases, plan)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
