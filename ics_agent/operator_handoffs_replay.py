"""Replay authored operator handoffs offline, without a plant or action broker."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any

from .operator_handoffs import EvidenceSample, HandoffTracker, specs_from_mapping
from .utils import load_yaml


def replay(mapping: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    if scenario.get("mode") != "synthetic_authored":
        raise ValueError("offline replay requires mode: synthetic_authored")
    if scenario.get("utility") != mapping.get("utility"):
        raise ValueError("scenario utility does not match its handoff mapping")

    tracker = HandoffTracker(specs_from_mapping(mapping))
    input_bytes = json.dumps(
        {"mapping": mapping, "scenario": scenario},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    handles: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    previous_at = -1.0

    for index, step in enumerate(scenario["steps"]):
        at = float(step["at_sec"])
        if not isfinite(at) or at < previous_at or at < 0:
            raise ValueError(f"step {index}: times must be finite and nondecreasing")
        previous_at = at
        kind = step["event"]
        alias = str(step["handoff"])

        if kind == "request":
            if alias in handles:
                raise ValueError(f"step {index}: duplicate handoff alias {alias}")
            request_id = tracker.request(str(step["spec"]), by=str(step["by"]), at=at)
            handles[alias] = request_id
            status = "requested"
            reasons: tuple[str, ...] = ()
        else:
            request_id = handles[alias]
            if kind == "acknowledge":
                assessment = tracker.acknowledge(request_id, by=str(step["by"]), at=at)
            elif kind == "complete":
                evidence = tuple(EvidenceSample(**item) for item in step["evidence"])
                assessment = tracker.complete(
                    request_id, by=str(step["by"]), at=at, evidence=evidence,
                )
            elif kind == "recommend":
                assessment = tracker.record_recommendation(
                    request_id, by=str(step["by"]), at=at,
                    choice=str(step["choice"]), basis=str(step["basis"]),
                )
            elif kind == "assess":
                assessment = tracker.assess(request_id, at=at)
            else:
                raise ValueError(f"step {index}: unknown event {kind}")
            status, reasons = assessment.status, assessment.reasons

        results.append({
            "step": index, "event": kind, "handoff": alias,
            "time_sec": at, "handoff_status": status, "reasons": list(reasons),
        })

    return {
        "schema": "ctf-operator-handoff-replay",
        "mode": "synthetic_authored",
        "utility": mapping["utility"],
        "scenario": scenario["scenario"],
        "source_revision": mapping["source"]["revision"],
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "results": results,
        "final_handoffs": {
            alias: tracker.assess(request_id, at=previous_at).status
            for alias, request_id in handles.items()
        },
        "events": tracker.events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(load_yaml(args.mapping), load_yaml(args.scenario)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
