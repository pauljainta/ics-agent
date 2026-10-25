"""Import CPS-VR read-only JSONL observations into the PLCBench contract."""

from __future__ import annotations

import json
from pathlib import Path

from .plcbench_observations import ExternalObservation


def import_cpsvr(path: Path) -> list[ExternalObservation]:
    results = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("schema_version") != "plcbench.external-observation/v1":
            raise ValueError(f"line {number}: unsupported CPS-VR schema")
        if raw.get("source") != "CPS-VR" or raw.get("authority") != "read_only":
            raise ValueError(f"line {number}: unexpected source or authority")
        targets = raw.get("targets")
        if not isinstance(targets, list):
            raise ValueError(f"line {number}: targets must be a list")
        results.append(ExternalObservation(raw["kind"], raw["source"],
            raw["observed_at_utc"], {"scene": raw["scene"], "targets": targets}))
    return results
