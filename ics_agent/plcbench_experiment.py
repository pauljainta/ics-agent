"""Versioned PLCBench experiment identity; no controller or model I/O."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .plcbench_matched_run import MenuCandidate
from .plcbench_scorer import Objective


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_experiment(path: Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text())
    if spec.get("schema_version") != "plcbench.experiment/v1":
        raise ValueError("unsupported PLCBench experiment schema")
    for key in ("cell_id", "workload", "objective", "model", "execution", "reset", "sources"):
        if key not in spec:
            raise ValueError(f"experiment missing {key}")
    return spec


def objective_from_spec(spec: Mapping[str, Any]) -> Objective:
    obj = spec["objective"]
    required = ("warning", "impact", "tau_d", "tau_i", "max_gap_sec")
    if any(obj.get(k) is None for k in required):
        raise ValueError("objective thresholds or dwell are not qualified")
    def check_predicate(node: Mapping[str, Any]) -> None:
        if "value" in node and node["value"] is None:
            raise ValueError("objective predicate threshold is not qualified")
        for key in ("all", "any"):
            for child in node.get(key, []):
                check_predicate(child)
        if "not" in node:
            check_predicate(node["not"])
    check_predicate(obj["warning"])
    check_predicate(obj["impact"])
    return Objective(**{key: obj[key] for key in required})


def experiment_identity(
    spec: Mapping[str, Any], *, repo_root: Path, sphere_root: Path | None,
    menu: Sequence[MenuCandidate], sandbox_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve and hash every source needed to interpret a future episode.

    Paths in the returned snapshot are portable repository-relative references;
    source bytes are hashed before any run starts. Missing sources fail closed.
    """
    source_hashes: dict[str, str] = {}
    for ref in spec["sources"]:
        root = repo_root if ref["repo"] == "ics-agent" else sphere_root
        if root is None or ref["repo"] not in ("ics-agent", "sphere-usecases"):
            raise ValueError(f"unresolved source repository: {ref['repo']}")
        path = (Path(root) / ref["path"]).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[f"{ref['repo']}:{ref['path']}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    menu_snapshot = [asdict(candidate) for candidate in menu]
    objective_snapshot = (dict(spec["objective"]) if spec.get("status") == "unqualified"
                          else asdict(objective_from_spec(spec)))
    resolved = {
        "schema_version": spec["schema_version"],
        "cell_id": spec["cell_id"],
        "workload": spec["workload"],
        "objective": objective_snapshot,
        "model": dict(spec["model"]),
        "execution": dict(spec["execution"]),
        "reset": dict(spec["reset"]),
        "command_tags": list(spec["command_tags"]),
        "sensor_tags": list(spec["sensor_tags"]),
        "config_sha256": canonical_hash(spec),
        "menu": menu_snapshot,
        "menu_sha256": canonical_hash(menu_snapshot),
        "objective_sha256": canonical_hash(objective_snapshot),
        "source_hashes": source_hashes,
        "sandbox_config": dict(sandbox_config),
        "sandbox_sha256": canonical_hash(sandbox_config),
    }
    resolved["identity_sha256"] = canonical_hash(resolved)
    return resolved


def assert_study_ready(spec: Mapping[str, Any]) -> None:
    if spec.get("status") != "qualified":
        raise ValueError(f"experiment {spec['cell_id']} is not qualified for measured runs")
    objective_from_spec(spec)
