"""Resolve SPHERE experiment references without duplicating its tag schema."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def resolve_sphere_descriptor(path: Path, *, plcbench_root: Path) -> dict[str, Any]:
    descriptor = yaml.safe_load(path.read_text())
    if descriptor.get("schema_version") != "sphere.plcbench-reference/v1":
        raise ValueError("unsupported SPHERE descriptor")
    base = path.parent
    refs = {}
    for key in ("tag_contract_ref", "generated_map_ref", "run_manifest_ref"):
        target = (base / descriptor[key]).resolve()
        if not target.is_file() or not target.is_relative_to(base.resolve()):
            raise FileNotFoundError(target)
        refs[key] = {"path": descriptor[key],
                     "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    config = (plcbench_root / descriptor["plcbench_config_ref"]).resolve()
    if not config.is_file() or not config.is_relative_to(plcbench_root.resolve()):
        raise FileNotFoundError(config)
    refs["plcbench_config_ref"] = {"path": descriptor["plcbench_config_ref"],
                                  "sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
    return {"schema_version": descriptor["schema_version"],
            "realization": descriptor["realization"], "refs": refs,
            "adapter_authority": descriptor["adapter_authority"]}
