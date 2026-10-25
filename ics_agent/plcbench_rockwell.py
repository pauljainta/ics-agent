"""Offline Rockwell admission, replay, and restoration evidence checks.

This module deliberately performs no controller I/O. A separate approved
executor must acquire program identity and signed readback evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_binding(binding: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    required = ("controller_family", "program_identity", "tag_binding_ref",
                "permitted_writes", "baseline_readback_ref", "restore_readback_ref")
    missing = [key for key in required if not binding.get(key)]
    if missing:
        raise ValueError(f"Rockwell binding missing {missing}")
    tag_path = (root / binding["tag_binding_ref"]).resolve()
    if not tag_path.is_relative_to(root.resolve()) or not tag_path.is_file():
        raise FileNotFoundError(tag_path)
    if not binding["program_identity"].get("checksum"):
        raise ValueError("program identity needs checksum")
    if len(set(binding["permitted_writes"])) != len(binding["permitted_writes"]):
        raise ValueError("duplicate permitted write")
    return {"status": "offline_binding_valid", "tag_binding_sha256": file_sha256(tag_path),
            "controller_family": binding["controller_family"],
            "program_identity": dict(binding["program_identity"]),
            "live_transactions": 0}


def check_replay_restoration(binding: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Compare externally acquired pre/post identity and readbacks; fail closed."""
    identity = binding["program_identity"]
    if evidence.get("before_program") != identity or evidence.get("after_program") != identity:
        raise ValueError("program identity changed or evidence missing")
    before, restored = evidence.get("baseline_readback"), evidence.get("restore_readback")
    if not isinstance(before, dict) or before != restored:
        raise ValueError("restoration readback mismatch or missing")
    replay = evidence.get("replay_log")
    if not isinstance(replay, list) or not all("timestamp" in step and "tag" in step
                                               for step in replay):
        raise ValueError("replay evidence missing timestamped tag entries")
    if any(step["tag"] not in binding["permitted_writes"] for step in replay):
        raise ValueError("replay contains unapproved tag")
    return {"status": "offline_evidence_consistent", "replayed_operations": len(replay),
            "evidence_sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()}
