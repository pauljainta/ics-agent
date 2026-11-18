#!/usr/bin/env python3
"""Static and deterministic pre-deployment gates for V2."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from ..policy import PolicyError
from .constants import PROCESS_CIDR, SIMULATOR_PROCESS
from .schemas import FINAL_SCHEMA, action_schema
from .worker import build_worker

ROOT = Path(__file__).resolve().parents[1]
V2 = Path(__file__).resolve().parent


def run() -> dict[str, Any]:
    topology_path = ROOT / "topology/scadaver_ot_realistic_v2.py"
    topology = topology_path.read_text()
    ast.parse(topology)
    engineering = (V2 / "engineering_runtime.py").read_text()
    worker = build_worker(live=False)
    private_rejected = False
    try:
        worker.policy.target_ip(SIMULATOR_PROCESS)
    except PolicyError:
        private_rejected = True
    schemas = json.dumps([action_schema(), FINAL_SCHEMA], sort_keys=True)
    checks = {
        "topology_parses": True,
        "simulator_not_on_ot_link": "net.connect([controller, access, engineering, decoy])" in topology,
        "private_link_declared": "net.connect([controller, simulator, evaluator])" in topology,
        "access_not_on_private_link": "net.connect([controller, simulator, evaluator, access])" not in topology,
        "routing_reticulator_disabled": "routing == static" not in topology,
        "worker_rejects_private_target": private_rejected,
        "worker_excludes_simulator_port": 1503 not in worker.policy.permitted_ports,
        "engineering_has_no_private_address": str(PROCESS_CIDR.network_address).rsplit(".", 1)[0] not in engineering,
        "engineering_uses_controller_telemetry": (
            "read_input_registers(100" in engineering
            and "read_holding_registers(1" in engineering
        ),
        "action_schema_has_no_private_target": SIMULATOR_PROCESS not in schemas and "1503" not in schemas,
        "action_is_controller_mediated": "bounded_controller_output" in schemas,
        "controller_forwarding_disabled": "net.ipv4.ip_forward=0" in (V2 / "deployment/prepare_controller.sh").read_text(),
        "original_topology_present": (ROOT / "topology/scadaver_openplc_water_fingerprint.py").exists(),
    }
    files = sorted(path for path in V2.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    return {
        "validation": "ot-realistic-v2-offline-v1",
        "checks": checks, "passed": all(checks.values()),
        "source_hashes": {str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in files},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
