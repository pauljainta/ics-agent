#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from ics_agent.config import load_config, require_fields
from ics_agent.environment import OpenPLCEnvironment, _SphereOpenPLCBackend
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from ics_agent.reporting import write_json
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.types import RunBundle
from ics_agent.utils import utc_now


CONFIG_PATH = REPO_ROOT / "configs" / "experiments" / "sphere_optionb_smoke.yaml"
PROGRESS_ROOT = (
    WORKSPACE_ROOT / "Artifact" / "Results" / "New" / "_progress_optionb"
)
UNIT_TEST_RESULT_PATH = PROGRESS_ROOT / "unit_test_result.json"


def _ensure_env_defaults() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTHONHASHSEED", "20260622")
    if not os.environ.get("QWEN_MODEL_PATH"):
        os.environ["QWEN_MODEL_PATH"] = str(
            (REPO_ROOT / "models" / "local_models" / "qwen3.5-9b").resolve()
        )
    if os.environ.get("QWEN_MODEL_ID") == "Qwen/Qwen3.5-8B":
        os.environ["QWEN_MODEL_ID"] = "Qwen/Qwen3.5-9B"
    os.environ.setdefault("QWEN_MODEL_ID", "Qwen/Qwen3.5-9B")


def _ssh_result(backend: _SphereOpenPLCBackend, target: str, command: str) -> dict[str, Any]:
    output = backend._ssh_checked(target, command).strip()
    return {
        "target": target,
        "command": command,
        "output": output,
    }


def _engineering_connectivity(backend: _SphereOpenPLCBackend, target: str) -> dict[str, Any]:
    backend._ensure_remote_packages(target, ["python3-pymodbus"])
    remote_command = f"""python3 - <<'PY'
import json
import socket

try:
    from pymodbus.client import ModbusTcpClient
except Exception:
    from pymodbus.client.sync import ModbusTcpClient

checks = {{}}
for name, host, port in [
    ("controller", "{backend.controller_process_ip}", {backend.remote_controller_port}),
    ("simulator", "{backend.simulator_process_ip}", {backend.remote_simulator_port}),
]:
    item = {{}}
    try:
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        item["tcp_connect"] = True
    except Exception as exc:
        item["tcp_connect"] = False
        item["tcp_error"] = str(exc)

    try:
        client = ModbusTcpClient(host, port=port, timeout=5)
        connected = bool(client.connect())
        item["modbus_connect"] = connected
        if connected:
            try:
                result = client.read_coils(0, 4)
            except TypeError:
                result = client.read_coils(address=0, count=4)
            item["coil_read_ok"] = not bool(result.isError())
            bits = getattr(result, "bits", None)
            if bits is not None:
                item["coil_sample"] = list(bits[:4])
        client.close()
    except Exception as exc:
        item["modbus_connect"] = False
        item["modbus_error"] = str(exc)

    checks[name] = item

print(json.dumps(checks))
PY"""
    return json.loads(backend._ssh_checked(target, remote_command))


def _reasoning_pass(
    *,
    config: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    require_fields(config, ["scenario_selectors", "conditions"], ctx="smoke config")
    orchestrator = ExperimentOrchestrator()
    catalog = ScenarioCatalog()
    scenario = catalog.resolve_selectors(list(config.get("scenario_selectors", [])))[0]
    condition_spec = list(config.get("conditions", []))[0]
    condition = dict(condition_spec.get("profiles") or {})
    bundle = RunBundle(
        experiment_id="sphere_optionb_smoke",
        condition_name=str(condition_spec.get("name") or "optionb_smoke"),
        run_id="sphere_optionb_smoke",
        scenario=scenario,
        run_dir=PROGRESS_ROOT,
        config_snapshot={},
    )
    planned_events = orchestrator._build_plan(scenario)
    last_state = orchestrator._record_observation(
        bundle,
        observation,
        elapsed=0.0,
        phase="pre_action",
        last_state=None,
    )
    _ = last_state
    context = orchestrator._live_context(
        scenario=scenario,
        bundle=bundle,
        planned_events=planned_events,
        observation=bundle.observations[-1],
        elapsed=0.0,
    )

    attacker_profile = condition["attacker"]
    defender_profile = condition["defender"]
    attacker = orchestrator._create_agent(attacker_profile)
    defender = orchestrator._create_agent(defender_profile)
    attacker.reset(scenario)
    defender.reset(scenario)

    if isinstance(getattr(attacker, "profile", None), dict):
        attacker.profile["allowed_action_types"] = ["inspect_surface", "noop"]
    if isinstance(getattr(defender, "profile", None), dict):
        defender.profile["allowed_action_types"] = ["hmi_stop", "hmi_estop", "noop"]

    attacker_decisions = [decision.as_dict() for decision in attacker.step(deepcopy(context))]
    defender_decisions = [decision.as_dict() for decision in defender.step(deepcopy(context))]

    return {
        "scenario": scenario.catalog_id,
        "condition": bundle.condition_name,
        "attacker_profile": attacker_profile,
        "defender_profile": defender_profile,
        "attacker_decisions": attacker_decisions,
        "defender_decisions": defender_decisions,
    }


def main() -> int:
    PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_env_defaults()
    config = load_config(CONFIG_PATH)
    env = OpenPLCEnvironment(config.get("environment", {}))
    result: dict[str, Any] = {
        "started_utc": utc_now(),
        "config_path": portable_path(CONFIG_PATH),
        "status": "failed",
    }

    try:
        startup = env.start()
        backend = env._backend_impl
        if not isinstance(backend, _SphereOpenPLCBackend):
            raise RuntimeError("Option B smoke requires the sphere_openplc backend")

        scenario_selector = list(config.get("scenario_selectors", []))[0]
        scenario = ScenarioCatalog().resolve_selectors([scenario_selector])[0]
        reset_snapshot = env.reset(scenario.execution.get("initial_conditions"))
        before_observation = env.observe_flat()
        reasoning = _reasoning_pass(config=config, observation=before_observation)
        benign_action = env.act({"type": "hmi_stop", "process_id": "WT-P1"})
        after_observation = env.observe_flat()

        result.update(
            {
                "status": "passed",
                "completed_utc": utc_now(),
                "startup": startup,
                "reset_state": {
                    "state_name": reset_snapshot.get("STATE_NAME")
                    or reset_snapshot.get("state_name"),
                    "rw_tank_level": reset_snapshot.get("RW_Tank_Level"),
                    "uf_tank_level": reset_snapshot.get("UF_UFFT_Tank_Level"),
                },
                "staged_reachability": {
                    "access": (
                        _ssh_result(backend, backend.access_host, "hostname")
                        if backend.access_host
                        else None
                    ),
                    "engineering": (
                        _ssh_result(backend, backend.engineering_host, "hostname")
                        if backend.engineering_host
                        else None
                    ),
                    "engineering_to_plant": (
                        _engineering_connectivity(backend, backend.engineering_host)
                        if backend.engineering_host
                        else None
                    ),
                },
                "local_observation": {
                    "before": {
                        "state_name": before_observation.get("STATE_NAME"),
                        "rw_tank_level": before_observation.get("RW_Tank_Level"),
                        "uf_tank_level": before_observation.get("UF_UFFT_Tank_Level"),
                    },
                    "after": {
                        "state_name": after_observation.get("STATE_NAME"),
                        "rw_tank_level": after_observation.get("RW_Tank_Level"),
                        "uf_tank_level": after_observation.get("UF_UFFT_Tank_Level"),
                    },
                },
                "benign_action": benign_action,
                "local_reasoning": reasoning,
            }
        )
        return 0
    except Exception as exc:
        result["completed_utc"] = utc_now()
        result["error"] = str(exc)
        return 1
    finally:
        write_json(UNIT_TEST_RESULT_PATH, result)
        try:
            env.stop()
        except Exception as exc:
            payload = json.loads(UNIT_TEST_RESULT_PATH.read_text())
            payload.setdefault("teardown", {})
            payload["teardown"]["error"] = str(exc)
            write_json(UNIT_TEST_RESULT_PATH, payload)


if __name__ == "__main__":
    raise SystemExit(main())
