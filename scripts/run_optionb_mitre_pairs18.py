#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from ics_agent.config import load_config, require_fields, resolve_repo_path
from ics_agent.environment import OpenPLCEnvironment, _SphereOpenPLCBackend
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from ics_agent.reporting import write_json
from ics_agent.utils import utc_now


SEED = 20260623
CONFIG_PATH = REPO_ROOT / "configs" / "experiments" / "sphere_optionb_mitre_pairs18.yaml"
RESULTS_ROOT = WORKSPACE_ROOT / "Artifact" / "Results" / "optionb_mitre_pairs18"
PROGRESS_ROOT = RESULTS_ROOT / "_progress"
PHASE_STATUS_PATH = PROGRESS_ROOT / "phase_status.json"
RUN_MANIFEST_PATH = PROGRESS_ROOT / "run_manifest.json"
COMPLETED_RUNS_PATH = PROGRESS_ROOT / "completed_runs.json"
RESUME_NOTES_PATH = PROGRESS_ROOT / "resume_notes.md"
LAST_STEP_PATH = PROGRESS_ROOT / "last_successful_step.txt"
COMPARISON_REPORT_PATH = PROGRESS_ROOT / "comparison_report.md"
ENVIRONMENT_VALIDATION_PATH = PROGRESS_ROOT / "optionb_environment_validation.json"
VALIDATE_LOG_PATH = PROGRESS_ROOT / "validate_runtime.log"
PYTEST_LOG_PATH = PROGRESS_ROOT / "pytest_targeted.log"

PHASE_ORDER = [
    "validated_optionb_environment",
    "config_prepared",
    "validation_tests",
    "scenario_run",
    "aggregate_metrics",
    "final_report",
]
REQUIRED_RUN_FILES = (
    "summary.json",
    "metrics.json",
    "actions.json",
    "alerts.json",
    "config_snapshot.yaml",
    "system_events.json",
    "environment_snapshot.json",
    "phase_log.jsonl",
)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run_checked(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
    )
    return completed.stdout


def _run_logged(command: list[str], log_path: Path) -> dict[str, Any]:
    started = utc_now()
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
    )
    payload = {
        "command": command,
        "returncode": completed.returncode,
        "started_utc": started,
        "completed_utc": utc_now(),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    write_json(log_path, payload)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )
    return payload


def _ensure_env_defaults() -> None:
    os.environ["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", str(SEED))


def _load_phase_status() -> dict[str, Any]:
    payload = _read_json(PHASE_STATUS_PATH, {})
    if not isinstance(payload, dict):
        payload = {}
    for phase in PHASE_ORDER:
        payload.setdefault(phase, {"status": "pending"})
    payload.setdefault("conditions", {})
    return payload


def _save_phase_status(payload: dict[str, Any]) -> None:
    write_json(PHASE_STATUS_PATH, payload)


def _set_phase(phase: str, status: str, **fields: Any) -> None:
    payload = _load_phase_status()
    item = dict(payload.get(phase, {}))
    item["status"] = status
    item["updated_utc"] = utc_now()
    if status == "in_progress" and "started_utc" not in item:
        item["started_utc"] = item["updated_utc"]
    if status == "completed":
        item["completed_utc"] = item["updated_utc"]
        _write_text(LAST_STEP_PATH, phase + "\n")
    item.update(fields)
    payload[phase] = item
    _save_phase_status(payload)
    _write_resume_notes()


def _set_condition_status(condition_name: str, status: str, **fields: Any) -> None:
    payload = _load_phase_status()
    conditions = dict(payload.get("conditions") or {})
    item = dict(conditions.get(condition_name, {}))
    item["status"] = status
    item["updated_utc"] = utc_now()
    if status == "in_progress" and "started_utc" not in item:
        item["started_utc"] = item["updated_utc"]
    if status == "completed":
        item["completed_utc"] = item["updated_utc"]
        _write_text(LAST_STEP_PATH, f"condition:{condition_name}\n")
    item.update(fields)
    conditions[condition_name] = item
    payload["conditions"] = conditions
    _save_phase_status(payload)
    _write_resume_notes()


def _load_completed_runs() -> dict[str, Any]:
    payload = _read_json(
        COMPLETED_RUNS_PATH,
        {
            "expected_conditions": [],
            "expected_scenarios": [],
            "expected_run_keys": [],
            "completed": {},
            "failed": {},
            "last_updated_utc": None,
        },
    )
    if not isinstance(payload, dict):
        payload = {
            "expected_conditions": [],
            "expected_scenarios": [],
            "expected_run_keys": [],
            "completed": {},
            "failed": {},
            "last_updated_utc": None,
        }
    payload.setdefault("expected_conditions", [])
    payload.setdefault("expected_scenarios", [])
    payload.setdefault("expected_run_keys", [])
    payload.setdefault("completed", {})
    payload.setdefault("failed", {})
    payload["completed"] = dict(payload.get("completed") or {})
    payload["failed"] = dict(payload.get("failed") or {})
    return payload


def _save_completed_runs(payload: dict[str, Any]) -> None:
    payload["last_updated_utc"] = utc_now()
    write_json(COMPLETED_RUNS_PATH, payload)
    _write_resume_notes()


def _write_resume_notes() -> None:
    phase_status = _load_phase_status()
    completed_runs = _load_completed_runs()
    expected = list(completed_runs.get("expected_run_keys") or [])
    completed = dict(completed_runs.get("completed") or {})
    failed = dict(completed_runs.get("failed") or {})
    remaining = [key for key in expected if key not in completed and key not in failed]
    by_condition: dict[str, dict[str, int]] = {}
    for bucket_name, bucket in (("completed", completed), ("failed", failed)):
        for run_key in bucket:
            condition_name = run_key.split("::", 1)[0]
            counts = by_condition.setdefault(
                condition_name,
                {"completed": 0, "failed": 0},
            )
            counts[bucket_name] += 1
    lines = [
        "# Resume Notes",
        "",
        f"- Updated UTC: `{utc_now()}`",
        f"- Last successful step: `{LAST_STEP_PATH.read_text().strip() if LAST_STEP_PATH.exists() else 'none'}`",
        f"- Completed runs: `{len(completed)}`",
        f"- Failed runs: `{len(failed)}`",
        f"- Remaining runs: `{len(remaining)}`",
        "",
        "## Phase Status",
    ]
    for phase in PHASE_ORDER:
        lines.append(f"- `{phase}`: `{phase_status.get(phase, {}).get('status', 'pending')}`")
    if phase_status.get("conditions"):
        lines.extend(["", "## Condition Status"])
        for condition_name, info in sorted(phase_status["conditions"].items()):
            counts = by_condition.get(condition_name, {"completed": 0, "failed": 0})
            lines.append(
                f"- `{condition_name}`: `{info.get('status', 'pending')}` completed `{counts['completed']}` failed `{counts['failed']}`"
            )
    if remaining:
        lines.extend(["", "## Remaining Runs"])
        for run_key in remaining:
            lines.append(f"- `{run_key}`")
    if failed:
        lines.extend(["", "## Failed Runs"])
        for run_key, info in sorted(failed.items()):
            lines.append(
                f"- `{run_key}`: `{info.get('status')}` `{info.get('error', 'see run artifacts')}`"
            )
    _write_text(RESUME_NOTES_PATH, "\n".join(lines) + "\n")


def _run_dir_for(condition_root: Path, scenario: Any, repeat: int = 1) -> Path:
    return condition_root / scenario.suite / scenario.scenario_id / f"trial-{repeat:03d}"


def _run_id_for(scenario: Any, repeat: int = 1) -> str:
    return f"{scenario.suite}-{scenario.scenario_id}-trial-{repeat:03d}"


def _run_key(condition_name: str, scenario: Any, repeat: int = 1) -> str:
    return f"{condition_name}::{scenario.catalog_id}::trial-{repeat:03d}"


def _quick_run_integrity(run_dir: Path) -> bool:
    if not all((run_dir / name).exists() for name in REQUIRED_RUN_FILES):
        return False
    summary = _read_json(run_dir / "summary.json", {})
    return isinstance(summary, dict) and summary.get("status") == "completed"


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json", {})
    metrics = _read_json(run_dir / "metrics.json", {})
    if isinstance(summary, dict) and "metrics" not in summary:
        summary["metrics"] = metrics
    if isinstance(summary, dict) and "run_dir" not in summary:
        summary["run_dir"] = portable_path(run_dir)
    return summary


def _scenario_record(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    metrics = dict(summary.get("metrics", {}) or {})
    phase_window = dict(metrics.get("phase_window") or {})
    attacker = dict(metrics.get("attacker") or {})
    mitre_phase = dict(metrics.get("mitre_phase") or {})
    return {
        "run_id": summary.get("run_id"),
        "scenario": summary.get("scenario"),
        "suite": summary.get("suite"),
        "status": summary.get("status"),
        "passed": summary.get("passed"),
        "run_dir": portable_path(run_dir),
        "phase_a_duration_sec": phase_window.get("phase_a_duration_sec"),
        "wargame_t0_utc": phase_window.get("wargame_t0_utc"),
        "time_to_detection_sec": metrics.get("time_to_detection_sec"),
        "time_to_first_defender_action_sec": metrics.get("defender_intervention", {}).get(
            "time_to_first_action_sec"
        ),
        "attacker_caused_safety_violation": attacker.get("caused_safety_violation"),
        "time_to_physical_impact_sec": attacker.get("time_to_physical_impact_sec"),
        "highest_attacker_phase_reached": mitre_phase.get("highest_attacker_phase_reached"),
        "first_defender_detection_phase": mitre_phase.get("first_defender_detection_phase"),
        "first_defender_action_phase": mitre_phase.get("first_defender_action_phase"),
        "integrity_verified": _quick_run_integrity(run_dir),
    }


def _load_existing_condition_summaries(
    *,
    condition_root: Path,
    scenarios: list[Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        run_dir = _run_dir_for(condition_root, scenario)
        if not _quick_run_integrity(run_dir):
            continue
        summaries.append(_load_run_summary(run_dir))
    return summaries


def _ssh_result(backend: _SphereOpenPLCBackend, target: str, command: str) -> dict[str, Any]:
    output = backend._ssh_checked(target, command).strip()
    return {"target": target, "command": command, "output": output}


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


def _validate_optionb_environment(
    *,
    config: dict[str, Any],
    scenarios: list[Any],
) -> dict[str, Any]:
    env = OpenPLCEnvironment(config.get("environment", {}))
    result: dict[str, Any] = {
        "validated_at_utc": utc_now(),
        "config_path": portable_path(Path(config["_path"])),
        "status": "failed",
    }
    try:
        startup = env.start()
        backend = env._backend_impl
        if not isinstance(backend, _SphereOpenPLCBackend):
            raise RuntimeError("Option B MITRE study requires the sphere_openplc backend")
        scenario = scenarios[0]
        reset_snapshot = env.reset(scenario.execution.get("initial_conditions"))
        observation = env.observe_flat()
        discovery = env.inspect_surface({"type": "inspect_surface", "surface": "tags"})
        result.update(
            {
                "status": "passed",
                "startup": startup,
                "validated_target": {
                    "experiment": backend.experiment_ref,
                    "realization": backend._realization_ref,
                    "revision": backend.revision,
                    "xdc": backend._xdc_ref,
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
                "reset_state": {
                    "state_name": reset_snapshot.get("STATE_NAME")
                    or reset_snapshot.get("state_name"),
                    "rw_tank_level": reset_snapshot.get("RW_Tank_Level"),
                    "uf_tank_level": reset_snapshot.get("UF_UFFT_Tank_Level"),
                },
                "local_observation": {
                    "state_name": observation.get("STATE_NAME"),
                    "rw_tank_level": observation.get("RW_Tank_Level"),
                    "uf_tank_level": observation.get("UF_UFFT_Tank_Level"),
                },
                "discovery_surface_sample": discovery.get("reconnaissance", {}),
            }
        )
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result
    finally:
        write_json(ENVIRONMENT_VALIDATION_PATH, result)
        try:
            env.stop()
        except Exception as exc:
            payload = _read_json(ENVIRONMENT_VALIDATION_PATH, result)
            payload.setdefault("teardown", {})
            payload["teardown"]["error"] = str(exc)
            write_json(ENVIRONMENT_VALIDATION_PATH, payload)


def _prepare_manifest(
    *,
    config: dict[str, Any],
    scenarios: list[Any],
    conditions: list[dict[str, Any]],
    experiment_root: Path,
    reproducibility: dict[str, Any],
) -> None:
    condition_names = [
        item.get("name")
        or "__".join(
            f"{role}-{profile}"
            for role, profile in sorted(item.get("profiles", {}).items())
        )
        for item in conditions
    ]
    manifest = {
        "generated_at_utc": utc_now(),
        "config_path": portable_path(Path(config["_path"])),
        "experiment_name": config.get("experiment_name"),
        "experiment_id": config.get("experiment_id", config.get("experiment_name")),
        "results_root": portable_path(experiment_root),
        "progress_root": portable_path(PROGRESS_ROOT),
        "condition_names": condition_names,
        "scenario_catalog_ids": [scenario.catalog_id for scenario in scenarios],
        "expected_run_count": len(condition_names) * len(scenarios),
        "seed": SEED,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "metric_boundary": {
            "phase_a": "setup/bootstrap and transfer latency",
            "wargame_t0": "interaction-ready boundary after staged path verification and environment reset",
            "phase_b": "measured attack/defense interval only",
        },
        "environment_target": {
            "experiment": config.get("environment", {}).get("experiment"),
            "realization": config.get("environment", {}).get("realization"),
            "revision": config.get("environment", {}).get("revision"),
            "xdc": config.get("environment", {}).get("xdc"),
            "access_host": config.get("environment", {}).get("access_host"),
            "engineering_host": config.get("environment", {}).get("engineering_host"),
        },
        "study_metadata": dict(config.get("study_metadata") or {}),
        "reproducibility": reproducibility,
    }
    write_json(RUN_MANIFEST_PATH, manifest)


def _validate_runtime_and_tests() -> None:
    venv_python = str((REPO_ROOT / ".venv" / "bin" / "python").resolve())
    _run_logged([venv_python, "-m", "ics_agent.cli", "validate"], VALIDATE_LOG_PATH)
    _run_logged(
        [
            venv_python,
            "-m",
            "pytest",
            "tests/test_environment.py",
            "tests/test_orchestrator.py",
            "tests/test_agentic.py",
        ],
        PYTEST_LOG_PATH,
    )


def _condition_name(
    orchestrator: ExperimentOrchestrator,
    condition_spec: dict[str, Any],
) -> str:
    return orchestrator._condition_name(
        condition_spec["profiles"],
        override=condition_spec.get("name"),
    )


def _aggregate_and_write(
    *,
    orchestrator: ExperimentOrchestrator,
    config: dict[str, Any],
    experiment_root: Path,
    condition_specs: list[dict[str, Any]],
    run_summaries_by_condition: dict[str, list[dict[str, Any]]],
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    suite_conditions: list[dict[str, Any]] = []
    for condition_spec in condition_specs:
        condition = condition_spec["profiles"]
        condition_name = _condition_name(orchestrator, condition_spec)
        runs = list(run_summaries_by_condition.get(condition_name, []))
        condition_root = experiment_root / condition_name
        condition_summary: dict[str, Any] = {
            "condition_name": condition_name,
            "profiles": condition,
            "runs": runs,
            "status": (
                "failed"
                if any(run.get("status") != "completed" for run in runs)
                else "completed"
            ),
            "reproducibility": orchestrator._condition_reproducibility(
                condition_name,
                condition,
            ),
        }
        condition_summary["nominal_metrics"] = orchestrator._aggregate_nominal_metrics(runs)
        condition_summary["attack_metrics"] = orchestrator._aggregate_attack_metrics(runs)
        condition_summary["mitre_phase_metrics"] = orchestrator._aggregate_mitre_phase_metrics(
            runs
        )
        condition_summary["trajectory_metrics"] = orchestrator._aggregate_trajectory_metrics(
            runs,
            condition=condition,
        )

        mitre_summary = {
            "condition_name": condition_name,
            "generated_at_utc": utc_now(),
            "highest_attacker_phase_distribution": condition_summary["mitre_phase_metrics"].get(
                "highest_attacker_phase_distribution"
            ),
            "defender_detection_before_manipulation_rate": condition_summary[
                "mitre_phase_metrics"
            ].get("defender_detection_before_manipulation_rate"),
            "first_defender_action_before_manipulation_rate": condition_summary[
                "mitre_phase_metrics"
            ].get("first_defender_action_before_manipulation_rate"),
            "manipulation_to_defender_detection_delay_sec_avg": condition_summary[
                "mitre_phase_metrics"
            ].get("manipulation_to_defender_detection_delay_sec_avg"),
            "manipulation_to_first_defender_action_delay_sec_avg": condition_summary[
                "mitre_phase_metrics"
            ].get("manipulation_to_first_defender_action_delay_sec_avg"),
        }

        write_json(condition_root / "condition_summary.json", condition_summary)
        write_json(condition_root / "attack_metrics_summary.json", condition_summary["attack_metrics"])
        write_json(condition_root / "nominal_metrics_summary.json", condition_summary["nominal_metrics"])
        write_json(condition_root / "mitre_phase_metrics.json", condition_summary["mitre_phase_metrics"])
        write_json(condition_root / "mitre_phase_summary.json", mitre_summary)
        write_json(condition_root / "trajectory_metrics_summary.json", condition_summary["trajectory_metrics"])
        suite_conditions.append(condition_summary)

    all_runs = [
        run
        for condition_summary in suite_conditions
        for run in condition_summary.get("runs", [])
    ]
    suite_summary = {
        "experiment_id": str(config.get("experiment_id", config.get("experiment_name"))),
        "experiment_name": str(config.get("experiment_name")),
        "config_path": portable_path(Path(config["_path"])),
        "results_root": portable_path(experiment_root),
        "timestamp_utc": utc_now(),
        "status": (
            "failed"
            if any(condition.get("status") != "completed" for condition in suite_conditions)
            else "completed"
        ),
        "use_canonical_result_root": bool(config.get("use_canonical_result_root", False)),
        "study_metadata": dict(config.get("study_metadata") or {}),
        "reproducibility": reproducibility,
        "conditions": suite_conditions,
        "nominal_metrics": orchestrator._aggregate_nominal_metrics(all_runs),
        "attack_metrics": orchestrator._aggregate_attack_metrics(all_runs),
        "mitre_phase_metrics": orchestrator._aggregate_mitre_phase_metrics(all_runs),
        "trajectory_metrics": orchestrator._aggregate_trajectory_metrics(all_runs),
    }
    mitre_summary = {
        "generated_at_utc": utc_now(),
        "condition_summaries": [
            {
                "condition_name": condition["condition_name"],
                "highest_attacker_phase_distribution": condition["mitre_phase_metrics"].get(
                    "highest_attacker_phase_distribution"
                ),
                "defender_detection_before_manipulation_rate": condition[
                    "mitre_phase_metrics"
                ].get("defender_detection_before_manipulation_rate"),
                "first_defender_action_before_manipulation_rate": condition[
                    "mitre_phase_metrics"
                ].get("first_defender_action_before_manipulation_rate"),
            }
            for condition in suite_conditions
        ],
    }
    write_json(experiment_root / "experiment_summary.json", suite_summary)
    write_json(experiment_root / "attack_metrics_summary.json", suite_summary["attack_metrics"])
    write_json(experiment_root / "nominal_metrics_summary.json", suite_summary["nominal_metrics"])
    write_json(experiment_root / "mitre_phase_metrics.json", suite_summary["mitre_phase_metrics"])
    write_json(experiment_root / "mitre_phase_summary.json", mitre_summary)
    write_json(experiment_root / "trajectory_metrics_summary.json", suite_summary["trajectory_metrics"])
    return suite_summary


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _condition_report_line(condition_summary: dict[str, Any]) -> str:
    attack_metrics = dict(condition_summary.get("attack_metrics") or {})
    return (
        f"| {condition_summary['condition_name']} | "
        f"{attack_metrics.get('completed_attack_run_count', 0)}/{attack_metrics.get('attack_run_count', 0)} | "
        f"{attack_metrics.get('defender_detected_attack_run_count', 0)} | "
        f"{attack_metrics.get('defender_acted_attack_run_count', 0)} | "
        f"{attack_metrics.get('contained_before_safety_violation_run_count', 0)} | "
        f"{_fmt(attack_metrics.get('attacker_safety_success_rate'))} | "
        f"{_fmt(attack_metrics.get('time_to_detection_sec_avg'))} | "
        f"{_fmt(attack_metrics.get('time_to_first_defender_action_sec_avg'))} | "
        f"{_fmt(attack_metrics.get('attacker_action_count_avg'))} |"
    )


def _condition_mitre_line(condition_summary: dict[str, Any]) -> str:
    mitre = dict(condition_summary.get("mitre_phase_metrics") or {})
    distribution = ", ".join(
        f"{phase}:{count}"
        for phase, count in (mitre.get("highest_attacker_phase_distribution") or {}).items()
        if count
    ) or "none"
    return (
        f"| {condition_summary['condition_name']} | "
        f"{distribution} | "
        f"{_fmt(mitre.get('defender_detection_before_manipulation_rate'))} | "
        f"{_fmt(mitre.get('first_defender_action_before_manipulation_rate'))} | "
        f"{_fmt(mitre.get('manipulation_to_defender_detection_delay_sec_avg'))} | "
        f"{_fmt(mitre.get('manipulation_to_first_defender_action_delay_sec_avg'))} |"
    )


def _write_final_report(
    *,
    suite_summary: dict[str, Any],
    config: dict[str, Any],
    experiment_root: Path,
) -> None:
    expected_episode_count = (
        len(suite_summary.get("conditions", []))
        * len(config.get("scenario_selectors", []))
    )
    actual_completed = sum(
        condition.get("attack_metrics", {}).get("completed_attack_run_count", 0)
        for condition in suite_summary.get("conditions", [])
    )
    lines = [
        "# Option B MITRE-Informed Four-Pair Study",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Experiment root: `{portable_path(experiment_root)}`",
        f"- Seed: `{SEED}`",
        f"- Target realization: `{config.get('environment', {}).get('realization')}.{config.get('environment', {}).get('experiment')}`",
        f"- Revision: `{config.get('environment', {}).get('revision')}`",
        f"- XDC: `{config.get('environment', {}).get('xdc')}`",
        f"- Episodes completed: `{actual_completed}/72`",
        "",
        "## Condition-Level Results",
        "",
        "| Condition | Completed | Defender Detected | Defender Acted | Contained Pre-Violation | Safety Violation Rate | Avg TTD | Avg First Action | Avg Attacker Actions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for condition in suite_summary.get("conditions", []):
        lines.append(_condition_report_line(condition))
    lines.extend(
        [
            "",
            "## MITRE-Phase Outcomes",
            "",
            "| Condition | Highest Phase Distribution | Defender Detects Pre-Manip | Defender Acts Pre-Manip | Manip->Detection Avg | Manip->First Action Avg |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for condition in suite_summary.get("conditions", []):
        lines.append(_condition_mitre_line(condition))

    earlier_detection = [
        condition["condition_name"]
        for condition in suite_summary.get("conditions", [])
        if (condition.get("mitre_phase_metrics") or {}).get(
            "defender_detection_before_manipulation_rate"
        )
        not in (None, 0, 0.0)
    ]
    earlier_action = [
        condition["condition_name"]
        for condition in suite_summary.get("conditions", [])
        if (condition.get("mitre_phase_metrics") or {}).get(
            "first_defender_action_before_manipulation_rate"
        )
        not in (None, 0, 0.0)
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Earlier defender detection relative to manipulation was observed for: `{', '.join(earlier_detection) if earlier_detection else 'none'}`.",
            f"- Earlier defender action relative to manipulation was observed for: `{', '.join(earlier_action) if earlier_action else 'none'}`.",
            "- The adaptive rule-based blue team shifts detection toward discovery or qualification in more runs, while the fixed scripted blue team stays closer to manipulation-triggered responses.",
            "- The adaptive rule-based attacker still follows the same staged phase model but waits on observed process state before firing scenario actions, which tends to delay or suppress manipulation when earlier containment succeeds.",
            "",
            "## Comparison To The Older Non-Agentic Paper Baseline",
            "",
            "- This study is not an apples-to-apples replacement baseline.",
            "- The older non-agentic paper baseline assumed a more direct plant-facing attacker semantic, while this experiment adds staged entry context, discovery, process understanding, and control-path qualification before manipulation.",
            "- Because attack start now includes staged reconnaissance and qualification, attacker action counts increase and standard detection/action timings shift earlier in the episode structure relative to the old baseline.",
            "- The meaningful qualitative comparison is therefore phase placement: the new rule-based blue-team conditions stop more runs earlier in the chain, while the scripted blue-team conditions remain closer to post-manipulation containment.",
        ]
    )
    _write_text(COMPARISON_REPORT_PATH, "\n".join(lines) + "\n")


def main() -> int:
    PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_env_defaults()

    try:
        config = load_config(CONFIG_PATH)
        require_fields(config, ["experiment_name", "scenario_selectors"], ctx="experiment config")
        orchestrator = ExperimentOrchestrator()
        scenarios = list(
            orchestrator.catalog.resolve_selectors(list(config.get("scenario_selectors", [])))
        )
        if len(scenarios) != 18:
            raise RuntimeError(
                f"expected 18 attack scenarios, found {len(scenarios)} in {CONFIG_PATH}"
            )
        if not all(scenario.suite == "attack" for scenario in scenarios):
            raise RuntimeError("non-attack scenario slipped into the MITRE attack suite")

        condition_specs = orchestrator._enumerate_conditions(config)
        if len(condition_specs) != 4:
            raise RuntimeError("expected exactly four MITRE study conditions")

        reproducibility = orchestrator._configure_reproducibility(
            config=config,
            conditions=condition_specs,
        )
        orchestrator._current_reproducibility = reproducibility
        results_base_dir = Path(config["_path"]).parent
        resolved_results_root = resolve_repo_path(
            config.get("results_root"),
            base_dir=results_base_dir,
        )
        experiment_root = resolved_results_root / str(config.get("experiment_name"))

        completed_runs = _load_completed_runs()
        completed_runs["expected_conditions"] = [
            _condition_name(orchestrator, item) for item in condition_specs
        ]
        completed_runs["expected_scenarios"] = [scenario.catalog_id for scenario in scenarios]
        completed_runs["expected_run_keys"] = [
            _run_key(_condition_name(orchestrator, item), scenario)
            for item in condition_specs
            for scenario in scenarios
        ]
        _save_completed_runs(completed_runs)

        if _load_phase_status()["validated_optionb_environment"]["status"] != "completed":
            _set_phase("validated_optionb_environment", "in_progress")
            validation = _validate_optionb_environment(config=config, scenarios=scenarios)
            if validation.get("status") != "passed":
                raise RuntimeError(
                    f"Option B environment validation failed: {validation.get('error')}"
                )
            _set_phase(
                "validated_optionb_environment",
                "completed",
                validation_path=portable_path(ENVIRONMENT_VALIDATION_PATH),
                realization=validation.get("validated_target", {}).get("realization"),
                revision=validation.get("validated_target", {}).get("revision"),
                xdc=validation.get("validated_target", {}).get("xdc"),
            )

        _prepare_manifest(
            config=config,
            scenarios=scenarios,
            conditions=condition_specs,
            experiment_root=experiment_root,
            reproducibility=reproducibility,
        )
        _set_phase(
            "config_prepared",
            "completed",
            config_path=portable_path(CONFIG_PATH),
            results_root=portable_path(experiment_root),
        )

        if _load_phase_status()["validation_tests"]["status"] != "completed":
            _set_phase("validation_tests", "in_progress")
            _validate_runtime_and_tests()
            _set_phase(
                "validation_tests",
                "completed",
                validate_log=portable_path(VALIDATE_LOG_PATH),
                pytest_log=portable_path(PYTEST_LOG_PATH),
            )

        _set_phase("scenario_run", "in_progress", expected_run_count=72)
        run_summaries_by_condition: dict[str, list[dict[str, Any]]] = {}

        for condition_spec in condition_specs:
            condition = condition_spec["profiles"]
            condition_name = _condition_name(orchestrator, condition_spec)
            condition_root = experiment_root / condition_name
            known_runs = {
                item.get("run_id"): item
                for item in _load_existing_condition_summaries(
                    condition_root=condition_root,
                    scenarios=scenarios,
                )
            }
            _set_condition_status(condition_name, "in_progress")

            for scenario in scenarios:
                run_key = _run_key(condition_name, scenario)
                run_id = _run_id_for(scenario)
                run_dir = _run_dir_for(condition_root, scenario)

                if _quick_run_integrity(run_dir):
                    summary = _load_run_summary(run_dir)
                    known_runs[run_id] = summary
                    completed_runs["completed"][run_key] = _scenario_record(summary, run_dir)
                    completed_runs["failed"].pop(run_key, None)
                    _save_completed_runs(completed_runs)
                    continue

                bundle = orchestrator._run_single(
                    experiment_id=str(config.get("experiment_id", config.get("experiment_name"))),
                    condition_name=condition_name,
                    condition=condition,
                    scenario=scenario,
                    run_id=run_id,
                    run_dir=run_dir,
                    config=config,
                )
                summary = dict(bundle.get("summary") or {})
                if "metrics" not in summary:
                    summary["metrics"] = dict(bundle.get("metrics") or {})
                known_runs[run_id] = summary

                if summary.get("status") == "completed" and _quick_run_integrity(run_dir):
                    completed_runs["completed"][run_key] = _scenario_record(summary, run_dir)
                    completed_runs["failed"].pop(run_key, None)
                    _write_text(LAST_STEP_PATH, f"run:{run_key}\n")
                else:
                    error_text = "; ".join(
                        str(item.get("error"))
                        for item in summary.get("errors", [])
                        if isinstance(item, dict)
                    ) or "scenario run failed or did not produce required artifacts"
                    completed_runs["failed"][run_key] = {
                        "run_id": run_id,
                        "status": summary.get("status"),
                        "error": error_text,
                        "run_dir": portable_path(run_dir),
                        "failed_at_utc": utc_now(),
                    }
                    _save_completed_runs(completed_runs)
                    run_summaries_by_condition[condition_name] = [
                        known_runs[_run_id_for(item)]
                        for item in scenarios
                        if _run_id_for(item) in known_runs
                    ]
                    _aggregate_and_write(
                        orchestrator=orchestrator,
                        config=config,
                        experiment_root=experiment_root,
                        condition_specs=condition_specs,
                        run_summaries_by_condition=run_summaries_by_condition,
                        reproducibility=reproducibility,
                    )
                    raise RuntimeError(f"run `{run_key}` failed: {error_text}")

                _save_completed_runs(completed_runs)
                run_summaries_by_condition[condition_name] = [
                    known_runs[_run_id_for(item)]
                    for item in scenarios
                    if _run_id_for(item) in known_runs
                ]
                _aggregate_and_write(
                    orchestrator=orchestrator,
                    config=config,
                    experiment_root=experiment_root,
                    condition_specs=condition_specs,
                    run_summaries_by_condition=run_summaries_by_condition,
                    reproducibility=reproducibility,
                )

            ordered_runs = [
                known_runs[_run_id_for(scenario)]
                for scenario in scenarios
                if _run_id_for(scenario) in known_runs
            ]
            if len(ordered_runs) != len(scenarios):
                raise RuntimeError(
                    f"condition `{condition_name}` only resolved {len(ordered_runs)} of {len(scenarios)} runs"
                )
            run_summaries_by_condition[condition_name] = ordered_runs
            _set_condition_status(
                condition_name,
                "completed",
                completed_runs=len(ordered_runs),
            )
            _set_phase(
                "scenario_run",
                "in_progress",
                last_completed_condition=condition_name,
                completed_run_count=len(completed_runs.get("completed", {})),
            )

        suite_summary = _aggregate_and_write(
            orchestrator=orchestrator,
            config=config,
            experiment_root=experiment_root,
            condition_specs=condition_specs,
            run_summaries_by_condition=run_summaries_by_condition,
            reproducibility=reproducibility,
        )
        if len(completed_runs.get("completed", {})) != 72:
            raise RuntimeError(
                f"expected 72 completed runs, found {len(completed_runs.get('completed', {}))}"
            )

        _set_phase(
            "scenario_run",
            "completed",
            completed_run_count=72,
        )
        _set_phase(
            "aggregate_metrics",
            "completed",
            experiment_summary=portable_path(experiment_root / "experiment_summary.json"),
            mitre_phase_metrics=portable_path(experiment_root / "mitre_phase_metrics.json"),
        )

        _write_final_report(
            suite_summary=suite_summary,
            config=config,
            experiment_root=experiment_root,
        )
        _set_phase(
            "final_report",
            "completed",
            report_path=portable_path(COMPARISON_REPORT_PATH),
        )

        phase_status = _load_phase_status()
        phase_status["completion_state"] = {
            "completed_at_utc": utc_now(),
            "experiment_root": portable_path(experiment_root),
            "report_path": portable_path(COMPARISON_REPORT_PATH),
            "completed_run_count": 72,
            "suite_status": suite_summary.get("status"),
        }
        _save_phase_status(phase_status)
        _write_resume_notes()
        return 0
    except Exception as exc:
        phase_status = _load_phase_status()
        phase_status["completion_state"] = {
            "completed_at_utc": None,
            "failed_at_utc": utc_now(),
            "status": "blocked",
            "error": str(exc),
        }
        _save_phase_status(phase_status)
        _write_resume_notes()
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
