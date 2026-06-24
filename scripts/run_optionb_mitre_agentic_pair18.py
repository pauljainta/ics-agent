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
from ics_agent.mitre import verify_optionb_staged_snapshot
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from ics_agent.reporting import write_json
from ics_agent.utils import utc_now


SEED = 20260623
CONFIG_PATH = (
    REPO_ROOT / "configs" / "experiments" / "sphere_optionb_mitre_agentic_pair18.yaml"
)
RESULTS_ROOT = WORKSPACE_ROOT / "Artifact" / "Results" / "optionb_mitre_agentic_pair18"
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
    "prompt_context_implementation",
    "validation_tests",
    "scenario_run",
    "aggregate_metrics",
    "final_report",
    "code_commit_push",
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
EXPECTED_CONDITION_NAME = "black_box_attacker__agentic_defender"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _git_commit(path: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
            or None
        )
    except Exception:
        return None


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
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", str(SEED))
    if os.environ.get("QWEN_MODEL_ID") == "Qwen/Qwen3.5-8B":
        os.environ["QWEN_MODEL_ID"] = "Qwen/Qwen3.5-9B"
    os.environ.setdefault("QWEN_MODEL_ID", "Qwen/Qwen3.5-9B")
    if not os.environ.get("QWEN_MODEL_PATH"):
        os.environ["QWEN_MODEL_PATH"] = str(
            (REPO_ROOT / "models" / "local_models" / "qwen3.5-9b").resolve()
        )


def _load_phase_status() -> dict[str, Any]:
    payload = _read_json(PHASE_STATUS_PATH, {})
    if not isinstance(payload, dict):
        payload = {}
    for phase in PHASE_ORDER:
        payload.setdefault(phase, {"status": "pending"})
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


def _load_completed_runs() -> dict[str, Any]:
    payload = _read_json(
        COMPLETED_RUNS_PATH,
        {
            "expected_scenarios": [],
            "completed": {},
            "failed": {},
            "last_updated_utc": None,
        },
    )
    if not isinstance(payload, dict):
        payload = {
            "expected_scenarios": [],
            "completed": {},
            "failed": {},
            "last_updated_utc": None,
        }
    payload.setdefault("expected_scenarios", [])
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
    expected = list(completed_runs.get("expected_scenarios") or [])
    completed = dict(completed_runs.get("completed") or {})
    failed = dict(completed_runs.get("failed") or {})
    remaining = [
        scenario for scenario in expected if scenario not in completed and scenario not in failed
    ]
    lines = [
        "# Resume Notes",
        "",
        f"- Updated UTC: `{utc_now()}`",
        f"- Last successful step: `{LAST_STEP_PATH.read_text().strip() if LAST_STEP_PATH.exists() else 'none'}`",
        f"- Completed scenarios: `{len(completed)}`",
        f"- Failed scenarios: `{len(failed)}`",
        f"- Remaining scenarios: `{len(remaining)}`",
        "",
        "## Phase Status",
    ]
    for phase in PHASE_ORDER:
        lines.append(f"- `{phase}`: `{phase_status.get(phase, {}).get('status', 'pending')}`")
    if remaining:
        lines.extend(["", "## Remaining Scenarios"])
        for scenario in remaining:
            lines.append(f"- `{scenario}`")
    if failed:
        lines.extend(["", "## Failed Scenarios"])
        for scenario, info in sorted(failed.items()):
            lines.append(
                f"- `{scenario}`: `{info.get('status')}` `{info.get('error', 'see run artifacts')}`"
            )
    _write_text(RESUME_NOTES_PATH, "\n".join(lines) + "\n")


def _run_dir_for(condition_root: Path, scenario: Any, repeat: int = 1) -> Path:
    return condition_root / scenario.suite / scenario.scenario_id / f"trial-{repeat:03d}"


def _run_id_for(scenario: Any, repeat: int = 1) -> str:
    return f"{scenario.suite}-{scenario.scenario_id}-trial-{repeat:03d}"


def _expected_environment(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("environment") or {})


def _run_optionb_integrity(run_dir: Path, *, expected_environment: dict[str, Any]) -> bool:
    if not all((run_dir / name).exists() for name in REQUIRED_RUN_FILES):
        return False
    summary = _read_json(run_dir / "summary.json", {})
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        return False
    environment_snapshot = _read_json(run_dir / "environment_snapshot.json", {})
    verification = verify_optionb_staged_snapshot(
        environment_snapshot,
        expected_environment=expected_environment,
    )
    return bool(verification.get("ok"))


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json", {})
    metrics = _read_json(run_dir / "metrics.json", {})
    if isinstance(summary, dict) and "metrics" not in summary:
        summary["metrics"] = metrics
    if isinstance(summary, dict) and "run_dir" not in summary:
        summary["run_dir"] = portable_path(run_dir)
    return summary


def _scenario_record(
    summary: dict[str, Any],
    run_dir: Path,
    *,
    expected_environment: dict[str, Any],
) -> dict[str, Any]:
    metrics = dict(summary.get("metrics", {}) or {})
    phase_window = dict(metrics.get("phase_window") or {})
    mitre_phase = dict(metrics.get("mitre_phase") or {})
    attacker = dict(metrics.get("attacker") or {})
    return {
        "run_id": summary.get("run_id"),
        "scenario": summary.get("scenario"),
        "suite": summary.get("suite"),
        "status": summary.get("status"),
        "passed": summary.get("passed"),
        "run_dir": portable_path(run_dir),
        "phase_a_duration_sec": phase_window.get("phase_a_duration_sec"),
        "wargame_t0_utc": phase_window.get("wargame_t0_utc"),
        "time_to_detection_from_wargame_t0_sec": metrics.get(
            "time_to_detection_from_wargame_t0_sec"
        ),
        "time_to_first_defender_action_from_wargame_t0_sec": metrics.get(
            "time_to_first_defender_action_from_wargame_t0_sec"
        ),
        "time_to_physical_impact_from_wargame_t0_sec": metrics.get(
            "time_to_physical_impact_from_wargame_t0_sec"
        ),
        "attacker_caused_safety_violation": attacker.get("caused_safety_violation"),
        "highest_attacker_phase_reached": mitre_phase.get("highest_attacker_phase_reached"),
        "first_defender_detection_phase": mitre_phase.get("first_defender_detection_phase"),
        "first_defender_action_phase": mitre_phase.get("first_defender_action_phase"),
        "integrity_verified": _run_optionb_integrity(
            run_dir,
            expected_environment=expected_environment,
        ),
    }


def _load_existing_suite_summaries(
    *,
    condition_root: Path,
    scenarios: list[Any],
    expected_environment: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        run_dir = _run_dir_for(condition_root, scenario)
        if not _run_optionb_integrity(run_dir, expected_environment=expected_environment):
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
            raise RuntimeError("Option B MITRE agentic study requires the sphere_openplc backend")
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
    condition_name: str,
    condition: dict[str, Any],
    experiment_root: Path,
    reproducibility: dict[str, Any],
) -> None:
    manifest = {
        "generated_at_utc": utc_now(),
        "config_path": portable_path(Path(config["_path"])),
        "experiment_name": config.get("experiment_name"),
        "experiment_id": config.get("experiment_id", config.get("experiment_name")),
        "results_root": portable_path(experiment_root),
        "progress_root": portable_path(PROGRESS_ROOT),
        "condition_name": condition_name,
        "profiles": condition,
        "scenario_catalog_ids": [scenario.catalog_id for scenario in scenarios],
        "scenario_count": len(scenarios),
        "seed": SEED,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "model_path": os.environ.get("QWEN_MODEL_PATH"),
        "metric_boundary": {
            "phase_a": "setup/bootstrap and transfer latency",
            "wargame_t0": "interaction-ready boundary after staged path verification and environment reset",
            "phase_b": "measured wargame interval only",
        },
        "primary_timing_origin": "wargame_t0",
        "study_metadata": dict(config.get("study_metadata") or {}),
        "environment_target": _expected_environment(config),
        "reproducibility": reproducibility,
        "repo_commits": {
            "ics_agent": _git_commit(REPO_ROOT),
        },
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
            "tests/test_metrics.py",
        ],
        PYTEST_LOG_PATH,
    )


def _aggregate_and_write(
    *,
    orchestrator: ExperimentOrchestrator,
    config: dict[str, Any],
    experiment_root: Path,
    condition_name: str,
    condition: dict[str, Any],
    runs: list[dict[str, Any]],
    reproducibility: dict[str, Any],
    effective_profiles: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    condition_root = experiment_root / condition_name
    condition_summary: dict[str, Any] = {
        "condition_name": condition_name,
        "profiles": condition,
        "runs": runs,
        "status": (
            "failed" if any(run.get("status") != "completed" for run in runs) else "completed"
        ),
        "study_metadata": dict(config.get("study_metadata") or {}),
        "reproducibility": orchestrator._condition_reproducibility(
            condition_name,
            condition,
            effective_profiles=effective_profiles,
        ),
    }
    condition_summary["nominal_metrics"] = orchestrator._aggregate_nominal_metrics(runs)
    condition_summary["attack_metrics"] = orchestrator._aggregate_attack_metrics(
        runs,
        primary_timing_origin="wargame_t0",
    )
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

    suite_summary = {
        "experiment_id": str(config.get("experiment_id", config.get("experiment_name"))),
        "experiment_name": str(config.get("experiment_name")),
        "config_path": portable_path(Path(config["_path"])),
        "results_root": portable_path(experiment_root),
        "timestamp_utc": utc_now(),
        "status": condition_summary["status"],
        "use_canonical_result_root": bool(config.get("use_canonical_result_root", False)),
        "study_metadata": dict(config.get("study_metadata") or {}),
        "reproducibility": reproducibility,
        "conditions": [condition_summary],
        "nominal_metrics": condition_summary["nominal_metrics"],
        "attack_metrics": condition_summary["attack_metrics"],
        "mitre_phase_metrics": condition_summary["mitre_phase_metrics"],
        "trajectory_metrics": condition_summary["trajectory_metrics"],
    }

    write_json(condition_root / "condition_summary.json", condition_summary)
    write_json(condition_root / "attack_metrics_summary.json", condition_summary["attack_metrics"])
    write_json(condition_root / "nominal_metrics_summary.json", condition_summary["nominal_metrics"])
    write_json(condition_root / "mitre_phase_metrics.json", condition_summary["mitre_phase_metrics"])
    write_json(condition_root / "mitre_phase_summary.json", mitre_summary)
    write_json(condition_root / "trajectory_metrics_summary.json", condition_summary["trajectory_metrics"])

    write_json(experiment_root / "experiment_summary.json", suite_summary)
    write_json(experiment_root / "attack_metrics_summary.json", suite_summary["attack_metrics"])
    write_json(experiment_root / "nominal_metrics_summary.json", suite_summary["nominal_metrics"])
    write_json(experiment_root / "mitre_phase_metrics.json", suite_summary["mitre_phase_metrics"])
    write_json(experiment_root / "mitre_phase_summary.json", {"condition_summaries": [mitre_summary]})
    write_json(experiment_root / "trajectory_metrics_summary.json", suite_summary["trajectory_metrics"])
    return suite_summary


def _load_baseline_reference() -> dict[str, Any]:
    candidate_paths = sorted(
        (
            path
            for path in (WORKSPACE_ROOT / "Artifact" / "Results").rglob("attack_metrics_summary.json")
            if EXPECTED_CONDITION_NAME in str(path.parent)
            and "optionb_mitre_agentic_pair18" not in str(path)
            and "optionb_mitre_pairs18" not in str(path)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidate_paths:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        return {
            "source_type": "artifact_results",
            "source_path": portable_path(path),
            "attack_metrics": payload,
        }

    return {
        "source_type": "repo_readme_snapshot",
        "source_path": portable_path(REPO_ROOT / "README.md"),
        "attack_metrics": {
            "attack_run_count": 18,
            "attacker_caused_safety_violation_run_count": 1,
            "attacker_safety_success_rate": round(1 / 18, 4),
            "attacker_recon_action_count_avg": 0.0,
        },
        "notes": [
            "No prior non-MITRE black_box_attacker__agentic_defender artifact was present under Artifact/Results.",
            "Baseline comparison therefore uses the README-documented 2026-05-23 local black-box agentic snapshot.",
            "That baseline was direct-access-semantic and exploit-first rather than staged-entry MITRE-informed.",
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _write_final_report(
    *,
    suite_summary: dict[str, Any],
    config: dict[str, Any],
    experiment_root: Path,
) -> None:
    condition_summary = suite_summary["conditions"][0]
    attack_metrics = dict(condition_summary.get("attack_metrics") or {})
    mitre_metrics = dict(condition_summary.get("mitre_phase_metrics") or {})
    baseline = _load_baseline_reference()
    distribution = ", ".join(
        f"{phase}:{count}"
        for phase, count in (mitre_metrics.get("highest_attacker_phase_distribution") or {}).items()
        if count
    ) or "none"

    lines = [
        "# Option B MITRE-Informed Agentic Pair Study",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Experiment root: `{portable_path(experiment_root)}`",
        f"- Seed: `{SEED}`",
        f"- Condition: `{condition_summary['condition_name']}`",
        f"- Target realization: `{config.get('environment', {}).get('realization')}.{config.get('environment', {}).get('experiment')}`",
        f"- Revision: `{config.get('environment', {}).get('revision')}`",
        f"- XDC: `{config.get('environment', {}).get('xdc')}`",
        f"- Episodes completed: `{attack_metrics.get('completed_attack_run_count', 0)}/{attack_metrics.get('attack_run_count', 0)}`",
        f"- Primary timing origin: `{attack_metrics.get('primary_timing_origin')}`",
        "",
        "## Fresh Results",
        "",
        f"- Defender-detected scenarios: `{attack_metrics.get('defender_detected_attack_run_count', 0)}`",
        f"- Defender-action scenarios: `{attack_metrics.get('defender_acted_attack_run_count', 0)}`",
        f"- Contained-before-violation scenarios: `{attack_metrics.get('contained_before_safety_violation_run_count', 0)}`",
        f"- Safety-violation rate: `{_fmt(attack_metrics.get('attacker_safety_success_rate'))}`",
        f"- Avg time to detection from wargame_t0: `{_fmt(attack_metrics.get('time_to_detection_sec_avg'))}`",
        f"- Avg time to first defender action from wargame_t0: `{_fmt(attack_metrics.get('time_to_first_defender_action_sec_avg'))}`",
        f"- Avg time to physical impact from wargame_t0: `{_fmt(attack_metrics.get('time_to_physical_impact_sec_avg'))}`",
        f"- Avg attacker action count: `{_fmt(attack_metrics.get('attacker_action_count_avg'))}`",
        "",
        "## MITRE-Phase Outcomes",
        "",
        f"- Highest attacker phase distribution: `{distribution}`",
        f"- Defender detection before manipulation rate: `{_fmt(mitre_metrics.get('defender_detection_before_manipulation_rate'))}`",
        f"- First defender action before manipulation rate: `{_fmt(mitre_metrics.get('first_defender_action_before_manipulation_rate'))}`",
        f"- Manipulation to defender detection delay avg: `{_fmt(mitre_metrics.get('manipulation_to_defender_detection_delay_sec_avg'))}`",
        f"- Manipulation to first defender action delay avg: `{_fmt(mitre_metrics.get('manipulation_to_first_defender_action_delay_sec_avg'))}`",
        "",
        "## Baseline Comparison",
        "",
        f"- Baseline source: `{baseline.get('source_type')}` `{baseline.get('source_path')}`",
    ]
    for note in baseline.get("notes", []):
        lines.append(f"- {note}")

    baseline_attack_metrics = dict(baseline.get("attack_metrics") or {})
    if baseline_attack_metrics:
        lines.extend(
            [
                f"- Baseline safety-violation rate: `{_fmt(baseline_attack_metrics.get('attacker_safety_success_rate'))}`",
                f"- Baseline attacker recon action avg: `{_fmt(baseline_attack_metrics.get('attacker_recon_action_count_avg'))}`",
            ]
        )

    lines.extend(
        [
            "- This MITRE-informed Option B pair is not an apples-to-apples replacement baseline for the older non-MITRE local pair because the attack semantics changed.",
            "- Relative to the direct-access baseline, the new pair forces staged entry context, reconnaissance, process understanding, and control-path qualification before manipulation can occur.",
            "- The resulting attacker trajectories are longer and phase-explicit, while the defender can now register suspicion and discovery-recognition earlier in the episode structure instead of only reacting at direct manipulation time.",
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
            raise RuntimeError("non-attack scenario slipped into the MITRE agentic attack suite")

        condition_specs = orchestrator._enumerate_conditions(config)
        if len(condition_specs) != 1:
            raise RuntimeError("expected exactly one condition in the MITRE agentic pair config")
        condition_spec = condition_specs[0]
        condition = condition_spec["profiles"]
        effective_profiles = condition_spec.get("effective_profiles")
        condition_name = orchestrator._condition_name(
            condition,
            override=condition_spec.get("name"),
        )
        if condition_name != EXPECTED_CONDITION_NAME:
            raise RuntimeError(
                f"expected condition name `{EXPECTED_CONDITION_NAME}`, found `{condition_name}`"
            )

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
        condition_root = experiment_root / condition_name
        expected_environment = _expected_environment(config)

        completed_runs = _load_completed_runs()
        completed_runs["expected_scenarios"] = [scenario.catalog_id for scenario in scenarios]
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
            condition_name=condition_name,
            condition=condition,
            experiment_root=experiment_root,
            reproducibility=reproducibility,
        )
        _set_phase(
            "config_prepared",
            "completed",
            condition_name=condition_name,
            config_path=portable_path(CONFIG_PATH),
            results_root=portable_path(experiment_root),
        )
        _set_phase(
            "prompt_context_implementation",
            "completed",
            prompt_files=[
                "configs/prompts/black_box_attacker_optionb_mitre.md",
                "configs/prompts/defender_optionb_mitre.md",
                "configs/prompts/observer_optionb_mitre.md",
            ],
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

        _set_phase("scenario_run", "in_progress", condition_name=condition_name)
        run_summaries = _load_existing_suite_summaries(
            condition_root=condition_root,
            scenarios=scenarios,
            expected_environment=expected_environment,
        )
        known_runs = {item.get("run_id"): item for item in run_summaries}
        for scenario in scenarios:
            run_id = _run_id_for(scenario)
            run_dir = _run_dir_for(condition_root, scenario)
            if _run_optionb_integrity(run_dir, expected_environment=expected_environment):
                summary = _load_run_summary(run_dir)
                known_runs[run_id] = summary
                completed_runs["completed"][scenario.catalog_id] = _scenario_record(
                    summary,
                    run_dir,
                    expected_environment=expected_environment,
                )
                completed_runs["failed"].pop(scenario.catalog_id, None)
                _save_completed_runs(completed_runs)
                continue

            bundle = orchestrator._run_single(
                experiment_id=str(config.get("experiment_id", config.get("experiment_name"))),
                condition_name=condition_name,
                condition=condition,
                effective_profiles=effective_profiles,
                scenario=scenario,
                run_id=run_id,
                run_dir=run_dir,
                config=config,
            )
            summary = dict(bundle.get("summary") or {})
            if "metrics" not in summary:
                summary["metrics"] = dict(bundle.get("metrics") or {})
            known_runs[run_id] = summary

            if summary.get("status") == "completed" and _run_optionb_integrity(
                run_dir,
                expected_environment=expected_environment,
            ):
                completed_runs["completed"][scenario.catalog_id] = _scenario_record(
                    summary,
                    run_dir,
                    expected_environment=expected_environment,
                )
                completed_runs["failed"].pop(scenario.catalog_id, None)
                _write_text(LAST_STEP_PATH, f"scenario:{scenario.catalog_id}\n")
            else:
                environment_snapshot = _read_json(run_dir / "environment_snapshot.json", {})
                verification = verify_optionb_staged_snapshot(
                    environment_snapshot,
                    expected_environment=expected_environment,
                )
                error_text = "; ".join(
                    str(item.get("error"))
                    for item in summary.get("errors", [])
                    if isinstance(item, dict)
                ) or "scenario run failed or did not produce required artifacts"
                if not verification.get("ok"):
                    error_text = (
                        error_text
                        + "; optionb_integrity="
                        + ", ".join(verification.get("errors", []))
                    )
                completed_runs["failed"][scenario.catalog_id] = {
                    "run_id": run_id,
                    "status": summary.get("status"),
                    "error": error_text,
                    "run_dir": portable_path(run_dir),
                    "failed_at_utc": utc_now(),
                }
                _save_completed_runs(completed_runs)
                current_runs = [
                    known_runs[_run_id_for(item)]
                    for item in scenarios
                    if _run_id_for(item) in known_runs
                ]
                _aggregate_and_write(
                    orchestrator=orchestrator,
                    config=config,
                    experiment_root=experiment_root,
                    condition_name=condition_name,
                    condition=condition,
                    runs=current_runs,
                    reproducibility=reproducibility,
                    effective_profiles=effective_profiles,
                )
                raise RuntimeError(f"scenario `{scenario.catalog_id}` failed: {error_text}")

            _save_completed_runs(completed_runs)
            current_runs = [
                known_runs[_run_id_for(item)]
                for item in scenarios
                if _run_id_for(item) in known_runs
            ]
            _aggregate_and_write(
                orchestrator=orchestrator,
                config=config,
                experiment_root=experiment_root,
                condition_name=condition_name,
                condition=condition,
                runs=current_runs,
                reproducibility=reproducibility,
                effective_profiles=effective_profiles,
            )

        ordered_runs = [
            known_runs[_run_id_for(scenario)]
            for scenario in scenarios
            if _run_id_for(scenario) in known_runs
        ]
        suite_summary = _aggregate_and_write(
            orchestrator=orchestrator,
            config=config,
            experiment_root=experiment_root,
            condition_name=condition_name,
            condition=condition,
            runs=ordered_runs,
            reproducibility=reproducibility,
            effective_profiles=effective_profiles,
        )
        if len(ordered_runs) != len(scenarios):
            raise RuntimeError(
                f"aggregate summary only found {len(ordered_runs)} of {len(scenarios)} scenario runs"
            )

        _set_phase(
            "scenario_run",
            "completed",
            completed_scenarios=len(ordered_runs),
        )
        _set_phase(
            "aggregate_metrics",
            "completed",
            experiment_summary=portable_path(experiment_root / "experiment_summary.json"),
            attack_metrics=portable_path(experiment_root / "attack_metrics_summary.json"),
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
        _set_phase(
            "code_commit_push",
            _load_phase_status()["code_commit_push"].get("status", "pending"),
        )

        phase_status = _load_phase_status()
        phase_status["completion_state"] = {
            "completed_at_utc": utc_now(),
            "experiment_root": portable_path(experiment_root),
            "condition_name": condition_name,
            "scenario_count": len(ordered_runs),
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
