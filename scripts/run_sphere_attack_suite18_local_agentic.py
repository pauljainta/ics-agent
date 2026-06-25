#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from ics_agent.agents.agentic import CachedQwenDecisionEngine
from ics_agent.config import load_config, require_fields, resolve_repo_path
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from ics_agent.reporting import write_json
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.utils import utc_now


SEED = 20260619
EXPERIMENT_REF = "openplcwater.lgarcia"
REALIZATION_REF = "wargamestable.openplcwater.lgarcia"
REVISION_REF = "7054831a885ac69f8cc5ffb2cccc432e892dadb6"
CONFIG_PATH = REPO_ROOT / "configs/experiments/sphere_attack_suite18_local_agentic.yaml"
RESULTS_ROOT = WORKSPACE_ROOT / "Artifact" / "Results" / "New"
PROGRESS_ROOT = RESULTS_ROOT / "_progress"
PORTAL_INSPECTION_PATH = PROGRESS_ROOT / "portal_inspection.json"
PHASE_STATUS_PATH = PROGRESS_ROOT / "phase_status.json"
RUN_MANIFEST_PATH = PROGRESS_ROOT / "run_manifest.json"
COMPLETED_RUNS_PATH = PROGRESS_ROOT / "completed_runs.json"
RESUME_NOTES_PATH = PROGRESS_ROOT / "resume_notes.md"
LAST_STEP_PATH = PROGRESS_ROOT / "last_successful_step.txt"
VALIDATE_LOG_PATH = PROGRESS_ROOT / "validate_runtime.log"
PYTEST_LOG_PATH = PROGRESS_ROOT / "pytest_targeted.log"
CUDA_LOG_PATH = PROGRESS_ROOT / "cuda_qwen_smoke.json"

PHASE_ORDER = [
    "portal_inspection",
    "rollback_decision",
    "config_prepared",
    "validation_tests",
    "scenario_run",
    "aggregate_metrics",
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


def _run_checked(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
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
        item = phase_status.get(phase, {})
        lines.append(
            f"- `{phase}`: `{item.get('status', 'pending')}`"
        )
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
    return (
        condition_root
        / scenario.suite
        / scenario.scenario_id
        / f"trial-{repeat:03d}"
    )


def _run_id_for(scenario: Any, repeat: int = 1) -> str:
    return f"{scenario.suite}-{scenario.scenario_id}-trial-{repeat:03d}"


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
    phase_window = dict(summary.get("metrics", {}).get("phase_window") or {})
    attacker = dict(summary.get("metrics", {}).get("attacker") or {})
    return {
        "run_id": summary.get("run_id"),
        "scenario": summary.get("scenario"),
        "suite": summary.get("suite"),
        "status": summary.get("status"),
        "passed": summary.get("passed"),
        "run_dir": portable_path(run_dir),
        "phase_a_duration_sec": phase_window.get("phase_a_duration_sec"),
        "wargame_t0_utc": phase_window.get("wargame_t0_utc"),
        "time_to_detection_sec": summary.get("metrics", {}).get("time_to_detection_sec"),
        "time_to_first_defender_action_sec": summary.get("metrics", {})
        .get("defender_intervention", {})
        .get("time_to_first_action_sec"),
        "attacker_caused_safety_violation": attacker.get("caused_safety_violation"),
        "time_to_physical_impact_sec": attacker.get("time_to_physical_impact_sec"),
        "integrity_verified": _quick_run_integrity(run_dir),
    }


def _aggregate_and_write(
    *,
    orchestrator: ExperimentOrchestrator,
    config: dict[str, Any],
    experiment_root: Path,
    condition_name: str,
    condition: dict[str, str],
    run_summaries: list[dict[str, Any]],
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    condition_root = experiment_root / condition_name
    condition_summary: dict[str, Any] = {
        "condition_name": condition_name,
        "profiles": condition,
        "runs": run_summaries,
        "status": (
            "failed"
            if any(run.get("status") != "completed" for run in run_summaries)
            else "completed"
        ),
        "reproducibility": orchestrator._condition_reproducibility(
            condition_name,
            condition,
        ),
    }
    condition_summary["nominal_metrics"] = orchestrator._aggregate_nominal_metrics(
        run_summaries
    )
    condition_summary["attack_metrics"] = orchestrator._aggregate_attack_metrics(
        run_summaries
    )
    condition_summary["trajectory_metrics"] = orchestrator._aggregate_trajectory_metrics(
        run_summaries,
        condition=condition,
    )

    suite_summary = {
        "experiment_id": str(
            config.get("experiment_id", config.get("experiment_name"))
        ),
        "experiment_name": str(config.get("experiment_name")),
        "config_path": portable_path(Path(config["_path"])),
        "results_root": portable_path(experiment_root),
        "timestamp_utc": utc_now(),
        "status": condition_summary["status"],
        "use_canonical_result_root": bool(config.get("use_canonical_result_root", False)),
        "reproducibility": reproducibility,
        "conditions": [condition_summary],
    }

    write_json(condition_root / "condition_summary.json", condition_summary)
    write_json(
        condition_root / "attack_metrics_summary.json",
        condition_summary["attack_metrics"],
    )
    write_json(
        condition_root / "nominal_metrics_summary.json",
        condition_summary["nominal_metrics"],
    )
    write_json(
        condition_root / "trajectory_metrics_summary.json",
        condition_summary["trajectory_metrics"],
    )
    write_json(experiment_root / "experiment_summary.json", suite_summary)
    write_json(experiment_root / "attack_metrics_summary.json", condition_summary["attack_metrics"])
    write_json(experiment_root / "nominal_metrics_summary.json", condition_summary["nominal_metrics"])
    write_json(
        experiment_root / "trajectory_metrics_summary.json",
        condition_summary["trajectory_metrics"],
    )
    return suite_summary


def _load_existing_suite_summaries(
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


def _inspect_portal() -> dict[str, Any]:
    merge_user = _run_checked(["mrg", "whoami"]).strip()
    experiment = json.loads(
        _run_checked(["mrg", "show", "experiment", EXPERIMENT_REF, "-j"])
    )
    realization = json.loads(
        _run_checked(["mrg", "show", "realization", REALIZATION_REF, "-j"])
    )
    materialization = json.loads(
        _run_checked(["mrg", "show", "materialization", REALIZATION_REF, "-j"])
    )
    realization_listing = json.loads(
        _run_checked(["mrg", "list", "realizations", "--filter", "all", "-j"])
    )

    filtered_realizations = []
    for item in realization_listing.get("results", []):
        payload = item.get("realization", {})
        if payload.get("eid") != "openplcwater" or payload.get("pid") != "lgarcia":
            continue
        node_ids = sorted(
            node_key.split("/", 1)[-1] for node_key in (payload.get("nodes") or {}).keys()
        )
        filtered_realizations.append(
            {
                "id": payload.get("id"),
                "xhash": payload.get("xhash"),
                "nodes": node_ids,
                "created": payload.get("created"),
                "expires": payload.get("expires"),
            }
        )

    revision_scan = []
    model_map = dict(experiment.get("experiment", {}).get("models") or {})
    for revision, meta in sorted(model_map.items()):
        revision_payload = json.loads(
            _run_checked(
                [
                    "mrg",
                    "show",
                    "revision",
                    EXPERIMENT_REF,
                    revision,
                    "--with-model",
                    "-j",
                ]
            )
        )
        serialized = json.dumps(revision_payload)
        revision_scan.append(
            {
                "revision": revision,
                "compile_time": meta.get("compileTime"),
                "realizations": list(meta.get("realizations") or []),
                "contains_attacker_foothold": "attacker_foothold" in serialized,
                "contains_defender_node": bool(
                    re.search(r'"id"\s*:\s*"defender"', serialized)
                ),
                "contains_viewer_node": bool(
                    re.search(r'"id"\s*:\s*"viewer"', serialized)
                ),
                "contains_controller_node": bool(
                    re.search(r'"id"\s*:\s*"controller"', serialized)
                ),
                "contains_simulator_node": bool(
                    re.search(r'"id"\s*:\s*"simulator"', serialized)
                ),
            }
        )

    four_vm_realizations = [
        item
        for item in filtered_realizations
        if {"attacker_foothold", "defender"}.intersection(item.get("nodes", []))
    ]
    four_vm_revisions = [
        item
        for item in revision_scan
        if item["contains_attacker_foothold"] or item["contains_defender_node"]
    ]

    realization_payload = realization.get("result", {}).get("realization", {})
    realized_nodes = sorted(
        node_key.split("/", 1)[-1]
        for node_key in (realization_payload.get("nodes") or {}).keys()
    )

    return {
        "checked_at_utc": utc_now(),
        "merge_user": merge_user,
        "experiment": {
            "ref": EXPERIMENT_REF,
            "description": experiment.get("experiment", {}).get("description"),
            "version": experiment.get("experiment", {}).get("ver"),
            "repository": experiment.get("experiment", {}).get("repository"),
        },
        "realization": {
            "ref": REALIZATION_REF,
            "revision": realization_payload.get("xhash"),
            "complete": realization_payload.get("complete"),
            "created": realization_payload.get("created"),
            "expires": realization_payload.get("expires"),
            "nodes": realized_nodes,
        },
        "materialization_present": bool(materialization.get("result", {}).get("materialization")),
        "all_realizations_for_experiment": filtered_realizations,
        "revision_scan": revision_scan,
        "four_vm_realization_candidates": four_vm_realizations,
        "four_vm_revision_candidates": four_vm_revisions,
        "usable_four_vm_deployment": bool(four_vm_realizations),
        "decision": {
            "use_remote_attacker_defender_vms": False,
            "reason": (
                "No usable 4-VM attacker/defender deployment is present in Launch. "
                "The only realization for openplcwater.lgarcia is wargamestable on "
                "revision 7054831a885ac69f8cc5ffb2cccc432e892dadb6 with only controller "
                "and simulator nodes."
            ),
            "selected_topology": {
                "backend": "sphere_openplc",
                "experiment": "openplcwater.lgarcia",
                "realization": "wargamestable",
                "revision": REVISION_REF,
                "remote_roles": {},
                "attacker_residency": "local_gpu_server",
                "defender_residency": "local_gpu_server",
            },
        },
    }


def _prepare_manifest(
    *,
    config: dict[str, Any],
    scenarios: list[Any],
    condition_name: str,
    condition: dict[str, str],
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
        "experiment_ref": EXPERIMENT_REF,
        "realization_ref": REALIZATION_REF,
        "revision_ref": REVISION_REF,
        "metric_boundary": {
            "phase_a": "setup/bootstrap",
            "wargame_t0": "first moment live scenario is ready for agent interaction",
            "phase_b": "measured wargaming interval only",
        },
        "reproducibility": reproducibility,
        "repo_commits": {
            "ics_agent": _git_commit(REPO_ROOT),
        },
    }
    write_json(RUN_MANIFEST_PATH, manifest)


def _validate_runtime_and_model() -> None:
    venv_python = str((REPO_ROOT / ".venv" / "bin" / "python").resolve())
    _run_logged([venv_python, "-m", "ics_agent.cli", "validate"], VALIDATE_LOG_PATH)

    orchestrator = ExperimentOrchestrator()
    profile = dict(orchestrator.agent_profiles["gray_box_attacker"])
    engine = CachedQwenDecisionEngine(profile, role="attacker")
    llm = engine._llm()
    payload = {
        "validated_at_utc": utc_now(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "model_path": os.environ.get("QWEN_MODEL_PATH"),
        "backend": engine.backend,
        "temperature": engine.temperature,
        "max_new_tokens": engine.max_new_tokens,
        "llm_type": type(llm).__name__,
    }
    try:
        import torch  # type: ignore

        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        payload["torch_import_error"] = str(exc)
        write_json(CUDA_LOG_PATH, payload)
        raise
    write_json(CUDA_LOG_PATH, payload)

    _run_logged(
        [
            venv_python,
            "-m",
            "pytest",
            "tests/test_agentic.py",
            "tests/test_environment.py",
            "tests/test_orchestrator.py",
        ],
        PYTEST_LOG_PATH,
    )


def main() -> int:
    PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_env_defaults()

    try:
        if _load_phase_status()["portal_inspection"]["status"] != "completed":
            _set_phase("portal_inspection", "in_progress")
            inspection = _inspect_portal()
            write_json(PORTAL_INSPECTION_PATH, inspection)
            _set_phase(
                "portal_inspection",
                "completed",
                usable_four_vm_deployment=inspection["usable_four_vm_deployment"],
            )
        else:
            inspection = _read_json(PORTAL_INSPECTION_PATH, {})

        _set_phase(
            "rollback_decision",
            "completed",
            use_remote_attacker_defender_vms=False,
            selected_realization=REALIZATION_REF,
            selected_revision=REVISION_REF,
            repo_rollback_required=False,
        )

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
            raise RuntimeError("non-attack scenario slipped into the attack suite")

        conditions = orchestrator._enumerate_conditions(config)
        if len(conditions) != 1:
            raise RuntimeError("expected exactly one live condition in dedicated config")
        condition_spec = conditions[0]
        condition = condition_spec["profiles"]
        condition_name = orchestrator._condition_name(
            condition,
            override=condition_spec.get("name"),
        )
        reproducibility = orchestrator._configure_reproducibility(
            config=config,
            conditions=conditions,
        )
        orchestrator._current_reproducibility = reproducibility
        results_base_dir = (
            Path(config["_path"]).parent if "results_root" in config else REPO_ROOT
        )
        resolved_results_root = resolve_repo_path(
            config.get("results_root"), base_dir=results_base_dir
        )
        experiment_root = resolved_results_root / str(config.get("experiment_name"))
        condition_root = experiment_root / condition_name

        completed_runs = _load_completed_runs()
        completed_runs["expected_scenarios"] = [scenario.catalog_id for scenario in scenarios]
        _save_completed_runs(completed_runs)
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
            results_root=portable_path(experiment_root),
        )

        if _load_phase_status()["validation_tests"]["status"] != "completed":
            _set_phase("validation_tests", "in_progress")
            _validate_runtime_and_model()
            _set_phase(
                "validation_tests",
                "completed",
                validate_log=portable_path(VALIDATE_LOG_PATH),
                pytest_log=portable_path(PYTEST_LOG_PATH),
                cuda_log=portable_path(CUDA_LOG_PATH),
            )

        _set_phase("scenario_run", "in_progress", condition_name=condition_name)
        run_summaries = _load_existing_suite_summaries(
            condition_root=condition_root,
            scenarios=scenarios,
        )
        known_runs = {item.get("run_id"): item for item in run_summaries}
        for scenario in scenarios:
            run_id = _run_id_for(scenario)
            run_dir = _run_dir_for(condition_root, scenario)
            if _quick_run_integrity(run_dir):
                summary = _load_run_summary(run_dir)
                known_runs[run_id] = summary
                completed_runs["completed"][scenario.catalog_id] = _scenario_record(
                    summary,
                    run_dir,
                )
                completed_runs["failed"].pop(scenario.catalog_id, None)
                _save_completed_runs(completed_runs)
                continue

            bundle = orchestrator._run_single(
                experiment_id=str(
                    config.get("experiment_id", config.get("experiment_name"))
                ),
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
                completed_runs["completed"][scenario.catalog_id] = _scenario_record(
                    summary,
                    run_dir,
                )
                completed_runs["failed"].pop(scenario.catalog_id, None)
                _write_text(LAST_STEP_PATH, f"scenario:{scenario.catalog_id}\n")
            else:
                error_text = "; ".join(
                    str(item.get("error"))
                    for item in summary.get("errors", [])
                    if isinstance(item, dict)
                ) or "scenario run failed or did not produce required artifacts"
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
                    run_summaries=current_runs,
                    reproducibility=reproducibility,
                )
                raise RuntimeError(
                    f"scenario `{scenario.catalog_id}` failed: {error_text}"
                )

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
                run_summaries=current_runs,
                reproducibility=reproducibility,
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
            run_summaries=ordered_runs,
            reproducibility=reproducibility,
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
        )
        _set_phase(
            "code_commit_push",
            _load_phase_status()["code_commit_push"].get("status", "pending"),
        )

        completion_state = {
            "completed_at_utc": utc_now(),
            "experiment_root": portable_path(experiment_root),
            "condition_name": condition_name,
            "scenario_count": len(ordered_runs),
            "suite_status": suite_summary.get("status"),
            "portal_inspection_path": portable_path(PORTAL_INSPECTION_PATH),
        }
        phase_status = _load_phase_status()
        phase_status["completion_state"] = completion_state
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
