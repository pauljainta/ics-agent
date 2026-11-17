"""Durable C0-C4 campaign orchestration for the B2-R1 study."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .b2_r1 import B2R1Profile
from .b2_r1_acceptance import frozen_baseline_checks
from .deployment_parity import verify_runtime_manifest


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CONDITIONS = ("C0", "C1", "C2", "C3", "C4")
PROFILE_HASHES = {
    "C0": "0a12a98f20c9c2e2afee72aa4700bc3c3488a4c097687e2dad9901ec37217a97",
    "C1": "ffe490ac32229a6cab84cc475ce36e697566a83e76efe63e2696b58a33cec1e4",
    "C2": "cd6411bed6cf68b7192c4c96995110c8891554a9f9e8586dd61672d76ad22b21",
    "C3": "cd6f75f58603b0db3b02b6f54909356bc95eac51b559853e50b673d10dd003b6",
    "C4": "9c09a989e46a07d8d104b9c66d232efbb5ba93cdb913d87b4776b487c5f54146",
}
PLAN_PATH = ROOT / "config/b2_r1_v1/pilot_plan.json"
RUBRIC_PATH = ROOT / "config/b2_r1_v1/scoring_rubric.json"
PLAN_SHA256 = "1dbbc1a7bea12b2ac14cff403172d18092f52d7fe67a78cd156c686c61fab08c"
RUBRIC_SHA256 = "8410927adaa26e02ec6adad9b299339d48cbacad5ed7a13119aa723d75e4c808"
ACCESS_VM_IP = "198.18.4.30"
INDUSTRIAL_CIDR = "198.18.4.0/24"
GLOBAL_RUN_LOCK = Path("/tmp/ics-agent-b2-r1-campaign.lock")


class CampaignError(RuntimeError):
    pass


def _validated_source_commit() -> str:
    value = os.environ.get("ICS_AGENT_SOURCE_COMMIT", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise CampaignError(
            "ICS_AGENT_SOURCE_COMMIT must identify the exact 40-character review commit"
        )
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


@contextlib.contextmanager
def campaign_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".campaign.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError("another B2-R1 campaign process holds the lock") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def global_run_lock() -> Iterator[None]:
    """Prevent concurrent live campaigns even when they use different roots."""
    with GLOBAL_RUN_LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError("another live B2-R1 campaign is active on this host") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _run_json(command: list[str], *, timeout: int = 1800,
              allow_nonzero: bool = False) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False,
    )
    if completed.returncode and not allow_nonzero:
        raise CampaignError(
            f"command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"{completed.stderr[-2000:]}\n{completed.stdout[-2000:]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CampaignError("command did not return JSON") from exc
    if not isinstance(value, dict):
        raise CampaignError("command returned a non-object JSON value")
    value.setdefault("command_returncode", completed.returncode)
    return value


def validate_profiles() -> dict[str, Any]:
    results = {}
    for condition, expected in PROFILE_HASHES.items():
        path = ROOT / f"config/b2_r1_v1/{condition}.json"
        profile = B2R1Profile.from_json(path)
        actual = _sha256(path)
        results[condition] = {
            "path": str(path), "expected": expected, "actual": actual,
            "passed": actual == expected and profile.condition == condition,
        }
    if not all(item["passed"] for item in results.values()):
        raise CampaignError("B2-R1 profile hash or semantic validation failed")
    return results


def load_pilot_plan() -> dict[str, Any]:
    try:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"pilot plan is unreadable: {exc}") from exc
    if _sha256(PLAN_PATH) != PLAN_SHA256:
        raise CampaignError("pilot plan hash does not match the frozen design")
    blocks = plan.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise CampaignError("pilot plan must contain at least one block")
    schedule: list[dict[str, Any]] = []
    for block in blocks:
        number = block.get("block")
        seed = block.get("seed")
        order = block.get("order")
        if (not isinstance(number, int) or not isinstance(seed, int)
                or not isinstance(order, list) or tuple(sorted(order)) != CONDITIONS):
            raise CampaignError("each pilot block must contain one ordered C0-C4 permutation")
        for position, condition in enumerate(order, start=1):
            schedule.append({
                "trial_id": f"block-{number:02d}-position-{position:02d}-{condition}-seed-{seed}",
                "block": number,
                "position": position,
                "condition": condition,
                "seed": seed,
            })
    if len(schedule) != plan.get("total_episodes"):
        raise CampaignError("pilot plan total_episodes does not match its schedule")
    expected_replicates = plan.get("replicates_per_condition")
    for condition in CONDITIONS:
        if sum(item["condition"] == condition for item in schedule) != expected_replicates:
            raise CampaignError(f"pilot plan has the wrong replicate count for {condition}")
    return {**plan, "schedule": schedule, "sha256": PLAN_SHA256}


def validate_study_assets() -> dict[str, Any]:
    plan = load_pilot_plan()
    try:
        rubric = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"scoring rubric is unreadable: {exc}") from exc
    if _sha256(RUBRIC_PATH) != RUBRIC_SHA256:
        raise CampaignError("scoring rubric hash does not match the frozen design")
    if rubric.get("rubric_version") != "b2-r1-readonly-evidence-v1":
        raise CampaignError("unexpected B2-R1 scoring rubric version")
    return {
        "passed": True,
        "pilot_plan": {"path": str(PLAN_PATH), "sha256": PLAN_SHA256,
                       "total_episodes": plan["total_episodes"]},
        "scoring_rubric": {"path": str(RUBRIC_PATH), "sha256": RUBRIC_SHA256,
                           "version": rubric["rubric_version"]},
    }


def access_vm_context() -> dict[str, Any]:
    """Verify that industrial checks are running from the deployed Access VM."""
    try:
        addresses = subprocess.run(
            ["ip", "-json", "address", "show"], capture_output=True, text=True,
            timeout=10, check=True,
        )
        routes = subprocess.run(
            ["ip", "route", "show", INDUSTRIAL_CIDR], capture_output=True,
            text=True, timeout=10, check=True,
        )
        values = json.loads(addresses.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"passed": False, "failure_layer": "access_vm_context",
                "error": type(exc).__name__}
    ips = sorted({
        item.get("local") for interface in values
        for item in interface.get("addr_info", [])
        if isinstance(item, dict) and isinstance(item.get("local"), str)
    })
    route = routes.stdout.strip()
    access_ip_present = ACCESS_VM_IP in ips
    industrial_route_present = INDUSTRIAL_CIDR in route
    return {
        "passed": access_ip_present and industrial_route_present,
        "hostname": socket.gethostname(),
        "access_vm_ip": ACCESS_VM_IP,
        "access_vm_ip_present": access_ip_present,
        "industrial_cidr": INDUSTRIAL_CIDR,
        "industrial_route_present": industrial_route_present,
        "failure_layer": None if access_ip_present and industrial_route_present
                         else "wrong_execution_host_or_industrial_route",
    }


def _operator(operation: str, *, allow_nonzero: bool = False) -> dict[str, Any]:
    return _run_json([
        sys.executable, "-m", "experiments.scadaver_sphere.operator_workflow", operation,
    ], timeout=900, allow_nonzero=allow_nonzero)


def _active_live_runners() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["pgrep", "-af", "experiments.scadaver_sphere.integrated_runner"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    result = []
    for line in completed.stdout.splitlines():
        if "--execute-live-model" not in line:
            continue
        pid, _, command = line.partition(" ")
        if pid.isdigit() and int(pid) != os.getpid():
            result.append({"pid": int(pid), "command": command})
    return result


def _initial_condition_from_readiness(readiness: dict[str, Any]) -> dict[str, Any] | None:
    closed_loop = (readiness.get("deployed_acceptance") or {}).get("closed_loop") or {}
    samples = closed_loop.get("samples") or []
    if not samples or not isinstance(samples[-1], dict):
        return None
    return {
        "captured_at_unix": time.time(),
        "source": "deployed_read_only_closed_loop_validation",
        "process_sample": samples[-1],
        "openplc_core": (readiness.get("status") or {}).get("openplc_lifecycle"),
        "service_health": (readiness.get("status") or {}).get("ports"),
        "controller_mode": "automatic_running",
        "process_phase": "continuously_evolving_uncontrolled_phase",
        "reset_performed": False,
    }


def _inspect_episode(directory: Path) -> dict[str, Any]:
    metadata_path = directory / "trial_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    episode_path = directory / "episode.json"
    if not episode_path.is_file():
        return {"episode_id": directory.name, "path": str(directory),
                "status": "INTERRUPTED", "evaluation": None,
                "trial_metadata": metadata}
    try:
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"episode_id": directory.name, "path": str(directory),
                "status": "TECHNICAL_ABORT", "evaluation": None,
                "trial_metadata": metadata}
    post_path = directory / "operator_post_verify.json"
    try:
        recovered = json.loads(post_path.read_text()).get("TESTBED_READY") is True
    except (OSError, json.JSONDecodeError):
        recovered = False
    evaluation = episode.get("evaluation")
    complete = (episode.get("status") == "submitted_integrated_report"
                and isinstance(evaluation, dict) and recovered)
    return {
        "episode_id": directory.name, "path": str(directory),
        "status": "EVALUATION_COMPLETE" if complete else "TECHNICAL_ABORT",
        "evaluation": evaluation, "statistics": episode.get("statistics"),
        "b2_r1_readonly_evaluation": episode.get("b2_r1_readonly_evaluation"),
        "reasoning_diagnostics": episode.get("reasoning_diagnostics"),
        "initial_condition_record": episode.get("initial_condition_record"),
        "recovery_complete": recovered,
        "trial_metadata": metadata,
        "reproducibility": episode.get("reproducibility"),
    }


def inventory(results_root: Path) -> dict[str, Any]:
    conditions = {}
    for condition in CONDITIONS:
        directories = {
            path.resolve(): path for pattern in (
                f"{condition}/{condition}-*",
                f"episodes/{condition}/*",
            ) for path in results_root.glob(pattern) if path.is_dir()
        }
        episodes = [_inspect_episode(path) for path in sorted(
            directories.values(), key=lambda item: item.name,
        )]
        conditions[condition] = {
            "episodes": episodes,
            "latest": episodes[-1] if episodes else None,
            "eligible_for_new_run": not episodes,
        }
    state_path = results_root / "campaign_state.json"
    state: dict[str, Any] | None = None
    state_error = None
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            state_error = type(exc).__name__
    return {"generated_at": _now(), "results_root": str(results_root),
            "conditions": conditions, "campaign_state": state,
            "campaign_state_error": state_error}


def _trial_index(current: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for condition in CONDITIONS:
        for episode in current["conditions"][condition]["episodes"]:
            trial_id = (episode.get("trial_metadata") or {}).get("trial_id")
            if isinstance(trial_id, str):
                result[trial_id] = episode
    return result


def validate(worker_url: str, validation_root: Path, *, production_smoke: bool) -> dict[str, Any]:
    if os.environ.get("SCADAVER_ENABLE_SITL_ACTIONS") or os.environ.get("SCADAVER_SITL_ACTION_TOKEN"):
        raise CampaignError("VALIDATE/PREPARE refuse active action environment variables")
    context = access_vm_context()
    if context.get("passed") is not True:
        raise CampaignError(
            "industrial validation must execute on the Access VM with its industrial route"
        )
    parity = verify_runtime_manifest()
    profiles = validate_profiles()
    study_assets = validate_study_assets()
    baseline = frozen_baseline_checks()
    if not baseline["passed"]:
        raise CampaignError("frozen B2/harness baseline changed")
    readiness = _run_json([
        sys.executable, "-m", "experiments.scadaver_sphere.operator_workflow", "verify",
    ], timeout=900)
    if readiness.get("TESTBED_READY") is not True:
        raise CampaignError("operator readiness did not report TESTBED_READY=true")

    production = None
    r1 = None
    production_record = None
    if production_smoke:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        production_root = validation_root / f"production-acceptance-{stamp}"
        production = _run_json([
            sys.executable, "-m", "experiments.scadaver_sphere.production_acceptance",
            "--worker-url", worker_url, "--out-dir", str(production_root),
        ])
        production_record = production_root / "production_acceptance.json"
        if production.get("EXPERIMENT_READY") is not True or not production_record.is_file():
            raise CampaignError("frozen production acceptance failed")
        r1_root = validation_root / f"b2-r1-acceptance-{stamp}"
        r1 = _run_json([
            sys.executable, "-m", "experiments.scadaver_sphere.b2_r1_acceptance",
            "--worker-url", worker_url, "--out-root", str(r1_root),
        ])
        if r1.get("B2_R1_READY") is not True:
            raise CampaignError("B2-R1 production-path acceptance failed")

    result = {
        "generated_at": _now(), "deployment_ready": True, "testbed_ready": True,
        "profiles": profiles, "frozen_baseline": baseline, "parity": parity,
        "execution_context": context, "study_assets": study_assets,
        "readiness": readiness, "production_acceptance": production,
        "b2_r1_acceptance": r1,
        "acceptance_record": str(production_record) if production_record else None,
        "initial_condition_record": (
            ((production or {}).get("testbed") or {}).get("initial_condition_record")
            or _initial_condition_from_readiness(readiness)
        ),
        "openrouter_calls": 0, "deployed_writes": 0,
        "action_tokens_created": 0, "live_experiment_executed": False,
        "DEPLOYMENT_READY": parity.get("passed") is True,
        "TESTBED_READY": readiness.get("TESTBED_READY") is True,
        "CAMPAIGN_RUNNER_READY": bool(
            parity.get("passed") is True and readiness.get("TESTBED_READY") is True
            and study_assets.get("passed") is True
            and (not production_smoke or (production or {}).get("EXPERIMENT_READY") is True)
            and (not production_smoke or (r1 or {}).get("B2_R1_READY") is True)
        ),
        "LIVE_QWEN_CALLS": 0,
        "EXPERIMENTAL_SIMULATOR_WRITES": 0,
    }
    _atomic_json(validation_root / "b2_r1_deployment_acceptance.json", result)
    return result


def prepare_environment(worker_url: str, validation_root: Path) -> dict[str, Any]:
    context = access_vm_context()
    if context.get("passed") is not True:
        raise CampaignError(
            "PREPARE must run on access (198.18.4.30); do not test industrial routing from the repository host"
        )
    before = _operator("status", allow_nonzero=True)
    recovery = None
    if before.get("TESTBED_READY") is not True:
        recovery = _operator("recover", allow_nonzero=True)
        if recovery.get("TESTBED_READY") is not True:
            raise CampaignError("non-destructive Access-VM service recovery did not restore readiness")
    accepted = validate(worker_url, validation_root, production_smoke=True)
    return {"execution_context": context, "status_before": before,
            "recovery": recovery, "validation": accepted,
            "CAMPAIGN_RUNNER_READY": accepted["CAMPAIGN_RUNNER_READY"],
            "LIVE_QWEN_CALLS": 0, "EXPERIMENTAL_SIMULATOR_WRITES": 0}


def campaign_status(results_root: Path, worker_url: str) -> dict[str, Any]:
    current = inventory(results_root)
    context = access_vm_context()
    operator = (_operator("status", allow_nonzero=True)
                if context.get("passed") is True else {
                    "TESTBED_READY": False,
                    "failure_layer": "wrong_execution_host_or_industrial_route",
                })
    try:
        parity = verify_runtime_manifest()
    except Exception as exc:
        parity = {"passed": False, "error": type(exc).__name__, "detail": str(exc)}
    try:
        profiles = validate_profiles()
        assets = validate_study_assets()
    except CampaignError as exc:
        profiles = {}
        assets = {"passed": False, "error": str(exc)}
    model = json.loads(
        (ROOT / "config/integrated_e1_e5_first_qwen.json").read_text(encoding="utf-8")
    ).get("model")
    schedule = load_pilot_plan().get("schedule", []) if assets.get("passed") else []
    trials = _trial_index(current)
    pending = [item for item in schedule if item["trial_id"] not in trials]
    interrupted = [value for value in trials.values()
                   if value.get("status") in {"INTERRUPTED", "TECHNICAL_ABORT"}]
    return {
        **current,
        "available_conditions": list(CONDITIONS),
        "model": model,
        "profiles": profiles,
        "study_assets": assets,
        "execution_context": context,
        "testbed_connectivity": operator,
        "deployment": parity,
        "active_live_runners": _active_live_runners(),
        "planned_trials": len(schedule),
        "completed_planned_trials": len(schedule) - len(pending),
        "pending_trials": pending,
        "interrupted_or_aborted_trials": interrupted,
        "CAMPAIGN_RUNNER_READY": bool(
            context.get("passed") is True and operator.get("TESTBED_READY") is True
            and parity.get("passed") is True and assets.get("passed") is True
        ),
        "LIVE_QWEN_CALLS": 0,
        "EXPERIMENTAL_SIMULATOR_WRITES": 0,
    }


def prepare(results_root: Path, state_path: Path, validation: dict[str, Any],
            level_min_raw: int | None = None,
            level_max_raw: int | None = None) -> dict[str, Any]:
    current = inventory(results_root)
    pilot = load_pilot_plan()
    existing = _trial_index(current)
    plan = []
    for trial in pilot["schedule"]:
        episode = existing.get(trial["trial_id"])
        disposition = "ELIGIBLE" if episode is None else (
            "PRESERVE_COMPLETED" if episode["status"] == "EVALUATION_COMPLETE"
            else "OPERATOR_DECISION_REQUIRED"
        )
        plan.append({**trial, "disposition": disposition,
                     "existing_episode": episode["episode_id"] if episode else None,
                     "profile_sha256": PROFILE_HASHES[trial["condition"]]})
    matched_state = _matched_initial_state(
        validation.get("initial_condition_record"), level_min_raw, level_max_raw,
    )
    state = {
        "campaign_version": "b2-r1-campaign-v2", "updated_at": _now(),
        "status": "READY", "inventory": current, "plan": plan,
        "deployment_ready": validation["deployment_ready"],
        "testbed_ready": validation["testbed_ready"],
        "campaign_runner_ready": validation.get("CAMPAIGN_RUNNER_READY") is True,
        "pilot_plan_sha256": pilot["sha256"],
        "scoring_rubric_sha256": RUBRIC_SHA256,
        "initial_condition_policy": {
            "reset_performed": False,
            "strategy": "read-only matched level band selected by operator",
            "record": validation.get("initial_condition_record"),
            "matched_state": matched_state,
            "limitation": "The continuously evolving plant is not reset by this campaign.",
        },
        "live_experiment_executed": False,
        "DEPLOYMENT_READY": validation["deployment_ready"],
        "TESTBED_READY": validation["testbed_ready"],
        "CAMPAIGN_RUNNER_READY": validation.get("CAMPAIGN_RUNNER_READY") is True,
        "LIVE_QWEN_CALLS": 0,
        "EXPERIMENTAL_SIMULATOR_WRITES": 0,
    }
    _atomic_json(state_path, state)
    _atomic_json(results_root / "campaign_readiness.json", {
        "generated_at": state["updated_at"],
        "DEPLOYMENT_READY": state["deployment_ready"],
        "TESTBED_READY": state["testbed_ready"],
        "CAMPAIGN_RUNNER_READY": state["campaign_runner_ready"],
        "LIVE_QWEN_CALLS": 0,
        "EXPERIMENTAL_SIMULATOR_WRITES": 0,
        "pilot_plan_sha256": pilot["sha256"],
        "scoring_rubric_sha256": RUBRIC_SHA256,
        "validation_manifest": (validation.get("parity") or {}).get("manifest_sha256"),
    })
    return state


def _flatten(condition: str, episode: dict[str, Any] | None) -> dict[str, Any]:
    if not episode:
        return {"condition": condition, "status": "NOT_AVAILABLE"}
    evaluation = episode.get("evaluation") or {}
    read_only_evaluation = episode.get("b2_r1_readonly_evaluation") or {}
    usage = ((episode.get("statistics") or {}).get("usage") or {})
    reasoning = episode.get("reasoning_diagnostics") or {}
    trial = episode.get("trial_metadata") or {}
    row = {
        "condition": condition, "status": episode.get("status"),
        "episode_id": episode.get("episode_id"),
        "trial_id": trial.get("trial_id"),
        "block": trial.get("block"),
        "position": trial.get("position"),
        "seed": trial.get("seed"),
        "read_only_core_success": read_only_evaluation.get("read_only_core_success"),
        # C0-C4 intentionally expose no action interface. Full E1-E5 success is
        # therefore not a defined outcome for this campaign.
        "full_system_overall_success": None,
        "model_calls": usage.get("model_calls"), "tool_calls": usage.get("tool_calls"),
        "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
        "api_cost_usd": usage.get("api_cost_usd"),
    }
    for key in (
        "unsuccessful_tool_calls", "redundant_probe_requests",
        "repeated_failed_request_count", "alternative_probe_after_failure_count",
        "successful_recovery_after_failure_count",
        "successful_evidence_per_investigative_call", "error_reviews",
        "hypotheses_with_successful_cited_test_results",
        "hypotheses_revised", "hypotheses_rejected",
        "contradictions_recognized", "uncertainty_reviews",
        "reviewed_supported_claims", "reviewed_uncertain_claims",
        "reviewed_contradicted_claims", "role_claims_with_role_evidence",
        "temporal_reviews_consistent", "unsupported_causal_claim_attempts",
    ):
        row[key] = reasoning.get(key)
    for stage, field in (("e1_pass", "e1_discovery"),
                         ("e2_pass", "e2_identification"),
                         ("e3_pass", "e3_understanding")):
        row[stage] = (read_only_evaluation.get(field) or {}).get("passed")
    planning = read_only_evaluation.get("read_only_planning") or {}
    row["e4_plan_present"] = planning.get("present")
    row["e4_plan_citations_valid"] = planning.get("citations_valid")
    e5 = evaluation.get("e5_execution") or {}
    row["e5_required"] = e5.get("required")
    row["e5_status"] = ("EVALUATED" if e5.get("required") is True
                        else "NOT_APPLICABLE_READ_ONLY")
    row["e5_pass"] = e5.get("passed") if e5.get("required") is True else None
    return row


def generate_reports(results_root: Path, output: Path) -> dict[str, Any]:
    current = inventory(results_root)
    rows = [
        _flatten(condition, episode)
        for condition in CONDITIONS
        for episode in current["conditions"][condition]["episodes"]
    ]
    if not rows:
        rows = [_flatten(condition, None) for condition in CONDITIONS]
    aggregates = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition
                          and row["status"] != "NOT_AVAILABLE"]
        complete = [row for row in condition_rows if row["status"] == "EVALUATION_COMPLETE"]
        aggregates[condition] = {
            "episodes": len(condition_rows),
            "evaluation_complete": len(complete),
            "technical_or_interrupted": len(condition_rows) - len(complete),
            "read_only_core_successes": sum(
                row.get("read_only_core_success") is True for row in complete
            ),
            "e5_status": "NOT_APPLICABLE_READ_ONLY",
            "full_system_overall_success": None,
        }
    output.mkdir(parents=True, exist_ok=True)
    generated_at = _now()
    _atomic_json(output / "C0_C4_comparison.json", {
        "generated_at": generated_at,
        "scope": "read-only E1-E3 plus descriptive planning; E5 is not applicable",
        "rubric_sha256": RUBRIC_SHA256,
        "rows": rows,
        "aggregates": aggregates,
    })
    fields = sorted({key for row in rows for key in row})
    with (output / "C0_C4_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# B2-R1 C0-C4 Comparison",
        "",
        "This campaign evaluates read-only E1-E3 investigation and descriptive planning. "
        "It does not evaluate E5 physical actions or complete E1-E5 success.",
        "",
        "| Condition | Episodes | Complete | Technical/Interrupted | Core Successes |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = aggregates[condition]
        lines.append(
            f"| {condition} | {item['episodes']} | {item['evaluation_complete']} | "
            f"{item['technical_or_interrupted']} | {item['read_only_core_successes']} |"
        )
    _atomic_text(output / "C0_C4_comparison.md", "\n".join(lines) + "\n")
    return {"generated_at": generated_at, "rows": rows,
            "aggregates": aggregates, "output": str(output)}


def _matched_initial_state(record: dict[str, Any] | None,
                           level_min_raw: int | None,
                           level_max_raw: int | None) -> dict[str, Any]:
    if (level_min_raw is None) != (level_max_raw is None):
        raise CampaignError("matched level band requires both minimum and maximum")
    if level_min_raw is None:
        return {"required": False, "matched": None, "record": record}
    if level_min_raw > level_max_raw:
        raise CampaignError("matched level minimum exceeds maximum")
    sample = (record or {}).get("process_sample") or {}
    level = sample.get("simulator_level")
    if not isinstance(level, int):
        raise CampaignError("initial condition record lacks simulator level")
    matched = level_min_raw <= level <= level_max_raw
    if not matched:
        raise CampaignError(
            f"initial level {level} is outside matched band [{level_min_raw}, {level_max_raw}]"
        )
    return {"required": True, "matched": True, "level_raw": level,
            "minimum_raw": level_min_raw, "maximum_raw": level_max_raw,
            "record": record}


def _write_artifact_checksums(directory: Path) -> dict[str, str]:
    output = directory / "artifact_checksums.json"
    checksums = {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != output
    }
    _atomic_json(output, {"algorithm": "sha256", "files": checksums})
    return checksums


def run_condition(condition: str, results_root: Path, worker_url: str,
                  seed: int | None = None,
                  level_min_raw: int | None = None,
                  level_max_raw: int | None = None,
                  *, trial_id: str | None = None,
                  block: int | None = None,
                  position: int | None = None,
                  accepted_validation: dict[str, Any] | None = None,
                  state_path: Path | None = None) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise CampaignError("RUN condition must be C0-C4")
    current = inventory(results_root)
    existing = _trial_index(current)
    if trial_id and trial_id in existing:
        raise CampaignError(f"trial {trial_id} already exists; automatic replacement is prohibited")
    if trial_id is None and current["conditions"][condition]["episodes"]:
        raise CampaignError(f"{condition} already has an episode; provide a new planned trial identity")
    source_commit = _validated_source_commit()
    lightweight = validate(
        worker_url, results_root / "engineering_validation", production_smoke=False,
    )
    validation = accepted_validation or validate(
        worker_url, results_root / "engineering_validation", production_smoke=True,
    )
    acceptance_record = validation.get("acceptance_record")
    if not acceptance_record or not Path(acceptance_record).is_file():
        raise CampaignError("live run requires a current production acceptance record")
    if ((validation.get("parity") or {}).get("manifest_sha256")
            != (lightweight.get("parity") or {}).get("manifest_sha256")):
        raise CampaignError("runtime manifest changed after production acceptance")
    active = _active_live_runners()
    if active:
        raise CampaignError("another live integrated runner is already active")
    matched_state = _matched_initial_state(
        lightweight.get("initial_condition_record") or validation.get("initial_condition_record"),
        level_min_raw, level_max_raw,
    )
    profile = ROOT / f"config/b2_r1_v1/{condition}.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    trial_id = trial_id or f"single-{condition}-seed-{seed if seed is not None else 'none'}"
    safe_trial = re.sub(r"[^A-Za-z0-9_.-]", "-", trial_id)
    out_dir = results_root / "episodes" / condition / f"{safe_trial}--{timestamp}"
    trial_metadata = {
        "trial_id": trial_id, "condition": condition, "block": block,
        "position": position, "seed": seed, "episode_id": out_dir.name,
        "source_commit": source_commit, "created_at": _now(),
        "pilot_plan_sha256": PLAN_SHA256, "scoring_rubric_sha256": RUBRIC_SHA256,
        "independent_context": True, "read_only": True,
    }
    objective = json.loads(
        (ROOT / "config/first_qwen_objective.json").read_text()
    )["agent_facing_objective"]
    environment = dict(os.environ, PYTHONPATH=str(REPO))
    environment["ICS_AGENT_SOURCE_COMMIT"] = source_commit
    environment.pop("SCADAVER_ENABLE_SITL_ACTIONS", None)
    environment.pop("SCADAVER_SITL_ACTION_TOKEN", None)
    command = [
        sys.executable, "-m", "experiments.scadaver_sphere.integrated_runner",
        "--execute-live-model", "--worker-url", worker_url,
        "--config", str(ROOT / "config/integrated_e1_e5_first_qwen.json"),
        "--reasoning-config", str(profile),
        "--acceptance-record", str(acceptance_record),
        "--objective", objective, "--out-dir", str(out_dir),
    ]
    if seed is not None:
        command += ["--seed", str(seed)]
    state_path = state_path or results_root / "campaign_state.json"
    _atomic_json(state_path, {
        "status": "RUNNING", "condition": condition, "episode_id": out_dir.name,
        "trial_id": trial_id, "block": block, "position": position,
        "started_at": _now(), "inference_seed": seed,
        "source_commit": source_commit,
        "read_only": True, "action_authorization_created": False,
        "matched_initial_state": matched_state, "command": shlex.join(command),
    })
    try:
        completed = subprocess.run(
            command, cwd=REPO, env=environment, capture_output=True,
            text=True, timeout=7500, check=False,
        )
    except BaseException:
        _atomic_json(state_path, {"status": "INTERRUPTED", "condition": condition,
                                  "episode_id": out_dir.name, "trial_id": trial_id,
                                  "updated_at": _now()})
        if out_dir.exists():
            _atomic_json(out_dir / "trial_metadata.json", trial_metadata)
            _write_artifact_checksums(out_dir)
        raise
    finally:
        environment.pop("SCADAVER_ENABLE_SITL_ACTIONS", None)
        environment.pop("SCADAVER_SITL_ACTION_TOKEN", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir / "trial_metadata.json", trial_metadata)
    _atomic_json(out_dir / "matched_initial_state.json", matched_state)
    _atomic_text(out_dir / "runner_stdout.log", completed.stdout)
    _atomic_text(out_dir / "runner_stderr.log", completed.stderr)
    try:
        post = _operator("verify")
    except CampaignError as exc:
        post = {"TESTBED_READY": False, "error": "postflight_verification_failed",
                "detail": str(exc)}
    _atomic_json(out_dir / "operator_post_verify.json", post)
    status = ("COMPLETED" if completed.returncode == 0
              and post.get("TESTBED_READY") is True else "TECHNICAL_ABORT")
    final = {
        "status": status, "condition": condition, "episode_id": out_dir.name,
        "trial_id": trial_id, "block": block, "position": position,
        "inference_seed": seed, "runner_returncode": completed.returncode,
        "source_commit": source_commit,
        "testbed_ready": post.get("TESTBED_READY"), "read_only": True,
        "action_authorization_created": False,
        "matched_initial_state": matched_state, "ended_at": _now(),
        "validation_manifest": validation.get("parity", {}).get("manifest_sha256"),
        "artifact_directory": str(out_dir),
    }
    _atomic_json(out_dir / "trial_metadata.json", trial_metadata)
    _atomic_json(out_dir / "campaign_episode_result.json", final)
    final["artifact_checksums"] = len(_write_artifact_checksums(out_dir))
    _atomic_json(state_path, final)
    return final


def run_pilot(results_root: Path, worker_url: str,
              level_min_raw: int | None, level_max_raw: int | None,
              *, max_episodes: int | None = None) -> dict[str, Any]:
    pilot = load_pilot_plan()
    schedule = pilot["schedule"]
    if max_episodes is not None:
        if max_episodes < 1 or max_episodes > len(schedule):
            raise CampaignError(f"--max-episodes must be between 1 and {len(schedule)}")
        schedule = schedule[:max_episodes]
    prepared = prepare_environment(worker_url, results_root / "engineering_validation")
    validation = prepared["validation"]
    state_path = results_root / "campaign_state.json"
    existing = _trial_index(inventory(results_root))
    blocked = [item for item in schedule if item["trial_id"] in existing]
    if blocked:
        raise CampaignError(
            "planned trial already has an artifact; use a new results root rather than rerunning it"
        )
    results = []
    _atomic_json(state_path, {
        "campaign_version": "b2-r1-campaign-v2", "status": "RUNNING",
        "started_at": _now(), "source_commit": _validated_source_commit(),
        "planned_episodes": len(schedule), "completed_episodes": 0,
        "schedule": schedule, "read_only": True,
        "action_authorization_created": False,
    })
    for index, trial in enumerate(schedule, start=1):
        try:
            result = run_condition(
                trial["condition"], results_root, worker_url, trial["seed"],
                level_min_raw, level_max_raw, trial_id=trial["trial_id"],
                block=trial["block"], position=trial["position"],
                accepted_validation=validation, state_path=state_path,
            )
        except CampaignError as exc:
            _atomic_json(state_path, {
                "campaign_version": "b2-r1-campaign-v2",
                "status": "STOPPED_ON_PREFLIGHT_FAILURE",
                "updated_at": _now(), "planned_episodes": len(schedule),
                "completed_episodes": len(results), "failed_trial": trial,
                "error": str(exc), "automatic_retry": False, "read_only": True,
            })
            raise
        results.append(result)
        _atomic_json(state_path, {
            "campaign_version": "b2-r1-campaign-v2",
            "status": "RUNNING" if index < len(schedule) else "COMPLETED",
            "updated_at": _now(), "source_commit": _validated_source_commit(),
            "planned_episodes": len(schedule), "completed_episodes": index,
            "last_trial": result, "schedule": schedule, "read_only": True,
            "action_authorization_created": False,
        })
        if result["status"] != "COMPLETED":
            _atomic_json(state_path, {
                "campaign_version": "b2-r1-campaign-v2", "status": "STOPPED_ON_TECHNICAL_ABORT",
                "updated_at": _now(), "planned_episodes": len(schedule),
                "completed_episodes": index, "failed_trial": result,
                "automatic_retry": False, "read_only": True,
            })
            break
    comparison = generate_reports(results_root, results_root / "comparisons")
    return {"status": json.loads(state_path.read_text())["status"],
            "results": results, "comparison": comparison,
            "LIVE_QWEN_CALLS": "model-driven only when RUN was explicitly invoked",
            "EXPERIMENTAL_SIMULATOR_WRITES": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "validate", "prepare", "report", "run"))
    parser.add_argument("--worker-url", default=os.environ.get(
        "SCADAVER_WORKER_URL", "http://127.0.0.1:18777"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--production-smoke", action="store_true")
    parser.add_argument("--execute-live", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--condition", choices=CONDITIONS)
    selection.add_argument("--all", action="store_true", dest="run_all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trial-id")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--matched-level-min-raw", type=int)
    parser.add_argument("--matched-level-max-raw", type=int)
    args = parser.parse_args()
    state_path = args.state_file or args.results_root / "campaign_state.json"
    validation_root = args.validation_root or args.results_root / "engineering_validation"
    try:
        with campaign_lock(args.results_root):
            if args.mode == "status":
                result = campaign_status(args.results_root, args.worker_url)
            elif args.mode == "validate":
                result = validate(args.worker_url, validation_root,
                                  production_smoke=True)
            elif args.mode == "prepare":
                prepared = prepare_environment(args.worker_url, validation_root)
                result = prepare(
                    args.results_root, state_path, prepared["validation"],
                    args.matched_level_min_raw, args.matched_level_max_raw,
                )
                result["preparation"] = prepared
            elif args.mode == "report":
                result = generate_reports(
                    args.results_root, args.report_dir or args.results_root / "comparisons")
            else:
                if not args.execute_live:
                    raise CampaignError("RUN requires the explicit --execute-live flag")
                if not args.condition and not args.run_all:
                    raise CampaignError("RUN requires either --condition or --all")
                if ((args.matched_level_min_raw is None)
                        != (args.matched_level_max_raw is None)):
                    raise CampaignError("RUN requires both matched-level bounds or neither")
                with global_run_lock():
                    if args.run_all:
                        if args.seed is not None or args.trial_id:
                            raise CampaignError("--all uses frozen plan seeds and trial identities")
                        if (args.matched_level_min_raw is None
                                and load_pilot_plan()["matched_initial_level_band_required"]):
                            raise CampaignError("the frozen pilot plan requires a matched initial level band")
                        result = run_pilot(
                            args.results_root, args.worker_url,
                            args.matched_level_min_raw, args.matched_level_max_raw,
                            max_episodes=args.max_episodes,
                        )
                    else:
                        if args.max_episodes is not None:
                            raise CampaignError("--max-episodes is valid only with --all")
                        if args.seed is None:
                            raise CampaignError("single-condition RUN requires --seed")
                        prepared = prepare_environment(args.worker_url, validation_root)
                        result = run_condition(
                            args.condition, args.results_root, args.worker_url,
                            args.seed, args.matched_level_min_raw,
                            args.matched_level_max_raw,
                            trial_id=args.trial_id,
                            accepted_validation=prepared["validation"],
                            state_path=state_path,
                        )
    except CampaignError as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error": str(exc)}, indent=2))
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"status": "INTERRUPTED", "automatic_retry": False}, indent=2))
        return 130
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
