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
import subprocess
import sys
import tempfile
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


def _run_json(command: list[str], *, timeout: int = 1800) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False,
    )
    if completed.returncode:
        raise CampaignError(
            f"command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"{completed.stderr[-2000:]}\n{completed.stdout[-2000:]}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CampaignError("command did not return JSON") from exc


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


def _inspect_episode(directory: Path) -> dict[str, Any]:
    episode_path = directory / "episode.json"
    if not episode_path.is_file():
        return {"episode_id": directory.name, "path": str(directory),
                "status": "INTERRUPTED", "evaluation": None}
    try:
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"episode_id": directory.name, "path": str(directory),
                "status": "TECHNICAL_ABORT", "evaluation": None}
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
    }


def inventory(results_root: Path) -> dict[str, Any]:
    conditions = {}
    for condition in CONDITIONS:
        base = results_root / condition
        episodes = [_inspect_episode(path) for path in sorted(base.glob(f"{condition}-*"))
                    if path.is_dir()]
        conditions[condition] = {
            "episodes": episodes,
            "latest": episodes[-1] if episodes else None,
            "eligible_for_new_run": not episodes,
        }
    return {"generated_at": _now(), "results_root": str(results_root),
            "conditions": conditions}


def validate(worker_url: str, validation_root: Path, *, production_smoke: bool) -> dict[str, Any]:
    if os.environ.get("SCADAVER_ENABLE_SITL_ACTIONS") or os.environ.get("SCADAVER_SITL_ACTION_TOKEN"):
        raise CampaignError("VALIDATE/PREPARE refuse active action environment variables")
    parity = verify_runtime_manifest()
    profiles = validate_profiles()
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
        "readiness": readiness, "production_acceptance": production,
        "b2_r1_acceptance": r1,
        "acceptance_record": str(production_record) if production_record else None,
        "initial_condition_record": ((production or {}).get("testbed") or {}).get(
            "initial_condition_record"),
        "openrouter_calls": 0, "deployed_writes": 0,
        "action_tokens_created": 0, "live_experiment_executed": False,
    }
    _atomic_json(validation_root / "b2_r1_deployment_acceptance.json", result)
    return result


def prepare(results_root: Path, state_path: Path, validation: dict[str, Any]) -> dict[str, Any]:
    current = inventory(results_root)
    plan = []
    for condition in CONDITIONS:
        latest = current["conditions"][condition]["latest"]
        disposition = "ELIGIBLE" if latest is None else (
            "PRESERVE_COMPLETED" if latest["status"] == "EVALUATION_COMPLETE"
            else "OPERATOR_DECISION_REQUIRED"
        )
        plan.append({
            "condition": condition, "disposition": disposition,
            "existing_episode": latest["episode_id"] if latest else None,
            "profile_sha256": PROFILE_HASHES[condition],
        })
    state = {
        "campaign_version": "b2-r1-campaign-v1", "updated_at": _now(),
        "status": "READY", "inventory": current, "plan": plan,
        "deployment_ready": validation["deployment_ready"],
        "testbed_ready": validation["testbed_ready"],
        "initial_condition_policy": {
            "reset_performed": False,
            "strategy": "read-only matched level band selected by operator",
            "record": validation.get("initial_condition_record"),
            "limitation": "The continuously evolving plant is not reset by this campaign.",
        },
        "live_experiment_executed": False,
    }
    _atomic_json(state_path, state)
    return state


def _flatten(condition: str, episode: dict[str, Any] | None) -> dict[str, Any]:
    if not episode:
        return {"condition": condition, "status": "NOT_AVAILABLE"}
    evaluation = episode.get("evaluation") or {}
    read_only_evaluation = episode.get("b2_r1_readonly_evaluation") or {}
    usage = ((episode.get("statistics") or {}).get("usage") or {})
    reasoning = episode.get("reasoning_diagnostics") or {}
    row = {
        "condition": condition, "status": episode.get("status"),
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
    rows = [_flatten(condition, current["conditions"][condition]["latest"])
            for condition in CONDITIONS]
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "C0_C4_comparison.json", {"generated_at": _now(), "rows": rows})
    fields = sorted({key for row in rows for key in row})
    with (output / "C0_C4_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": rows, "output": str(output)}


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


def run_condition(condition: str, results_root: Path, worker_url: str,
                  seed: int | None = None,
                  level_min_raw: int | None = None,
                  level_max_raw: int | None = None) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise CampaignError("RUN condition must be C0-C4")
    if inventory(results_root)["conditions"][condition]["episodes"]:
        raise CampaignError(f"{condition} already has an episode; automatic replacement is prohibited")
    source_commit = _validated_source_commit()
    validation = validate(worker_url, results_root / "engineering_validation",
                          production_smoke=True)
    matched_state = _matched_initial_state(
        validation.get("initial_condition_record"), level_min_raw, level_max_raw,
    )
    profile = ROOT / f"config/b2_r1_v1/{condition}.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = results_root / condition / f"{condition}-{timestamp}"
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
        "--acceptance-record", str(validation["acceptance_record"]),
        "--objective", objective, "--out-dir", str(out_dir),
    ]
    if seed is not None:
        command += ["--seed", str(seed)]
    state_path = results_root / "campaign_state.json"
    _atomic_json(state_path, {
        "status": "RUNNING", "condition": condition, "episode_id": out_dir.name,
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
                                  "episode_id": out_dir.name, "updated_at": _now()})
        raise
    finally:
        environment.pop("SCADAVER_ENABLE_SITL_ACTIONS", None)
        environment.pop("SCADAVER_SITL_ACTION_TOKEN", None)
    post = _run_json([
        sys.executable, "-m", "experiments.scadaver_sphere.operator_workflow", "verify",
    ], timeout=900)
    if out_dir.exists():
        _atomic_json(out_dir / "operator_post_verify.json", post)
        _atomic_json(out_dir / "matched_initial_state.json", matched_state)
        (out_dir / "runner_stdout.log").write_text(completed.stdout)
        (out_dir / "runner_stderr.log").write_text(completed.stderr)
    status = ("COMPLETED" if completed.returncode == 0
              and post.get("TESTBED_READY") is True else "TECHNICAL_ABORT")
    final = {
        "status": status, "condition": condition, "episode_id": out_dir.name,
        "inference_seed": seed, "runner_returncode": completed.returncode,
        "source_commit": source_commit,
        "testbed_ready": post.get("TESTBED_READY"), "read_only": True,
        "action_authorization_created": False,
        "matched_initial_state": matched_state, "ended_at": _now(),
        "validation_manifest": validation.get("parity", {}).get("manifest_sha256"),
    }
    _atomic_json(state_path, final)
    return final


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
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--matched-level-min-raw", type=int)
    parser.add_argument("--matched-level-max-raw", type=int)
    args = parser.parse_args()
    state_path = args.state_file or args.results_root / "campaign_state.json"
    validation_root = args.validation_root or args.results_root / "engineering_validation"
    try:
        with campaign_lock(args.results_root):
            if args.mode == "status":
                result = inventory(args.results_root)
            elif args.mode == "validate":
                result = validate(args.worker_url, validation_root,
                                  production_smoke=args.production_smoke)
            elif args.mode == "prepare":
                accepted = validate(args.worker_url, validation_root,
                                    production_smoke=args.production_smoke)
                result = prepare(args.results_root, state_path, accepted)
            elif args.mode == "report":
                result = generate_reports(
                    args.results_root, args.report_dir or args.results_root / "comparisons")
            else:
                if not args.execute_live:
                    raise CampaignError("RUN requires the explicit --execute-live flag")
                if not args.condition:
                    raise CampaignError("RUN requires --condition")
                result = run_condition(
                    args.condition, args.results_root, args.worker_url,
                    args.seed,
                    args.matched_level_min_raw, args.matched_level_max_raw,
                )
    except CampaignError as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
