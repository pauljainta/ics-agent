"""Durable orchestration for the B1-B4 reasoning-ablation campaign."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
import secrets
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .deployment_parity import verify_runtime_manifest
from .reasoning_ablation import ReasoningProfile


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CONDITIONS = ("B1", "B2", "B3", "B4")
PROFILE_HASHES = {
    "B1": "2e45ef3a1965df8548734909878e14c69c5ed72496c570c06c4c0de9e96b7aa3",
    "B2": "9ebf50871aaf09967394d1b12f5dbdcf1657ed0040be6d49c424ed5cf93e4b5b",
    "B3": "ac55208e232dbbe7ffd12b4f74cd1a799d50130f4fc860f2b69bd0655aabdfee",
    "B4": "cc01791df795a6a7cf4d7bba205f1bb3cdcbdc92efa24b1e87a102158a6dbd0b",
}


class CampaignError(RuntimeError):
    pass


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
    path = root / ".campaign.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError("another campaign process holds the lock") from exc
        handle.seek(0); handle.truncate(); handle.write(f"{os.getpid()}\n"); handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_profiles() -> dict[str, Any]:
    results = {}
    for condition, expected in PROFILE_HASHES.items():
        path = ROOT / f"config/reasoning_ablation_v1/{condition}.json"
        profile = ReasoningProfile.from_json(path)
        actual = _sha256(path)
        results[condition] = {"path": str(path), "expected": expected, "actual": actual,
                              "passed": actual == expected and profile.condition == condition}
    if not all(item["passed"] for item in results.values()):
        raise CampaignError("reasoning profile hash or semantic validation failed")
    return results


def _transaction_status(directory: Path) -> dict[str, Any]:
    states = list(directory.glob("e5_transactions/*/transaction_state.json"))
    if not states:
        return {"present": False, "complete": None}
    values = []
    for path in states:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            value = {"complete": False, "error": "unreadable"}
        values.append({"path": str(path), "complete": value.get("complete") is True,
                       "last_event": value.get("last_event")})
    return {"present": True, "complete": all(item["complete"] for item in values),
            "transactions": values}


def inspect_episode(directory: Path) -> dict[str, Any]:
    episode_path = directory / "episode.json"
    transaction = _transaction_status(directory)
    if not episode_path.exists():
        return {"episode_id": directory.name, "path": str(directory), "status": "INTERRUPTED",
                "valid_report": False, "evaluation_complete": False,
                "transaction": transaction, "recovery_complete": False,
                "checksums_present": (directory / "SHA256SUMS").exists()}
    try:
        episode = json.loads(episode_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"episode_id": directory.name, "path": str(directory), "status": "TECHNICAL_ABORT",
                "valid_report": False, "evaluation_complete": False,
                "transaction": transaction, "recovery_complete": False,
                "checksums_present": (directory / "SHA256SUMS").exists()}
    evaluation = episode.get("evaluation") if isinstance(episode.get("evaluation"), dict) else None
    post_path = directory / "operator_post_verify.json"
    try:
        recovery = json.loads(post_path.read_text()).get("TESTBED_READY") is True
    except (OSError, json.JSONDecodeError):
        recovery = False
    submitted = episode.get("status") == "submitted_integrated_report"
    return {
        "episode_id": directory.name, "path": str(directory),
        "status": "EVALUATION_COMPLETE" if submitted and evaluation else "TECHNICAL_ABORT",
        "valid_report": bool(submitted and evaluation and evaluation.get("valid_submission") is True),
        "evaluation_complete": evaluation is not None, "transaction": transaction,
        "recovery_complete": recovery, "checksums_present": (directory / "SHA256SUMS").exists(),
        "statistics": episode.get("statistics"), "evaluation": evaluation,
        "reasoning_profile": episode.get("reasoning_profile"),
    }


def inventory(results_root: Path) -> dict[str, Any]:
    conditions = {}
    for condition in ("B0", *CONDITIONS):
        base = results_root / condition
        episodes = [inspect_episode(path) for path in sorted(base.glob(f"{condition}-*")) if path.is_dir()]
        conditions[condition] = {"episodes": episodes,
                                 "latest": episodes[-1] if episodes else None,
                                 "eligible_for_new_run": condition != "B0" and not episodes}
    return {"generated_at": _now(), "results_root": str(results_root), "conditions": conditions}


def _run_json(command: list[str], *, cwd: Path = REPO, timeout: int = 900) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                               timeout=timeout, check=False)
    if completed.returncode:
        raise CampaignError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr[-2000:]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CampaignError("command did not return JSON") from exc


def validate(worker_url: str, validation_root: Path, *, production_smoke: bool) -> dict[str, Any]:
    if os.environ.get("SCADAVER_ENABLE_SITL_ACTIONS") or os.environ.get("SCADAVER_SITL_ACTION_TOKEN"):
        raise CampaignError("VALIDATE/PREPARE refuse active action environment variables")
    parity = verify_runtime_manifest()
    profiles = validate_profiles()
    readiness = _run_json([sys.executable, "-m", "experiments.scadaver_sphere.operator_workflow", "verify"])
    if readiness.get("TESTBED_READY") is not True:
        raise CampaignError("operator readiness did not report TESTBED_READY=true")
    smoke = None
    if production_smoke:
        smoke_root = validation_root / f"mock-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        smoke = _run_json([sys.executable, "-m", "experiments.scadaver_sphere.production_path_smoke",
                           "--worker-url", worker_url, "--out-root", str(smoke_root)])
        if smoke.get("passed") is not True:
            raise CampaignError("production-path smoke failed")
    result = {"generated_at": _now(), "deployment_ready": True, "testbed_ready": True,
              "profiles": profiles, "parity": parity, "readiness": readiness,
              "production_smoke": smoke, "openrouter_calls": 0, "deployed_writes": 0,
              "live_experiment_executed": False}
    _atomic_json(validation_root / "deployment_acceptance.json", result)
    return result


def prepare(results_root: Path, state_path: Path, validation: dict[str, Any]) -> dict[str, Any]:
    current = inventory(results_root)
    plan = []
    for condition in CONDITIONS:
        latest = current["conditions"][condition]["latest"]
        if latest is None:
            disposition = "ELIGIBLE"
        elif latest["status"] == "EVALUATION_COMPLETE":
            disposition = "PRESERVE_COMPLETED"
        else:
            disposition = "OPERATOR_DECISION_REQUIRED"
        plan.append({"condition": condition, "disposition": disposition,
                     "existing_episode": latest["episode_id"] if latest else None,
                     "profile_sha256": PROFILE_HASHES[condition]})
    state = {"campaign_version": "reasoning-ablation-campaign-v1", "updated_at": _now(),
             "status": "READY", "inventory": current, "plan": plan,
             "deployment_ready": validation["deployment_ready"],
             "testbed_ready": validation["testbed_ready"],
             "campaign_runner_ready": True, "live_experiment_executed": False}
    _atomic_json(state_path, state)
    return state


def _flatten(condition: str, episode: dict[str, Any] | None) -> dict[str, Any]:
    if not episode:
        return {"condition": condition, "status": "NOT_AVAILABLE"}
    evaluation = episode.get("evaluation") or {}
    usage = ((episode.get("statistics") or {}).get("usage") or {})
    e5 = evaluation.get("e5_execution") or {}
    return {"condition": condition, "episode_id": episode.get("episode_id"),
            "status": episode.get("status"), "valid_report": episode.get("valid_report"),
            "e1_pass": (evaluation.get("e1_discovery") or {}).get("passed"),
            "e2_pass": (evaluation.get("e2_service_protocol_identification") or {}).get("passed"),
            "e3_pass": (evaluation.get("e3_understanding") or {}).get("passed"),
            "e4_pass": (evaluation.get("e4_planning") or {}).get("passed"),
            "e5_pass": e5.get("passed"), "causal_attribution": e5.get("causal_attribution"),
            "overall_success": evaluation.get("overall_success"),
            "model_calls": usage.get("model_calls"), "tool_calls": usage.get("tool_calls"),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "api_cost_usd": usage.get("api_cost_usd")}


def generate_reports(results_root: Path, output: Path) -> dict[str, Any]:
    current = inventory(results_root)
    selected = {condition: current["conditions"][condition]["latest"]
                for condition in ("B0", *CONDITIONS)}
    corrected_path = results_root / "comparisons/B0_corrected_causal_evaluation.json"
    if selected["B0"] and corrected_path.is_file():
        corrected = json.loads(corrected_path.read_text())
        selected["B0"] = json.loads(json.dumps(selected["B0"]))
        evaluation = selected["B0"].setdefault("evaluation", {})
        evaluation["e5_execution"] = corrected["corrected_e5"]
        evaluation["evaluator_version"] = corrected["corrected_evaluator_version"]
        evaluation["overall_success"] = corrected["corrected_overall_success"]
        selected["B0"]["corrected_evaluation_applied"] = True
    rows = [_flatten(condition, selected[condition]) for condition in ("B0", *CONDITIONS)]
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "B0_B4_comparison.json", {"generated_at": _now(), "rows": rows})
    fields = sorted({key for row in rows for key in row})
    with (output / "B0_B4_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["# B0-B4 comparison", "", "| Condition | Status | E1 | E2 | E3 | E4 | E5 | Causal attribution | Overall |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for row in rows:
        lines.append("| {condition} | {status} | {e1_pass} | {e2_pass} | {e3_pass} | {e4_pass} | {e5_pass} | {causal_attribution} | {overall_success} |".format(**{key: row.get(key, "N/A") for key in ("condition", "status", "e1_pass", "e2_pass", "e3_pass", "e4_pass", "e5_pass", "causal_attribution", "overall_success")}))
    (output / "B0_B4_comparison.md").write_text("\n".join(lines) + "\n")
    return {"rows": rows, "output": str(output)}


def run_condition(condition: str, results_root: Path, worker_url: str,
                  authorization_file: Path | None = None) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise CampaignError("RUN condition must be B1-B4")
    existing = inventory(results_root)["conditions"][condition]["episodes"]
    if existing:
        raise CampaignError(f"{condition} already has an episode; automatic replacement is prohibited")
    generated_authorization = authorization_file is None
    if generated_authorization:
        packet = {
            "authorization_status": "AUTHORIZED",
            "authorization_id": f"{condition}-campaign-{secrets.token_hex(16)}",
            "condition": condition,
            "allowed_operation": "bounded_binary_output",
            "maximum_action_count": 1,
            "maximum_action_duration_seconds": 1.25,
        }
    else:
        assert authorization_file is not None
        if not authorization_file.is_file():
            raise CampaignError("condition-specific authorization file is missing")
        packet = json.loads(authorization_file.read_text())
        if packet.get("authorization_status") != "AUTHORIZED" or packet.get("condition") != condition:
            raise CampaignError("authorization packet is not active for the selected condition")
    validation = validate(worker_url, results_root / "engineering_validation",
                          production_smoke=False)
    profile = ROOT / f"config/reasoning_ablation_v1/{condition}.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = results_root / condition / f"{condition}-{timestamp}"
    objective = json.loads((ROOT / "config/first_qwen_objective.json").read_text())["agent_facing_objective"]
    token = secrets.token_urlsafe(32)
    environment = dict(os.environ, PYTHONPATH=str(REPO), SCADAVER_ENABLE_SITL_ACTIONS="1",
                       SCADAVER_SITL_ACTION_TOKEN=token)
    private_authorization = authorization_file
    if generated_authorization:
        private_root = Path.home() / ".config/ics-agent/campaign-authorizations"
        private_root.mkdir(parents=True, exist_ok=True)
        private_root.chmod(0o700)
        private_authorization = private_root / f"{packet['authorization_id']}.json"
        _atomic_json(private_authorization, packet)
        private_authorization.chmod(0o600)
    assert private_authorization is not None
    command = [sys.executable, "-m", "experiments.scadaver_sphere.integrated_runner",
               "--execute-live-model", "--enable-actions", "--worker-url", worker_url,
               "--config", str(ROOT / "config/integrated_e1_e5_first_qwen.json"),
               "--reasoning-config", str(profile), "--authorization-file", str(private_authorization),
               "--objective", objective, "--out-dir", str(out_dir)]
    state_path = results_root / "campaign_state.json"
    _atomic_json(state_path, {"status": "RUNNING", "condition": condition,
                              "episode_id": out_dir.name, "started_at": _now(),
                              "authorization_id": packet.get("authorization_id"),
                              "command": shlex.join(command)})
    try:
        completed = subprocess.run(command, cwd=REPO, env=environment, capture_output=True,
                                   text=True, timeout=7500, check=False)
    except BaseException:
        _atomic_json(state_path, {"status": "INTERRUPTED", "condition": condition,
                                  "episode_id": out_dir.name, "updated_at": _now()})
        raise
    finally:
        environment.pop("SCADAVER_SITL_ACTION_TOKEN", None)
        environment.pop("SCADAVER_ENABLE_SITL_ACTIONS", None)
        token = ""
        if generated_authorization and private_authorization is not None:
            private_authorization.unlink(missing_ok=True)
    post = _run_json([sys.executable, "-m", "experiments.scadaver_sphere.operator_workflow", "verify"])
    if out_dir.exists():
        _atomic_json(out_dir / "operator_post_verify.json", post)
        (out_dir / "runner_stdout.log").write_text(completed.stdout)
        (out_dir / "runner_stderr.log").write_text(completed.stderr)
    status = "COMPLETED" if completed.returncode == 0 and post.get("TESTBED_READY") is True else "TECHNICAL_ABORT"
    final = {"status": status, "condition": condition, "episode_id": out_dir.name,
             "runner_returncode": completed.returncode, "testbed_ready": post.get("TESTBED_READY"),
             "authorization_revoked": True, "ended_at": _now(),
             "validation_manifest": validation.get("parity", {}).get("manifest_sha256")}
    _atomic_json(state_path, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "validate", "prepare", "report", "run"))
    parser.add_argument("--worker-url", default=os.environ.get("SCADAVER_WORKER_URL", "http://127.0.0.1:18777"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--production-smoke", action="store_true")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--authorization-file", type=Path)
    args = parser.parse_args()
    state_path = args.state_file or args.results_root / "campaign_state.json"
    validation_root = args.validation_root or args.results_root / "engineering_validation"
    try:
        with campaign_lock(args.results_root):
            if args.mode == "status": result = inventory(args.results_root)
            elif args.mode == "validate": result = validate(args.worker_url, validation_root, production_smoke=args.production_smoke)
            elif args.mode == "prepare":
                validation = validate(args.worker_url, validation_root, production_smoke=args.production_smoke)
                result = prepare(args.results_root, state_path, validation)
            elif args.mode == "report": result = generate_reports(args.results_root, args.report_dir or args.results_root / "comparisons")
            else:
                if not args.execute_live:
                    raise CampaignError("RUN requires the explicit --execute-live flag")
                if not args.condition:
                    raise CampaignError("RUN requires --condition")
                result = run_condition(args.condition, args.results_root, args.worker_url,
                                       args.authorization_file)
    except CampaignError as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
