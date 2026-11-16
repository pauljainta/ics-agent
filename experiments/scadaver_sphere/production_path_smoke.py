"""No-model, no-write production-path acceptance for the integrated runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .agent import run_agent
from .budgets import EpisodeBudget, EpisodeLimits
from .deployment_parity import DEFAULT_MANIFEST, DeploymentParityError, verify_runtime_manifest
from .integrated_runner import CompositeExecutor


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


class SubmissionExecutor:
    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "submit_integrated_report":
            return {"tool": name, "status": "accepted", "report": arguments["report"]}
        return {"tool": name, "status": "rejected", "error": "smoke_read_only"}


def report() -> dict[str, Any]:
    return {
        "run_id": "production-smoke", "objective": "offline acceptance", "systems": [],
        "relationships": [], "permission_assessment": [],
        "objective_plan": {"summary": "No action.", "steps": [],
                           "supporting_observation_ids": [], "requires_action": False,
                           "limitations": ["synthetic acceptance"]},
        "action_attempted": False, "outcome": "synthetic acceptance",
        "outcome_observation_ids": [], "overall_assessment": "synthetic acceptance",
    }


def submission_call() -> dict[str, Any]:
    return {"tool_calls": [{"id": "submit", "function": {
        "name": "submit_integrated_report", "arguments": json.dumps({"report": report()}),
    }}]}


def synthetic_checks() -> dict[str, Any]:
    tools = [{"type": "function", "function": {
        "name": "submit_integrated_report", "parameters": {"type": "object"},
    }}]

    class Normal:
        def chat(self, messages, schemas): return submission_call()

    normal = run_agent(Normal(), SubmissionExecutor(), allowed_cidr="192.0.2.0/24",
                       max_steps=2, tool_schemas=tools,
                       submission_tool_name="submit_integrated_report",
                       report_validator=lambda value: value)

    class Truncated:
        def __init__(self): self.calls = 0
        def chat(self, messages, schemas):
            self.calls += 1
            if self.calls == 1:
                return {"content": '{"partial":true',
                        "_scadaver_response_metadata": {"finish_reason": "length"}}
            return submission_call()

    truncated_client = Truncated()
    recovered = run_agent(truncated_client, SubmissionExecutor(), allowed_cidr="192.0.2.0/24",
                          max_steps=3, tool_schemas=tools,
                          submission_tool_name="submit_integrated_report",
                          report_validator=lambda value: value)

    exhausted_budget = EpisodeBudget(EpisodeLimits(max_model_calls=1, max_tool_calls=2))
    exhausted = run_agent(Truncated(), SubmissionExecutor(), allowed_cidr="192.0.2.0/24",
                          max_steps=3, tool_schemas=tools,
                          submission_tool_name="submit_integrated_report",
                          report_validator=lambda value: value, budget=exhausted_budget)

    class Failure:
        def chat(self, messages, schemas): raise RuntimeError("controlled smoke failure")

    failed = run_agent(Failure(), SubmissionExecutor(), allowed_cidr="192.0.2.0/24",
                       max_steps=1, tool_schemas=tools,
                       submission_tool_name="submit_integrated_report",
                       report_validator=lambda value: value)

    class Transport:
        def execute(self, name, arguments):
            return {"tool": name, "status": "ok"}

    action_disabled = CompositeExecutor(Transport(), None, None).execute(
        "bounded_binary_output", {"ip": "192.0.2.1", "port": 1, "address": 0,
                                  "active": True, "duration_seconds": 0.1})

    with tempfile.TemporaryDirectory() as temporary:
        value = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        value["dependencies"]["definitely-missing-scadaver-package"] = "0"
        invalid = Path(temporary) / "invalid-manifest.json"
        invalid.write_text(json.dumps(value), encoding="utf-8")
        missing_dependency_rejected = False
        try:
            verify_runtime_manifest(invalid)
        except DeploymentParityError:
            missing_dependency_rejected = True

    checks = {
        "normal_termination": normal.stop_reason == "submitted_integrated_report",
        "truncation_recovery": recovered.stop_reason == "submitted_integrated_report"
                               and truncated_client.calls == 2,
        "budget_exhaustion": exhausted.stop_reason == "submission_recovery_budget_exhausted",
        "error_handling": failed.stop_reason == "inference_error",
        "missing_dependency_detection": missing_dependency_rejected,
        "action_disabled": action_disabled.get("error") == "actions_not_authorized",
    }
    return {"passed": all(checks.values()), "checks": checks}


def production_mock(worker_url: str, out_root: Path) -> dict[str, Any]:
    target = out_root / "production_integrated_mock"
    objective = json.loads((ROOT / "config/first_qwen_objective.json").read_text())["agent_facing_objective"]
    command = [
        sys.executable, "-m", "experiments.scadaver_sphere.integrated_runner",
        "--mock", "--worker-url", worker_url,
        "--config", str(ROOT / "config/integrated_e1_e5_first_qwen.json"),
        "--reasoning-config", str(ROOT / "config/reasoning_ablation_v1/B0.json"),
        "--objective", objective, "--out-dir", str(target),
    ]
    environment = dict(os.environ, PYTHONPATH=str(REPO))
    completed = subprocess.run(command, cwd=REPO, env=environment, capture_output=True,
                               text=True, timeout=600, check=False)
    episode_path = target / "episode.json"
    episode = json.loads(episode_path.read_text()) if episode_path.exists() else {}
    checks = {
        "returncode_zero": completed.returncode == 0,
        "episode_created": episode_path.exists(),
        "parity_artifact_created": (target / "deployment_parity.json").exists(),
        "structured_submission": episode.get("status") == "submitted_integrated_report",
        "valid_evaluation": (episode.get("evaluation") or {}).get("valid_submission") is True,
        "b0_no_reasoning_state": episode.get("reasoning_state") is None,
        "actions_disabled": episode.get("action_enabled") is False,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "returncode": completed.returncode, "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:], "output": str(target)}


def reasoning_profile_mocks(worker_url: str, out_root: Path) -> dict[str, Any]:
    results = {}
    objective = json.loads((ROOT / "config/first_qwen_objective.json").read_text())["agent_facing_objective"]
    for condition in ("B1", "B2", "B3", "B4"):
        target = out_root / f"production_{condition.lower()}_mock"
        command = [
            sys.executable, "-m", "experiments.scadaver_sphere.integrated_runner",
            "--mock", "--worker-url", worker_url,
            "--config", str(ROOT / "config/integrated_e1_e5_first_qwen.json"),
            "--reasoning-config", str(ROOT / f"config/reasoning_ablation_v1/{condition}.json"),
            "--objective", objective, "--out-dir", str(target),
        ]
        completed = subprocess.run(command, cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO)),
                                   capture_output=True, text=True, timeout=600, check=False)
        episode_path = target / "episode.json"
        episode = json.loads(episode_path.read_text()) if episode_path.exists() else {}
        state = episode.get("reasoning_state") or {}
        current = state.get("current") or {}
        checks = {
            "returncode_zero": completed.returncode == 0,
            "structured_submission": episode.get("status") == "submitted_integrated_report",
            "reasoning_update": len(state.get("updates", [])) >= 1,
            "evidence_ledger": isinstance(current.get("evidence_ledger"), list),
            "hypotheses": condition == "B1" or isinstance(current.get("hypotheses"), list),
            "pre_action_review": condition in {"B1", "B2"} or "pre_action_review" in current,
            "information_gain": condition != "B4" or isinstance(current.get("information_gain_selection"), dict),
            "actions_disabled": episode.get("action_enabled") is False,
        }
        results[condition] = {"passed": all(checks.values()), "checks": checks,
                              "stderr": completed.stderr[-1000:]}
    return {"passed": all(value["passed"] for value in results.values()), "conditions": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=False)
    result = {
        "parity": verify_runtime_manifest(),
        "synthetic": synthetic_checks(),
        "production_mock": production_mock(args.worker_url, args.out_root),
        "reasoning_profile_mocks": reasoning_profile_mocks(args.worker_url, args.out_root),
        "openrouter_calls": 0,
        "deployed_writes": 0,
    }
    result["passed"] = all((result["parity"]["passed"], result["synthetic"]["passed"],
                            result["production_mock"]["passed"], result["reasoning_profile_mocks"]["passed"]))
    (args.out_root / "acceptance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
