"""No-model, no-write production-path acceptance for B2-R1 C0-C4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .b2_r1 import B2R1Profile, FROZEN_BASELINE, load_prompt, reasoning_tool_schema
from .deployment_parity import verify_runtime_manifest
from .integrated_runner import ACTION_SCHEMA, FINAL_SCHEMA
from .reasoning_ablation import (
    ReasoningProfile,
    load_prompt as load_original_prompt,
    reasoning_tool_schema as original_reasoning_tool_schema,
)
from .schemas import E4_TOOL_SCHEMAS


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CONDITIONS = ("C0", "C1", "C2", "C3", "C4")


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_baseline_checks() -> dict[str, Any]:
    original = ReasoningProfile.from_json(ROOT / "config/reasoning_ablation_v1/B2.json")
    c0 = B2R1Profile.from_json(ROOT / "config/b2_r1_v1/C0.json")
    actual = {
        "original_b2_config_sha256": _hash_file(ROOT / "config/reasoning_ablation_v1/B2.json"),
        "original_b2_prompt_sha256": _hash_file(ROOT / "prompts/reasoning_ablation_v1/B2.md"),
        "base_scenario_sha256": _hash_file(ROOT / "config/integrated_e1_e5_first_qwen.json"),
        "system_prompt_sha256": _hash_file(ROOT / "prompts/integrated_system_v1.md"),
        "task_prompt_sha256": _hash_file(ROOT / "prompts/integrated_e1_e5_v1.md"),
        "integrated_evaluator_sha256": _hash_file(ROOT / "integrated_evaluation.py"),
        "e5_evaluator_sha256": _hash_file(ROOT / "e5_evaluation.py"),
        "action_broker_sha256": _hash_file(ROOT / "heterogeneous_sitl_v1/action_broker.py"),
        "investigative_tools_schema_sha256": _hash_json(E4_TOOL_SCHEMAS),
        "action_schema_sha256": _hash_json(ACTION_SCHEMA),
        "final_report_schema_sha256": _hash_json(FINAL_SCHEMA),
    }
    expected = {key: FROZEN_BASELINE[key] for key in actual}
    cases = {
        "hashes_unchanged": actual == expected,
        "c0_prompt_equals_original_b2": load_prompt(c0, ROOT) == load_original_prompt(original, ROOT),
        "c0_schema_equals_original_b2": reasoning_tool_schema(c0) == original_reasoning_tool_schema(original),
    }
    return {"passed": all(cases.values()), "cases": cases,
            "expected": expected, "actual": actual}


def profile_hashes() -> dict[str, Any]:
    values = {}
    for condition in CONDITIONS:
        path = ROOT / f"config/b2_r1_v1/{condition}.json"
        profile = B2R1Profile.from_json(path)
        values[condition] = {
            "config_sha256": _hash_file(path),
            "effective_prompt_sha256": hashlib.sha256(
                load_prompt(profile, ROOT).encode()).hexdigest(),
            "reasoning_schema_sha256": _hash_json(reasoning_tool_schema(profile)),
            "components": {name: getattr(profile, name) for name in (
                "error_aware_investigation", "evidence_verification", "uncertainty_review",
            )},
        }
    return values


def run_profile_mocks(worker_url: str, out_root: Path) -> dict[str, Any]:
    objective = json.loads(
        (ROOT / "config/first_qwen_objective.json").read_text(encoding="utf-8")
    )["agent_facing_objective"]
    results: dict[str, Any] = {}
    for condition in CONDITIONS:
        target = out_root / f"production_{condition.lower()}_mock"
        command = [
            sys.executable, "-m", "experiments.scadaver_sphere.integrated_runner",
            "--mock", "--worker-url", worker_url,
            "--config", str(ROOT / "config/integrated_e1_e5_first_qwen.json"),
            "--reasoning-config", str(ROOT / f"config/b2_r1_v1/{condition}.json"),
            "--objective", objective, "--out-dir", str(target),
        ]
        completed = subprocess.run(
            command, cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO)),
            capture_output=True, text=True, timeout=600, check=False,
        )
        episode_path = target / "episode.json"
        episode = json.loads(episode_path.read_text()) if episode_path.exists() else {}
        state = episode.get("reasoning_state") or {}
        updates = state.get("updates") or []
        current = state.get("current") or {}
        diagnostics = episode.get("reasoning_diagnostics") or {}
        checks = {
            "returncode_zero": completed.returncode == 0,
            "episode_created": episode_path.exists(),
            "structured_submission": episode.get("status") == "submitted_integrated_report",
            "valid_evaluation": (episode.get("evaluation") or {}).get("valid_submission") is True,
            "profile_family": state.get("profile_family") == "b2-r1-v1",
            "reasoning_state_persisted": len(updates) >= (2 if condition != "C0" else 1),
            "observation_registry": len(state.get("observation_registry") or {}) >= 1,
            "action_reasoning_not_added": "action_requests" not in state,
            "actions_disabled": episode.get("action_enabled") is False,
            "error_review_field": condition not in {"C1", "C4"} or "error_review" in current,
            "verification_cycle": condition not in {"C2", "C4"}
                                  or diagnostics.get("hypotheses_meaningfully_tested", 0) >= 1,
            "uncertainty_review": condition not in {"C3", "C4"}
                                  or diagnostics.get("uncertainty_reviews", 0) >= 1,
            "no_model_requests": episode.get("inference_requests") == [],
        }
        results[condition] = {
            "passed": all(checks.values()), "checks": checks,
            "stderr": completed.stderr[-2000:], "output": str(target),
        }
    return {"passed": all(item["passed"] for item in results.values()),
            "conditions": results}


def run_b2_r1_acceptance(worker_url: str, out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=False)
    parity = verify_runtime_manifest()
    baseline = frozen_baseline_checks()
    profiles = profile_hashes()
    mocks = run_profile_mocks(worker_url, out_root)
    result = {
        "acceptance_version": "b2-r1-engineering-acceptance-v1",
        "B2_R1_READY": bool(parity["passed"] and baseline["passed"] and mocks["passed"]),
        "openrouter_calls": 0,
        "deployed_writes": 0,
        "action_tokens_created": 0,
        "parity": parity,
        "frozen_baseline": baseline,
        "profiles": profiles,
        "production_mocks": mocks,
    }
    (out_root / "b2_r1_acceptance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        parser.error("--out-root must not already exist")
    try:
        result = run_b2_r1_acceptance(args.worker_url, args.out_root)
    except Exception as exc:
        print(json.dumps({"B2_R1_READY": False, "error": type(exc).__name__,
                          "detail": str(exc)}, indent=2))
        return 1
    print(json.dumps({
        "B2_R1_READY": result["B2_R1_READY"],
        "openrouter_calls": result["openrouter_calls"],
        "deployed_writes": result["deployed_writes"],
        "conditions": {name: value["passed"] for name, value
                       in result["production_mocks"]["conditions"].items()},
    }, indent=2, sort_keys=True))
    return 0 if result["B2_R1_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
