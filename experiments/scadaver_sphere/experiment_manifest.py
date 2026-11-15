"""Generate a secret-free, hash-linked first-run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def build_manifest(baseline: Path, output_root: Path) -> dict[str, Any]:
    critical = [
        ROOT / "config/integrated_e1_e5_first_qwen.json",
        ROOT / "config/first_qwen_objective.json",
        ROOT / "config/first_qwen_e5_authorization.template.json",
        ROOT / "config/qwen_openrouter_model_v1.json",
        ROOT / "private/integrated_e1_e5_first_qwen_spec.json",
        ROOT / "prompts/integrated_system_v1.md",
        ROOT / "prompts/integrated_e1_e5_v1.md",
        ROOT / "integrated_runner.py",
        ROOT / "integrated_evaluation.py",
        ROOT / "operator_workflow.py",
    ]
    baseline_value = json.loads(baseline.read_text())
    return {
        "manifest_version": "scadaver-first-qwen-e1-e5-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_tag": git_value("describe", "--tags", "--exact-match", "HEAD"),
        "realization": "scadaver.lgarcia",
        "testbed_profile": "heterogeneous-sitl-v1",
        "startup_procedure": "operator_workflow-v1",
        "initial_condition_policy": "record continuously evolving state; do not claim deterministic reset",
        "model_configuration": json.loads((ROOT / "config/qwen_openrouter_model_v1.json").read_text()),
        "objective": json.loads((ROOT / "config/first_qwen_objective.json").read_text()),
        "episode_profile": json.loads((ROOT / "config/integrated_e1_e5_first_qwen.json").read_text()),
        "authorization_template_sha256": digest(ROOT / "config/first_qwen_e5_authorization.template.json"),
        "evaluator_version": "integrated-e1-e5-first-qwen-v1",
        "trace_schema": "integrated_runner episode.json v1; raw message and request records retained",
        "recovery": "fail closed, persist partial trace, disable token, operator workflow recover and verify",
        "stopping_conditions": ["valid final submission", "global budget", "deadline", "inference failure", "tool transport failure", "operator interruption"],
        "output_root": str(output_root),
        "baseline_sha256": digest(baseline),
        "baseline_summary": {
            "testbed_ready": baseline_value.get("TESTBED_READY"),
            "initial_process_state": baseline_value.get("initial_process_state"),
            "openplc_activation": baseline_value.get("status", {}).get("openplc_lifecycle", {}).get("activation"),
        },
        "critical_files": {str(path.relative_to(ROOT)): digest(path) for path in critical},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = build_manifest(args.baseline, args.output_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(args.out), "sha256": digest(args.out)}, indent=2))


if __name__ == "__main__":
    main()
