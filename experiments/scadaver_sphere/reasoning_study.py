"""Build reproducibility artifacts for the B0-B4 reasoning ablation study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .reasoning_ablation import ReasoningProfile


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("B0", "B1", "B2", "B3", "B4")
CONFIG_ROOT = ROOT / "config/reasoning_ablation_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(out_root: Path) -> None:
    for name in (*CONDITIONS, "configurations", "offline_validation", "comparisons", "reports"):
        (out_root / name).mkdir(parents=True, exist_ok=True)
    profiles = {}
    for condition in CONDITIONS:
        source = CONFIG_ROOT / f"{condition}.json"
        profile = ReasoningProfile.from_json(source)
        profiles[condition] = profile.public_snapshot()
        shutil.copyfile(source, out_root / "configurations" / source.name)
    tracked_inputs = [
        ROOT / "integrated_runner.py",
        ROOT / "agent.py",
        ROOT / "integrated_evaluation.py",
        ROOT / "reasoning_ablation.py",
        ROOT / "reasoning_metrics.py",
        ROOT / "config/integrated_e1_e5_first_qwen.json",
        ROOT / "prompts/integrated_system_v1.md",
        ROOT / "prompts/integrated_e1_e5_v1.md",
    ]
    tracked_inputs += [CONFIG_ROOT / f"{condition}.json" for condition in CONDITIONS]
    tracked_inputs += [ROOT / f"prompts/reasoning_ablation_v1/{condition}.md" for condition in CONDITIONS[1:]]
    fixture_rows = []
    for condition in CONDITIONS:
        episode_path = out_root / "offline_validation" / f"{condition}_fixture/episode.json"
        if episode_path.exists():
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            fixture_rows.append({
                "condition": condition,
                "status": episode.get("status"),
                **episode.get("reasoning_diagnostics", {}),
                "scientific_result": False,
            })
    write_json(out_root / "study_manifest.json", {
        "study": "qwen-agentic-reasoning-ablation-B0-B4",
        "version": "reasoning-ablation-v1",
        "branch": git_value("branch", "--show-current"),
        "implementation_commit": git_value("rev-parse", "HEAD"),
        "frozen_baseline_commit": None,
        "frozen_baseline_tag": None,
        "baseline_provenance_note": (
            "Original baseline references were retired during publication sanitization. "
            "implementation_commit and input_hashes identify these public inputs; "
            "historical results do not qualify the sanitized deployment."
        ),
        "live_episode_status": "PENDING_SEPARATE_AUTHORIZATION",
        "live_order": list(CONDITIONS),
        "profiles": profiles,
        "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in tracked_inputs},
        "safety": {"maximum_actions_per_episode": 1, "authorization_reuse": False,
                   "token_reuse": False, "live_actions_executed_by_builder": 0},
    })
    write_json(out_root / "offline_validation/fixture_summary.json", fixture_rows)
    live_rows = [{"condition": condition, "status": "Not executed - separate authorization required"}
                 for condition in CONDITIONS]
    write_json(out_root / "comparisons/live_comparison.json", live_rows)
    with (out_root / "comparisons/live_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "status"])
        writer.writeheader()
        writer.writerows(live_rows)
    (out_root / "comparisons/live_comparison.md").write_text(
        "| Condition | Status |\n|---|---|\n" +
        "".join(f"| {row['condition']} | {row['status']} |\n" for row in live_rows),
        encoding="utf-8",
    )
    (out_root / "study_protocol.md").write_text(
        "# Reasoning ablation v1\n\n"
        "B0 is the frozen baseline. B1 adds an evidence ledger; B2 adds hypothesis "
        "verification; B3 adds pre-action review; B4 adds information-gain selection. "
        "Conditions are cumulative. Each live condition requires a separate authorization "
        "and fresh action token and is executed once in B0-B4 order. Fixture outputs are "
        "engineering validation only and must not be reported as scientific outcomes. "
        "The public inputs were sanitized; historical deployment evidence must be "
        "revalidated before reuse.\n",
        encoding="utf-8",
    )
    (out_root / "reports/implementation_and_readiness.md").write_text(
        "# Implementation and readiness\n\n"
        "The cumulative reasoning profiles, model-authored state tool, diagnostics, and "
        "offline fixtures are implemented. Existing evaluator and action safety paths are "
        "unchanged. Live B0-B4 episodes remain pending separate per-condition authorization. "
        "Consult `offline_validation/deployed_read_only_status.json` for current deployment "
        "reachability; readiness must be re-established immediately before each episode.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.out_root)


if __name__ == "__main__":
    main()
