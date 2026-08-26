from __future__ import annotations

import argparse
from pathlib import Path

from ...environment import OpenPLCEnvironment, _SphereOpenPLCBackend
from ...paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from .scripted_black_box_baseline import (
    ScriptedBlackBoxBaselineRunner,
)
from .seeded_backbone import FamilySpec, PROFILE_REGISTRY_PATHS


EXPERIMENT_VERSION = "water_treatment_scripted_white_box_baseline_v1"
CANONICAL_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "Water-treatment"
    / "scripted_baselines"
    / "white_box_attacker_vs_full_containment_defender"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_scripted_white_box_attacker_vs_agentic_full_containment_defender.yaml"
)
FAMILY_SPEC = FamilySpec(
    family_name="scripted_white_box_attacker__agentic_defender",
    attacker_profile_id="scripted_white_box_treatment_attacker",
    knowledge_variant="white_box_scripted",
    config_path=CONFIG_PATH,
)
REPO_LOCAL_REPORT_PATH = (
    REPO_ROOT
    / "testbeds"
    / "water_treatment"
    / "docs"
    / "2026-08-20-scripted-white-box-baseline-pass.md"
)
RESUME_COMPATIBLE_CONFIG_HASH_PATHS = {
    portable_path(REPO_ROOT / "ics_agent" / "experiments" / "water_treatment" / "scripted_white_box_baseline.py"),
}
VALIDATION_TEST_PATHS = [
    "tests/test_agentic.py",
    "tests/test_orchestrator.py",
    "tests/test_metrics.py",
    "tests/test_seeded_backbone.py",
    "tests/test_phase35_profiles.py",
    "tests/test_defender_authority_ablation.py",
    "tests/test_white_box_ablation.py",
    "tests/test_scripted_white_box_baseline.py",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume the water-treatment scripted white-box baseline."
    )
    subparsers = parser.add_subparsers(dest="mode")

    controller = subparsers.add_parser("controller")
    controller.add_argument(
        "--result-root",
        default=str(CANONICAL_RESULT_ROOT),
        help="Scripted white-box baseline result root.",
    )

    worker = subparsers.add_parser("worker")
    worker.add_argument(
        "--result-root",
        default=str(CANONICAL_RESULT_ROOT),
        help="Scripted white-box baseline result root.",
    )
    worker.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum attempts for one scenario before surfacing a failure.",
    )

    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode = "controller"
        args.result_root = str(CANONICAL_RESULT_ROOT)
    return args


class ScriptedWhiteBoxBaselineRunner(ScriptedBlackBoxBaselineRunner):
    experiment_version = EXPERIMENT_VERSION
    config_path = CONFIG_PATH
    family_spec = FAMILY_SPEC
    runtime_policy = "deterministic_scripted_white_box"
    run_id_policy_token = "scriptedWhiteBox"
    controller_run_label = "scripted white-box baseline"
    validation_test_paths = VALIDATION_TEST_PATHS
    resume_compatible_config_hash_paths = RESUME_COMPATIBLE_CONFIG_HASH_PATHS
    report_filename = "scripted_white_box_baseline_report.md"
    repo_note_filename = "scripted_white_box_baseline_note.md"
    repo_local_report_path = REPO_LOCAL_REPORT_PATH

    def _optionb_environment_cls(self):
        return OpenPLCEnvironment

    def _optionb_backend_cls(self):
        return _SphereOpenPLCBackend

    def _config_hash_tracked_paths(self) -> list[Path]:
        return [
            *PROFILE_REGISTRY_PATHS,
            self.config_path,
            REPO_ROOT / "ics_agent" / "agents" / "agentic.py",
            REPO_ROOT / "ics_agent" / "experiments" / "water_treatment" / "scripted_white_box_baseline.py",
            REPO_ROOT / "testbeds" / "water_treatment" / "prompts" / "white_box_attacker.md",
            REPO_ROOT / "testbeds" / "water_treatment" / "prompts" / "defender.md",
            REPO_ROOT / "testbeds" / "water_treatment" / "prompts" / "observer.md",
        ]

    def _report_lines(self) -> list[str]:
        lines = super()._report_lines()
        lines[0] = "# Scripted White-Box Baseline Report"
        lines[4] = (
            f"- Attacker: `{self.family_spec.attacker_profile_id}` deterministic scripted white-box baseline"
        )
        notes_index = lines.index("## Notes")
        lines[notes_index + 1] = (
            "- The attacker used deterministic fixed logic over the bounded white-box runtime interface."
        )
        return lines

    def _repo_note_lines(self, resume_state: dict[str, object]) -> list[str]:
        return [
            "# Scripted White-Box Baseline Note",
            "",
            f"- Experiment version: `{self.experiment_version}`",
            f"- Result root: `{portable_path(self.result_root)}`",
            f"- Seed root: `{portable_path(self.seed_root)}`",
            f"- Config: `{portable_path(self.config_path)}`",
            f"- Attacker profile: `{self.family_spec.attacker_profile_id}`",
            f"- Seed: `{self.seed}`",
            f"- Resume event occurred: `{'true' if resume_state.get('resume_event_detected') else 'false'}`",
            f"- Scenarios rerun after interruption: `{'true' if resume_state.get('scenarios_rerun_after_interruption') else 'false'}`",
            f"- All {self.runs_per_seed} runs completed: `{'true' if resume_state.get('completed_runs') == self.runs_per_seed else 'false'}`",
            "- The attacker baseline is deterministic and does not invoke an LLM at runtime.",
            "- Defender and observer remained the current agentic full-containment stack.",
            "- Water-distribution execution is handled by the separate distribution scripted white-box runner.",
        ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runner = ScriptedWhiteBoxBaselineRunner(Path(args.result_root))
    if args.mode == "worker":
        return runner.run_worker(max_attempts=int(args.max_attempts))
    return runner.run_controller()


if __name__ == "__main__":
    raise SystemExit(main())
