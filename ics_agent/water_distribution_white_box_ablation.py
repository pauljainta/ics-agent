from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Any

from . import water_distribution_seeded_backbone as seeded_backbone
from .config import resolve_repo_path
from .environment import OpenPLCEnvironment, _SphereOpenPLCBackend
from .mitre import verify_optionb_staged_snapshot
from .orchestrator import ExperimentOrchestrator
from .paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from .reporting import write_csv, write_json
from .utils import utc_now
from .water_distribution_seeded_backbone import (
    CONTAINMENT_SEMANTICS_VERSION,
    FamilySpec,
    SeededBackboneRunner,
    _append_log,
    _canonical_json_hash,
    _format_mean_std,
    _python_executable,
    _read_json,
    _sha256_path,
    _write_text,
)


TESTBED_ID = "water_distribution"
MODEL_LABEL = "Qwen3.5-9B-local-bf16"
EXPERIMENT_VERSION = "water_distribution_white_box_ablation_qwen35_9b_v1"
CANONICAL_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "water_distribution_openplc"
    / "White-Box-Ablation"
)
BASELINE_SEED_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "water_distribution_openplc"
    / "seeded_backbone_qwen35_9b"
    / "white_box_attacker__agentic_defender"
    / "seed_101"
)
SEEDS = [101]
RUNS_PER_SEED = 11
RUNS_PER_FAMILY = RUNS_PER_SEED
TOTAL_RUNS = 33
ABLATED_PROMPT_PATH = (
    REPO_ROOT / "testbeds" / "water_distribution" / "prompts" / "white_box_attacker_ablation.md"
)
DEFENDER_PROMPT_PATH = (
    REPO_ROOT / "testbeds" / "water_distribution" / "prompts" / "defender.md"
)
CONFIG_PATHS = [
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_white_box_no_safety_bounds_vs_agentic_defender.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_white_box_no_surface_priors_vs_agentic_defender.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_white_box_no_control_semantics_vs_agentic_defender.yaml",
]
FAMILY_SPECS = [
    FamilySpec(
        family_name="white_box_no_safety_bounds__agentic_defender",
        attacker_profile_id="white_box_no_safety_bounds_attacker",
        knowledge_variant="white_box_no_safety_bounds",
        config_path=CONFIG_PATHS[0],
    ),
    FamilySpec(
        family_name="white_box_no_surface_priors__agentic_defender",
        attacker_profile_id="white_box_no_surface_priors_attacker",
        knowledge_variant="white_box_no_surface_priors",
        config_path=CONFIG_PATHS[1],
    ),
    FamilySpec(
        family_name="white_box_no_control_semantics__agentic_defender",
        attacker_profile_id="white_box_no_control_semantics_attacker",
        knowledge_variant="white_box_no_control_semantics",
        config_path=CONFIG_PATHS[2],
    ),
]
FAMILY_BY_NAME = {item.family_name: item for item in FAMILY_SPECS}
FAMILY_ORDER = {item.family_name: index for index, item in enumerate(FAMILY_SPECS)}
ACTION_SPACE_SOURCE_PATHS = [
    REPO_ROOT / "ics_agent" / "action_semantics.py",
    REPO_ROOT / "ics_agent" / "environment.py",
    REPO_ROOT / "ics_agent" / "agents" / "action_space.py",
    REPO_ROOT / "ics_agent" / "agents" / "agentic.py",
    REPO_ROOT / "ics_agent" / "testbeds" / "water_distribution.py",
    REPO_ROOT
    / "testbeds"
    / "water_distribution"
    / "openplc"
    / "configs"
    / "openplc_map.yaml",
    REPO_ROOT / "configs" / "agent_profiles.yaml",
    REPO_ROOT / "configs" / "agent_profiles_overrides.yaml",
    *CONFIG_PATHS,
]
TOP_LEVEL_PHASES = [
    "manifest_frozen",
    "validated_distribution_environment",
    "validation_tests",
    "running_grid",
    "aggregating",
    "reporting",
]
VALIDATION_TEST_PATHS = [
    "tests/test_water_distribution.py",
    "tests/test_water_distribution_seeded_backbone.py",
    "tests/test_water_distribution_white_box_ablation.py",
]


seeded_backbone.EXPERIMENT_VERSION = EXPERIMENT_VERSION
seeded_backbone.CANONICAL_RESULT_ROOT = CANONICAL_RESULT_ROOT
seeded_backbone.SEEDS = SEEDS
seeded_backbone.SCENARIO_COUNT = RUNS_PER_SEED
seeded_backbone.RUNS_PER_FAMILY = RUNS_PER_FAMILY
seeded_backbone.TOTAL_RUNS = TOTAL_RUNS
seeded_backbone.FROZEN_PROMPT_PATHS = [
    ABLATED_PROMPT_PATH,
    ABLATED_PROMPT_PATH,
    ABLATED_PROMPT_PATH,
    DEFENDER_PROMPT_PATH,
]
seeded_backbone.FROZEN_CONFIG_PATHS = CONFIG_PATHS
seeded_backbone.FAMILY_SPECS = FAMILY_SPECS
seeded_backbone.FAMILY_BY_NAME = FAMILY_BY_NAME
seeded_backbone.FAMILY_ORDER = FAMILY_ORDER
seeded_backbone.ACTION_SPACE_SOURCE_PATHS = ACTION_SPACE_SOURCE_PATHS
seeded_backbone.TOP_LEVEL_PHASES = TOP_LEVEL_PHASES


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume the water-distribution white-box ablation pass."
    )
    subparsers = parser.add_subparsers(dest="mode")

    controller = subparsers.add_parser("controller")
    controller.add_argument(
        "--result-root",
        default=str(CANONICAL_RESULT_ROOT),
        help="White-box ablation result root.",
    )

    worker = subparsers.add_parser("worker")
    worker.add_argument("--family", required=True, choices=sorted(FAMILY_BY_NAME))
    worker.add_argument("--seed", required=True, type=int, choices=SEEDS)
    worker.add_argument(
        "--result-root",
        default=str(CANONICAL_RESULT_ROOT),
        help="White-box ablation result root.",
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


class WhiteBoxAblationRunner(SeededBackboneRunner):
    def __init__(self, result_root: Path) -> None:
        super().__init__(result_root)
        self.validate_log_path = self.progress_root / "validate_runtime.json"
        self.pytest_log_path = self.progress_root / "pytest_targeted.json"
        self.report_path = self.result_root / "white_box_ablation_report.md"
        self.comparison_csv_path = self.result_root / "baseline_comparison_seed_101.csv"
        self.comparison_json_path = self.result_root / "baseline_comparison_seed_101.json"
        self.comparison_md_path = self.result_root / "baseline_comparison_seed_101.md"
        self.repo_note_path = self.result_root / "white_box_ablation_note.md"

    def _prompt_hash_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        agent_profiles_path = REPO_ROOT / "configs" / "agent_profiles.yaml"
        profiles_overrides_path = REPO_ROOT / "configs" / "agent_profiles_overrides.yaml"
        tracked_paths = [
            ABLATED_PROMPT_PATH,
            DEFENDER_PROMPT_PATH,
            agent_profiles_path,
            profiles_overrides_path,
            *CONFIG_PATHS,
        ]
        file_hashes = {
            portable_path(path): _sha256_path(path)
            for path in tracked_paths
            if path.exists()
        }
        family_hashes: dict[str, dict[str, Any]] = {}
        for family in FAMILY_SPECS:
            config = self._runtime_config(family, SEEDS[0])
            condition_spec = self._condition_spec(orchestrator, family, config)
            attacker_profile = dict(
                (condition_spec.get("effective_profiles") or {}).get("attacker") or {}
            )
            manual_hashes = {}
            for raw_path in attacker_profile.get("knowledge_files", []):
                resolved = resolve_repo_path(raw_path, base_dir=REPO_ROOT)
                manual_hashes[portable_path(resolved)] = _sha256_path(resolved)
            bundle = {
                "attacker_prompt": file_hashes[portable_path(ABLATED_PROMPT_PATH)],
                "defender_prompt": file_hashes[portable_path(DEFENDER_PROMPT_PATH)],
                "agent_profiles": file_hashes[portable_path(agent_profiles_path)],
                "agent_profiles_overrides": file_hashes.get(
                    portable_path(profiles_overrides_path)
                ),
                "experiment_config": file_hashes[portable_path(family.config_path)],
                "knowledge_files": manual_hashes,
            }
            family_hashes[family.family_name] = {
                "bundle_hash": _canonical_json_hash(bundle),
                "bundle": bundle,
            }
            file_hashes.update(manual_hashes)
        return {
            "experiment_version": EXPERIMENT_VERSION,
            "file_hashes": file_hashes,
            "family_prompt_hashes": family_hashes,
        }

    def _write_resume_notes(
        self,
        queue_rows: list[dict[str, Any]],
        next_run: dict[str, Any] | None,
    ) -> None:
        completed_count = sum(1 for row in queue_rows if row["status"] == "completed")
        failed_count = sum(1 for row in queue_rows if row["status"] == "failed")
        pending_count = len(queue_rows) - completed_count - failed_count
        lines = [
            "# Resume Notes",
            "",
            f"- Updated UTC: `{utc_now()}`",
            f"- Experiment version: `{EXPERIMENT_VERSION}`",
            f"- Result root: `{portable_path(self.result_root)}`",
            f"- Baseline root reused: `{portable_path(BASELINE_SEED_ROOT)}`",
            f"- Completed runs: `{completed_count}`",
            f"- Failed runs: `{failed_count}`",
            f"- Pending runs: `{pending_count}`",
            "- Paper repo touched: `false`",
        ]
        if next_run:
            lines.extend(
                [
                    "",
                    "## Next Run",
                    f"- Family: `{next_run['attacker_family']}`",
                    f"- Seed: `{next_run['seed']}`",
                    f"- Scenario: `{next_run['scenario_id']}`",
                    f"- Split: `{next_run['split']}`",
                ]
            )
        _write_text(self.resume_notes_path, "\n".join(lines) + "\n")

    def _run_logged(
        self,
        command: list[str],
        log_path: Path,
        *,
        cwd: Path = REPO_ROOT,
    ) -> None:
        env = os.environ.copy()
        if len(command) >= 3 and command[1:3] == ["-m", "pytest"]:
            env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        started_utc = utc_now()
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=cwd,
            env=env,
        )
        write_json(
            log_path,
            {
                "cwd": portable_path(cwd),
                "command": command,
                "returncode": completed.returncode,
                "started_utc": started_utc,
                "completed_utc": utc_now(),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"command failed with exit code {completed.returncode}: {' '.join(command)}"
            )

    def _validate_runtime_and_tests(self) -> None:
        payload = _read_json(self.validate_log_path, {})
        pytest_payload = _read_json(self.pytest_log_path, {})
        if payload.get("returncode") == 0 and pytest_payload.get("returncode") == 0:
            self._set_phase("validation_tests", "completed")
            return
        self._set_phase("validation_tests", "in_progress")
        python_exec = _python_executable()
        self._run_logged(
            [python_exec, "-m", "ics_agent.cli", "validate"],
            self.validate_log_path,
        )
        self._run_logged(
            [python_exec, "-m", "pytest", *VALIDATION_TEST_PATHS],
            self.pytest_log_path,
        )
        self._set_phase(
            "validation_tests",
            "completed",
            validate_log=portable_path(self.validate_log_path),
            pytest_log=portable_path(self.pytest_log_path),
        )

    def _validate_distribution_environment(self) -> None:
        payload = _read_json(self.environment_validation_path, {})
        if (
            payload.get("status") == "passed"
            and (payload.get("initial_state_validation") or {}).get("status") == "passed"
        ):
            self._set_phase("validated_distribution_environment", "completed")
            return
        self._set_phase("validated_distribution_environment", "in_progress")
        config = self._runtime_config(FAMILY_SPECS[0], SEEDS[0])
        scenarios = self.scenarios()
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
                raise RuntimeError(
                    "expected sphere_openplc backend for water distribution"
                )
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
                    "reset_state": {
                        "state_name": reset_snapshot.get("STATE_NAME")
                        or reset_snapshot.get("state_name"),
                    },
                    "initial_state_validation": reset_snapshot.get(
                        "initial_state_validation"
                    ),
                    "post_reset_observation": reset_snapshot.get(
                        "post_reset_observation"
                    ),
                    "local_observation": {
                        "state_name": observation.get("STATE_NAME"),
                        "Supply_Tank_Level": observation.get("Supply_Tank_Level"),
                        "Grid_Elev_Tank_Level": observation.get("Grid_Elev_Tank_Level"),
                        "RWS_Tank_Level": observation.get("RWS_Tank_Level"),
                    },
                    "discovery_surface_sample": discovery.get("reconnaissance", {}),
                }
            )
            write_json(self.environment_validation_path, result)
            self._set_phase(
                "validated_distribution_environment",
                "completed",
                validation_path=portable_path(self.environment_validation_path),
            )
            return
        except Exception as exc:
            result["error"] = str(exc)
            write_json(self.environment_validation_path, result)
            raise
        finally:
            try:
                env.stop()
            except Exception:
                pass

    def run_controller(self) -> int:
        self.freeze_metadata()
        self._mark_resume_event_if_needed(
            list(csv.DictReader(self.run_queue_path.open(newline="")))
            if self.run_queue_path.exists()
            else []
        )
        self._validate_distribution_environment()
        self._validate_runtime_and_tests()
        self._set_phase("running_grid", "in_progress")
        _append_log(
            self.experiment_log_path,
            f"controller starting white-box ablation under {portable_path(self.result_root)}",
        )

        for family in FAMILY_SPECS:
            for seed in SEEDS:
                seed_status = _read_json(
                    self._seed_status_path(family.family_name, seed),
                    {},
                )
                if seed_status.get("completed_scenarios") == RUNS_PER_SEED:
                    continue
                worker_log_path = (
                    self.seed_progress_root(family.family_name, seed) / "worker_run.json"
                )
                started_utc = utc_now()
                try:
                    returncode = self.run_worker(family.family_name, seed, max_attempts=2)
                    write_json(
                        worker_log_path,
                        {
                            "mode": "in_process",
                            "family": family.family_name,
                            "seed": seed,
                            "started_utc": started_utc,
                            "completed_utc": utc_now(),
                            "returncode": returncode,
                        },
                    )
                except Exception as exc:
                    write_json(
                        worker_log_path,
                        {
                            "mode": "in_process",
                            "family": family.family_name,
                            "seed": seed,
                            "started_utc": started_utc,
                            "completed_utc": utc_now(),
                            "returncode": 1,
                            "error": str(exc),
                        },
                    )
                    self._sync_all_ledgers()
                    self._set_phase(
                        "running_grid",
                        "in_progress",
                        last_failure_family=family.family_name,
                        last_failure_seed=seed,
                    )
                    _append_log(
                        self.experiment_log_path,
                        f"worker failed for {family.family_name} seed {seed}: {exc}",
                    )
                    return 1
                self._sync_all_ledgers()
                if returncode != 0:
                    self._set_phase(
                        "running_grid",
                        "in_progress",
                        last_failure_family=family.family_name,
                        last_failure_seed=seed,
                    )
                    _append_log(
                        self.experiment_log_path,
                        f"worker failed for {family.family_name} seed {seed}",
                    )
                    return returncode
                _append_log(
                    self.experiment_log_path,
                    f"worker completed for {family.family_name} seed {seed}",
                )

        self._sync_all_ledgers()
        resume_state = _read_json(self.resume_state_path, {})
        if resume_state.get("completed_runs") != TOTAL_RUNS:
            raise RuntimeError(
                f"expected {TOTAL_RUNS} completed runs, found {resume_state.get('completed_runs')}"
            )
        self._set_phase("running_grid", "completed", completed_runs=TOTAL_RUNS)
        self.write_family_aggregates()
        self._set_phase("reporting", "in_progress")
        self.write_comparison_artifacts()
        self.write_report()
        self.write_repo_note()
        self._set_phase("reporting", "completed")
        return 0

    def _comparison_rows(self) -> list[dict[str, Any]]:
        baseline_summary = dict(
            _read_json(BASELINE_SEED_ROOT / "seed_summary.json", {}).get("summary_row")
            or {}
        )
        family_summaries = {
            family.family_name: dict(
                _read_json(
                    self.seed_root(family.family_name, SEEDS[0]) / "seed_summary.json",
                    {},
                ).get("summary_row")
                or {}
            )
            for family in FAMILY_SPECS
        }
        return [
            {
                "metric": "Attack scenarios",
                "baseline_white_box_seed_101": RUNS_PER_SEED,
                "white_box_no_safety_bounds": RUNS_PER_SEED,
                "white_box_no_surface_priors": RUNS_PER_SEED,
                "white_box_no_control_semantics": RUNS_PER_SEED,
            },
            {
                "metric": "Detected scenarios",
                "baseline_white_box_seed_101": baseline_summary.get("detected_scenarios"),
                "white_box_no_safety_bounds": family_summaries[
                    "white_box_no_safety_bounds__agentic_defender"
                ].get("detected_scenarios"),
                "white_box_no_surface_priors": family_summaries[
                    "white_box_no_surface_priors__agentic_defender"
                ].get("detected_scenarios"),
                "white_box_no_control_semantics": family_summaries[
                    "white_box_no_control_semantics__agentic_defender"
                ].get("detected_scenarios"),
            },
            {
                "metric": "Defender-action scenarios",
                "baseline_white_box_seed_101": baseline_summary.get(
                    "defender_action_scenarios"
                ),
                "white_box_no_safety_bounds": family_summaries[
                    "white_box_no_safety_bounds__agentic_defender"
                ].get("defender_action_scenarios"),
                "white_box_no_surface_priors": family_summaries[
                    "white_box_no_surface_priors__agentic_defender"
                ].get("defender_action_scenarios"),
                "white_box_no_control_semantics": family_summaries[
                    "white_box_no_control_semantics__agentic_defender"
                ].get("defender_action_scenarios"),
            },
            {
                "metric": "Safety-violation scenarios",
                "baseline_white_box_seed_101": baseline_summary.get(
                    "safety_violation_scenarios"
                ),
                "white_box_no_safety_bounds": family_summaries[
                    "white_box_no_safety_bounds__agentic_defender"
                ].get("safety_violation_scenarios"),
                "white_box_no_surface_priors": family_summaries[
                    "white_box_no_surface_priors__agentic_defender"
                ].get("safety_violation_scenarios"),
                "white_box_no_control_semantics": family_summaries[
                    "white_box_no_control_semantics__agentic_defender"
                ].get("safety_violation_scenarios"),
            },
            {
                "metric": "Average wall-clock time to detection from attack start (sec)",
                "baseline_white_box_seed_101": baseline_summary.get(
                    "avg_wall_clock_time_to_detection_from_attack_start_sec"
                ),
                "white_box_no_safety_bounds": family_summaries[
                    "white_box_no_safety_bounds__agentic_defender"
                ].get("avg_wall_clock_time_to_detection_from_attack_start_sec"),
                "white_box_no_surface_priors": family_summaries[
                    "white_box_no_surface_priors__agentic_defender"
                ].get("avg_wall_clock_time_to_detection_from_attack_start_sec"),
                "white_box_no_control_semantics": family_summaries[
                    "white_box_no_control_semantics__agentic_defender"
                ].get("avg_wall_clock_time_to_detection_from_attack_start_sec"),
            },
            {
                "metric": "Average attacker action count",
                "baseline_white_box_seed_101": baseline_summary.get(
                    "avg_attacker_action_count"
                ),
                "white_box_no_safety_bounds": family_summaries[
                    "white_box_no_safety_bounds__agentic_defender"
                ].get("avg_attacker_action_count"),
                "white_box_no_surface_priors": family_summaries[
                    "white_box_no_surface_priors__agentic_defender"
                ].get("avg_attacker_action_count"),
                "white_box_no_control_semantics": family_summaries[
                    "white_box_no_control_semantics__agentic_defender"
                ].get("avg_attacker_action_count"),
            },
            {
                "metric": "Average downtime burden (timesteps)",
                "baseline_white_box_seed_101": baseline_summary.get(
                    "avg_downtime_timestep_count"
                ),
                "white_box_no_safety_bounds": family_summaries[
                    "white_box_no_safety_bounds__agentic_defender"
                ].get("avg_downtime_timestep_count"),
                "white_box_no_surface_priors": family_summaries[
                    "white_box_no_surface_priors__agentic_defender"
                ].get("avg_downtime_timestep_count"),
                "white_box_no_control_semantics": family_summaries[
                    "white_box_no_control_semantics__agentic_defender"
                ].get("avg_downtime_timestep_count"),
            },
        ]

    def write_comparison_artifacts(self) -> None:
        rows = self._comparison_rows()
        write_csv(self.comparison_csv_path, rows)
        write_json(
            self.comparison_json_path,
            {
                "generated_at_utc": utc_now(),
                "experiment_version": EXPERIMENT_VERSION,
                "baseline_seed_root": portable_path(BASELINE_SEED_ROOT),
                "rows": rows,
            },
        )
        header = (
            "| Metric | Baseline white-box seed 101 | No safety bounds | No surface priors | No control semantics |\n"
            "| --- | ---: | ---: | ---: | ---: |\n"
        )
        body = "\n".join(
            "| {metric} | {baseline_white_box_seed_101} | {white_box_no_safety_bounds} | {white_box_no_surface_priors} | {white_box_no_control_semantics} |".format(
                **row
            )
            for row in rows
        )
        _write_text(
            self.comparison_md_path,
            "\n".join(
                [
                    "# Seed-101 Baseline Comparison",
                    "",
                    f"- Baseline root: `{portable_path(BASELINE_SEED_ROOT)}`",
                    f"- Ablation result root: `{portable_path(self.result_root)}`",
                    "",
                    header + body,
                    "",
                ]
            ),
        )

    def write_report(self) -> None:
        rows = {row["metric"]: row for row in self._comparison_rows()}
        family_roots = [
            portable_path(self.family_root(family.family_name)) for family in FAMILY_SPECS
        ]
        lines = [
            "# Water-Distribution White-Box Ablation Report",
            "",
            f"- Full white-box seed-101 baseline source: `{portable_path(BASELINE_SEED_ROOT)}`",
            "- Ablations run:",
            "  - `white_box_no_safety_bounds__agentic_defender`: explicit unsafe thresholds and published safety limits removed.",
            "  - `white_box_no_surface_priors__agentic_defender`: direct leverage ranking and scenario-level target hints removed.",
            "  - `white_box_no_control_semantics__agentic_defender`: operational meaning for named signals, commands, actuators, and interlocks removed.",
            f"- Seed used: `{SEEDS[0]}` only.",
            f"- Distribution suite size: `{RUNS_PER_SEED}` attack scenarios; suite remained fixed.",
            f"- Per-condition result roots: `{', '.join(family_roots)}`",
            f"- Comparison artifacts: `{portable_path(self.comparison_csv_path)}`, `{portable_path(self.comparison_json_path)}`, `{portable_path(self.comparison_md_path)}`",
            "",
            "## Side-by-Side Metrics",
            "",
            f"- Safety-violation scenarios: baseline `{rows['Safety-violation scenarios']['baseline_white_box_seed_101']}`, no safety bounds `{rows['Safety-violation scenarios']['white_box_no_safety_bounds']}`, no surface priors `{rows['Safety-violation scenarios']['white_box_no_surface_priors']}`, no control semantics `{rows['Safety-violation scenarios']['white_box_no_control_semantics']}`.",
            f"- Detected scenarios: baseline `{rows['Detected scenarios']['baseline_white_box_seed_101']}`, no safety bounds `{rows['Detected scenarios']['white_box_no_safety_bounds']}`, no surface priors `{rows['Detected scenarios']['white_box_no_surface_priors']}`, no control semantics `{rows['Detected scenarios']['white_box_no_control_semantics']}`.",
            f"- Defender-action scenarios: baseline `{rows['Defender-action scenarios']['baseline_white_box_seed_101']}`, no safety bounds `{rows['Defender-action scenarios']['white_box_no_safety_bounds']}`, no surface priors `{rows['Defender-action scenarios']['white_box_no_surface_priors']}`, no control semantics `{rows['Defender-action scenarios']['white_box_no_control_semantics']}`.",
            f"- Average attacker action count: baseline `{rows['Average attacker action count']['baseline_white_box_seed_101']}`, no safety bounds `{rows['Average attacker action count']['white_box_no_safety_bounds']}`, no surface priors `{rows['Average attacker action count']['white_box_no_surface_priors']}`, no control semantics `{rows['Average attacker action count']['white_box_no_control_semantics']}`.",
            f"- Average downtime burden: baseline `{rows['Average downtime burden (timesteps)']['baseline_white_box_seed_101']}`, no safety bounds `{rows['Average downtime burden (timesteps)']['white_box_no_safety_bounds']}`, no surface priors `{rows['Average downtime burden (timesteps)']['white_box_no_surface_priors']}`, no control semantics `{rows['Average downtime burden (timesteps)']['white_box_no_control_semantics']}`.",
            "",
            "- No paper files were changed in this pass.",
            "- No secrets or credentials were committed.",
        ]
        _write_text(self.report_path, "\n".join(lines) + "\n")

    def write_repo_note(self) -> None:
        resume_state = _read_json(self.resume_state_path, {})
        lines = [
            "# White-Box Ablation Note",
            "",
            f"- Experiment version: `{EXPERIMENT_VERSION}`",
            f"- Result root: `{portable_path(self.result_root)}`",
            f"- Baseline root: `{portable_path(BASELINE_SEED_ROOT)}`",
            f"- Configs: `{', '.join(portable_path(path) for path in CONFIG_PATHS)}`",
            f"- Prompt: `{portable_path(ABLATED_PROMPT_PATH)}`",
            f"- Seed: `{SEEDS[0]}`",
            f"- Resume event occurred: `{'true' if resume_state.get('resume_event_detected') else 'false'}`",
            f"- All {TOTAL_RUNS} runs completed: `{'true' if resume_state.get('completed_runs') == TOTAL_RUNS else 'false'}`",
            f"- Report: `{portable_path(self.report_path)}`",
            "- Paper repo touched: `false`",
            "- No secrets or credentials were committed.",
        ]
        _write_text(self.repo_note_path, "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runner = WhiteBoxAblationRunner(Path(args.result_root))
    if args.mode == "worker":
        return runner.run_worker(
            args.family,
            int(args.seed),
            max_attempts=int(args.max_attempts),
        )
    return runner.run_controller()


if __name__ == "__main__":
    raise SystemExit(main())
