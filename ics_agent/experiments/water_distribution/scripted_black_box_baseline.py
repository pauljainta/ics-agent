from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from ...config import load_config, require_fields
from ...environment import OpenPLCEnvironment, _SphereOpenPLCBackend
from ...mitre import verify_optionb_staged_snapshot
from ...orchestrator import ExperimentOrchestrator
from ...paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from ...reporting import write_csv, write_json
from ...utils import dump_yaml, load_yaml, utc_now
from .seeded_backbone import (
    ACTION_SPACE_SOURCE_PATHS,
    CONTAINMENT_SEMANTICS_VERSION,
    LEGACY_FAMILY_ROOTS,
    PROFILE_REGISTRY_PATHS,
    REQUIRED_RUN_FILES,
    TESTBED_ID,
    FamilySpec,
    _append_log,
    _attack_family_for,
    _canonical_json_hash,
    _format_mean_std,
    _infer_logger_schema,
    _infer_metrics_schema,
    _mean,
    _normalize_csv_rows,
    _python_executable,
    _read_json,
    _sanitize_token,
    _scenario_family_for,
    _scenario_split_map,
    _sha256_path,
    _stdev,
    _write_text,
    build_manifest_rows,
    build_split_rows,
)


EXPERIMENT_VERSION = "water_distribution_scripted_black_box_baseline_v1"
CANONICAL_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "water_distribution_openplc"
    / "scripted_baselines"
    / "black_box_attacker_vs_full_containment_defender"
)
SEED = 101
RUNS_PER_SEED = 11
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_scripted_black_box_vs_agentic_defender.yaml"
)
FAMILY_SPEC = FamilySpec(
    family_name="scripted_black_box_attacker__agentic_defender",
    attacker_profile_id="scripted_black_box_distribution_attacker",
    knowledge_variant="black_box_scripted",
    config_path=CONFIG_PATH,
)
REPO_LOCAL_REPORT_PATH = (
    REPO_ROOT
    / "testbeds"
    / "water_distribution"
    / "docs"
    / "2026-08-20-scripted-black-box-baseline-pass.md"
)
RESUME_COMPATIBLE_CONFIG_HASH_PATHS = {
    portable_path(
        REPO_ROOT / "ics_agent" / "experiments" / "water_distribution" / "scripted_black_box_baseline.py"
    ),
}
TOP_LEVEL_PHASES = [
    "manifest_frozen",
    "validated_optionb_environment",
    "validation_tests",
    "scenario_run",
    "aggregating",
    "final_report",
]
VALIDATION_TEST_PATHS = [
    "tests/test_agentic.py",
    "tests/test_orchestrator.py",
    "tests/test_metrics.py",
    "tests/test_water_distribution.py",
    "tests/test_water_distribution_seeded_backbone.py",
    "tests/test_phase35_profiles.py",
    "tests/test_water_distribution_scripted_black_box_baseline.py",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume the water-distribution scripted black-box baseline."
    )
    subparsers = parser.add_subparsers(dest="mode")

    controller = subparsers.add_parser("controller")
    controller.add_argument(
        "--result-root",
        default=str(CANONICAL_RESULT_ROOT),
        help="Scripted black-box baseline result root.",
    )

    worker = subparsers.add_parser("worker")
    worker.add_argument(
        "--result-root",
        default=str(CANONICAL_RESULT_ROOT),
        help="Scripted black-box baseline result root.",
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


class ScriptedBlackBoxBaselineRunner:
    experiment_version = EXPERIMENT_VERSION
    config_path = CONFIG_PATH
    family_spec = FAMILY_SPEC
    seed = SEED
    runs_per_seed = RUNS_PER_SEED
    runtime_policy = "deterministic_scripted_black_box"
    run_id_policy_token = "scriptedBlackBox"
    controller_run_label = "scripted black-box baseline"
    validation_test_paths = VALIDATION_TEST_PATHS
    resume_compatible_config_hash_paths = RESUME_COMPATIBLE_CONFIG_HASH_PATHS
    report_filename = "scripted_black_box_baseline_report.md"
    repo_note_filename = "scripted_black_box_baseline_note.md"
    default_repo_local_report_path = REPO_LOCAL_REPORT_PATH
    preferred_xdc_short_names: tuple[str, ...] = ()
    candidate_xdc_prefix = ""
    optionb_validation_scenario_id = "wd-attack-coordinated-grid-rws-delay"
    validation_attempt_timeout_sec = 480.0

    def __init__(self, result_root: Path) -> None:
        self.result_root = result_root.resolve()
        self.seed_root = self.result_root / f"seed_{self.seed}"
        self.progress_root = self.result_root / "_progress"
        self.seed_progress_root = self.seed_root / "_progress"
        self.global_phase_status_path = self.progress_root / "global_phase_status.json"
        self.completed_runs_path = self.progress_root / "completed_runs.json"
        self.failed_runs_path = self.progress_root / "failed_runs.json"
        self.resume_state_path = self.progress_root / "resume_state.json"
        self.last_step_path = self.progress_root / "last_successful_step.txt"
        self.run_queue_path = self.progress_root / "run_queue.csv"
        self.experiment_log_path = self.progress_root / "experiment_run.log"
        self.resume_notes_path = self.progress_root / "resume_notes.md"
        self.optionb_validation_path = self.progress_root / "optionb_environment_validation.json"
        self.validate_log_path = self.progress_root / "validate_runtime.json"
        self.pytest_log_path = self.progress_root / "pytest_targeted.json"
        self.scenario_status_path = self.seed_progress_root / "scenario_status.csv"
        self.interrupted_root = self.seed_progress_root / "interrupted_artifacts"
        self.run_catalog_path = self.result_root / "run_catalog.csv"
        self.run_failures_path = self.result_root / "run_failures.csv"
        self.report_path = self.result_root / self.report_filename
        self.repo_note_path = self.result_root / self.repo_note_filename
        self.manifest_path = self.result_root / "scenario_manifest_water_distribution.csv"
        self.splits_path = self.result_root / "scenario_splits_water_distribution.csv"
        self.seeds_path = self.result_root / "seeds.txt"
        self.execution_registry_path = self.result_root / "execution_registry.yaml"
        self.experiment_version_path = self.result_root / "experiment_version.txt"
        self.config_hashes_path = self.result_root / "config_hashes.json"
        self.action_space_hashes_path = self.result_root / "action_space_hashes.json"
        self.containment_semantics_path = (
            self.result_root / "containment_semantics_version.txt"
        )
        self.logger_schema_path = self.result_root / "logger_schema.json"
        self.metrics_schema_path = self.result_root / "metrics_schema.json"
        self.condition_summary_path = self.result_root / "condition_summary.json"
        self.attack_metrics_path = self.result_root / "attack_metrics_summary.json"
        self.trajectory_metrics_path = self.result_root / "trajectory_metrics_summary.json"
        self.mitre_phase_metrics_path = self.result_root / "mitre_phase_metrics.json"
        self.nominal_metrics_path = self.result_root / "nominal_metrics_summary.json"
        self.aggregate_json_path = self.result_root / "aggregate_over_seed_101.json"
        self.aggregate_csv_path = self.result_root / "aggregate_over_seed_101.csv"
        self.repo_local_report_path = self.default_repo_local_report_path

    def ensure_root_layout(self) -> None:
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.progress_root.mkdir(parents=True, exist_ok=True)
        self.seed_root.mkdir(parents=True, exist_ok=True)
        self.seed_progress_root.mkdir(parents=True, exist_ok=True)
        self.interrupted_root.mkdir(parents=True, exist_ok=True)

    def _orchestrator(self) -> ExperimentOrchestrator:
        return ExperimentOrchestrator(testbed_id=TESTBED_ID)

    def scenarios(self) -> list[Any]:
        return self._orchestrator().catalog.resolve_selectors(["attack/*"])

    def manifest_rows(self) -> list[dict[str, Any]]:
        rows = build_manifest_rows(self.scenarios())
        enabled = [row for row in rows if row.get("enabled_for_final") == "true"]
        if len(enabled) != self.runs_per_seed:
            raise RuntimeError(
                f"expected exactly {self.runs_per_seed} enabled water-distribution scenarios, found {len(enabled)}"
            )
        return rows

    def split_rows(self) -> list[dict[str, Any]]:
        rows = build_split_rows(self.manifest_rows())
        if len(rows) != self.runs_per_seed:
            raise RuntimeError(
                f"expected {self.runs_per_seed} split rows, found {len(rows)}"
            )
        return rows

    def _base_config(self) -> dict[str, Any]:
        config = load_config(str(self.config_path))
        require_fields(
            config,
            ["experiment_name", "scenario_selectors"],
            ctx="experiment config",
        )
        return config

    def _runtime_config(self) -> dict[str, Any]:
        config = deepcopy(self._base_config())
        config.setdefault("reproducibility", {})
        config["reproducibility"]["seed"] = self.seed
        config["resume"] = True
        config["use_canonical_result_root"] = True
        config["testbed"] = TESTBED_ID
        selected_xdc = self._selected_xdc_short_name()
        if selected_xdc:
            config.setdefault("environment", {})
            config["environment"]["xdc"] = selected_xdc
        return config

    def _selected_xdc_short_name(self) -> str | None:
        override = str(os.environ.get("ICS_AGENT_SPHERE_XDC_OVERRIDE", "")).strip()
        if override:
            return override.split(".", 1)[0]
        payload = _read_json(self.optionb_validation_path, {})
        if str(payload.get("status", "")).strip().lower() != "passed":
            return None
        selected = dict(payload.get("selected_xdc") or {})
        short_name = str(selected.get("short_name", "")).strip()
        if short_name:
            return short_name
        validated_target = dict(payload.get("validated_target") or {})
        xdc_ref = str(validated_target.get("xdc", "")).strip()
        if not xdc_ref:
            return None
        return xdc_ref.split(".", 1)[0]

    @staticmethod
    def _required_preflight_refs(config: dict[str, Any]) -> dict[str, str]:
        environment = dict(config.get("environment") or {})
        experiment_ref = str(environment.get("experiment", "")).strip()
        realization_name = str(environment.get("realization", "")).strip()
        if not experiment_ref or "." not in experiment_ref:
            raise RuntimeError(
                "expected environment.experiment in `experiment.project` form"
            )
        experiment_name, project_name = experiment_ref.split(".", 1)
        if realization_name.count(".") == 2:
            realization_ref = realization_name
        elif "." in realization_name:
            raise RuntimeError(
                "expected environment.realization as a short name or full `realization.experiment.project` reference"
            )
        else:
            realization_ref = f"{realization_name}.{experiment_name}.{project_name}"
        return {
            "merge_server": str(environment.get("merge_server", "")).strip(),
            "realization": realization_ref,
            "materialization": realization_ref,
        }

    def _condition_spec(
        self,
        orchestrator: ExperimentOrchestrator,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        condition_specs = orchestrator._enumerate_conditions(config)
        if len(condition_specs) != 1:
            raise RuntimeError(f"expected exactly one condition in {self.config_path}")
        condition_spec = condition_specs[0]
        condition_name = orchestrator._condition_name(
            condition_spec["profiles"],
            override=condition_spec.get("name"),
        )
        if condition_name != self.family_spec.family_name:
            raise RuntimeError(
                f"expected condition `{self.family_spec.family_name}`, found `{condition_name}`"
            )
        return condition_spec

    def _blue_team_model_metadata(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        config = self._runtime_config()
        condition_spec = self._condition_spec(orchestrator, config)
        settings = orchestrator._collect_decoding_settings(
            condition_spec["profiles"],
            effective_profiles=condition_spec.get("effective_profiles"),
        )
        metadata: dict[str, Any] = {}
        for role in ("defender", "observer"):
            payload = dict(settings.get(role) or {})
            if not payload:
                continue
            metadata[role] = {
                "model_id": payload.get("model_id"),
                "model_path": payload.get("model_path"),
                "model_quantization": payload.get("quantization"),
                "context_limit": payload.get("context_limit"),
                "temperature": payload.get("temperature"),
                "top_p": payload.get("top_p"),
                "backend": payload.get("backend"),
                "framework": payload.get("framework"),
                "local_files_only": payload.get("local_files_only"),
                "require_cuda": payload.get("require_cuda"),
            }
        return metadata

    def _execution_registry_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        config = self._runtime_config()
        condition_spec = self._condition_spec(orchestrator, config)
        effective_profiles = condition_spec.get("effective_profiles") or {}
        attacker_profile = dict(effective_profiles.get("attacker") or {})
        return {
            "experiment_version": self.experiment_version,
            "result_root": portable_path(self.result_root),
            "seed_root": portable_path(self.seed_root),
            "testbed": TESTBED_ID,
            "seed": self.seed,
            "condition_name": self.family_spec.family_name,
            "config_path": portable_path(self.config_path),
            "attacker": {
                "profile_id": self.family_spec.attacker_profile_id,
                "family": attacker_profile.get("family"),
                "knowledge_mode": attacker_profile.get("knowledge_mode"),
                "behavior_model": attacker_profile.get("behavior_model"),
                "allowed_action_types": list(
                    attacker_profile.get("allowed_action_types", [])
                ),
                "runtime_policy": self.runtime_policy,
                "llm_invocation_required": False,
            },
            "blue_team_models": self._blue_team_model_metadata(),
            "planner_profile": condition_spec["profiles"].get("planner"),
            "operator_profile": condition_spec["profiles"].get("operator"),
            "offline_analysis_profile": condition_spec["profiles"].get(
                "offline_analysis"
            ),
        }

    def _config_hash_tracked_paths(self) -> list[Path]:
        return [
            *PROFILE_REGISTRY_PATHS,
            self.config_path,
            REPO_ROOT / "ics_agent" / "agents" / "agentic.py",
            REPO_ROOT / "ics_agent" / "experiments" / "water_distribution" / "scripted_black_box_baseline.py",
            REPO_ROOT / "testbeds" / "water_distribution" / "prompts" / "black_box_attacker.md",
            REPO_ROOT / "testbeds" / "water_distribution" / "prompts" / "defender.md",
            REPO_ROOT
            / "configs"
            / "experiments"
            / "water_distribution_black_box_vs_agentic_defender.yaml",
        ]

    def _config_hash_payload(self) -> dict[str, Any]:
        tracked_paths = self._config_hash_tracked_paths()
        file_hashes = {
            portable_path(path): _sha256_path(path)
            for path in tracked_paths
            if path.exists()
        }
        bundle = {
            "condition_name": self.family_spec.family_name,
            "experiment_version": self.experiment_version,
            "seed": self.seed,
            "file_hashes": file_hashes,
        }
        return {
            "experiment_version": self.experiment_version,
            "file_hashes": file_hashes,
            "bundle_hash": _canonical_json_hash(bundle),
        }

    def _action_space_hash_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        config = self._runtime_config()
        condition_spec = self._condition_spec(orchestrator, config)
        effective_profiles = condition_spec.get("effective_profiles") or {}
        source_hashes = {
            portable_path(path): _sha256_path(path)
            for path in [
                *ACTION_SPACE_SOURCE_PATHS,
                REPO_ROOT / "ics_agent" / "agents" / "agentic.py",
                self.config_path,
            ]
            if path.exists()
        }
        role_payload = {}
        for role, profile_id in condition_spec["profiles"].items():
            profile = dict(
                effective_profiles.get(role) or orchestrator.agent_profiles[profile_id]
            )
            role_payload[role] = {
                "profile_id": profile_id,
                "allowed_action_types": list(profile.get("allowed_action_types", [])),
                "behavior_model": profile.get("behavior_model"),
                "knowledge_mode": profile.get("knowledge_mode"),
            }
        bundle = {
            "runtime_source_hashes": source_hashes,
            "roles": role_payload,
            "containment_semantics_version": CONTAINMENT_SEMANTICS_VERSION,
        }
        return {
            "experiment_version": self.experiment_version,
            "source_hashes": source_hashes,
            "bundle_hash": _canonical_json_hash(bundle),
            "bundle": bundle,
        }

    def _containment_semantics_text(self) -> str:
        return "\n".join(
            [
                CONTAINMENT_SEMANTICS_VERSION,
                "",
                "Concrete runtime containment semantics frozen for this scripted baseline pass:",
                "- runtime-enforced surface isolation remains enabled",
                "- defender containment actions: block_sensor_write, block_actuator_write, disable_hmi_session, isolate_engineering_host, quarantine_signal_source, hmi_stop, hmi_estop",
                "- symbolic containment remains disabled",
                "- defender profile remains the current full agentic defender used in the attacker-knowledge comparison",
                "",
                "Primary source files:",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'action_semantics.py')}",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'environment.py')}",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'orchestrator.py')}",
                "",
            ]
        )

    def _expected_metadata_payloads(self) -> dict[Path, tuple[str, Any]]:
        return {
            self.manifest_path: ("csv", self.manifest_rows()),
            self.splits_path: ("csv", self.split_rows()),
            self.seeds_path: ("text", f"{self.seed}\n"),
            self.execution_registry_path: ("yaml", self._execution_registry_payload()),
            self.experiment_version_path: ("text", self.experiment_version + "\n"),
            self.config_hashes_path: ("json", self._config_hash_payload()),
            self.action_space_hashes_path: ("json", self._action_space_hash_payload()),
            self.containment_semantics_path: (
                "text",
                self._containment_semantics_text(),
            ),
            self.logger_schema_path: ("json", _infer_logger_schema(LEGACY_FAMILY_ROOTS)),
            self.metrics_schema_path: (
                "json",
                _infer_metrics_schema(LEGACY_FAMILY_ROOTS),
            ),
        }

    @classmethod
    def _is_resume_compatible_config_hash_update(
        cls,
        actual_payload: Any,
        expected_payload: Any,
    ) -> bool:
        if not isinstance(actual_payload, dict) or not isinstance(expected_payload, dict):
            return False
        actual_hashes = dict(actual_payload.get("file_hashes") or {})
        expected_hashes = dict(expected_payload.get("file_hashes") or {})
        if set(actual_hashes) != set(expected_hashes):
            return False
        changed_paths = {
            path
            for path, expected_hash in expected_hashes.items()
            if actual_hashes.get(path) != expected_hash
        }
        return bool(changed_paths) and changed_paths.issubset(
            cls.resume_compatible_config_hash_paths
        )

    def freeze_metadata(self) -> None:
        self.ensure_root_layout()
        has_progress = bool(_read_json(self.completed_runs_path, {}))
        expected_payloads = self._expected_metadata_payloads()
        for actual_path, (kind, payload) in expected_payloads.items():
            if kind == "csv":
                expected_rows = list(payload)
                if actual_path.exists():
                    with actual_path.open(newline="") as handle:
                        reader = csv.DictReader(handle)
                        actual_rows = list(reader)
                    if (
                        _normalize_csv_rows(actual_rows)
                        != _normalize_csv_rows(expected_rows)
                        and has_progress
                    ):
                        raise RuntimeError(
                            f"frozen metadata mismatch in {actual_path}; create a new experiment version before continuing"
                        )
                write_csv(actual_path, expected_rows)
            elif kind == "json":
                if actual_path.exists():
                    actual_payload = _read_json(actual_path, None)
                    if actual_payload != payload and has_progress:
                        if (
                            actual_path == self.config_hashes_path
                            and self._is_resume_compatible_config_hash_update(
                                actual_payload,
                                payload,
                            )
                        ):
                            _append_log(
                                self.experiment_log_path,
                                "updating config hashes after a resume-compatible runner-only bookkeeping change",
                            )
                        else:
                            raise RuntimeError(
                                f"frozen metadata mismatch in {actual_path}; create a new experiment version before continuing"
                            )
                write_json(actual_path, payload)
            elif kind == "yaml":
                if actual_path.exists():
                    actual_payload = load_yaml(actual_path)
                    if actual_payload != payload and has_progress:
                        raise RuntimeError(
                            f"frozen metadata mismatch in {actual_path}; create a new experiment version before continuing"
                        )
                dump_yaml(actual_path, payload)
            else:
                if actual_path.exists():
                    actual_text = actual_path.read_text()
                    if actual_text != payload and has_progress:
                        raise RuntimeError(
                            f"frozen metadata mismatch in {actual_path}; create a new experiment version before continuing"
                        )
                _write_text(actual_path, str(payload))
        if not self.run_catalog_path.exists():
            write_csv(self.run_catalog_path, [])
        if not self.run_failures_path.exists():
            write_csv(self.run_failures_path, [])
        self._write_queue_and_resume(self._load_scenario_rows())
        self._set_last_step("manifest_frozen")
        self._set_phase("manifest_frozen", "completed")

    def _set_last_step(self, step: str) -> None:
        _write_text(self.last_step_path, step + "\n")

    def _load_global_phase_status(self) -> dict[str, Any]:
        payload = _read_json(self.global_phase_status_path, {})
        if not isinstance(payload, dict):
            payload = {}
        phases = dict(payload.get("phases") or {})
        for phase in TOP_LEVEL_PHASES:
            phases.setdefault(phase, {"status": "pending"})
        payload["phases"] = phases
        return payload

    def _set_phase(self, phase: str, status: str, **fields: Any) -> None:
        payload = self._load_global_phase_status()
        phases = dict(payload.get("phases") or {})
        item = dict(phases.get(phase) or {})
        item["status"] = status
        item["updated_utc"] = utc_now()
        if status != "failed" and "error_message" not in fields:
            item.pop("error_message", None)
        if status == "in_progress" and "started_utc" not in item:
            item["started_utc"] = item["updated_utc"]
        if status == "completed":
            item["completed_utc"] = item["updated_utc"]
            self._set_last_step(phase)
        item.update(fields)
        phases[phase] = item
        payload["phases"] = phases
        payload["updated_utc"] = utc_now()
        write_json(self.global_phase_status_path, payload)

    def _initial_scenario_rows(self) -> list[dict[str, Any]]:
        split_map = _scenario_split_map(self.split_rows())
        rows: list[dict[str, Any]] = []
        for index, scenario in enumerate(self.scenarios()):
            rows.append(
                {
                    "manifest_order": index + 1,
                    "attacker_family": self.family_spec.family_name,
                    "seed": str(self.seed),
                    "scenario_id": scenario.scenario_id,
                    "catalog_id": scenario.catalog_id,
                    "split": split_map[scenario.scenario_id],
                    "status": "pending",
                    "attempt_count": "0",
                    "started_at_utc": "",
                    "completed_at_utc": "",
                    "failed_at_utc": "",
                    "last_error": "",
                    "run_id": "",
                    "run_dir": portable_path(self._run_dir(scenario)),
                    "archived_interrupted_dir": "",
                }
            )
        return rows

    def _load_scenario_rows(self) -> list[dict[str, Any]]:
        if not self.scenario_status_path.exists():
            rows = self._initial_scenario_rows()
            write_csv(self.scenario_status_path, rows)
            return rows
        with self.scenario_status_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != self.runs_per_seed:
            raise RuntimeError(
                f"expected {self.runs_per_seed} rows in {self.scenario_status_path}, found {len(rows)}"
            )
        return rows

    def _save_scenario_rows(self, rows: list[dict[str, Any]]) -> None:
        write_csv(self.scenario_status_path, rows)

    def _run_dir(self, scenario: Any) -> Path:
        return self.seed_root / scenario.suite / scenario.scenario_id / "trial-001"

    def _run_id(self, scenario: Any, split: str) -> str:
        return "_".join(
            [
                TESTBED_ID,
                _sanitize_token(scenario.scenario_id),
                split,
                self.run_id_policy_token,
                "fullAgenticDefender",
                f"seed{self.seed}",
            ]
        )

    @staticmethod
    def _expected_environment(config: dict[str, Any]) -> dict[str, Any]:
        return dict(config.get("environment") or {})

    def _verify_run_completion(
        self,
        run_dir: Path,
        *,
        expected_environment: dict[str, Any],
    ) -> bool:
        if not all((run_dir / name).exists() for name in REQUIRED_RUN_FILES):
            return False
        summary = _read_json(run_dir / "summary.json", {})
        if not isinstance(summary, dict) or summary.get("status") != "completed":
            return False
        environment_snapshot = _read_json(run_dir / "environment_snapshot.json", {})
        verification = verify_optionb_staged_snapshot(
            environment_snapshot,
            expected_environment=expected_environment,
        )
        return bool(verification.get("ok"))

    @staticmethod
    def _load_run_summary(run_dir: Path) -> dict[str, Any]:
        summary = _read_json(run_dir / "summary.json", {})
        metrics = _read_json(run_dir / "metrics.json", {})
        if isinstance(summary, dict) and "metrics" not in summary:
            summary["metrics"] = metrics
        return summary

    def _archive_partial_run(self, scenario_id: str, run_dir: Path) -> str:
        destination = self.interrupted_root / (
            f"{scenario_id}__{utc_now().replace(':', '').replace('-', '')}"
        )
        counter = 1
        while destination.exists():
            counter += 1
            destination = self.interrupted_root / f"{scenario_id}__{counter}"
        shutil.move(str(run_dir), str(destination))
        return portable_path(destination)

    def _completed_row(
        self,
        *,
        scenario: Any,
        run_dir: Path,
        summary: dict[str, Any],
        split: str,
        config_hashes: dict[str, Any],
        action_space_hashes: dict[str, Any],
        blue_team_models: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = dict(summary.get("metrics") or {})
        attack_metrics = dict(metrics.get("attacker") or {})
        return {
            "progress_key": f"{self.family_spec.family_name}::{self.seed}::{scenario.scenario_id}",
            "run_id": summary.get("run_id"),
            "experiment_version": self.experiment_version,
            "git_commit": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "timestamp_start_utc": summary.get("timestamp_start_utc"),
            "timestamp_end_utc": summary.get("timestamp_end_utc"),
            "testbed": TESTBED_ID,
            "scenario_id": scenario.scenario_id,
            "scenario_family": _scenario_family_for(scenario.scenario_id),
            "attack_family": _attack_family_for(scenario.scenario_id),
            "split": split,
            "seed": self.seed,
            "attacker_profile_id": self.family_spec.attacker_profile_id,
            "attacker_runtime_policy": self.runtime_policy,
            "defender_profile_id": "agentic_defender",
            "defender_model_id": (blue_team_models.get("defender") or {}).get(
                "model_id"
            ),
            "observer_model_id": (blue_team_models.get("observer") or {}).get(
                "model_id"
            ),
            "config_hash": config_hashes.get("bundle_hash"),
            "action_space_hash": action_space_hashes.get("bundle_hash"),
            "containment_semantics_version": CONTAINMENT_SEMANTICS_VERSION,
            "status": summary.get("status"),
            "error_message": "",
            "family_name": self.family_spec.family_name,
            "run_dir": portable_path(run_dir),
            "catalog_id": scenario.catalog_id,
            "detected": int(
                metrics.get("defender_detection", {}).get("time_to_detection_sec")
                is not None
            ),
            "defender_acted": int((metrics.get("defender_action_count") or 0) > 0),
            "safety_violation": int(
                bool(attack_metrics.get("caused_safety_violation"))
            ),
            "wall_clock_time_to_detection_from_attack_start_sec": metrics.get(
                "wall_clock_time_to_detection_from_attack_start_sec"
            ),
            "attacker_action_count": metrics.get("attacker_action_count"),
            "downtime_timestep_count": metrics.get("downtime_timestep_count"),
            "summary_path": portable_path(run_dir / "summary.json"),
            "metrics_path": portable_path(run_dir / "metrics.json"),
        }

    def _failure_row(
        self,
        *,
        scenario_id: str,
        split: str,
        run_dir: Path,
        error_message: str,
    ) -> dict[str, Any]:
        return {
            "progress_key": f"{self.family_spec.family_name}::{self.seed}::{scenario_id}",
            "attacker_family": self.family_spec.family_name,
            "seed": self.seed,
            "scenario_id": scenario_id,
            "split": split,
            "status": "failed",
            "error_message": error_message,
            "run_dir": portable_path(run_dir),
            "updated_utc": utc_now(),
        }

    def _write_seed_summaries(
        self,
        *,
        completed_rows: list[dict[str, Any]],
        run_summaries: list[dict[str, Any]],
        reproducibility: dict[str, Any],
    ) -> None:
        orchestrator = self._orchestrator()
        attack_metrics = orchestrator._aggregate_attack_metrics(
            run_summaries,
            primary_timing_origin="attack_start",
        )
        trajectory_metrics = orchestrator._aggregate_trajectory_metrics(run_summaries)
        mitre_phase_metrics = orchestrator._aggregate_mitre_phase_metrics(run_summaries)
        nominal_metrics = orchestrator._aggregate_nominal_metrics(run_summaries)
        split_counts: dict[str, int] = {}
        for row in completed_rows:
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        condition_summary = {
            "condition_name": self.family_spec.family_name,
            "seed": self.seed,
            "status": "completed" if len(completed_rows) == self.runs_per_seed else "partial",
            "experiment_version": self.experiment_version,
            "testbed": TESTBED_ID,
            "runs": run_summaries,
            "completed_run_count": len(completed_rows),
            "expected_run_count": self.runs_per_seed,
            "split_counts": split_counts,
            "reproducibility": reproducibility,
            "attack_metrics": attack_metrics,
            "trajectory_metrics": trajectory_metrics,
            "mitre_phase_metrics": mitre_phase_metrics,
            "nominal_metrics": nominal_metrics,
        }
        for path, payload in (
            (self.seed_root / "condition_summary.json", condition_summary),
            (self.condition_summary_path, condition_summary),
            (self.seed_root / "attack_metrics_summary.json", attack_metrics),
            (self.attack_metrics_path, attack_metrics),
            (self.seed_root / "trajectory_metrics_summary.json", trajectory_metrics),
            (self.trajectory_metrics_path, trajectory_metrics),
            (self.seed_root / "mitre_phase_metrics.json", mitre_phase_metrics),
            (self.mitre_phase_metrics_path, mitre_phase_metrics),
            (self.seed_root / "nominal_metrics_summary.json", nominal_metrics),
            (self.nominal_metrics_path, nominal_metrics),
        ):
            write_json(path, payload)

        summary_row = {
            "attacker_family": self.family_spec.family_name,
            "seed": self.seed,
            "completed_scenarios": len(completed_rows),
            "expected_scenarios": self.runs_per_seed,
            "detected_scenarios": attack_metrics.get("defender_detected_attack_run_count"),
            "defender_action_scenarios": attack_metrics.get(
                "defender_acted_attack_run_count"
            ),
            "safety_violation_scenarios": attack_metrics.get(
                "attacker_caused_safety_violation_run_count"
            ),
            "avg_wall_clock_time_to_detection_from_attack_start_sec": attack_metrics.get(
                "wall_clock_time_to_detection_from_attack_start_sec_avg"
            ),
            "avg_attacker_action_count": attack_metrics.get("attacker_action_count_avg"),
            "avg_downtime_timestep_count": attack_metrics.get(
                "downtime_timestep_count_avg"
            ),
            "primary_timing_origin": attack_metrics.get("primary_timing_origin"),
        }
        seed_summary = {
            "generated_at_utc": utc_now(),
            "experiment_version": self.experiment_version,
            "seed": self.seed,
            "family_name": self.family_spec.family_name,
            "completed_scenarios": len(completed_rows),
            "expected_scenarios": self.runs_per_seed,
            "split_counts": split_counts,
            "attack_metrics": attack_metrics,
            "trajectory_metrics": trajectory_metrics,
            "summary_row": summary_row,
        }
        write_json(self.seed_root / "seed_summary.json", seed_summary)
        write_csv(self.seed_root / "seed_summary.csv", [summary_row])

    def _write_queue_and_resume(
        self,
        scenario_rows: list[dict[str, Any]],
        *,
        completed_rows: list[dict[str, Any]] | None = None,
        failed_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        if completed_rows is None:
            completed_rows = list(dict(_read_json(self.completed_runs_path, {})).values())
        if failed_rows is None:
            failed_rows = list(dict(_read_json(self.failed_runs_path, {})).values())
        write_csv(self.run_queue_path, scenario_rows)
        write_csv(self.run_catalog_path, completed_rows)
        write_csv(self.run_failures_path, failed_rows)
        pending_rows = [row for row in scenario_rows if row["status"] != "completed"]
        next_run = pending_rows[0] if pending_rows else None
        previous_resume_state = _read_json(self.resume_state_path, {})
        resume_state = {
            "updated_utc": utc_now(),
            "experiment_version": self.experiment_version,
            "completed_runs": len(completed_rows),
            "failed_runs": len(failed_rows),
            "pending_runs": len(pending_rows),
            "all_complete": len(completed_rows) == self.runs_per_seed,
            "next_run": next_run,
            "resume_event_detected": bool(
                previous_resume_state.get("resume_event_detected")
            ),
            "resume_event_evidence": list(
                previous_resume_state.get("resume_event_evidence") or []
            ),
            "rerun_required_scenarios": [
                row["scenario_id"]
                for row in scenario_rows
                if row["status"] in {"failed", "pending"}
                and row.get("attempt_count") not in {"", "0"}
            ],
            "scenarios_rerun_after_interruption": sorted(
                {
                    row["scenario_id"]
                    for row in scenario_rows
                    if row.get("attempt_count") not in {"", "0", "1"}
                    or row.get("archived_interrupted_dir")
                }
            ),
        }
        write_json(self.resume_state_path, resume_state)
        self._write_resume_notes(scenario_rows, next_run)

    def _write_resume_notes(
        self,
        scenario_rows: list[dict[str, Any]],
        next_run: dict[str, Any] | None,
    ) -> None:
        completed_count = sum(1 for row in scenario_rows if row["status"] == "completed")
        failed_count = sum(1 for row in scenario_rows if row["status"] == "failed")
        pending_count = len(scenario_rows) - completed_count - failed_count
        lines = [
            "# Resume Notes",
            "",
            f"- Updated UTC: `{utc_now()}`",
            f"- Experiment version: `{self.experiment_version}`",
            f"- Result root: `{portable_path(self.result_root)}`",
            f"- Seed root: `{portable_path(self.seed_root)}`",
            f"- Completed runs: `{completed_count}`",
            f"- Failed runs: `{failed_count}`",
            f"- Pending runs: `{pending_count}`",
            f"- Testbed: `{TESTBED_ID}`",
            f"- Attacker: `{self.family_spec.attacker_profile_id}`",
            f"- Defender: `agentic_defender`",
        ]
        if next_run:
            lines.extend(
                [
                    "",
                    "## Next Run",
                    f"- Scenario: `{next_run['scenario_id']}`",
                    f"- Split: `{next_run['split']}`",
                    f"- Attempts so far: `{next_run['attempt_count']}`",
                ]
            )
        _write_text(self.resume_notes_path, "\n".join(lines) + "\n")

    def _sync_state(
        self,
        *,
        expected_environment: dict[str, Any],
        reproducibility: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        config_hashes = self._config_hash_payload()
        action_space_hashes = self._action_space_hash_payload()
        blue_team_models = self._blue_team_model_metadata()
        scenario_rows = self._load_scenario_rows()
        scenario_by_id = {scenario.scenario_id: scenario for scenario in self.scenarios()}
        completed_rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        run_summaries: list[dict[str, Any]] = []
        updated_rows: list[dict[str, Any]] = []

        for row in scenario_rows:
            row = dict(row)
            scenario = scenario_by_id[row["scenario_id"]]
            run_dir = self._run_dir(scenario)
            if row["status"] == "completed" and self._verify_run_completion(
                run_dir,
                expected_environment=expected_environment,
            ):
                summary = self._load_run_summary(run_dir)
                completed_rows.append(
                    self._completed_row(
                        scenario=scenario,
                        run_dir=run_dir,
                        summary=summary,
                        split=row["split"],
                        config_hashes=config_hashes,
                        action_space_hashes=action_space_hashes,
                        blue_team_models=blue_team_models,
                    )
                )
                run_summaries.append(summary)
            else:
                if row["status"] == "completed":
                    row["status"] = "pending"
                    row["completed_at_utc"] = ""
                    row["last_error"] = (
                        "completed marker existed but required artifacts were incomplete"
                    )
                if row["status"] == "failed":
                    failed_rows.append(
                        self._failure_row(
                            scenario_id=row["scenario_id"],
                            split=row["split"],
                            run_dir=run_dir,
                            error_message=row.get("last_error", ""),
                        )
                    )
            updated_rows.append(row)

        self._save_scenario_rows(updated_rows)
        write_json(
            self.completed_runs_path,
            {row["scenario_id"]: row for row in completed_rows},
        )
        write_json(
            self.failed_runs_path,
            {row["scenario_id"]: row for row in failed_rows},
        )
        self._write_seed_summaries(
            completed_rows=completed_rows,
            run_summaries=run_summaries,
            reproducibility=reproducibility,
        )
        self._write_queue_and_resume(
            updated_rows,
            completed_rows=completed_rows,
            failed_rows=failed_rows,
        )
        return completed_rows, failed_rows

    def _ensure_env_defaults(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        os.environ["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", str(self.seed))
        os.environ.setdefault("QWEN_MODEL_ID", "Qwen/Qwen3.5-9B")
        if not os.environ.get("QWEN_MODEL_PATH"):
            os.environ["QWEN_MODEL_PATH"] = str(
                (REPO_ROOT / "models" / "local_models" / "qwen3.5-9b").resolve()
            )

    def _check_mrg_transport(
        self, *, merge_server: str | None = None, timeout_sec: float = 20.0
    ) -> str:
        command = ["mrg"]
        if merge_server:
            command.extend(["-s", merge_server])
        command.append("whoami")
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"mrg transport preflight timed out after {timeout_sec:.0f}s: {' '.join(command)}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"mrg transport preflight failed ({completed.returncode}): {detail}"
            )
        return completed.stdout.strip()

    def _check_mrg_show_json(
        self,
        kind: str,
        ref: str,
        *,
        merge_server: str,
        timeout_sec: float = 20.0,
    ) -> dict[str, Any]:
        command = ["mrg"]
        if merge_server:
            command.extend(["-s", merge_server])
        command.extend(["show", kind, ref, "--json"])
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"mrg {kind} preflight timed out after {timeout_sec:.0f}s: {' '.join(command)}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"mrg {kind} preflight failed ({completed.returncode}) for `{ref}`: {detail}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"mrg {kind} preflight returned non-JSON output for `{ref}`"
            ) from exc

    def _check_mrg_text(
        self,
        args: list[str],
        *,
        merge_server: str,
        timeout_sec: float = 20.0,
    ) -> str:
        command = ["mrg"]
        if merge_server:
            command.extend(["-s", merge_server])
        command.extend(args)
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"mrg {' '.join(args)} timed out after {timeout_sec:.0f}s"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"mrg {' '.join(args)} failed ({completed.returncode}): {detail}"
            )
        return completed.stdout

    @staticmethod
    def _qualified_xdc_ref(short_name: str, merge_user: str) -> str:
        return short_name if short_name.count(".") == 1 else f"{short_name}.{merge_user}"

    @staticmethod
    def _extract_xdc_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload.get("xdc"), dict):
            return dict(payload["xdc"])
        result = payload.get("result")
        if isinstance(result, dict) and isinstance(result.get("xdc"), dict):
            return dict(result["xdc"])
        return {}

    @staticmethod
    def _parse_xdc_refs(list_output: str) -> list[str]:
        refs: list[str] = []
        for line in list_output.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("Name.Project"):
                continue
            token = stripped.split()[0]
            if re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token) and token not in refs:
                refs.append(token)
        return refs

    def _runner_import_name(self) -> str:
        module_name = str(self.__class__.__module__)
        if module_name != "__main__":
            return module_name
        source_path = Path(inspect.getsourcefile(self.__class__) or "").resolve()
        try:
            relative = source_path.relative_to(REPO_ROOT)
        except Exception as exc:
            raise RuntimeError(
                f"unable to derive import path for `{self.__class__.__name__}`"
            ) from exc
        return ".".join(relative.with_suffix("").parts)

    def _listed_xdc_refs(self, *, merge_server: str) -> tuple[str, list[str]]:
        output = self._check_mrg_text(
            ["list", "xdcs"],
            merge_server=merge_server,
        )
        return output, self._parse_xdc_refs(output)

    def _candidate_xdc_short_names(
        self,
        *,
        available_refs: list[str],
        configured_xdc: str,
        preferred_xdcs: list[str] | tuple[str, ...] = (),
        candidate_xdc_prefix: str = "",
    ) -> list[str]:
        names: list[str] = []
        available_short_names = sorted(
            {
                ref.split(".", 1)[0]
                for ref in available_refs
                if candidate_xdc_prefix
                and ref.split(".", 1)[0].startswith(candidate_xdc_prefix)
            }
        )
        for name in (*preferred_xdcs, configured_xdc, *available_short_names):
            short_name = str(name).strip()
            if short_name and short_name not in names:
                names.append(short_name)
        return names

    def _inspect_xdc_candidates(
        self,
        *,
        candidate_short_names: list[str],
        merge_server: str,
        merge_user: str,
        realization_ref: str,
    ) -> list[dict[str, Any]]:
        inspections: list[dict[str, Any]] = []
        for short_name in candidate_short_names:
            ref = self._qualified_xdc_ref(short_name, merge_user)
            entry: dict[str, Any] = {
                "short_name": short_name,
                "ref": ref,
                "available": False,
                "healthy": False,
                "attached": False,
            }
            try:
                payload = self._check_mrg_show_json(
                    "xdc",
                    ref,
                    merge_server=merge_server,
                )
            except Exception as exc:
                entry["error"] = str(exc)
                inspections.append(entry)
                continue
            xdc_payload = self._extract_xdc_payload(payload)
            materialization = str(
                xdc_payload.get("materialization")
                or xdc_payload.get("materialization_ref")
                or xdc_payload.get("realization")
                or ""
            ).strip()
            status = str(
                xdc_payload.get("status")
                or xdc_payload.get("state")
                or xdc_payload.get("health")
                or ""
            ).strip()
            status_token = status.lower()
            healthy = status_token not in {
                "error",
                "failed",
                "stale",
                "deleting",
                "deleted",
            }
            entry.update(
                {
                    "available": True,
                    "healthy": healthy,
                    "attached": materialization == realization_ref,
                    "materialization": materialization,
                    "status": status,
                }
            )
            inspections.append(entry)
        return inspections

    def _ordered_xdc_candidates(
        self,
        *,
        inspections: list[dict[str, Any]],
        configured_xdc: str,
    ) -> list[dict[str, Any]]:
        attached = [
            entry
            for entry in inspections
            if entry.get("available") and entry.get("healthy") and entry.get("attached")
        ]
        attached.sort(
            key=lambda entry: (
                entry.get("short_name") != configured_xdc,
                str(entry.get("short_name", "")),
            )
        )
        if attached:
            return attached
        detached = [
            entry for entry in inspections if entry.get("available") and entry.get("healthy")
        ]
        detached.sort(
            key=lambda entry: (
                entry.get("short_name") != configured_xdc,
                str(entry.get("short_name", "")),
            )
        )
        return detached

    def _attempt_validation_path(self, xdc_short_name: str, attempt_index: int) -> Path:
        token = _sanitize_token(xdc_short_name) or f"attempt_{attempt_index}"
        return (
            self.progress_root
            / f"optionb_environment_validation_attempt_{attempt_index:02d}_{token}.json"
        )

    def _run_optionb_validation_attempt(
        self,
        *,
        xdc_short_name: str,
        validation_path: Path,
        timeout_sec: float,
    ) -> dict[str, Any]:
        command = [
            _python_executable(),
            "-c",
            (
                "from pathlib import Path; "
                f"from {self._runner_import_name()} import {self.__class__.__name__}; "
                "import sys; "
                "runner = "
                f"{self.__class__.__name__}(Path(sys.argv[1])); "
                "runner._validate_optionb_environment("
                "validation_path=Path(sys.argv[2]), update_phase=False)"
            ),
            str(self.result_root),
            str(validation_path),
        ]
        env = os.environ.copy()
        env["ICS_AGENT_SPHERE_XDC_OVERRIDE"] = xdc_short_name
        started_utc = utc_now()
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                cwd=REPO_ROOT,
                env=env,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "hung",
                "short_name": xdc_short_name,
                "validation_path": portable_path(validation_path),
                "started_utc": started_utc,
                "completed_utc": utc_now(),
                "timeout_sec": timeout_sec,
                "stdout": "",
                "stderr": "",
                "error": (
                    "optionb environment validation remained in progress without "
                    "completing before the smoke timeout"
                ),
            }
        payload = _read_json(validation_path, {})
        status = "passed" if completed.returncode == 0 else "failed"
        result = {
            "status": status,
            "short_name": xdc_short_name,
            "validation_path": portable_path(validation_path),
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "payload_status": payload.get("status"),
        }
        if completed.returncode != 0:
            result["error"] = str(payload.get("error") or completed.stderr.strip() or completed.stdout.strip())
        return result

    def _cleanup_failed_validation_attempt(self, xdc_short_name: str) -> None:
        config = self._base_config()
        environment = deepcopy(dict(config.get("environment") or {}))
        environment["testbed"] = TESTBED_ID
        environment["xdc"] = xdc_short_name
        helper_root: Path | None = None
        try:
            env = self._optionb_environment_cls()(environment)
            backend = env._backend_impl
            helper_root = getattr(backend, "helper_root", None)
            env.stop()
        except Exception:
            pass
        if isinstance(helper_root, Path) and helper_root.exists():
            shutil.rmtree(helper_root, ignore_errors=True)

    def _optionb_validation_scenario(self) -> Any:
        for scenario in self.scenarios():
            if scenario.scenario_id == self.optionb_validation_scenario_id:
                return scenario
        raise RuntimeError(
            f"missing distribution validation scenario `{self.optionb_validation_scenario_id}`"
        )

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _validate_optionb_smoke_snapshot(
        self,
        *,
        scenario: Any,
        reset_snapshot: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        expected = {
            key: float(value)
            for key, value in (scenario.execution.get("initial_conditions") or {}).items()
            if key
            in {
                "Supply_Tank_Level",
                "Grid_Elev_Tank_Level",
                "RWS_Tank_Level",
            }
        }
        if expected != {
            "Supply_Tank_Level": 930.0,
            "Grid_Elev_Tank_Level": 920.0,
            "RWS_Tank_Level": 320.0,
        }:
            raise RuntimeError(
                "distribution smoke validation scenario no longer matches the required "
                "`wd-attack-coordinated-grid-rws-delay` reset contract"
            )
        initial_state_validation = dict(reset_snapshot.get("initial_state_validation") or {})
        if str(initial_state_validation.get("status", "")).strip().lower() != "passed":
            raise RuntimeError(
                "distribution smoke validation failed: initial-state validation did not pass"
            )
        post_reset_observation = dict(reset_snapshot.get("post_reset_observation") or {})
        required_fields = ["STATE_NAME", *expected.keys()]
        missing = [
            field
            for field in required_fields
            if post_reset_observation.get(field) is None and observation.get(field) is None
        ]
        if missing:
            raise RuntimeError(
                "distribution smoke validation failed: reset observation is missing "
                f"required distribution fields {missing}"
            )
        observed = {
            field: post_reset_observation.get(field, observation.get(field))
            for field in expected
        }
        mismatches: dict[str, dict[str, Any]] = {}
        tolerance = 1.0
        for field, expected_value in expected.items():
            actual_value = self._coerce_float(observed.get(field))
            if actual_value is None:
                mismatches[field] = {
                    "expected": expected_value,
                    "observed": observed.get(field),
                }
                continue
            difference = round(abs(actual_value - expected_value), 3)
            if difference > tolerance:
                mismatches[field] = {
                    "expected": expected_value,
                    "observed": actual_value,
                    "difference": difference,
                }
        if mismatches:
            details = "; ".join(
                f"{field}: expected {payload.get('expected')}, observed {payload.get('observed')}"
                for field, payload in mismatches.items()
            )
            raise RuntimeError(
                "distribution smoke validation failed after reset: " + details
            )
        return {
            "status": "passed",
            "family": "water_distribution",
            "scenario_id": scenario.scenario_id,
            "expected": expected,
            "observed": observed,
            "tolerance": tolerance,
            "state_name": post_reset_observation.get("STATE_NAME")
            or observation.get("STATE_NAME"),
            "initial_state_validation": initial_state_validation,
        }

    def _optionb_environment_cls(self):
        return OpenPLCEnvironment

    def _optionb_backend_cls(self):
        return _SphereOpenPLCBackend

    def _validate_optionb_environment(
        self,
        *,
        validation_path: Path | None = None,
        update_phase: bool = True,
    ) -> None:
        target_validation_path = validation_path or self.optionb_validation_path
        if update_phase:
            self._set_phase("validated_optionb_environment", "in_progress")
        self._ensure_env_defaults()
        config = self._runtime_config()
        env = self._optionb_environment_cls()(config.get("environment", {}))
        scenario = self._optionb_validation_scenario()
        preflight_refs = self._required_preflight_refs(config)
        result: dict[str, Any] = {
            "validated_at_utc": utc_now(),
            "config_path": portable_path(Path(config["_path"])),
            "status": "failed",
        }
        try:
            merge_user = self._check_mrg_transport(
                merge_server=preflight_refs["merge_server"]
            )
            realization_payload = self._check_mrg_show_json(
                "realization",
                preflight_refs["realization"],
                merge_server=preflight_refs["merge_server"],
            )
            materialization_payload = self._check_mrg_show_json(
                "materialization",
                preflight_refs["materialization"],
                merge_server=preflight_refs["merge_server"],
            )
            startup = env.start()
            backend = env._backend_impl
            if not isinstance(backend, self._optionb_backend_cls()):
                raise RuntimeError("expected sphere_openplc backend for water distribution")
            reset_snapshot = env.reset(scenario.execution.get("initial_conditions"))
            observation = env.observe_flat()
            discovery = env.inspect_surface({"type": "inspect_surface", "surface": "tags"})
            smoke_validation = self._validate_optionb_smoke_snapshot(
                scenario=scenario,
                reset_snapshot=reset_snapshot,
                observation=observation,
            )
            result.update(
                {
                    "status": "passed",
                    "preflight": {
                        "merge_server_configured": bool(preflight_refs["merge_server"]),
                        "mrg_whoami_ok": bool(merge_user),
                        "realization_check_ok": bool(realization_payload),
                        "materialization_check_ok": bool(materialization_payload),
                    },
                    "startup": startup,
                    "validated_target": {
                        "experiment_configured": bool(backend.experiment_ref),
                        "realization_configured": bool(backend._realization_ref),
                        "revision_configured": bool(backend.revision),
                        "xdc_configured": bool(backend._xdc_ref),
                    },
                    "smoke_validation": smoke_validation,
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
            write_json(target_validation_path, result)
            if update_phase:
                self._set_phase(
                    "validated_optionb_environment",
                    "completed",
                    validation_path=portable_path(target_validation_path),
                )
            return
        except Exception as exc:
            result["error"] = str(exc)
            write_json(target_validation_path, result)
            if update_phase:
                self._set_phase(
                    "validated_optionb_environment",
                    "failed",
                    error_message=str(exc),
                    validation_path=portable_path(target_validation_path),
                )
                _append_log(
                    self.experiment_log_path,
                    f"optionb environment validation failed: {exc}",
                )
            raise
        finally:
            try:
                env.stop()
            except Exception:
                pass

    def _validate_optionb_environment_with_xdc_discovery(self) -> None:
        self._set_phase("validated_optionb_environment", "in_progress")
        config = self._base_config()
        environment = dict(config.get("environment") or {})
        preflight_refs = self._required_preflight_refs(config)
        configured_xdc = str(environment.get("xdc", "")).strip()
        discovery_allowed = bool(environment.get("allow_xdc_discovery", False))
        preferred_xdcs = list(environment.get("preferred_xdcs") or self.preferred_xdc_short_names)
        candidate_xdc_prefix = str(
            environment.get("xdc_candidate_prefix") or self.candidate_xdc_prefix
        ).strip()
        candidate_selection: dict[str, Any] = {
            "configured_xdc": configured_xdc,
            "preferred_xdcs": preferred_xdcs,
            "xdc_discovery_allowed": discovery_allowed,
            "realization_ref": preflight_refs["realization"],
            "materialization_ref": preflight_refs["materialization"],
        }
        final_result: dict[str, Any] = {
            "validated_at_utc": utc_now(),
            "config_path": portable_path(Path(config["_path"])),
            "status": "failed",
            "candidate_selection": candidate_selection,
            "attempts": [],
        }
        try:
            merge_user = self._check_mrg_transport(
                merge_server=preflight_refs["merge_server"]
            )
            candidate_selection["mrg_whoami_ok"] = bool(merge_user)
            if discovery_allowed:
                list_output, available_refs = self._listed_xdc_refs(
                    merge_server=preflight_refs["merge_server"]
                )
                candidate_selection["listed_xdc_count"] = len(available_refs)
                candidate_selection["xdc_candidate_prefix"] = candidate_xdc_prefix
                candidate_short_names = self._candidate_xdc_short_names(
                    available_refs=available_refs,
                    configured_xdc=configured_xdc,
                    preferred_xdcs=preferred_xdcs,
                    candidate_xdc_prefix=candidate_xdc_prefix,
                )
                inspections = self._inspect_xdc_candidates(
                    candidate_short_names=candidate_short_names,
                    merge_server=preflight_refs["merge_server"],
                    merge_user=merge_user,
                    realization_ref=preflight_refs["realization"],
                )
                candidate_selection["inspected_xdc_count"] = len(inspections)
                ordered_candidates = self._ordered_xdc_candidates(
                    inspections=inspections,
                    configured_xdc=configured_xdc,
                )
                if not ordered_candidates:
                    raise RuntimeError(
                        "no healthy distribution XDC candidate is currently available for "
                        f"`{preflight_refs['realization']}`"
                    )
            else:
                ordered_candidates = [
                    {
                        "short_name": configured_xdc,
                        "ref": self._qualified_xdc_ref(configured_xdc, merge_user),
                        "configured_only": True,
                    }
                ]
            for attempt_index, candidate in enumerate(ordered_candidates, start=1):
                short_name = str(candidate.get("short_name", "")).strip()
                candidate_selection["chosen_xdc"] = short_name
                candidate_selection["chosen_xdc_ref"] = candidate.get("ref")
                _append_log(
                    self.experiment_log_path,
                    f"trying distribution smoke validation with XDC `{short_name}`",
                )
                attempt_path = self._attempt_validation_path(short_name, attempt_index)
                attempt = self._run_optionb_validation_attempt(
                    xdc_short_name=short_name,
                    validation_path=attempt_path,
                    timeout_sec=self.validation_attempt_timeout_sec,
                )
                final_result["attempts"].append(attempt)
                if attempt["status"] == "passed":
                    payload = _read_json(attempt_path, {})
                    payload["selected_xdc"] = {
                        "short_name": short_name,
                        "ref": candidate.get("ref"),
                    }
                    payload["candidate_selection"] = candidate_selection
                    payload["attempts"] = final_result["attempts"]
                    write_json(self.optionb_validation_path, payload)
                    os.environ["ICS_AGENT_SPHERE_XDC_OVERRIDE"] = short_name
                    self._set_phase(
                        "validated_optionb_environment",
                        "completed",
                        validation_path=portable_path(self.optionb_validation_path),
                        chosen_xdc=short_name,
                    )
                    _append_log(
                        self.experiment_log_path,
                        f"distribution smoke validation passed with XDC `{short_name}`",
                    )
                    return
                self._cleanup_failed_validation_attempt(short_name)
            last_attempt = final_result["attempts"][-1] if final_result["attempts"] else {}
            raise RuntimeError(str(last_attempt.get("error") or "all distribution XDC attempts failed"))
        except Exception as exc:
            final_result["error"] = str(exc)
            write_json(self.optionb_validation_path, final_result)
            self._set_phase(
                "validated_optionb_environment",
                "failed",
                error_message=str(exc),
                validation_path=portable_path(self.optionb_validation_path),
                chosen_xdc=candidate_selection.get("chosen_xdc"),
            )
            _append_log(
                self.experiment_log_path,
                f"optionb environment validation failed: {exc}",
            )
            raise

    def _run_logged(
        self,
        command: list[str],
        log_path: Path,
    ) -> None:
        env = os.environ.copy()
        if len(command) >= 3 and command[1:3] == ["-m", "pytest"]:
            env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        started_utc = utc_now()
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            env=env,
        )
        payload = {
            "cwd": portable_path(REPO_ROOT),
            "command": command,
            "returncode": completed.returncode,
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        write_json(log_path, payload)
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
            [python_exec, "-m", "pytest", *self.validation_test_paths],
            self.pytest_log_path,
        )
        self._set_phase(
            "validation_tests",
            "completed",
            validate_log=portable_path(self.validate_log_path),
            pytest_log=portable_path(self.pytest_log_path),
        )

    def _mark_resume_event_if_needed(self, rows: list[dict[str, Any]]) -> None:
        evidence = [
            {
                "scenario_id": row["scenario_id"],
                "status": row["status"],
                "attempt_count": row["attempt_count"],
                "archived_interrupted_dir": row.get("archived_interrupted_dir", ""),
            }
            for row in rows
            if row["status"] in {"running", "failed"}
            or row.get("attempt_count") not in {"", "0"}
            or row.get("archived_interrupted_dir")
        ]
        if not evidence:
            return
        resume_state = _read_json(self.resume_state_path, {})
        if resume_state.get("resume_event_detected"):
            return
        resume_state["resume_event_detected"] = True
        resume_state["resume_event_detected_at_utc"] = utc_now()
        resume_state["resume_event_evidence"] = evidence
        write_json(self.resume_state_path, resume_state)

    def run_controller(self) -> int:
        self.freeze_metadata()
        _append_log(
            self.experiment_log_path,
            f"controller starting {self.controller_run_label} under {portable_path(self.result_root)}",
        )
        rows = self._load_scenario_rows()
        self._mark_resume_event_if_needed(rows)
        self._validate_optionb_environment_with_xdc_discovery()
        self._validate_runtime_and_tests()
        self._set_phase("scenario_run", "in_progress")
        completed_rows, _ = self._sync_state(
            expected_environment=self._expected_environment(self._runtime_config()),
            reproducibility={"seed": self.seed},
        )
        if len(completed_rows) != self.runs_per_seed:
            exit_code = self.run_worker(max_attempts=2)
            if exit_code != 0:
                return exit_code
        completed_rows, failed_rows = self._sync_state(
            expected_environment=self._expected_environment(self._runtime_config()),
            reproducibility={"seed": self.seed},
        )
        if failed_rows:
            return 1
        if len(completed_rows) != self.runs_per_seed:
            raise RuntimeError(
                f"expected {self.runs_per_seed} completed runs, found {len(completed_rows)}"
            )
        self._set_phase(
            "scenario_run",
            "completed",
            completed_runs=self.runs_per_seed,
        )
        self.write_aggregate()
        self.write_report()
        self.write_repo_note()
        self.write_repo_local_report()
        return 0

    def run_worker(self, *, max_attempts: int) -> int:
        self.ensure_root_layout()
        self._ensure_env_defaults()
        config = self._runtime_config()
        orchestrator = self._orchestrator()
        condition_spec = self._condition_spec(orchestrator, config)
        expected_environment = self._expected_environment(config)
        reproducibility = orchestrator._configure_reproducibility(
            config=config,
            conditions=[condition_spec],
        )
        orchestrator._current_reproducibility = reproducibility
        scenario_rows = self._load_scenario_rows()
        self._mark_resume_event_if_needed(scenario_rows)
        completed_rows, _ = self._sync_state(
            expected_environment=expected_environment,
            reproducibility=reproducibility,
        )
        if len(completed_rows) == self.runs_per_seed:
            return 0

        scenario_by_id = {scenario.scenario_id: scenario for scenario in self.scenarios()}
        for row in scenario_rows:
            if row["status"] == "completed":
                continue
            scenario = scenario_by_id[row["scenario_id"]]
            run_dir = self._run_dir(scenario)
            archived_path = ""
            if run_dir.exists():
                archived_path = self._archive_partial_run(scenario.scenario_id, run_dir)
            attempts = int(row["attempt_count"] or "0")
            success = False
            last_error = ""
            for _ in range(max_attempts):
                attempts += 1
                row["status"] = "running"
                row["attempt_count"] = str(attempts)
                row["started_at_utc"] = utc_now()
                row["run_id"] = self._run_id(scenario, row["split"])
                row["archived_interrupted_dir"] = archived_path
                self._save_scenario_rows(scenario_rows)
                self._write_queue_and_resume(scenario_rows)

                reproducibility = orchestrator._configure_reproducibility(
                    config=config,
                    conditions=[condition_spec],
                )
                orchestrator._current_reproducibility = reproducibility
                bundle = orchestrator._run_single(
                    experiment_id=self.experiment_version,
                    condition_name=self.family_spec.family_name,
                    condition=condition_spec["profiles"],
                    effective_profiles=condition_spec.get("effective_profiles"),
                    scenario=scenario,
                    run_id=row["run_id"],
                    run_dir=run_dir,
                    config=config,
                )
                summary = dict(bundle.get("summary") or {})
                if self._verify_run_completion(
                    run_dir,
                    expected_environment=expected_environment,
                ):
                    row["status"] = "completed"
                    row["completed_at_utc"] = utc_now()
                    row["failed_at_utc"] = ""
                    row["last_error"] = ""
                    success = True
                    break
                error_text = "; ".join(
                    str(item.get("error"))
                    for item in summary.get("errors", [])
                    if isinstance(item, dict)
                ) or "scenario run failed or did not produce required artifacts"
                last_error = error_text
                row["status"] = "failed"
                row["failed_at_utc"] = utc_now()
                row["last_error"] = error_text
                self._save_scenario_rows(scenario_rows)
                self._write_queue_and_resume(scenario_rows)
                if run_dir.exists():
                    archived_path = self._archive_partial_run(
                        scenario.scenario_id,
                        run_dir,
                    )
                    row["archived_interrupted_dir"] = archived_path
            self._save_scenario_rows(scenario_rows)
            completed_rows, failed_rows = self._sync_state(
                expected_environment=expected_environment,
                reproducibility=reproducibility,
            )
            if not success:
                _append_log(
                    self.experiment_log_path,
                    f"scenario failed after {attempts} attempts: {scenario.scenario_id}: {last_error}",
                )
                return 1
            _append_log(
                self.experiment_log_path,
                f"completed scripted baseline scenario {scenario.scenario_id}",
            )
            if len(completed_rows) == self.runs_per_seed and not failed_rows:
                break
        return 0

    def write_aggregate(self) -> None:
        self._set_phase("aggregating", "in_progress")
        with self.run_catalog_path.open(newline="") as handle:
            catalog_rows = list(csv.DictReader(handle))
        if len(catalog_rows) != self.runs_per_seed:
            raise RuntimeError(
                f"expected {self.runs_per_seed} catalog rows, found {len(catalog_rows)}"
            )
        detection_values = [
            float(row["wall_clock_time_to_detection_from_attack_start_sec"])
            for row in catalog_rows
            if row.get("wall_clock_time_to_detection_from_attack_start_sec")
            not in {"", "None", None}
        ]
        action_values = [
            float(row["attacker_action_count"])
            for row in catalog_rows
            if row.get("attacker_action_count") not in {"", "None", None}
        ]
        downtime_values = [
            float(row["downtime_timestep_count"])
            for row in catalog_rows
            if row.get("downtime_timestep_count") not in {"", "None", None}
        ]
        aggregate_row = {
            "family_name": self.family_spec.family_name,
            "seed": self.seed,
            "total_runs": self.runs_per_seed,
            "total_completed_runs": len(catalog_rows),
            "safety_violation_count": sum(
                int(row["safety_violation"]) for row in catalog_rows
            ),
            "detected_count": sum(int(row["detected"]) for row in catalog_rows),
            "defender_action_count": sum(
                int(row["defender_acted"]) for row in catalog_rows
            ),
            "mean_wall_clock_detection_from_attack_start_sec": _mean(detection_values),
            "std_wall_clock_detection_from_attack_start_sec": _stdev(detection_values),
            "mean_attacker_action_count": _mean(action_values),
            "std_attacker_action_count": _stdev(action_values),
            "mean_downtime_timestep_count": _mean(downtime_values),
            "std_downtime_timestep_count": _stdev(downtime_values),
        }
        write_json(self.aggregate_json_path, aggregate_row)
        write_csv(self.aggregate_csv_path, [aggregate_row])
        write_json(self.seed_root / "aggregate_over_seed_101.json", aggregate_row)
        write_csv(self.seed_root / "aggregate_over_seed_101.csv", [aggregate_row])
        self._set_phase("aggregating", "completed")

    def _report_lines(self) -> list[str]:
        condition_summary = _read_json(self.condition_summary_path, {})
        attack_metrics = dict(condition_summary.get("attack_metrics") or {})
        aggregate_row = _read_json(self.aggregate_json_path, {})
        return [
            "# Scripted Black-Box Baseline Report",
            "",
            f"- Generated UTC: `{utc_now()}`",
            f"- Testbed: `{TESTBED_ID}` only",
            f"- Attacker: `{self.family_spec.attacker_profile_id}` deterministic scripted black-box baseline",
            f"- Defender: `agentic_defender` current full containment defender",
            f"- Seed: `{self.seed}` only",
            f"- Results root: `{portable_path(self.result_root)}`",
            f"- Seed root: `{portable_path(self.seed_root)}`",
            "- This pass is a deterministic attacker-baseline comparison and does not replace the LLM attacker results.",
            "",
            "## Outcome",
            f"- Completed scenarios: `{condition_summary.get('completed_run_count')}/{condition_summary.get('expected_run_count')}`",
            f"- Detected scenarios: `{attack_metrics.get('defender_detected_attack_run_count')}/{self.runs_per_seed}`",
            f"- Defender-action scenarios: `{attack_metrics.get('defender_acted_attack_run_count')}/{self.runs_per_seed}`",
            f"- Safety-violation scenarios: `{attack_metrics.get('attacker_caused_safety_violation_run_count')}/{self.runs_per_seed}`",
            f"- Runtime containment cutoff activations: `{attack_metrics.get('attacker_execution_cutoff_run_count', attack_metrics.get('attacker_execution_cutoff_rate'))}`",
            f"- Average wall-clock detection from attack start: `{_format_mean_std(aggregate_row.get('mean_wall_clock_detection_from_attack_start_sec'), aggregate_row.get('std_wall_clock_detection_from_attack_start_sec'))}`",
            f"- Average attacker action count: `{_format_mean_std(aggregate_row.get('mean_attacker_action_count'), aggregate_row.get('std_attacker_action_count'))}`",
            f"- Average downtime burden: `{_format_mean_std(aggregate_row.get('mean_downtime_timestep_count'), aggregate_row.get('std_downtime_timestep_count'))}`",
            "",
            "## Notes",
            "- The attacker used deterministic fixed logic over the black-box-visible runtime interface.",
            "- The blue-team roles remained the current scripted observer and full containment defender.",
            "- Resume ledgers were written under the result-root `_progress` directory and preserved per-scenario completion state.",
        ]

    def write_report(self) -> None:
        self._set_phase("final_report", "in_progress")
        _write_text(self.report_path, "\n".join(self._report_lines()) + "\n")
        self._set_phase(
            "final_report",
            "completed",
            report_path=portable_path(self.report_path),
        )

    def write_repo_note(self) -> None:
        resume_state = _read_json(self.resume_state_path, {})
        lines = [
            "# Scripted Black-Box Baseline Note",
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
            "- Defender remained the current full-containment distribution stack.",
            "- Existing LLM attacker and defender families were left intact.",
        ]
        _write_text(self.repo_note_path, "\n".join(lines) + "\n")

    def write_repo_local_report(self) -> None:
        self.repo_local_report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            *self._report_lines(),
            "",
            "## Repo Note",
            "- This repo-local note summarizes the local scripted-baseline pass and references the local result root above.",
            "- Raw experiment artifacts remain under `Artifact/Results` and are not intended for git commit.",
        ]
        _write_text(self.repo_local_report_path, "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runner = ScriptedBlackBoxBaselineRunner(Path(args.result_root))
    if args.mode == "worker":
        return runner.run_worker(max_attempts=int(args.max_attempts))
    return runner.run_controller()


if __name__ == "__main__":
    raise SystemExit(main())
