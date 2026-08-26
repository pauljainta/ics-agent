from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from ...config import load_config, require_fields, resolve_repo_path
from ...environment import OpenPLCEnvironment, _SphereOpenPLCBackend
from ...mitre import verify_optionb_staged_snapshot
from ...orchestrator import ExperimentOrchestrator
from ...paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from ...reporting import write_csv, write_json
from ...utils import dump_yaml, load_yaml, utc_now
from .seeded_backbone import (
    ACTION_SPACE_SOURCE_PATHS,
    CONTAINMENT_SEMANTICS_VERSION,
    MODEL_LABEL,
    REQUIRED_RUN_FILES,
    TESTBED_ID,
    TOP_LEVEL_PHASES,
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


EXPERIMENT_VERSION = "water_treatment_white_box_ablation_qwen35_9b_v1"
CANONICAL_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "Water-treatment"
    / "White-Box-Ablation"
)
BASELINE_SEED_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "Water-treatment"
    / "seeded_backbone_qwen35_9b"
    / "white_box_attacker__agentic_defender"
    / "seed_101"
)
SEEDS = [101]
RUNS_PER_SEED = 18
RUNS_PER_FAMILY = len(SEEDS) * RUNS_PER_SEED
ABLATED_PROMPT_PATH = (
    REPO_ROOT / "testbeds" / "water_treatment" / "prompts" / "white_box_attacker_ablation.md"
)
DEFENDER_PROMPT_PATH = REPO_ROOT / "testbeds" / "water_treatment" / "prompts" / "defender.md"
OBSERVER_PROMPT_PATH = REPO_ROOT / "testbeds" / "water_treatment" / "prompts" / "observer.md"
CONFIG_PATHS = [
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_white_box_no_safety_bounds_attacker_vs_agentic_full_containment_defender.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_white_box_no_surface_priors_attacker_vs_agentic_full_containment_defender.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_white_box_no_control_semantics_attacker_vs_agentic_full_containment_defender.yaml",
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
TOTAL_RUNS = len(FAMILY_SPECS) * RUNS_PER_FAMILY
VALIDATION_TEST_PATHS = [
    "tests/test_agentic.py",
    "tests/test_orchestrator.py",
    "tests/test_seeded_backbone.py",
    "tests/test_white_box_ablation.py",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume the water-treatment white-box ablation pass."
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


class WhiteBoxAblationRunner:
    def __init__(self, result_root: Path) -> None:
        self.result_root = result_root.resolve()
        self.progress_root = self.result_root / "_progress"
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
        self.run_catalog_path = self.result_root / "run_catalog.csv"
        self.run_failures_path = self.result_root / "run_failures.csv"
        self.report_path = self.result_root / "white_box_ablation_report.md"
        self.comparison_csv_path = self.result_root / "baseline_comparison_seed_101.csv"
        self.comparison_json_path = self.result_root / "baseline_comparison_seed_101.json"
        self.comparison_md_path = self.result_root / "baseline_comparison_seed_101.md"
        self.repo_note_path = self.result_root / "white_box_ablation_note.md"
        self.manifest_path = self.result_root / "scenario_manifest_water_treatment.csv"
        self.splits_path = self.result_root / "scenario_splits_water_treatment.csv"
        self.seeds_path = self.result_root / "seeds.txt"
        self.model_registry_path = self.result_root / "model_registry.yaml"
        self.experiment_version_path = self.result_root / "experiment_version.txt"
        self.prompt_hashes_path = self.result_root / "prompt_hashes.json"
        self.action_space_hashes_path = self.result_root / "action_space_hashes.json"
        self.containment_semantics_path = (
            self.result_root / "containment_semantics_version.txt"
        )
        self.logger_schema_path = self.result_root / "logger_schema.json"
        self.metrics_schema_path = self.result_root / "metrics_schema.json"

    def ensure_root_layout(self) -> None:
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.progress_root.mkdir(parents=True, exist_ok=True)
        for family in FAMILY_SPECS:
            (self.family_root(family.family_name) / "_progress").mkdir(
                parents=True, exist_ok=True
            )
            for seed in SEEDS:
                (self.seed_progress_root(family.family_name, seed)).mkdir(
                    parents=True, exist_ok=True
                )

    def family_root(self, family_name: str) -> Path:
        return self.result_root / family_name

    def seed_root(self, family_name: str, seed: int) -> Path:
        return self.family_root(family_name) / f"seed_{seed}"

    def seed_progress_root(self, family_name: str, seed: int) -> Path:
        return self.seed_root(family_name, seed) / "_progress"

    def family_progress_root(self, family_name: str) -> Path:
        return self.family_root(family_name) / "_progress"

    def _orchestrator(self) -> ExperimentOrchestrator:
        return ExperimentOrchestrator(testbed_id=TESTBED_ID)

    def scenarios(self) -> list[Any]:
        return self._orchestrator().catalog.resolve_selectors(["attack/*"])

    def manifest_rows(self) -> list[dict[str, Any]]:
        rows = build_manifest_rows(self.scenarios())
        enabled = [row for row in rows if row.get("enabled_for_final") == "true"]
        if len(enabled) != RUNS_PER_SEED:
            raise RuntimeError(
                f"expected exactly {RUNS_PER_SEED} enabled water-treatment scenarios, found {len(enabled)}"
            )
        return rows

    def split_rows(self) -> list[dict[str, Any]]:
        rows = build_split_rows(self.manifest_rows())
        if len(rows) != RUNS_PER_SEED:
            raise RuntimeError(
                f"expected {RUNS_PER_SEED} split rows, found {len(rows)}"
            )
        return rows

    def _base_config(self, family_spec: FamilySpec) -> dict[str, Any]:
        config = load_config(str(family_spec.config_path))
        require_fields(
            config,
            ["experiment_name", "scenario_selectors"],
            ctx="experiment config",
        )
        return config

    def _runtime_config(self, family_spec: FamilySpec, seed: int) -> dict[str, Any]:
        config = deepcopy(self._base_config(family_spec))
        config.setdefault("reproducibility", {})
        config["reproducibility"]["seed"] = seed
        config["resume"] = True
        config["use_canonical_result_root"] = True
        config["testbed"] = TESTBED_ID
        return config

    def _condition_spec(
        self,
        orchestrator: ExperimentOrchestrator,
        family_spec: FamilySpec,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        condition_specs = orchestrator._enumerate_conditions(config)
        if len(condition_specs) != 1:
            raise RuntimeError(
                f"expected exactly one condition in {family_spec.config_path}"
            )
        condition_spec = condition_specs[0]
        condition_name = orchestrator._condition_name(
            condition_spec["profiles"],
            override=condition_spec.get("name"),
        )
        if condition_name != family_spec.family_name:
            raise RuntimeError(
                f"expected condition `{family_spec.family_name}`, found `{condition_name}`"
            )
        return condition_spec

    def _model_metadata(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        sample_config = self._runtime_config(FAMILY_SPECS[0], SEEDS[0])
        condition_spec = self._condition_spec(orchestrator, FAMILY_SPECS[0], sample_config)
        settings = orchestrator._collect_decoding_settings(
            condition_spec["profiles"],
            effective_profiles=condition_spec.get("effective_profiles"),
        )
        preferred = None
        for role in ("attacker", "defender", "observer"):
            if settings.get(role):
                preferred = dict(settings[role])
                break
        if preferred is None:
            raise RuntimeError("unable to resolve model metadata from ablation profiles")
        return {
            "model_label": MODEL_LABEL,
            "model_id": preferred.get("model_id"),
            "model_path": preferred.get("model_path"),
            "model_quantization": preferred.get("quantization"),
            "context_limit": preferred.get("context_limit"),
            "temperature": preferred.get("temperature"),
            "top_p": preferred.get("top_p"),
            "backend": preferred.get("backend"),
            "framework": preferred.get("framework"),
            "local_files_only": preferred.get("local_files_only"),
            "require_cuda": preferred.get("require_cuda"),
        }

    def _model_registry_payload(self) -> dict[str, Any]:
        return {
            "experiment_version": EXPERIMENT_VERSION,
            "models": [
                {
                    **self._model_metadata(),
                    "source_profiles": [
                        "white_box_attacker",
                        "agentic_defender",
                        "agentic_observer",
                    ],
                }
            ],
        }

    def _prompt_hash_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        tracked_paths = [
            ABLATED_PROMPT_PATH,
            DEFENDER_PROMPT_PATH,
            OBSERVER_PROMPT_PATH,
            *seeded_backbone.PROFILE_REGISTRY_PATHS,
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
                "observer_prompt": file_hashes[portable_path(OBSERVER_PROMPT_PATH)],
                "profile_registries": {
                    portable_path(path): file_hashes[portable_path(path)]
                    for path in seeded_backbone.PROFILE_REGISTRY_PATHS
                },
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

    def _action_space_hash_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        source_hashes = {
            portable_path(path): _sha256_path(path)
            for path in ACTION_SPACE_SOURCE_PATHS
            if path.exists()
        }
        family_hashes: dict[str, dict[str, Any]] = {}
        for family in FAMILY_SPECS:
            config = self._runtime_config(family, SEEDS[0])
            condition_spec = self._condition_spec(orchestrator, family, config)
            effective_profiles = condition_spec.get("effective_profiles") or {}
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
                    "knowledge_files": profile.get("knowledge_files", []),
                    "white_box_knowledge_ablation": profile.get(
                        "white_box_knowledge_ablation"
                    ),
                }
            containment_payload = {
                "runtime_source_hashes": source_hashes,
                "roles": role_payload,
                "containment_semantics_version": CONTAINMENT_SEMANTICS_VERSION,
            }
            family_hashes[family.family_name] = {
                "bundle_hash": _canonical_json_hash(containment_payload),
                "bundle": containment_payload,
            }
        return {
            "experiment_version": EXPERIMENT_VERSION,
            "source_hashes": source_hashes,
            "family_action_space_hashes": family_hashes,
        }

    def _containment_semantics_text(self) -> str:
        return "\n".join(
            [
                CONTAINMENT_SEMANTICS_VERSION,
                "",
                "Concrete runtime containment semantics frozen for this white-box ablation pass:",
                "- runtime-enforced surface isolation remains enabled",
                "- defender containment actions: block_sensor_write, quarantine_signal_source, disable_hmi_session, block_process2_manual_path, isolate_engineering_host",
                "- symbolic containment remains disabled",
                "- white-box baseline on August 4, 2026 was reused as a read-only reference",
                "",
                "Primary source files:",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'action_semantics.py')}",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'environment.py')}",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'orchestrator.py')}",
                "",
            ]
        )

    def _expected_metadata_payloads(self) -> dict[Path, tuple[str, Any]]:
        manifest_rows = self.manifest_rows()
        split_rows = self.split_rows()
        return {
            self.manifest_path: ("csv", manifest_rows),
            self.splits_path: ("csv", split_rows),
            self.seeds_path: ("text", "\n".join(str(seed) for seed in SEEDS) + "\n"),
            self.model_registry_path: ("yaml", self._model_registry_payload()),
            self.experiment_version_path: ("text", EXPERIMENT_VERSION + "\n"),
            self.prompt_hashes_path: ("json", self._prompt_hash_payload()),
            self.action_space_hashes_path: ("json", self._action_space_hash_payload()),
            self.containment_semantics_path: (
                "text",
                self._containment_semantics_text(),
            ),
            self.logger_schema_path: ("json", _infer_logger_schema()),
            self.metrics_schema_path: ("json", _infer_metrics_schema()),
        }

    def freeze_metadata(self) -> None:
        self.ensure_root_layout()
        self._ensure_env_defaults(SEEDS[0])
        completed_runs = _read_json(self.completed_runs_path, {})
        has_progress = bool(completed_runs)
        for actual_path, (kind, payload) in self._expected_metadata_payloads().items():
            if kind == "csv":
                expected_rows = list(payload)
                if actual_path.exists():
                    with actual_path.open(newline="") as handle:
                        actual_rows = list(csv.DictReader(handle))
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
        self._sync_all_ledgers()
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

    def _seed_status_path(self, family_name: str, seed: int) -> Path:
        return self.seed_progress_root(family_name, seed) / "seed_status.json"

    def _scenario_status_path(self, family_name: str, seed: int) -> Path:
        return self.seed_progress_root(family_name, seed) / "scenario_status.csv"

    def _seed_completed_runs_path(self, family_name: str, seed: int) -> Path:
        return self.seed_progress_root(family_name, seed) / "seed_completed_runs.json"

    def _seed_failed_runs_path(self, family_name: str, seed: int) -> Path:
        return self.seed_progress_root(family_name, seed) / "seed_failed_runs.json"

    def _seed_last_completed_path(self, family_name: str, seed: int) -> Path:
        return self.seed_progress_root(family_name, seed) / "last_completed_scenario.txt"

    def _family_status_path(self, family_name: str) -> Path:
        return self.family_progress_root(family_name) / "family_status.json"

    def _family_completed_runs_path(self, family_name: str) -> Path:
        return self.family_progress_root(family_name) / "family_completed_runs.json"

    def _family_failed_runs_path(self, family_name: str) -> Path:
        return self.family_progress_root(family_name) / "family_failed_runs.json"

    def _initial_scenario_rows(
        self,
        family_name: str,
        seed: int,
    ) -> list[dict[str, Any]]:
        split_map = _scenario_split_map(self.split_rows())
        rows: list[dict[str, Any]] = []
        for index, scenario in enumerate(self.scenarios()):
            rows.append(
                {
                    "manifest_order": index + 1,
                    "attacker_family": family_name,
                    "seed": str(seed),
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
                    "run_dir": portable_path(
                        self.seed_root(family_name, seed)
                        / scenario.suite
                        / scenario.scenario_id
                        / "trial-001"
                    ),
                    "archived_interrupted_dir": "",
                }
            )
        return rows

    def _load_scenario_rows(self, family_name: str, seed: int) -> list[dict[str, Any]]:
        path = self._scenario_status_path(family_name, seed)
        if not path.exists():
            rows = self._initial_scenario_rows(family_name, seed)
            write_csv(path, rows)
            return rows
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != RUNS_PER_SEED:
            raise RuntimeError(
                f"expected {RUNS_PER_SEED} rows in {path}, found {len(rows)}"
            )
        return rows

    def _save_scenario_rows(
        self,
        family_name: str,
        seed: int,
        rows: list[dict[str, Any]],
    ) -> None:
        write_csv(self._scenario_status_path(family_name, seed), rows)

    @staticmethod
    def _expected_environment(config: dict[str, Any]) -> dict[str, Any]:
        return dict(config.get("environment") or {})

    def _run_dir(self, family_name: str, seed: int, scenario: Any) -> Path:
        return (
            self.seed_root(family_name, seed)
            / scenario.suite
            / scenario.scenario_id
            / "trial-001"
        )

    def _run_id(self, family_spec: FamilySpec, seed: int, scenario: Any, split: str) -> str:
        return "_".join(
            [
                TESTBED_ID,
                _sanitize_token(scenario.scenario_id),
                split,
                family_spec.attacker_profile_id,
                "fullAgenticDefender",
                "Qwen35_9B",
                f"seed{seed}",
            ]
        )

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

    def _load_run_summary(self, run_dir: Path) -> dict[str, Any]:
        summary = _read_json(run_dir / "summary.json", {})
        metrics = _read_json(run_dir / "metrics.json", {})
        if isinstance(summary, dict) and "metrics" not in summary:
            summary["metrics"] = metrics
        return summary

    def _archive_partial_run(
        self,
        family_name: str,
        seed: int,
        scenario_id: str,
        run_dir: Path,
    ) -> str:
        archive_root = self.seed_progress_root(family_name, seed) / "interrupted_artifacts"
        archive_root.mkdir(parents=True, exist_ok=True)
        destination = archive_root / f"{scenario_id}__{utc_now().replace(':', '').replace('-', '')}"
        counter = 1
        while destination.exists():
            counter += 1
            destination = archive_root / f"{scenario_id}__{counter}"
        shutil.move(str(run_dir), str(destination))
        return portable_path(destination)

    def _metadata_hash_context(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            dict(_read_json(self.prompt_hashes_path, {})),
            dict(_read_json(self.action_space_hashes_path, {})),
        )

    def _seed_completed_row(
        self,
        *,
        family_spec: FamilySpec,
        seed: int,
        scenario: Any,
        run_dir: Path,
        summary: dict[str, Any],
        split: str,
        model_metadata: dict[str, Any],
        prompt_hashes: dict[str, Any],
        action_space_hashes: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = dict(summary.get("metrics") or {})
        attack_metrics = dict(metrics.get("attacker") or {})
        family_prompt_hash = (
            prompt_hashes.get("family_prompt_hashes", {})
            .get(family_spec.family_name, {})
            .get("bundle_hash")
        )
        family_action_space_hash = (
            action_space_hashes.get("family_action_space_hashes", {})
            .get(family_spec.family_name, {})
            .get("bundle_hash")
        )
        runtime_hash_payload = {
            "family": family_spec.family_name,
            "seed": seed,
            "scenario": scenario.scenario_id,
            "scenario_config_path": portable_path(scenario.execution_path),
            "experiment_config_path": portable_path(family_spec.config_path),
            "experiment_version": EXPERIMENT_VERSION,
        }
        return {
            "progress_key": f"{family_spec.family_name}::{seed}::{scenario.scenario_id}",
            "run_id": summary.get("run_id"),
            "experiment_version": EXPERIMENT_VERSION,
            "git_commit": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "timestamp_start_utc": summary.get("timestamp_start_utc"),
            "timestamp_end_utc": summary.get("timestamp_end_utc"),
            "testbed": TESTBED_ID,
            "scenario_id": scenario.scenario_id,
            "scenario_family": _scenario_family_for(scenario.scenario_id),
            "split": split,
            "seed": seed,
            "model_label": model_metadata.get("model_label"),
            "model_id": model_metadata.get("model_id"),
            "model_quantization": model_metadata.get("model_quantization"),
            "context_limit": model_metadata.get("context_limit"),
            "temperature": model_metadata.get("temperature"),
            "top_p": model_metadata.get("top_p"),
            "prompt_hash": family_prompt_hash,
            "config_hash": _canonical_json_hash(runtime_hash_payload),
            "action_space_hash": family_action_space_hash,
            "attacker_regime": family_spec.attacker_profile_id,
            "defender_variant": "full_agentic_defender",
            "knowledge_variant": family_spec.knowledge_variant,
            "observability_variant": family_spec.knowledge_variant,
            "robustness_variant": "white_box_ablation",
            "containment_semantics_version": CONTAINMENT_SEMANTICS_VERSION,
            "status": summary.get("status"),
            "error_message": "",
            "family_name": family_spec.family_name,
            "run_dir": portable_path(run_dir),
            "catalog_id": scenario.catalog_id,
            "detected": int(
                metrics.get("defender_detection", {}).get("time_to_detection_sec")
                is not None
            ),
            "defender_acted": int((metrics.get("defender_action_count") or 0) > 0),
            "safety_violation": int(bool(attack_metrics.get("caused_safety_violation"))),
            "wall_clock_time_to_detection_from_attack_start_sec": metrics.get(
                "wall_clock_time_to_detection_from_attack_start_sec"
            ),
            "attacker_action_count": metrics.get("attacker_action_count"),
            "downtime_timestep_count": metrics.get("downtime_timestep_count"),
            "summary_path": portable_path(run_dir / "summary.json"),
            "metrics_path": portable_path(run_dir / "metrics.json"),
        }

    def _seed_failure_row(
        self,
        *,
        family_spec: FamilySpec,
        seed: int,
        scenario_id: str,
        split: str,
        run_dir: Path,
        error_message: str,
    ) -> dict[str, Any]:
        return {
            "progress_key": f"{family_spec.family_name}::{seed}::{scenario_id}",
            "attacker_family": family_spec.family_name,
            "seed": seed,
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
        family_spec: FamilySpec,
        seed: int,
        completed_run_rows: list[dict[str, Any]],
        run_summaries: list[dict[str, Any]],
        reproducibility: dict[str, Any],
    ) -> None:
        orchestrator = self._orchestrator()
        attack_metrics = orchestrator._aggregate_attack_metrics(
            run_summaries,
            primary_timing_origin="wargame_t0",
        )
        trajectory_metrics = orchestrator._aggregate_trajectory_metrics(run_summaries)
        mitre_phase_metrics = orchestrator._aggregate_mitre_phase_metrics(run_summaries)
        nominal_metrics = orchestrator._aggregate_nominal_metrics(run_summaries)
        split_counts: dict[str, int] = {}
        for row in completed_run_rows:
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        seed_root = self.seed_root(family_spec.family_name, seed)
        condition_summary = {
            "condition_name": family_spec.family_name,
            "seed": seed,
            "status": "completed" if len(completed_run_rows) == RUNS_PER_SEED else "partial",
            "experiment_version": EXPERIMENT_VERSION,
            "testbed": TESTBED_ID,
            "runs": run_summaries,
            "completed_run_count": len(completed_run_rows),
            "expected_run_count": RUNS_PER_SEED,
            "split_counts": split_counts,
            "reproducibility": reproducibility,
            "attack_metrics": attack_metrics,
            "trajectory_metrics": trajectory_metrics,
            "mitre_phase_metrics": mitre_phase_metrics,
            "nominal_metrics": nominal_metrics,
        }
        write_json(seed_root / "condition_summary.json", condition_summary)
        write_json(seed_root / "attack_metrics_summary.json", attack_metrics)
        write_json(seed_root / "trajectory_metrics_summary.json", trajectory_metrics)
        write_json(seed_root / "mitre_phase_metrics.json", mitre_phase_metrics)
        write_json(seed_root / "nominal_metrics_summary.json", nominal_metrics)

        summary_row = {
            "attacker_family": family_spec.family_name,
            "seed": seed,
            "completed_scenarios": len(completed_run_rows),
            "expected_scenarios": RUNS_PER_SEED,
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
            "experiment_version": EXPERIMENT_VERSION,
            "seed": seed,
            "family_name": family_spec.family_name,
            "completed_scenarios": len(completed_run_rows),
            "expected_scenarios": RUNS_PER_SEED,
            "split_counts": split_counts,
            "attack_metrics": attack_metrics,
            "trajectory_metrics": trajectory_metrics,
            "summary_row": summary_row,
        }
        write_json(seed_root / "seed_summary.json", seed_summary)
        write_csv(seed_root / "seed_summary.csv", [summary_row])

    def _sync_seed_state(
        self,
        *,
        family_spec: FamilySpec,
        seed: int,
        expected_environment: dict[str, Any],
        reproducibility: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        model_metadata = self._model_metadata()
        prompt_hashes, action_space_hashes = self._metadata_hash_context()
        scenario_rows = self._load_scenario_rows(family_spec.family_name, seed)
        scenario_by_id = {scenario.scenario_id: scenario for scenario in self.scenarios()}
        completed_rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        run_summaries: list[dict[str, Any]] = []
        updated_rows: list[dict[str, Any]] = []
        for row in scenario_rows:
            scenario = scenario_by_id[row["scenario_id"]]
            run_dir = self._run_dir(family_spec.family_name, seed, scenario)
            if row["status"] == "completed" and self._verify_run_completion(
                run_dir,
                expected_environment=expected_environment,
            ):
                summary = self._load_run_summary(run_dir)
                completed_row = self._seed_completed_row(
                    family_spec=family_spec,
                    seed=seed,
                    scenario=scenario,
                    run_dir=run_dir,
                    summary=summary,
                    split=row["split"],
                    model_metadata=model_metadata,
                    prompt_hashes=prompt_hashes,
                    action_space_hashes=action_space_hashes,
                )
                completed_rows.append(completed_row)
                run_summaries.append(summary)
            else:
                if row["status"] == "completed":
                    row = dict(row)
                    row["status"] = "pending"
                    row["completed_at_utc"] = ""
                    row["last_error"] = (
                        "completed marker existed but required artifacts were incomplete"
                    )
                if row["status"] == "failed":
                    failed_rows.append(
                        self._seed_failure_row(
                            family_spec=family_spec,
                            seed=seed,
                            scenario_id=row["scenario_id"],
                            split=row["split"],
                            run_dir=run_dir,
                            error_message=row.get("last_error", ""),
                        )
                    )
            updated_rows.append(row)
        self._save_scenario_rows(family_spec.family_name, seed, updated_rows)
        write_json(
            self._seed_completed_runs_path(family_spec.family_name, seed),
            {row["scenario_id"]: row for row in completed_rows},
        )
        write_json(
            self._seed_failed_runs_path(family_spec.family_name, seed),
            {row["scenario_id"]: row for row in failed_rows},
        )
        self._write_seed_summaries(
            family_spec=family_spec,
            seed=seed,
            completed_run_rows=completed_rows,
            run_summaries=run_summaries,
            reproducibility=reproducibility,
        )
        status_payload = {
            "updated_utc": utc_now(),
            "experiment_version": EXPERIMENT_VERSION,
            "family_name": family_spec.family_name,
            "seed": seed,
            "status": "completed" if len(completed_rows) == RUNS_PER_SEED else "in_progress",
            "completed_scenarios": len(completed_rows),
            "failed_scenarios": len(failed_rows),
            "pending_scenarios": RUNS_PER_SEED - len(completed_rows) - len(failed_rows),
            "last_completed_scenario": (
                completed_rows[-1]["scenario_id"] if completed_rows else None
            ),
        }
        write_json(self._seed_status_path(family_spec.family_name, seed), status_payload)
        if completed_rows:
            _write_text(
                self._seed_last_completed_path(family_spec.family_name, seed),
                completed_rows[-1]["scenario_id"] + "\n",
            )
        self._sync_all_ledgers()
        return completed_rows, failed_rows

    def _sync_all_ledgers(self) -> None:
        manifest_rows = self.manifest_rows()
        all_completed: dict[str, Any] = {}
        all_failed: dict[str, Any] = {}
        queue_rows: list[dict[str, Any]] = []

        for family in FAMILY_SPECS:
            family_completed_rows: dict[str, Any] = {}
            family_failed_rows: dict[str, Any] = {}
            for seed in SEEDS:
                rows = self._load_scenario_rows(family.family_name, seed)
                completed_payload = dict(
                    _read_json(self._seed_completed_runs_path(family.family_name, seed), {})
                )
                failed_payload = dict(
                    _read_json(self._seed_failed_runs_path(family.family_name, seed), {})
                )
                for row in rows:
                    scenario_key = f"{family.family_name}::{seed}::{row['scenario_id']}"
                    queue_rows.append(
                        {
                            "attacker_family": family.family_name,
                            "seed": seed,
                            "scenario_id": row["scenario_id"],
                            "split": row["split"],
                            "status": row["status"],
                            "attempt_count": row["attempt_count"],
                            "run_id": row["run_id"],
                            "run_dir": row["run_dir"],
                            "archived_interrupted_dir": row["archived_interrupted_dir"],
                            "last_error": row["last_error"],
                        }
                    )
                    if row["status"] == "completed":
                        record = dict(completed_payload.get(row["scenario_id"]) or {})
                        if record:
                            all_completed[scenario_key] = record
                            family_completed_rows[scenario_key] = record
                    elif row["status"] == "failed":
                        record = dict(failed_payload.get(row["scenario_id"]) or {})
                        if record:
                            all_failed[scenario_key] = record
                            family_failed_rows[scenario_key] = record
            write_json(
                self._family_completed_runs_path(family.family_name),
                family_completed_rows,
            )
            write_json(
                self._family_failed_runs_path(family.family_name),
                family_failed_rows,
            )
            completed_count = len(family_completed_rows)
            failed_count = len(family_failed_rows)
            family_status = {
                "updated_utc": utc_now(),
                "experiment_version": EXPERIMENT_VERSION,
                "family_name": family.family_name,
                "completed_runs": completed_count,
                "failed_runs": failed_count,
                "pending_runs": RUNS_PER_FAMILY - completed_count - failed_count,
                "status": "completed" if completed_count == RUNS_PER_FAMILY else "in_progress",
                "seed_status": {
                    seed: _read_json(self._seed_status_path(family.family_name, seed), {})
                    for seed in SEEDS
                },
            }
            write_json(self._family_status_path(family.family_name), family_status)

        write_json(self.completed_runs_path, all_completed)
        write_json(self.failed_runs_path, all_failed)
        write_csv(self.run_queue_path, queue_rows)
        catalog_rows = [
            all_completed[key]
            for key in sorted(
                all_completed,
                key=lambda item: (
                    FAMILY_ORDER[item.split("::", 1)[0]],
                    SEEDS.index(int(item.split("::")[1])),
                    next(
                        index
                        for index, row in enumerate(manifest_rows)
                        if row["scenario_id"] == item.split("::")[2]
                    ),
                ),
            )
        ]
        failure_rows = [all_failed[key] for key in sorted(all_failed)]
        write_csv(self.run_catalog_path, catalog_rows)
        write_csv(self.run_failures_path, failure_rows)

        pending_rows = [row for row in queue_rows if row["status"] != "completed"]
        next_run = pending_rows[0] if pending_rows else None
        previous_resume_state = _read_json(self.resume_state_path, {})
        rerun_scenarios = sorted(
            {
                row["scenario_id"]
                for row in queue_rows
                if row.get("attempt_count") not in {"", "0", "1"}
                or row.get("archived_interrupted_dir")
            }
        )
        resume_state = {
            "updated_utc": utc_now(),
            "experiment_version": EXPERIMENT_VERSION,
            "completed_runs": len(catalog_rows),
            "failed_runs": len(failure_rows),
            "pending_runs": len(pending_rows),
            "all_complete": len(catalog_rows) == TOTAL_RUNS,
            "next_run": next_run,
            "resume_event_detected": bool(
                previous_resume_state.get("resume_event_detected")
            ),
            "resume_event_evidence": list(
                previous_resume_state.get("resume_event_evidence") or []
            ),
            "rerun_required_scenarios": [
                row["scenario_id"]
                for row in queue_rows
                if row["status"] in {"failed", "pending"} and row.get("attempt_count") not in {"", "0"}
            ],
            "scenarios_rerun_after_interruption": rerun_scenarios,
        }
        write_json(self.resume_state_path, resume_state)
        self._write_resume_notes(queue_rows, next_run)

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
            f"- Completed runs: `{completed_count}`",
            f"- Failed runs: `{failed_count}`",
            f"- Pending runs: `{pending_count}`",
            f"- Baseline root reused: `{portable_path(BASELINE_SEED_ROOT)}`",
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

    def _ensure_env_defaults(self, seed: int) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        os.environ["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", str(seed))
        os.environ.setdefault("QWEN_MODEL_ID", "Qwen/Qwen3.5-9B")
        if not os.environ.get("QWEN_MODEL_PATH"):
            os.environ["QWEN_MODEL_PATH"] = str(
                (REPO_ROOT / "models" / "local_models" / "qwen3.5-9b").resolve()
            )

    def _validate_optionb_environment(self) -> None:
        payload = _read_json(self.optionb_validation_path, {})
        if payload.get("status") == "passed":
            self._set_phase("validated_optionb_environment", "completed")
            return
        self._set_phase("validated_optionb_environment", "in_progress")
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
                raise RuntimeError("expected sphere_openplc backend for water treatment")
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
                    "local_observation": {
                        "state_name": observation.get("STATE_NAME"),
                    },
                    "discovery_surface_sample": discovery.get("reconnaissance", {}),
                }
            )
            write_json(self.optionb_validation_path, result)
            self._set_phase(
                "validated_optionb_environment",
                "completed",
                validation_path=portable_path(self.optionb_validation_path),
            )
            return
        except Exception as exc:
            result["error"] = str(exc)
            write_json(self.optionb_validation_path, result)
            raise
        finally:
            try:
                env.stop()
            except Exception:
                pass

    def _mark_resume_event_if_needed(self, rows: list[dict[str, Any]]) -> None:
        evidence = [
            {
                "attacker_family": row["attacker_family"],
                "seed": row["seed"],
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

    def run_controller(self) -> int:
        self.freeze_metadata()
        self._mark_resume_event_if_needed(
            list(csv.DictReader(self.run_queue_path.open(newline="")))
            if self.run_queue_path.exists()
            else []
        )
        self._validate_optionb_environment()
        self._validate_runtime_and_tests()
        self._set_phase("running_grid", "in_progress")
        _append_log(
            self.experiment_log_path,
            f"controller starting white-box ablation under {portable_path(self.result_root)}",
        )

        for family in FAMILY_SPECS:
            for seed in SEEDS:
                seed_status = _read_json(self._seed_status_path(family.family_name, seed), {})
                if seed_status.get("completed_scenarios") == RUNS_PER_SEED:
                    continue
                worker_log_path = self.seed_progress_root(family.family_name, seed) / "worker_run.json"
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
        self.write_comparison_artifacts()
        self.write_report()
        self.write_repo_note()
        return 0

    def run_worker(self, family_name: str, seed: int, *, max_attempts: int) -> int:
        family_spec = FAMILY_BY_NAME[family_name]
        self.ensure_root_layout()
        self._ensure_env_defaults(seed)
        config = self._runtime_config(family_spec, seed)
        orchestrator = self._orchestrator()
        condition_spec = self._condition_spec(orchestrator, family_spec, config)
        expected_environment = self._expected_environment(config)
        reproducibility = orchestrator._configure_reproducibility(
            config=config,
            conditions=[condition_spec],
        )
        orchestrator._current_reproducibility = reproducibility
        self._mark_resume_event_if_needed(self._load_scenario_rows(family_name, seed))
        completed_rows, _ = self._sync_seed_state(
            family_spec=family_spec,
            seed=seed,
            expected_environment=expected_environment,
            reproducibility=reproducibility,
        )
        if len(completed_rows) == RUNS_PER_SEED:
            return 0

        scenario_rows = self._load_scenario_rows(family_name, seed)
        scenario_by_id = {scenario.scenario_id: scenario for scenario in self.scenarios()}
        for row in scenario_rows:
            if row["status"] == "completed":
                continue
            scenario = scenario_by_id[row["scenario_id"]]
            run_dir = self._run_dir(family_name, seed, scenario)
            archived_path = ""
            if run_dir.exists():
                archived_path = self._archive_partial_run(
                    family_name,
                    seed,
                    scenario.scenario_id,
                    run_dir,
                )
            attempts = int(row["attempt_count"] or "0")
            success = False
            last_error = ""
            for _ in range(max_attempts):
                attempts += 1
                row["status"] = "running"
                row["attempt_count"] = str(attempts)
                row["started_at_utc"] = utc_now()
                row["run_id"] = self._run_id(family_spec, seed, scenario, row["split"])
                row["archived_interrupted_dir"] = archived_path
                self._save_scenario_rows(family_name, seed, scenario_rows)
                self._sync_all_ledgers()

                reproducibility = orchestrator._configure_reproducibility(
                    config=config,
                    conditions=[condition_spec],
                )
                orchestrator._current_reproducibility = reproducibility
                bundle = orchestrator._run_single(
                    experiment_id=EXPERIMENT_VERSION,
                    condition_name=family_name,
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
                self._save_scenario_rows(family_name, seed, scenario_rows)
                self._sync_all_ledgers()
                if run_dir.exists():
                    archived_path = self._archive_partial_run(
                        family_name,
                        seed,
                        scenario.scenario_id,
                        run_dir,
                    )
                    row["archived_interrupted_dir"] = archived_path
            self._save_scenario_rows(family_name, seed, scenario_rows)
            completed_rows, failed_rows = self._sync_seed_state(
                family_spec=family_spec,
                seed=seed,
                expected_environment=expected_environment,
                reproducibility=reproducibility,
            )
            if not success:
                _append_log(
                    self.experiment_log_path,
                    f"scenario failed after {attempts} attempts: {family_name} seed {seed} scenario {scenario.scenario_id}: {last_error}",
                )
                return 1
            _append_log(
                self.experiment_log_path,
                f"completed {family_name} seed {seed} scenario {scenario.scenario_id}",
            )
            if len(completed_rows) == RUNS_PER_SEED and not failed_rows:
                break
        return 0

    def write_family_aggregates(self) -> None:
        self._set_phase("aggregating", "in_progress")
        with self.run_catalog_path.open(newline="") as handle:
            catalog_rows = list(csv.DictReader(handle))
        for family in FAMILY_SPECS:
            family_rows = [
                row for row in catalog_rows if row["family_name"] == family.family_name
            ]
            if len(family_rows) != RUNS_PER_FAMILY:
                continue
            seed_summary = _read_json(
                self.seed_root(family.family_name, SEEDS[0]) / "seed_summary.json", {}
            )
            summary_row = dict(seed_summary.get("summary_row") or {})
            aggregate_row = {
                "family_name": family.family_name,
                "seed": SEEDS[0],
                "total_runs": RUNS_PER_FAMILY,
                "total_completed_runs": len(family_rows),
                "safety_violation_count": sum(
                    int(row["safety_violation"]) for row in family_rows
                ),
                "detected_count": sum(int(row["detected"]) for row in family_rows),
                "defender_action_count": sum(
                    int(row["defender_acted"]) for row in family_rows
                ),
                "mean_wall_clock_detection_from_attack_start_sec": summary_row.get(
                    "avg_wall_clock_time_to_detection_from_attack_start_sec"
                ),
                "std_wall_clock_detection_from_attack_start_sec": 0.0,
                "mean_attacker_action_count": summary_row.get("avg_attacker_action_count"),
                "std_attacker_action_count": 0.0,
                "mean_downtime_timestep_count": summary_row.get(
                    "avg_downtime_timestep_count"
                ),
                "std_downtime_timestep_count": 0.0,
            }
            family_root = self.family_root(family.family_name)
            write_json(family_root / "aggregate_over_seed_101.json", aggregate_row)
            write_csv(family_root / "aggregate_over_seed_101.csv", [aggregate_row])
            report_lines = [
                f"# {family.family_name}",
                "",
                f"- Seed: `{SEEDS[0]}`",
                f"- Completed runs: `{aggregate_row['total_completed_runs']}/{aggregate_row['total_runs']}`",
                f"- Safety-violation count: `{aggregate_row['safety_violation_count']}`",
                f"- Detected count: `{aggregate_row['detected_count']}`",
                f"- Defender-action count: `{aggregate_row['defender_action_count']}`",
                f"- Wall-clock detection from attack start: `{_format_mean_std(aggregate_row['mean_wall_clock_detection_from_attack_start_sec'], aggregate_row['std_wall_clock_detection_from_attack_start_sec'])}`",
                f"- Attacker action count: `{_format_mean_std(aggregate_row['mean_attacker_action_count'], aggregate_row['std_attacker_action_count'])}`",
                f"- Downtime burden: `{_format_mean_std(aggregate_row['mean_downtime_timestep_count'], aggregate_row['std_downtime_timestep_count'])}`",
            ]
            _write_text(family_root / "family_report.md", "\n".join(report_lines) + "\n")
        self._set_phase("aggregating", "completed")

    @staticmethod
    def _summary_metric(summary_row: dict[str, Any], key: str) -> Any:
        value = summary_row.get(key)
        if value in {"", "None"}:
            return None
        return value

    def _comparison_rows(self) -> list[dict[str, Any]]:
        baseline_summary = dict(
            _read_json(BASELINE_SEED_ROOT / "seed_summary.json", {}).get("summary_row") or {}
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
            "# Water-Treatment White-Box Ablation Report",
            "",
            f"- Full white-box seed-101 baseline source: `{portable_path(BASELINE_SEED_ROOT)}`",
            "- Ablations run:",
            "  - `white_box_no_safety_bounds__agentic_defender`: explicit unsafe thresholds and published safety limits removed.",
            "  - `white_box_no_surface_priors__agentic_defender`: direct leverage ranking and safety-critical target hints removed.",
            "  - `white_box_no_control_semantics__agentic_defender`: operational meaning for named signals, commands, actuators, and interlocks removed.",
            f"- Seed used: `{SEEDS[0]}` only.",
            f"- Treatment suite size: `{RUNS_PER_SEED}` attack scenarios; suite remained fixed.",
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
