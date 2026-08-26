from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable

from ...config import ConfigError, load_config, require_fields
from ...environment import OpenPLCEnvironment, _SphereOpenPLCBackend
from ...mitre import verify_optionb_staged_snapshot
from ...orchestrator import ExperimentOrchestrator
from ...paths import REPO_ROOT, WORKSPACE_ROOT, portable_path, profile_registry_paths
from ...reporting import write_csv, write_json
from ...scenarios import ScenarioCatalog
from ...utils import dump_yaml, load_yaml, stable_hash, utc_now


TESTBED_ID = "water_treatment"
PROFILE_REGISTRY_PATHS = profile_registry_paths(TESTBED_ID)
EXPERIMENT_VERSION = "water_treatment_seeded_backbone_qwen35_9b_v1"
CONTAINMENT_SEMANTICS_VERSION = "current_concrete_containment"
MODEL_LABEL = "Qwen3.5-9B-local-bf16"
CANONICAL_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "Water-treatment"
    / "seeded_backbone_qwen35_9b"
)
PAPER_REPO_ROOT = WORKSPACE_ROOT / "ics-agent-paper"
PAPER_TABLE_PATH = PAPER_REPO_ROOT / "Tables" / "results-attackers-agentic.tex"
PAPER_RESULTS_PATH = PAPER_REPO_ROOT / "Sections" / "results.tex"
SEEDS = [101, 202, 303, 404, 505]
LEGACY_RESULT_ROOTS = [
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "Water-treatment"
    / "DefendervsBlackBoxAttackerConcreteContainment",
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "Water-treatment"
    / "DefendervsGrayBoxAttackerConcreteContainment",
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "Water-treatment"
    / "DefendervsWhiteBoxAttackerConcreteContainment",
]
FROZEN_PROMPT_PATHS = [
    REPO_ROOT / "testbeds" / "water_treatment" / "prompts" / "black_box_attacker.md",
    REPO_ROOT / "testbeds" / "water_treatment" / "prompts" / "gray_box_attacker.md",
    REPO_ROOT / "testbeds" / "water_treatment" / "prompts" / "white_box_attacker.md",
    REPO_ROOT / "testbeds" / "water_treatment" / "prompts" / "defender.md",
    REPO_ROOT / "testbeds" / "water_treatment" / "prompts" / "observer.md",
]
FROZEN_CONFIG_PATHS = [
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_black_box_attacker_vs_agentic_full_containment_defender.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_gray_box_attacker_vs_agentic_full_containment_defender.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_white_box_attacker_vs_agentic_full_containment_defender.yaml",
]
ACTION_SPACE_SOURCE_PATHS = [
    REPO_ROOT / "ics_agent" / "action_semantics.py",
    REPO_ROOT / "ics_agent" / "environment.py",
    REPO_ROOT / "ics_agent" / "orchestrator.py",
    *PROFILE_REGISTRY_PATHS,
    *FROZEN_CONFIG_PATHS,
]
REQUIRED_RUN_FILES = (
    "summary.json",
    "metrics.json",
    "actions.json",
    "alerts.json",
    "config_snapshot.yaml",
    "system_events.json",
    "environment_snapshot.json",
    "phase_log.jsonl",
)
TOP_LEVEL_PHASES = [
    "manifest_frozen",
    "validated_optionb_environment",
    "validation_tests",
    "running_grid",
    "aggregating",
    "paper_export",
    "paper_update",
]


@dataclass(frozen=True)
class FamilySpec:
    family_name: str
    attacker_profile_id: str
    knowledge_variant: str
    config_path: Path


FAMILY_SPECS = [
    FamilySpec(
        family_name="black_box_attacker__agentic_defender",
        attacker_profile_id="black_box_attacker",
        knowledge_variant="black_box",
        config_path=FROZEN_CONFIG_PATHS[0],
    ),
    FamilySpec(
        family_name="gray_box_attacker__agentic_defender",
        attacker_profile_id="gray_box_attacker",
        knowledge_variant="gray_box",
        config_path=FROZEN_CONFIG_PATHS[1],
    ),
    FamilySpec(
        family_name="white_box_attacker__agentic_defender",
        attacker_profile_id="white_box_attacker",
        knowledge_variant="white_box",
        config_path=FROZEN_CONFIG_PATHS[2],
    ),
]
FAMILY_BY_NAME = {item.family_name: item for item in FAMILY_SPECS}
FAMILY_ORDER = {item.family_name: index for index, item in enumerate(FAMILY_SPECS)}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume the canonical seeded water-treatment backbone."
    )
    subparsers = parser.add_subparsers(dest="mode")

    controller = subparsers.add_parser("controller")
    controller.add_argument(
        "--result-root",
        default=str(CANONICAL_RESULT_ROOT),
        help="Canonical seeded result root.",
    )

    worker = subparsers.add_parser("worker")
    worker.add_argument("--family", required=True, choices=sorted(FAMILY_BY_NAME))
    worker.add_argument("--seed", required=True, type=int, choices=SEEDS)
    worker.add_argument(
        "--result-root",
        default=str(CANONICAL_RESULT_ROOT),
        help="Canonical seeded result root.",
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


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(f"[{utc_now()}] {message}\n")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _mean(values: Iterable[float]) -> float | None:
    prepared = [float(value) for value in values]
    if not prepared:
        return None
    return round(sum(prepared) / len(prepared), 4)


def _stdev(values: Iterable[float]) -> float | None:
    prepared = [float(value) for value in values]
    if len(prepared) < 2:
        return 0.0 if prepared else None
    return round(statistics.stdev(prepared), 4)


def _format_mean_std(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None:
        return "n/a"
    std_rendered = 0.0 if std_value is None else std_value
    return f"{mean_value:.2f} ± {std_rendered:.2f}"


def _sanitize_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _scenario_family_for(scenario_id: str) -> str:
    if "hmi" in scenario_id:
        return "hmi_control"
    if "p2-" in scenario_id:
        return "process2_manual_path"
    if "sensor-rw-" in scenario_id:
        return "raw_water_sensor"
    if "sensor-uf-" in scenario_id:
        return "uf_sensor"
    return "other"


def _attack_family_for(scenario_id: str) -> str:
    if "hmi" in scenario_id:
        return "hmi_session_abuse"
    if "p2-" in scenario_id:
        return "manual_dosing_abuse"
    if "sensor-" in scenario_id:
        return "sensor_spoofing"
    return "other"


def _split_for_index(index: int) -> str:
    if index < 12:
        return "train_dev"
    if index < 15:
        return "validation"
    return "heldout_test"


def build_manifest_rows(scenarios: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        rows.append(
            {
                "testbed": TESTBED_ID,
                "scenario_id": scenario.scenario_id,
                "scenario_family": _scenario_family_for(scenario.scenario_id),
                "attack_family": _attack_family_for(scenario.scenario_id),
                "default_initial_condition": json.dumps(
                    scenario.execution.get("initial_conditions", {}),
                    sort_keys=True,
                ),
                "scenario_config_path": portable_path(scenario.execution_path),
                "enabled_for_final": "true",
                "notes": scenario.execution.get("description", "").strip(),
            }
        )
    return rows


def build_split_rows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(manifest_rows):
        rows.append(
            {
                "testbed": row["testbed"],
                "scenario_id": row["scenario_id"],
                "manifest_order": index + 1,
                "split": _split_for_index(index),
            }
        )
    return rows


def _scenario_split_map(split_rows: list[dict[str, Any]]) -> dict[str, str]:
    return {row["scenario_id"]: row["split"] for row in split_rows}


def _schema_for_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": ["null"]}
    if isinstance(value, bool):
        return {"type": ["boolean"]}
    if isinstance(value, int):
        return {"type": ["integer"]}
    if isinstance(value, float):
        return {"type": ["number"]}
    if isinstance(value, str):
        return {"type": ["string"]}
    if isinstance(value, list):
        schema = {"type": ["array"], "items": {}}
        if not value:
            return schema
        item_schema: dict[str, Any] = {}
        for item in value:
            item_schema = _merge_schema(item_schema, _schema_for_value(item))
        schema["items"] = item_schema
        return schema
    if isinstance(value, dict):
        return {
            "type": ["object"],
            "properties": {key: _schema_for_value(item) for key, item in value.items()},
        }
    return {"type": [type(value).__name__]}


def _normalize_csv_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append(
            {
                str(key): "" if value is None else str(value)
                for key, value in row.items()
            }
        )
    return normalized


def _merge_schema(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if not left:
        return deepcopy(right)
    if not right:
        return deepcopy(left)
    merged: dict[str, Any] = {}
    merged["type"] = sorted(
        {
            *list(left.get("type", [])),
            *list(right.get("type", [])),
        }
    )
    left_props = left.get("properties")
    right_props = right.get("properties")
    if isinstance(left_props, dict) or isinstance(right_props, dict):
        props: dict[str, Any] = {}
        for key in sorted({*(left_props or {}).keys(), *(right_props or {}).keys()}):
            props[key] = _merge_schema(
                dict((left_props or {}).get(key) or {}),
                dict((right_props or {}).get(key) or {}),
            )
        merged["properties"] = props
    left_items = left.get("items")
    right_items = right.get("items")
    if isinstance(left_items, dict) or isinstance(right_items, dict):
        merged["items"] = _merge_schema(
            dict(left_items or {}),
            dict(right_items or {}),
        )
    return merged


def _infer_metrics_schema() -> dict[str, Any]:
    schema: dict[str, Any] = {}
    sources: list[str] = []
    for legacy_root in LEGACY_RESULT_ROOTS:
        for path in sorted(legacy_root.glob("*/*/*/*/metrics.json")):
            payload = _read_json(path, {})
            if not isinstance(payload, dict):
                continue
            sources.append(portable_path(path))
            schema = _merge_schema(schema, _schema_for_value(payload))
    return {
        "sources": sources,
        "schema": schema,
    }


def _infer_logger_schema() -> dict[str, Any]:
    schema: dict[str, Any] = {}
    events: set[str] = set()
    phases: set[str] = set()
    sources: list[str] = []
    for legacy_root in LEGACY_RESULT_ROOTS:
        for path in sorted(legacy_root.glob("*/*/*/*/phase_log.jsonl")):
            sources.append(portable_path(path))
            with path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        continue
                    events.add(str(payload.get("event", "")))
                    phases.add(str(payload.get("phase", "")))
                    schema = _merge_schema(schema, _schema_for_value(payload))
    return {
        "sources": sources,
        "schema": schema,
        "phase_values": sorted(item for item in phases if item),
        "event_values": sorted(item for item in events if item),
    }


def _python_executable() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python.resolve())
    return sys.executable


class SeededBackboneRunner:
    def __init__(self, result_root: Path) -> None:
        self.result_root = result_root.resolve()
        self.progress_root = self.result_root / "_progress"
        self.paper_export_root = self.result_root / "paper_export"
        self.global_phase_status_path = self.progress_root / "global_phase_status.json"
        self.completed_runs_path = self.progress_root / "completed_runs.json"
        self.failed_runs_path = self.progress_root / "failed_runs.json"
        self.resume_state_path = self.progress_root / "resume_state.json"
        self.last_step_path = self.progress_root / "last_successful_step.txt"
        self.run_queue_path = self.progress_root / "run_queue.csv"
        self.experiment_log_path = self.progress_root / "experiment_run.log"
        self.resume_notes_path = self.progress_root / "resume_notes.md"
        self.run_catalog_path = self.result_root / "run_catalog.csv"
        self.run_failures_path = self.result_root / "run_failures.csv"
        self.optionb_validation_path = self.progress_root / "optionb_environment_validation.json"
        self.validate_log_path = self.progress_root / "validate_runtime.json"
        self.pytest_log_path = self.progress_root / "pytest_targeted.json"
        self.paper_compile_log_path = self.progress_root / "paper_compile.json"
        self.repo_note_path = self.result_root / "seeded_backbone_note.md"
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
        self.paper_export_root.mkdir(parents=True, exist_ok=True)
        for family in FAMILY_SPECS:
            family_root = self.family_root(family.family_name)
            (family_root / "_progress").mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                seed_root = self.seed_root(family.family_name, seed)
                (seed_root / "_progress").mkdir(parents=True, exist_ok=True)

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
        if len(enabled) != 18:
            raise RuntimeError(
                f"expected exactly 18 enabled water-treatment scenarios, found {len(enabled)}"
            )
        return rows

    def split_rows(self) -> list[dict[str, Any]]:
        rows = build_split_rows(self.manifest_rows())
        if len(rows) != 18:
            raise RuntimeError(f"expected 18 split rows, found {len(rows)}")
        return rows

    def _base_config(self, family_spec: FamilySpec) -> dict[str, Any]:
        config = load_config(str(family_spec.config_path))
        require_fields(config, ["experiment_name", "scenario_selectors"], ctx="experiment config")
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
            raise RuntimeError("unable to resolve model metadata from agentic profiles")
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
        model_metadata = self._model_metadata()
        return {
            "experiment_version": EXPERIMENT_VERSION,
            "models": [
                {
                    **model_metadata,
                    "source_profiles": [
                        "black_box_attacker",
                        "gray_box_attacker",
                        "white_box_attacker",
                        "agentic_defender",
                        "agentic_observer",
                    ],
                }
            ],
        }

    def _prompt_hash_payload(self) -> dict[str, Any]:
        file_hashes = {
            portable_path(path): _sha256_path(path)
            for path in [
                *FROZEN_PROMPT_PATHS,
                *PROFILE_REGISTRY_PATHS,
                *FROZEN_CONFIG_PATHS,
            ]
            if path.exists()
        }
        family_hashes: dict[str, dict[str, Any]] = {}
        for family in FAMILY_SPECS:
            prompt_path = {
                "black_box_attacker": FROZEN_PROMPT_PATHS[0],
                "gray_box_attacker": FROZEN_PROMPT_PATHS[1],
                "white_box_attacker": FROZEN_PROMPT_PATHS[2],
            }[family.attacker_profile_id]
            bundle = {
                "attacker_prompt": file_hashes[portable_path(prompt_path)],
                "defender_prompt": file_hashes[portable_path(FROZEN_PROMPT_PATHS[3])],
                "observer_prompt": file_hashes[portable_path(FROZEN_PROMPT_PATHS[4])],
                "profile_registries": {
                    portable_path(path): file_hashes[portable_path(path)]
                    for path in PROFILE_REGISTRY_PATHS
                },
                "experiment_config": file_hashes[portable_path(family.config_path)],
            }
            family_hashes[family.family_name] = {
                "bundle_hash": _canonical_json_hash(bundle),
                "bundle": bundle,
            }
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
                profile = dict(effective_profiles.get(role) or orchestrator.agent_profiles[profile_id])
                role_payload[role] = {
                    "profile_id": profile_id,
                    "allowed_action_types": list(profile.get("allowed_action_types", [])),
                    "behavior_model": profile.get("behavior_model"),
                    "knowledge_mode": profile.get("knowledge_mode"),
                    "knowledge_files": profile.get("knowledge_files", []),
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
                "Concrete runtime containment semantics frozen for this pass:",
                "- runtime-enforced surface isolation remains enabled",
                "- defender containment actions: block_sensor_write, quarantine_signal_source, disable_hmi_session, block_process2_manual_path, isolate_engineering_host",
                "- symbolic containment remains disabled",
                "- water-distribution semantics were not touched in this pass",
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
        if len(rows) != 18:
            raise RuntimeError(
                f"expected 18 rows in {path}, found {len(rows)}"
            )
        return rows

    def _save_scenario_rows(
        self,
        family_name: str,
        seed: int,
        rows: list[dict[str, Any]],
    ) -> None:
        write_csv(self._scenario_status_path(family_name, seed), rows)

    def _expected_environment(self, config: dict[str, Any]) -> dict[str, Any]:
        return dict(config.get("environment") or {})

    def _run_dir(self, family_name: str, seed: int, scenario: Any) -> Path:
        return self.seed_root(family_name, seed) / scenario.suite / scenario.scenario_id / "trial-001"

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

    def _verify_run_completion(self, run_dir: Path, *, expected_environment: dict[str, Any]) -> bool:
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
            "robustness_variant": "seeded_backbone",
            "containment_semantics_version": CONTAINMENT_SEMANTICS_VERSION,
            "status": summary.get("status"),
            "error_message": "",
            "family_name": family_spec.family_name,
            "run_dir": portable_path(run_dir),
            "catalog_id": scenario.catalog_id,
            "detected": int(
                metrics.get("defender_detection", {}).get("time_to_detection_sec") is not None
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
            "status": "completed" if len(completed_run_rows) == 18 else "partial",
            "experiment_version": EXPERIMENT_VERSION,
            "testbed": TESTBED_ID,
            "runs": run_summaries,
            "completed_run_count": len(completed_run_rows),
            "expected_run_count": 18,
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
            "expected_scenarios": 18,
            "detected_scenarios": attack_metrics.get("defender_detected_attack_run_count"),
            "defender_action_scenarios": attack_metrics.get("defender_acted_attack_run_count"),
            "safety_violation_scenarios": attack_metrics.get(
                "attacker_caused_safety_violation_run_count"
            ),
            "avg_wall_clock_time_to_detection_from_attack_start_sec": attack_metrics.get(
                "wall_clock_time_to_detection_from_attack_start_sec_avg"
            ),
            "avg_attacker_action_count": attack_metrics.get("attacker_action_count_avg"),
            "avg_downtime_timestep_count": attack_metrics.get("downtime_timestep_count_avg"),
            "primary_timing_origin": attack_metrics.get("primary_timing_origin"),
        }
        seed_summary = {
            "generated_at_utc": utc_now(),
            "experiment_version": EXPERIMENT_VERSION,
            "seed": seed,
            "family_name": family_spec.family_name,
            "completed_scenarios": len(completed_run_rows),
            "expected_scenarios": 18,
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
                    row["last_error"] = "completed marker existed but required artifacts were incomplete"
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
            "status": "completed" if len(completed_rows) == 18 else "in_progress",
            "completed_scenarios": len(completed_rows),
            "failed_scenarios": len(failed_rows),
            "pending_scenarios": 18 - len(completed_rows) - len(failed_rows),
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
        split_map = _scenario_split_map(self.split_rows())
        all_completed: dict[str, Any] = {}
        all_failed: dict[str, Any] = {}
        queue_rows: list[dict[str, Any]] = []
        family_completed: dict[str, dict[str, Any]] = {}
        family_failed: dict[str, dict[str, Any]] = {}

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
            family_completed[family.family_name] = family_completed_rows
            family_failed[family.family_name] = family_failed_rows
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
                "pending_runs": 90 - completed_count - failed_count,
                "status": "completed" if completed_count == 90 else "in_progress",
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
        failure_rows = [
            all_failed[key]
            for key in sorted(all_failed)
        ]
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
            "all_complete": len(catalog_rows) == 270,
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
            f"- Canonical result root: `{portable_path(self.result_root)}`",
            f"- Completed runs: `{completed_count}`",
            f"- Failed runs: `{failed_count}`",
            f"- Pending runs: `{pending_count}`",
            f"- Legacy one-off roots preserved: `true`",
            f"- Water-distribution touched: `false`",
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
                        "state_name": reset_snapshot.get("STATE_NAME") or reset_snapshot.get("state_name"),
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
        payload = {
            "cwd": portable_path(cwd),
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
            [
                python_exec,
                "-m",
                "pytest",
                "tests/test_agentic.py",
                "tests/test_orchestrator.py",
                "tests/test_metrics.py",
                "tests/test_seeded_backbone.py",
            ],
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
            [
                row
                for row in csv.DictReader(self.run_queue_path.open(newline=""))
            ]
            if self.run_queue_path.exists()
            else []
        )
        self._validate_optionb_environment()
        self._validate_runtime_and_tests()
        self._set_phase("running_grid", "in_progress")
        _append_log(
            self.experiment_log_path,
            f"controller starting seeded backbone under {portable_path(self.result_root)}",
        )

        python_exec = _python_executable()
        for family in FAMILY_SPECS:
            for seed in SEEDS:
                seed_status = _read_json(self._seed_status_path(family.family_name, seed), {})
                if seed_status.get("completed_scenarios") == 18:
                    continue
                worker_log_path = self.seed_progress_root(family.family_name, seed) / "worker_run.json"
                command = [
                    python_exec,
                    "-m",
                    "ics_agent.experiments.water_treatment.seeded_backbone",
                    "worker",
                    "--family",
                    family.family_name,
                    "--seed",
                    str(seed),
                    "--result-root",
                    str(self.result_root),
                ]
                env = os.environ.copy()
                env["PYTHONHASHSEED"] = str(seed)
                env.setdefault("CUDA_VISIBLE_DEVICES", "0")
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    cwd=REPO_ROOT,
                    env=env,
                )
                write_json(
                    worker_log_path,
                    {
                        "command": command,
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                        "updated_utc": utc_now(),
                    },
                )
                self._sync_all_ledgers()
                if completed.returncode != 0:
                    self._set_phase(
                        "running_grid",
                        "in_progress",
                        last_failure_family=family.family_name,
                        last_failure_seed=seed,
                    )
                    _append_log(
                        self.experiment_log_path,
                        f"worker failed for {family.family_name} seed {seed}: {completed.stderr.strip()}",
                    )
                    return completed.returncode
                _append_log(
                    self.experiment_log_path,
                    f"worker completed for {family.family_name} seed {seed}",
                )

        self._sync_all_ledgers()
        resume_state = _read_json(self.resume_state_path, {})
        if resume_state.get("completed_runs") != 270:
            raise RuntimeError(
                f"expected 270 completed runs, found {resume_state.get('completed_runs')}"
            )
        self._set_phase("running_grid", "completed", completed_runs=270)
        self.write_family_aggregates()
        self.write_paper_export()
        try:
            self.update_paper_repo()
        finally:
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
        if len(completed_rows) == 18:
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
            if len(completed_rows) == 18 and not failed_rows:
                break
        return 0

    def write_family_aggregates(self) -> None:
        self._set_phase("aggregating", "in_progress")
        with self.run_catalog_path.open(newline="") as handle:
            catalog_rows = list(csv.DictReader(handle))
        for family in FAMILY_SPECS:
            family_root = self.family_root(family.family_name)
            seed_rows = []
            per_seed_summary_rows = []
            for seed in SEEDS:
                seed_summary = _read_json(self.seed_root(family.family_name, seed) / "seed_summary.json", {})
                summary_row = dict(seed_summary.get("summary_row") or {})
                if not summary_row:
                    continue
                summary_row["seed"] = seed
                per_seed_summary_rows.append(summary_row)
                seed_rows.extend(
                    row
                    for row in catalog_rows
                    if row["family_name"] == family.family_name and int(row["seed"]) == seed
                )
            if len(seed_rows) != 90:
                continue
            completed_runs = len(seed_rows)
            detection_values = [
                float(row["avg_wall_clock_time_to_detection_from_attack_start_sec"])
                for row in per_seed_summary_rows
                if row.get("avg_wall_clock_time_to_detection_from_attack_start_sec") not in {"", "None", None}
            ]
            action_values = [
                float(row["avg_attacker_action_count"])
                for row in per_seed_summary_rows
                if row.get("avg_attacker_action_count") not in {"", "None", None}
            ]
            downtime_values = [
                float(row["avg_downtime_timestep_count"])
                for row in per_seed_summary_rows
                if row.get("avg_downtime_timestep_count") not in {"", "None", None}
            ]
            aggregate_row = {
                "family_name": family.family_name,
                "total_runs": 90,
                "total_completed_runs": completed_runs,
                "safety_violation_count": sum(int(row["safety_violation"]) for row in seed_rows),
                "detected_count": sum(int(row["detected"]) for row in seed_rows),
                "defender_action_count": sum(int(row["defender_acted"]) for row in seed_rows),
                "mean_wall_clock_detection_from_attack_start_sec": _mean(detection_values),
                "std_wall_clock_detection_from_attack_start_sec": _stdev(detection_values),
                "mean_attacker_action_count": _mean(action_values),
                "std_attacker_action_count": _stdev(action_values),
                "mean_downtime_timestep_count": _mean(downtime_values),
                "std_downtime_timestep_count": _stdev(downtime_values),
            }
            write_json(family_root / "aggregate_over_seeds.json", aggregate_row)
            write_csv(family_root / "aggregate_over_seeds.csv", [aggregate_row])

            stability_rows = []
            for scenario in self.scenarios():
                matching = [
                    row
                    for row in seed_rows
                    if row["scenario_id"] == scenario.scenario_id
                ]
                stability_rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "safety_violation_count_over_5": sum(
                            int(row["safety_violation"]) for row in matching
                        ),
                        "detected_count_over_5": sum(int(row["detected"]) for row in matching),
                        "defender_action_count_over_5": sum(
                            int(row["defender_acted"]) for row in matching
                        ),
                        "mean_downtime_over_5": _mean(
                            float(row["downtime_timestep_count"])
                            for row in matching
                            if row.get("downtime_timestep_count") not in {"", "None", None}
                        ),
                        "std_downtime_over_5": _stdev(
                            float(row["downtime_timestep_count"])
                            for row in matching
                            if row.get("downtime_timestep_count") not in {"", "None", None}
                        ),
                        "mean_attacker_action_count_over_5": _mean(
                            float(row["attacker_action_count"])
                            for row in matching
                            if row.get("attacker_action_count") not in {"", "None", None}
                        ),
                        "std_attacker_action_count_over_5": _stdev(
                            float(row["attacker_action_count"])
                            for row in matching
                            if row.get("attacker_action_count") not in {"", "None", None}
                        ),
                    }
                )
            write_csv(family_root / "scenario_stability_over_seeds.csv", stability_rows)

            seed_variance_rows = []
            for row in per_seed_summary_rows:
                seed_variance_rows.append(
                    {
                        "seed": row["seed"],
                        "safety_violation_rate": round(
                            float(row["safety_violation_scenarios"]) / 18.0,
                            4,
                        ),
                        "detection_rate": round(
                            float(row["detected_scenarios"]) / 18.0,
                            4,
                        ),
                        "defender_action_rate": round(
                            float(row["defender_action_scenarios"]) / 18.0,
                            4,
                        ),
                        "mean_downtime": row["avg_downtime_timestep_count"],
                        "mean_attacker_action_count": row["avg_attacker_action_count"],
                    }
                )
            write_csv(family_root / "seed_variance.csv", seed_variance_rows)

            report_lines = [
                f"# {family.family_name}",
                "",
                f"- Total runs: `{aggregate_row['total_runs']}`",
                f"- Completed runs: `{aggregate_row['total_completed_runs']}`",
                f"- Safety-violation count: `{aggregate_row['safety_violation_count']}`",
                f"- Detected count: `{aggregate_row['detected_count']}`",
                f"- Defender-action count: `{aggregate_row['defender_action_count']}`",
                f"- Wall-clock detection from attack start: `{_format_mean_std(aggregate_row['mean_wall_clock_detection_from_attack_start_sec'], aggregate_row['std_wall_clock_detection_from_attack_start_sec'])}`",
                f"- Attacker action count: `{_format_mean_std(aggregate_row['mean_attacker_action_count'], aggregate_row['std_attacker_action_count'])}`",
                f"- Downtime burden: `{_format_mean_std(aggregate_row['mean_downtime_timestep_count'], aggregate_row['std_downtime_timestep_count'])}`",
            ]
            _write_text(family_root / "family_report.md", "\n".join(report_lines) + "\n")
        self._set_phase("aggregating", "completed")

    def write_paper_export(self) -> None:
        self._set_phase("paper_export", "in_progress")
        family_rows = {
            family.family_name: _read_json(
                self.family_root(family.family_name) / "aggregate_over_seeds.json",
                {},
            )
            for family in FAMILY_SPECS
        }
        export_rows = [
            {
                "metric": "Attack scenarios",
                "unit": "scenarios",
                "black_box": "90",
                "gray_box": "90",
                "white_box": "90",
            },
            {
                "metric": "Detected scenarios",
                "unit": "scenarios",
                "black_box": str(
                    family_rows["black_box_attacker__agentic_defender"].get("detected_count")
                ),
                "gray_box": str(
                    family_rows["gray_box_attacker__agentic_defender"].get("detected_count")
                ),
                "white_box": str(
                    family_rows["white_box_attacker__agentic_defender"].get("detected_count")
                ),
            },
            {
                "metric": "Defender-action scenarios",
                "unit": "scenarios",
                "black_box": str(
                    family_rows["black_box_attacker__agentic_defender"].get("defender_action_count")
                ),
                "gray_box": str(
                    family_rows["gray_box_attacker__agentic_defender"].get("defender_action_count")
                ),
                "white_box": str(
                    family_rows["white_box_attacker__agentic_defender"].get("defender_action_count")
                ),
            },
            {
                "metric": "Safety-violation scenarios",
                "unit": "scenarios",
                "black_box": f"{family_rows['black_box_attacker__agentic_defender'].get('safety_violation_count')}/90",
                "gray_box": f"{family_rows['gray_box_attacker__agentic_defender'].get('safety_violation_count')}/90",
                "white_box": f"{family_rows['white_box_attacker__agentic_defender'].get('safety_violation_count')}/90",
            },
            {
                "metric": "Average wall-clock time to detection from attack start",
                "unit": "seconds",
                "black_box": _format_mean_std(
                    family_rows["black_box_attacker__agentic_defender"].get(
                        "mean_wall_clock_detection_from_attack_start_sec"
                    ),
                    family_rows["black_box_attacker__agentic_defender"].get(
                        "std_wall_clock_detection_from_attack_start_sec"
                    ),
                ),
                "gray_box": _format_mean_std(
                    family_rows["gray_box_attacker__agentic_defender"].get(
                        "mean_wall_clock_detection_from_attack_start_sec"
                    ),
                    family_rows["gray_box_attacker__agentic_defender"].get(
                        "std_wall_clock_detection_from_attack_start_sec"
                    ),
                ),
                "white_box": _format_mean_std(
                    family_rows["white_box_attacker__agentic_defender"].get(
                        "mean_wall_clock_detection_from_attack_start_sec"
                    ),
                    family_rows["white_box_attacker__agentic_defender"].get(
                        "std_wall_clock_detection_from_attack_start_sec"
                    ),
                ),
            },
            {
                "metric": "Average attacker action count",
                "unit": "actions",
                "black_box": _format_mean_std(
                    family_rows["black_box_attacker__agentic_defender"].get(
                        "mean_attacker_action_count"
                    ),
                    family_rows["black_box_attacker__agentic_defender"].get(
                        "std_attacker_action_count"
                    ),
                ),
                "gray_box": _format_mean_std(
                    family_rows["gray_box_attacker__agentic_defender"].get(
                        "mean_attacker_action_count"
                    ),
                    family_rows["gray_box_attacker__agentic_defender"].get(
                        "std_attacker_action_count"
                    ),
                ),
                "white_box": _format_mean_std(
                    family_rows["white_box_attacker__agentic_defender"].get(
                        "mean_attacker_action_count"
                    ),
                    family_rows["white_box_attacker__agentic_defender"].get(
                        "std_attacker_action_count"
                    ),
                ),
            },
            {
                "metric": "Average downtime burden",
                "unit": "timesteps",
                "black_box": _format_mean_std(
                    family_rows["black_box_attacker__agentic_defender"].get(
                        "mean_downtime_timestep_count"
                    ),
                    family_rows["black_box_attacker__agentic_defender"].get(
                        "std_downtime_timestep_count"
                    ),
                ),
                "gray_box": _format_mean_std(
                    family_rows["gray_box_attacker__agentic_defender"].get(
                        "mean_downtime_timestep_count"
                    ),
                    family_rows["gray_box_attacker__agentic_defender"].get(
                        "std_downtime_timestep_count"
                    ),
                ),
                "white_box": _format_mean_std(
                    family_rows["white_box_attacker__agentic_defender"].get(
                        "mean_downtime_timestep_count"
                    ),
                    family_rows["white_box_attacker__agentic_defender"].get(
                        "std_downtime_timestep_count"
                    ),
                ),
            },
        ]
        write_csv(
            self.paper_export_root / "water_treatment_seeded_attacker_table.csv",
            export_rows,
        )
        write_json(
            self.paper_export_root / "water_treatment_seeded_attacker_table.json",
            {
                "generated_at_utc": utc_now(),
                "experiment_version": EXPERIMENT_VERSION,
                "rows": export_rows,
            },
        )
        self._set_phase("paper_export", "completed")

    def _paper_export_by_metric(self) -> dict[str, dict[str, str]]:
        export_path = self.paper_export_root / "water_treatment_seeded_attacker_table.csv"
        with export_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {row["metric"]: row for row in rows}

    def _replace_required(self, text: str, pattern: str, replacement: str) -> str:
        updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError(f"unable to apply required paper update pattern: {pattern}")
        return updated

    def _update_paper_table(self, export_rows: dict[str, dict[str, str]]) -> None:
        table_text = PAPER_TABLE_PATH.read_text()
        treatment_block = "\n".join(
            [
                r"\multicolumn{5}{@{}l}{\textbf{Water Treatment}} \\",
                (
                    "Attack scenarios & scenarios & "
                    f"{export_rows['Attack scenarios']['black_box']} & "
                    f"{export_rows['Attack scenarios']['gray_box']} & "
                    f"{export_rows['Attack scenarios']['white_box']} \\\\"
                ),
                (
                    "Detected scenarios & scenarios & "
                    f"{export_rows['Detected scenarios']['black_box']} & "
                    f"{export_rows['Detected scenarios']['gray_box']} & "
                    f"{export_rows['Detected scenarios']['white_box']} \\\\"
                ),
                (
                    "Defender-action scenarios & scenarios & "
                    f"{export_rows['Defender-action scenarios']['black_box']} & "
                    f"{export_rows['Defender-action scenarios']['gray_box']} & "
                    f"{export_rows['Defender-action scenarios']['white_box']} \\\\"
                ),
                (
                    "Safety-violation scenarios & scenarios & "
                    f"{export_rows['Safety-violation scenarios']['black_box']} & "
                    f"{export_rows['Safety-violation scenarios']['gray_box']} & "
                    f"{export_rows['Safety-violation scenarios']['white_box']} \\\\"
                ),
                (
                    "Average wall-clock time to detection from attack start & seconds & "
                    f"{export_rows['Average wall-clock time to detection from attack start']['black_box']} & "
                    f"{export_rows['Average wall-clock time to detection from attack start']['gray_box']} & "
                    f"{export_rows['Average wall-clock time to detection from attack start']['white_box']} \\\\"
                ),
                (
                    "Average attacker action count & actions & "
                    f"{export_rows['Average attacker action count']['black_box']} & "
                    f"{export_rows['Average attacker action count']['gray_box']} & "
                    f"{export_rows['Average attacker action count']['white_box']} \\\\"
                ),
                (
                    "Average downtime burden & timesteps & "
                    f"{export_rows['Average downtime burden']['black_box']} & "
                    f"{export_rows['Average downtime burden']['gray_box']} & "
                    f"{export_rows['Average downtime burden']['white_box']} \\\\"
                ),
            ]
        )
        table_text = self._replace_required(
            table_text,
            r"\\multicolumn\{5\}\{@\{\}l\}\{\\textbf\{Water Treatment\}\} \\\\\n.*?\n\\addlinespace\[2pt\]",
            treatment_block + "\n\\addlinespace[2pt]",
        )
        table_notes = dedent(
            r"""
            \paragraph{Table notes.}
            Both blocks are staged Option~B reruns under the same defender family
            and current concrete containment execution semantics. The
            water-treatment block reports the seeded five-seed aggregate over the
            18-scenario suite, and the water-distribution block reports the
            retained 11-scenario suite. The detection row in this table is
            reported as a wall-clock latency derived from recorded UTC timestamps
            between the first attacker action and the first defender detection
            event, rather than as a simulated process-time delta.
            """
        ).strip()
        table_text = self._replace_required(
            table_text,
            r"\\paragraph\{Table notes\.\}\n.*?$",
            table_notes,
        )
        _write_text(PAPER_TABLE_PATH, table_text + "\n")

    def _update_paper_results_section(
        self,
        export_rows: dict[str, dict[str, str]],
    ) -> None:
        results_text = PAPER_RESULTS_PATH.read_text()
        results_text = results_text.replace(
            "The water-treatment block reports the 18-scenario suite. The\n"
            "water-distribution block reports the retained 11-scenario suite.",
            "The water-treatment block reports the 18-scenario suite aggregated\n"
            "over the five final seeds. The water-distribution block reports the\n"
            "retained 11-scenario suite.",
        )
        treatment_paragraph = dedent(
            f"""
            The treatment block still shows the clearest attacker-knowledge split. Over 90 seeded runs per attacker regime (18 scenarios across 5 seeds), the black-box attacker is detected in {export_rows['Detected scenarios']['black_box']}/90 scenarios, triggers defender action in {export_rows['Defender-action scenarios']['black_box']}/90, and records safety violation in {export_rows['Safety-violation scenarios']['black_box']}. The gray-box attacker remains similarly stable at {export_rows['Detected scenarios']['gray_box']}/90 detected scenarios, {export_rows['Defender-action scenarios']['gray_box']}/90 defender-action scenarios, and {export_rows['Safety-violation scenarios']['gray_box']} safety-violation scenarios. The white-box attacker changes that outcome qualitatively: it is also detected in {export_rows['Detected scenarios']['white_box']}/90 runs and triggers defender action in {export_rows['Defender-action scenarios']['white_box']}/90, but it reaches safety violation in {export_rows['Safety-violation scenarios']['white_box']} scenarios and imposes the heaviest defender burden at {export_rows['Average downtime burden']['white_box']} downtime timesteps on average across seeds. The wall-clock detection latencies remain tightly clustered in treatment at {export_rows['Average wall-clock time to detection from attack start']['black_box']}~s for the black-box attacker, {export_rows['Average wall-clock time to detection from attack start']['gray_box']}~s for the gray-box attacker, and {export_rows['Average wall-clock time to detection from attack start']['white_box']}~s for the white-box attacker, which indicates that richer attacker knowledge changes physical impact much more than it changes the defender's elapsed time to notice the attack. In other words, richer attacker knowledge still matters sharply in treatment, even under the same concrete containment semantics.
            """
        ).strip()
        results_text = self._replace_required(
            results_text,
            r"The treatment block still shows the clearest attacker-knowledge split\..*?same concrete containment semantics\.\n",
            treatment_paragraph + "\n\n",
        )
        refreshed_takeaway = dedent(
            f"""
            Under the current concrete containment semantics used in the refreshed same-defender treatment ablation, the agentic defender records safety violation in {export_rows['Safety-violation scenarios']['black_box']} black-box runs and {export_rows['Safety-violation scenarios']['gray_box']} gray-box runs, while the retained black-box-versus-rule-based reference still suffers safety violation in all 18.
            """
        ).strip()
        results_text = results_text.replace(
            "Under the current concrete containment semantics used in the refreshed black-box-versus-agentic rerun, the agentic\n"
            "defender holds safety violation at 0/18 scenarios against both the black-box\n"
            "and gray-box attackers, while the retained black-box-versus-rule-based\n"
            "reference still suffers safety violation in all 18.",
            refreshed_takeaway,
        )
        white_box_takeaway = dedent(
            f"""
            That conclusion, however, does not extend to the white-box attacker in either testbed. In the same-defender ablation, the white-box attacker is detected in {export_rows['Detected scenarios']['white_box']}/90 treatment runs and all 11 distribution runs, and triggers defender action in {export_rows['Defender-action scenarios']['white_box']}/90 treatment runs and all 11 distribution runs, yet it still achieves safety violation in {export_rows['Safety-violation scenarios']['white_box']} and 11/11 scenarios respectively.
            """
        ).strip()
        results_text = results_text.replace(
            "That conclusion, however, does not extend to the white-box attacker in either\n"
            "testbed. In the same-defender ablation, the white-box attacker is detected in\n"
            "all 18 treatment runs and all 11 distribution runs, and triggers defender\n"
            "action in every case, yet it still achieves safety violation in 18/18 and\n"
            "11/11 scenarios respectively.",
            white_box_takeaway,
        )
        _write_text(PAPER_RESULTS_PATH, results_text)

    def update_paper_repo(self) -> None:
        self._set_phase("paper_update", "in_progress")
        try:
            export_rows = self._paper_export_by_metric()
            self._update_paper_table(export_rows)
            self._update_paper_results_section(export_rows)
            self._run_logged(
                [
                    "latexmk",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "main.tex",
                ],
                self.paper_compile_log_path,
                cwd=PAPER_REPO_ROOT,
            )
        except Exception as exc:
            self._set_phase("paper_update", "failed", error_message=str(exc))
            raise
        self._set_phase(
            "paper_update",
            "completed",
            table_path=portable_path(PAPER_TABLE_PATH),
            results_path=portable_path(PAPER_RESULTS_PATH),
            compile_log=portable_path(self.paper_compile_log_path),
        )

    def write_repo_note(self) -> None:
        resume_state = _read_json(self.resume_state_path, {})
        lines = [
            "# Seeded Backbone Note",
            "",
            f"- Experiment version: `{EXPERIMENT_VERSION}`",
            f"- Canonical result root: `{portable_path(self.result_root)}`",
            f"- Frozen configs: `{', '.join(portable_path(path) for path in FROZEN_CONFIG_PATHS)}`",
            f"- Frozen prompts: `{', '.join(portable_path(path) for path in FROZEN_PROMPT_PATHS)}`",
            f"- Final seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
            f"- Resume event occurred: `{'true' if resume_state.get('resume_event_detected') else 'false'}`",
            f"- Scenarios rerun after interruption: `{'true' if resume_state.get('scenarios_rerun_after_interruption') else 'false'}`",
            f"- All 270 runs completed: `{'true' if resume_state.get('completed_runs') == 270 else 'false'}`",
            f"- Paper export written to: `{portable_path(self.paper_export_root)}`",
            "- Legacy one-off water-treatment roots were preserved.",
            "- No water-distribution runs were touched.",
            "- No secrets or credentials were committed.",
        ]
        _write_text(self.repo_note_path, "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runner = SeededBackboneRunner(Path(args.result_root))
    if args.mode == "worker":
        return runner.run_worker(
            args.family,
            int(args.seed),
            max_attempts=int(args.max_attempts),
        )
    return runner.run_controller()


if __name__ == "__main__":
    raise SystemExit(main())
