from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .action_semantics import RUNTIME_ENFORCED_CONTAINMENT_ACTION_TYPES
from .config import resolve_repo_path
from .orchestrator import ExperimentOrchestrator
from .paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from .reporting import write_json
from .utils import utc_now
from . import water_treatment_seeded_backbone as seeded


AUTHORITY_EXPERIMENT_VERSION = "water_treatment_defender_authority_ablation_seed101_v1"
AUTHORITY_SEED = 101
AUTHORITY_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "Water-treatment"
    / "defender-authority-ablation"
)
AUTHORITY_CONFIG_PATHS = [
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_black_box_attacker_vs_agentic_observe_only_and_delayed_containment_defenders.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_gray_box_attacker_vs_agentic_observe_only_and_delayed_containment_defenders.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_white_box_attacker_vs_agentic_observe_only_and_delayed_containment_defenders.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_black_box_high_cyber_high_domain_attacker_vs_agentic_observe_only_and_delayed_containment_defenders.yaml",
]
AUTHORITY_FAMILY_SPECS = [
    seeded.FamilySpec(
        family_name="black_box_attacker__observe_alert_only_defender",
        attacker_profile_id="black_box_attacker",
        knowledge_variant="black_box",
        config_path=AUTHORITY_CONFIG_PATHS[0],
    ),
    seeded.FamilySpec(
        family_name="black_box_attacker__delayed_containment_defender",
        attacker_profile_id="black_box_attacker",
        knowledge_variant="black_box",
        config_path=AUTHORITY_CONFIG_PATHS[0],
    ),
    seeded.FamilySpec(
        family_name="gray_box_attacker__observe_alert_only_defender",
        attacker_profile_id="gray_box_attacker",
        knowledge_variant="gray_box",
        config_path=AUTHORITY_CONFIG_PATHS[1],
    ),
    seeded.FamilySpec(
        family_name="gray_box_attacker__delayed_containment_defender",
        attacker_profile_id="gray_box_attacker",
        knowledge_variant="gray_box",
        config_path=AUTHORITY_CONFIG_PATHS[1],
    ),
    seeded.FamilySpec(
        family_name="white_box_attacker__observe_alert_only_defender",
        attacker_profile_id="white_box_attacker",
        knowledge_variant="white_box",
        config_path=AUTHORITY_CONFIG_PATHS[2],
    ),
    seeded.FamilySpec(
        family_name="white_box_attacker__delayed_containment_defender",
        attacker_profile_id="white_box_attacker",
        knowledge_variant="white_box",
        config_path=AUTHORITY_CONFIG_PATHS[2],
    ),
    seeded.FamilySpec(
        family_name="black_box_high_cyber_high_domain__observe_alert_only_defender",
        attacker_profile_id="black_box_high_cyber_high_domain_treatment_attacker",
        knowledge_variant="black_box_high_cyber_high_domain",
        config_path=AUTHORITY_CONFIG_PATHS[3],
    ),
    seeded.FamilySpec(
        family_name="black_box_high_cyber_high_domain__delayed_containment_defender",
        attacker_profile_id="black_box_high_cyber_high_domain_treatment_attacker",
        knowledge_variant="black_box_high_cyber_high_domain",
        config_path=AUTHORITY_CONFIG_PATHS[3],
    ),
]
AUTHORITY_FAMILY_BY_NAME = {
    item.family_name: item for item in AUTHORITY_FAMILY_SPECS
}
AUTHORITY_FAMILY_ORDER = {
    item.family_name: index for index, item in enumerate(AUTHORITY_FAMILY_SPECS)
}
AUTHORITY_VALIDATION_TESTS = [
    "tests/test_agentic.py",
    "tests/test_environment.py",
    "tests/test_orchestrator.py",
    "tests/test_metrics.py",
    "tests/test_seeded_backbone.py",
    "tests/test_phase35_profiles.py",
    "tests/test_defender_authority_ablation.py",
]
BASELINE_TREATMENT_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "Water-treatment"
    / "seeded_backbone_qwen35_9b"
)
BASELINE_DOMAIN_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "Water-treatment"
    / "domain-ablation"
    / "black_box_high_cyber_high_domain__agentic_defender"
)


def _read_json(path: Path, default: Any) -> Any:
    return seeded._read_json(path, default)


def _family_defender_profile_id(family_name: str) -> str:
    if family_name.endswith("__observe_alert_only_defender"):
        return "observe_alert_only_defender"
    if family_name.endswith("__delayed_containment_defender"):
        return "delayed_containment_defender"
    return "agentic_defender"


def _defender_variant_label(profile_id: str) -> str:
    return {
        "agentic_defender": "full_containment",
        "observe_alert_only_defender": "observe_alert_only",
        "delayed_containment_defender": "delayed_containment",
    }.get(profile_id, profile_id)


def _family_attacker_label(family_name: str) -> str:
    return family_name.split("__", 1)[0]


def _baseline_family_name(family_spec: seeded.FamilySpec) -> str:
    return {
        "black_box_attacker": "black_box_attacker__agentic_defender",
        "gray_box_attacker": "gray_box_attacker__agentic_defender",
        "white_box_attacker": "white_box_attacker__agentic_defender",
        "black_box_high_cyber_high_domain_treatment_attacker": (
            "black_box_high_cyber_high_domain__agentic_defender"
        ),
    }[family_spec.attacker_profile_id]


def _baseline_condition_summary_path(family_spec: seeded.FamilySpec) -> Path:
    if family_spec.attacker_profile_id == "black_box_high_cyber_high_domain_treatment_attacker":
        return BASELINE_DOMAIN_RESULT_ROOT / "condition_summary.json"
    return (
        BASELINE_TREATMENT_RESULT_ROOT
        / _baseline_family_name(family_spec)
        / f"seed_{AUTHORITY_SEED}"
        / "condition_summary.json"
    )


def _baseline_reference_root(family_spec: seeded.FamilySpec) -> Path:
    if family_spec.attacker_profile_id == "black_box_high_cyber_high_domain_treatment_attacker":
        return BASELINE_DOMAIN_RESULT_ROOT
    return (
        BASELINE_TREATMENT_RESULT_ROOT
        / _baseline_family_name(family_spec)
        / f"seed_{AUTHORITY_SEED}"
    )


def _completed_attack_runs(condition_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        run
        for run in condition_summary.get("runs", [])
        if run.get("suite") == "attack" and run.get("status") == "completed"
    ]


def _mean(values: list[float]) -> float | None:
    return seeded._mean(values)


def _rate(count: int | None, total: int) -> float | None:
    if count is None or total <= 0:
        return None
    return round(float(count) / float(total), 4)


def _cutoff_time_values(condition_summary: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for run in _completed_attack_runs(condition_summary):
        value = (
            run.get("metrics", {})
            .get("runtime_containment", {})
            .get("attacker_execution_cutoff_activation_from_wargame_t0_sec")
        )
        if value is not None:
            values.append(float(value))
    return values


def _resolve_run_dir(run: dict[str, Any]) -> Path | None:
    raw_path = run.get("run_dir")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _first_plant_containment_time_from_wargame_t0(run: dict[str, Any]) -> float | None:
    run_dir = _resolve_run_dir(run)
    if run_dir is None:
        return None
    actions_path = run_dir / "actions.json"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    if not actions_path.exists() or not config_snapshot_path.exists():
        return None

    actions = _read_json(actions_path, [])
    if not isinstance(actions, list):
        return None
    config_snapshot = seeded.load_yaml(config_snapshot_path)
    if not isinstance(config_snapshot, dict):
        return None
    phase_boundary = dict(config_snapshot.get("phase_boundary") or {})
    wargame_origin = float(phase_boundary.get("wargame_t0_elapsed_sec", 0.0) or 0.0)

    plant_action_times: list[float] = []
    for item in actions:
        if not isinstance(item, dict) or item.get("role") != "defender":
            continue
        containment = dict(item.get("containment") or {})
        if not containment.get("is_containment_action"):
            continue
        if str(containment.get("execution_mode", "")) != "plant_action":
            continue
        time_sec = item.get("time_sec")
        if time_sec is None:
            continue
        plant_action_times.append(round(float(time_sec) - wargame_origin, 3))
    if not plant_action_times:
        return None
    return min(plant_action_times)


def _effective_containment_time_from_wargame_t0(run: dict[str, Any]) -> float | None:
    metrics = dict(run.get("metrics") or {})
    runtime_metrics = dict(metrics.get("runtime_containment") or {})
    cutoff_time = runtime_metrics.get(
        "attacker_execution_cutoff_activation_from_wargame_t0_sec"
    )
    plant_action_time = _first_plant_containment_time_from_wargame_t0(run)
    candidates = [
        float(value)
        for value in (plant_action_time, cutoff_time)
        if value is not None
    ]
    if candidates:
        return round(min(candidates), 3)

    first_containment_time = runtime_metrics.get("first_containment_time_from_wargame_t0_sec")
    if first_containment_time is None:
        return None
    first_containment_action_type = str(
        runtime_metrics.get("first_containment_action_type") or ""
    )
    if first_containment_action_type in RUNTIME_ENFORCED_CONTAINMENT_ACTION_TYPES:
        return None
    return float(first_containment_time)


def _effective_contained_before_safety_violation_run_count(
    condition_summary: dict[str, Any],
) -> int:
    count = 0
    for run in _completed_attack_runs(condition_summary):
        effective_containment_time = _effective_containment_time_from_wargame_t0(run)
        if effective_containment_time is None:
            continue
        first_violation_time = (
            run.get("metrics", {})
            .get("phase_window", {})
            .get("time_to_first_safety_violation_sec")
        )
        if first_violation_time is None or effective_containment_time < float(
            first_violation_time
        ):
            count += 1
    return count


def _safety_violation_before_cutoff_run_count(condition_summary: dict[str, Any]) -> int:
    count = 0
    for run in _completed_attack_runs(condition_summary):
        metrics = dict(run.get("metrics") or {})
        safety_total = int(metrics.get("safety_violation_count", 0) or 0)
        if safety_total <= 0:
            continue
        cutoff_time = (
            metrics.get("runtime_containment", {})
            .get("attacker_execution_cutoff_activation_from_wargame_t0_sec")
        )
        first_violation_time = (
            metrics.get("phase_window", {})
            .get("time_to_first_safety_violation_sec")
        )
        if cutoff_time is None:
            count += 1
            continue
        if first_violation_time is None:
            continue
        if float(first_violation_time) < float(cutoff_time):
            count += 1
    return count


def _condition_metric_row(
    family_spec: seeded.FamilySpec,
    condition_summary: dict[str, Any],
) -> dict[str, Any]:
    attack_metrics = dict(condition_summary.get("attack_metrics") or {})
    total_runs = int(condition_summary.get("expected_run_count", 18) or 18)
    completed_runs = int(condition_summary.get("completed_run_count", 0) or 0)
    detected_count = int(attack_metrics.get("defender_detected_attack_run_count", 0) or 0)
    defender_action_count = int(
        attack_metrics.get("defender_acted_attack_run_count", 0) or 0
    )
    cutoff_activation_count = sum(
        1
        for run in _completed_attack_runs(condition_summary)
        if (
            run.get("metrics", {})
            .get("runtime_containment", {})
            .get("attacker_execution_cutoff_activated")
        )
    )
    contained_before_count = _effective_contained_before_safety_violation_run_count(
        condition_summary
    )
    safety_violation_count = int(
        attack_metrics.get("attacker_caused_safety_violation_run_count", 0) or 0
    )
    cutoff_times = _cutoff_time_values(condition_summary)
    defender_profile_id = _family_defender_profile_id(family_spec.family_name)
    return {
        "family_name": family_spec.family_name,
        "attacker_profile_id": family_spec.attacker_profile_id,
        "attacker_family": _family_attacker_label(family_spec.family_name),
        "defender_profile_id": defender_profile_id,
        "defender_variant": _defender_variant_label(defender_profile_id),
        "completed_runs": completed_runs,
        "expected_runs": total_runs,
        "detection_count": detected_count,
        "detection_rate": _rate(detected_count, total_runs),
        "defender_action_count": defender_action_count,
        "defender_action_rate": _rate(defender_action_count, total_runs),
        "cutoff_activation_count": cutoff_activation_count,
        "cutoff_activation_rate": _rate(cutoff_activation_count, total_runs),
        "contained_before_safety_violation_count": contained_before_count,
        "contained_before_safety_violation_rate": _rate(
            contained_before_count,
            total_runs,
        ),
        "safety_violation_count": safety_violation_count,
        "safety_violation_rate": _rate(safety_violation_count, total_runs),
        "time_to_detection_from_wargame_t0_sec_avg": attack_metrics.get(
            "time_to_detection_from_wargame_t0_sec_avg"
        ),
        "time_to_first_defender_action_from_wargame_t0_sec_avg": attack_metrics.get(
            "time_to_first_defender_action_from_wargame_t0_sec_avg"
        ),
        "time_to_cutoff_from_wargame_t0_sec_avg": _mean(cutoff_times),
        "safety_violation_before_cutoff_run_count": (
            _safety_violation_before_cutoff_run_count(condition_summary)
        ),
        "attacker_action_count_avg": attack_metrics.get("attacker_action_count_avg"),
        "defender_action_count_avg": attack_metrics.get("defender_action_count_avg"),
        "downtime_timestep_count_avg": attack_metrics.get("downtime_timestep_count_avg"),
    }


class DefenderAuthorityAblationRunner(seeded.SeededBackboneRunner):
    def __init__(self, result_root: Path = AUTHORITY_RESULT_ROOT) -> None:
        self._apply_authority_globals()
        super().__init__(result_root)
        self.repo_note_path = self.result_root / "defender_authority_ablation_note.md"
        self.phase4_report_path = self.result_root / "phase4_report.md"
        self.results_summary_path = self.result_root / "results_summary.md"
        self.phase4_condition_summary_path = (
            self.result_root / "phase4_condition_summary.json"
        )
        self.phase4_condition_summary_csv_path = (
            self.result_root / "phase4_condition_summary.csv"
        )
        self.baseline_comparison_path = self.result_root / "baseline_comparison.json"
        self.validation_tests = list(AUTHORITY_VALIDATION_TESTS)

    @staticmethod
    def _apply_authority_globals() -> None:
        seeded.EXPERIMENT_VERSION = AUTHORITY_EXPERIMENT_VERSION
        seeded.SEEDS = [AUTHORITY_SEED]
        seeded.FROZEN_CONFIG_PATHS = list(AUTHORITY_CONFIG_PATHS)
        seeded.FAMILY_SPECS = list(AUTHORITY_FAMILY_SPECS)
        seeded.FAMILY_BY_NAME = dict(AUTHORITY_FAMILY_BY_NAME)
        seeded.FAMILY_ORDER = dict(AUTHORITY_FAMILY_ORDER)
        seeded.CONTAINMENT_SEMANTICS_VERSION = "phase4_defender_authority_ablation_v1"
        seeded.TOP_LEVEL_PHASES = [
            "manifest_frozen",
            "validated_optionb_environment",
            "validation_tests",
            "running_grid",
            "aggregating",
            "phase4_report",
        ]

    def _condition_spec(
        self,
        orchestrator: ExperimentOrchestrator,
        family_spec: seeded.FamilySpec,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        condition_specs = orchestrator._enumerate_conditions(config)
        for condition_spec in condition_specs:
            condition_name = orchestrator._condition_name(
                condition_spec["profiles"],
                override=condition_spec.get("name"),
            )
            if condition_name == family_spec.family_name:
                return condition_spec
        raise RuntimeError(
            f"unable to resolve condition `{family_spec.family_name}` from {family_spec.config_path}"
        )

    def _model_registry_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        config = self._runtime_config(AUTHORITY_FAMILY_SPECS[0], AUTHORITY_SEED)
        condition_spec = self._condition_spec(
            orchestrator,
            AUTHORITY_FAMILY_SPECS[0],
            config,
        )
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
            raise RuntimeError("unable to resolve model metadata for authority ablation")
        return {
            "experiment_version": AUTHORITY_EXPERIMENT_VERSION,
            "models": [
                {
                    "model_label": seeded.MODEL_LABEL,
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
                    "source_profiles": sorted(
                        {
                            spec.attacker_profile_id for spec in AUTHORITY_FAMILY_SPECS
                        }
                        | {
                            "observe_alert_only_defender",
                            "delayed_containment_defender",
                            "agentic_observer",
                        }
                    ),
                }
            ],
        }

    def _prompt_hash_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        agent_profiles_path = REPO_ROOT / "configs" / "agent_profiles.yaml"
        profiles_overrides_path = REPO_ROOT / "configs" / "agent_profiles_overrides.yaml"
        hashed_paths = {
            portable_path(path): seeded._sha256_path(path)
            for path in [
                agent_profiles_path,
                profiles_overrides_path,
                *AUTHORITY_CONFIG_PATHS,
            ]
            if path.exists()
        }
        family_hashes: dict[str, dict[str, Any]] = {}
        for family in AUTHORITY_FAMILY_SPECS:
            config = self._runtime_config(family, AUTHORITY_SEED)
            condition_spec = self._condition_spec(orchestrator, family, config)
            effective_profiles = condition_spec.get("effective_profiles") or {}
            attacker = dict(
                effective_profiles.get("attacker")
                or orchestrator.agent_profiles[family.attacker_profile_id]
            )
            defender = dict(
                effective_profiles.get("defender")
                or orchestrator.agent_profiles[_family_defender_profile_id(family.family_name)]
            )
            observer = dict(
                effective_profiles.get("observer")
                or orchestrator.agent_profiles["agentic_observer"]
            )
            prompt_paths = {
                "attacker_prompt": resolve_repo_path(
                    attacker["llm"]["prompt_file"],
                    base_dir=REPO_ROOT,
                ),
                "defender_prompt": resolve_repo_path(
                    defender["llm"]["prompt_file"],
                    base_dir=REPO_ROOT,
                ),
                "observer_prompt": resolve_repo_path(
                    observer["llm"]["prompt_file"],
                    base_dir=REPO_ROOT,
                ),
            }
            knowledge_paths = [
                resolve_repo_path(path, base_dir=REPO_ROOT)
                for path in attacker.get("knowledge_files", [])
            ] + [
                resolve_repo_path(path, base_dir=REPO_ROOT)
                for path in defender.get("knowledge_files", [])
            ]
            for path in [*prompt_paths.values(), *knowledge_paths]:
                key = portable_path(path)
                if key not in hashed_paths and path.exists():
                    hashed_paths[key] = seeded._sha256_path(path)
            bundle = {
                "attacker_prompt": hashed_paths[
                    portable_path(prompt_paths["attacker_prompt"])
                ],
                "defender_prompt": hashed_paths[
                    portable_path(prompt_paths["defender_prompt"])
                ],
                "observer_prompt": hashed_paths[
                    portable_path(prompt_paths["observer_prompt"])
                ],
                "attacker_knowledge_files": {
                    portable_path(path): hashed_paths[portable_path(path)]
                    for path in knowledge_paths
                },
                "agent_profiles": hashed_paths[portable_path(agent_profiles_path)],
                "agent_profiles_overrides": hashed_paths.get(
                    portable_path(profiles_overrides_path)
                ),
                "experiment_config": hashed_paths[portable_path(family.config_path)],
            }
            family_hashes[family.family_name] = {
                "bundle_hash": seeded._canonical_json_hash(bundle),
                "bundle": bundle,
            }
        return {
            "experiment_version": AUTHORITY_EXPERIMENT_VERSION,
            "file_hashes": hashed_paths,
            "family_prompt_hashes": family_hashes,
        }

    def _containment_semantics_text(self) -> str:
        return "\n".join(
            [
                seeded.CONTAINMENT_SEMANTICS_VERSION,
                "",
                "Water-treatment Phase 4 defender-authority ablation semantics:",
                "- observe_alert_only_defender may alert and reason but may not actuate live containment or shutdown actions",
                "- delayed_containment_defender preserves the current full defender action space",
                "- delayed_containment_defender records containment actions immediately and activates runtime cutoff after 2 timesteps",
                "- full-containment baseline was preserved and reused for comparison",
                "- water-distribution semantics were not touched in this pass",
                "",
                "Primary source files:",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'action_semantics.py')}",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'environment.py')}",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'orchestrator.py')}",
                "",
            ]
        )

    def _run_id(
        self,
        family_spec: seeded.FamilySpec,
        seed: int,
        scenario: Any,
        split: str,
    ) -> str:
        defender_variant = _defender_variant_label(
            _family_defender_profile_id(family_spec.family_name)
        )
        return "_".join(
            [
                seeded.TESTBED_ID,
                seeded._sanitize_token(scenario.scenario_id),
                split,
                seeded._sanitize_token(family_spec.attacker_profile_id),
                seeded._sanitize_token(defender_variant),
                "Qwen35_9B",
                f"seed{seed}",
            ]
        )

    def _seed_completed_row(self, **kwargs: Any) -> dict[str, Any]:
        row = super()._seed_completed_row(**kwargs)
        family_spec = kwargs["family_spec"]
        defender_profile_id = _family_defender_profile_id(family_spec.family_name)
        row["defender_variant"] = _defender_variant_label(defender_profile_id)
        row["robustness_variant"] = "defender_authority_ablation"
        return row

    def _validate_runtime_and_tests(self) -> None:
        payload = _read_json(self.validate_log_path, {})
        pytest_payload = _read_json(self.pytest_log_path, {})
        if payload.get("returncode") == 0 and pytest_payload.get("returncode") == 0:
            self._set_phase("validation_tests", "completed")
            return
        self._set_phase("validation_tests", "in_progress")
        python_exec = seeded._python_executable()
        self._run_logged(
            [python_exec, "-m", "ics_agent.cli", "validate"],
            self.validate_log_path,
        )
        self._run_logged(
            [python_exec, "-m", "pytest", *self.validation_tests],
            self.pytest_log_path,
        )
        self._set_phase(
            "validation_tests",
            "completed",
            validate_log=portable_path(self.validate_log_path),
            pytest_log=portable_path(self.pytest_log_path),
        )

    def _sync_all_ledgers(self) -> None:
        manifest_rows = self.manifest_rows()
        all_completed: dict[str, Any] = {}
        all_failed: dict[str, Any] = {}
        queue_rows: list[dict[str, Any]] = []
        for family in AUTHORITY_FAMILY_SPECS:
            family_completed_rows: dict[str, Any] = {}
            family_failed_rows: dict[str, Any] = {}
            rows = self._load_scenario_rows(family.family_name, AUTHORITY_SEED)
            completed_payload = dict(
                _read_json(self._seed_completed_runs_path(family.family_name, AUTHORITY_SEED), {})
            )
            failed_payload = dict(
                _read_json(self._seed_failed_runs_path(family.family_name, AUTHORITY_SEED), {})
            )
            for row in rows:
                scenario_key = (
                    f"{family.family_name}::{AUTHORITY_SEED}::{row['scenario_id']}"
                )
                queue_rows.append(
                    {
                        "attacker_family": family.family_name,
                        "seed": AUTHORITY_SEED,
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
            family_completed_count = len(family_completed_rows)
            family_failed_count = len(family_failed_rows)
            family_status = {
                "updated_utc": utc_now(),
                "experiment_version": AUTHORITY_EXPERIMENT_VERSION,
                "family_name": family.family_name,
                "completed_runs": family_completed_count,
                "failed_runs": family_failed_count,
                "pending_runs": 18 - family_completed_count - family_failed_count,
                "status": "completed" if family_completed_count == 18 else "in_progress",
                "seed_status": {
                    AUTHORITY_SEED: _read_json(
                        self._seed_status_path(family.family_name, AUTHORITY_SEED),
                        {},
                    )
                },
            }
            write_json(self._family_status_path(family.family_name), family_status)

        write_json(self.completed_runs_path, all_completed)
        write_json(self.failed_runs_path, all_failed)
        seeded.write_csv(self.run_queue_path, queue_rows)
        catalog_rows = [
            all_completed[key]
            for key in sorted(
                all_completed,
                key=lambda item: (
                    AUTHORITY_FAMILY_ORDER[item.split("::", 1)[0]],
                    next(
                        index
                        for index, row in enumerate(manifest_rows)
                        if row["scenario_id"] == item.split("::")[2]
                    ),
                ),
            )
        ]
        failure_rows = [all_failed[key] for key in sorted(all_failed)]
        seeded.write_csv(self.run_catalog_path, catalog_rows)
        seeded.write_csv(self.run_failures_path, failure_rows)
        write_json(self.result_root / "run_catalog.json", catalog_rows)
        write_json(self.result_root / "run_failures.json", failure_rows)

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
            "experiment_version": AUTHORITY_EXPERIMENT_VERSION,
            "completed_runs": len(catalog_rows),
            "failed_runs": len(failure_rows),
            "pending_runs": len(pending_rows),
            "all_complete": len(catalog_rows) == 144,
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
                if row["status"] in {"failed", "pending"}
                and row.get("attempt_count") not in {"", "0"}
            ],
            "scenarios_rerun_after_interruption": rerun_scenarios,
        }
        write_json(self.resume_state_path, resume_state)
        self._write_resume_notes(queue_rows, next_run)

    def write_family_aggregates(self) -> None:
        self._set_phase("aggregating", "in_progress")
        summary_rows: list[dict[str, Any]] = []
        for family in AUTHORITY_FAMILY_SPECS:
            seed_root = self.seed_root(family.family_name, AUTHORITY_SEED)
            family_root = self.family_root(family.family_name)
            seed_summary = _read_json(seed_root / "seed_summary.json", {})
            attack_metrics = _read_json(seed_root / "attack_metrics_summary.json", {})
            trajectory_metrics = _read_json(
                seed_root / "trajectory_metrics_summary.json",
                {},
            )
            mitre_phase_metrics = _read_json(seed_root / "mitre_phase_metrics.json", {})
            nominal_metrics = _read_json(seed_root / "nominal_metrics_summary.json", {})
            condition_summary = _read_json(seed_root / "condition_summary.json", {})
            if not seed_summary or not condition_summary:
                continue

            metric_row = _condition_metric_row(family, condition_summary)
            summary_rows.append(metric_row)
            aggregate_row = {
                "family_name": family.family_name,
                "seed": AUTHORITY_SEED,
                "total_runs": 18,
                "total_completed_runs": seed_summary.get("completed_scenarios"),
                "detected_count": metric_row["detection_count"],
                "defender_action_count": metric_row["defender_action_count"],
                "cutoff_activation_count": metric_row["cutoff_activation_count"],
                "contained_before_safety_violation_count": (
                    metric_row["contained_before_safety_violation_count"]
                ),
                "safety_violation_count": metric_row["safety_violation_count"],
                "avg_time_to_detection_from_wargame_t0_sec": (
                    metric_row["time_to_detection_from_wargame_t0_sec_avg"]
                ),
                "avg_time_to_first_defender_action_from_wargame_t0_sec": (
                    metric_row["time_to_first_defender_action_from_wargame_t0_sec_avg"]
                ),
                "avg_time_to_cutoff_from_wargame_t0_sec": (
                    metric_row["time_to_cutoff_from_wargame_t0_sec_avg"]
                ),
                "avg_attacker_action_count": metric_row["attacker_action_count_avg"],
                "avg_defender_action_count": metric_row["defender_action_count_avg"],
                "avg_downtime_timestep_count": metric_row["downtime_timestep_count_avg"],
            }
            write_json(family_root / "aggregate_over_seeds.json", aggregate_row)
            seeded.write_csv(family_root / "aggregate_over_seeds.csv", [aggregate_row])
            write_json(family_root / "family_summary.json", seed_summary)
            write_json(family_root / "family_condition_summary.json", condition_summary)
            seeded._write_text(
                family_root / "family_report.md",
                "\n".join(
                    [
                        f"# {family.family_name}",
                        "",
                        f"- Attacker profile: `{family.attacker_profile_id}`",
                        f"- Defender profile: `{_family_defender_profile_id(family.family_name)}`",
                        f"- Baseline reference root: `{portable_path(_baseline_reference_root(family))}`",
                        f"- Seed: `{AUTHORITY_SEED}`",
                        f"- Completed runs: `{seed_summary.get('completed_scenarios')}/18`",
                        f"- Detection rate: `{metric_row['detection_rate']}`",
                        f"- Defender-action rate: `{metric_row['defender_action_rate']}`",
                        f"- Cutoff activation rate: `{metric_row['cutoff_activation_rate']}`",
                        f"- Contained-before-safety-violation rate: `{metric_row['contained_before_safety_violation_rate']}`",
                        f"- Safety-violation rate: `{metric_row['safety_violation_rate']}`",
                        f"- Average time to detection from wargame_t0: `{metric_row['time_to_detection_from_wargame_t0_sec_avg']}`",
                        f"- Average time to first defender action from wargame_t0: `{metric_row['time_to_first_defender_action_from_wargame_t0_sec_avg']}`",
                        f"- Average time to cutoff from wargame_t0: `{metric_row['time_to_cutoff_from_wargame_t0_sec_avg']}`",
                        f"- Safety violation before cutoff runs: `{metric_row['safety_violation_before_cutoff_run_count']}`",
                        f"- Average attacker action count: `{metric_row['attacker_action_count_avg']}`",
                        f"- Average defender action count: `{metric_row['defender_action_count_avg']}`",
                        f"- Average downtime burden: `{metric_row['downtime_timestep_count_avg']}`",
                    ]
                )
                + "\n",
            )
            write_json(family_root / "attack_metrics_summary.json", attack_metrics)
            write_json(
                family_root / "trajectory_metrics_summary.json",
                trajectory_metrics,
            )
            write_json(family_root / "mitre_phase_metrics.json", mitre_phase_metrics)
            write_json(family_root / "nominal_metrics_summary.json", nominal_metrics)

        write_json(self.phase4_condition_summary_path, summary_rows)
        seeded.write_csv(self.phase4_condition_summary_csv_path, summary_rows)
        self._set_phase("aggregating", "completed")

    def write_phase4_report(self) -> None:
        self._set_phase("phase4_report", "in_progress")
        condition_rows = list(_read_json(self.phase4_condition_summary_path, []))
        row_by_family = {
            row["family_name"]: row for row in condition_rows if row.get("family_name")
        }
        comparison_rows: list[dict[str, Any]] = []
        report_lines = [
            "# Phase 4 Defender-Authority Ablation",
            "",
            f"- Generated UTC: `{utc_now()}`",
            f"- Result root: `{portable_path(self.result_root)}`",
            f"- Seed: `{AUTHORITY_SEED}`",
            f"- New conditions completed: `{sum(int(row.get('completed_runs', 0)) for row in condition_rows)}/144`",
            f"- Full-containment baselines reused from: `{portable_path(BASELINE_TREATMENT_RESULT_ROOT)}` and `{portable_path(BASELINE_DOMAIN_RESULT_ROOT)}`",
            "",
            "## Condition Summary",
            "",
            "| Condition | Detect | Act | Cutoff | Contained First | Safety Violations | t_detect | t_action | t_cutoff | Violation Before Cutoff | Attacker Actions | Defender Actions | Downtime |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]

        for family in AUTHORITY_FAMILY_SPECS:
            row = row_by_family.get(family.family_name)
            if not row:
                continue
            report_lines.append(
                "| "
                + " | ".join(
                    [
                        family.family_name,
                        str(row.get("detection_rate")),
                        str(row.get("defender_action_rate")),
                        str(row.get("cutoff_activation_rate")),
                        str(row.get("contained_before_safety_violation_rate")),
                        str(row.get("safety_violation_rate")),
                        str(row.get("time_to_detection_from_wargame_t0_sec_avg")),
                        str(
                            row.get(
                                "time_to_first_defender_action_from_wargame_t0_sec_avg"
                            )
                        ),
                        str(row.get("time_to_cutoff_from_wargame_t0_sec_avg")),
                        str(row.get("safety_violation_before_cutoff_run_count")),
                        str(row.get("attacker_action_count_avg")),
                        str(row.get("defender_action_count_avg")),
                        str(row.get("downtime_timestep_count_avg")),
                    ]
                )
                + " |"
            )

        report_lines.extend(["", "## Baseline Comparison", ""])
        for attacker_group in [
            "black_box_attacker",
            "gray_box_attacker",
            "white_box_attacker",
            "black_box_high_cyber_high_domain",
        ]:
            matching_specs = [
                spec
                for spec in AUTHORITY_FAMILY_SPECS
                if spec.family_name.startswith(f"{attacker_group}__")
            ]
            if not matching_specs:
                continue
            baseline_spec = matching_specs[0]
            baseline_summary = _read_json(
                _baseline_condition_summary_path(baseline_spec),
                {},
            )
            baseline_family = seeded.FamilySpec(
                family_name=_baseline_family_name(baseline_spec),
                attacker_profile_id=baseline_spec.attacker_profile_id,
                knowledge_variant=baseline_spec.knowledge_variant,
                config_path=baseline_spec.config_path,
            )
            baseline_row = _condition_metric_row(baseline_family, baseline_summary)
            comparison_rows.append(
                {
                    "attacker_family": attacker_group,
                    "baseline": baseline_row,
                    "observe_alert_only": row_by_family.get(
                        f"{attacker_group}__observe_alert_only_defender"
                    ),
                    "delayed_containment": row_by_family.get(
                        f"{attacker_group}__delayed_containment_defender"
                    ),
                    "baseline_reference_root": portable_path(
                        _baseline_reference_root(baseline_spec)
                    ),
                }
            )
            report_lines.extend(
                [
                    f"### {attacker_group}",
                    f"- Full containment baseline safety-violation rate: `{baseline_row.get('safety_violation_rate')}`",
                    f"- Observe-alert-only safety-violation rate: `{(row_by_family.get(f'{attacker_group}__observe_alert_only_defender') or {}).get('safety_violation_rate')}`",
                    f"- Delayed-containment safety-violation rate: `{(row_by_family.get(f'{attacker_group}__delayed_containment_defender') or {}).get('safety_violation_rate')}`",
                    f"- Full containment cutoff activation rate: `{baseline_row.get('cutoff_activation_rate')}`",
                    f"- Observe-alert-only cutoff activation rate: `{(row_by_family.get(f'{attacker_group}__observe_alert_only_defender') or {}).get('cutoff_activation_rate')}`",
                    f"- Delayed-containment cutoff activation rate: `{(row_by_family.get(f'{attacker_group}__delayed_containment_defender') or {}).get('cutoff_activation_rate')}`",
                    f"- Baseline reference root: `{portable_path(_baseline_reference_root(baseline_spec))}`",
                    "",
                ]
            )

        write_json(self.baseline_comparison_path, comparison_rows)

        interpretation_lines = ["## Interpretation", ""]
        for item in comparison_rows:
            attacker_family = item["attacker_family"]
            baseline = dict(item.get("baseline") or {})
            observe = dict(item.get("observe_alert_only") or {})
            delayed = dict(item.get("delayed_containment") or {})
            if observe:
                interpretation_lines.append(
                    f"- `{attacker_family}` observe-alert-only detected `{observe.get('detection_count')}/18` runs while safety violations reached `{observe.get('safety_violation_count')}/18`, versus `{baseline.get('safety_violation_count')}/18` under full containment."
                )
            if delayed:
                interpretation_lines.append(
                    f"- `{attacker_family}` delayed containment activated cutoff in `{delayed.get('cutoff_activation_count')}/18` runs with average cutoff time `{delayed.get('time_to_cutoff_from_wargame_t0_sec_avg')}`, versus full containment cutoff rate `{baseline.get('cutoff_activation_rate')}`."
                )

        stronger_delayed = []
        for attacker_family in [
            "white_box_attacker",
            "black_box_high_cyber_high_domain",
        ]:
            delayed = next(
                (item for item in comparison_rows if item["attacker_family"] == attacker_family),
                None,
            )
            if delayed is None:
                continue
            baseline_rate = float(
                (delayed.get("baseline") or {}).get("safety_violation_rate") or 0.0
            )
            delayed_rate = float(
                (delayed.get("delayed_containment") or {}).get("safety_violation_rate")
                or 0.0
            )
            stronger_delayed.append((attacker_family, round(delayed_rate - baseline_rate, 4)))
        if stronger_delayed:
            strongest = ", ".join(
                f"`{family}` delta `{delta}`" for family, delta in stronger_delayed
            )
            interpretation_lines.append(
                f"- Delayed containment harms the stronger attacker regimes most clearly where the safety-violation delta versus full containment is highest: {strongest}."
            )
        report_lines.extend(interpretation_lines)
        report_text = "\n".join(report_lines) + "\n"
        seeded._write_text(self.phase4_report_path, report_text)
        seeded._write_text(self.results_summary_path, report_text)
        self._set_phase("phase4_report", "completed")

    def write_repo_note(self) -> None:
        resume_state = _read_json(self.resume_state_path, {})
        lines = [
            "# Defender Authority Ablation Note",
            "",
            f"- Experiment version: `{AUTHORITY_EXPERIMENT_VERSION}`",
            f"- Result root: `{portable_path(self.result_root)}`",
            f"- Config paths: `{', '.join(portable_path(path) for path in AUTHORITY_CONFIG_PATHS)}`",
            f"- Seed: `{AUTHORITY_SEED}`",
            f"- Condition count: `{len(AUTHORITY_FAMILY_SPECS)}`",
            f"- Resume event occurred: `{'true' if resume_state.get('resume_event_detected') else 'false'}`",
            f"- All 144 runs completed: `{'true' if resume_state.get('completed_runs') == 144 else 'false'}`",
            f"- Phase 4 report: `{portable_path(self.phase4_report_path)}`",
            "- Water-treatment only.",
            "- Existing full-containment baselines were reused for comparison.",
            "- No paper files were touched.",
            "- No secrets or credentials were committed.",
        ]
        seeded._write_text(self.repo_note_path, "\n".join(lines) + "\n")

    def run_controller(self) -> int:
        self._apply_authority_globals()
        self.freeze_metadata()
        self._mark_resume_event_if_needed(
            [row for row in csv.DictReader(self.run_queue_path.open(newline=""))]
            if self.run_queue_path.exists()
            else []
        )
        self._validate_optionb_environment()
        self._validate_runtime_and_tests()
        self._set_phase("running_grid", "in_progress")
        seeded._append_log(
            self.experiment_log_path,
            f"controller starting defender-authority ablation under {portable_path(self.result_root)}",
        )

        python_exec = seeded._python_executable()
        for family in AUTHORITY_FAMILY_SPECS:
            seed_status = _read_json(
                self._seed_status_path(family.family_name, AUTHORITY_SEED),
                {},
            )
            if seed_status.get("completed_scenarios") == 18:
                continue
            worker_log_path = (
                self.seed_progress_root(family.family_name, AUTHORITY_SEED)
                / "worker_run.json"
            )
            command = [
                python_exec,
                "-m",
                "ics_agent.water_treatment_defender_authority_ablation",
                "worker",
                "--family",
                family.family_name,
                "--seed",
                str(AUTHORITY_SEED),
                "--result-root",
                str(self.result_root),
            ]
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(AUTHORITY_SEED)
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
                    last_failure_seed=AUTHORITY_SEED,
                )
                seeded._append_log(
                    self.experiment_log_path,
                    f"worker failed for {family.family_name} seed {AUTHORITY_SEED}: {completed.stderr.strip()}",
                )
                return completed.returncode
            seeded._append_log(
                self.experiment_log_path,
                f"worker completed for {family.family_name} seed {AUTHORITY_SEED}",
            )

        self._sync_all_ledgers()
        resume_state = _read_json(self.resume_state_path, {})
        if resume_state.get("completed_runs") != 144:
            raise RuntimeError(
                f"expected 144 completed runs, found {resume_state.get('completed_runs')}"
            )
        self._set_phase("running_grid", "completed", completed_runs=144)
        self.write_family_aggregates()
        self.write_phase4_report()
        self.write_repo_note()
        return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the water-treatment Phase 4 defender-authority ablation."
        )
    )
    subparsers = parser.add_subparsers(dest="mode")

    controller = subparsers.add_parser("controller")
    controller.add_argument(
        "--result-root",
        default=str(AUTHORITY_RESULT_ROOT),
        help="Defender-authority-ablation result root.",
    )

    worker = subparsers.add_parser("worker")
    worker.add_argument(
        "--family",
        required=True,
        choices=sorted(AUTHORITY_FAMILY_BY_NAME),
    )
    worker.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[AUTHORITY_SEED],
    )
    worker.add_argument(
        "--result-root",
        default=str(AUTHORITY_RESULT_ROOT),
        help="Defender-authority-ablation result root.",
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
        args.result_root = str(AUTHORITY_RESULT_ROOT)
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runner = DefenderAuthorityAblationRunner(Path(args.result_root))
    if args.mode == "worker":
        return runner.run_worker(
            args.family,
            int(args.seed),
            max_attempts=int(args.max_attempts),
        )
    return runner.run_controller()


if __name__ == "__main__":
    raise SystemExit(main())
