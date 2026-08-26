from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import resolve_repo_path
from .orchestrator import ExperimentOrchestrator
from .paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from .reporting import write_json
from .utils import utc_now
from . import water_distribution_seeded_backbone as seeded
from . import water_treatment_defender_authority_ablation as authority_common


AUTHORITY_EXPERIMENT_VERSION = (
    "water_distribution_defender_authority_ablation_seed101_v6"
)
AUTHORITY_SEED = 101
AUTHORITY_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "water_distribution_openplc"
    / "defender-authority-ablation"
)
AUTHORITY_CONFIG_PATHS = [
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_black_box_vs_defender_authority_ablation.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_gray_box_vs_defender_authority_ablation.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_white_box_vs_defender_authority_ablation.yaml",
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_black_box_high_cyber_high_domain_vs_defender_authority_ablation.yaml",
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
        attacker_profile_id="black_box_high_cyber_high_domain_distribution_attacker",
        knowledge_variant="black_box_high_cyber_high_domain",
        config_path=AUTHORITY_CONFIG_PATHS[3],
    ),
    seeded.FamilySpec(
        family_name="black_box_high_cyber_high_domain__delayed_containment_defender",
        attacker_profile_id="black_box_high_cyber_high_domain_distribution_attacker",
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
    "tests/test_environment.py",
    "tests/test_metrics.py",
    "tests/test_orchestrator.py",
    "tests/test_phase35_profiles.py",
    "tests/test_defender_authority_ablation.py",
    "tests/test_water_distribution_seeded_backbone.py",
    "tests/test_water_distribution_domain_ablation.py",
    "tests/test_water_distribution_defender_authority_ablation.py",
]
BASELINE_DISTRIBUTION_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "water_distribution_openplc"
    / "seeded_backbone_qwen35_9b"
)
BASELINE_DOMAIN_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "water_distribution_openplc"
    / "domain-ablation"
    / "black_box_high_cyber_high_domain__agentic_defender"
)


def _condition_metric_row(
    family_spec: seeded.FamilySpec,
    condition_summary: dict[str, Any],
) -> dict[str, Any]:
    normalized_summary = dict(condition_summary)
    normalized_summary.setdefault("expected_run_count", seeded.SCENARIO_COUNT)
    return authority_common._condition_metric_row(family_spec, normalized_summary)


def _baseline_family_name(family_spec: seeded.FamilySpec) -> str:
    return {
        "black_box_attacker": "black_box_attacker__agentic_defender",
        "gray_box_attacker": "gray_box_attacker__agentic_defender",
        "white_box_attacker": "white_box_attacker__agentic_defender",
        "black_box_high_cyber_high_domain_distribution_attacker": (
            "black_box_high_cyber_high_domain__agentic_defender"
        ),
    }[family_spec.attacker_profile_id]


def _baseline_condition_summary_path(family_spec: seeded.FamilySpec) -> Path:
    if family_spec.attacker_profile_id == "black_box_high_cyber_high_domain_distribution_attacker":
        return BASELINE_DOMAIN_RESULT_ROOT / "condition_summary.json"
    return (
        BASELINE_DISTRIBUTION_RESULT_ROOT
        / _baseline_family_name(family_spec)
        / f"seed_{AUTHORITY_SEED}"
        / "condition_summary.json"
    )


def _baseline_reference_root(family_spec: seeded.FamilySpec) -> Path:
    if family_spec.attacker_profile_id == "black_box_high_cyber_high_domain_distribution_attacker":
        return BASELINE_DOMAIN_RESULT_ROOT
    return (
        BASELINE_DISTRIBUTION_RESULT_ROOT
        / _baseline_family_name(family_spec)
        / f"seed_{AUTHORITY_SEED}"
    )


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
        self.validate_log_path = self.progress_root / "validate_runtime.json"
        self.pytest_log_path = self.progress_root / "pytest_targeted.json"
        self.validation_tests = list(AUTHORITY_VALIDATION_TESTS)

    @staticmethod
    def _apply_authority_globals() -> None:
        seeded.EXPERIMENT_VERSION = AUTHORITY_EXPERIMENT_VERSION
        seeded.SEEDS = [AUTHORITY_SEED]
        seeded.FROZEN_CONFIG_PATHS = list(AUTHORITY_CONFIG_PATHS)
        seeded.FAMILY_SPECS = list(AUTHORITY_FAMILY_SPECS)
        seeded.FAMILY_BY_NAME = dict(AUTHORITY_FAMILY_BY_NAME)
        seeded.FAMILY_ORDER = dict(AUTHORITY_FAMILY_ORDER)
        seeded.ACTION_SPACE_SOURCE_PATHS = [
            REPO_ROOT / "ics_agent" / "action_semantics.py",
            REPO_ROOT / "ics_agent" / "environment.py",
            REPO_ROOT / "ics_agent" / "agents" / "action_space.py",
            REPO_ROOT / "ics_agent" / "agents" / "agentic.py",
            REPO_ROOT / "testbeds" / "runtime" / "water_distribution_runtime.py",
            REPO_ROOT
            / "testbeds"
            / "water_distribution"
            / "openplc"
            / "configs"
            / "openplc_map.yaml",
            REPO_ROOT / "configs" / "agent_profiles.yaml",
            REPO_ROOT / "configs" / "agent_profiles_overrides.yaml",
            *AUTHORITY_CONFIG_PATHS,
        ]
        seeded.RUNS_PER_FAMILY = seeded.SCENARIO_COUNT * len(seeded.SEEDS)
        seeded.TOTAL_RUNS = seeded.RUNS_PER_FAMILY * len(seeded.FAMILY_SPECS)
        seeded.CONTAINMENT_SEMANTICS_VERSION = "phase4_defender_authority_ablation_v1"
        seeded.TOP_LEVEL_PHASES = [
            "manifest_frozen",
            "validated_distribution_environment",
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

    def _family_model_settings(
        self,
        family_spec: seeded.FamilySpec,
    ) -> dict[str, dict[str, Any]]:
        orchestrator = self._orchestrator()
        config = self._runtime_config(family_spec, AUTHORITY_SEED)
        condition_spec = self._condition_spec(orchestrator, family_spec, config)
        effective_profiles = condition_spec.get("effective_profiles") or {}

        # Freeze experiment metadata from the configured profile view rather than
        # runtime-local environment defaults like QWEN_MODEL_PATH.
        settings: dict[str, dict[str, Any]] = {}
        for role, profile_id in condition_spec["profiles"].items():
            profile = (
                dict(effective_profiles.get(role) or {})
                or orchestrator.agent_profiles.get(profile_id, {})
            )
            if not isinstance(profile, dict) or profile.get("family") != "agentic":
                continue
            llm = dict(profile.get("llm") or {})
            if not llm:
                continue
            model_id = llm.get("model_id")
            settings[str(role)] = {
                "profile_id": str(profile_id),
                "backend": llm.get("backend"),
                "framework": llm.get("framework"),
                "model_id": model_id,
                "model_path": model_id,
                "temperature": float(llm.get("temperature", 0.0) or 0.0),
                "top_p": llm.get("top_p"),
                "do_sample": float(llm.get("temperature", 0.0) or 0.0) > 0.0,
                "max_new_tokens": llm.get("max_new_tokens"),
                "quantization": llm.get("quantization"),
                "context_limit": llm.get("context_limit"),
                "local_files_only": bool(llm.get("local_files_only", True)),
                "require_cuda": bool(llm.get("require_cuda", True)),
                "strict_backend": bool(llm.get("strict_backend", True)),
                "allow_langchain_fallback": bool(
                    llm.get("allow_langchain_fallback", True)
                ),
            }
        return settings

    def _prompt_hash_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        agent_profiles_path = REPO_ROOT / "configs" / "agent_profiles.yaml"
        profiles_overrides_path = REPO_ROOT / "configs" / "agent_profiles_overrides.yaml"
        hashed_paths: dict[str, str] = {
            portable_path(path): seeded._sha256_path(path)
            for path in [agent_profiles_path, profiles_overrides_path, *AUTHORITY_CONFIG_PATHS]
            if path.exists()
        }
        family_hashes: dict[str, dict[str, Any]] = {}
        for family in AUTHORITY_FAMILY_SPECS:
            config = self._runtime_config(family, AUTHORITY_SEED)
            condition_spec = self._condition_spec(orchestrator, family, config)
            effective_profiles = condition_spec.get("effective_profiles") or {}
            role_profiles: dict[str, dict[str, Any]] = {}
            for role, profile_id in condition_spec["profiles"].items():
                role_profiles[role] = dict(
                    effective_profiles.get(role)
                    or orchestrator.agent_profiles[profile_id]
                )

            prompt_paths: dict[str, Path] = {}
            knowledge_paths: dict[str, list[Path]] = {}
            for role, profile in role_profiles.items():
                llm = dict(profile.get("llm") or {})
                prompt_file = llm.get("prompt_file")
                if prompt_file:
                    prompt_paths[role] = resolve_repo_path(prompt_file, base_dir=REPO_ROOT)
                knowledge_paths[role] = [
                    resolve_repo_path(path, base_dir=REPO_ROOT)
                    for path in profile.get("knowledge_files", [])
                ]
            for path in [*prompt_paths.values(), *(path for paths in knowledge_paths.values() for path in paths)]:
                key = portable_path(path)
                if key not in hashed_paths and path.exists():
                    hashed_paths[key] = seeded._sha256_path(path)

            bundle = {
                "prompt_files": {
                    role: hashed_paths[portable_path(path)]
                    for role, path in prompt_paths.items()
                },
                "knowledge_files": {
                    role: {
                        portable_path(path): hashed_paths[portable_path(path)]
                        for path in paths
                    }
                    for role, paths in knowledge_paths.items()
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
                "Water-distribution Phase 4 defender-authority ablation semantics:",
                "- observe_alert_only_defender may detect, reason, alert, and note but may not actuate live containment or shutdown actions",
                "- delayed_containment_defender preserves the current distribution full defender action space",
                "- delayed_containment_defender records containment actions immediately and activates runtime cutoff after 2 timesteps",
                "- the retained 11-scenario suite, attacker profiles, and existing full-containment distribution baselines remain unchanged",
                "",
                "Primary source files:",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'action_semantics.py')}",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'environment.py')}",
                f"- {portable_path(REPO_ROOT / 'ics_agent' / 'orchestrator.py')}",
                f"- {portable_path(REPO_ROOT / 'testbeds' / 'runtime' / 'water_distribution_runtime.py')}",
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
        defender_variant = authority_common._defender_variant_label(
            authority_common._family_defender_profile_id(family_spec.family_name)
        )
        return "_".join(
            [
                seeded.TESTBED_ID,
                seeded._sanitize_token(scenario.scenario_id),
                split,
                seeded._sanitize_token(family_spec.attacker_profile_id),
                seeded._sanitize_token(defender_variant),
                f"seed{seed}",
            ]
        )

    def _seed_completed_row(self, **kwargs: Any) -> dict[str, Any]:
        row = super()._seed_completed_row(**kwargs)
        family_spec = kwargs["family_spec"]
        defender_profile_id = authority_common._family_defender_profile_id(
            family_spec.family_name
        )
        row["defender_variant"] = authority_common._defender_variant_label(
            defender_profile_id
        )
        row["robustness_variant"] = "defender_authority_ablation"
        return row

    def _validate_runtime_and_tests(self) -> None:
        payload = seeded._read_json(self.validate_log_path, {})
        pytest_payload = seeded._read_json(self.pytest_log_path, {})
        if payload.get("returncode") == 0 and pytest_payload.get("returncode") == 0:
            self._set_phase("validation_tests", "completed")
            return
        self._set_phase("validation_tests", "in_progress")
        python_exec = seeded._python_executable()
        self._run_logged(
            [python_exec, "-m", "ics_agent.cli", "validate"],
            self.validate_log_path,
        )
        os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
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

    def write_family_aggregates(self) -> None:
        self._set_phase("aggregating", "in_progress")
        summary_rows: list[dict[str, Any]] = []
        for family in AUTHORITY_FAMILY_SPECS:
            seed_root = self.seed_root(family.family_name, AUTHORITY_SEED)
            family_root = self.family_root(family.family_name)
            seed_summary = seeded._read_json(seed_root / "seed_summary.json", {})
            attack_metrics = seeded._read_json(seed_root / "attack_metrics_summary.json", {})
            trajectory_metrics = seeded._read_json(
                seed_root / "trajectory_metrics_summary.json",
                {},
            )
            mitre_phase_metrics = seeded._read_json(
                seed_root / "mitre_phase_metrics.json",
                {},
            )
            nominal_metrics = seeded._read_json(
                seed_root / "nominal_metrics_summary.json",
                {},
            )
            condition_summary = seeded._read_json(
                seed_root / "condition_summary.json",
                {},
            )
            if not seed_summary or not condition_summary:
                continue

            metric_row = _condition_metric_row(family, condition_summary)
            summary_rows.append(metric_row)
            aggregate_row = {
                "family_name": family.family_name,
                "seed": AUTHORITY_SEED,
                "total_runs": seeded.SCENARIO_COUNT,
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
                        (
                            f"- Defender profile: "
                            f"`{authority_common._family_defender_profile_id(family.family_name)}`"
                        ),
                        f"- Baseline reference root: `{portable_path(_baseline_reference_root(family))}`",
                        f"- Testbed: `{seeded.TESTBED_ID}`",
                        f"- Seed: `{AUTHORITY_SEED}`",
                        f"- Retained scenarios: `{seeded.SCENARIO_COUNT}`",
                        (
                            f"- Completed runs: "
                            f"`{seed_summary.get('completed_scenarios')}/{seeded.SCENARIO_COUNT}`"
                        ),
                        f"- Detection rate: `{metric_row['detection_rate']}`",
                        f"- Defender-action rate: `{metric_row['defender_action_rate']}`",
                        f"- Cutoff activation rate: `{metric_row['cutoff_activation_rate']}`",
                        (
                            f"- Contained-before-safety-violation rate: "
                            f"`{metric_row['contained_before_safety_violation_rate']}`"
                        ),
                        f"- Safety-violation rate: `{metric_row['safety_violation_rate']}`",
                        (
                            f"- Average time to detection from wargame_t0: "
                            f"`{metric_row['time_to_detection_from_wargame_t0_sec_avg']}`"
                        ),
                        (
                            f"- Average time to first defender action from wargame_t0: "
                            f"`{metric_row['time_to_first_defender_action_from_wargame_t0_sec_avg']}`"
                        ),
                        (
                            f"- Average time to cutoff from wargame_t0: "
                            f"`{metric_row['time_to_cutoff_from_wargame_t0_sec_avg']}`"
                        ),
                        (
                            f"- Safety violation before cutoff runs: "
                            f"`{metric_row['safety_violation_before_cutoff_run_count']}`"
                        ),
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
        condition_rows = list(seeded._read_json(self.phase4_condition_summary_path, []))
        row_by_family = {
            row["family_name"]: row for row in condition_rows if row.get("family_name")
        }
        comparison_rows: list[dict[str, Any]] = []
        total_completed = sum(int(row.get("completed_runs", 0) or 0) for row in condition_rows)
        report_lines = [
            "# Phase 4 Defender-Authority Ablation",
            "",
            f"- Generated UTC: `{utc_now()}`",
            f"- Testbed: `{seeded.TESTBED_ID}`",
            f"- Result root: `{portable_path(self.result_root)}`",
            f"- Seed set: `{AUTHORITY_SEED}`",
            f"- Retained suite size: `{seeded.SCENARIO_COUNT}` scenarios",
            f"- New conditions completed: `{total_completed}/{seeded.TOTAL_RUNS}`",
            (
                f"- Full-containment baselines reused from: "
                f"`{portable_path(BASELINE_DISTRIBUTION_RESULT_ROOT)}` and "
                f"`{portable_path(BASELINE_DOMAIN_RESULT_ROOT)}`"
            ),
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
            baseline_summary = seeded._read_json(
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
                    (
                        f"- Full containment safety-violation rate: "
                        f"`{baseline_row.get('safety_violation_rate')}`"
                    ),
                    (
                        f"- Observe-alert-only safety-violation rate: "
                        f"`{(row_by_family.get(f'{attacker_group}__observe_alert_only_defender') or {}).get('safety_violation_rate')}`"
                    ),
                    (
                        f"- Delayed-containment safety-violation rate: "
                        f"`{(row_by_family.get(f'{attacker_group}__delayed_containment_defender') or {}).get('safety_violation_rate')}`"
                    ),
                    (
                        f"- Full containment cutoff activation rate: "
                        f"`{baseline_row.get('cutoff_activation_rate')}`"
                    ),
                    (
                        f"- Observe-alert-only cutoff activation rate: "
                        f"`{(row_by_family.get(f'{attacker_group}__observe_alert_only_defender') or {}).get('cutoff_activation_rate')}`"
                    ),
                    (
                        f"- Delayed-containment cutoff activation rate: "
                        f"`{(row_by_family.get(f'{attacker_group}__delayed_containment_defender') or {}).get('cutoff_activation_rate')}`"
                    ),
                    (
                        f"- Baseline reference root: "
                        f"`{portable_path(_baseline_reference_root(baseline_spec))}`"
                    ),
                    "",
                ]
            )

        write_json(self.baseline_comparison_path, comparison_rows)

        report_lines.extend(
            [
                "## Interpretation",
                "",
            ]
        )
        for item in comparison_rows:
            attacker_family = item["attacker_family"]
            baseline = dict(item.get("baseline") or {})
            observe = dict(item.get("observe_alert_only") or {})
            delayed = dict(item.get("delayed_containment") or {})
            if observe:
                report_lines.append(
                    (
                        f"- `{attacker_family}` observe-alert-only detected "
                        f"`{observe.get('detection_count')}/{seeded.SCENARIO_COUNT}` runs while "
                        f"safety violations reached `{observe.get('safety_violation_count')}/{seeded.SCENARIO_COUNT}`, "
                        f"versus `{baseline.get('safety_violation_count')}/{seeded.SCENARIO_COUNT}` under full containment."
                    )
                )
            if delayed:
                report_lines.append(
                    (
                        f"- `{attacker_family}` delayed containment activated cutoff in "
                        f"`{delayed.get('cutoff_activation_count')}/{seeded.SCENARIO_COUNT}` runs with "
                        f"average cutoff time `{delayed.get('time_to_cutoff_from_wargame_t0_sec_avg')}`, "
                        f"versus full containment cutoff rate `{baseline.get('cutoff_activation_rate')}`."
                    )
                )

        report_text = "\n".join(report_lines) + "\n"
        seeded._write_text(self.phase4_report_path, report_text)
        seeded._write_text(self.results_summary_path, report_text)
        self._set_phase("phase4_report", "completed")

    def write_repo_note(self) -> None:
        resume_state = seeded._read_json(self.resume_state_path, {})
        lines = [
            "# Defender Authority Ablation Note",
            "",
            f"- Experiment version: `{AUTHORITY_EXPERIMENT_VERSION}`",
            f"- Result root: `{portable_path(self.result_root)}`",
            (
                f"- Config paths: "
                f"`{', '.join(portable_path(path) for path in AUTHORITY_CONFIG_PATHS)}`"
            ),
            f"- Testbed: `{seeded.TESTBED_ID}`",
            f"- Seed: `{AUTHORITY_SEED}`",
            f"- Retained scenarios: `{seeded.SCENARIO_COUNT}`",
            f"- Condition count: `{len(AUTHORITY_FAMILY_SPECS)}`",
            (
                f"- Resume event occurred: "
                f"`{'true' if resume_state.get('resume_event_detected') else 'false'}`"
            ),
            (
                f"- All {seeded.TOTAL_RUNS} runs completed: "
                f"`{'true' if resume_state.get('completed_runs') == seeded.TOTAL_RUNS else 'false'}`"
            ),
            f"- Phase 4 report: `{portable_path(self.phase4_report_path)}`",
            "- Water-distribution only.",
            "- Existing full-containment baselines were reused for comparison.",
            "- Water-treatment was not rerun.",
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
        self._validate_distribution_environment()
        self._validate_runtime_and_tests()
        self._set_phase("running_grid", "in_progress")
        seeded._append_log(
            self.experiment_log_path,
            f"controller starting defender-authority ablation under {portable_path(self.result_root)}",
        )

        python_exec = seeded._python_executable()
        for family in AUTHORITY_FAMILY_SPECS:
            seed_status = seeded._read_json(
                self._seed_status_path(family.family_name, AUTHORITY_SEED),
                {},
            )
            if seed_status.get("completed_scenarios") == seeded.SCENARIO_COUNT:
                continue
            worker_log_path = (
                self.seed_progress_root(family.family_name, AUTHORITY_SEED)
                / "worker_run.json"
            )
            command = [
                python_exec,
                "-m",
                "ics_agent.water_distribution_defender_authority_ablation",
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
                    (
                        f"worker failed for {family.family_name} seed {AUTHORITY_SEED}; "
                        "inspect the family worker_run.json for detailed stderr/stdout"
                    ),
                )
                return completed.returncode
            seeded._append_log(
                self.experiment_log_path,
                f"worker completed for {family.family_name} seed {AUTHORITY_SEED}",
            )

        self._sync_all_ledgers()
        resume_state = seeded._read_json(self.resume_state_path, {})
        if resume_state.get("completed_runs") != seeded.TOTAL_RUNS:
            raise RuntimeError(
                f"expected {seeded.TOTAL_RUNS} completed runs, found {resume_state.get('completed_runs')}"
            )
        self._set_phase("running_grid", "completed", completed_runs=seeded.TOTAL_RUNS)
        self.write_family_aggregates()
        self.write_phase4_report()
        self.write_repo_note()
        return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the water-distribution Phase 4 defender-authority ablation."
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
