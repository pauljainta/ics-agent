from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ...agents.agentic import AgenticRoleAgent
from ...config import resolve_repo_path
from ...orchestrator import ExperimentOrchestrator
from ...paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from ...reporting import write_json
from ...scenarios import ScenarioCatalog
from ...utils import utc_now
from . import seeded_backbone as seeded


DOMAIN_EXPERIMENT_VERSION = (
    "water_distribution_domain_ablation_black_box_high_cyber_high_domain_wadi_enriched_seed101_v1"
)
DOMAIN_SEED = 101
DOMAIN_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_domain_ablation_black_box_high_cyber_high_domain_vs_agentic_defender.yaml"
)
DOMAIN_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "water_distribution_openplc"
    / "domain-ablation"
    / "black_box_high_cyber_high_domain__agentic_defender"
)
DOMAIN_FAMILY_SPEC = seeded.FamilySpec(
    family_name="black_box_high_cyber_high_domain__agentic_defender",
    attacker_profile_id="black_box_high_cyber_high_domain_distribution_attacker",
    knowledge_variant="black_box_high_cyber_high_domain",
    config_path=DOMAIN_CONFIG_PATH,
)
FORBIDDEN_LEAKAGE_PHRASES = (
    "best leverage surface",
    "highest leverage",
    "preferred attack surface",
    "surface ranking",
    "surface_leverage_hints",
)
DEFAULT_VALIDATION_TESTS = [
    "tests/test_phase35_profiles.py",
    "tests/test_water_distribution_seeded_backbone.py",
    "tests/test_water_distribution_domain_ablation.py",
]
ENRICHED_CONDITION_LABEL = "WADI-informed documentation-enriched black-box attacker"
ENRICHED_CONDITION_NOTE = (
    "Distribution primer intentionally enriched with WADI-derived attack/process "
    "knowledge; this is not the earlier strict target-blind high-domain primer."
)
BLACK_BOX_RUNTIME_NOTE = (
    "Runtime attacker interface remained black-box with knowledge_mode = "
    "black_box, system_knowledge = None, and the existing distribution black-box "
    "observation/action interface unchanged."
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_prompt_path(path_text: str | os.PathLike[str]) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _profile_prompt_path(profile: dict[str, Any]) -> Path | None:
    llm = dict(profile.get("llm") or {})
    prompt_file = llm.get("prompt_file")
    if not prompt_file:
        return None
    return _resolve_prompt_path(prompt_file)


def _profile_knowledge_paths(profile: dict[str, Any]) -> list[Path]:
    return [
        resolve_repo_path(path, base_dir=REPO_ROOT)
        for path in profile.get("knowledge_files", [])
    ]


def _forbidden_distribution_tokens() -> set[str]:
    contract = yaml.safe_load(
        (REPO_ROOT / "testbeds" / "water_distribution" / "tag_contract.yaml").read_text()
    )
    tokens = set((contract.get("tags") or {}).keys())
    tokens.update({"WD-UC0", "WD-Supply", "WD-Grid", "WD-Return"})
    catalog = ScenarioCatalog(testbed_id=seeded.TESTBED_ID)
    tokens.update(
        scenario.catalog_id.split("/")[-1]
        for scenario in catalog.resolve_selectors(["attack/*"])
    )
    return tokens


def _audit_distribution_observation(**overrides: float | str) -> dict[str, float | str]:
    observation: dict[str, float | str] = {
        "STATE_NAME": "RUNNING",
        "Supply_Tank_Level": 920.0,
        "Supply_Pump_Flow": 140.0,
        "Supply_Pump_Speed_Cmd": 72.0,
        "Supply_Pump_Sts": 1.0,
        "Supply_Tank_Valve_Cmd": 1.0,
        "Supply_Tank_Valve_Sts": 1.0,
        "Grid_Elev_Tank_Level": 1250.0,
        "Grid_Consum_Tank_Level": 830.0,
        "Grid_Elev_Valve_Cmd": 0.0,
        "Grid_Elev_Valve_Sts": 0.0,
        "RWS_Tank_Level": 540.0,
        "RWS_Pump_Speed_Cmd": 0.0,
        "RWS_Pump_Sts": 0.0,
        "RWS_Tank_Valve_Cmd": 0.0,
        "RWS_Tank_Valve_Sts": 0.0,
        "Alarm_Supply_Tank_HH": 0.0,
        "HMI_Start_PB": 0.0,
        "HMI_Stop_PB": 0.0,
    }
    observation.update(overrides)
    return observation


def _scan_text_for_leakage(text: str, *, tokens: set[str]) -> list[str]:
    findings: list[str] = []
    for token in sorted(tokens):
        if token and token in text:
            findings.append(f"forbidden token `{token}` present")
    lowered = text.lower()
    for phrase in FORBIDDEN_LEAKAGE_PHRASES:
        if phrase in lowered:
            findings.append(f"forbidden phrase `{phrase}` present")
    return findings


def _domain_python_executable() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python.resolve())
    return seeded._python_executable()


class DomainAblationRunner(seeded.SeededBackboneRunner):
    def __init__(self, result_root: Path = DOMAIN_RESULT_ROOT) -> None:
        self._apply_domain_globals()
        super().__init__(result_root)
        self.audit_root = self.result_root / "prompt_knowledge_audit"
        self.audit_packages_root = self.audit_root / "scenario_packages"
        self.audit_summary_path = self.audit_root / "audit_summary.json"
        self.audit_report_path = self.audit_root / "audit_report.md"
        self.audit_manifest_path = self.audit_root / "prompt_knowledge_manifest.json"
        self.repo_note_path = self.result_root / "domain_ablation_note.md"
        self.validate_log_path = self.progress_root / "validate_runtime.json"
        self.pytest_log_path = self.progress_root / "pytest_targeted.json"
        self.validation_tests = list(DEFAULT_VALIDATION_TESTS)

    @staticmethod
    def _apply_domain_globals() -> None:
        seeded.EXPERIMENT_VERSION = DOMAIN_EXPERIMENT_VERSION
        seeded.SEEDS = [DOMAIN_SEED]
        seeded.FROZEN_CONFIG_PATHS = [DOMAIN_CONFIG_PATH]
        seeded.FAMILY_SPECS = [DOMAIN_FAMILY_SPEC]
        seeded.FAMILY_BY_NAME = {DOMAIN_FAMILY_SPEC.family_name: DOMAIN_FAMILY_SPEC}
        seeded.FAMILY_ORDER = {DOMAIN_FAMILY_SPEC.family_name: 0}
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
            *seeded.PROFILE_REGISTRY_PATHS,
            DOMAIN_CONFIG_PATH,
        ]
        seeded.RUNS_PER_FAMILY = seeded.SCENARIO_COUNT * len(seeded.SEEDS)
        seeded.TOTAL_RUNS = seeded.RUNS_PER_FAMILY * len(seeded.FAMILY_SPECS)

    def _attacker_profile_bundle(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        orchestrator = self._orchestrator()
        config = self._runtime_config(DOMAIN_FAMILY_SPEC, DOMAIN_SEED)
        condition_spec = self._condition_spec(orchestrator, DOMAIN_FAMILY_SPEC, config)
        effective_profiles = condition_spec.get("effective_profiles") or {}
        attacker_profile = dict(
            effective_profiles.get("attacker")
            or orchestrator.agent_profiles[DOMAIN_FAMILY_SPEC.attacker_profile_id]
        )
        return config, condition_spec, effective_profiles, attacker_profile

    def _attacker_prompt_and_knowledge_paths(self) -> tuple[Path, list[Path]]:
        _, _, _, attacker_profile = self._attacker_profile_bundle()
        prompt_path = _profile_prompt_path(attacker_profile)
        if prompt_path is None:
            raise RuntimeError("distribution attacker prompt path is required")
        return prompt_path, _profile_knowledge_paths(attacker_profile)

    def _prompt_hash_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        config = self._runtime_config(DOMAIN_FAMILY_SPEC, DOMAIN_SEED)
        condition_spec = self._condition_spec(orchestrator, DOMAIN_FAMILY_SPEC, config)
        effective_profiles = condition_spec.get("effective_profiles") or {}
        role_profiles: dict[str, dict[str, Any]] = {}
        for role, profile_id in condition_spec["profiles"].items():
            role_profiles[role] = dict(
                effective_profiles.get(role) or orchestrator.agent_profiles[profile_id]
            )

        prompt_paths = {
            role: _profile_prompt_path(profile)
            for role, profile in role_profiles.items()
            if _profile_prompt_path(profile) is not None
        }
        knowledge_paths = {
            role: _profile_knowledge_paths(profile)
            for role, profile in role_profiles.items()
            if profile.get("knowledge_files")
        }
        hashed_paths = [
            *(path for path in prompt_paths.values()),
            *(path for paths in knowledge_paths.values() for path in paths),
            *seeded.PROFILE_REGISTRY_PATHS,
            DOMAIN_CONFIG_PATH,
        ]
        file_hashes = {
            portable_path(path): seeded._sha256_path(path)
            for path in hashed_paths
            if path.exists()
        }
        knowledge_bundle = {
            role: {
                portable_path(path): file_hashes[portable_path(path)]
                for path in paths
            }
            for role, paths in knowledge_paths.items()
        }
        bundle = {
            "prompt_files": {
                role: file_hashes[portable_path(path)]
                for role, path in prompt_paths.items()
            },
            "knowledge_files": knowledge_bundle,
            "profile_registries": {
                portable_path(path): file_hashes[portable_path(path)]
                for path in seeded.PROFILE_REGISTRY_PATHS
            },
            "experiment_config": file_hashes[portable_path(DOMAIN_CONFIG_PATH)],
        }
        return {
            "experiment_version": DOMAIN_EXPERIMENT_VERSION,
            "file_hashes": file_hashes,
            "family_prompt_hashes": {
                DOMAIN_FAMILY_SPEC.family_name: {
                    "bundle_hash": seeded._canonical_json_hash(bundle),
                    "bundle": bundle,
                }
            },
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
            f"- Experiment version: `{DOMAIN_EXPERIMENT_VERSION}`",
            f"- Condition root: `{portable_path(self.result_root)}`",
            f"- Testbed: `{seeded.TESTBED_ID}`",
            f"- Family: `{DOMAIN_FAMILY_SPEC.family_name}`",
            f"- Attacker profile: `{DOMAIN_FAMILY_SPEC.attacker_profile_id}`",
            f"- Seed: `{DOMAIN_SEED}`",
            f"- Completed runs: `{completed_count}`",
            f"- Failed runs: `{failed_count}`",
            f"- Pending runs: `{pending_count}`",
            f"- Prompt audit root: `{portable_path(self.audit_root)}`",
            "- Water-distribution only.",
            "- Water-treatment touched: `false`",
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
        seeded._write_text(self.resume_notes_path, "\n".join(lines) + "\n")

    def _validate_runtime_and_tests(self) -> None:
        payload = _read_json(self.validate_log_path, {})
        pytest_payload = _read_json(self.pytest_log_path, {})
        if payload.get("returncode") == 0 and pytest_payload.get("returncode") == 0:
            self._set_phase("validation_tests", "completed")
            return
        self._set_phase("validation_tests", "in_progress")
        python_exec = _domain_python_executable()
        self._run_logged(
            [python_exec, "-m", "ics_agent.cli", "validate", "--testbed", seeded.TESTBED_ID],
            self.validate_log_path,
        )
        previous_pytest_autoload = os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        try:
            self._run_logged(
                [python_exec, "-m", "pytest", *self.validation_tests],
                self.pytest_log_path,
            )
        finally:
            if previous_pytest_autoload is None:
                os.environ.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
            else:
                os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = previous_pytest_autoload
        self._set_phase(
            "validation_tests",
            "completed",
            validate_log=portable_path(self.validate_log_path),
            pytest_log=portable_path(self.pytest_log_path),
        )

    def write_prompt_knowledge_audit(self) -> None:
        self.audit_root.mkdir(parents=True, exist_ok=True)
        self.audit_packages_root.mkdir(parents=True, exist_ok=True)
        _, condition_spec, _, attacker_profile = self._attacker_profile_bundle()
        prompt_path, knowledge_paths = self._attacker_prompt_and_knowledge_paths()
        role_profiles = self._condition_role_profiles(condition_spec)
        defender_prompt = _profile_prompt_path(role_profiles["defender"])
        defender_knowledge_paths = _profile_knowledge_paths(role_profiles["defender"])
        attacker_hash_inputs = [
            prompt_path,
            *knowledge_paths,
            *( [defender_prompt] if defender_prompt else [] ),
            *defender_knowledge_paths,
            DOMAIN_CONFIG_PATH,
        ]
        file_hashes = {
            portable_path(path): seeded._sha256_path(path)
            for path in attacker_hash_inputs
        }
        prompt_text = prompt_path.read_text()
        knowledge_text = "\n\n".join(path.read_text() for path in knowledge_paths)
        static_findings = _scan_text_for_leakage(
            prompt_text + "\n\n" + knowledge_text,
            tokens=_forbidden_distribution_tokens(),
        )
        if re.search(r"\b\d{3,}\b", knowledge_text):
            static_findings.append("knowledge bundle contains 3+ digit numeric token(s)")

        package_rows: list[dict[str, Any]] = []
        all_findings = list(static_findings)
        for scenario in self.scenarios():
            observation = _audit_distribution_observation()
            agent = AgenticRoleAgent("domain_ablation_prompt_audit", attacker_profile)
            agent.reset(scenario)
            context = agent._build_online_context(
                {
                    "scenario": scenario,
                    "observation": observation,
                    "observations": [observation],
                    "executed_actions": [],
                    "decisions": [],
                    "alerts": [],
                    "playbooks": {},
                    "time_sec": 7.0,
                }
            )
            assembled_prompt = agent.engine._format_prompt(context)
            findings = []
            if context.get("system_knowledge") is not None:
                findings.append("system_knowledge must remain None for black-box attacker")
            if context.get("process_id") is not None:
                findings.append("process_id leaked into black-box context")
            if context.get("scenario_id") != "opaque-interface-live":
                findings.append(
                    f"scenario id leaked into black-box context: {context.get('scenario_id')}"
                )
            if context.get("safety_bounds") != {}:
                findings.append("safety_bounds must remain empty for black-box context")
            findings.extend(
                _scan_text_for_leakage(
                    assembled_prompt,
                    tokens=_forbidden_distribution_tokens(),
                )
            )
            package_payload = {
                "generated_at_utc": utc_now(),
                "scenario_catalog_id": scenario.catalog_id,
                "scenario_id": scenario.scenario_id,
                "reset_state_name": observation.get("STATE_NAME"),
                "context": {
                    "scenario_id": context.get("scenario_id"),
                    "process_id": context.get("process_id"),
                    "system_knowledge": context.get("system_knowledge"),
                    "knowledge_ablation": context.get("knowledge_ablation"),
                    "safety_bounds": context.get("safety_bounds"),
                    "role_constraints": context.get("role_constraints"),
                    "candidate_actions": context.get("candidate_actions"),
                    "observation": context.get("observation"),
                    "short_horizon_trends": context.get("short_horizon_trends"),
                },
                "prompt_sha256": _sha256_text(assembled_prompt),
                "prompt_path": portable_path(prompt_path),
                "knowledge_paths": [portable_path(path) for path in knowledge_paths],
                "assembled_prompt_path": portable_path(
                    self.audit_packages_root / f"{scenario.scenario_id}.prompt.txt"
                ),
                "findings": findings,
            }
            write_json(
                self.audit_packages_root / f"{scenario.scenario_id}.context.json",
                package_payload,
            )
            seeded._write_text(
                self.audit_packages_root / f"{scenario.scenario_id}.prompt.txt",
                assembled_prompt,
            )
            package_rows.append(package_payload)
            all_findings.extend(f"{scenario.scenario_id}: {item}" for item in findings)

        summary = {
            "generated_at_utc": utc_now(),
            "experiment_version": DOMAIN_EXPERIMENT_VERSION,
            "result_root": portable_path(self.result_root),
            "condition_name": DOMAIN_FAMILY_SPEC.family_name,
            "attacker_profile": DOMAIN_FAMILY_SPEC.attacker_profile_id,
            "defender_profile": condition_spec["profiles"]["defender"],
            "observer_profile": condition_spec["profiles"]["observer"],
            "attacker_condition_label": ENRICHED_CONDITION_LABEL,
            "attacker_condition_note": ENRICHED_CONDITION_NOTE,
            "knowledge_mode": attacker_profile.get("knowledge_mode"),
            "system_knowledge": None,
            "runtime_interface_note": BLACK_BOX_RUNTIME_NOTE,
            "prompt_paths": {
                "attacker": portable_path(prompt_path),
                "defender": portable_path(defender_prompt) if defender_prompt else None,
            },
            "knowledge_paths": {
                "attacker": [portable_path(path) for path in knowledge_paths],
                "defender": [portable_path(path) for path in defender_knowledge_paths],
            },
            "file_hashes": file_hashes,
            "bundle_hash": seeded._canonical_json_hash(file_hashes),
            "scenario_package_count": len(package_rows),
            "leakage_findings": all_findings,
            "packages": package_rows,
        }
        write_json(self.audit_manifest_path, summary)
        write_json(self.audit_summary_path, summary)
        report_lines = [
            "# Prompt And Knowledge Audit",
            "",
            f"- Generated UTC: `{utc_now()}`",
            f"- Experiment version: `{DOMAIN_EXPERIMENT_VERSION}`",
            f"- Condition: `{DOMAIN_FAMILY_SPEC.family_name}`",
            f"- Attacker profile: `{DOMAIN_FAMILY_SPEC.attacker_profile_id}`",
            "- Defender profile: `agentic_defender`",
            f"- Condition interpretation: `{ENRICHED_CONDITION_LABEL}`",
            f"- Enrichment note: `{ENRICHED_CONDITION_NOTE}`",
            f"- Runtime interface note: `{BLACK_BOX_RUNTIME_NOTE}`",
            f"- Attacker prompt path: `{portable_path(prompt_path)}`",
            f"- Attacker knowledge files: `{', '.join(portable_path(path) for path in knowledge_paths)}`",
            f"- Defender prompt path: `{portable_path(defender_prompt) if defender_prompt else 'n/a'}`",
            f"- Defender knowledge files: `{', '.join(portable_path(path) for path in defender_knowledge_paths)}`",
            f"- Audit manifest: `{portable_path(self.audit_manifest_path)}`",
            f"- Package count: `{len(package_rows)}`",
            f"- Leakage findings: `{len(all_findings)}`",
        ]
        seeded._write_text(self.audit_report_path, "\n".join(report_lines) + "\n")
        if all_findings:
            raise RuntimeError(
                "prompt/knowledge audit failed: " + "; ".join(all_findings[:8])
            )

    def _condition_role_profiles(
        self,
        condition_spec: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        orchestrator = self._orchestrator()
        effective_profiles = condition_spec.get("effective_profiles") or {}
        return {
            role: dict(
                effective_profiles.get(role)
                or orchestrator.agent_profiles[profile_id]
            )
            for role, profile_id in condition_spec["profiles"].items()
        }

    def write_family_aggregates(self) -> None:
        self._set_phase("aggregating", "in_progress")
        seed_root = self.seed_root(DOMAIN_FAMILY_SPEC.family_name, DOMAIN_SEED)
        family_root = self.family_root(DOMAIN_FAMILY_SPEC.family_name)
        family_root.mkdir(parents=True, exist_ok=True)
        prompt_path, knowledge_paths = self._attacker_prompt_and_knowledge_paths()
        _, condition_spec, _, _ = self._attacker_profile_bundle()
        role_profiles = self._condition_role_profiles(condition_spec)
        defender_prompt = _profile_prompt_path(role_profiles["defender"])
        defender_knowledge_paths = _profile_knowledge_paths(role_profiles["defender"])
        audit_summary = _read_json(self.audit_summary_path, {})
        seed_summary = _read_json(seed_root / "seed_summary.json", {})
        if not seed_summary:
            raise RuntimeError("seed summary missing for domain ablation run")
        attack_metrics = _read_json(seed_root / "attack_metrics_summary.json", {})
        trajectory_metrics = _read_json(seed_root / "trajectory_metrics_summary.json", {})
        condition_summary = _read_json(seed_root / "condition_summary.json", {})
        aggregate_row = {
            "family_name": DOMAIN_FAMILY_SPEC.family_name,
            "seed": DOMAIN_SEED,
            "total_runs": seeded.SCENARIO_COUNT,
            "total_completed_runs": seed_summary.get("completed_scenarios"),
            "safety_violation_count": attack_metrics.get(
                "attacker_caused_safety_violation_run_count"
            ),
            "detected_count": attack_metrics.get("defender_detected_attack_run_count"),
            "defender_action_count": attack_metrics.get("defender_acted_attack_run_count"),
            "avg_wall_clock_detection_from_attack_start_sec": attack_metrics.get(
                "wall_clock_time_to_detection_from_attack_start_sec_avg"
            ),
            "avg_attacker_action_count": attack_metrics.get("attacker_action_count_avg"),
            "avg_downtime_timestep_count": attack_metrics.get("downtime_timestep_count_avg"),
        }
        write_json(family_root / "aggregate_over_seeds.json", aggregate_row)
        seeded.write_csv(family_root / "aggregate_over_seeds.csv", [aggregate_row])
        write_json(family_root / "family_summary.json", seed_summary)
        write_json(family_root / "family_condition_summary.json", condition_summary)
        seeded._write_text(
            family_root / "family_report.md",
            "\n".join(
                [
                    f"# {DOMAIN_FAMILY_SPEC.family_name}",
                    "",
                    f"- Attacker profile: `{DOMAIN_FAMILY_SPEC.attacker_profile_id}`",
                    "- Defender profile: `agentic_defender`",
                    f"- Testbed: `{seeded.TESTBED_ID}`",
                    f"- Seed: `{DOMAIN_SEED}`",
                    f"- Condition interpretation: `{ENRICHED_CONDITION_LABEL}`",
                    f"- Enrichment note: `{ENRICHED_CONDITION_NOTE}`",
                    f"- Runtime interface note: `{BLACK_BOX_RUNTIME_NOTE}`",
                    f"- Attacker prompt: `{portable_path(prompt_path)}`",
                    f"- Attacker knowledge files: `{', '.join(portable_path(path) for path in knowledge_paths)}`",
                    f"- Defender prompt: `{portable_path(defender_prompt) if defender_prompt else 'n/a'}`",
                    f"- Defender knowledge files: `{', '.join(portable_path(path) for path in defender_knowledge_paths)}`",
                    f"- Prompt and knowledge audit manifest: `{portable_path(self.audit_manifest_path)}`",
                    f"- Prompt and knowledge bundle hash: `{audit_summary.get('bundle_hash')}`",
                    f"- Completed runs: `{seed_summary.get('completed_scenarios')}/{seeded.SCENARIO_COUNT}`",
                    f"- Detected scenarios: `{attack_metrics.get('defender_detected_attack_run_count')}`",
                    f"- Defender-action scenarios: `{attack_metrics.get('defender_acted_attack_run_count')}`",
                    f"- Safety-violation scenarios: `{attack_metrics.get('attacker_caused_safety_violation_run_count')}`",
                    f"- Average wall-clock detection from attack start: `{seeded._format_mean_std(attack_metrics.get('wall_clock_time_to_detection_from_attack_start_sec_avg'), 0.0)}`",
                    f"- Average attacker action count: `{seeded._format_mean_std(attack_metrics.get('attacker_action_count_avg'), 0.0)}`",
                    f"- Average downtime burden: `{seeded._format_mean_std(attack_metrics.get('downtime_timestep_count_avg'), 0.0)}`",
                ]
            )
            + "\n",
        )
        write_json(self.result_root / "condition_summary.json", condition_summary)
        write_json(self.result_root / "attack_metrics_summary.json", attack_metrics)
        write_json(self.result_root / "trajectory_metrics_summary.json", trajectory_metrics)
        summary_lines = [
            "# Results Summary",
            "",
            f"- Generated UTC: `{utc_now()}`",
            f"- Attacker profile: `{DOMAIN_FAMILY_SPEC.attacker_profile_id}`",
            "- Defender profile: `agentic_defender`",
            f"- Testbed: `{seeded.TESTBED_ID}`",
            f"- Seed: `{DOMAIN_SEED}`",
            f"- Results root: `{portable_path(self.result_root)}`",
            f"- Condition interpretation: `{ENRICHED_CONDITION_LABEL}`",
            f"- Enrichment note: `{ENRICHED_CONDITION_NOTE}`",
            f"- Runtime interface note: `{BLACK_BOX_RUNTIME_NOTE}`",
            f"- Attacker prompt: `{portable_path(prompt_path)}`",
            f"- Attacker knowledge files: `{', '.join(portable_path(path) for path in knowledge_paths)}`",
            f"- Defender prompt: `{portable_path(defender_prompt) if defender_prompt else 'n/a'}`",
            f"- Defender knowledge files: `{', '.join(portable_path(path) for path in defender_knowledge_paths)}`",
            f"- Prompt audit root: `{portable_path(self.audit_root)}`",
            f"- Prompt and knowledge audit manifest: `{portable_path(self.audit_manifest_path)}`",
            f"- Prompt and knowledge bundle hash: `{audit_summary.get('bundle_hash')}`",
            f"- Detected scenarios: `{attack_metrics.get('defender_detected_attack_run_count')}`",
            f"- Defender-action scenarios: `{attack_metrics.get('defender_acted_attack_run_count')}`",
            f"- Safety-violation scenarios: `{attack_metrics.get('attacker_caused_safety_violation_run_count')}`",
            f"- Average detection time: `{attack_metrics.get('wall_clock_time_to_detection_from_attack_start_sec_avg')}`",
            f"- Attacker action count: `{attack_metrics.get('attacker_action_count_avg')}`",
            f"- Downtime burden: `{attack_metrics.get('downtime_timestep_count_avg')}`",
            "- Water-distribution only.",
            "- Seed 101 only.",
        ]
        seeded._write_text(self.result_root / "results_summary.md", "\n".join(summary_lines) + "\n")
        self._set_phase("aggregating", "completed")

    def write_repo_note(self) -> None:
        resume_state = _read_json(self.resume_state_path, {})
        prompt_path, knowledge_paths = self._attacker_prompt_and_knowledge_paths()
        _, condition_spec, _, _ = self._attacker_profile_bundle()
        role_profiles = self._condition_role_profiles(condition_spec)
        defender_prompt = _profile_prompt_path(role_profiles["defender"])
        defender_knowledge_paths = _profile_knowledge_paths(role_profiles["defender"])
        lines = [
            "# Domain Ablation Note",
            "",
            f"- Experiment version: `{DOMAIN_EXPERIMENT_VERSION}`",
            f"- Condition root: `{portable_path(self.result_root)}`",
            f"- Config path: `{portable_path(DOMAIN_CONFIG_PATH)}`",
            f"- Attacker profile: `{DOMAIN_FAMILY_SPEC.attacker_profile_id}`",
            "- Defender profile: `agentic_defender`",
            f"- Testbed: `{seeded.TESTBED_ID}`",
            f"- Seed: `{DOMAIN_SEED}`",
            f"- Condition interpretation: `{ENRICHED_CONDITION_LABEL}`",
            f"- Enrichment note: `{ENRICHED_CONDITION_NOTE}`",
            f"- Runtime interface note: `{BLACK_BOX_RUNTIME_NOTE}`",
            f"- Attacker prompt: `{portable_path(prompt_path)}`",
            f"- Attacker knowledge files: `{', '.join(portable_path(path) for path in knowledge_paths)}`",
            f"- Defender prompt: `{portable_path(defender_prompt) if defender_prompt else 'n/a'}`",
            f"- Defender knowledge files: `{', '.join(portable_path(path) for path in defender_knowledge_paths)}`",
            f"- Prompt audit root: `{portable_path(self.audit_root)}`",
            f"- Prompt and knowledge audit manifest: `{portable_path(self.audit_manifest_path)}`",
            f"- Resume event occurred: `{'true' if resume_state.get('resume_event_detected') else 'false'}`",
            f"- All {seeded.SCENARIO_COUNT} runs completed: `{'true' if resume_state.get('completed_runs') == seeded.TOTAL_RUNS else 'false'}`",
            "- Water-distribution only.",
            "- Water-treatment was not touched.",
            "- No paper files were touched.",
            "- No secrets or credentials were committed.",
        ]
        seeded._write_text(self.repo_note_path, "\n".join(lines) + "\n")

    def run_controller(self) -> int:
        self._apply_domain_globals()
        self.freeze_metadata()
        self._mark_resume_event_if_needed(
            [row for row in csv.DictReader(self.run_queue_path.open(newline=""))]
            if self.run_queue_path.exists()
            else []
        )
        self._validate_distribution_environment()
        self.write_prompt_knowledge_audit()
        self._validate_runtime_and_tests()
        self._set_phase("running_grid", "in_progress")
        seeded._append_log(
            self.experiment_log_path,
            f"controller starting domain ablation under {portable_path(self.result_root)}",
        )

        seed_status = _read_json(
            self._seed_status_path(DOMAIN_FAMILY_SPEC.family_name, DOMAIN_SEED),
            {},
        )
        if seed_status.get("completed_scenarios") != seeded.SCENARIO_COUNT:
            python_exec = _domain_python_executable()
            worker_log_path = (
                self.seed_progress_root(DOMAIN_FAMILY_SPEC.family_name, DOMAIN_SEED)
                / "worker_run.json"
            )
            command = [
                python_exec,
                "-m",
                "ics_agent.experiments.water_distribution.domain_ablation",
                "worker",
                "--family",
                DOMAIN_FAMILY_SPEC.family_name,
                "--seed",
                str(DOMAIN_SEED),
                "--result-root",
                str(self.result_root),
            ]
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(DOMAIN_SEED)
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
                    last_failure_family=DOMAIN_FAMILY_SPEC.family_name,
                    last_failure_seed=DOMAIN_SEED,
                )
                seeded._append_log(
                    self.experiment_log_path,
                    "worker failed for black_box_high_cyber_high_domain__agentic_defender "
                    f"seed {DOMAIN_SEED}: {completed.stderr.strip()}",
                )
                return completed.returncode
            seeded._append_log(
                self.experiment_log_path,
                f"worker completed for {DOMAIN_FAMILY_SPEC.family_name} seed {DOMAIN_SEED}",
            )

        self._sync_all_ledgers()
        resume_state = _read_json(self.resume_state_path, {})
        if resume_state.get("completed_runs") != seeded.TOTAL_RUNS:
            raise RuntimeError(
                f"expected {seeded.TOTAL_RUNS} completed runs, found {resume_state.get('completed_runs')}"
            )
        self._set_phase("running_grid", "completed", completed_runs=seeded.TOTAL_RUNS)
        self.write_family_aggregates()
        self.write_repo_note()
        return 0


def _read_json(path: Path, default: Any) -> Any:
    return seeded._read_json(path, default)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the single-family water-distribution domain-ablation pass."
        )
    )
    subparsers = parser.add_subparsers(dest="mode")

    controller = subparsers.add_parser("controller")
    controller.add_argument(
        "--result-root",
        default=str(DOMAIN_RESULT_ROOT),
        help="Domain-ablation result root.",
    )

    worker = subparsers.add_parser("worker")
    worker.add_argument(
        "--family",
        required=True,
        choices=[DOMAIN_FAMILY_SPEC.family_name],
    )
    worker.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[DOMAIN_SEED],
    )
    worker.add_argument(
        "--result-root",
        default=str(DOMAIN_RESULT_ROOT),
        help="Domain-ablation result root.",
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
        args.result_root = str(DOMAIN_RESULT_ROOT)
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runner = DomainAblationRunner(Path(args.result_root))
    if args.mode == "worker":
        return runner.run_worker(
            args.family,
            int(args.seed),
            max_attempts=int(args.max_attempts),
        )
    return runner.run_controller()


if __name__ == "__main__":
    raise SystemExit(main())
