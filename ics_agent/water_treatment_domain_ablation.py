from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .agents.agentic import AgenticRoleAgent
from .config import load_config, require_fields, resolve_repo_path
from .orchestrator import ExperimentOrchestrator
from .paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from .reporting import write_json
from .scenarios import ScenarioCatalog
from .testbeds.registry import observation_focus_fields, trend_fields
from .utils import dump_yaml, utc_now
from . import water_treatment_seeded_backbone as seeded


DOMAIN_EXPERIMENT_VERSION = (
    "water_treatment_domain_ablation_black_box_high_cyber_high_domain_seed101_v1"
)
DOMAIN_SEED = 101
DOMAIN_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "sphere_optionb_mitre_black_box_high_cyber_high_domain_vs_agentic_defender_fast_containment.yaml"
)
DOMAIN_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "Artifact"
    / "Results"
    / "Water-treatment"
    / "domain-ablation"
    / "black_box_high_cyber_high_domain__agentic_defender"
)
DOMAIN_FAMILY_SPEC = seeded.FamilySpec(
    family_name="black_box_high_cyber_high_domain__agentic_defender",
    attacker_profile_id="black_box_high_cyber_high_domain_treatment_attacker",
    knowledge_variant="black_box_high_cyber_high_domain",
    config_path=DOMAIN_CONFIG_PATH,
)
DOMAIN_PROMPT_PATHS = [
    REPO_ROOT / "configs" / "prompts" / "black_box_attacker_optionb_mitre.md",
    REPO_ROOT / "configs" / "prompts" / "defender_optionb_mitre.md",
    REPO_ROOT / "configs" / "prompts" / "observer_optionb_mitre.md",
]
FORBIDDEN_LEAKAGE_PHRASES = (
    "best leverage surface",
    "highest leverage",
    "preferred attack surface",
    "surface ranking",
    "surface_leverage_hints",
)
DEFAULT_VALIDATION_TESTS = [
    "tests/test_phase35_profiles.py",
    "tests/test_seeded_backbone.py",
    "tests/test_water_treatment_domain_ablation.py",
]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_prompt_path(path_text: str | os.PathLike[str]) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _forbidden_treatment_tokens() -> set[str]:
    catalog = ScenarioCatalog(testbed_id=seeded.TESTBED_ID)
    tokens = set(observation_focus_fields(seeded.TESTBED_ID))
    tokens.update(trend_fields(seeded.TESTBED_ID))
    tokens.update({"WT-P1", "WT-P2", "WT-P3"})
    tokens.update(
        scenario.catalog_id.split("/")[-1]
        for scenario in catalog.resolve_selectors(["attack/*"])
    )
    return tokens


def _audit_treatment_observation(**overrides: float | str) -> dict[str, float | str]:
    observation: dict[str, float | str] = {
        "STATE_NAME": "RUNNING",
        "RW_Tank_Level": 640.0,
        "UF_UFFT_Tank_Level": 480.0,
        "ChemTreat_NaCl_Level": 780.0,
        "ChemTreat_NaOCl_Level": 760.0,
        "ChemTreat_HCl_Level": 750.0,
        "RW_Pump_Flow": 118.0,
        "RW_Pump_Speed_Cmd": 1.0,
        "RW_Tank_PR_Valve_Cmd": 1.0,
        "RW_Tank_P_Valve_Cmd": 0.0,
        "RW_Tank_PR_Valve_Sts": 1.0,
        "RW_Tank_P_Valve_Sts": 0.0,
        "RW_Pump_Sts": 1.0,
        "ChemTreat_NaCl_Valve_Cmd": 0.0,
        "ChemTreat_NaOCl_Valve_Cmd": 0.0,
        "ChemTreat_HCl_Valve_Cmd": 0.0,
        "ChemTreat_NaCl_Valve_Sts": 0.0,
        "ChemTreat_NaOCl_Valve_Sts": 0.0,
        "ChemTreat_HCl_Valve_Sts": 0.0,
        "SYS_IDLE": 0.0,
        "SYS_START": 0.0,
        "SYS_RUNNING": 1.0,
        "SYS_SHUTDOWN": 0.0,
        "P2_Selected": 0.0,
        "P2_NaCl_Cmd": 0.0,
        "P2_NaOCl_Cmd": 0.0,
        "P2_HCl_Cmd": 0.0,
        "Alarm_RW_Tank_LL": 0.0,
        "Alarm_RW_Tank_L": 0.0,
        "Alarm_RW_Tank_H": 0.0,
        "Alarm_RW_Tank_HH": 0.0,
        "HMI_Start_Active": 0.0,
        "HMI_Stop_Active": 0.0,
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
        self.validation_tests = list(DEFAULT_VALIDATION_TESTS)

    @staticmethod
    def _apply_domain_globals() -> None:
        seeded.EXPERIMENT_VERSION = DOMAIN_EXPERIMENT_VERSION
        seeded.SEEDS = [DOMAIN_SEED]
        seeded.FROZEN_CONFIG_PATHS = [DOMAIN_CONFIG_PATH]
        seeded.FAMILY_SPECS = [DOMAIN_FAMILY_SPEC]
        seeded.FAMILY_BY_NAME = {DOMAIN_FAMILY_SPEC.family_name: DOMAIN_FAMILY_SPEC}
        seeded.FAMILY_ORDER = {DOMAIN_FAMILY_SPEC.family_name: 0}

    def _model_registry_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        config = self._runtime_config(DOMAIN_FAMILY_SPEC, DOMAIN_SEED)
        condition_spec = self._condition_spec(orchestrator, DOMAIN_FAMILY_SPEC, config)
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
            raise RuntimeError("unable to resolve model metadata for domain ablation runner")
        return {
            "experiment_version": DOMAIN_EXPERIMENT_VERSION,
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
                    "source_profiles": [
                        DOMAIN_FAMILY_SPEC.attacker_profile_id,
                        "agentic_defender",
                        "agentic_observer",
                    ],
                }
            ],
        }

    def _prompt_hash_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        config = self._runtime_config(DOMAIN_FAMILY_SPEC, DOMAIN_SEED)
        condition_spec = self._condition_spec(orchestrator, DOMAIN_FAMILY_SPEC, config)
        effective_profiles = condition_spec.get("effective_profiles") or {}
        attacker = dict(
            effective_profiles.get("attacker")
            or orchestrator.agent_profiles[DOMAIN_FAMILY_SPEC.attacker_profile_id]
        )
        defender = dict(
            effective_profiles.get("defender")
            or orchestrator.agent_profiles["agentic_defender"]
        )
        observer = dict(
            effective_profiles.get("observer")
            or orchestrator.agent_profiles["agentic_observer"]
        )
        prompt_paths = {
            "attacker_prompt": _resolve_prompt_path(attacker["llm"]["prompt_file"]),
            "defender_prompt": _resolve_prompt_path(defender["llm"]["prompt_file"]),
            "observer_prompt": _resolve_prompt_path(observer["llm"]["prompt_file"]),
        }
        knowledge_paths = [
            resolve_repo_path(path, base_dir=REPO_ROOT)
            for path in attacker.get("knowledge_files", [])
        ]
        agent_profiles_path = REPO_ROOT / "configs" / "agent_profiles.yaml"
        profiles_overrides_path = REPO_ROOT / "configs" / "agent_profiles_overrides.yaml"
        hashed_paths = [
            *prompt_paths.values(),
            *knowledge_paths,
            agent_profiles_path,
            profiles_overrides_path,
            DOMAIN_CONFIG_PATH,
        ]
        file_hashes = {
            portable_path(path): seeded._sha256_path(path)
            for path in hashed_paths
            if path.exists()
        }
        knowledge_bundle = {
            portable_path(path): file_hashes[portable_path(path)]
            for path in knowledge_paths
        }
        bundle = {
            "attacker_prompt": file_hashes[portable_path(prompt_paths["attacker_prompt"])],
            "defender_prompt": file_hashes[portable_path(prompt_paths["defender_prompt"])],
            "observer_prompt": file_hashes[portable_path(prompt_paths["observer_prompt"])],
            "attacker_knowledge_files": knowledge_bundle,
            "agent_profiles": file_hashes[portable_path(agent_profiles_path)],
            "agent_profiles_overrides": file_hashes[portable_path(profiles_overrides_path)],
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

    def _sync_all_ledgers(self) -> None:
        family_name = DOMAIN_FAMILY_SPEC.family_name
        seed = DOMAIN_SEED
        rows = self._load_scenario_rows(family_name, seed)
        completed_payload = dict(_read_json(self._seed_completed_runs_path(family_name, seed), {}))
        failed_payload = dict(_read_json(self._seed_failed_runs_path(family_name, seed), {}))
        all_completed: dict[str, Any] = {}
        all_failed: dict[str, Any] = {}
        queue_rows: list[dict[str, Any]] = []

        for row in rows:
            scenario_key = f"{family_name}::{seed}::{row['scenario_id']}"
            queue_rows.append(
                {
                    "attacker_family": family_name,
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
            elif row["status"] == "failed":
                record = dict(failed_payload.get(row["scenario_id"]) or {})
                if record:
                    all_failed[scenario_key] = record

        write_json(self._family_completed_runs_path(family_name), all_completed)
        write_json(self._family_failed_runs_path(family_name), all_failed)
        family_completed_count = len(all_completed)
        family_failed_count = len(all_failed)
        family_status = {
            "updated_utc": utc_now(),
            "experiment_version": DOMAIN_EXPERIMENT_VERSION,
            "family_name": family_name,
            "completed_runs": family_completed_count,
            "failed_runs": family_failed_count,
            "pending_runs": 18 - family_completed_count - family_failed_count,
            "status": "completed" if family_completed_count == 18 else "in_progress",
            "seed_status": {
                DOMAIN_SEED: _read_json(self._seed_status_path(family_name, seed), {})
            },
        }
        write_json(self._family_status_path(family_name), family_status)

        write_json(self.completed_runs_path, all_completed)
        write_json(self.failed_runs_path, all_failed)
        seeded.write_csv(self.run_queue_path, queue_rows)
        catalog_rows = [
            all_completed[key]
            for key in sorted(
                all_completed,
                key=lambda item: next(
                    index
                    for index, manifest_row in enumerate(self.manifest_rows())
                    if manifest_row["scenario_id"] == item.split("::")[2]
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
            "experiment_version": DOMAIN_EXPERIMENT_VERSION,
            "completed_runs": len(catalog_rows),
            "failed_runs": len(failure_rows),
            "pending_runs": len(pending_rows),
            "all_complete": len(catalog_rows) == 18,
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
        seeded._write_text(self.resume_notes_path, "\n".join(lines) + "\n")

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

    def write_prompt_knowledge_audit(self) -> None:
        self.audit_root.mkdir(parents=True, exist_ok=True)
        self.audit_packages_root.mkdir(parents=True, exist_ok=True)
        config, condition_spec, _, attacker_profile = self._attacker_profile_bundle()
        prompt_path = _resolve_prompt_path(attacker_profile["llm"]["prompt_file"])
        knowledge_paths = [
            resolve_repo_path(path, base_dir=REPO_ROOT)
            for path in attacker_profile.get("knowledge_files", [])
        ]
        file_hashes = {
            portable_path(path): seeded._sha256_path(path)
            for path in [prompt_path, *knowledge_paths, DOMAIN_CONFIG_PATH]
        }
        prompt_text = prompt_path.read_text()
        knowledge_text = "\n\n".join(path.read_text() for path in knowledge_paths)
        static_findings = _scan_text_for_leakage(
            prompt_text + "\n\n" + knowledge_text,
            tokens=_forbidden_treatment_tokens(),
        )
        if re.search(r"\b\d{3,}\b", knowledge_text):
            static_findings.append("knowledge bundle contains 3+ digit numeric token(s)")
        package_rows: list[dict[str, Any]] = []
        all_findings = list(static_findings)
        for scenario in self.scenarios():
            observation = _audit_treatment_observation()
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
                    tokens=_forbidden_treatment_tokens(),
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
            all_findings.extend(
                f"{scenario.scenario_id}: {item}" for item in findings
            )

        summary = {
            "generated_at_utc": utc_now(),
            "experiment_version": DOMAIN_EXPERIMENT_VERSION,
            "result_root": portable_path(self.result_root),
            "condition_name": DOMAIN_FAMILY_SPEC.family_name,
            "attacker_profile": DOMAIN_FAMILY_SPEC.attacker_profile_id,
            "knowledge_mode": attacker_profile.get("knowledge_mode"),
            "prompt_path": portable_path(prompt_path),
            "knowledge_paths": [portable_path(path) for path in knowledge_paths],
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
            f"- Prompt path: `{portable_path(prompt_path)}`",
            f"- Knowledge files: `{', '.join(portable_path(path) for path in knowledge_paths)}`",
            f"- Package count: `{len(package_rows)}`",
            f"- Leakage findings: `{len(all_findings)}`",
        ]
        seeded._write_text(self.audit_report_path, "\n".join(report_lines) + "\n")
        if all_findings:
            raise RuntimeError(
                "prompt/knowledge audit failed: " + "; ".join(all_findings[:8])
            )

    def write_family_aggregates(self) -> None:
        self._set_phase("aggregating", "in_progress")
        seed_root = self.seed_root(DOMAIN_FAMILY_SPEC.family_name, DOMAIN_SEED)
        family_root = self.family_root(DOMAIN_FAMILY_SPEC.family_name)
        seed_summary = _read_json(seed_root / "seed_summary.json", {})
        if not seed_summary:
            raise RuntimeError("seed summary missing for domain ablation run")
        attack_metrics = _read_json(seed_root / "attack_metrics_summary.json", {})
        trajectory_metrics = _read_json(seed_root / "trajectory_metrics_summary.json", {})
        mitre_phase_metrics = _read_json(seed_root / "mitre_phase_metrics.json", {})
        nominal_metrics = _read_json(seed_root / "nominal_metrics_summary.json", {})
        condition_summary = _read_json(seed_root / "condition_summary.json", {})
        aggregate_row = {
            "family_name": DOMAIN_FAMILY_SPEC.family_name,
            "seed": DOMAIN_SEED,
            "total_runs": 18,
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
                    f"- Seed: `{DOMAIN_SEED}`",
                    f"- Completed runs: `{seed_summary.get('completed_scenarios')}/18`",
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
        write_json(self.result_root / "mitre_phase_metrics.json", mitre_phase_metrics)
        write_json(self.result_root / "nominal_metrics_summary.json", nominal_metrics)
        summary_lines = [
            "# Results Summary",
            "",
            f"- Generated UTC: `{utc_now()}`",
            f"- Attacker profile: `{DOMAIN_FAMILY_SPEC.attacker_profile_id}`",
            f"- Defender profile: `agentic_defender`",
            f"- Testbed: `{seeded.TESTBED_ID}`",
            f"- Seed: `{DOMAIN_SEED}`",
            f"- Results root: `{portable_path(self.result_root)}`",
            "- Target-blind black-box regime preserved: `true`",
            f"- Prompt audit root: `{portable_path(self.audit_root)}`",
            f"- Detected scenarios: `{attack_metrics.get('defender_detected_attack_run_count')}`",
            f"- Defender-action scenarios: `{attack_metrics.get('defender_acted_attack_run_count')}`",
            f"- Safety-violation scenarios: `{attack_metrics.get('attacker_caused_safety_violation_run_count')}`",
            f"- Average detection time: `{attack_metrics.get('wall_clock_time_to_detection_from_attack_start_sec_avg')}`",
            f"- Average attacker action count: `{attack_metrics.get('attacker_action_count_avg')}`",
            f"- Downtime burden: `{attack_metrics.get('downtime_timestep_count_avg')}`",
        ]
        seeded._write_text(self.result_root / "results_summary.md", "\n".join(summary_lines) + "\n")
        self._set_phase("aggregating", "completed")

    def write_repo_note(self) -> None:
        resume_state = _read_json(self.resume_state_path, {})
        lines = [
            "# Domain Ablation Note",
            "",
            f"- Experiment version: `{DOMAIN_EXPERIMENT_VERSION}`",
            f"- Condition root: `{portable_path(self.result_root)}`",
            f"- Config path: `{portable_path(DOMAIN_CONFIG_PATH)}`",
            f"- Prompt audit root: `{portable_path(self.audit_root)}`",
            f"- Final seed: `{DOMAIN_SEED}`",
            f"- Resume event occurred: `{'true' if resume_state.get('resume_event_detected') else 'false'}`",
            f"- All 18 runs completed: `{'true' if resume_state.get('completed_runs') == 18 else 'false'}`",
            "- Water-treatment only.",
            "- Water-distribution was not touched.",
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
        self._validate_optionb_environment()
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
        if seed_status.get("completed_scenarios") != 18:
            python_exec = seeded._python_executable()
            worker_log_path = (
                self.seed_progress_root(DOMAIN_FAMILY_SPEC.family_name, DOMAIN_SEED)
                / "worker_run.json"
            )
            command = [
                python_exec,
                "-m",
                "ics_agent.water_treatment_domain_ablation",
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
        if resume_state.get("completed_runs") != 18:
            raise RuntimeError(
                f"expected 18 completed runs, found {resume_state.get('completed_runs')}"
            )
        self._set_phase("running_grid", "completed", completed_runs=18)
        self.write_family_aggregates()
        self.write_repo_note()
        return 0


def _read_json(path: Path, default: Any) -> Any:
    return seeded._read_json(path, default)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the single-family water-treatment domain-ablation pass."
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
