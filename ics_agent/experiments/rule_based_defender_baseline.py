from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..paths import REPO_ROOT, WORKSPACE_ROOT, portable_path, profile_registry_paths
from ..reporting import write_json
from .water_distribution.scripted_black_box_baseline import (
    ScriptedBlackBoxBaselineRunner as DistributionBaselineRunner,
)
from .water_distribution.seeded_backbone import FamilySpec as DistributionFamilySpec
from .water_treatment.scripted_black_box_baseline import (
    ScriptedBlackBoxBaselineRunner as TreatmentBaselineRunner,
)
from .water_treatment.seeded_backbone import FamilySpec as TreatmentFamilySpec


SEED = 101

ATTACKER_SPECS = {
    "black_box": {
        "profile_id": "black_box_attacker",
        "label": "Black-box agentic attacker",
    },
    "gray_box": {
        "profile_id": "gray_box_attacker",
        "label": "Gray-box agentic attacker",
    },
    "white_box": {
        "profile_id": "white_box_attacker",
        "label": "White-box agentic attacker",
    },
}

DEFENDER_MODE_SPECS = {
    "active_intervention": {
        "profile_id": "rule_based_active_intervention_defender",
        "formal_name": "Rule-based defender, active intervention",
        "result_token": "active_intervention",
    },
    "advisory_response": {
        "profile_id": "rule_based_advisory_response_defender",
        "formal_name": "Rule-based defender, advisory response",
        "result_token": "advisory_response",
    },
    "latency_constrained_intervention": {
        "profile_id": "rule_based_latency_constrained_intervention_defender",
        "formal_name": "Rule-based defender, latency-constrained intervention",
        "result_token": "latency_constrained_intervention",
    },
}

TREATMENT_CONFIGS = {
    "black_box": REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_black_box_attacker_vs_agentic_full_containment_defender.yaml",
    "gray_box": REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_gray_box_attacker_vs_agentic_full_containment_defender.yaml",
    "white_box": REPO_ROOT
    / "configs"
    / "experiments"
    / "water_treatment_agentic_white_box_attacker_vs_agentic_full_containment_defender.yaml",
}

DISTRIBUTION_CONFIGS = {
    "black_box": REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_black_box_vs_agentic_defender.yaml",
    "gray_box": REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_gray_box_vs_agentic_defender.yaml",
    "white_box": REPO_ROOT
    / "configs"
    / "experiments"
    / "water_distribution_white_box_vs_agentic_defender.yaml",
}


def _allowed_defender_actions(mode: str) -> list[str]:
    if mode == "advisory_response":
        return ["noop"]
    return [
        "hmi_stop",
        "hmi_estop",
        "select_process",
        "disable_hmi_session",
        "isolate_engineering_host",
        "block_sensor_write",
        "block_actuator_write",
        "block_process2_manual_path",
        "quarantine_signal_source",
        "noop",
    ]


def _condition_name(attacker: str, mode: str) -> str:
    return f"{attacker}_attacker__rule_based_{mode}_defender"


def _result_root(testbed: str, attacker: str, mode: str) -> Path:
    base = WORKSPACE_ROOT / "Artifact" / "Results"
    if testbed == "water_treatment":
        base = base / "Water-treatment" / "scripted_baselines"
    else:
        base = base / "water_distribution_openplc" / "scripted_baselines"
    return base / f"{attacker}_agentic_attacker_vs_rule_based_defender_{mode}"


def _parse_result_root_tokens(result_root: Path) -> tuple[str, str]:
    name = result_root.name
    for attacker in ATTACKER_SPECS:
        prefix = f"{attacker}_agentic_attacker_vs_rule_based_defender_"
        if not name.startswith(prefix):
            continue
        mode = name.removeprefix(prefix)
        if mode in DEFENDER_MODE_SPECS:
            return attacker, mode
    raise ValueError(
        "could not infer attacker/defender mode from result root name "
        f"`{result_root}`"
    )


def _resume_compatible_hash_paths() -> set[str]:
    return {
        *[
            portable_path(path)
            for testbed in ("water_treatment", "water_distribution")
            for path in profile_registry_paths(testbed)
        ],
        portable_path(REPO_ROOT / "ics_agent" / "agents" / "rule_based.py"),
        portable_path(REPO_ROOT / "ics_agent" / "experiments" / "rule_based_defender_baseline.py"),
    }


class RuleBasedDefenderRunnerMixin:
    attacker_key: str
    defender_mode: str
    testbed_label: str
    mode_spec: dict[str, Any]
    attacker_spec: dict[str, Any]

    def _runtime_config(self) -> dict[str, Any]:
        config = super()._runtime_config()
        condition = config["conditions"][0]
        condition["name"] = self.family_spec.family_name
        condition["profiles"]["defender"] = self.mode_spec["profile_id"]
        condition["profiles"]["observer"] = "scripted_observer"
        overrides = deepcopy(condition.get("profile_overrides") or {})
        overrides["defender"] = {
            "allowed_action_types": _allowed_defender_actions(self.defender_mode),
            "behavior_model": {
                "id": "optionb_mitre_staged",
                "policy": "rule_based_defender",
                "response_mode": self.defender_mode,
                "notes": self.mode_spec["formal_name"],
            },
        }
        if self.defender_mode == "latency_constrained_intervention":
            overrides["defender"]["runtime_containment_activation_delay_steps"] = 2
        overrides.pop("observer", None)
        condition["profile_overrides"] = overrides
        config["reproducibility"]["seed"] = SEED
        return config

    def _execution_registry_payload(self) -> dict[str, Any]:
        orchestrator = self._orchestrator()
        config = self._runtime_config()
        condition_spec = self._condition_spec(orchestrator, config)
        effective_profiles = condition_spec.get("effective_profiles") or {}
        attacker_profile = dict(effective_profiles.get("attacker") or {})
        defender_profile = dict(effective_profiles.get("defender") or {})
        return {
            "experiment_version": self.experiment_version,
            "result_root": portable_path(self.result_root),
            "seed_root": portable_path(self.seed_root),
            "testbed": self.testbed_label,
            "seed": self.seed,
            "condition_name": self.family_spec.family_name,
            "config_path": portable_path(self.config_path),
            "attacker": {
                "profile_id": self.family_spec.attacker_profile_id,
                "family": attacker_profile.get("family"),
                "knowledge_mode": attacker_profile.get("knowledge_mode"),
                "behavior_model": attacker_profile.get("behavior_model"),
                "allowed_action_types": list(attacker_profile.get("allowed_action_types", [])),
                "runtime_policy": self.runtime_policy,
                "llm_invocation_required": True,
            },
            "defender": {
                "profile_id": self.mode_spec["profile_id"],
                "family": defender_profile.get("family"),
                "response_mode": self.defender_mode,
                "allowed_action_types": list(defender_profile.get("allowed_action_types", [])),
                "llm_invocation_required": False,
            },
            "observer_profile": condition_spec["profiles"].get("observer"),
            "planner_profile": condition_spec["profiles"].get("planner"),
            "operator_profile": condition_spec["profiles"].get("operator"),
            "offline_analysis_profile": condition_spec["profiles"].get("offline_analysis"),
        }

    def _completed_row(self, **kwargs: Any) -> dict[str, Any]:
        row = super()._completed_row(**kwargs)
        row["defender_profile_id"] = self.mode_spec["profile_id"]
        row["defender_response_mode"] = self.defender_mode
        row["attacker_runtime_policy"] = self.runtime_policy
        return row

    def _config_hash_tracked_paths(self) -> list[Path]:
        paths = list(super()._config_hash_tracked_paths())
        paths.extend(
            [
                REPO_ROOT / "ics_agent" / "agents" / "rule_based.py",
                REPO_ROOT / "ics_agent" / "experiments" / "rule_based_defender_baseline.py",
            ]
        )
        return paths

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
            _resume_compatible_hash_paths()
        )

    def _run_id(self, scenario: Any, split: str) -> str:
        return "_".join(
            [
                self.testbed_label,
                scenario.scenario_id.replace("-", "_"),
                split,
                self.attacker_key,
                "ruleBased",
                self.defender_mode,
                f"seed{self.seed}",
            ]
        )

    def _report_lines(self) -> list[str]:
        condition_summary = self._read_condition_summary()
        attack_metrics = dict(condition_summary.get("attack_metrics") or {})
        aggregate_row = self._read_aggregate()
        return [
            "# Rule-Based Defender Baseline Report",
            "",
            f"- Testbed: `{self.testbed_label}`",
            f"- Attacker: `{self.attacker_spec['label']}`",
            f"- Defender: `{self.mode_spec['formal_name']}`",
            f"- Seed: `{self.seed}`",
            f"- Results root: `{portable_path(self.result_root)}`",
            "",
            "## Outcome",
            f"- Completed scenarios: `{condition_summary.get('completed_run_count')}/{condition_summary.get('expected_run_count')}`",
            f"- Detected scenarios: `{attack_metrics.get('defender_detected_attack_run_count')}/{self.runs_per_seed}`",
            f"- Defender-action scenarios: `{attack_metrics.get('defender_acted_attack_run_count')}/{self.runs_per_seed}`",
            f"- Safety-violation scenarios: `{attack_metrics.get('attacker_caused_safety_violation_run_count')}/{self.runs_per_seed}`",
            f"- Average wall-clock detection from attack start: `{aggregate_row.get('mean_wall_clock_detection_from_attack_start_sec')}`",
            f"- Average attacker action count: `{aggregate_row.get('mean_attacker_action_count')}`",
            f"- Average downtime burden: `{aggregate_row.get('mean_downtime_timestep_count')}`",
        ]

    def _read_condition_summary(self) -> dict[str, Any]:
        if not self.condition_summary_path.exists():
            return {}
        return json.loads(self.condition_summary_path.read_text())

    def _read_aggregate(self) -> dict[str, Any]:
        if not self.aggregate_json_path.exists():
            return {}
        return json.loads(self.aggregate_json_path.read_text())

    def write_repo_local_report(self) -> None:
        if getattr(self, "repo_local_report_path", None) is None:
            return
        return super().write_repo_local_report()

    def smoke(self) -> dict[str, Any]:
        self.freeze_metadata()
        orchestrator = self._orchestrator()
        config = self._runtime_config()
        condition_spec = self._condition_spec(orchestrator, config)
        profiles = condition_spec.get("effective_profiles") or {}
        attacker_profile = dict(profiles.get("attacker") or {})
        defender_profile = dict(profiles.get("defender") or {})
        scenarios = self.scenarios()
        errors: list[str] = []
        if len(scenarios) != self.runs_per_seed:
            errors.append(f"expected {self.runs_per_seed} scenarios, found {len(scenarios)}")
        if attacker_profile.get("family") != "agentic":
            errors.append("attacker profile is not agentic")
        if defender_profile.get("family") != "rule_based":
            errors.append("defender profile is not rule_based")
        behavior = defender_profile.get("behavior_model") or {}
        if behavior.get("response_mode") != self.defender_mode:
            errors.append("defender response mode did not resolve correctly")
        payload = {
            "status": "failed" if errors else "passed",
            "errors": errors,
            "testbed": self.testbed_label,
            "attacker": self.attacker_key,
            "defender_mode": self.defender_mode,
            "condition_name": self.family_spec.family_name,
            "scenario_count": len(scenarios),
            "result_root": portable_path(self.result_root),
            "seed": self.seed,
        }
        write_json(self.progress_root / "rule_based_defender_smoke.json", payload)
        if errors:
            raise RuntimeError("; ".join(errors))
        return payload


class TreatmentRuleBasedDefenderRunner(
    RuleBasedDefenderRunnerMixin,
    TreatmentBaselineRunner,
):
    def __init__(
        self,
        result_root: Path,
        attacker_key: str | None = None,
        defender_mode: str | None = None,
    ) -> None:
        if attacker_key is None or defender_mode is None:
            attacker_key, defender_mode = _parse_result_root_tokens(result_root)
        self.attacker_key = attacker_key
        self.defender_mode = defender_mode
        self.testbed_label = "water_treatment"
        self.mode_spec = DEFENDER_MODE_SPECS[defender_mode]
        self.attacker_spec = ATTACKER_SPECS[attacker_key]
        self.experiment_version = (
            f"water_treatment_{attacker_key}_vs_rule_based_{defender_mode}_seed101_v1"
        )
        self.config_path = TREATMENT_CONFIGS[attacker_key]
        self.family_spec = TreatmentFamilySpec(
            family_name=_condition_name(attacker_key, defender_mode),
            attacker_profile_id=self.attacker_spec["profile_id"],
            knowledge_variant=f"{attacker_key}_agentic",
            config_path=self.config_path,
        )
        self.runtime_policy = f"agentic_{attacker_key}"
        self.run_id_policy_token = f"{attacker_key}_ruleBased_{defender_mode}"
        self.controller_run_label = (
            f"{self.attacker_spec['label']} vs {self.mode_spec['formal_name']}"
        )
        self.report_filename = "rule_based_defender_baseline_report.md"
        self.repo_note_filename = "rule_based_defender_baseline_note.md"
        self.repo_local_report_path = None
        super().__init__(result_root)


class DistributionRuleBasedDefenderRunner(
    RuleBasedDefenderRunnerMixin,
    DistributionBaselineRunner,
):
    def __init__(
        self,
        result_root: Path,
        attacker_key: str | None = None,
        defender_mode: str | None = None,
    ) -> None:
        if attacker_key is None or defender_mode is None:
            attacker_key, defender_mode = _parse_result_root_tokens(result_root)
        self.attacker_key = attacker_key
        self.defender_mode = defender_mode
        self.testbed_label = "water_distribution"
        self.mode_spec = DEFENDER_MODE_SPECS[defender_mode]
        self.attacker_spec = ATTACKER_SPECS[attacker_key]
        self.experiment_version = (
            f"water_distribution_{attacker_key}_vs_rule_based_{defender_mode}_seed101_v1"
        )
        self.config_path = DISTRIBUTION_CONFIGS[attacker_key]
        self.family_spec = DistributionFamilySpec(
            family_name=_condition_name(attacker_key, defender_mode),
            attacker_profile_id=self.attacker_spec["profile_id"],
            knowledge_variant=f"{attacker_key}_agentic",
            config_path=self.config_path,
        )
        self.runtime_policy = f"agentic_{attacker_key}"
        self.run_id_policy_token = f"{attacker_key}_ruleBased_{defender_mode}"
        self.controller_run_label = (
            f"{self.attacker_spec['label']} vs {self.mode_spec['formal_name']}"
        )
        self.report_filename = "rule_based_defender_baseline_report.md"
        self.repo_note_filename = "rule_based_defender_baseline_note.md"
        self.default_repo_local_report_path = None
        super().__init__(result_root)
        self.repo_local_report_path = None


def build_runner(testbed: str, attacker: str, defender_mode: str):
    result_root = _result_root(testbed, attacker, defender_mode)
    if testbed == "water_treatment":
        return TreatmentRuleBasedDefenderRunner(
            result_root,
            attacker_key=attacker,
            defender_mode=defender_mode,
        )
    if testbed == "water_distribution":
        return DistributionRuleBasedDefenderRunner(
            result_root,
            attacker_key=attacker,
            defender_mode=defender_mode,
        )
    raise ValueError(f"unknown testbed: {testbed}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic rule-based defenders against agentic attackers."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name in ("smoke", "controller", "worker"):
        sub = subparsers.add_parser(name)
        sub.add_argument(
            "--testbed",
            choices=["water_treatment", "water_distribution"],
            required=name != "smoke",
        )
        sub.add_argument(
            "--attacker",
            choices=sorted(ATTACKER_SPECS),
            required=name != "smoke",
        )
        sub.add_argument(
            "--defender-mode",
            choices=sorted(DEFENDER_MODE_SPECS),
            required=name != "smoke",
        )
        sub.add_argument(
            "--all",
            action="store_true",
            help="Smoke-test all testbed/attacker/defender-mode combinations.",
        )
        if name == "worker":
            sub.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args(argv)


def _selected_jobs(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if args.mode == "smoke" and args.all:
        return [
            (testbed, attacker, defender_mode)
            for testbed in ("water_treatment", "water_distribution")
            for attacker in ("black_box", "gray_box", "white_box")
            for defender_mode in (
                "active_intervention",
                "advisory_response",
                "latency_constrained_intervention",
            )
        ]
    if not args.testbed or not args.attacker or not args.defender_mode:
        raise SystemExit("--testbed, --attacker, and --defender-mode are required")
    return [(args.testbed, args.attacker, args.defender_mode)]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    outputs: list[dict[str, Any]] = []
    for testbed, attacker, defender_mode in _selected_jobs(args):
        runner = build_runner(testbed, attacker, defender_mode)
        if args.mode == "smoke":
            outputs.append(runner.smoke())
        elif args.mode == "worker":
            rc = runner.run_worker(max_attempts=int(args.max_attempts))
            if rc != 0:
                return rc
        else:
            rc = runner.run_controller()
            if rc != 0:
                return rc
    if outputs:
        print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
