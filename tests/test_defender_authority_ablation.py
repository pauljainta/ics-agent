from __future__ import annotations

import json
from pathlib import Path

from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.experiments.water_treatment.defender_authority_ablation import (
    AUTHORITY_CONFIG_PATHS,
    AUTHORITY_FAMILY_SPECS,
    AUTHORITY_RESULT_ROOT,
    AUTHORITY_SEED,
    DefenderAuthorityAblationRunner,
    _condition_metric_row,
)
from ics_agent.experiments.water_treatment import seeded_backbone as seeded


REPO_ROOT = Path(__file__).resolve().parents[1]


def _condition_by_name(
    orchestrator: ExperimentOrchestrator,
    config_path: Path,
    condition_name: str,
) -> dict[str, object]:
    config = load_config(str(config_path))
    for condition in orchestrator._enumerate_conditions(config):
        if condition.get("name") == condition_name:
            return condition
    raise AssertionError(f"missing condition {condition_name}")


def test_defender_authority_profiles_exist_with_expected_delay_controls() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")

    observe = orchestrator.agent_profiles["observe_alert_only_defender"]
    delayed = orchestrator.agent_profiles["delayed_containment_defender"]

    assert observe["allowed_action_types"] == ["noop"]
    assert delayed["runtime_containment_activation_delay_steps"] == 2


def test_authority_runner_resolves_all_eight_condition_pairs() -> None:
    runner = DefenderAuthorityAblationRunner(AUTHORITY_RESULT_ROOT)
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")

    assert len(AUTHORITY_FAMILY_SPECS) == 8
    for family in AUTHORITY_FAMILY_SPECS:
        config = runner._runtime_config(family, AUTHORITY_SEED)
        condition_spec = runner._condition_spec(orchestrator, family, config)
        assert condition_spec["name"] == family.family_name
        assert condition_spec["profiles"]["attacker"] == family.attacker_profile_id


def test_authority_runner_sync_ledgers_tracks_144_pending_runs(tmp_path: Path) -> None:
    runner = DefenderAuthorityAblationRunner(tmp_path / "defender-authority-ablation")
    runner.ensure_root_layout()
    runner._sync_all_ledgers()
    resume_state = runner.resume_state_path.read_text()
    assert '"completed_runs": 0' in resume_state
    assert '"pending_runs": 144' in resume_state


def test_authority_configs_preserve_existing_attacker_resolutions() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    baseline_paths = {
        "black_box_attacker": REPO_ROOT
        / "configs/experiments/water_treatment_agentic_black_box_attacker_vs_agentic_full_containment_defender.yaml",
        "gray_box_attacker": REPO_ROOT
        / "configs/experiments/water_treatment_agentic_gray_box_attacker_vs_agentic_full_containment_defender.yaml",
        "white_box_attacker": REPO_ROOT
        / "configs/experiments/water_treatment_agentic_white_box_attacker_vs_agentic_full_containment_defender.yaml",
        "black_box_high_cyber_high_domain_treatment_attacker": REPO_ROOT
        / "configs/experiments/water_treatment_agentic_black_box_high_cyber_high_domain_attacker_vs_agentic_full_containment_defender.yaml",
    }
    authority_conditions = {
        "black_box_attacker": (
            AUTHORITY_CONFIG_PATHS[0],
            "black_box_attacker__observe_alert_only_defender",
        ),
        "gray_box_attacker": (
            AUTHORITY_CONFIG_PATHS[1],
            "gray_box_attacker__observe_alert_only_defender",
        ),
        "white_box_attacker": (
            AUTHORITY_CONFIG_PATHS[2],
            "white_box_attacker__observe_alert_only_defender",
        ),
        "black_box_high_cyber_high_domain_treatment_attacker": (
            AUTHORITY_CONFIG_PATHS[3],
            "black_box_high_cyber_high_domain__observe_alert_only_defender",
        ),
    }
    baseline_conditions = {
        "black_box_attacker": "black_box_attacker__agentic_defender",
        "gray_box_attacker": "gray_box_attacker__agentic_defender",
        "white_box_attacker": "white_box_attacker__agentic_defender",
        "black_box_high_cyber_high_domain_treatment_attacker": (
            "black_box_high_cyber_high_domain__agentic_defender"
        ),
    }

    for attacker_profile_id, baseline_path in baseline_paths.items():
        baseline_condition = _condition_by_name(
            orchestrator,
            baseline_path,
            baseline_conditions[attacker_profile_id],
        )
        authority_condition = _condition_by_name(
            orchestrator,
            authority_conditions[attacker_profile_id][0],
            authority_conditions[attacker_profile_id][1],
        )
        baseline_attacker = baseline_condition["effective_profiles"]["attacker"]
        authority_attacker = authority_condition["effective_profiles"]["attacker"]
        assert authority_attacker == baseline_attacker


def test_existing_full_containment_defender_resolution_remains_unchanged() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    baseline = _condition_by_name(
        orchestrator,
        REPO_ROOT
        / "configs/experiments/water_treatment_agentic_black_box_attacker_vs_agentic_full_containment_defender.yaml",
        "black_box_attacker__agentic_defender",
    )
    defender = baseline["effective_profiles"]["defender"]

    assert baseline["profiles"]["defender"] == "agentic_defender"
    assert "block_sensor_write" in defender["allowed_action_types"]
    assert "quarantine_signal_source" in defender["allowed_action_types"]


def test_phase4_condition_summary_uses_effective_cutoff_timing(tmp_path: Path) -> None:
    run_dir = tmp_path / "delayed-run"
    run_dir.mkdir(parents=True)
    (run_dir / "actions.json").write_text(
        json.dumps(
            [
                {
                    "time_sec": 11.0,
                    "role": "defender",
                    "action": {"type": "block_sensor_write"},
                    "containment": {
                        "is_containment_action": True,
                        "execution_mode": "runtime_enforced",
                    },
                }
            ]
        )
    )
    seeded.dump_yaml(
        run_dir / "config_snapshot.yaml",
        {"phase_boundary": {"wargame_t0_elapsed_sec": 10.0}},
    )
    condition_summary = {
        "expected_run_count": 18,
        "completed_run_count": 1,
        "attack_metrics": {
            "defender_detected_attack_run_count": 1,
            "defender_acted_attack_run_count": 1,
            "attacker_caused_safety_violation_run_count": 1,
            "time_to_detection_from_wargame_t0_sec_avg": 1.0,
            "time_to_first_defender_action_from_wargame_t0_sec_avg": 1.0,
            "attacker_action_count_avg": 3.0,
            "defender_action_count_avg": 1.0,
            "downtime_timestep_count_avg": 0.0,
        },
        "runs": [
            {
                "suite": "attack",
                "status": "completed",
                "run_dir": str(run_dir),
                "metrics": {
                    "safety_violation_count": 1,
                    "phase_window": {
                        "time_to_first_safety_violation_sec": 2.0,
                    },
                    "runtime_containment": {
                        "first_containment_action_type": "block_sensor_write",
                        "first_containment_time_from_wargame_t0_sec": 1.0,
                        "attacker_execution_cutoff_activated": True,
                        "attacker_execution_cutoff_activation_from_wargame_t0_sec": 3.0,
                    },
                },
            }
        ],
    }

    row = _condition_metric_row(
        AUTHORITY_FAMILY_SPECS[1],
        condition_summary,
    )

    assert row["contained_before_safety_violation_count"] == 0
    assert row["safety_violation_before_cutoff_run_count"] == 1
