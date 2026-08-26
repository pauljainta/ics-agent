from __future__ import annotations

from pathlib import Path

from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.water_distribution_defender_authority_ablation import (
    AUTHORITY_CONFIG_PATHS,
    AUTHORITY_FAMILY_SPECS,
    AUTHORITY_RESULT_ROOT,
    AUTHORITY_SEED,
    DefenderAuthorityAblationRunner,
)


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


def test_distribution_authority_runner_resolves_all_eight_condition_pairs() -> None:
    runner = DefenderAuthorityAblationRunner(AUTHORITY_RESULT_ROOT)
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")

    assert len(AUTHORITY_FAMILY_SPECS) == 8
    for family in AUTHORITY_FAMILY_SPECS:
        config = runner._runtime_config(family, AUTHORITY_SEED)
        condition_spec = runner._condition_spec(orchestrator, family, config)
        assert condition_spec["name"] == family.family_name
        assert condition_spec["profiles"]["attacker"] == family.attacker_profile_id


def test_distribution_authority_runner_sync_ledgers_tracks_88_pending_runs(
    tmp_path: Path,
) -> None:
    runner = DefenderAuthorityAblationRunner(tmp_path / "defender-authority-ablation")
    runner.ensure_root_layout()
    runner._sync_all_ledgers()
    resume_state = runner.resume_state_path.read_text()
    assert '"completed_runs": 0' in resume_state
    assert '"pending_runs": 88' in resume_state


def test_distribution_authority_configs_preserve_existing_attacker_resolutions() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
    baseline_paths = {
        "black_box_attacker": REPO_ROOT
        / "configs/experiments/water_distribution_black_box_vs_agentic_defender.yaml",
        "gray_box_attacker": REPO_ROOT
        / "configs/experiments/water_distribution_gray_box_vs_agentic_defender.yaml",
        "white_box_attacker": REPO_ROOT
        / "configs/experiments/water_distribution_white_box_vs_agentic_defender.yaml",
        "black_box_high_cyber_high_domain_distribution_attacker": REPO_ROOT
        / "configs/experiments/water_distribution_domain_ablation_black_box_high_cyber_high_domain_vs_agentic_defender.yaml",
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
        "black_box_high_cyber_high_domain_distribution_attacker": (
            AUTHORITY_CONFIG_PATHS[3],
            "black_box_high_cyber_high_domain__observe_alert_only_defender",
        ),
    }
    baseline_conditions = {
        "black_box_attacker": "black_box_attacker__agentic_defender",
        "gray_box_attacker": "gray_box_attacker__agentic_defender",
        "white_box_attacker": "white_box_attacker__agentic_defender",
        "black_box_high_cyber_high_domain_distribution_attacker": (
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


def test_distribution_authority_defender_profiles_match_expected_semantics() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
    observe = _condition_by_name(
        orchestrator,
        AUTHORITY_CONFIG_PATHS[0],
        "black_box_attacker__observe_alert_only_defender",
    )
    delayed = _condition_by_name(
        orchestrator,
        AUTHORITY_CONFIG_PATHS[0],
        "black_box_attacker__delayed_containment_defender",
    )
    baseline = _condition_by_name(
        orchestrator,
        REPO_ROOT / "configs/experiments/water_distribution_black_box_vs_agentic_defender.yaml",
        "black_box_attacker__agentic_defender",
    )

    observe_defender = observe["effective_profiles"]["defender"]
    delayed_defender = delayed["effective_profiles"]["defender"]
    baseline_defender = baseline["effective_profiles"]["defender"]

    assert observe_defender["allowed_action_types"] == ["noop"]
    assert delayed_defender["runtime_containment_activation_delay_steps"] == 2
    assert delayed_defender["allowed_action_types"] == baseline_defender["allowed_action_types"]


def test_distribution_authority_manifest_uses_retained_eleven_scenarios() -> None:
    runner = DefenderAuthorityAblationRunner(AUTHORITY_RESULT_ROOT)
    manifest_rows = runner.manifest_rows()

    assert len(manifest_rows) == 11
    assert manifest_rows[0]["scenario_id"] == "wd-attack-coordinated-grid-rws-delay"
    assert manifest_rows[-1]["scenario_id"] == "wd-attack-supply-low-spoof"


def test_distribution_authority_run_id_uses_defender_variant_label() -> None:
    runner = DefenderAuthorityAblationRunner(AUTHORITY_RESULT_ROOT)
    scenario = ScenarioCatalog(testbed_id="water_distribution").get(
        "attack/wd-attack-supply-low-spoof"
    )

    observe_run_id = runner._run_id(
        AUTHORITY_FAMILY_SPECS[0],
        AUTHORITY_SEED,
        scenario,
        "heldout_test",
    )
    delayed_run_id = runner._run_id(
        AUTHORITY_FAMILY_SPECS[1],
        AUTHORITY_SEED,
        scenario,
        "heldout_test",
    )

    assert "observe_alert_only" in observe_run_id
    assert "delayed_containment" in delayed_run_id


def test_distribution_authority_freeze_metadata_is_stable_under_local_model_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = tmp_path / "defender-authority-ablation"
    runner = DefenderAuthorityAblationRunner(result_root)

    runner.freeze_metadata()
    runner.completed_runs_path.write_text("{}\n")

    monkeypatch.delenv("QWEN_MODEL_PATH", raising=False)
    monkeypatch.delenv("QWEN_MODEL_ID", raising=False)
    runner.freeze_metadata()

    monkeypatch.setenv("QWEN_MODEL_PATH", "alternate-qwen-path")
    runner.freeze_metadata()
