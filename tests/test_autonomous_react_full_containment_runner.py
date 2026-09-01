from __future__ import annotations

from pathlib import Path

from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog


CONFIG_NAME = (
    "water_treatment_autonomous_react_white_box_attacker_vs_agentic_full_containment_defender.yaml"
)
SCRIPT_NAME = (
    "run_water_treatment_autonomous_react_white_box_attacker_vs_agentic_full_containment_defender.py"
)


def test_autonomous_react_full_containment_runner_config_is_full_mitre18() -> None:
    config = load_config(CONFIG_NAME)
    catalog = ScenarioCatalog(testbed_id="water_treatment")
    selected = catalog.resolve_selectors(list(config["scenario_selectors"]))

    assert Path(config["_path"]).name == CONFIG_NAME
    assert config["testbed"] == "water_treatment"
    assert config["reproducibility"]["seed"] == 101
    assert config["resume"] is True
    assert config["use_canonical_result_root"] is True
    assert config["results_root"].endswith(
        "Artifact/Results/Water-treatment/Autonomous-React/White-Box"
    )
    assert len(selected) == 18
    assert all(scenario.catalog_id.startswith("attack/") for scenario in selected)


def test_autonomous_react_full_containment_condition_profiles_are_expected() -> None:
    config = load_config(CONFIG_NAME)
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    condition = orchestrator._enumerate_conditions(config)[0]
    effective_profiles = condition["effective_profiles"]

    assert condition["name"] == "white_box_autonomous_react_attacker__agentic_defender"
    assert condition["profiles"]["attacker"] == "white_box_autonomous_react_attacker"
    assert condition["profiles"]["defender"] == "agentic_defender"

    attacker = effective_profiles["attacker"]
    defender = effective_profiles["defender"]

    assert attacker["knowledge_mode"] == "white_box"
    assert attacker["autonomous_react_tools"] is True
    assert attacker["white_box_commit_guardrail"] is False
    assert attacker["llm"]["model_id"] == "Qwen/Qwen3.5-9B"
    assert attacker["llm"]["quantization"] == "4bit"

    assert "block_sensor_write" in defender["allowed_action_types"]
    assert "block_process2_manual_path" in defender["allowed_action_types"]
    assert "quarantine_signal_source" in defender["allowed_action_types"]
    assert defender["llm"]["model_id"] == "Qwen/Qwen3.5-9B"


def test_autonomous_react_full_containment_runner_script_exists() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / SCRIPT_NAME

    assert script_path.exists()
    assert "CONFIG_PATH" in script_path.read_text()
