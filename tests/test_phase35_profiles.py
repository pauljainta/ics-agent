from __future__ import annotations

import re
from pathlib import Path

import yaml

from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.testbeds.registry import observation_focus_fields, trend_fields


REPO_ROOT = Path(__file__).resolve().parents[1]
TREATMENT_PROFILE_ID = "black_box_high_cyber_high_domain_treatment_attacker"
DISTRIBUTION_PROFILE_ID = "black_box_high_cyber_high_domain_distribution_attacker"


def _distribution_black_box_baseline(orchestrator: ExperimentOrchestrator) -> dict[str, object]:
    config = yaml.safe_load(
        (
            REPO_ROOT
            / "configs/experiments/water_distribution_black_box_vs_agentic_defender.yaml"
        ).read_text()
    )
    return orchestrator._effective_condition_profiles(
        profiles=config["conditions"][0]["profiles"],
        profile_overrides=config["conditions"][0]["profile_overrides"],
    )["attacker"]


def _treatment_observation(**overrides: float | str) -> dict[str, float | str]:
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


def _distribution_observation(**overrides: float | str) -> dict[str, float | str]:
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


def _treatment_black_box_context(profile: dict[str, object]) -> dict[str, object]:
    catalog = ScenarioCatalog()
    scenario = catalog.get("attack/wt-attack-sensor-rw-refill-spoof")
    observation = _treatment_observation()
    agent = AgenticRoleAgent("phase35_treatment", profile)
    agent.reset(scenario)
    return agent._build_online_context(
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


def _distribution_black_box_context(profile: dict[str, object]) -> dict[str, object]:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-supply-low-spoof")
    observation = _distribution_observation()
    agent = AgenticRoleAgent("phase35_distribution", profile)
    agent.reset(scenario)
    return agent._build_online_context(
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


def test_phase35_profiles_remain_black_box_and_keep_expected_knowledge_files() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")

    treatment = orchestrator.agent_profiles[TREATMENT_PROFILE_ID]
    distribution = orchestrator.agent_profiles[DISTRIBUTION_PROFILE_ID]

    assert treatment["knowledge_mode"] == "black_box"
    assert distribution["knowledge_mode"] == "black_box"
    assert treatment["role"] == "attacker"
    assert distribution["role"] == "attacker"
    assert treatment["family"] == "agentic"
    assert distribution["family"] == "agentic"

    assert treatment["knowledge_files"] == [
        "configs/knowledge/phase35/black_box_constraints_manual.md",
        "configs/knowledge/phase35/high_cyber_ics_methodology_manual.md",
        "testbeds/water_treatment/knowledge/phase35/water_treatment_domain_primer.md",
    ]
    assert distribution["knowledge_files"] == [
        "configs/knowledge/phase35/black_box_constraints_manual.md",
        "configs/knowledge/phase35/high_cyber_ics_methodology_manual.md",
        "testbeds/water_distribution/knowledge/phase35/water_distribution_domain_primer.md",
    ]


def test_phase35_treatment_profile_preserves_black_box_action_space_and_interface() -> None:
    orchestrator = ExperimentOrchestrator()
    baseline = orchestrator.agent_profiles["black_box_attacker"]
    candidate = orchestrator.agent_profiles[TREATMENT_PROFILE_ID]

    assert candidate["allowed_action_types"] == baseline["allowed_action_types"]

    baseline_context = _treatment_black_box_context(baseline)
    candidate_context = _treatment_black_box_context(candidate)

    assert baseline_context["system_knowledge"] is None
    assert candidate_context["system_knowledge"] is None
    assert candidate_context["observation"].keys() == baseline_context["observation"].keys()
    assert candidate_context["decision_epoch"].keys() == baseline_context["decision_epoch"].keys()
    assert {item["action"]["type"] for item in candidate_context["candidate_actions"]} == {
        item["action"]["type"] for item in baseline_context["candidate_actions"]
    }


def test_phase35_distribution_profile_preserves_black_box_action_space_and_interface() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
    baseline = _distribution_black_box_baseline(orchestrator)
    candidate = orchestrator.agent_profiles[DISTRIBUTION_PROFILE_ID]

    assert candidate["allowed_action_types"] == baseline["allowed_action_types"]

    baseline_context = _distribution_black_box_context(baseline)
    candidate_context = _distribution_black_box_context(candidate)

    assert baseline_context["system_knowledge"] is None
    assert candidate_context["system_knowledge"] is None
    assert candidate_context["observation"].keys() == baseline_context["observation"].keys()
    assert candidate_context["decision_epoch"].keys() == baseline_context["decision_epoch"].keys()
    assert {item["action"]["type"] for item in candidate_context["candidate_actions"]} == {
        item["action"]["type"] for item in baseline_context["candidate_actions"]
    }


def test_phase35_primers_strip_exact_tokens_scenarios_and_thresholds() -> None:
    treatment_primer = (
        REPO_ROOT
        / "testbeds/water_treatment/knowledge/phase35/water_treatment_domain_primer.md"
    ).read_text()
    distribution_primer = (
        REPO_ROOT
        / "testbeds/water_distribution/knowledge/phase35/water_distribution_domain_primer.md"
    ).read_text()

    treatment_tokens = set(observation_focus_fields("water_treatment"))
    treatment_tokens.update(trend_fields("water_treatment"))
    treatment_tokens.update({"WT-P1", "WT-P2", "WT-P3"})

    distribution_contract = yaml.safe_load(
        (REPO_ROOT / "testbeds/water_distribution/tag_contract.yaml").read_text()
    )
    distribution_tokens = set((distribution_contract.get("tags") or {}).keys())
    distribution_tokens.update({"WD-UC0", "WD-Supply", "WD-Grid", "WD-Return"})

    for token in sorted(treatment_tokens):
        assert token not in treatment_primer
    for token in sorted(distribution_tokens):
        assert token not in distribution_primer

    treatment_catalog = ScenarioCatalog()
    distribution_catalog = ScenarioCatalog(testbed_id="water_distribution")
    treatment_markers = {
        scenario.catalog_id.split("/")[-1]
        for scenario in treatment_catalog.resolve_selectors(["attack/*"])
    }
    distribution_markers = {
        scenario.catalog_id.split("/")[-1]
        for scenario in distribution_catalog.resolve_selectors(["attack/*"])
    }
    for marker in sorted(treatment_markers):
        assert marker not in treatment_primer
    for marker in sorted(distribution_markers):
        assert marker not in distribution_primer

    assert re.search(r"\b\d{3,}\b", treatment_primer) is None
    assert re.search(r"\b\d{3,}\b", distribution_primer) is None
    for phrase in (
        "best leverage surface",
        "highest leverage",
        "preferred attack surface",
        "surface ranking",
    ):
        assert phrase not in treatment_primer.lower()
        assert phrase not in distribution_primer.lower()


def test_phase35_treatment_primer_includes_swat_informed_enrichment_sections() -> None:
    treatment_primer = (
        REPO_ROOT
        / "testbeds/water_treatment/knowledge/phase35/water_treatment_domain_primer.md"
    ).read_text().lower()

    for heading in (
        "swat-informed water-treatment domain primer",
        "## process layout",
        "## component functions",
        "## stage dependencies and control relationships",
        "## failure modes",
        "## known attack sequences",
        "## successful attack recipes",
        "## attack-to-process mapping",
        "## black-box use discipline",
    ):
        assert heading in treatment_primer


def test_phase35_distribution_primer_includes_wadi_informed_enrichment_sections() -> None:
    distribution_primer = (
        REPO_ROOT
        / "testbeds/water_distribution/knowledge/phase35/water_distribution_domain_primer.md"
    ).read_text().lower()

    for heading in (
        "wadi-informed water-distribution domain primer",
        "## process layout",
        "## component functions",
        "## stage dependencies and control relationships",
        "## failure modes",
        "## known attack sequences",
        "## successful attack recipes",
        "## attack-to-process mapping",
        "## black-box use discipline",
    ):
        assert heading in distribution_primer
