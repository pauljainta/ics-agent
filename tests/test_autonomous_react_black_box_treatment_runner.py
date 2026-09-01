from __future__ import annotations

import json
from pathlib import Path

from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog


CONFIG_NAME = (
    "water_treatment_black_box_autonomous_react_attacker_vs_silent_defender_smoke.yaml"
)
SCRIPT_NAME = (
    "run_water_treatment_autonomous_react_black_box_attacker_vs_silent_defender_smoke.py"
)
SMOKE_SCENARIOS = [
    "wt-attack-p2-multi-chemical-drain",
    "wt-attack-p2-dual-hijack",
    "wt-attack-sensor-rw-cascade-low-hh",
    "wt-attack-sensor-uf-high-disrupt",
]


def _black_box_agent() -> tuple[AgenticRoleAgent, dict]:
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    profile = orchestrator.agent_profiles["black_box_autonomous_react_attacker"]
    return AgenticRoleAgent("black_box_autonomous_react_attacker", profile), profile


def _context(*, recon_done: bool) -> dict:
    catalog = ScenarioCatalog(testbed_id="water_treatment")
    scenario = catalog.resolve_selectors(["attack/wt-attack-sensor-rw-cascade-low-hh"])[0]
    executed_actions = []
    if recon_done:
        executed_actions = [
            {
                "role": "attacker",
                "time_sec": 0.0,
                "action": {"type": "inspect_surface", "surface": "tags"},
            },
            {
                "role": "attacker",
                "time_sec": 0.5,
                "action": {"type": "inspect_surface", "surface": "alarm_bits"},
            },
        ]
    return {
        "scenario": scenario,
        "time_sec": 1.0,
        "observation": {
            "STATE_NAME": "RUNNING",
            "RW_Tank_Level": 600,
            "UF_UFFT_Tank_Level": 400,
            "ChemTreat_NaCl_Level": 800,
            "ChemTreat_NaOCl_Level": 800,
            "ChemTreat_HCl_Level": 800,
        },
        "executed_actions": executed_actions,
    }


def _action_types(candidates: list[dict]) -> set[str]:
    return {
        str((candidate.get("action") or {}).get("type"))
        for candidate in candidates
        if isinstance(candidate.get("action"), dict)
    }


def test_black_box_autonomous_react_config_selects_four_silent_runs() -> None:
    config = load_config(CONFIG_NAME)
    catalog = ScenarioCatalog(testbed_id="water_treatment")
    selected = catalog.resolve_selectors(list(config["scenario_selectors"]))
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    condition = orchestrator._enumerate_conditions(config)[0]
    effective = condition["effective_profiles"]

    assert [scenario.scenario_id for scenario in selected] == SMOKE_SCENARIOS
    assert config["reproducibility"]["seed"] == 101
    assert config["resume"] is True
    assert config["environment"]["backend"] == "sphere_openplc"
    assert config["environment"]["infrastructure_profile"] == "water_treatment"
    assert config["results_root"].endswith(
        "Artifact/Results/Water-treatment/Autonomous-React/Black-Box-Silent-Smoke"
    )
    assert condition["name"] == "black_box_autonomous_react_attacker__silent_defender"
    assert condition["profiles"]["attacker"] == "black_box_autonomous_react_attacker"
    assert condition["profiles"]["defender"] == "silent_defender"
    assert effective["attacker"]["knowledge_mode"] == "black_box"
    assert effective["attacker"]["scaffold_mode"] == "pure_react"
    assert effective["attacker"]["autonomous_react_tools"] is True
    assert effective["attacker"]["llm"]["model_id"] == "Qwen/Qwen3.5-9B"
    assert effective["attacker"]["llm"]["quantization"] == "4bit"
    assert effective["attacker"]["reasoning_cooldown_sec"] == 0.0
    assert effective["attacker"]["reasoning_refresh_sec"] == 0.0
    assert effective["defender"]["allowed_action_types"] == ["noop"]


def test_black_box_autonomous_react_starts_with_opaque_recon_only() -> None:
    agent, _profile = _black_box_agent()
    context = _context(recon_done=False)
    agent.reset(context["scenario"])

    candidates = agent._black_box_autonomous_react_candidates(
        context=context,
        telemetry_summary={"active_alarms": [], "risk_flags": [], "relationship_notes": []},
        structured_memory={},
        mitre_context={},
    )

    assert candidates
    assert _action_types(candidates) <= {"inspect_surface", "noop"}
    serialized = json.dumps(candidates)
    assert "surf_" in serialized
    assert "RW_Tank_Level" not in serialized
    assert "ChemTreat" not in serialized
    assert all(candidate["source"] == "black_box_autonomous_react_tool" for candidate in candidates)


def test_black_box_autonomous_react_offers_opaque_probe_after_recon() -> None:
    agent, _profile = _black_box_agent()
    context = _context(recon_done=True)
    agent.reset(context["scenario"])

    candidates = agent._black_box_autonomous_react_candidates(
        context=context,
        telemetry_summary={"active_alarms": [], "risk_flags": [], "relationship_notes": []},
        structured_memory={},
        mitre_context={},
    )

    types = _action_types(candidates)
    serialized = json.dumps(candidates)
    assert "write_signal" in types
    assert "toggle_control" in types
    assert "sig_" in serialized
    assert "ctrl_" in serialized
    assert "RW_Tank_Level" not in serialized
    assert "UF_UFFT_Tank_Level" not in serialized
    assert "ChemTreat" not in serialized


def test_black_box_autonomous_react_resolves_opaque_candidate() -> None:
    agent, _profile = _black_box_agent()
    context = _context(recon_done=True)
    agent.reset(context["scenario"])
    candidates = agent._black_box_autonomous_react_candidates(
        context=context,
        telemetry_summary={"active_alarms": [], "risk_flags": [], "relationship_notes": []},
        structured_memory={},
        mitre_context={},
    )
    candidate = next(
        item
        for item in candidates
        if (item.get("action") or {}).get("type") == "write_signal"
    )

    resolved = agent._black_box_resolved_candidate(candidate["id"])

    assert resolved is not None
    assert (resolved.get("action") or {}).get("type") == "set_sensor"
    assert (resolved.get("visible_action") or {}).get("type") == "write_signal"


def test_black_box_autonomous_react_runner_script_exists() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / SCRIPT_NAME

    assert script_path.exists()
    assert "CONFIG_PATH" in script_path.read_text()
