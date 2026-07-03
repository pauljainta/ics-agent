from __future__ import annotations

from pathlib import Path

from ics_agent.mitre import (
    ATTACKER_PHASES,
    DEFENDER_PHASES,
    OPTIONB_MITRE_MODEL_ID,
    OPTIONB_STAGE_PATH,
    optionb_mitre_context,
)
from ics_agent.orchestrator import ExperimentOrchestrator


def test_optionb_mitre_context_preserves_staged_runtime_model() -> None:
    empty_context = {"decisions": [], "executed_actions": [], "alerts": []}

    attacker_context = optionb_mitre_context(role="attacker", context=empty_context)
    defender_context = optionb_mitre_context(role="defender", context=empty_context)

    assert attacker_context["phase_model"] == OPTIONB_MITRE_MODEL_ID
    assert attacker_context["staged_path"] == OPTIONB_STAGE_PATH
    assert attacker_context["phase_order"] == ATTACKER_PHASES
    assert attacker_context["current_phase"] == ATTACKER_PHASES[0]
    assert attacker_context["direct_plant_shortcuts_forbidden"] is True

    assert defender_context["phase_model"] == OPTIONB_MITRE_MODEL_ID
    assert defender_context["staged_path"] == OPTIONB_STAGE_PATH
    assert defender_context["phase_order"] == DEFENDER_PHASES
    assert defender_context["current_phase"] == DEFENDER_PHASES[0]


def test_configure_reproducibility_keeps_seeded_run_metadata(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "20260623")
    orchestrator = ExperimentOrchestrator()
    payload = orchestrator._configure_reproducibility(
        config={
            "reproducibility": {"seed": 20260623},
            "_path": Path("/tmp/concrete-containment.yaml"),
        },
        conditions=[
            {
                "profiles": {
                    "attacker": "gray_box_attacker",
                    "defender": "agentic_defender",
                    "observer": "agentic_observer",
                    "planner": "scripted_planner",
                    "operator": "scripted_operator",
                    "offline_analysis": "scripted_offline_analyst",
                }
            }
        ],
    )

    assert payload["seed"] == 20260623
    assert payload["pythonhashseed"] == "20260623"
    assert payload["suite_decoding"]
    assert payload["experiment_config_path"] == "/tmp/concrete-containment.yaml"
