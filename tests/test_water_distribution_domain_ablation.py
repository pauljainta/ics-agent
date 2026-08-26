from __future__ import annotations

from pathlib import Path

from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.experiments.water_distribution.domain_ablation import (
    BLACK_BOX_RUNTIME_NOTE,
    DOMAIN_CONFIG_PATH,
    DOMAIN_FAMILY_SPEC,
    DOMAIN_RESULT_ROOT,
    ENRICHED_CONDITION_NOTE,
    DomainAblationRunner,
)


def test_domain_ablation_runner_resolves_single_family_seed_and_result_root() -> None:
    runner = DomainAblationRunner(DOMAIN_RESULT_ROOT)
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
    config = runner._runtime_config(DOMAIN_FAMILY_SPEC, 101)
    condition_spec = runner._condition_spec(orchestrator, DOMAIN_FAMILY_SPEC, config)

    assert runner.result_root == DOMAIN_RESULT_ROOT.resolve()
    assert condition_spec["profiles"]["attacker"] == (
        "black_box_high_cyber_high_domain_distribution_attacker"
    )
    assert condition_spec["name"] == DOMAIN_FAMILY_SPEC.family_name


def test_domain_ablation_config_and_prompt_hash_payload_track_phase35_inputs() -> None:
    runner = DomainAblationRunner(DOMAIN_RESULT_ROOT)
    config = load_config(str(DOMAIN_CONFIG_PATH))
    payload = runner._prompt_hash_payload()
    family_payload = payload["family_prompt_hashes"][DOMAIN_FAMILY_SPEC.family_name]["bundle"]

    assert Path(config["_path"]).resolve() == DOMAIN_CONFIG_PATH.resolve()
    assert config["reproducibility"]["seed"] == 101
    attacker_knowledge = family_payload["knowledge_files"]["attacker"]
    assert attacker_knowledge
    assert any(key.endswith("black_box_constraints_manual.md") for key in attacker_knowledge)
    assert any(key.endswith("water_distribution_domain_primer.md") for key in attacker_knowledge)


def test_domain_ablation_sync_ledgers_use_single_seed_single_family_counts(
    tmp_path: Path,
) -> None:
    runner = DomainAblationRunner(tmp_path / "domain-ablation")
    runner.ensure_root_layout()
    runner._sync_all_ledgers()
    resume_state = runner.resume_state_path.read_text()
    assert '"completed_runs": 0' in resume_state
    assert '"pending_runs": 11' in resume_state


def test_domain_ablation_audit_uses_decision_engine_prompt_formatter() -> None:
    runner = DomainAblationRunner(DOMAIN_RESULT_ROOT)
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
    config = runner._runtime_config(DOMAIN_FAMILY_SPEC, 101)
    condition_spec = runner._condition_spec(orchestrator, DOMAIN_FAMILY_SPEC, config)
    effective_profiles = condition_spec.get("effective_profiles") or {}
    attacker = dict(
        effective_profiles.get("attacker")
        or orchestrator.agent_profiles[DOMAIN_FAMILY_SPEC.attacker_profile_id]
    )
    agent = AgenticRoleAgent("domain_ablation_audit", attacker)
    scenario = ScenarioCatalog(testbed_id="water_distribution").get(
        "attack/wd-attack-supply-low-spoof"
    )
    agent.reset(scenario)
    context = agent._build_online_context(
        {
            "scenario": scenario,
            "observation": {"STATE_NAME": "RUNNING"},
            "observations": [{"STATE_NAME": "RUNNING"}],
            "executed_actions": [],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
            "time_sec": 0.0,
        }
    )

    prompt = agent.engine._format_prompt(context)
    assert "Operational manuals:" in prompt
    assert '"scenario_id": "opaque-interface-live"' in prompt


def test_domain_ablation_repo_note_mentions_enriched_primer_and_black_box_runtime(
    tmp_path: Path,
) -> None:
    runner = DomainAblationRunner(tmp_path / "domain-ablation")
    runner.ensure_root_layout()
    runner.write_repo_note()

    note_text = runner.repo_note_path.read_text()

    assert ENRICHED_CONDITION_NOTE in note_text
    assert BLACK_BOX_RUNTIME_NOTE in note_text
