from agents.scenario_generation.agent import ScenarioGenerationAgent


def test_scenario_generation_agent_stub():
    out = ScenarioGenerationAgent().run()
    assert out["agent"] == "scenario_generation"
    assert out["read_only"] is True
