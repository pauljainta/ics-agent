from agents.deployment_node.agent import DeploymentNodeAgent


def test_deployment_node_agent_stub():
    out = DeploymentNodeAgent().run()
    assert out["agent"] == "deployment_node"
    assert out["read_only"] is True
    assert out["actuation_consent"] is False
