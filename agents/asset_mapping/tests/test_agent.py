from agents.asset_mapping.agent import AssetMappingAgent


def test_asset_mapping_agent_stub():
    out = AssetMappingAgent().run()
    assert out["agent"] == "asset_mapping"
    assert out["read_only"] is True
