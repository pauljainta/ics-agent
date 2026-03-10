"""Asset Mapping agent entrypoint."""


class AssetMappingAgent:
    """Discovers and normalizes ICS assets into a KG."""

    def run(self) -> dict:
        return {"agent": "asset_mapping", "status": "stub", "read_only": True}
