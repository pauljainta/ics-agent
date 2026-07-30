from __future__ import annotations

from pathlib import Path

from ics_agent.environment import OpenPLCEnvironment
from ics_agent.scenarios import ScenarioCatalog


def test_water_distribution_catalog_loads_local_attack_suite() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")

    scenarios = catalog.resolve_selectors(["attack/*"])

    assert len(scenarios) == 12
    assert all(item.testbed_id == "water_distribution" for item in scenarios)
    assert {
        item.catalog_id for item in scenarios
    } >= {
        "attack/wd-attack-supply-low-spoof",
        "attack/wd-attack-grid-low-spoof",
        "attack/wd-attack-coordinated-grid-rws-delay",
    }

    grid_low = catalog.get("attack/wd-attack-grid-low-spoof")
    assert grid_low.metadata["name"] == "Elevated Tank Low Spoof"
    assert (
        grid_low.metadata["ground_truth"]["attack_class"] == "single_point_deception"
    )
    assert Path(grid_low.execution_path).is_file()


def test_water_distribution_environment_uses_local_testbed_and_actuator_containment() -> None:
    env = OpenPLCEnvironment(
        {
            "testbed": "water_distribution",
            "backend": "auto",
            "start_bridge": False,
        }
    )

    start = env.start()
    reset = env.reset()
    surface = env.inspect_surface({"type": "inspect_surface", "surface": "tags"})
    containment = env.act({"type": "block_actuator_write"}, role="defender")
    blocked = env.act(
        {"type": "set_actuator", "name": "Grid_Elev_Valve", "value": True},
        role="attacker",
    )

    assert start["started"] is True
    assert start["testbed"] == "water_distribution"
    assert Path(start["support_root"]).name == "openplc"
    assert reset["state_name"] == "IDLE"
    assert surface["reconnaissance"]["surface"] == "tags"
    assert "commands" in surface["reconnaissance"]["groups"]
    assert containment["containment"]["execution_mode"] == "runtime_enforced"
    assert containment["runtime_containment"]["target_surfaces"] == [
        "actuator_write_path"
    ]
    assert blocked["blocked_by_runtime_containment"] is True
    assert blocked["blocked_surface"] == "actuator_write_path"
