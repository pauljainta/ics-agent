from __future__ import annotations

from ics_agent.environment import OpenPLCEnvironment


def _fresh_env(fake_openplc) -> OpenPLCEnvironment:
    env = OpenPLCEnvironment({"backend": "auto", "start_bridge": False})
    env.start()
    env.reset()
    return env


def test_block_sensor_write_blocks_later_attacker_sensor_writes(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    containment = env.act({"type": "block_sensor_write"}, role="defender")
    blocked = env.act(
        {"type": "set_sensor", "name": "RW_Tank_Level", "value": 123},
        role="attacker",
    )

    assert containment["containment"]["execution_mode"] == "runtime_enforced"
    assert containment["runtime_containment"]["target_surfaces"] == ["sensor_write_path"]
    assert blocked["blocked_by_runtime_containment"] is True
    assert blocked["blocked_surface"] == "sensor_write_path"
    assert env.testbed.actions == []


def test_quarantine_signal_source_strengthens_same_sensor_surface(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    env.act({"type": "block_sensor_write"}, role="defender")
    containment = env.act({"type": "quarantine_signal_source"}, role="defender")
    blocked = env.act(
        {"type": "set_sensor", "name": "UF_UFFT_Tank_Level", "value": 999},
        role="attacker",
    )

    rule = containment["runtime_containment"]["rules"]["sensor_write_path"]
    assert rule["source_action_type"] == "quarantine_signal_source"
    assert rule["strength"] == 2
    assert blocked["blocked_by_action_type"] == "quarantine_signal_source"


def test_disable_hmi_session_blocks_later_attacker_hmi_commands(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    env.act({"type": "disable_hmi_session"}, role="defender")
    blocked = env.act({"type": "hmi_stop", "process_id": "WT-P1"}, role="attacker")

    assert blocked["blocked_by_runtime_containment"] is True
    assert blocked["blocked_surface"] == "hmi_session"
    assert env.testbed.actions == []


def test_block_process2_manual_path_blocks_process2_select_and_valve_actions(
    fake_openplc,
) -> None:
    env = _fresh_env(fake_openplc)

    env.act({"type": "block_process2_manual_path"}, role="defender")
    blocked_select = env.act(
        {"type": "select_process", "process_id": "WT-P2", "active": True},
        role="attacker",
    )
    blocked_valve = env.act(
        {
            "type": "set_manual_valve",
            "process_id": "WT-P2",
            "chemical": "NaCl",
            "open": True,
        },
        role="attacker",
    )

    assert blocked_select["blocked_surface"] == "process2_manual_path"
    assert blocked_valve["blocked_surface"] == "process2_manual_path"
    assert env.testbed.actions == []


def test_isolate_engineering_host_blocks_all_later_attacker_actions(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    env.act({"type": "isolate_engineering_host"}, role="defender")
    blocked_recon = env.act(
        {"type": "inspect_surface", "surface": "tags"},
        role="attacker",
    )
    blocked_control = env.act(
        {"type": "set_sensor", "name": "RW_Tank_Level", "value": 101},
        role="attacker",
    )

    assert blocked_recon["blocked_surface"] == "all"
    assert blocked_control["blocked_surface"] == "all"


def test_shutdown_actions_still_flow_to_concrete_plant_path(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    stop_result = env.act({"type": "hmi_stop", "process_id": "WT-P1"}, role="defender")
    estop_result = env.act({"type": "hmi_estop", "process_id": "WT-P1"}, role="defender")

    assert stop_result["state_name"] == "STOPPED"
    assert estop_result["state_name"] == "ESTOPPED"
    assert [item["type"] for item in env.testbed.actions] == ["hmi_stop", "hmi_estop"]


def test_runtime_containment_state_resets_between_episodes(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    env.act({"type": "block_sensor_write"}, role="defender")
    env.reset()
    allowed = env.act(
        {"type": "set_sensor", "name": "RW_Tank_Level", "value": 456},
        role="attacker",
    )

    assert allowed.get("blocked_by_runtime_containment") is not True
    assert env.runtime_containment_snapshot()["active"] is False
    assert env.testbed.actions[-1]["type"] == "set_sensor"
