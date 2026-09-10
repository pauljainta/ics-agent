from __future__ import annotations

from ics_agent.metrics import compute_metrics


def test_metrics_distinguish_runtime_enforced_from_symbolic_containment() -> None:
    bundle = {
        "observations": [
            {"elapsed_sec": 0.0, "STATE_NAME": "RUNNING"},
            {"elapsed_sec": 1.0, "STATE_NAME": "RUNNING"},
            {"elapsed_sec": 2.0, "STATE_NAME": "RUNNING"},
        ],
        "decisions": [],
        "alerts": [],
        "system_events": [
            {
                "timestamp_utc": "2026-07-03T00:00:00Z",
                "time_sec": 1.0,
                "event_type": "attacker_execution_cutoff_activated",
            },
            {
                "timestamp_utc": "2026-07-03T00:00:01Z",
                "time_sec": 2.0,
                "event_type": "attacker_action_blocked",
                "attempted_action_type": "set_sensor",
            },
        ],
        "executed_actions": [
            {
                "time_sec": 0.5,
                "role": "attacker",
                "action": {"type": "set_sensor", "name": "RW_Tank_Level", "value": 100},
                "metadata": {},
            },
            {
                "time_sec": 1.0,
                "role": "defender",
                "action": {"type": "block_sensor_write"},
                "containment": {
                    "is_containment_action": True,
                    "is_symbolic": False,
                    "execution_mode": "runtime_enforced",
                    "target_surfaces": ["sensor_write_path"],
                },
                "metadata": {},
            },
        ],
    }
    scenario = {
        "suite": "attack",
        "scenario_id": "attack_rw_refill_spoof",
        "metadata": {
            "ground_truth": {
                "incident_type": "attack",
                "mitigation_actions": ["block_sensor_write"],
            }
        },
    }

    metrics = compute_metrics(bundle, scenario)

    assert metrics["defender_runtime_enforced_containment_action_count"] == 1
    assert metrics["defender_symbolic_containment_action_count"] == 0
    assert metrics["runtime_containment"]["runtime_enforced_containment_action_count"] == 1
    assert metrics["runtime_containment"]["symbolic_containment_action_count"] == 0
    assert metrics["runtime_containment"]["attacker_execution_cutoff_activated"] is True
    assert metrics["runtime_containment"]["blocked_attacker_action_count"] == 1


def test_water_distribution_wargame_logs_wall_clock_attack_progress_metrics() -> None:
    bundle = {
        "config_snapshot": {
            "phase_boundary": {
                "wargame_t0_utc": "2026-09-10T00:00:05+00:00",
                "wargame_t0_elapsed_sec": 0.0,
                "metric_window": "phase_b_only",
                "metric_clock_origin": "wargame_t0",
            }
        },
        "observations": [
            {
                "timestamp_utc": "2026-09-10T00:00:05+00:00",
                "elapsed_sec": 0.0,
                "STATE_NAME": "RUNNING",
                "Grid_Elev_Tank_Level": 2200,
            },
            {
                "timestamp_utc": "2026-09-10T00:00:22+00:00",
                "elapsed_sec": 2.5,
                "STATE_NAME": "RUNNING",
                "Grid_Elev_Tank_Level": 2600,
            },
            {
                "timestamp_utc": "2026-09-10T00:00:40+00:00",
                "elapsed_sec": 4.0,
                "STATE_NAME": "RUNNING",
                "Grid_Elev_Tank_Level": 2700,
            },
        ],
        "decisions": [],
        "alerts": [
            {
                "timestamp_utc": "2026-09-10T00:00:23+00:00",
                "time_sec": 2.6,
                "role": "defender",
                "decision_type": "alert",
                "diagnosis": "grid_level_overflow",
            }
        ],
        "executed_actions": [
            {
                "timestamp_utc": "2026-09-10T00:00:10+00:00",
                "time_sec": 1.0,
                "role": "attacker",
                "agent_name": "white_box_attacker",
                "action": {"type": "inspect_surface", "surface": "tags"},
                "result": {"category": "reconnaissance", "surface": "tags"},
                "metadata": {"phase_family": "attacker"},
            },
            {
                "timestamp_utc": "2026-09-10T00:00:18+00:00",
                "time_sec": 2.0,
                "role": "attacker",
                "agent_name": "white_box_attacker",
                "action": {"type": "set_actuator", "name": "Grid_Elev_Valve", "value": True},
                "metadata": {"phase_family": "attacker"},
            },
            {
                "timestamp_utc": "2026-09-10T00:00:25+00:00",
                "time_sec": 3.0,
                "role": "defender",
                "agent_name": "agentic_defender",
                "action": {"type": "block_actuator_write"},
                "containment": {
                    "is_containment_action": True,
                    "is_symbolic": False,
                    "execution_mode": "runtime_enforced",
                    "target_surfaces": ["actuator_write_path"],
                },
                "metadata": {},
            },
        ],
        "system_events": [
            {
                "timestamp_utc": "2026-09-10T00:00:05+00:00",
                "time_sec": 0.0,
                "event_type": "wargame_t0",
            },
            {
                "timestamp_utc": "2026-09-10T00:00:29+00:00",
                "time_sec": 3.5,
                "event_type": "attacker_execution_cutoff_activated",
                "role": "defender",
                "action": {"type": "block_actuator_write"},
            },
            {
                "timestamp_utc": "2026-09-10T00:00:31+00:00",
                "time_sec": 4.0,
                "event_type": "attacker_action_blocked",
                "attempted_action_type": "set_actuator",
            },
        ],
    }
    scenario = {
        "suite": "attack",
        "scenario_id": "wd-attack-grid-valve-force-open",
        "metadata": {
            "ground_truth": {
                "incident_type": "attack",
                "mitigation_actions": ["block_actuator_write"],
            },
            "safety_bounds": {
                "Grid_Elev_Tank_Level": {"min": 400, "max": 2500},
            },
        },
    }

    metrics = compute_metrics(bundle, scenario)

    assert metrics["wall_clock_time_to_detection_from_attack_start_sec"] == 13.0
    assert metrics["wall_clock_time_to_first_useful_attacker_action_from_attack_start_sec"] == 8.0
    assert metrics["wall_clock_time_to_first_defender_action_from_attack_start_sec"] == 15.0
    assert (
        metrics["wall_clock_time_to_effective_containment_cutoff_from_attack_start_sec"]
        == 19.0
    )
    assert metrics["wall_clock_time_to_first_safety_violation_from_attack_start_sec"] == 12.0
    assert metrics["attacker_action_count"] == 2
    assert metrics["attacker_attempted_action_count"] == 3
    assert metrics["attacker_ineffective_action_count"] == 2
    assert metrics["distinct_surfaces_probed"] == ["tags"]
    assert metrics["distinct_surfaces_probed_before_first_useful_count"] == 1
    assert metrics["physical_impact"]["violated_field_count"] == 1
    assert metrics["physical_impact"]["unsafe_wall_clock_duration_sec"] == 18.0
    assert metrics["physical_impact"]["max_safety_violation_magnitude_by_field"] == {
        "Grid_Elev_Tank_Level": 200.0
    }
