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
