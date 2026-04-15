from __future__ import annotations

from ics_agent.metrics import compute_metrics


def test_compute_metrics_detects_ttd_ttm_and_diagnosis() -> None:
    bundle = {
        "observations": [
            {"STATE_NAME": "RUNNING", "RW_Tank_Level": 600},
            {"STATE_NAME": "RUNNING", "RW_Tank_Level": 390},
            {"STATE_NAME": "SHUTDOWN", "RW_Tank_Level": 390},
        ],
        "decisions": [
            {
                "role": "observer",
                "time_sec": 8.0,
                "diagnosis": "raw_water_low_or_spoof",
            },
            {
                "role": "defender",
                "time_sec": 8.2,
                "diagnosis": "raw_water_low_or_spoof",
            },
        ],
        "alerts": [
            {"role": "observer", "time_sec": 8.0},
        ],
        "executed_actions": [
            {"role": "attacker", "time_sec": 7.0, "action": {"type": "set_sensor"}},
            {"role": "defender", "time_sec": 8.2, "action": {"type": "hmi_stop"}},
        ],
        "passed": True,
    }
    scenario = {
        "metadata": {
            "ground_truth": {
                "incident_start_time_sec": 7.0,
                "expected_diagnosis": "raw_water_low_or_spoof",
                "expected_defender_actions": ["hmi_stop"],
                "mitigation_actions": ["hmi_stop"],
            },
            "safety_bounds": {
                "RW_Tank_Level": {"min": 0, "max": 1200},
            },
        }
    }
    metrics = compute_metrics(bundle, scenario)
    assert metrics["time_to_detection_sec"] == 1.0
    assert metrics["time_to_mitigation_sec"] == 1.2
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["true_positive_rate"] == 1.0
    assert metrics["diagnosis_accuracy"] == 1.0
    assert metrics["decision_correctness"] == 1.0
    assert metrics["observer_detection"]["time_to_detection_sec"] == 1.0
    assert metrics["defender_detection"]["time_to_detection_sec"] == 1.2
    assert metrics["defender_intervention"]["time_to_mitigation_sec"] == 1.2
    assert metrics["observer_detection"]["true_positive_count"] == 1
    assert metrics["defender_intervention"]["mitigation_action_count"] == 1


def test_compute_metrics_credits_defender_detection_from_action_without_alert() -> None:
    bundle = {
        "observations": [{"STATE_NAME": "RUNNING", "RW_Tank_Level": 400}],
        "decisions": [
            {
                "role": "defender",
                "time_sec": 9.0,
                "decision_type": "action",
                "diagnosis": "raw_water_low_or_spoof",
                "action": {"type": "hmi_stop"},
            }
        ],
        "alerts": [],
        "executed_actions": [
            {"role": "attacker", "time_sec": 8.0, "action": {"type": "set_sensor"}},
            {"role": "defender", "time_sec": 9.0, "action": {"type": "hmi_stop"}},
        ],
        "passed": True,
    }
    scenario = {
        "metadata": {
            "ground_truth": {
                "incident_start_time_sec": 8.0,
                "expected_diagnosis": "raw_water_low_or_spoof",
                "expected_defender_actions": ["hmi_stop"],
                "mitigation_actions": ["hmi_stop"],
            },
            "safety_bounds": {
                "RW_Tank_Level": {"min": 0, "max": 1200},
            },
        }
    }

    metrics = compute_metrics(bundle, scenario)

    assert metrics["time_to_detection_sec"] == 1.0
    assert metrics["defender_detection"]["time_to_detection_sec"] == 1.0
    assert metrics["true_positive_count"] == 0
    assert metrics["defender_intervention"]["mitigation_action_count"] == 1
    assert metrics["defender_recognition"]["time_to_recognition_sec"] == 1.0


def test_compute_metrics_does_not_credit_diagnosis_only_note_as_defender_detection() -> None:
    bundle = {
        "observations": [{"STATE_NAME": "RUNNING", "RW_Tank_Level": 400}],
        "decisions": [
            {
                "role": "defender",
                "time_sec": 9.0,
                "decision_type": "note",
                "diagnosis": "raw_water_low_or_spoof",
            }
        ],
        "alerts": [],
        "executed_actions": [
            {"role": "attacker", "time_sec": 8.0, "action": {"type": "set_sensor"}}
        ],
        "passed": True,
    }
    scenario = {
        "metadata": {
            "ground_truth": {
                "incident_start_time_sec": 8.0,
                "expected_diagnosis": "raw_water_low_or_spoof",
                "expected_defender_actions": ["hmi_stop"],
                "mitigation_actions": ["hmi_stop"],
            },
            "safety_bounds": {
                "RW_Tank_Level": {"min": 0, "max": 1200},
            },
        }
    }

    metrics = compute_metrics(bundle, scenario)

    assert metrics["time_to_detection_sec"] is None
    assert metrics["defender_detection"]["time_to_detection_sec"] is None
    assert metrics["defender_detection"]["detection_rate"] == 0.0
    assert metrics["defender_recognition"]["time_to_recognition_sec"] == 1.0
    assert metrics["time_to_recognition_sec"] == 1.0
    assert metrics["defender_intervention"]["mitigation_action_count"] == 0


def test_compute_metrics_excludes_planner_alerts_from_blue_team_alert_counts() -> None:
    bundle = {
        "observations": [{"STATE_NAME": "RUNNING", "RW_Tank_Level": 400}],
        "decisions": [
            {
                "role": "observer",
                "time_sec": 8.5,
                "decision_type": "alert",
                "diagnosis": "raw_water_low_or_spoof",
            },
            {
                "role": "planner",
                "time_sec": 8.6,
                "decision_type": "alert",
                "diagnosis": "raw_water_low_or_spoof",
            },
        ],
        "alerts": [
            {"role": "observer", "time_sec": 8.5},
            {"role": "planner", "time_sec": 8.6},
        ],
        "executed_actions": [
            {"role": "attacker", "time_sec": 8.0, "action": {"type": "set_sensor"}}
        ],
        "passed": True,
    }
    scenario = {
        "metadata": {
            "ground_truth": {
                "incident_start_time_sec": 8.0,
                "expected_diagnosis": "raw_water_low_or_spoof",
                "expected_defender_actions": ["hmi_stop"],
                "mitigation_actions": ["hmi_stop"],
            },
            "safety_bounds": {
                "RW_Tank_Level": {"min": 0, "max": 1200},
            },
        }
    }

    metrics = compute_metrics(bundle, scenario)

    assert metrics["alert_count"] == 1
    assert metrics["blue_team_alert_count"] == 1
    assert metrics["planner_alert_count"] == 1
    assert metrics["observer_detection"]["alert_count"] == 1
