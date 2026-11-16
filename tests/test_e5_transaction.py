from __future__ import annotations

import json

import pytest

from experiments.scadaver_sphere.e5_evaluation import evaluate_e5
from experiments.scadaver_sphere.e5_fixture import (
    ControlledProcessFixture,
    FixtureController,
    FixtureFaults,
)
from experiments.scadaver_sphere.e5_runner import write_final_artifact
from experiments.scadaver_sphere.e5_transaction import (
    DurableEvidenceStore,
    E5Transaction,
    TransactionFailure,
    TransactionPolicy,
)


def execute(tmp_path, *, faults=None, mode="operational_recovery",
            authorized=True, require_effect=True, fail_hook=None):
    process = ControlledProcessFixture(faults)
    controller = FixtureController(process)
    store = DurableEvidenceStore(tmp_path, fail_hook=fail_hook)
    stops = []
    transaction = E5Transaction(
        process,
        store,
        TransactionPolicy(
            restoration_mode=mode,
            max_duration_seconds=2,
            require_physical_effect=require_effect,
            observation_interval_seconds=1,
        ),
        controller=controller,
        emergency_stop=stops.append,
        sleeper=process.advance_time,
    )
    try:
        result = transaction.execute(
            active=True,
            duration_seconds=1.25,
            authorized=authorized,
            request={"action": "bounded_inlet_test", "parameters": {"active": True, "duration_seconds": 1.25}},
        )
        return result, process, controller, stops, store
    except TransactionFailure as exc:
        return exc.result, process, controller, stops, store


def assert_durable(store):
    assert store.events_path.exists()
    events = [json.loads(line) for line in store.events_path.read_text().splitlines()]
    assert events
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))


def test_successful_bounded_action_and_operational_recovery(tmp_path):
    result, process, controller, stops, store = execute(tmp_path)
    assert result["status"] == "completed"
    assert result["action_applied"] and result["physical_response_observed"]
    assert result["safe_operational_recovery"] and result["restoration_verified"]
    assert controller.active and process.inlet is False and not stops
    assert_durable(store)


def test_normal_process_drift_is_not_exact_restoration(tmp_path):
    result, *_ = execute(tmp_path)
    assert result["during"]["level_raw"] != result["before"]["level_raw"]
    assert result["exact_state_restored"] is False
    assert result["safe_operational_recovery"] is True


def test_exact_restoration_uses_coordinated_pause(tmp_path):
    result, process, controller, stops, store = execute(tmp_path, mode="exact_snapshot")
    assert result["status"] == "completed"
    assert result["exact_state_restored"] is True
    assert result["restored"] == result["before"]
    assert process.paused is False and controller.active and not stops
    assert_durable(store)


def test_operational_recovery_returns_to_safe_closed_loop(tmp_path):
    result, process, controller, *_ = execute(tmp_path)
    assert result["recovery_observations"][-1]["scan_counter"] > result["recovery_observations"][0]["scan_counter"]
    assert 0 <= process.level <= 1000 and controller.health()["passed"]


def test_controller_update_during_restore_fails_closed(tmp_path):
    result, _, controller, stops, store = execute(
        tmp_path, faults=FixtureFaults(controller_overwrite_on_restore=True)
    )
    assert result["status"] == "failed"
    assert result["classification"] == "restoration_failure"
    assert result["emergency_stop_active"] and stops and controller.active
    assert_durable(store)


def test_action_accepted_without_physical_effect_is_distinct(tmp_path):
    result, _, controller, stops, store = execute(
        tmp_path, faults=FixtureFaults(no_physical_effect=True)
    )
    assert result["action_applied"] is True
    assert result["physical_response_observed"] is False
    assert result["classification"] == "applied_without_physical_effect"
    assert controller.active and stops
    assert_durable(store)


def test_action_rejection_is_pre_action_failure(tmp_path):
    result, process, controller, stops, store = execute(
        tmp_path, faults=FixtureFaults(reject_action=True)
    )
    assert result["classification"] == "pre_action_failure"
    assert result["action_applied"] is False
    assert process.inlet is False and controller.active and not stops
    assert_durable(store)


def test_action_timeout_is_classified(tmp_path):
    result, _, controller, stops, store = execute(
        tmp_path, faults=FixtureFaults(action_timeout=True)
    )
    assert result["classification"] == "timeout"
    assert controller.active and not stops
    assert_durable(store)


def test_worker_interruption_preserves_baseline_and_recovers(tmp_path):
    result, process, controller, stops, store = execute(
        tmp_path, faults=FixtureFaults(interrupt_after_apply=True)
    )
    assert result["status"] == "failed" and "before" in result
    assert process.inlet is False and controller.active and stops
    assert_durable(store)


def test_controller_restart_failure_activates_stop(tmp_path):
    result, _, controller, stops, store = execute(
        tmp_path, faults=FixtureFaults(controller_restart_failure=True)
    )
    assert result["status"] == "failed"
    assert result["recovery_error"] == "OSError"
    assert not controller.active and stops
    assert_durable(store)


def test_snapshot_persistence_failure_prevents_action(tmp_path):
    failed = {"done": False}
    def hook(phase, status):
        if phase == "capture_baseline" and status == "authoritative" and not failed["done"]:
            failed["done"] = True
            raise OSError("disk unavailable")
    result, process, controller, stops, store = execute(tmp_path, fail_hook=hook)
    assert result["action_applied"] is False
    assert process.action_calls == 0 and controller.active and not stops
    assert_durable(store)


def test_restoration_failure_activates_stop_and_keeps_evidence(tmp_path):
    result, _, controller, stops, store = execute(
        tmp_path, faults=FixtureFaults(restoration_failure=True)
    )
    assert result["classification"] == "restoration_failure"
    assert result["emergency_stop_active"] and controller.active and stops
    assert result["before"] and result["acknowledged"] and result["during"]
    assert_durable(store)


def test_authorization_rejection_executes_nothing(tmp_path):
    result, process, controller, stops, store = execute(tmp_path, authorized=False)
    assert result["classification"] == "authorization_rejected"
    assert process.action_calls == 0 and controller.quiesce_calls == 0 and not stops
    assert_durable(store)


def test_evaluator_failure_is_persisted_without_changing_transaction(tmp_path):
    result, *_ = execute(tmp_path / "tx")
    def broken(_result): raise RuntimeError("evaluator failed")
    document = write_final_artifact(tmp_path / "tx", result, evaluator=broken)
    assert document["evaluation_status"] == "failed"
    assert json.loads((tmp_path / "tx/result.json").read_text())["evaluation"]["error"] == "RuntimeError"


def test_interrupted_final_result_write_leaves_recoverable_journal(tmp_path):
    def hook(phase, status):
        if phase == "finalize": raise OSError("result filesystem interrupted")
    result, _, controller, stops, store = execute(tmp_path, fail_hook=hook)
    assert result["persistence_error"] == "OSError"
    assert controller.active and stops
    phases = [json.loads(line)["phase"] for line in store.events_path.read_text().splitlines()]
    assert "evaluate" in phases
    fallback = json.loads(store.failure_path.read_text())
    assert fallback["complete"] is True
    assert fallback["canonical_finalize_failed"] is True
    assert json.loads(store.state_path.read_text())["complete"] is True


def test_twenty_sequential_transactions_are_isolated(tmp_path):
    ids = set()
    for index in range(20):
        result, process, controller, stops, store = execute(tmp_path / f"run-{index:02d}")
        assert result["status"] == "completed"
        assert process.inlet is False and controller.active and not stops
        assert json.loads(store.state_path.read_text())["complete"] is True
        ids.add(store.transaction_id)
    assert len(ids) == 20


def test_finalize_projects_transaction_identity_to_caller_result(tmp_path):
    result, _, _, _, store = execute(tmp_path)
    durable = json.loads(store.result_path.read_text())
    assert result["transaction_id"] == store.transaction_id
    assert result["complete"] is True
    assert durable["transaction_id"] == result["transaction_id"]
    assert durable["complete"] is True


def test_independent_evaluator_distinguishes_all_outcomes(tmp_path):
    result, *_ = execute(tmp_path)
    assessment = evaluate_e5(result)
    assert assessment["action_requested"]
    assert assessment["action_authorized"]
    assert assessment["action_accepted"]
    assert assessment["action_applied"]
    assert assessment["physical_response_observed"]
    assert assessment["physical_objective_achieved"]
    assert assessment["safe_operational_recovery"]
    assert assessment["restoration_verified"]
    assert assessment["safe"]


def test_toggle_is_resolved_from_durable_post_quiescence_baseline(tmp_path):
    process = ControlledProcessFixture(); process.inlet = True; process.flow = 120
    controller = FixtureController(process); store = DurableEvidenceStore(tmp_path)
    transaction = E5Transaction(
        process, store,
        TransactionPolicy(restoration_mode="exact_snapshot", max_duration_seconds=2),
        controller=controller, sleeper=process.advance_time,
    )
    result = transaction.execute(
        active="toggle", duration_seconds=1.25, authorized=True,
        request={"action": "bounded_inlet_test", "parameters": {"active": "toggle", "duration_seconds": 1.25}},
    )
    assert result["before"]["inlet"] is True
    assert result["acknowledged"]["inlet"] is False


def _causal_result(*, before_inlet=False, target=True, before_level=450,
                   during_level=462, before_scan=10, during_scan=11,
                   before_time=1.0, acknowledge_time=1.1, during_time=2.0,
                   flow_before=0, flow_during=120):
    return {
        "action_requested": True, "action_authorized": True,
        "action_accepted": True, "action_applied": True,
        "resolved_active": target,
        "before": {"inlet": before_inlet, "level_raw": before_level,
                   "flow_raw": flow_before, "scan_counter": before_scan,
                   "time": before_time},
        "acknowledged": {"inlet": target, "level_raw": before_level,
                         "flow_raw": flow_during, "scan_counter": before_scan,
                         "time": acknowledge_time},
        "during": {"inlet": target, "level_raw": during_level,
                   "flow_raw": flow_during, "scan_counter": during_scan,
                   "time": during_time},
        "restored": {"inlet": before_inlet, "level_raw": during_level,
                     "flow_raw": flow_before, "scan_counter": during_scan,
                     "time": during_time + .1},
        "safe_operational_recovery": True, "restoration_verified": True,
        "recovery_observations": [
            {"level_raw": during_level, "scan_counter": during_scan},
            {"level_raw": during_level - 1, "scan_counter": during_scan + 1},
        ],
        "controller_health": {"passed": True},
    }


@pytest.mark.parametrize(
    "changes,expected,reason",
    [
        ({}, "OBJECTIVE_ACHIEVED", "intervention_window_direction_and_magnitude_supported"),
        ({"before_inlet": True, "target": False, "flow_before": 120,
          "flow_during": 0, "during_level": 443}, "OBJECTIVE_NOT_ACHIEVED",
         "response_direction_or_magnitude_failed"),
        ({"target": False, "during_level": 443, "flow_during": 0},
         "OBJECTIVE_NOT_ACHIEVED", "requested_state_already_active"),
        ({"before_inlet": True, "target": True, "flow_before": 120},
         "OBJECTIVE_NOT_ACHIEVED", "requested_state_already_active"),
        ({"during_level": 443}, "OBJECTIVE_NOT_ACHIEVED",
         "response_direction_or_magnitude_failed"),
        ({"during_scan": 10}, "INDETERMINATE",
         "insufficient_or_ambiguous_intervention_window"),
        ({"acknowledge_time": 2.1, "during_time": 2.0}, "INDETERMINATE",
         "insufficient_or_ambiguous_intervention_window"),
        ({"during_level": 459}, "OBJECTIVE_NOT_ACHIEVED",
         "response_direction_or_magnitude_failed"),
    ],
)
def test_causal_attribution_cases(changes, expected, reason):
    result = _causal_result(**changes)
    assessment = evaluate_e5(
        result, {"minimum_level_increase_raw": 10, "minimum_scan_advance": 1})
    assert assessment["causal_attribution"] == expected
    assert assessment["objective_measurement"]["reason"] == reason


def test_causal_attribution_missing_sample_is_indeterminate():
    result = _causal_result()
    del result["during"]["scan_counter"]
    assessment = evaluate_e5(
        result, {"minimum_level_increase_raw": 10, "minimum_scan_advance": 1})
    assert assessment["causal_attribution"] == "INDETERMINATE"


def test_command_acknowledged_without_actuator_change_is_not_impact():
    result = _causal_result(target=False, during_level=462, flow_during=0)
    assessment = evaluate_e5(
        result, {"minimum_level_increase_raw": 10, "minimum_scan_advance": 1})
    assert assessment["action_applied"] is True
    assert assessment["actuator_state_changed"] is False
    assert assessment["physical_objective_achieved"] is False


def test_pre_action_level_change_is_not_attributed_to_intervention():
    result = _causal_result(target=False, before_level=455, during_level=448,
                            flow_before=120, flow_during=0)
    result["pre_authorization"] = {"inlet": True, "level_raw": 443,
                                   "scan_counter": 9, "time": .5}
    assessment = evaluate_e5(
        result, {"minimum_level_increase_raw": 10, "minimum_scan_advance": 1})
    assert assessment["objective_measurement"]["level_increase_raw"] == -7
    assert assessment["physical_objective_achieved"] is False


def test_twenty_repeated_causal_attribution_evaluations_are_deterministic():
    outcomes = []
    for index in range(20):
        result = (_causal_result() if index % 2 == 0 else
                  _causal_result(target=False, during_level=443, flow_during=0))
        assessment = evaluate_e5(
            result, {"minimum_level_increase_raw": 10, "minimum_scan_advance": 1})
        outcomes.append(assessment["causal_attribution"])
    assert outcomes == [
        "OBJECTIVE_ACHIEVED" if index % 2 == 0 else "OBJECTIVE_NOT_ACHIEVED"
        for index in range(20)
    ]
