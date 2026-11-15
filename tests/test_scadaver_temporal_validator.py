from __future__ import annotations

import random

from experiments.scadaver_sphere.e5_fixture import ControlledProcessFixture
from experiments.scadaver_sphere.heterogeneous_sitl_v1.deployment.validate_heterogeneous_sitl import (
    CONSISTENT,
    INCONSISTENT,
    INDETERMINATE,
    classify_process_response,
)


def sample(scan, level, flow, inlet, *, time=None):
    return {
        "time": float(scan if time is None else time),
        "controller_input_level": level,
        "simulator_level": level,
        "simulator_flow": flow,
        "simulator_scan_counter": scan,
        "simulator_inlet": inlet,
    }


def classify(values, minimum=1, tolerance=1):
    return classify_process_response(
        values, minimum_confirmed=minimum, level_tolerance=tolerance
    )


def test_stable_inlet_on():
    values = [sample(i, 500 + 12 * i, 120, True) for i in range(5)]
    assert classify(values, 3)["outcome"] == CONSISTENT


def test_stable_inlet_off():
    values = [sample(i, 500 - 7 * i, 0, False) for i in range(5)]
    assert classify(values, 3)["outcome"] == CONSISTENT


def test_switch_immediately_before_scan_uses_completed_scan_flow():
    values = [sample(1, 500, 0, False), sample(2, 512, 120, True)]
    result = classify(values)
    assert result["outcome"] == CONSISTENT
    assert result["intervals"][0]["coil_transition_observed"] is True


def test_switch_immediately_after_scan_does_not_rewrite_scan_history():
    values = [sample(1, 500, 0, False), sample(2, 493, 0, True)]
    assert classify(values)["outcome"] == CONSISTENT


def test_multiple_unobserved_transitions_are_indeterminate():
    values = [sample(1, 500, 0, False), sample(4, 510, 120, True)]
    assert classify(values)["outcome"] == INDETERMINATE


def test_delayed_or_missing_response_is_inconsistent_for_atomic_model():
    values = [sample(1, 500, 0, False), sample(2, 500, 120, True)]
    assert classify(values)["outcome"] == INCONSISTENT


def test_measurement_jitter_within_explicit_tolerance():
    values = [sample(1, 500, 0, False), sample(2, 511, 120, True)]
    assert classify(values, tolerance=1)["outcome"] == CONSISTENT


def test_measurement_jitter_outside_tolerance_is_detected():
    values = [sample(1, 500, 0, False), sample(2, 509, 120, True)]
    assert classify(values, tolerance=1)["outcome"] == INCONSISTENT


def test_missing_sample_and_scan_gap_are_indeterminate():
    values = [sample(1, 500, 0, False), {"time": 2, "status": "read_error"}, sample(3, 505, 120, True)]
    result = classify(values)
    assert result["outcome"] == INDETERMINATE
    assert result["missing_samples"] == 1


def test_non_monotonic_delayed_sample_is_indeterminate():
    values = [sample(1, 500, 0, False, time=2), sample(2, 493, 0, False, time=1)]
    assert classify(values)["outcome"] == INDETERMINATE


def test_genuine_process_inconsistency_wins_over_consistent_intervals():
    values = [
        sample(1, 500, 0, False),
        sample(2, 493, 0, False),
        sample(3, 505, 120, True),
        sample(4, 530, 120, True),
    ]
    assert classify(values, 2)["outcome"] == INCONSISTENT


def test_normal_closed_loop_oscillation():
    values = [
        sample(1, 450, 0, False),
        sample(2, 443, 0, False),
        sample(3, 455, 120, True),
        sample(4, 448, 0, False),
        sample(5, 460, 120, True),
        sample(6, 453, 0, False),
    ]
    result = classify(values, 4)
    assert result["outcome"] == CONSISTENT
    assert result["confirmed_intervals"] == 5


def test_exact_aborted_deployment_trace_is_consistent():
    raw = [
        (16234, 450, 0, False), (16235, 443, 0, False),
        (16236, 455, 120, True), (16237, 448, 0, False),
        (16238, 460, 120, True), (16239, 453, 0, False),
        (16240, 446, 0, False), (16241, 458, 120, True),
        (16242, 451, 0, False), (16243, 444, 0, False),
        (16244, 456, 120, True), (16245, 449, 0, False),
    ]
    values = [sample(scan, level, flow, inlet, time=index) for index, (scan, level, flow, inlet) in enumerate(raw)]
    result = classify(values, 8)
    assert result["outcome"] == CONSISTENT
    assert result["confirmed_intervals"] == 11


def test_clamped_process_boundaries():
    assert classify([sample(1, 999, 120, True), sample(2, 1000, 120, True)])["outcome"] == CONSISTENT
    assert classify([sample(1, 3, 0, False), sample(2, 0, 0, False)])["outcome"] == CONSISTENT


def test_twenty_deterministic_switching_runs():
    for seed in range(20):
        rng = random.Random(seed)
        process = ControlledProcessFixture()
        values = [sample(0, process.level, process.flow, process.inlet)]
        for scan in range(1, 13):
            inlet = rng.choice([False, True])
            process.inlet = inlet
            process.advance_scan()
            # Current coil can already reflect the next controller decision;
            # applied flow remains authoritative for this completed scan.
            observed_coil = rng.choice([inlet, not inlet])
            values.append(sample(
                process.scan_counter, process.level, process.flow, observed_coil
            ))
        result = classify(values, 8)
        assert result["outcome"] == CONSISTENT, (seed, result)
