from __future__ import annotations

import json

import pytest

from ics_agent.operator_handoffs import (
    EvidenceSample,
    HandoffTracker,
    specs_from_mapping,
    validate_contract_tags,
)
from ics_agent.operator_handoffs_replay import replay
from ics_agent.paths import REPO_ROOT
from ics_agent.utils import load_yaml


MAP_DIR = REPO_ROOT / "configs" / "ctf" / "operator_handoffs"


def _mapping(name: str) -> dict:
    return load_yaml(MAP_DIR / f"{name}.yaml")


def _evidence(spec, at: float) -> tuple[EvidenceSample, ...]:
    return tuple(
        EvidenceSample(tag=req.tag, source=req.source, value=0, observed_at_sec=at)
        for req in spec.requirements
    )


@pytest.mark.parametrize("name", ["hydro_power", "water_distribution"])
def test_handoff_completes_only_with_current_declared_evidence(name: str) -> None:
    spec = specs_from_mapping(_mapping(name))[0]
    tracker = HandoffTracker((spec,))
    request_id = tracker.request(spec.id, by=spec.requester, at=10)

    assert tracker.assess(request_id, at=11).status == "requested"
    assert tracker.complete(request_id, by=spec.responder, at=11, evidence=_evidence(spec, 11)).status == "not_acknowledged"
    assert tracker.acknowledge(request_id, by=spec.responder, at=12).status == "acknowledged"
    with pytest.raises(ValueError, match="acknowledgement"):
        tracker.complete(request_id, by=spec.responder, at=11, evidence=_evidence(spec, 11))

    stale = tracker.complete(request_id, by=spec.responder, at=20, evidence=_evidence(spec, 10))
    assert stale.status == "rejected"
    assert f"stale {spec.requirements[0].tag}" in stale.reasons
    wrong = list(_evidence(spec, 20))
    wrong[0] = EvidenceSample(wrong[0].tag, "hmi_echo", 0, 20)
    assert tracker.complete(request_id, by=spec.responder, at=20, evidence=tuple(wrong)).status == "rejected"

    assert tracker.complete(request_id, by=spec.responder, at=21, evidence=_evidence(spec, 21)).status == "completed"
    assert tracker.assess(request_id, at=25).status == "completed"
    assert tracker.assess(request_id, at=27).status == "stale"
    assert [event["event_type"] for event in tracker.events] == [
        "requested", "acknowledged", "completion_rejected", "completion_rejected", "completed"
    ]


def test_late_or_wrong_role_cannot_complete_handoff() -> None:
    spec = specs_from_mapping(_mapping("hydro_power"))[0]
    tracker = HandoffTracker((spec,))
    with pytest.raises(ValueError, match="requester"):
        tracker.request(spec.id, by="attacker", at=0)
    request_id = tracker.request(spec.id, by=spec.requester, at=0)
    with pytest.raises(ValueError, match="responder"):
        tracker.acknowledge(request_id, by="attacker", at=1)
    assert tracker.acknowledge(request_id, by=spec.responder, at=31).status == "expired"
    assert tracker.assess(request_id, at=31).status == "expired"
    assert tracker.complete(request_id, by=spec.responder, at=32, evidence=_evidence(spec, 32)).status == "expired"
    assert [event["event_type"] for event in tracker.events] == ["requested", "expired"]


def test_binding_validation_catches_wrong_contract_owner() -> None:
    mapping = _mapping("water_distribution")
    spec = specs_from_mapping(mapping)[0]
    tags = {
        req.tag: {"role": req.contract_role, "direction": req.contract_direction}
        for req in spec.requirements
    }
    assert validate_contract_tags(mapping, {"tags": tags}) == []
    tags[spec.requirements[0].tag]["role"] = "controller"
    assert "role is 'controller'" in validate_contract_tags(mapping, {"tags": tags})[0]


def test_handoff_has_no_actuation_method() -> None:
    spec = specs_from_mapping(_mapping("hydro_power"))[0]
    tracker = HandoffTracker((spec,))
    assert not hasattr(tracker, "act")
    assert not hasattr(tracker, "execute")


@pytest.mark.parametrize(
    ("mapping_name", "scenario_name", "recommendation_status", "final_status"),
    [
        ("hydro_power", "hydro_checked", "completed", "completed"),
        ("hydro_power", "hydro_skipped", "requested", "expired"),
        ("water_distribution", "water_distribution_echo", "acknowledged", "expired"),
    ],
)
def test_authored_replay_records_handoff_without_actuation(
    mapping_name: str, scenario_name: str, recommendation_status: str, final_status: str,
) -> None:
    scenario = load_yaml(MAP_DIR / "examples" / f"{scenario_name}.yaml")
    output = replay(_mapping(mapping_name), scenario)

    assert output["mode"] == "synthetic_authored"
    assert output["scenario"] == scenario_name
    assert len(output["input_sha256"]) == 64
    assert output["input_sha256"] == replay(_mapping(mapping_name), scenario)["input_sha256"]
    recommendation = next(item for item in output["events"] if item["event_type"] == "recommendation")
    assert recommendation["handoff_status"] == recommendation_status
    assert recommendation["verification_state"] == (
        "verified_current" if recommendation_status == "completed" else "unverified"
    )
    assert next(iter(output["final_handoffs"].values())) == final_status
    assert all(item["event_type"] not in {"actuation", "controller_write"} for item in output["events"])
    json.dumps(output)


def test_rejected_hmi_echo_retains_its_source_and_value() -> None:
    scenario = load_yaml(MAP_DIR / "examples" / "water_distribution_echo.yaml")
    output = replay(_mapping("water_distribution"), scenario)
    rejected = next(item for item in output["events"] if item["event_type"] == "completion_rejected")
    assert "wrong source for Grid_Elev_Valve_Sts" in rejected["reasons"]
    assert rejected["evidence"][0]["source"] == "hmi_echo"
    assert rejected["evidence"][0]["value"] is True
    recommendation = next(item for item in output["events"] if item["event_type"] == "recommendation")
    assert "wrong source for Grid_Elev_Valve_Sts" in recommendation["verification_reasons"]


def test_replay_rejects_unlabeled_or_out_of_order_traces() -> None:
    scenario = load_yaml(MAP_DIR / "examples" / "hydro_checked.yaml")
    scenario["mode"] = "live"
    with pytest.raises(ValueError, match="synthetic_authored"):
        replay(_mapping("hydro_power"), scenario)

    scenario["mode"] = "synthetic_authored"
    scenario["steps"][1]["at_sec"] = -1
    with pytest.raises(ValueError, match="nondecreasing"):
        replay(_mapping("hydro_power"), scenario)


def test_replay_digest_changes_with_authored_evidence() -> None:
    scenario = load_yaml(MAP_DIR / "examples" / "hydro_checked.yaml")
    original = replay(_mapping("hydro_power"), scenario)["input_sha256"]
    scenario["steps"][2]["evidence"][0]["value"] = 1
    assert replay(_mapping("hydro_power"), scenario)["input_sha256"] != original


def test_only_declared_requester_can_record_recommendation() -> None:
    spec = specs_from_mapping(_mapping("hydro_power"))[0]
    tracker = HandoffTracker((spec,))
    request_id = tracker.request(spec.id, by=spec.requester, at=0)
    with pytest.raises(ValueError, match="requester"):
        tracker.record_recommendation(
            request_id, by=spec.responder, at=1,
            choice="recommend_operation", basis="test",
        )
    assert tracker.record_recommendation(
        request_id, by=spec.requester, at=1,
        choice="recommend_operation", basis="simulated omitted check",
    ).status == "requested"
