from __future__ import annotations

import pytest

from ics_agent.operator_handoffs import (
    EvidenceSample,
    HandoffTracker,
    specs_from_mapping,
    validate_contract_tags,
)
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
