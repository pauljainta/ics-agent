"""Evidence handoffs for simulated operator roles; never actuates a process."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class EvidenceRequirement:
    tag: str
    source: str
    max_age_sec: float
    contract_role: str
    contract_direction: str = "sensor"


@dataclass(frozen=True)
class HandoffSpec:
    id: str
    requester: str
    responder: str
    deadline_sec: float
    requirements: tuple[EvidenceRequirement, ...]


@dataclass(frozen=True)
class EvidenceSample:
    tag: str
    source: str
    value: Any
    observed_at_sec: float


@dataclass(frozen=True)
class HandoffAssessment:
    status: str
    reasons: tuple[str, ...] = ()


@dataclass
class _Handoff:
    spec: HandoffSpec
    requested_at_sec: float
    acknowledged_at_sec: float | None = None
    completed_at_sec: float | None = None
    evidence: dict[str, EvidenceSample] = field(default_factory=dict)
    expired: bool = False


def _time(value: float) -> float:
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError("handoff time must be a finite nonnegative number")
    return result


def specs_from_mapping(mapping: dict[str, Any]) -> tuple[HandoffSpec, ...]:
    roles = set(mapping.get("roles", {}))
    specs = []
    for item in mapping.get("handoffs", []):
        if item["requester"] not in roles or item["responder"] not in roles:
            raise ValueError("handoff requester and responder must be declared roles")
        requirements = tuple(
            EvidenceRequirement(
                tag=str(req["tag"]),
                source=str(req["source"]),
                max_age_sec=_time(req["max_age_sec"]),
                contract_role=str(req["contract_role"]),
                contract_direction=str(req.get("contract_direction", "sensor")),
            )
            for req in item["requirements"]
        )
        if not requirements or len({req.tag for req in requirements}) != len(requirements):
            raise ValueError("handoff requirements must be nonempty and have unique tags")
        spec = HandoffSpec(
            id=str(item["id"]),
            requester=str(item["requester"]),
            responder=str(item["responder"]),
            deadline_sec=_time(item["deadline_sec"]),
            requirements=requirements,
        )
        if spec.deadline_sec == 0:
            raise ValueError("handoff deadline must be positive")
        specs.append(spec)
    if len({spec.id for spec in specs}) != len(specs):
        raise ValueError("handoff IDs must be unique")
    return tuple(specs)


def validate_contract_tags(mapping: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    """Check declared tag ownership against a canonical sphere-usecases contract."""
    raw_tags = contract.get("tags", {})
    if isinstance(raw_tags, list):
        tags = {item["name"]: item for item in raw_tags}
    elif isinstance(raw_tags, dict):
        tags = raw_tags
    else:
        raise ValueError("canonical contract has no supported tags collection")
    errors = []
    for spec in specs_from_mapping(mapping):
        for req in spec.requirements:
            actual = tags.get(req.tag)
            if actual is None:
                errors.append(f"{spec.id}: unknown tag {req.tag}")
                continue
            for field_name, expected in (
                ("role", req.contract_role),
                ("direction", req.contract_direction),
            ):
                if actual.get(field_name) != expected:
                    errors.append(
                        f"{spec.id}: {req.tag} {field_name} is {actual.get(field_name)!r}, expected {expected!r}"
                    )
    return errors


class HandoffTracker:
    """Records request, acknowledgement, and completion without granting authority."""

    def __init__(self, specs: tuple[HandoffSpec, ...]) -> None:
        self.specs = {spec.id: spec for spec in specs}
        if len(self.specs) != len(specs):
            raise ValueError("handoff IDs must be unique")
        self.requests: dict[str, _Handoff] = {}
        self.events: list[dict[str, Any]] = []
        self._next_id = 1

    def _event(self, request_id: str, kind: str, actor: str, at: float, **detail: Any) -> None:
        self.events.append(
            {"request_id": request_id, "event_type": kind, "actor": actor,
             "time_sec": at, **detail}
        )

    def request(self, spec_id: str, *, by: str, at: float) -> str:
        now = _time(at)
        spec = self.specs[spec_id]
        if by != spec.requester:
            raise ValueError("only the declared requester may open this handoff")
        request_id = f"{spec_id}:{self._next_id}"
        self._next_id += 1
        self.requests[request_id] = _Handoff(spec=spec, requested_at_sec=now)
        self._event(request_id, "requested", by, now)
        return request_id

    def _record(self, request_id: str, now: float) -> _Handoff:
        record = self.requests[request_id]
        if now < record.requested_at_sec:
            raise ValueError("handoff event precedes its request")
        if record.acknowledged_at_sec is not None and now < record.acknowledged_at_sec:
            raise ValueError("handoff event precedes its acknowledgement")
        if record.completed_at_sec is not None and now < record.completed_at_sec:
            raise ValueError("handoff event precedes its completion")
        if (
            not record.expired
            and record.completed_at_sec is None
            and now > record.requested_at_sec + record.spec.deadline_sec
        ):
            record.expired = True
            self._event(request_id, "expired", "workflow", now)
        return record

    def acknowledge(self, request_id: str, *, by: str, at: float) -> HandoffAssessment:
        now = _time(at)
        record = self._record(request_id, now)
        if by != record.spec.responder:
            raise ValueError("only the declared responder may acknowledge")
        if record.expired:
            return HandoffAssessment("expired")
        if record.acknowledged_at_sec is not None:
            return HandoffAssessment("already_acknowledged")
        record.acknowledged_at_sec = now
        self._event(request_id, "acknowledged", by, now)
        return HandoffAssessment("acknowledged")

    def complete(
        self,
        request_id: str,
        *,
        by: str,
        at: float,
        evidence: tuple[EvidenceSample, ...],
    ) -> HandoffAssessment:
        now = _time(at)
        record = self._record(request_id, now)
        if by != record.spec.responder:
            raise ValueError("only the declared responder may complete")
        if record.expired:
            return HandoffAssessment("expired")
        if record.acknowledged_at_sec is None:
            return HandoffAssessment("not_acknowledged")
        if record.completed_at_sec is not None:
            return HandoffAssessment("already_completed")
        by_tag = {sample.tag: sample for sample in evidence}
        reasons = []
        if len(by_tag) != len(evidence):
            reasons.append("duplicate evidence tags")
        extra = set(by_tag) - {req.tag for req in record.spec.requirements}
        if extra:
            reasons.append(f"undeclared evidence: {', '.join(sorted(extra))}")
        for req in record.spec.requirements:
            sample = by_tag.get(req.tag)
            if sample is None:
                reasons.append(f"missing {req.tag}")
                continue
            if sample.source != req.source:
                reasons.append(f"wrong source for {req.tag}")
            observed_at = _time(sample.observed_at_sec)
            if observed_at > now:
                reasons.append(f"future observation for {req.tag}")
            elif now - observed_at > req.max_age_sec:
                reasons.append(f"stale {req.tag}")
        if reasons:
            self._event(
                request_id, "completion_rejected", by, now,
                reasons=reasons, evidence=[_sample_record(item) for item in evidence],
            )
            return HandoffAssessment("rejected", tuple(reasons))
        record.completed_at_sec = now
        record.evidence = by_tag
        self._event(
            request_id, "completed", by, now,
            evidence=[_sample_record(item) for item in evidence],
        )
        return HandoffAssessment("completed")

    def record_recommendation(
        self, request_id: str, *, by: str, at: float, choice: str, basis: str,
    ) -> HandoffAssessment:
        """Record a simulated operator recommendation; never issue an operation."""
        if choice not in {"recommend_operation", "hold_operation"}:
            raise ValueError("unknown operator recommendation")
        if not basis.strip():
            raise ValueError("operator recommendation needs a stated basis")
        record = self.requests[request_id]
        if by != record.spec.requester:
            raise ValueError("only the declared requester may recommend")
        now = _time(at)
        assessment = self.assess(request_id, at=now)
        verification_reasons = list(assessment.reasons)
        if assessment.status != "completed":
            previous_rejection = next(
                (
                    event for event in reversed(self.events)
                    if event["request_id"] == request_id
                    and event["event_type"] == "completion_rejected"
                    and event["time_sec"] <= now
                ),
                None,
            )
            if previous_rejection is not None:
                verification_reasons.extend(previous_rejection["reasons"])
            if not verification_reasons:
                verification_reasons.append(f"handoff {assessment.status}")
        self._event(
            request_id, "recommendation", by, now, choice=choice, basis=basis,
            handoff_status=assessment.status,
            verification_state=(
                "verified_current" if assessment.status == "completed" else "unverified"
            ),
            verification_reasons=verification_reasons,
        )
        return assessment

    def assess(self, request_id: str, *, at: float) -> HandoffAssessment:
        now = _time(at)
        record = self._record(request_id, now)
        if record.expired:
            return HandoffAssessment("expired")
        if record.completed_at_sec is None:
            status = "acknowledged" if record.acknowledged_at_sec is not None else "requested"
            return HandoffAssessment(status)
        stale = tuple(
            f"stale {req.tag}"
            for req in record.spec.requirements
            if now - record.evidence[req.tag].observed_at_sec > req.max_age_sec
        )
        return HandoffAssessment("stale" if stale else "completed", stale)


def _sample_record(sample: EvidenceSample) -> dict[str, Any]:
    return {
        "tag": sample.tag,
        "source": sample.source,
        "value": sample.value,
        "observed_at_sec": sample.observed_at_sec,
    }
