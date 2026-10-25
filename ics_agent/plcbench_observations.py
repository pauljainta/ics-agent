"""Typed, timestamped read-only visual/robot/social observation contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


ALLOWED_KINDS = {"scene", "robot", "swarm", "social"}


@dataclass(frozen=True)
class ExternalObservation:
    kind: str
    source: str
    observed_at_utc: str
    payload: Mapping[str, Any]
    authority: str = "read_only"

    def __post_init__(self) -> None:
        if self.kind not in ALLOWED_KINDS or self.authority != "read_only":
            raise ValueError("external adapter has unsupported kind or authority")
        datetime.fromisoformat(self.observed_at_utc.replace("Z", "+00:00"))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def observation(kind: str, source: str, payload: Mapping[str, Any],
                *, timestamp_ms: int | None = None) -> ExternalObservation:
    stamp = (datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)
             if timestamp_ms is not None else datetime.now(timezone.utc))
    return ExternalObservation(kind, source, stamp.isoformat(), dict(payload))
