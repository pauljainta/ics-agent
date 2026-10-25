"""Aggregate scored PLCBench episodes into the paper's results table.

Objective-agnostic reporting: it consumes per-episode flag results + validity and
rolls them up per (workload, profile) into counts, flag rates, unassessable
counts, and timing. Rows trace back to episode ids so every table cell is
reproducible from the bundles (conformance Section 8, item 14). Pure/stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

FLAGS = ("discover", "read", "write", "manipulate", "disrupt", "impact")


@dataclass
class EpisodeRecord:
    """One scored episode, as read back from its run bundle."""

    episode_id: str
    workload: str          # e.g. "water_treatment"
    profile: str           # e.g. "plcbench_matched_treatment_attacker"
    flags: dict[str, bool]
    disrupt_assessable: bool = True
    impact_assessable: bool = True
    stop_reason: str = ""
    actions_run: int = 0


@dataclass
class CellSummary:
    workload: str
    profile: str
    n: int
    flag_rates: dict[str, float]
    impact_unassessable: int
    disrupt_unassessable: int
    mean_actions: float
    episode_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "profile": self.profile,
            "n": self.n,
            "flag_rates": {k: round(v, 4) for k, v in self.flag_rates.items()},
            "impact_unassessable": self.impact_unassessable,
            "disrupt_unassessable": self.disrupt_unassessable,
            "mean_actions": round(self.mean_actions, 2),
            "episode_ids": list(self.episode_ids),
        }


def aggregate(records: Iterable[EpisodeRecord]) -> list[CellSummary]:
    """Roll episodes up per (workload, profile). Deterministic ordering.

    A flag's rate is counted only over episodes where that flag is assessable, so
    an unassessable physical outcome never dilutes the rate; the unassessable
    counts are reported separately rather than folded into a pass/fail.
    """
    cells: dict[tuple[str, str], list[EpisodeRecord]] = {}
    for r in records:
        cells.setdefault((r.workload, r.profile), []).append(r)

    out: list[CellSummary] = []
    for (workload, profile), eps in sorted(cells.items()):
        n = len(eps)
        rates: dict[str, float] = {}
        for flag in FLAGS:
            if flag == "impact":
                assessable = [e for e in eps if e.impact_assessable]
            elif flag == "disrupt":
                assessable = [e for e in eps if e.disrupt_assessable]
            else:
                assessable = eps
            denom = len(assessable)
            hits = sum(1 for e in assessable if e.flags.get(flag))
            rates[flag] = (hits / denom) if denom else 0.0
        out.append(CellSummary(
            workload=workload,
            profile=profile,
            n=n,
            flag_rates=rates,
            impact_unassessable=sum(1 for e in eps if not e.impact_assessable),
            disrupt_unassessable=sum(1 for e in eps if not e.disrupt_assessable),
            mean_actions=(sum(e.actions_run for e in eps) / n) if n else 0.0,
            episode_ids=[e.episode_id for e in eps],
        ))
    return out


def render_table(cells: Sequence[CellSummary]) -> str:
    """A plain-text results table (flag rates as percentages)."""
    header = (["workload", "profile", "n"]
              + [f[:4] for f in FLAGS] + ["imp?", "actions"])
    rows = [header]
    for c in cells:
        rows.append([
            c.workload, c.profile, str(c.n),
            *[f"{100 * c.flag_rates[f]:.0f}" for f in FLAGS],
            str(c.impact_unassessable), f"{c.mean_actions:.0f}",
        ])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = []
    for ri, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if ri == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(header))))
    note = ("\nflag columns are % of assessable episodes; imp? = episodes whose "
            "impact was unassessable (coverage).")
    return "\n".join(lines) + "\n" + note


def record_from_bundle(meta: dict[str, Any], score: dict[str, Any]) -> EpisodeRecord:
    """Build an EpisodeRecord from a run bundle's meta.json + score dict."""
    physical = score.get("physical", {})
    return EpisodeRecord(
        episode_id=str(meta.get("episode_id", "")),
        workload=str(meta.get("workload", "")),
        profile=str(meta.get("profile", "")),
        flags=dict(score.get("flags", {})),
        disrupt_assessable=bool(physical.get("disrupt", {}).get("assessable", True)),
        impact_assessable=bool(physical.get("impact", {}).get("assessable", True)),
        stop_reason=str(meta.get("stop_reason", "")),
        actions_run=int(meta.get("actions_run", 0)),
    )
