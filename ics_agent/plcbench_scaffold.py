"""PLCBench-style decision scaffold pieces (pure, backend-free).

Two things distinguish the `plcbench_style_matched` baseline from the existing
ICS-agent agentic condition while holding world-facing capability fixed (same
candidate menu, observations, knowledge, action grants, and LLM backend):

1. A three-part prompt -- SYSTEM / TASK / ENVIRONMENT -- in the spirit of the
   source's prompt organization (arXiv:2608.26882v1 Section 3.1.1, Appendix F),
   instead of the flat context payload.
2. Deterministic history compaction under a token budget (Section 3.1.3,
   Algorithm 1): when the calibrated token estimate exceeds a soft budget Bs the
   older prefix is compacted progressively while a recent-K suffix is protected,
   down to a target budget Bt.

Both functions here are pure and import nothing heavy, so they are unit-testable
without a GPU/model. The exact Bs/Bt/K are not published by the source; ours are
configured in the profile and recorded in docs/baselines/plcbench_conformance.md.
Compaction is deterministic: identical input yields identical output.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def estimate_tokens(text: str) -> int:
    """Deterministic, model-free token estimate (~4 chars/token, ceil)."""
    return math.ceil(len(text) / 4) if text else 0


@dataclass
class HistoryEntry:
    """One compactable history item rendered to text, tagged by kind."""

    kind: str
    text: str


@dataclass
class CompactionResult:
    protected: list[HistoryEntry]        # recent-K suffix, kept verbatim
    prefix_lines: list[str]              # compacted older prefix (may be empty)
    levels_applied: int                  # 0 = none, up to 4
    omitted: int                         # count folded into a rollup/omission line
    tokens: int                          # final estimated token total

    def render(self) -> str:
        lines = list(self.prefix_lines)
        lines.extend(f"[{e.kind}] {e.text}" for e in self.protected)
        return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _kind_counts(entries: Sequence[HistoryEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    return counts


def _level_render(prefix: Sequence[HistoryEntry], level: int) -> tuple[list[str], int]:
    """Render the older prefix at a given compaction level. Returns (lines, omitted)."""
    if not prefix:
        return [], 0
    if level <= 0:
        return [f"[{e.kind}] {e.text}" for e in prefix], 0
    if level == 1:  # terse: one short line per entry
        return [f"[{e.kind}] {_truncate(e.text, 80)}" for e in prefix], 0
    if level == 2:  # coalesce runs of the same kind, keep the latest text of each run
        lines: list[str] = []
        i = 0
        while i < len(prefix):
            j = i
            while j + 1 < len(prefix) and prefix[j + 1].kind == prefix[i].kind:
                j += 1
            run = prefix[i : j + 1]
            if len(run) == 1:
                lines.append(f"[{run[0].kind}] {_truncate(run[0].text, 60)}")
            else:
                lines.append(
                    f"[{run[0].kind} x{len(run)}] latest: {_truncate(run[-1].text, 60)}"
                )
            i = j + 1
        return lines, 0
    if level == 3:  # single rollup line with per-kind counts
        counts = _kind_counts(prefix)
        summary = ", ".join(f"{k}={n}" for k, n in sorted(counts.items()))
        return [f"[earlier {len(prefix)} steps: {summary}]"], len(prefix)
    # level >= 4: drop the prefix entirely
    return [f"[{len(prefix)} earlier steps omitted]"], len(prefix)


def compact_history(
    entries: Sequence[HistoryEntry],
    *,
    soft_budget: int,
    target_budget: int,
    protected_k: int,
) -> CompactionResult:
    """Compact history to fit a token budget, protecting the recent-K suffix.

    If the full rendering already fits ``soft_budget`` nothing is compacted. Else
    the prefix (everything but the last ``protected_k`` entries) is compacted
    through increasing levels until the total estimate is at or below
    ``target_budget`` (or the most aggressive level is reached).
    """
    if target_budget > soft_budget:
        raise ValueError("target_budget must be <= soft_budget")
    protected_k = max(0, protected_k)
    entries = list(entries)

    full = CompactionResult(
        protected=entries,
        prefix_lines=[],
        levels_applied=0,
        omitted=0,
        tokens=estimate_tokens("\n".join(f"[{e.kind}] {e.text}" for e in entries)),
    )
    if full.tokens <= soft_budget:
        return full

    split = max(0, len(entries) - protected_k)
    prefix, protected = entries[:split], entries[split:]
    protected_text = "\n".join(f"[{e.kind}] {e.text}" for e in protected)
    protected_tokens = estimate_tokens(protected_text)

    best: CompactionResult | None = None
    for level in range(1, 5):
        lines, omitted = _level_render(prefix, level)
        tokens = protected_tokens + estimate_tokens("\n".join(lines))
        best = CompactionResult(
            protected=protected,
            prefix_lines=lines,
            levels_applied=level,
            omitted=omitted,
            tokens=tokens,
        )
        if tokens <= target_budget:
            break
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Three-part prompt
# ---------------------------------------------------------------------------
# Response contract -- kept identical to the agentic engine's so the matched
# baseline selects from the SAME candidate menu with the SAME output schema
# (capability parity). Mirror of CachedQwenDecisionEngine._format_prompt's
# footer in agents/agentic.py; tests assert the required keys/rules are present.
RESPONSE_CONTRACT = (
    "Return exactly one JSON object and no prose before or after it.\n"
    "Required keys: summary, candidate_id, confidence, diagnosis, labels, decision_type.\n"
    "Rules:\n"
    "- candidate_id must be either null or match one of candidate_actions[].id.\n"
    "- Never invent a new action or a new candidate id.\n"
    "- Use decision_type of note, alert, action, or analysis.\n"
    "- diagnosis must be null or a short snake_case style label, never a sentence.\n"
    "- summary must stay under 18 words and should not restate the full prompt context.\n"
    "- If the evidence is weak, set diagnosis to null instead of guessing.\n"
    "- If you want to hold position, prefer the noop candidate when available.\n\n"
    "Example JSON:\n"
    '{"summary":"Monitor and hold.","candidate_id":null,"confidence":0.42,'
    '"diagnosis":null,"labels":["agentic"],"decision_type":"note"}'
)

_HISTORY_FIELDS: tuple[tuple[str, str], ...] = (
    ("recent_observations", "obs"),
    ("recent_actions", "action"),
    ("recent_action_effects", "effect"),
    ("recent_alerts", "alert"),
    ("recent_decisions", "decision"),
)


def _history_entries(context: Mapping[str, Any]) -> list[HistoryEntry]:
    """Flatten the context's history streams into ordered compactable entries.

    Entries keep each stream's own order; streams are interleaved oldest-first by
    the order of `_HISTORY_FIELDS` within a step (there is no global cross-stream
    timestamp in the flat context, so this ordering is fixed and deterministic).
    """
    entries: list[HistoryEntry] = []
    for field, kind in _HISTORY_FIELDS:
        for item in context.get(field, []) or []:
            entries.append(HistoryEntry(kind=kind, text=_render_item(item)))
    sm = context.get("structured_memory")
    if sm:
        entries.append(HistoryEntry(kind="memory", text=_render_item(sm)))
    return entries


def _render_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    try:
        return json.dumps(item, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(item)


def build_matched_prompt(
    context: Mapping[str, Any],
    *,
    system_prompt: str,
    operational_knowledge: str = "",
    soft_budget: int,
    target_budget: int,
    protected_k: int,
) -> tuple[str, CompactionResult]:
    """Assemble the SYSTEM/TASK/ENVIRONMENT/HISTORY prompt + response contract.

    Capabilities (candidate_actions, observation) are taken verbatim from the
    same context the agentic agent builds; only the organization and the
    history compaction differ. Returns the prompt and the compaction result (for
    manifest/telemetry).
    """
    system = system_prompt.strip() or (
        "You are an ICS role agent. Reason carefully and answer with strict JSON only."
    )
    knowledge_block = (
        f"\n\nOperational manuals:\n{operational_knowledge.strip()}"
        if operational_knowledge.strip()
        else ""
    )

    task = {
        "role": context.get("role"),
        "scenario_id": context.get("scenario_id"),
        "intent": context.get("intent"),
        "process_id": context.get("process_id"),
        "role_goal": context.get("role_goal"),
        "role_constraints": context.get("role_constraints", []),
        "expected_outcomes": context.get("expected_outcomes", []),
        "safety_bounds": context.get("safety_bounds", {}),
    }
    environment = {
        "time_sec": context.get("time_sec"),
        "state_name": context.get("state_name"),
        "observation": context.get("observation"),
        "active_alarms": context.get("active_alarms", []),
        "active_states": context.get("active_states", []),
        "active_commands": context.get("active_commands", []),
        "short_horizon_trends": context.get("short_horizon_trends", {}),
        "candidate_actions": context.get("candidate_actions", []),
    }

    compaction = compact_history(
        _history_entries(context),
        soft_budget=soft_budget,
        target_budget=target_budget,
        protected_k=protected_k,
    )

    prompt = (
        f"[SYSTEM]\n{system}{knowledge_block}\n\n"
        f"[TASK]\n{json.dumps(task, indent=2)}\n\n"
        f"[ENVIRONMENT]\n{json.dumps(environment, indent=2)}\n\n"
        f"[HISTORY]\n{compaction.render() or '(none)'}\n\n"
        f"[RESPONSE]\n{RESPONSE_CONTRACT}"
    )
    return prompt, compaction
