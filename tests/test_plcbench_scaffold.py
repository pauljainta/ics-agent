"""Tests for the PLCBench matched-scaffold pieces (pure; no GPU/model)."""

from __future__ import annotations

import json

from ics_agent.plcbench_scaffold import (
    RESPONSE_CONTRACT,
    HistoryEntry,
    build_matched_prompt,
    compact_history,
    estimate_tokens,
)


def _entries(n: int, *, kind: str = "action", size: int = 40) -> list[HistoryEntry]:
    return [HistoryEntry(kind=kind, text=f"{kind}-{i}-" + "x" * size) for i in range(n)]


def test_estimate_tokens_deterministic_and_monotone():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("a" * 41) >= estimate_tokens("a" * 40)


def test_no_compaction_when_under_soft_budget():
    entries = _entries(3, size=4)
    res = compact_history(entries, soft_budget=10_000, target_budget=5_000, protected_k=6)
    assert res.levels_applied == 0
    assert res.prefix_lines == []
    assert res.protected == entries  # all kept verbatim


def test_protected_suffix_kept_verbatim_under_pressure():
    entries = _entries(20, size=60)
    res = compact_history(entries, soft_budget=200, target_budget=120, protected_k=5)
    assert res.levels_applied >= 1
    # The last 5 entries are protected and appear verbatim in the render.
    assert res.protected == entries[-5:]
    rendered = res.render()
    for e in entries[-5:]:
        assert e.text in rendered


def test_compaction_is_deterministic():
    entries = _entries(30, size=50)
    kw = dict(soft_budget=150, target_budget=90, protected_k=4)
    a = compact_history(entries, **kw)
    b = compact_history(entries, **kw)
    assert a.render() == b.render()
    assert (a.levels_applied, a.omitted, a.tokens) == (b.levels_applied, b.omitted, b.tokens)


def test_more_pressure_never_increases_tokens():
    entries = _entries(40, size=60)
    loose = compact_history(entries, soft_budget=400, target_budget=300, protected_k=6)
    tight = compact_history(entries, soft_budget=120, target_budget=60, protected_k=6)
    assert tight.tokens <= loose.tokens


def test_aggressive_level_omits_prefix_and_counts_it():
    entries = _entries(25, size=80)
    res = compact_history(entries, soft_budget=60, target_budget=20, protected_k=2)
    # Only the protected suffix survives; the prefix is rolled up/omitted.
    assert res.omitted >= 1
    assert res.levels_applied >= 3
    assert res.protected == entries[-2:]


def test_target_must_not_exceed_soft():
    try:
        compact_history([], soft_budget=10, target_budget=20, protected_k=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError when target_budget > soft_budget")


# ---------------------------------------------------------------------------
# Three-part prompt
# ---------------------------------------------------------------------------
def _context():
    return {
        "role": "attacker",
        "scenario_id": "wt-x",
        "intent": "drive raw water low",
        "process_id": "WT-P1",
        "time_sec": 12.0,
        "state_name": "RUNNING",
        "observation": {"RW_Tank_Level": 300, "RW_Pump_Sts": 1},
        "candidate_actions": [
            {"id": "a1", "summary": "stop", "action": {"type": "hmi_stop"}},
            {"id": "a2", "summary": "hold", "action": {"type": "noop"}},
        ],
        "active_alarms": ["Alarm_RW_Tank_LL"],
        "recent_actions": [{"type": "hmi_start"}],
        "recent_observations": [{"RW_Tank_Level": 320}],
        "role_goal": "sustain loss of transfer",
    }


def test_matched_prompt_has_all_sections_and_capability_fields():
    prompt, _ = build_matched_prompt(
        _context(), system_prompt="SYS RULES", operational_knowledge="",
        soft_budget=4096, target_budget=3072, protected_k=6,
    )
    for marker in ("[SYSTEM]", "[TASK]", "[ENVIRONMENT]", "[HISTORY]", "[RESPONSE]"):
        assert marker in prompt
    # Capability parity: the candidate menu is present verbatim, by id.
    assert '"a1"' in prompt and '"a2"' in prompt
    # The observation is present (same world-facing view).
    assert "RW_Tank_Level" in prompt
    # The response contract carries the same selection rules as the agentic engine.
    assert "candidate_id" in prompt
    assert "Never invent a new action" in prompt


def test_response_contract_requires_the_six_keys():
    for key in ("summary", "candidate_id", "confidence", "diagnosis", "labels", "decision_type"):
        assert key in RESPONSE_CONTRACT


def test_matched_prompt_is_deterministic():
    ctx = _context()
    kw = dict(system_prompt="SYS", operational_knowledge="manual",
              soft_budget=4096, target_budget=3072, protected_k=6)
    assert build_matched_prompt(ctx, **kw)[0] == build_matched_prompt(ctx, **kw)[0]
