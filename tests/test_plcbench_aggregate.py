"""Tests for PLCBench episode aggregation."""

from __future__ import annotations

from ics_agent.plcbench_aggregate import (
    EpisodeRecord,
    aggregate,
    record_from_bundle,
    render_table,
)


def _rec(eid, profile, flags, **kw):
    base = {f: False for f in
            ("discover", "read", "write", "manipulate", "disrupt", "impact")}
    base.update(flags)
    return EpisodeRecord(episode_id=eid, workload="wt", profile=profile,
                         flags=base, **kw)


def test_flag_rate_over_assessable_only():
    # 3 episodes; impact True in 1, but 1 of the others is impact-unassessable.
    recs = [
        _rec("e1", "matched", {"impact": True, "write": True}),
        _rec("e2", "matched", {"write": True}, impact_assessable=False),
        _rec("e3", "matched", {}),
    ]
    (cell,) = aggregate(recs)
    assert cell.n == 3
    # impact rate is over the 2 assessable episodes -> 1/2 = 0.5, not 1/3.
    assert cell.flag_rates["impact"] == 0.5
    assert cell.impact_unassessable == 1
    # write is a cyber flag, always assessable: 2/3.
    assert abs(cell.flag_rates["write"] - 2 / 3) < 1e-9


def test_grouping_and_ordering_deterministic():
    recs = [
        _rec("b", "native", {"discover": True}),
        _rec("a", "matched", {"discover": True}),
    ]
    cells = aggregate(recs)
    assert [(c.profile) for c in cells] == ["matched", "native"]  # sorted
    assert all(c.n == 1 for c in cells)


def test_mean_actions_and_ids():
    recs = [_rec("e1", "m", {}, actions_run=4), _rec("e2", "m", {}, actions_run=6)]
    (cell,) = aggregate(recs)
    assert cell.mean_actions == 5.0
    assert cell.episode_ids == ["e1", "e2"]


def test_render_table_has_headers_and_rows():
    (cell,) = aggregate([_rec("e1", "matched", {"impact": True})])
    out = render_table([cell])
    assert "workload" in out and "profile" in out and "imp?" in out
    assert "matched" in out


def test_record_from_bundle_reads_assessable():
    meta = {"episode_id": "e9", "workload": "water_distribution",
            "profile": "plcbench_matched_distribution_attacker",
            "stop_reason": "final", "actions_run": 7}
    score = {"flags": {"write": True, "impact": False},
             "physical": {"impact": {"assessable": False},
                          "disrupt": {"assessable": True}}}
    rec = record_from_bundle(meta, score)
    assert rec.episode_id == "e9" and rec.profile.endswith("distribution_attacker")
    assert rec.flags["write"] is True
    assert rec.impact_assessable is False and rec.disrupt_assessable is True
    assert rec.actions_run == 7
    # And it aggregates without counting the unassessable impact in the rate.
    (cell,) = aggregate([rec])
    assert cell.impact_unassessable == 1 and cell.flag_rates["impact"] == 0.0
