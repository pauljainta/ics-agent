from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ics_agent import regional_power_notice
from ics_agent.paths import REPO_ROOT
from ics_agent.utils import load_yaml


PLAN_PATH = REPO_ROOT / "configs" / "ctf" / "operator_handoffs" / "regional" / "power_notice.yaml"


def _fixture() -> dict:
    return {
        "schema": "sphere.composition-fixture/1",
        "scenarios": {
            "short_outage": {
                "power_schedule": [
                    {"from_s": 0, "valid_until_s": 60, "available_kw": 1.0,
                     "quality": "supplied"},
                    {"from_s": 60, "valid_until_s": 120, "available_kw": 0.0,
                     "quality": "supplied"},
                    {"from_s": 120, "valid_until_s": 240, "available_kw": 1.0,
                     "quality": "supplied"},
                ],
            },
        },
    }


def test_notice_distinguishes_current_supplied_from_expired() -> None:
    output = regional_power_notice.replay(load_yaml(PLAN_PATH), _fixture())
    timely, expired = output["cases"]

    assert output["mode"] == "synthetic_authored"
    assert output["physical_feeder_claim"] is False
    assert timely["notice_status"] == "current_supplied_notice"
    assert expired["notice_status"] == "outside_validity"
    assert timely["events"][2]["evidence_class"] == "supplied"
    assert timely["events"][2]["available_kw"] == 0.0
    assert expired["staff_source_window"]["available_kw"] == 0.0
    assert expired["staff_decision_window"]["available_kw"] == 1.0
    assert all("staff_decision_window" not in event for event in expired["events"])
    assert all(case["events"][-1]["actuation"] is False for case in output["cases"])
    assert len(output["plan_sha256"]) == 64
    json.dumps(output)


def test_notice_does_not_treat_missing_or_unknown_as_zero() -> None:
    plan = load_yaml(PLAN_PATH)
    plan["cases"][0]["notice_at_sec"] = 64
    assert regional_power_notice.replay(plan, _fixture())["cases"][0]["notice_status"] == "not_received"

    plan["cases"][0]["notice_at_sec"] = 62
    fixture = _fixture()
    fixture["scenarios"]["short_outage"]["power_schedule"][1].update(
        available_kw=None, quality="unknown",
    )
    result = regional_power_notice.replay(plan, fixture)["cases"][0]
    assert result["notice_status"] == "unknown_supplied_input"
    assert result["events"][2]["available_kw"] is None


def test_operator_view_excludes_decision_time_staff_truth() -> None:
    staff = regional_power_notice.replay(load_yaml(PLAN_PATH), _fixture())
    view = regional_power_notice.operator_view(staff)
    expired = view["cases"][1]

    assert view["schema"] == "ctf-regional-power-notice-operator-view"
    assert expired["notice_status"] == "outside_validity"
    assert "staff_source_window" not in expired
    assert "staff_decision_window" not in expired
    assert expired["events"][2]["available_kw"] == 0.0
    assert '"available_kw": 1.0' not in json.dumps(expired)


def test_notice_rejects_feeder_claim_and_incoherent_time() -> None:
    plan = load_yaml(PLAN_PATH)
    plan["source"]["physical_feeder_claim"] = True
    with pytest.raises(ValueError, match="not a hydro feeder"):
        regional_power_notice.replay(plan, _fixture())

    plan["source"]["physical_feeder_claim"] = False
    plan["cases"][0]["notice_at_sec"] = 59
    with pytest.raises(ValueError, match="chronology"):
        regional_power_notice.replay(plan, _fixture())


def test_pinned_fixture_reader_checks_revision_bytes_and_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan = load_yaml(PLAN_PATH)
    source_path = tmp_path / plan["source"]["fixture"]
    source_path.parent.mkdir(parents=True)
    contents = json.dumps(_fixture()).encode()
    source_path.write_bytes(contents)
    plan["source"]["fixture_sha256"] = hashlib.sha256(contents).hexdigest()
    monkeypatch.setattr(
        regional_power_notice.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=plan["source"]["revision"] + "\n"),
    )

    assert regional_power_notice.read_pinned_fixture(tmp_path, plan) == _fixture()
    source_path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="pinned source bytes"):
        regional_power_notice.read_pinned_fixture(tmp_path, plan)

    plan["source"]["fixture"] = "../outside.json"
    with pytest.raises(ValueError, match="within the source checkout"):
        regional_power_notice.read_pinned_fixture(tmp_path, plan)
