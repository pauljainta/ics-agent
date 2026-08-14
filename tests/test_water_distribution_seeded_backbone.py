from __future__ import annotations

import subprocess

from ics_agent.scenarios import ScenarioCatalog
from ics_agent.water_distribution_seeded_backbone import (
    DEFAULT_TMUX_SESSION_NAME,
    SeededBackboneRunner,
    TESTBED_ID,
    _read_json,
    _format_mean_std,
    _normalize_csv_rows,
    _parse_args,
    build_manifest_rows,
    build_split_rows,
)


def test_water_distribution_seeded_manifest_and_splits_are_fixed() -> None:
    catalog = ScenarioCatalog(testbed_id=TESTBED_ID)
    scenarios = catalog.resolve_selectors(["attack/*"])

    manifest_rows = build_manifest_rows(scenarios)
    split_rows = build_split_rows(manifest_rows)

    assert len(manifest_rows) == 11
    assert all(row["testbed"] == TESTBED_ID for row in manifest_rows)
    assert all(row["enabled_for_final"] == "true" for row in manifest_rows)
    assert manifest_rows[0]["scenario_id"] == "wd-attack-coordinated-grid-rws-delay"
    assert manifest_rows[-1]["scenario_id"] == "wd-attack-supply-low-spoof"

    split_counts = {}
    for row in split_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    assert split_counts == {
        "train_dev": 7,
        "validation": 2,
        "heldout_test": 2,
    }


def test_distribution_format_mean_std_handles_missing_values() -> None:
    assert _format_mean_std(None, None) == "n/a"
    assert _format_mean_std(8.3897, 0.4123) == "8.39 ± 0.41"


def test_distribution_normalize_csv_rows_stabilizes_string_comparison() -> None:
    assert _normalize_csv_rows(
        [{"manifest_order": 1, "split": "train_dev", "enabled_for_final": True}]
    ) == [
        {
            "manifest_order": "1",
            "split": "train_dev",
            "enabled_for_final": "True",
        }
    ]


def test_distribution_parse_args_launch_defaults() -> None:
    args = _parse_args(["launch"])
    assert args.mode == "launch"
    assert args.session_name == DEFAULT_TMUX_SESSION_NAME
    assert args.force_restart is False


def test_distribution_launch_controller_uses_detached_tmux(
    monkeypatch, tmp_path
) -> None:
    runner = SeededBackboneRunner(tmp_path / "seeded_backbone_qwen35_9b")
    calls: list[list[str]] = []

    monkeypatch.setattr("ics_agent.water_distribution_seeded_backbone.shutil.which", lambda name: "/usr/bin/tmux" if name == "tmux" else None)
    monkeypatch.setattr(runner, "_tmux_session_exists", lambda session_name: False)

    def fake_run(command, text=True, capture_output=True):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("ics_agent.water_distribution_seeded_backbone.subprocess.run", fake_run)

    result = runner.launch_controller("wd_seeded_backbone_test")

    assert result == 0
    assert calls
    assert calls[0][:4] == ["/usr/bin/tmux", "new-session", "-d", "-s"]
    assert calls[0][4] == "wd_seeded_backbone_test"
    launch_command = calls[0][5]
    assert "ics_agent.water_distribution_seeded_backbone controller" in launch_command
    assert str(runner.result_root) in launch_command
    assert str(runner.controller_stdout_path) in launch_command
    assert runner.launcher_state_path.exists()


def test_distribution_clear_stale_running_rows_resets_resume_state(tmp_path) -> None:
    runner = SeededBackboneRunner(tmp_path / "seeded_backbone_qwen35_9b")
    family_name = "black_box_attacker__agentic_defender"
    seed = 101

    rows = runner._load_scenario_rows(family_name, seed)
    rows[0]["status"] = "running"
    rows[0]["attempt_count"] = "1"
    rows[0]["started_at_utc"] = "2026-08-14T00:00:00+00:00"
    rows[0]["run_id"] = "stale-run"
    runner._save_scenario_rows(family_name, seed, rows)

    cleared = runner._clear_stale_running_rows(
        family_names=[family_name],
        seeds=[seed],
    )

    assert len(cleared) == 1
    refreshed_rows = runner._load_scenario_rows(family_name, seed)
    assert refreshed_rows[0]["status"] == "pending"
    assert refreshed_rows[0]["run_id"] == ""
    assert (
        refreshed_rows[0]["last_error"]
        == "stale running marker cleared before resume"
    )

    resume_state = _read_json(runner.resume_state_path, {})
    assert resume_state["pending_runs"] > 0
    assert resume_state["next_run"]["scenario_id"] == refreshed_rows[0]["scenario_id"]
