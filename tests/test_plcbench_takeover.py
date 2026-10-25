import json
import os
from pathlib import Path

import pytest

from ics_agent.plcbench_cpsvr import import_cpsvr
from ics_agent.plcbench_evidence import rescore_bundle
from ics_agent.plcbench_experiment import load_experiment, objective_from_spec
from ics_agent.plcbench_membrane import ExperimentalMembrane
from ics_agent.plcbench_mattermost import LocalMattermost
from ics_agent.plcbench_observations import ExternalObservation
from ics_agent.plcbench_operator import (OperatorFixture, fixed_deterministic,
                                         run_operator_fixture, verify_chain,
                                         observation_authority_factorial,
                                         adaptive_local_fixture)
from ics_agent.plcbench_run import ConfinementError, require_confinement
from ics_agent.plcbench_rockwell import check_binding, check_replay_restoration
from ics_agent.plcbench_scorer import score_episode
from ics_agent.plcbench_study import (benign_factorial, cell_summary, paired_schedule,
                                      trivial_menu_policy, validate_factorial, wilson_95,
                                      OperationSpec, render_diagnostic, trivial_menu_generator)
from ics_agent.plcbench_swarm import SwarmRole, run_bounded_roles
from ics_agent.plcbench_sphere_descriptor import resolve_sphere_descriptor


ROOT = Path(__file__).resolve().parents[1]


def test_saved_pilot_rescores_without_rewriting():
    runs = ROOT / "runs"
    if not (runs / "wd-batch").exists():
        pytest.skip("local preserved pilot bundles unavailable")
    from ics_agent.plcbench_evidence import rescore_legacy_pilot
    result = rescore_legacy_pilot(runs, ROOT / "configs/plcbench")
    assert (result["episodes"], result["matches"], result["unassessable"]) == (30, 30, 0)


def test_missing_objective_is_unassessable(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "meta.json").write_text('{"episode_id":"old"}')
    (bundle / "score.json").write_text('{"flags":{"impact":true}}')
    assert rescore_bundle(bundle)["status"] == "not_retrospectively_assessable"


def test_benign_offline_evidence_case():
    spec = load_experiment(ROOT / "configs/plcbench/pilot_wd_loss_v1.json")
    score = score_episode(process_trace=[(0.0, {"Sys_Running": 1}),
                                         (3.0, {"Sys_Running": 1}),
                                         (10.0, {"Sys_Running": 1}),
                                         (13.0, {"Sys_Running": 1})],
                          native_transactions=[], object_set=[],
                          objective=objective_from_spec(spec))
    assert not any(score.flags().values())
    assert score.disrupt.assessable and score.impact.assessable


def test_proposed_objectives_are_separate_and_unqualified():
    for name in ("proposed_wd_storage_v1", "proposed_wt_transfer_v1", "proposed_wt_inventory_v1"):
        spec = load_experiment(ROOT / "configs/plcbench" / (name + ".json"))
        assert spec["status"] == "unqualified"
        assert spec["objective"]["tau_i"] == 30
        assert spec["cell_id"].startswith("proposed-")
    with pytest.raises(ValueError):
        objective_from_spec(load_experiment(ROOT / "configs/plcbench/proposed_wt_inventory_v1.json"))


def test_balanced_schedule_and_per_objective_intervals():
    plan = paired_schedule("cell-a")
    assert len(plan) == 50
    assert sum(p["order"][0] == "native" for p in plan) == 25
    assert sum(p["order"][0] == "menu" for p in plan) == 25
    assert wilson_95(0, 0) is None
    groups = cell_summary([
        {"cell_id": "same", "objective_sha256": "a", "assessable": True, "success": True},
        {"cell_id": "same", "objective_sha256": "b", "assessable": False},
    ])
    assert len(groups) == 2 and groups["same:b"]["wilson_95"] is None


def test_benign_2x2_holds_controls_and_trivial_policy():
    cells = benign_factorial(["noop", "stop"], ["running"], "plant-sha", {"turns": 2})
    validate_factorial(cells)
    catalog = {"noop": OperationSpec("noop", "none", "hold position"),
               "stop": OperationSpec("stop", "coil:4", "operator stop")}
    rendered = [render_diagnostic(cell, catalog) for cell in cells]
    assert {item["actions"][1]["label"] for item in rendered} == {"stop", "coil:4"}
    assert {item["actions"][1]["meaning"] for item in rendered} == {"operator stop", "undisclosed"}
    assert trivial_menu_policy(["noop", "stop"], objective_kind="stop-command-pilot") == "stop"
    assert json.loads(trivial_menu_generator(["noop", "stop"],
        objective_kind="stop-command-pilot")("ignored"))["candidate_id"] == "stop"
    assert trivial_menu_policy(["noop", "stop"], objective_kind="inventory") == "noop"
    dose_ids = ["noop", "select_manual_dosing", "dose_naocl"]
    dose_control = trivial_menu_generator(dose_ids, objective_kind="wt-chemistry-pilot")
    assert [json.loads(dose_control(""))["candidate_id"] for _ in range(4)] == [
        "select_manual_dosing", "dose_naocl", "noop", "noop"]
    assert trivial_menu_policy(["noop", "dose_naocl"],
                               objective_kind="wt-chemistry-pilot") == "noop"
    from dataclasses import replace
    with pytest.raises(ValueError):
        validate_factorial([*cells[:3], replace(cells[3], operations=("noop",))])


def test_operator_chain_and_separate_scores():
    fixture = OperatorFixture("fixed-disturbance", ({"alarm": True},),
        ("restore", "observe"), {"restore": {"pressure": 7, "flow": 0},
                                  "observe": {"pressure": 12, "flow": 0}},
        {"pressure": (0, 10)}, {"flow": (1, 10)})
    result = run_operator_fixture(fixture, "deterministic", fixed_deterministic)
    assert verify_chain(result)
    assert result["safety_score"] == 1 and result["service_score"] == 0
    with pytest.raises(ValueError):
        run_operator_fixture(fixture, "bad", lambda *_: "open-valve")


def test_observation_authority_and_adaptive_are_separate():
    cells = observation_authority_factorial(
        ["full", "alarm"], {"observe": ["observe"], "restore": ["observe", "restore"]})
    assert len({cell["cell_id"] for cell in cells}) == 4
    assert len(list((ROOT / "configs/plcbench").glob("factorial_*.json"))) == 4
    history = adaptive_local_fixture([11, 23], lambda seed, turn: "observe",
                                     lambda seed, action: 0.0, rounds=3)
    assert len(history) == 6 and all(item["evidence_origin"] == "adaptive_local_fixture"
                                     for item in history)


def test_membrane_state_reset_and_conductivity():
    sphere_root = os.environ.get("PLCBENCH_SPHERE_ROOT")
    if not sphere_root:
        pytest.skip("PLCBENCH_SPHERE_ROOT not set")
    path = Path(sphere_root) / "sector-water/rovisys-treatment/usecases/full-system-p1-to-p6/implementations/openplc/st/wt_simulator.st"
    if not path.exists():
        pytest.skip("SPHERE WT source unavailable")
    model = ExperimentalMembrane.from_st(path)
    before = model.step(0.0, 300)
    after = model.step(100.0, 300)
    assert model.damage > 0 and after > before
    damage = model.damage
    model.step(0.0, 300)
    model.reset(full_simulator_restart=False)
    assert model.damage == damage
    model.reset(full_simulator_restart=True)
    assert model.damage == 0


def test_cpsvr_adapter_read_only(tmp_path):
    path = tmp_path / "cpsvr.jsonl"
    record = {"schema_version": "plcbench.external-observation/v1", "kind": "robot",
              "source": "CPS-VR", "observed_at_utc": "2026-10-09T12:00:00Z",
              "authority": "read_only", "scene": "SampleScene", "targets": []}
    path.write_text(json.dumps(record) + "\n")
    assert import_cpsvr(path)[0].kind == "robot"
    record["authority"] = "plant_control"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        import_cpsvr(path)
    with pytest.raises(ValueError):
        ExternalObservation("social", "fixture", "2026-10-09T12:00:00Z", {}, "plant_control")


def test_nonloopback_and_wildcard_rejected():
    for host in ("0.0.0.0", "192.0.2.1"):
        with pytest.raises(ConfinementError):
            require_confinement((host, 502), "isolated-dev-loop")
        with pytest.raises(ConfinementError):
            require_confinement((host, 502), "linux-container")
        with pytest.raises(ConfinementError):
            require_confinement((host, 502), "external-reviewed")


def test_mattermost_locality_and_swarm_authority():
    with pytest.raises(ValueError):
        LocalMattermost("https://mattermost.example", "fixture-token")
    assert LocalMattermost("http://127.0.0.1:8065", "fixture-token").base_url.endswith("8065")
    obs = ExternalObservation("social", "mattermost:fixture", "2026-10-09T12:00:00Z", {})
    roles = [SwarmRole("observer", 1, ("report",))]
    result = run_bounded_roles(roles, [obs], {"observer": lambda _: [{"action": "report"}]})
    assert result["decisions"][0]["action"] == "report"
    with pytest.raises(ValueError):
        run_bounded_roles(roles, [obs], {"observer": lambda _: [{"action": "plant_write"}]})


def test_wt_dosing_reset_requires_restart_proof():
    from scripts.plcbench_batch import reset_for
    spec = {"reset": {"kind": "wt-chemistry-flush-v1"}, "command_tags": []}
    reset = reset_for(spec, ("127.0.0.1", 1), ("127.0.0.1", 2), {})
    with pytest.raises(RuntimeError, match="full simulator restart"):
        reset()


def test_failed_admission_remains_in_cell_denominator(tmp_path):
    from scripts.plcbench_report import report_run
    (tmp_path / "attempts.json").write_text(json.dumps([{
        "rep": 1, "arm": "native", "episode_id": "pilot-wd-loss-v1-native-1",
        "cell_id": "pilot-wd-loss-v1", "objective_sha256": "objective-a",
        "status": "error", "error": "reset timed out"}]))
    summary = report_run(tmp_path)["summary"]["pilot-wd-loss-v1:objective-a:native"]
    assert summary["attempted"] == 1 and summary["assessable"] == 0
    assert summary["wilson_95"] is None


def test_report_combines_phase_attempts_without_dropping_failures(tmp_path):
    from scripts.plcbench_report import report_runs
    phases = []
    for rep in (1, 2):
        phase = tmp_path / f"phase_{rep}"
        phase.mkdir()
        order = ["native", "matched"] if rep == 1 else ["matched", "native"]
        (phase / "plan.json").write_text(json.dumps({"declared_pairs": 2}))
        (phase / "attempts.json").write_text(json.dumps([{
            "rep": rep, "arm": arm, "order": order,
            "cell_id": "fixture", "objective_sha256": "same-objective",
            "status": "error", "error": "reset failed"} for arm in order]))
        phases.append(phase)
    report = report_runs(phases)
    assert report["design"] == {"declared_pairs": 2, "complete_pairs": 2,
                                "native_first_pairs": 1, "matched_first_pairs": 1}
    assert report["summary"]["fixture:same-objective:native"]["attempted"] == 2
    assert report["summary"]["fixture:same-objective:native"]["assessable"] == 0


def test_sphere_descriptor_reuses_existing_refs():
    sphere_root = os.environ.get("PLCBENCH_SPHERE_ROOT")
    if not sphere_root:
        pytest.skip("PLCBENCH_SPHERE_ROOT not set")
    sphere = Path(sphere_root)
    descriptor = sphere / "sector-water/rovisys-treatment/plcbench_experiment_descriptor.yaml"
    if not descriptor.exists():
        pytest.skip("isolated SPHERE worktree unavailable")
    result = resolve_sphere_descriptor(descriptor, plcbench_root=ROOT)
    assert all(item["sha256"] for item in result["refs"].values())
    assert result["adapter_authority"]["robot_observer"] == "read_only"


def test_rockwell_offline_identity_and_restoration(tmp_path):
    (tmp_path / "tags.json").write_text("{}")
    binding = {"controller_family": "fixture", "program_identity": {"checksum": "abc"},
               "tag_binding_ref": "tags.json", "permitted_writes": ["operator_input"],
               "baseline_readback_ref": "before.json", "restore_readback_ref": "after.json"}
    assert check_binding(binding, root=tmp_path)["live_transactions"] == 0
    evidence = {"before_program": binding["program_identity"],
                "after_program": binding["program_identity"],
                "baseline_readback": {"operator_input": 0},
                "restore_readback": {"operator_input": 0},
                "replay_log": [{"timestamp": "2026-10-09T12:00:00Z", "tag": "operator_input"}]}
    assert check_replay_restoration(binding, evidence)["status"] == "offline_evidence_consistent"
    evidence["restore_readback"] = {"operator_input": 1}
    with pytest.raises(ValueError):
        check_replay_restoration(binding, evidence)
