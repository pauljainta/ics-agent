"""Run an exploratory menu/native PLCBench diagnostic over a live OpenPLC pair.

This is the single place that drives the comparison the baseline is built to
measure: per repetition, reset the plant and run one **native** episode
(free-form discovery) and one **matched** episode (named candidate menu) against
the same live pair. The arms differ in presentation, available actions, history,
and execution path; they are not a matched causal comparison. Each episode is
written as a bundle and the set is aggregated into a diagnostic table.

Workloads (water distribution, water treatment, ...) differ only in the
``ComparisonConfig`` passed in -- the map, the writable command tags, the sensor
tags, the objective, and the matched menu -- never in this code. The model and the
plant are injected (``generate_for(arm) -> generate`` and the endpoints), so this
module reaches neither a model nor a PLC by itself and is testable offline.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .plcbench_aggregate import EpisodeRecord, aggregate, record_from_bundle, render_table
from .plcbench_experiment import canonical_hash
from .plcbench_live import (ContinuousTraceRecorder, ModbusObserver,
                            object_set_from_map, run_live_native_episode,
                            sensor_specs_from_map)
from .plcbench_matched_run import MenuCandidate, run_matched_episode
from .plcbench_run import write_native_bundle
from .plcbench_scorer import NativeTransaction, Objective, score_episode

Generate = Callable[[str], str]
GenerateFor = Callable[[str], Generate]  # arm name ("native"|"matched") -> generate


@dataclass
class ComparisonConfig:
    """Everything that makes one workload's comparison concrete."""

    workload: str
    controller_endpoint: tuple[str, int]
    simulator_endpoint: tuple[str, int]
    mappings: Mapping[str, Any]
    command_tags: Sequence[str]      # writable command tag NAMES; the object-set source
    sensor_tags: Sequence[str]       # canonical sensors the observer samples
    objective: Objective
    menu: Sequence[MenuCandidate]    # matched arm's named candidates
    reset: Callable[[], Any]         # put the plant in the pre-attack state; returns a probe value
    native_profile: str
    matched_profile: str
    settle_sec: float = 13.0         # observe sustained impact past the last action (> tau_i)
    settle_interval_sec: float = 2.0
    native_max_actions: int = 10
    matched_max_actions: int = 8
    horizon_sec: float = 600.0
    continuous_observation_sec: float = 0.0
    continuous_interval_sec: float = 1.0
    # Sandbox interpreter for the native arm. It must expose the client the actor
    # is told about (pymodbus 3.x) in its *main* site-packages, because the sandbox
    # runs Python in isolated mode (-I), which ignores the user site. A dedicated
    # venv with pymodbus satisfies this; None uses the current interpreter (only
    # correct if its main site already has pymodbus 3.x).
    sandbox_python: str | None = None
    container_image: str | None = None
    # Required safety acknowledgement for the native arm's free-form code execution
    # (see plcbench_run.require_confinement). 'isolated-dev-loop' for a loopback pair.
    confinement: str = "isolated-dev-loop"
    cell_id: str = ""
    experiment: Mapping[str, Any] = field(default_factory=dict)


def run_comparison(
    cfg: ComparisonConfig,
    *,
    generate_for: GenerateFor,
    reps: int,
    out_dir: Path,
    deployment: Mapping[str, Any] | None = None,
    log: Callable[[str], Any] = print,
    start_rep: int = 1,
    end_rep: int | None = None,
    stop_on_error: bool = False,
) -> tuple[list[EpisodeRecord], str]:
    """Run ``reps`` native+matched episode pairs and return (records, table).

    Both arms share one object set (derived from ``cfg.command_tags``) so the
    manipulate flag means the same thing on each side. Each episode persists a
    bundle under ``out_dir/<arm>-<rep>``; the aggregate is also written there as
    ``comparison_table.txt``.
    """
    object_set = object_set_from_map(cfg.mappings, cfg.command_tags)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[EpisodeRecord] = []
    attempts: list[dict[str, Any]] = []
    objective_sha = cfg.experiment.get("objective_sha256") or canonical_hash(asdict(cfg.objective))
    end_rep = reps if end_rep is None else end_rep
    if not (1 <= start_rep <= end_rep <= reps):
        raise ValueError("phase range must be inside 1..reps")
    plan = {"schema_version": "plcbench.paired-plan/v1",
            "cell_id": cfg.cell_id or cfg.workload,
            "objective_sha256": objective_sha,
            "experiment_identity_sha256": cfg.experiment.get("identity_sha256"),
            "declared_pairs": reps, "phase_start_rep": start_rep,
            "phase_end_rep": end_rep,
            "order": [{"rep": i, "arms": ["native", "matched"] if i % 2
                       else ["matched", "native"]} for i in range(1, reps + 1)]}
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    for i in range(start_rep, end_rep + 1):
        # With an even number of repetitions each condition runs first equally often.
        order = ("native", "matched") if i % 2 else ("matched", "native")
        for arm in order:
            run_dir = out_dir / f"{arm}-{i}"
            episode_id = (f"{cfg.cell_id}-{arm}-{i}" if cfg.cell_id
                          else f"{cfg.workload}-{arm}-{i}")
            recorder = None
            probe = None
            try:
                if run_dir.exists():
                    raise FileExistsError(f"refusing to overwrite saved episode: {run_dir}")
                probe = cfg.reset()
                log(f"[{time.strftime('%H:%M:%S')}] rep {i} {arm}: reset (state={probe})")
                if cfg.continuous_observation_sec > 0:
                    specs = sensor_specs_from_map(cfg.mappings, cfg.sensor_tags)
                    recorder = ContinuousTraceRecorder(
                        ModbusObserver(*cfg.simulator_endpoint, specs),
                        interval_sec=cfg.continuous_interval_sec)
                    recorder.start()
                if arm == "native":
                    result, _ = run_live_native_episode(
                        controller_endpoint=cfg.controller_endpoint,
                        simulator_endpoint=cfg.simulator_endpoint,
                        mappings=cfg.mappings,
                        sensor_tags=cfg.sensor_tags,
                        object_tags=list(cfg.command_tags),
                        objective=cfg.objective,
                        generate=generate_for(arm),
                        workload=cfg.workload,
                        profile=cfg.native_profile,
                        episode_id=episode_id,
                        run_dir=run_dir,
                        deployment=dict(deployment or {}),
                        experiment=cfg.experiment,
                        max_actions=cfg.native_max_actions,
                        horizon_sec=cfg.horizon_sec,
                        settle_sec=0.0 if recorder else cfg.settle_sec,
                        settle_interval_sec=cfg.settle_interval_sec,
                        sandbox_python=cfg.sandbox_python,
                        container_image=cfg.container_image,
                        confinement=cfg.confinement,
                    )
                else:
                    result, _ = run_matched_episode(
                        controller_endpoint=cfg.controller_endpoint,
                        simulator_endpoint=cfg.simulator_endpoint,
                        mappings=cfg.mappings,
                        sensor_tags=cfg.sensor_tags,
                        object_set=object_set,
                        objective=cfg.objective,
                        generate=generate_for(arm),
                        menu=cfg.menu,
                        workload=cfg.workload,
                        profile=cfg.matched_profile,
                        episode_id=episode_id,
                        run_dir=run_dir,
                        deployment=dict(deployment or {}),
                        experiment=cfg.experiment,
                        max_actions=cfg.matched_max_actions,
                        horizon_sec=cfg.horizon_sec,
                        settle_sec=0.0 if recorder else cfg.settle_sec,
                        settle_interval_sec=cfg.settle_interval_sec,
                    )
                if recorder is not None:
                    recorder.wait_until(cfg.continuous_observation_sec)
                    recorder.stop()
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "recorder_errors.json").write_text(
                        json.dumps(recorder.errors, indent=2) + "\n")
                    result.trace = list(recorder.trace)
                    result.score = score_episode(
                        process_trace=result.trace,
                        native_transactions=[NativeTransaction(**txn)
                                             for txn in result.transactions],
                        object_set=object_set, objective=cfg.objective)
                    meta = json.loads((run_dir / "meta.json").read_text())
                    meta["continuous_observation"] = {
                        "duration_sec": cfg.continuous_observation_sec,
                        "interval_sec": cfg.continuous_interval_sec,
                        "read_errors": len(recorder.errors)}
                    meta["assessment_status"] = ("assessable" if all(
                        result.score.to_dict()["physical"][name]["assessable"]
                        for name in ("disrupt", "impact")) else "partly_unassessable")
                    write_native_bundle(run_dir, meta, result)
                records.append(_record(run_dir))
                attempts.append({"rep": i, "arm": arm, "order": list(order),
                                 "episode_id": episode_id, "status": "completed",
                                 "cell_id": cfg.cell_id or cfg.workload,
                                 "objective_sha256": objective_sha,
                                 "reset_probe": str(probe)})
                (out_dir / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
                log(f"[{time.strftime('%H:%M:%S')}] rep {i} {arm}: flags={result.score.flags()}")
            except Exception as exc:  # preserve every attempted cell and its failure
                import traceback
                if recorder is not None:
                    recorder.stop()
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "partial_recorder.json").write_text(json.dumps({
                        "trace": recorder.trace, "errors": recorder.errors}, indent=2) + "\n")
                attempts.append({"rep": i, "arm": arm, "order": list(order),
                                 "episode_id": episode_id,
                                 "cell_id": cfg.cell_id or cfg.workload,
                                 "objective_sha256": objective_sha,
                                 "status": "error", "reset_probe": str(probe),
                                 "error_type": type(exc).__name__,
                                 "error": str(exc)})
                (out_dir / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
                log(f"[{time.strftime('%H:%M:%S')}] rep {i} {arm}: ERROR {exc}\n"
                    f"{traceback.format_exc()[-600:]}")
                if stop_on_error:
                    raise

    (out_dir / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
    table = render_table(aggregate(records))
    (out_dir / "comparison_table.txt").write_text(table + "\n")
    return records, table


def _record(run_dir: Path) -> EpisodeRecord:
    """Read a persisted bundle back into an EpisodeRecord (same for both arms)."""
    meta = json.loads((run_dir / "meta.json").read_text())
    score = json.loads((run_dir / "score.json").read_text())
    return record_from_bundle(meta, score)
