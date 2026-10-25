"""Reproduce fixed, factorial, or adaptive offline PLCBench fixture evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ics_agent.plcbench_operator import (OperatorFixture, adaptive_local_fixture,
    fixed_deterministic, fixed_reference, run_operator_fixture, verify_chain)


ROOT = Path(__file__).resolve().parents[1] / "configs/plcbench"


def load_fixture() -> OperatorFixture:
    raw = json.loads((ROOT / "operator_fixed_disturbance_v1.json").read_text())
    return OperatorFixture(raw["fixture_id"], tuple(raw["observations"]),
        tuple(raw["allowed_interventions"]), raw["plant_responses"],
        {k: tuple(v) for k, v in raw["safe_envelope"].items()},
        {k: tuple(v) for k, v in raw["service_target"].items()})


def fixed(decisions: dict | None) -> dict:
    fixture = load_fixture()
    runs = [run_operator_fixture(fixture, "reference", fixed_reference),
            run_operator_fixture(fixture, "deterministic", fixed_deterministic)]
    if decisions is not None:
        selected = decisions[fixture.fixture_id]
        runs.append(run_operator_fixture(fixture, "llm-decision-file", lambda *_: selected))
    assert all(verify_chain(item) for item in runs)
    return {"study": "fixed", "runs": runs,
            "llm_status": "decision_file_fixture" if decisions is not None else "unassessable_no_decision"}


def factorial() -> dict:
    base = load_fixture()
    runs = []
    for path in sorted(ROOT.glob("factorial_*.json")):
        cell = json.loads(path.read_text())
        observations = (base.observations if cell["observation_mode"] == "full"
                        else tuple({"t_sec": o["t_sec"], "alarm": o["alarm"]}
                                   for o in base.observations))
        fixture = OperatorFixture(cell["cell_id"], observations,
            base.allowed_interventions, base.plant_responses,
            base.safe_envelope, base.service_target)
        runs.append(run_operator_fixture(fixture, "deterministic", fixed_deterministic,
                                         authority=cell["allowed_interventions"]))
    if len(runs) != 4:
        raise ValueError("factorial requires four separate configs")
    return {"study": "observation_x_authority", "runs": runs}


def adaptive() -> dict:
    raw = json.loads((ROOT / "adaptive_local_fixture_v1.json").read_text())
    history = adaptive_local_fixture(raw["seeds"],
        lambda seed, round_no: "restore" if (seed + round_no) % 2 else "observe",
        lambda _seed, action: 1.0 if action == "restore" else 0.0,
        rounds=raw["rounds"])
    return {"study": "adaptive_local_fixture", "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("fixed", "factorial", "adaptive"))
    parser.add_argument("--llm-decisions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    decisions = json.loads(args.llm_decisions.read_text()) if args.llm_decisions else None
    result = (fixed(decisions) if args.study == "fixed" else
              factorial() if args.study == "factorial" else adaptive())
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
