"""Read and re-score immutable PLCBench bundles without plant access."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .plcbench_experiment import load_experiment, objective_from_spec
from .plcbench_scorer import NativeTransaction, Objective, score_episode


def _value(raw: str) -> Any:
    if raw == "":
        return None
    if raw in ("True", "False"):
        return raw == "True"
    try:
        return float(raw)
    except ValueError:
        return raw


def load_trace(path: Path) -> list[tuple[float, dict[str, Any]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "t" not in reader.fieldnames:
            raise ValueError("trace.csv lacks time column")
        return [(float(row["t"]), {key: value for key, raw in row.items()
                                    if key != "t" and (value := _value(raw)) is not None})
                for row in reader]


def _objective(value: Mapping[str, Any]) -> Objective:
    return Objective(**{key: value[key] for key in
                        ("warning", "impact", "tau_d", "tau_i", "max_gap_sec")})


def rescore_bundle(bundle: Path, *, objective: Objective | None = None,
                   objective_source: str | None = None) -> dict[str, Any]:
    """Preserve stored verdicts while adding an independently computed verdict.

    Legacy bundles lack an embedded objective. Without a supplied objective they
    remain not retrospectively assessable; no objective is inferred from flags.
    """
    bundle = Path(bundle)
    meta = json.loads((bundle / "meta.json").read_text())
    original = json.loads((bundle / "score.json").read_text())
    report: dict[str, Any] = {
        "bundle": str(bundle), "episode_id": meta.get("episode_id"),
        "original": {"flags": original.get("flags"),
                     "physical": original.get("physical")},
    }
    if objective is None and meta.get("objective"):
        objective = _objective(meta["objective"])
        objective_source = "bundle:meta.objective"
    if objective is None:
        report.update(status="not_retrospectively_assessable",
                      reason="objective absent from bundle and no versioned external objective supplied")
        return report
    try:
        trace = load_trace(bundle / "trace.csv")
        transactions = [NativeTransaction(**record) for record in
                        json.loads((bundle / "transactions.json").read_text())]
        computed = score_episode(process_trace=trace,
                                 native_transactions=transactions,
                                 object_set=meta.get("object_set", []), objective=objective)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report.update(status="not_retrospectively_assessable", reason=str(exc))
        return report
    scored = computed.to_dict()
    report.update(status="assessable", objective_source=objective_source,
                  task_specific=scored,
                  matches_original=(scored["flags"] == original.get("flags") and
                                    all(scored["physical"][name]["assessable"] ==
                                        original.get("physical", {}).get(name, {}).get("assessable")
                                        for name in ("disrupt", "impact"))))
    return report


LEGACY_PILOT_CONFIGS = {
    "wd-batch": "pilot_wd_loss_v1.json",
    "wt-lot": "pilot_wt_loss_v1.json",
    "wt-dosing": "pilot_wt_dosing_v1.json",
}


def rescore_legacy_pilot(run_root: Path, config_root: Path) -> dict[str, Any]:
    cases = []
    for directory, config_name in LEGACY_PILOT_CONFIGS.items():
        spec = load_experiment(config_root / config_name)
        objective = objective_from_spec(spec)
        for bundle in sorted((run_root / directory).glob("*/meta.json")):
            cases.append(rescore_bundle(bundle.parent, objective=objective,
                                        objective_source=f"versioned:{config_name}"))
    return {"episodes": len(cases),
            "matches": sum(case.get("matches_original") is True for case in cases),
            "unassessable": sum(case["status"] != "assessable" for case in cases),
            "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path)
    parser.add_argument("--objective-config", type=Path)
    parser.add_argument("--pilot-root", type=Path)
    args = parser.parse_args()
    if args.pilot_root:
        if args.bundle or args.objective_config:
            parser.error("--pilot-root cannot be combined with bundle arguments")
        config_root = Path(__file__).resolve().parents[1] / "configs" / "plcbench"
        result = rescore_legacy_pilot(args.pilot_root, config_root)
        print(json.dumps({key: result[key] for key in ("episodes", "matches", "unassessable")}, indent=2))
    elif args.bundle:
        spec = load_experiment(args.objective_config) if args.objective_config else None
        result = rescore_bundle(args.bundle,
                                objective=objective_from_spec(spec) if spec else None,
                                objective_source=str(args.objective_config) if spec else None)
        print(json.dumps(result, indent=2))
    else:
        parser.error("provide a bundle or --pilot-root")


if __name__ == "__main__":
    main()
