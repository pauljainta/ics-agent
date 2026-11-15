"""Re-evaluate or summarize saved E2/E3 results without model calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .e23_evaluation import evaluate_e2, evaluate_e3, summarize
from .e23_runner import TRUTH

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--experiment", choices=("e2", "e3"), required=True); p.add_argument("--out-dir", type=Path, required=True); args = p.parse_args()
    truth = json.loads(TRUTH.read_text()); records = []
    for path in sorted(args.out_dir.glob("*/episode.json")):
        record = json.loads(path.read_text()); record["evaluation"] = (evaluate_e2 if args.experiment == "e2" else evaluate_e3)(record.get("report"), truth) if record.get("status") == "completed" else {"valid_submission": False}; path.with_name("private_evaluation.json").write_text(json.dumps(record["evaluation"], indent=2, sort_keys=True) + "\n"); records.append(record)
    result = summarize(records, args.experiment); (args.out_dir / f"{args.experiment}_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); (args.out_dir / "summary.md").write_text("# Experiment Summary\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n"); print(json.dumps(result, indent=2, sort_keys=True))
if __name__ == "__main__": main()
