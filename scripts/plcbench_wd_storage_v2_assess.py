#!/usr/bin/env python3
"""Assess saved WD v2 admission traces without altering the PLC pair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ics_agent.plcbench_experiment import objective_from_spec
from ics_agent.plcbench_scorer import score_episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--config",type=Path,
                   help="defaults to the frozen descriptor in the admission directory")
    p.add_argument("--out",type=Path,
                   help="write a new assessment here; existing files are never replaced")
    p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args()
    config_path = a.config or (a.input_dir/"experiment_at_admission.json")
    if not config_path.is_file():
        p.error("frozen admission descriptor missing; pass --config")
    out = a.out or (a.input_dir/"assessment.json")
    if out.exists():
        p.error(f"refusing to overwrite assessment: {out}")
    spec=json.loads(config_path.read_text())
    admission=json.loads((a.input_dir/"admission.json").read_text())
    objective=objective_from_spec(spec)
    result={"schema_version":"plcbench.wd-storage-assessment/v2",
            "cell_id":spec["cell_id"],"evidence_origin":"isolated_local_openplc_pair",
            "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "source_hashes":{},"admission_error":admission.get("admission_error")}
    for ref in spec["sources"]:
        if ref["repo"]!="ics-agent":
            raise ValueError("unexpected source repository")
        path=a.repo_root/ref["path"]
        result["source_hashes"][ref["path"]]=hashlib.sha256(path.read_bytes()).hexdigest()
    complete=not admission.get("admission_error")
    for label in ("zero_action","reference_stop"):
        path=a.input_dir/f"{label}.jsonl"
        if not path.is_file():
            result[label]={"status":"missing"}
            complete=False
            continue
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        trace=[(float(row["t_sec"]),row["values"]) for row in rows]
        score=score_episode(process_trace=trace,native_transactions=[],object_set=[],objective=objective)
        # Check the independently sampled consumer inventory against the
        # simulator's reported actual transfer and served-demand flows. Endpoint
        # flow bounds allow one unsampled scan transition within each interval.
        balance_residual_liters = 0.0
        for (t0, v0), (t1, v1) in zip(trace, trace[1:]):
            observed = 0.5 * (v1["Grid_Consum_Tank_Level"] - v0["Grid_Consum_Tank_Level"])
            net0 = v0["Grid_To_Consumer_Flow"] - v0["Consumer_Served_Flow"]
            net1 = v1["Grid_To_Consumer_Flow"] - v1["Consumer_Served_Flow"]
            lower = min(net0, net1) * (t1-t0) / 60.0
            upper = max(net0, net1) * (t1-t0) / 60.0
            balance_residual_liters = max(balance_residual_liters,
                max(lower-observed, observed-upper, 0.0))
        result[label]={"n_samples":len(rows),"duration_sec":trace[-1][0]-trace[0][0] if rows else 0,
                       "score":score.to_dict(),"consumer_balance_max_residual_liters":balance_residual_liters,
                       "min_consumer_mm":min(row["values"]["Grid_Consum_Tank_Level"] for row in rows) if rows else None,
                       "min_elevated_mm":min(row["values"]["Grid_Elev_Tank_Level"] for row in rows) if rows else None,
                       "max_supply_mm":max(row["values"]["Supply_Tank_Level"] for row in rows) if rows else None,
                       "min_delivered_lpm":min(row["values"]["Consumer_Served_Flow"] for row in rows) if rows else None,
                       "max_spill_lpm":max(row["values"]["Total_Spill_Flow"] for row in rows) if rows else None}
        if result[label]["duration_sec"]<spec["execution"]["horizon_sec"]-1.0:
            complete=False
    reset_ok=all(admission.get(f"{label}_reset",{}).get("generation_after") ==
                 admission.get(f"{label}_reset",{}).get("generation_before",-2)+1
                 for label in ("zero_action","reference_stop"))
    reset_ok=reset_ok and bool(admission.get("restoration_reset",{}).get("Sys_Running"))
    result["reset_verified"]=reset_ok
    if complete and reset_ok:
        zero=result["zero_action"]["score"]["physical"]
        ref=result["reference_stop"]["score"]["physical"]
        zero_ok=(not zero["disrupt"]["value"] and not zero["impact"]["value"]
                 and zero["disrupt"]["assessable"] and zero["impact"]["assessable"])
        ref_ok=ref["impact"]["value"] and ref["impact"]["assessable"]
        gap_ok=all(result[label]["score"]["evidence_quality"]["max_gap_sec"] <= objective.max_gap_sec
                   for label in ("zero_action","reference_stop"))
        result["mechanistic_gate_passed"]=zero_ok and ref_ok and gap_ok
        result["zero_action_gate_passed"]=zero_ok
        result["reference_gate_passed"]=ref_ok
        result["sample_gap_gate_passed"]=gap_ok
    else:
        result["mechanistic_gate_passed"]=False
    result["interpretation"]="The reference is the deterministic operator stop; a menu policy can select it without an LLM. This gate tests the experimental process realization, not agent reasoning."
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("cell_id","mechanistic_gate_passed","reset_verified","zero_action","reference_stop")},indent=2))

if __name__=="__main__":
    main()
