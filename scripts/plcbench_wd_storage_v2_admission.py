#!/usr/bin/env python3
"""Live local admission for WD storage v2, with no model invocation.

Run inside a disposable container on the isolated plcbench-wd-v2_pair network.
It performs only the declared fixture reset and HMI stop on that pair.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import struct
import time
from pathlib import Path

from pymodbus.client import ModbusTcpClient

from ics_agent.plcbench_wd_storage_v2 import reset_wd_storage_v2, _request


def sample(endpoint, mappings):
    # One contiguous read is less disruptive to OpenPLC than eight reads and
    # keeps the observation fields from the same process scan window.
    lo = min(int(mappings[tag]["address"]) for tag in (
        "Supply_Tank_Level", "Grid_Elev_Tank_Level", "Grid_Consum_Tank_Level",
        "Grid_To_Consumer_Flow", "Consumer_Served_Flow", "Fixture_Demand_Flow",
        "Total_Spill_Flow", "Fixture_Reset_Generation"))
    hi = max(int(mappings[tag]["address"]) + 2 for tag in (
        "Supply_Tank_Level", "Grid_Elev_Tank_Level", "Grid_Consum_Tank_Level",
        "Grid_To_Consumer_Flow", "Consumer_Served_Flow", "Fixture_Demand_Flow",
        "Total_Spill_Flow", "Fixture_Reset_Generation"))
    for attempt in range(3):
        client = ModbusTcpClient(endpoint[0], port=endpoint[1], timeout=1, retries=0)
        try:
            if not client.connect():
                raise RuntimeError("simulator sample connect failed")
            response = client.read_holding_registers(lo, count=hi-lo)
            if response.isError() or len(response.registers) != hi-lo:
                raise RuntimeError("bulk simulator read rejected")
            values = {}
            for tag in (
                "Supply_Tank_Level", "Grid_Elev_Tank_Level", "Grid_Consum_Tank_Level",
                "Grid_To_Consumer_Flow", "Consumer_Served_Flow", "Fixture_Demand_Flow",
                "Total_Spill_Flow", "Fixture_Reset_Generation"):
                offset = int(mappings[tag]["address"]) - lo
                values[tag] = struct.unpack(">f", struct.pack(">HH",
                    *response.registers[offset:offset+2]))[0]
            return values
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.1)
        finally:
            client.close()
    raise AssertionError("unreachable")


def observe(label, duration, interval, sim_ep, mappings, out):
    trace=[]
    started=time.monotonic()
    next_sample=started
    next_report=60
    with (out/f"{label}.jsonl").open("w") as handle:
        while True:
            now=time.monotonic()
            if now<next_sample:
                time.sleep(next_sample-now)
            t=time.monotonic()-started
            row={"t_sec":t,"t_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "values":sample(sim_ep,mappings)}
            trace.append(row)
            handle.write(json.dumps(row,sort_keys=True)+"\n")
            handle.flush()
            if t>=next_report:
                print(json.dumps({"phase":label,"elapsed_sec":round(t,1),"consumer_mm":row["values"]["Grid_Consum_Tank_Level"],"elevated_mm":row["values"]["Grid_Elev_Tank_Level"]}),flush=True)
                next_report+=60
            if t>=duration:
                break
            next_sample+=interval
    return trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-json",type=Path,required=True)
    p.add_argument("--config",type=Path,
                   default=Path(__file__).resolve().parents[1]/"configs/plcbench/wd_storage_mass_balance_v2.json")
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--controller",default="controller:502")
    p.add_argument("--simulator",default="simulator:502")
    p.add_argument("--duration",type=float,default=360.0)
    p.add_argument("--interval",type=float,default=1.0)
    a=p.parse_args()
    if a.duration<35 or not (0.2<=a.interval<=2.0):
        p.error("admission needs >=35 s coverage and 0.2-2.0 s samples")
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        p.error("output directory must be empty")
    a.out_dir.mkdir(parents=True,exist_ok=True)
    config_bytes=a.config.read_bytes()
    spec=json.loads(config_bytes)
    if spec.get("cell_id")!="wd-storage-mass-balance-v2":
        raise ValueError("wrong experiment descriptor for WD v2 admission")
    (a.out_dir/"experiment_at_admission.json").write_bytes(config_bytes)
    repo_root=Path(__file__).resolve().parents[1]
    source_hashes={}
    for ref in spec["sources"]:
        if ref["repo"]!="ics-agent":
            raise ValueError("unexpected source repository")
        source_path=repo_root/ref["path"]
        source_hashes[ref["path"]]=hashlib.sha256(source_path.read_bytes()).hexdigest()
    data=json.loads(a.map_json.read_text())
    source_hash=hashlib.sha256(a.map_json.with_suffix(".yaml").read_bytes()).hexdigest()
    if data.get("source_yaml_sha256")!=source_hash:
        raise RuntimeError("generated map differs from YAML source")
    mappings=data["mappings"]
    def ep(value):
        h,port=value.rsplit(":",1)
        return h,int(port)
    ctrl_ep,sim_ep=ep(a.controller),ep(a.simulator)
    metadata={"schema_version":"plcbench.wd-storage-admission/v2",
              "evidence_origin":"isolated_local_openplc_pair",
              "cell_id":"wd-storage-mass-balance-v2",
              "map_yaml_sha256":source_hash,
              "config_sha256":hashlib.sha256(config_bytes).hexdigest(),
              "source_hashes":source_hashes,
              "duration_sec":a.duration,"interval_sec":a.interval,
              "started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        for label in ("zero_action","reference_stop"):
            proof=reset_wd_storage_v2(ctrl_ep,sim_ep,mappings)
            metadata[f"{label}_reset"]=proof
            print(json.dumps({"phase":label,"reset_generation":proof["generation_after"]}),flush=True)
            if label=="reference_stop":
                _request(ctrl_ep,"write",int(mappings["HMI_Stop_PB"]["address"]),True)
                metadata["reference_intervention"]={"action":"HMI_Stop_PB=true", "accepted":True,
                    "t_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
            trace=observe(label,a.duration,a.interval,sim_ep,mappings,a.out_dir)
            metadata[f"{label}_samples"]=len(trace)
            metadata[f"{label}_start"]=trace[0]["values"]
            metadata[f"{label}_end"]=trace[-1]["values"]
            (a.out_dir/"admission.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    except Exception as exc:
        metadata["admission_error"]={"type":type(exc).__name__,"message":str(exc),
            "t_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (a.out_dir/"admission.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
        raise
    finally:
        try:
            metadata["restoration_reset"]=reset_wd_storage_v2(ctrl_ep,sim_ep,mappings)
        except Exception as exc:
            metadata["restoration_error"]={"type":type(exc).__name__,"message":str(exc)}
            raise
        finally:
            metadata["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
            (a.out_dir/"admission.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":
    main()
