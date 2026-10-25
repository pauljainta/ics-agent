"""Inspect or run a versioned PLCBench pilot cell.

Inspection is the default. Live episodes require --run and loopback endpoints.
The saved pilot bundles are never used as an output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from ics_agent.plcbench_compare import ComparisonConfig, run_comparison
from ics_agent.plcbench_container import SIDECAR_IMAGE
from ics_agent.plcbench_experiment import canonical_hash, experiment_identity, load_experiment, objective_from_spec
from ics_agent.plcbench_live import ModbusObserver, sensor_specs_from_map
from ics_agent.plcbench_matched_run import wd_command_menu, wt_command_menu
from ics_agent.plcbench_model_identity import probe_vllm, verify_live_target_deployment
from ics_agent.plcbench_native import build_ollama_generate, build_openai_generate
from ics_agent.plcbench_run import require_confinement
from ics_agent.plcbench_wt_restart import verify_wt_restart_proof

REPO = Path(__file__).resolve().parents[1]


def endpoint(value: str) -> tuple[str, int]:
    host, port = value.rsplit(":", 1)
    return host, int(port)


def map_for(spec: dict, sphere_root: Path | None) -> dict:
    ref = next(ref for ref in spec["sources"] if ref["path"].endswith("openplc_map.yaml"))
    root = REPO if ref["repo"] == "ics-agent" else sphere_root
    if root is None:
        raise ValueError("--sphere-root is required for WT configs")
    return yaml.safe_load((root / ref["path"]).read_text())["mappings"]


def _read_coil(client, address: int) -> int:
    result = client.read_coils(address, count=1)
    if result.isError() or not result.bits:
        raise RuntimeError(f"reset read failed at coil {address}")
    return int(result.bits[0])


def reset_for(spec: dict, controller: tuple[str, int], simulator: tuple[str, int], mapping: dict,
              restart_hook: Path | None = None):
    from pymodbus.client import ModbusTcpClient

    address = lambda name: int(mapping[name]["address"])
    kind = spec["reset"]["kind"]
    if kind == "wd-storage-v2-fixture":
        from ics_agent.plcbench_wd_storage_v2 import reset_wd_storage_v2
        return lambda: reset_wd_storage_v2(controller, simulator, mapping)
    if kind not in {"wd-running-v1", "wt-running-v1", "wt-chemistry-flush-v1"}:
        raise ValueError(f"unsupported reset contract: {kind}")

    previous_generation = None

    def reset():
        nonlocal previous_generation
        restart_proof = None
        if kind == "wt-chemistry-flush-v1":
            if restart_hook is None:
                raise RuntimeError("WT dosing requires a full simulator restart hook per episode")
            completed = subprocess.run([str(restart_hook)], check=True, capture_output=True,
                                       text=True, timeout=120)
            restart_proof = json.loads(completed.stdout)
            previous_generation = verify_wt_restart_proof(
                restart_proof, previous_generation=previous_generation,
                expected_simulator_sha256=spec["reset"].get("expected_simulator_sha256"),
                expected_controller_endpoint=f"{controller[0]}:{controller[1]}",
                expected_simulator_endpoint=f"{simulator[0]}:{simulator[1]}")
        client = ModbusTcpClient(controller[0], port=controller[1], timeout=3)
        if not client.connect():
            raise RuntimeError("reset cannot connect to controller")
        try:
            clear_tags = (["HMI_Stop_PB"] if kind == "wd-running-v1"
                          else spec["command_tags"])
            for tag in clear_tags:
                if mapping[tag]["register_type"] == "coil":
                    result = client.write_coil(address(tag), False)
                    if result.isError():
                        raise RuntimeError(f"reset failed to clear {tag}")
            if kind == "wd-running-v1":
                result = client.write_coil(address("HMI_Start_PB"), True)
                if result.isError():
                    raise RuntimeError("reset failed to start WD")
                time.sleep(3)
                if _read_coil(client, address("Sys_Running")) != 1:
                    raise RuntimeError("WD reset did not reach running state")
                return "Sys_Running=1"

            # WT start input is edge triggered; leave it released after starting.
            for _ in range(3):
                client.write_coil(address("HMI_Start_PB"), False)
                time.sleep(1)
                client.write_coil(address("HMI_Start_PB"), True)
                time.sleep(3)
                client.write_coil(address("HMI_Start_PB"), False)
                time.sleep(1)
                if _read_coil(client, address("SYS_RUNNING")) == 1:
                    break
            else:
                raise RuntimeError("WT reset did not reach running state")
            if kind == "wt-running-v1":
                return "SYS_RUNNING=1"

            tags = ["ChemTreat_Conductivity", "ChemTreat_ORP", "ChemTreat_pH"]
            observer = ModbusObserver(*simulator, sensor_specs_from_map(mapping, tags))
            deadline = time.monotonic() + 180.0
            while True:
                _, values = observer()
                clean = (values["ChemTreat_Conductivity"] < 340.0
                         and values["ChemTreat_ORP"] < 260.0
                         and values["ChemTreat_pH"] > 7.30)
                if clean:
                    return {"SYS_RUNNING": 1, "restart_proof": restart_proof, **values}
                if time.monotonic() >= deadline:
                    raise RuntimeError("WT chemistry flush timed out; episode not admitted")
                time.sleep(5)
        finally:
            client.close()

    return reset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sphere-root", type=Path,
                        default=Path(os.environ["SPHERE_USECASES_ROOT"]) if "SPHERE_USECASES_ROOT" in os.environ else None)
    parser.add_argument("--sandbox-python", default="python3")
    parser.add_argument("--container-image", default="plcbench-native:py311-pymodbus386",
                        help="explicit local image with pymodbus; build Dockerfile.native first")
    parser.add_argument("--controller", help="loopback host:port for --run")
    parser.add_argument("--simulator", help="loopback host:port for --run")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--start-rep", type=int, default=1,
                        help="first absolute pair number for a new, empty execution phase")
    parser.add_argument("--end-rep", type=int,
                        help="last absolute pair number for this execution phase")
    parser.add_argument("--model-url", help="Ollama host or OpenAI-compatible base URL")
    parser.add_argument("--deployment-json", type=Path,
                        help="target-model weight revision, vLLM version, runtime digest")
    parser.add_argument("--restart-hook", type=Path,
                        help="host executable returning verified wt-full-restart/v1 JSON after full simulator restart")
    parser.add_argument("--run", action="store_true", help="run live episodes; default is identity inspection")
    args = parser.parse_args()
    if args.restart_hook:
        args.restart_hook = args.restart_hook.resolve(strict=True)
    spec = load_experiment(args.config)
    mapping = map_for(spec, args.sphere_root)
    menu = wd_command_menu(mapping) if spec["workload"] == "water_distribution" else wt_command_menu(mapping)
    sandbox = {"python_executable": args.sandbox_python, "cpu_seconds": 10,
               "wall_timeout_sec": 15.0, "output_cap_bytes": 65536,
               "max_file_bytes": 16777216, "confinement": "linux-container",
               "container_image": args.container_image, "relay_sidecar_image": SIDECAR_IMAGE}
    if args.restart_hook:
        sandbox["restart_hook_sha256"] = hashlib.sha256(args.restart_hook.read_bytes()).hexdigest()
    if args.run:
        image_probe = subprocess.run(["docker", "image", "inspect", args.container_image,
                                      "--format", "{{.Id}}"], check=True, capture_output=True,
                                     text=True, timeout=10)
        sandbox["container_image_id"] = image_probe.stdout.strip()
    identity = experiment_identity(spec, repo_root=REPO, sphere_root=args.sphere_root,
                                   menu=menu, sandbox_config=sandbox)
    print(json.dumps({"cell_id": spec["cell_id"], "identity_sha256": identity["identity_sha256"],
                      "status": spec["status"]}, indent=2))
    if not args.run:
        return
    if spec["status"] != "qualified":
        parser.error("this cell has open validation gates and cannot be run")
    if spec["reset"]["kind"] == "wt-chemistry-flush-v1" and not args.restart_hook:
        parser.error("WT dosing requires --restart-hook for each episode")
    if not (args.controller and args.simulator and args.out_dir):
        parser.error("--run requires --controller, --simulator, and --out-dir")
    if args.reps <= 0:
        parser.error("--reps must be positive")
    if not 1 <= args.start_rep <= (args.end_rep or args.reps) <= args.reps:
        parser.error("phase range must be inside 1..--reps")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("--out-dir must be empty to preserve existing bundles")
    controller, simulator = endpoint(args.controller), endpoint(args.simulator)
    require_confinement(controller, "linux-container")
    model = spec["model"]
    deployment = dict(model)
    if model.get("role") == "target":
        if not args.model_url or not args.deployment_json:
            parser.error("target model requires --model-url and --deployment-json")
        deployment.update(json.loads(args.deployment_json.read_text()))
        live_probe = probe_vllm(args.model_url, model["model_id"],
                                token=os.environ.get("PLCBENCH_API_KEY"))
        verify_live_target_deployment(model, deployment, live_probe)
        # The frozen record defines the run identity; the fresh probe is retained
        # separately in bundle deployment evidence, since its timestamp and raw
        # vLLM metadata change on every request.
        identity["deployment"] = dict(deployment)
        deployment.update(live_probe)
    else:
        identity["deployment"] = dict(deployment)
    identity["identity_sha256"] = canonical_hash({k: v for k, v in identity.items()
                                                   if k != "identity_sha256"})

    def generate_for(_arm):
        if model["backend"] == "ollama":
            return build_ollama_generate(model["model_id"], host=args.model_url,
                                         temperature=model["temperature"],
                                         num_predict=model["max_tokens"])
        if model["backend"] == "openai":
            return build_openai_generate(model["model_id"], base_url=args.model_url,
                                         api_key=os.environ.get("PLCBENCH_API_KEY"),
                                         temperature=model["temperature"],
                                         max_tokens=model["max_tokens"])
        raise ValueError(f"unsupported backend: {model['backend']}")

    execution = spec["execution"]
    cfg = ComparisonConfig(
        workload=spec["workload"], controller_endpoint=controller,
        simulator_endpoint=simulator, mappings=mapping,
        command_tags=spec["command_tags"], sensor_tags=spec["sensor_tags"],
        objective=objective_from_spec(spec), menu=menu,
        reset=reset_for(spec, controller, simulator, mapping, args.restart_hook),
        native_profile=f"plcbench_native_{spec['cell_id']}",
        matched_profile=f"plcbench_menu_{spec['cell_id']}",
        sandbox_python=args.sandbox_python, container_image=args.container_image,
        confinement="linux-container", cell_id=spec["cell_id"],
        experiment=identity, **execution)
    _, table = run_comparison(cfg, generate_for=generate_for, reps=args.reps,
                              out_dir=args.out_dir, deployment=deployment,
                              start_rep=args.start_rep, end_rep=args.end_rep,
                              stop_on_error=True)
    print(table)


if __name__ == "__main__":
    main()
