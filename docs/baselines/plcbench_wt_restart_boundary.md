# WT per-episode restart boundary

The `wt-chemistry-dosing-v2` descriptor is a separate local PLCBench-inspired cell. It retains the pilot's 3 s/10 s chemistry objective but pins the WT simulator source and uses the actual map-derived bridge. The full 600 s zero-action/reference evidence in `runs/qualifications/wt_chemistry_pilot_v1/attempt_2/` admits the **mechanism only**. Its deterministic two-command menu control must be reported in any later comparison.

`plcbench_batch.py` now requires a `wt-full-restart/v1` proof from its host-side `--restart-hook` before each WT dosing episode. The verifier checks a changed simulator process generation, the expected simulator ST hash, declared source-bound damage basis, clean conductivity/pH/ORP/permeate readback, running state, and closed P2 dosing state. Missing or dirty proof fails admission, and the comparison runner records the failed attempt in `attempts.json`. The old pilot descriptor and saved pilot bundles are unchanged.

The local executable `scripts/plcbench_wt_restart_hook.py` uses Docker CLI on the **host** to restart only the isolated `plcbench-wt-chem` simulator, wait for it, reconnect its bridge, and invoke the readback sidecar. It writes a retained `restart.json`, command logs, and probe record under `runs/qualifications/wt_restart_hook/`; stdout contains only the proof JSON expected by the batch runner. The native-code child is not given a Docker socket. Controller, simulator, and bridge remain on the internal Compose network without host ports. Two fixed-target trusted relays in `wt_chemistry_loopback.compose.yaml` publish only `127.0.0.1:19502` and `127.0.0.1:19503`; their source is hashed in the experiment descriptor. The hook checks Docker's actual bindings and receives host-side Modbus replies before issuing a proof. The batch runner binds that proof to its exact controller and simulator endpoints.

Two sequential live-local hook calls passed on the isolated pair. Their `StartedAt` values differed, and both returned conductivity 300 µS/cm, pH 7.5, ORP 200 mV, and permeate conductivity about 1.50 µS/cm. The source initializes private `membrane_damage` to zero on process restart; the permeate reading is consistent with that zero under the hashed experimental equation. It is not a direct memory measurement or physical validation. The pair was stopped afterward, with volumes and evidence retained.

To run the hook on the isolated pair:

```sh
PLCBENCH_WT_SPHERE_ROOT=/path/to/pinned/sphere-usecases \
  scripts/plcbench_wt_restart_hook.py
```

An initial 2026-10-15 attempt tried to publish ports directly from Docker Desktop's `internal` network. Compose accepted the mapping, but both host endpoints refused connections; its original proof is explicitly invalidated by `host_endpoint_assessment.json` in that attempt's evidence directory. The trusted-relay attempt then passed host readback and clean-state checks. A batch-runner preflight called the same hook twice through the loopback endpoints, checked Docker's actual port bindings, and recorded two distinct clean simulator generations at `runs/qualifications/wt_restart_hook/batch_boundary_20261015_v2.json`. This is local reset/endpoint evidence, not a measured or model episode.

For a future batch, set `PLCBENCH_WT_SPHERE_ROOT`, pass the executable as `--restart-hook`, and use `--controller 127.0.0.1:19502 --simulator 127.0.0.1:19503`. The host batch process must have Docker CLI access; the model and native child do not. A target Qwen/vLLM deployment identity was captured on 2026-10-15, but that temporary job does not qualify this cell. The cell remains `unqualified` for measured runs until these reset and relay checks are repeated in the final execution environment with a fresh deployment identity and the separate deterministic control.
