# OpenPLC v3 Additive Deployment

## Scope and version

This bundle targets a **new** realization built from
`topology/scadaver_openplc_water_fingerprint.py`. It never changes the legacy
Python-controller realization used by E1-A/E1-B. The selected runtime is
OpenPLC v3 at upstream commit
`b5d41356dab4aeadca0dd7ca64ba542f870b595d`.

OpenPLC v3 was selected because its upstream source exposes a native
Modbus-master configuration (`mbconfig.cfg`) and a configurable Modbus server,
which map directly to the existing simulator without an external decision
bridge. It is marked end-of-life upstream; the pin is therefore important for
reproducibility, not an endorsement for production use.

## I/O mapping

`mbconfig.cfg` maps the simulator's holding register 0 into OpenPLC's native
remote-I/O input offset 100. The Structured Text program reads `%IW100` and
writes `%QX100.0`; native OpenPLC maps that output to remote Modbus coil 0.
The controller decision exists only in `water_level_controller.st`.

## Operator sequence

1. Materialize the **new** OpenPLC topology; retain the legacy realization.
2. Stage this repository to each node at `$HOME/scadaver-lab`.
3. On controller `.10`, run `bootstrap-openplc-node.sh controller`; on the
   other nodes run it with their existing role names.
4. Confirm `systemctl status openplc-controller.service` and verify that port
   1502 answers bounded Modbus reads. The OpenPLC web UI listens on 8080; it is
   not exposed as an agent management surface. The worker only permits `GET /`.
5. From an operator host, run:

```bash
python3 validate_openplc_non_llm.py --controller 198.18.4.10 --simulator 198.18.4.20
```

6. Inspect the returned samples manually. They prove reachability and changing
   observations, but not causality by themselves. Confirm service logs show the
   native OpenPLC runtime process and successful Modbus-master connection before
   declaring that OpenPLC controls the simulator.

## Rollback

On the OpenPLC realization, run
`rollback_to_legacy_python_controller.sh` only if that realization has the
legacy controller service staged. It disables rather than deletes OpenPLC.
The original legacy realization requires no action.

## Agent runs, after operator validation

Create a reviewed private copy of `config/runtime_process_live.example.json`
with both live gates set to `true`, then use either condition:

```bash
SCADAVER_WORKER_URL=http://127.0.0.1:18777 \
  .venv/bin/python -m experiments.scadaver_sphere.runtime_process_batch \
  --config /secure/runtime-process.json --condition network --episodes 1 \
  --out-dir /secure/results/runtime-process-network-001
```

Use `--condition service` only after verifying that the selected safe HTTP root
contains genuine unauthenticated metadata. Do not run a batch until the
non-LLM validation is recorded.
