# SPHERE OpenPLC Backend

`sphere_openplc` makes the SPHERE-hosted OpenPLC water-treatment runtime a
first-class `ics-agent` environment backend.

The backend keeps scenario variation in `ics-agent` and keeps the portal-side
runtime pinned to one durable realization:

- experiment: `openplcwater.lgarcia`
- realization: `wargamestable.openplcwater.lgarcia`
- revision: `7054831a885ac69f8cc5ffb2cccc432e892dadb6`

The backend can also target a derived Option B realization by overriding the
environment config with:

- `merge_server: merge.example.invalid`
- `realization: optionbsmoke`
- `revision: <derived optionb revision>`
- `xdc: optionb0`
- `access_host: access`
- `engineering_host: engineering`
- `remote_bind_host: 0.0.0.0`

On this server, the installed `mrg` requires the SPHERE server to be passed
explicitly, so the backend now injects `-s merge.example.invalid` whenever
`merge_server` is configured.

What the backend manages locally and non-interactively:

- verifies `mrg whoami`
- reuses or creates the configured realization and XDC
- attaches the XDC to the stable realization
- writes local-only SSH transport state under `~/.cache/ics-sphere/`
- stages the native PLC runtime payload to the remote controller and simulator
- ensures remote native controller and simulator services are listening
- creates fresh local SSH forwards for the controller and simulator
- starts a fresh local Modbus bridge for each run

What still stays scenario-driven in `ics-agent`:

- `hmi_start`
- `hmi_stop`
- `hmi_estop`
- `select_process`
- `set_manual_valve`
- `set_sensor`

Representative remote subset:

```bash
python -m ics_agent.cli run /home/researcher/Agentic_ICS/ics-agent/configs/experiments/sphere_phase1_representative.yaml
```

Full remote `18`-scenario attack suite:

```bash
python -m ics_agent.cli run /home/researcher/Agentic_ICS/ics-agent/configs/experiments/sphere_attack_suite18.yaml
```

Remote result artifacts are written under:

`/home/researcher/Agentic_ICS/Artifact/Results/remote_sphere_openplc/`

Backend validation used during development:

```bash
python -m pytest -q /home/researcher/Agentic_ICS/ics-agent/tests/test_environment.py /home/researcher/Agentic_ICS/ics-agent/tests/test_orchestrator.py
```
