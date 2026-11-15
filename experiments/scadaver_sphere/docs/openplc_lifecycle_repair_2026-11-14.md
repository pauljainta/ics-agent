# OpenPLC lifecycle repair and validation

Date: 2026-11-14 (America/Denver)

## Verdict

The controller lifecycle defect is repaired. The intended deployment is
restored with OpenPLC active, the legacy controller inactive, one compiled PLC
core, HTTP `8080`, runtime RPC `43628`, and Modbus TCP `1502`. A clean systemd
restart and the E5 broker's no-actuation quiesce/recovery rehearsal both pass.

The testbed is ready for a **freshly authorized E5 acceptance test**. No E5
simulator write and no Qwen/OpenRouter request were performed during this
repair. The complete autonomous E1-E5 experiment remains blocked until the
separately authorized deployed E5 test passes.

## Root cause

The active IEC program was intact and correctly registered:

- active file: `339003.st`
- program: `SCADAver water level controller`
- compiled core SHA-256:
  `120035e2ed832e12dcc128ac63eddca89eecc6c4a7e566c8c3e4923b8a0d4d9d`
- configured Modbus server port: `1502`

The defect was lifecycle-specific. Database setting `Start_run_mode` was
`false`, while the systemd unit launched only `webserver.py`. Consequently,
systemd reported the web service active after restart even though
`core/openplc`, runtime RPC, and Modbus `1502` were absent.

Starting `core/openplc` alone was also insufficient: OpenPLC's management layer
enables the Modbus server through its local runtime RPC. The repaired supervisor
therefore starts the registered compiled core and issues the same
`start_modbus(1502)` RPC used by OpenPLC's management code. It does not modify
the Programs table, active-program file, credentials, or IEC program.

## Changes

- `openplc_lifecycle.py` validates activation, supervises the OpenPLC web and
  core processes, enables Modbus through runtime RPC, monitors both children,
  and sends systemd readiness/stopping notifications.
- `scadaver-openplc-controller.service` now uses `Type=notify` and runs a strict
  `ExecStartPost` validator only after the supervisor reports readiness.
- `validate_openplc_lifecycle.py` requires a registered active program, exactly
  one core process, HTTP `8080`, RPC `43628`, Modbus `1502`, OpenPLC service
  active/activating, and legacy controller inactive.
- `validate_heterogeneous_sitl.py` now includes the seventh service
  (`.10:8080`), controller lifecycle evidence, and a 12-sample synchronized
  closed-loop trace.
- The E5 broker controller coordinator remains fail-closed. Controller recovery
  failure activates its persistent emergency-stop latch rather than accepting
  the legacy controller as research-ready.

## Live evidence

### Controller lifecycle

- `systemctl restart scadaver-openplc-controller.service`: exit `0`, `0.784 s`.
- `NRestarts=0`, `ActiveState=active`, `SubState=running`.
- exactly one `/opt/openplc-v3/webserver/core/openplc` process.
- ports `8080`, `43628`, and `1502` open.
- OpenPLC active; legacy controller inactive.

### Closed-loop control

The final 12-sample trace passed all checks:

- controller input register `100` matched simulator level within one process
  scan for every valid sample;
- both inlet states were observed;
- simulator scan counters advanced;
- level increased after inlet-on and decreased after inlet-off, within the
  bounded timing tolerance;
- controller, simulator, decoy, HTTP, OPC UA, S7, and engineering endpoints all
  remained reachable.

### Broker recovery rehearsal

The operator-only E5 coordinator was exercised without any simulator action:

- before: OpenPLC active, legacy inactive, `1502` open;
- quiesced: OpenPLC inactive, legacy inactive, `1502` closed;
- recovered: OpenPLC active, legacy inactive, `1502` open;
- post-recovery simulator scans advanced and controller input tracked process
  level within one scan;
- `action_attempted=false`.

### Agent-facing acceptance

- deployed tool coverage: pass (`19/19`);
- seven service checks: pass (`7/7`);
- E3 evidence-sufficiency suite: pass for all identifiable conclusions, with
  packet direction and independent causality still correctly marked not
  identifiable under network-only access;
- deterministic continuous E1-E4: 4/4 hosts, 7/7 services, 5/5 protocol
  endpoints, OpenPLC identity pass, E3 pass, E4 pass, 8/8 valid citations, and
  `read_only_workflow_success=true`;
- no inference requests and no action.

## Regression results

- SCADAver-focused: `69 passed`.
- Full repository: `145 passed, 1 skipped`.
- `git diff --check`: clean.

The new regression test explicitly rejects a state where the OpenPLC systemd
service is active and HTTP `8080` is available but the core process and Modbus
listener are absent.

## Evidence artifacts

Stored under:

`Artifact/Results/scadaver_sphere/openplc_lifecycle_repair_20261114/`

- `controller_lifecycle.json`
- `deployed_acceptance.json`
- `broker_recovery_rehearsal.json`
- `tool_coverage.json`
- `e3_evidence_sufficiency.json`
- `integrated_e1_e4.json`

## Remaining gate

The lifecycle repair does not reuse the previous E5 authorization. A new
authorization is required before one deployed simulator-only toggle can be
performed. Qwen/OpenRouter must remain disabled until that acceptance passes.
