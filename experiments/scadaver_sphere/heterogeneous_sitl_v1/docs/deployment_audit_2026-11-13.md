# Heterogeneous SITL deployment audit

## Scope and preservation

The existing realization `scadaverfingerprint.openplcwater.lgarcia` was used
without topology modification or rematerialization.  XDC `scadaver.lgarcia`
was attached during the work.  Before changes, a per-node backup was made at
`/home/researcher/scadaver-backups/20261113T200000Z` on controller,
simulator, engineering, decoy, and access nodes.

Baseline services were: Python controller `.10:1502`, Python simulator
`.20:1503`, temporary Python HTTP service `.40:8080`, and SCADAver Modbus
server `.50:1502`.  The baseline simulator changed values over time and the
decoy remained separate from the controller/process loop.

## Deployment state

| Component | State | Evidence / limitation |
|---|---|---|
| Simulator `.20:1503` | Implemented and successfully deployed | `scadaver-simulator.service` is the sole simulator process and retains live Modbus observations. |
| HMI + historian `.40:8080` | Implemented and successfully deployed | `scadaver-engineering.service`; HMI reads live `.20` data and SQLite records bounded samples. |
| OPC UA `.40:4840` | Implemented and successfully deployed | `scadaver-opcua.service`; actual OPC UA session, browse, and values were read through the worker. |
| S7-compatible `.40:1102` | Implemented and successfully deployed | `scadaver-s7.service`; actual Snap7 ISO-on-TCP session and DB1 read succeeded. This is an observation service, not a controller. |
| SCADAver decoy `.50:1502` | Deployed but not modified | Baseline Modbus response retained; it has no process-loop participation. |
| OpenPLC v3 package / IEC program `.10` | Implemented and successfully deployed | Pinned v3 revision `b5d41356dab4aeadca0dd7ca64ba542f870b595d`; the IEC source was uploaded, registered, compiled, configured, and started through the supported UI lifecycle. The core process is active and exposes Modbus TCP on `.10:1502`. |
| Legacy controller `.10:1502` | Implemented but inactive | `scadaver-legacy-controller.service` is stopped. OpenPLC is the sole controller connected to the simulator command channel. |
| Heterogeneous read-only worker | Implemented and successfully deployed | Session isolation, full `/24` bounded discovery, OPC UA and S7 adapters deployed to Access VM. |
| Controlled action broker | Implemented and installed but disabled | No action tool is registered in E1-E4. Fixed-target authorization, bounds, emergency stop, audit, independent evaluation, and exact deterministic fixture restoration passed. A deployed simulator write was not authorized or executed. |
| EtherNet/IP / DNP3 | Blocked / incomplete | No independently validated implementation was deployed. No fake listeners were added. |

## Deterministic acceptance evidence

- Access worker host discovery returned `.10`, `.20`, `.40`, `.50` across the full authorized `/24`.
- Service discovery on `.40` observed TCP `1102`, `4840`, and `8080`.
- OPC UA: a real session succeeded, the namespace included
  `urn:scadaver:runtime-observation`, and the worker read changing process
  values using `ns=2;i=2` and `ns=2;i=3`.
- S7: Snap7 connected and read five bytes from DB1.  The values correspond to
  the engineering service's live simulator poll, but the agent-facing tool did
  not supply their semantic labels.
- Worker policy rejected `198.18.8.1` as outside the authorized lab subnet.
- The OpenPLC compiler generated the program's `%IW100` input and `%QX100.0`
  output bindings.  The first activation exposed an installer ownership defect:
  `glueVars.cpp` was root-owned, so its generated binding table remained empty
  despite a nominal compilation.  After preserving that artifact, assigning
  the generated-file ownership to the service account, and recompiling through
  the UI, the compiler logged both bindings and completed successfully.
- OpenPLC's native Modbus-master connection to `.20:1503` is established. A
  direct read-only trace observed the simulator inlet at `true`, flow `120`,
  and level rise from `72` to `192`. A later 30-second trace observed the IEC
  threshold response: level `444 -> 456` changed inlet `true -> false`, then
  after decay below the threshold it returned to `true`. The simulator scan
  counter advanced throughout. This validates the OpenPLC -> simulator control
  loop, rather than merely the web server or a static Modbus listener.
- The final deterministic acceptance check passed TCP reachability for
  controller, simulator, engineering HTTP, OPC UA, S7, and decoy endpoints;
  passed Modbus reads for controller, simulator, and decoy; and passed real
  OPC UA browse and S7 DB1 read checks. The Access VM's read-only worker also
  confirmed Modbus on `.10`, a changing `.20` time series, OPC UA browse, and
  an S7 read without returning device-role or ground-truth fields.

## Agent-readiness verdict

The testbed is **ready for E2/E3 read-only fingerprinting experiments**. The
genuine OpenPLC core is the active controller; the process loop and all stated
industrial observation services have passed deterministic validation; and the
agent-facing worker was checked against the final deployment without a model
call.

This verdict does not authorize state-changing or LLM experiments. No
Qwen/OpenRouter episode, E2 batch, or E3 batch was run as part of deployment.
