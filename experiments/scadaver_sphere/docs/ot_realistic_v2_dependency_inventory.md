# OT Realistic V2 Dependency Inventory

This inventory was produced from the local `scadaver` working tree before V2
deployment. It records fixed-address and architecture dependencies that must not
be silently inherited by the new topology.

## Original Topology and Lifecycle

| Area | V1 dependency | V2 treatment |
|---|---|---|
| SPHERE model | `topology/scadaver_openplc_water_fingerprint.py` places `.10`, `.20`, `.30`, `.40`, and `.50` on one link | New model uses one OT link and one disjoint process link |
| XDC/materialization | `scadaver.lgarcia` → `scadaverfingerprint.openplcwater.lgarcia` | Preserved; V2 realization name remains operator approval pending |
| Operator workflow | `operator_workflow.py` identifies `.20:1503` as simulator | V1-only; V2 uses viewpoint-specific acceptance code |
| OpenPLC master | `openplc_v3/deployment/mbconfig.cfg` polls `.20:1503` | V2 polls `198.18.6.2:1503` and localhost supervisory telemetry |
| IEC logic | `%IW100` drives `%QX100.0` | Preserved, with a bounded expiring supervisory branch in a new ST program |

## Agent Observation Dependencies

| Area | V1 dependency | V2 treatment |
|---|---|---|
| Worker policy | `access_worker_service.py` and `n3_worker_service.py` allow port `1503` on `198.18.4.0/24` | New worker allows only `1102`, `1502`, `4840`, and `8080` |
| Deterministic acceptance | `acceptance.py` reads `.20:1503` and cross-samples `.10/.20` | V2 checks controller telemetry and monitoring data; private data is operator-only |
| Evidence sufficiency | `evidence_sufficiency.py` treats `.20` as the process endpoint | V2 evaluator represents the physical process as a logical entity |
| N3 configuration | N3 profiles include port `1503` | New inactive V2 profile removes it |
| Skills/prompts | N3 wording permits a directly reachable simulator | V2 overlay explains that physical processes need not be network hosts |

## Monitoring Dependencies

| Area | V1 dependency | V2 treatment |
|---|---|---|
| HMI/historian | `engineering_runtime.py` polls `.20:1503` | New runtime polls PLC input register `100` and the IEC-visible output register `1`; direct OpenPLC output verification uses flattened coil `800` for `%QX100.0` |
| OPC UA | V1 reads simulator state directly | V2 publishes PLC-observed values |
| S7 | V1 reads simulator state directly | V2 publishes the same PLC-observed values |
| Service units | V1 sets `SCADAVER_SIMULATOR_HOST=198.18.4.20` | New units set only `SCADAVER_CONTROLLER_HOST=198.18.4.10` |

## Action and Evaluation Dependencies

| Area | V1 dependency | V2 treatment |
|---|---|---|
| Broker | `heterogeneous_sitl_v1/action_broker.py` writes simulator coil `0` | New broker accepts only a controller endpoint and never writes the simulator |
| Transaction | `e5_transaction.py` quiesces OpenPLC before direct process writes | V2 leaves OpenPLC running and verifies command application through its output |
| Evaluator | `integrated_evaluation.py` expects `.10 → .20` | New evaluator expects `.10 → process:<logical-id>` |
| Ground truth | V1 specs list `.20` as a host/service | V2 private spec has three hosts and six services; no simulator endpoint |
| Action schema | `bounded_binary_output` accepts arbitrary endpoint fields subject to private validation | V2 exposes `bounded_controller_output`; the private process address never appears |

## Deliberately Preserved V1 Paths

All V1 topology, worker, prompt, evaluator, broker, result, and deployment files
remain available. The only shared runtime changes are an optional
`--bind-address` argument and `pymodbus` compatibility aliases in
`plant_runtime.py`. Its default remains `0.0.0.0`, verified by regression test.
