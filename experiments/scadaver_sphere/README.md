# SCADAver + SPHERE Read-Only ICS Fingerprinting

This is a separate experiment for read-only inventory and device-authenticity
assessment in an authorized isolated SPHERE lab. It is not a PLCBench runner and
does not reuse any previous realization.

The completed E1-A workflow remains on `fingerprinting_basic`. The opt-in
[`fingerprinting_enhanced` harness](docs/enhanced_fingerprinting_harness.md)
adds bounded value-bearing Modbus reads, short temporal sampling, conservative
identity probing, and fixed-root HTTP inspection without changing E1-A results,
the topology, the plant, or the decoy.

The separate E1-B role-inference orchestration and exact manual commands are in
[docs/e1b_runbook.md](docs/e1b_runbook.md). It is intentionally not executed by
the implementation or offline test suite.

The authoritative deployed E1-E5 architecture, master acceptance results, and
remaining E5 gate are recorded in
[the master readiness report](docs/master_readiness_report_2026-11-14.md). The
earlier [readiness report](docs/final_readiness_report_2026-11-13.md) is retained
as historical engineering evidence. Exact
commands are in [the E1-E5 operator runbook](docs/e1_e5_operator_runbook.md),
and the answer-free human-equivalent condition is documented in
[the blind analyst package](docs/blind_human_investigation_guide.md).

The opt-in OpenPLC runtime/process path is separate again: it adds a new
OpenPLC v3 realization template, cross-device timestamped sampling, and two
fingerprinting conditions without modifying E1-A/E1-B or the legacy Python
controller. Read [the deployment runbook](openplc_v3/docs/deployment_and_validation.md)
before materializing it.

## Safety boundary

The model receives only typed tools. The deployed worker validates literal IPv4
targets against the authorized `198.18.4.0/24` lab CIDR, restricts ports to
`1102`, `1502`, `1503`, `4840`, and `8080`, limits Modbus reads to ten values,
and has no write, shell, SSH, credential, exploit, reset, or Internet tool.
State-changing E5 infrastructure is a separate private broker and is never
registered in E1-E4.

## Offline checks

```bash
cd /home/researcher/Agentic_ICS/ics-agent
./.venv/bin/python -m experiments.scadaver_sphere.scripts.audit /tmp/scadaver-audit.json
./.venv/bin/python -m pytest -q tests/test_scadaver_sphere.py
```

## Live gate

Live model runs require a reviewed configuration, the deployed loopback worker,
and an environment-only API key. Never put a key in arguments or results. Use
the versioned E1-E4 runners in the operator runbook; the older generic command
below remains for the original D1 workflow only:

```bash
./.venv/bin/python -m experiments.scadaver_sphere.scripts.run \
  --config /secure/scadaver-d1.json --mode agent \
  --allow-live-worker --allow-live-openrouter \
  --out-dir /secure/results/scadaver-d1-run-001
```

The topology model is [topology/scadaver_sphere.py](topology/scadaver_sphere.py).
It is a template only until a SPHERE operator validates image names, the current
water-treatment controller/simulator deployment, bridge placement, and the new
realization plan.

## Deployment prerequisites

Before a live realization is proposed, independently verify the SCADAver release
at the pinned source revision, current SPHERE model syntax and node images,
controller/simulator staging steps, access-node egress isolation, and the hidden
ground-truth evaluator inputs. The private live configuration and its result
metadata must record the model identifier, selected configuration, lab CIDR,
tool limits, and SCADAver binary revision; they must never record API-key values.
