# Harness Production Acceptance

Date: 2026-11-16
Acceptance version: `scadaver-production-acceptance-v1`

## Scope and invariants

The acceptance used deterministic model fixtures and deployed read-only checks. It made zero OpenRouter/Qwen calls, created zero action tokens, and performed zero deployed writes. The PLC program, process, prompts, scientific configs, evaluators, and historical results were not modified.

## Source state and changed files

- Branch: `scadaver`
- Base commit: `6027eb985dcd98cddfbc0bd2f49d17d5fbfdc8ea` (`feat(scadaver): harden reasoning ablation campaign`)
- Commit created by this task: none; the hardening changes remain uncommitted for review.
- Runtime changes: `acceptance.py`, `agent.py`, `budgets.py`, `enhanced_worker.py`, `integrated_runner.py`, `openrouter.py`, `operator_workflow.py`, `production_acceptance.py`, `reasoning_campaign.py`, `tool_results.py`, and `worker.py`.
- Configuration/documentation changes: `config/production_runtime_manifest_v1.json`, `docs/reasoning_campaign_operator_runbook.md`, and the reports under `reports/`.
- Regression changes: `tests/test_scadaver_sphere.py`.
- The much larger pre-existing deletion set in this working tree was not created, reverted, staged, or otherwise modified by this task.

## Source and deployment

- Branch: `scadaver`
- Access VM: `access` in `scadaverfingerprint.openplcwater.lgarcia`
- XDC: `scadaver.lgarcia`
- Runtime: Python 3.10, `pymodbus 3.6.9`
- Deployment manifest: `production-runtime-parity-v1`
- Production tool-result contract: `scadaver-tool-result-v2`
- E5 evaluator: `e5-causal-attribution-v3`
- Frozen integrated evaluator: `integrated-e1-e5-causal-v3`

## Validation summary

| Area | Evidence | Result |
|---|---|---|
| Unit/integration/fault/safety tests | Full repository suite | 251 passed, 1 skipped |
| Focused hardening regressions | Worker, HTTP, budget, campaign | 109 passed |
| Broader SCADAver/E5 suite | Temporal, workflow, transaction, reasoning | 179 passed |
| Deployment parity | File hashes, imports, dependencies, Python, interfaces, configs | PASS |
| Industrial services | OpenPLC Modbus+HTTP, simulator Modbus, engineering HTTP+OPC UA+S7, independent Modbus | 7/7 healthy |
| Controller lifecycle | One OpenPLC core, Modbus 1502, HTTP 8080, legacy controller inactive | PASS |
| Closed-loop behavior | 11 confirmed scan intervals; controller decisions observed; no inconsistent/indeterminate intervals | PASS |
| Worker | Refreshed worker, 19 frozen read-only tools, isolated sessions | PASS |
| Tool coverage | Every investigative tool plus submission exercised | PASS |
| Budget boundaries | Calls, tools, input/output reserve, retries, unknown usage, crossing | PASS |
| Broker | Disabled default, fixed allowlist behavior, one-request policy, no replay | PASS |
| E5 evaluator | Applied/unapplied, intervention response, recovery-only rejection, restoration | PASS |
| B0-B4 production path | Actual Access-VM runner with deterministic mock model | PASS |
| Historical integrity | Pre/post SHA-256 manifest for 35 multiseed files | PASS |

Canonical deployed manifest SHA-256: `76b6104327e98adfc6e7ef7f8db12db5a59b01782aab348812929176947f4067`.
The complete machine-readable gate output is `reports/harness_production_acceptance.json`.

## Mandatory readiness states

The canonical machine-readable result is produced by:

```bash
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.production_acceptance \
  --worker-url http://127.0.0.1:18777 \
  --out-dir /tmp/scadaver-production-acceptance-$(date -u +%Y%m%dT%H%M%SZ)
```

The gate reports `DEPLOYMENT_READY`, `TOOLS_READY`, `BUDGETS_READY`, `TESTBED_READY`, `BROKER_READY`, `EVALUATOR_READY`, `FULL_SMOKE_READY`, `SCIENTIFIC_CONFIG_FROZEN`, and `EXPERIMENT_READY`. Any incomplete component forces `EXPERIMENT_READY=false`.

## Production mock scenarios

| Scenario | Coverage |
|---|---|
| A complete E1-E5 path | Access-VM integrated runner and structured evaluation for B0-B4 fixtures |
| B generic HTTP failure | Timeout/error contract plus successful specialized fixture inspection |
| C redirects | Root no-follow and bounded same-host resource follow, loop/limit/cross-host tests |
| D error storm | Stable structured policy/protocol/rate/budget errors, no hidden actions |
| E budget exhaustion | Conservative admission, finalization reserve, recovery and hard termination tests |
| F stale state | Observed and post-quiescence phases remain separately timestamped |
| G wrong action | Private broker rejects; composite executor consumes only the model's request path |
| H correct fixture action | Isolated process fixture verifies transition, effect, restoration, and health |
| I wrong attribution | Recovery-only change is not objective achievement |
| J interruption | Durable transaction evidence and no automatic write replay |
| K stale deployment | Manifest mismatch fails before model creation or authorization |
| L missing worker/service | Transport/operator preflight fails closed with component status |

## Remaining limitations

- Provider input usage is known exactly only after response. The harness uses a conservative estimate and explicit crossing record; it cannot promise a provider-side billing hard cap that the provider does not enforce.
- Read-only association cannot establish packet direction or causal controller influence.
- A continuously evolving plant cannot provide identical initial state without a separately authorized reset/band-selection protocol.
- Passing deterministic mocks establishes engineering readiness, not Qwen scientific capability.
- This harness is not claimed to be universally bug-free; future source/dependency changes invalidate parity until reaccepted.
