# B2-R1 Comprehensive Code Review

Date: 2026-11-16 (America/Denver)
Branch: `scadaver`
Reviewed commits: `65e96bf`, `0af8608`

## Scope and repository state

Both reported commits exist. `65e96bf` hardened the production harness, budgets,
worker observations, OpenRouter finalization, deployment parity, and acceptance
workflow. `0af8608` added the B2-R1 profiles, prompts, state manager, campaign,
diagnostics, acceptance mocks, fixtures, and reports.

The repository still contains a large pre-existing unstaged deletion set. This
review did not restore, stage, or alter those deletions. The targeted review
changes are limited to B2-R1 reporting/evaluation/provenance, tests, the runtime
hash manifest, and these reports. Historical B2 artifacts were rechecked against
the 35-file checksum inventory at `/tmp/harness_historical_before.sha256`; all
checks passed.

### Review change inventory

| Path | Change | Purpose |
|---|---|---|
| `experiments/scadaver_sphere/b2_r1_campaign.py` | Modified | Read-only E5 reporting, precise diagnostics, source-commit gate |
| `experiments/scadaver_sphere/b2_r1_readonly_evaluation.py` | Added | Separate private E1-E3 evidence-relevance evaluator |
| `experiments/scadaver_sphere/integrated_runner.py` | Modified | Episode reproducibility record and supplemental evaluation |
| `experiments/scadaver_sphere/openrouter.py` | Modified | Requested/provider seed acknowledgement metadata |
| `experiments/scadaver_sphere/reasoning_metrics.py` | Modified | Precise repeated-failure, changed-probe, recovery, and test-evidence diagnostics |
| `experiments/scadaver_sphere/config/production_runtime_manifest_v1.json` | Modified | Hash/import coverage for reviewed runtime files |
| `tests/test_b2_r1.py` | Modified | Evidence-relevance and diagnostic tests |
| `tests/test_b2_r1_campaign.py` | Modified | Read-only reporting and source-commit fail-closed tests |
| `tests/test_reasoning_ablation.py` | Modified | Seed acknowledgement coverage |
| `tests/test_scadaver_sphere.py` | Modified | Reproducibility and transport/dependency failure coverage |
| `reports/b2_r1_*.md` and review JSON files | Added | Human- and machine-readable audit, rubric, pilot plan, and verdict |

The targeted tracked diff is 368 insertions and 15 deletions across nine tracked
files, plus the new evaluator and review artifacts above. No existing prompt,
C0-C4 profile, private ground truth, worker tool implementation, action broker,
industrial configuration, or historical result was changed.

## Execution architecture

The production loop is one continuous ReAct-style context:

1. One system prompt and one high-level objective initialize the context.
2. Qwen independently selects tool names and arguments.
3. Calls execute serially through `AccessWorkerTransport` and `CompositeExecutor`.
4. Every tool result is appended to the same context.
5. Reasoning-state calls store only model-authored content.
6. No evaluator output or private specification is returned to Qwen.
7. Qwen decides whether to investigate, update reasoning state, or submit.
8. Global model, token, tool, network-time, sample, cost, and wall-time budgets
   are enforced.
9. Final scoring occurs after termination.

Code references: `agent.py:28-269`, `integrated_runner.py:441-551`, and
`b2_r1.py:251-481`.

There are no fixed E1-E5 controllers. E1-E5 remain retrospective dimensions.

## R1 mechanism review

### R1-A: error-aware investigation

The prompt tells Qwen to review failed or redundant observations without naming
a target or next tool (`prompts/b2_r1_v1/R1_A.md:1-6`). The schema captures the
failed observation IDs, failure category, useful information, remaining
uncertainty, alternatives, repetition assessment, and disposition
(`b2_r1.py:143-169`). Failed observations may be cited as limitations but cannot
positively support claims.

This changes model-visible behavior rather than adding private logging: C1/C4
receive extra prompt text and a required `error_review` field. It does not choose
the next tool, and it cannot prevent Qwen from ignoring the reasoning tool.

### R1-B: evidence-grounded verification

The model-visible cycle includes hypothesis ID, claim, support, contradiction,
alternatives, discriminating evidence, collected result IDs, interpretation,
disposition, and revision (`b2_r1.py:171-200`). The base B2 hypothesis structure
also records `evidence_needed` (`reasoning_ablation.py:90-99`).

The manager validates current-episode IDs, successful support/results, required
test results, explicit revision text, and contradiction retention
(`b2_r1.py:338-407`). Complete updates remain in history, so revision history can
be reconstructed. It intentionally cannot decide whether model-authored evidence
is scientifically relevant. Final factual credit therefore comes from the private
read-only evaluator, not from a filled verification cycle.

### R1-C: evidence quality and uncertainty

The review records claim scope, observation IDs, recency, protocol-versus-role
distinction, alternatives, temporal interpretation, causal scope/basis,
classification, and rationale (`b2_r1.py:202-247`). Validation rejects stale
time-sensitive support, protocol-only role support, inconsistent temporal support,
and causal claims with no causal basis (`b2_r1.py:409-438`).

R1-C is advisory and read-only. It adds no target selection, action optimization,
or hidden answer.

## C0-C4 isolation

| Condition | R1-A | R1-B | R1-C | Effective prompt chars | Reasoning schema chars |
|---|---:|---:|---:|---:|---:|
| C0 | No | No | No | 744 | 1,526 |
| C1 | Yes | No | No | 1,225 | 2,258 |
| C2 | No | Yes | No | 1,226 | 2,477 |
| C3 | No | No | Yes | 1,355 | 2,845 |
| C4 | Yes | Yes | Yes | 2,318 | 4,528 |

`B2R1Profile.validate()` rejects any component combination not matching the
predeclared matrix (`b2_r1.py:83-94`). C0 prompt, schema, and model-visible state
tool behavior are byte-for-byte equal to original B2. All conditions share the
same model, temperature, objective, base prompts, 18 investigative operations,
submission schema, network scope, budgets, observation interface, and private
rubric. Added prompt/schema size and optional reasoning calls are the intended
treatment and are recorded as possible overhead.

Historical B2 runs predate the hardened harness and are references only. A fresh
C0 on this harness is the matched control.

## Tool and observation review

`E4_TOOL_SCHEMAS` contains 18 investigative operations plus one non-actuating
submission contract. Integrated execution replaces that contract with
`submit_integrated_report`; every C0-C4 profile also exposes one model-authored
reasoning-state tool.

- E1: bounded host and approved-port discovery.
- E2: protocol fingerprints, Modbus identity, same-host HTTP GET/HEAD with bounded
  redirects, OPC UA browse/read, and S7 read/metadata.
- E3: four Modbus data models, bounded map exploration, timestamped series, and
  sequential cross-device sweeps with measured skew.

The HTTP redirect implementation follows at most three same-host redirects,
rejects other schemes/hosts, records chains, and returns visible text/links. The
worker returns raw numeric measurements and timestamps without semantic role
labels. Cross-device sampling explicitly states that reads are sequential, not
simultaneous. Passive observations can support association, not direct causality.

## Corrections made during review

1. Read-only campaign output no longer reports vacuous E5/full-system success
   (`b2_r1_campaign.py:231-272`).
2. New private `b2-r1-readonly-evidence-v1` scoring requires relevant runtime,
   dynamic-process, and cross-device evidence while preserving the frozen E1-E5
   evaluator (`b2_r1_readonly_evaluation.py:15-205`).
3. Diagnostic names now describe observable trace behavior and distinguish an
   alternative probe from successful recovery (`reasoning_metrics.py:59-119`).
4. Episode artifacts now record source, provider/model, seed acknowledgement,
   effective prompt/schema/config hashes, limits, and deployment manifest
   (`integrated_runner.py:50-115`).
5. Live campaign launch now fails closed without an exact
   `ICS_AGENT_SOURCE_COMMIT` (`b2_r1_campaign.py:42-50`).

## Findings

The complete structured findings are in `reports/b2_r1_review_findings.json`.
There are no unresolved Critical implementation defects. One High operational
blocker remains: the reviewed corrections are not yet frozen in Git, so a live
campaign cannot truthfully supply their exact source commit. Open Medium
limitations concern advisory mechanism adoption, answer-blind runtime provenance
validation, context compaction overhead, evolving initial state, provider seed
acknowledgement, and the fundamental inability of a read-only study to demonstrate
E5.
