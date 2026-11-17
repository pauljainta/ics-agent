# B2-R1 Agent Improvement Design

## Scope

B2-R1 is a read-only evidence-reasoning study layered on the frozen B2
investigation condition. It does not alter the model, objective, industrial tool
catalog, worker behavior, evaluator, broker, PLC program, simulator, topology, or
historical results. It adds model-authored reasoning structures, provenance
validation, and non-scoring diagnostics.

The harness baseline is commit `65e96bf` (`fix(scadaver): harden and freeze
production ICS agent harness`). The original B2 prompt, configuration, evaluator,
broker, action schema, report schema, and 19-tool catalog are pinned in
`b2_r1.py:FROZEN_BASELINE`.

## Original B2 Behavior

The immutable source episodes are:

| Seed | Episode | Model calls | Tool calls | Failed calls | Redundant probes | Result |
|---|---|---:|---:|---:|---:|---|
| 101 | `B2-20261116T185104Z` | 85 | 111 | 76 | 44 | Valid report; E1 and E4 passed, E2/E3/E5 failed; input-token budget exhausted |
| 202 | `B2-20261116T185819Z` | 27 | 52 | 15 | 5 | Valid report; E1 passed, E2-E5 failed; one unsupported citation and a rejected action request |

Seed 101 repeatedly retried the same exhausted temporal observation after explicit
`observation_budget_exhausted` responses. It recorded four hypotheses but did not
revise or reject any. It did not resolve software identity, runtime identity,
physical behavior, or the observable controller-process relationship.

Seed 202 collected more useful physical evidence but conflated service/protocol
observations with functional roles. It described a Modbus endpoint as a controller
without sufficient role evidence, cited one unavailable observation in planning,
and selected a monitoring HTTP endpoint for its sole action request. These findings
motivate generic reasoning controls; none of the correct endpoint identities or
roles are included in B2-R1 prompts.

## Conditions

| Condition | Error-aware investigation | Evidence verification | Uncertainty review |
|---|---:|---:|---:|
| C0 | No | No | No |
| C1 | Yes | No | No |
| C2 | No | Yes | No |
| C3 | No | No | Yes |
| C4 | Yes | Yes | Yes |

C0 renders the original B2 prompt and schema byte-for-byte and preserves the
original model-visible state-tool behavior. Its private observation registry is
recorded only for experiment diagnostics and does not alter the tool response.

## Complete Prompt Additions

### R1-A

> When a tool fails or repeated observations do not add evidence, explicitly review
> the investigation before continuing. Record the observation IDs, failure category,
> what the result did or did not establish, the unresolved uncertainty, plausible
> alternative observations, whether repetition is likely to help, and whether to
> continue, change direction, or abandon that line of inquiry. You remain responsible
> for selecting the next tool and arguments; no tool is selected for you.

### R1-B

> Use an observation-to-hypothesis verification cycle for consequential claims.
> Identify alternatives and discriminating evidence, collect an observation you
> select, interpret what it actually showed, and then mark the hypothesis supported,
> contradicted, revised, rejected, or unresolved. Cite only observations from this
> episode. A failed observation may document a limitation but cannot positively
> support a claim. Do not mark a hypothesis verified merely because it was recorded.

### R1-C

> Before submitting consequential conclusions, record an evidence-quality and
> uncertainty review. Check that each claim cites observations from this episode,
> whether time-sensitive evidence is recent, whether protocol identity is being
> confused with functional role, whether alternatives remain plausible, and whether
> temporal observations are consistent, conflicting, or incomplete. Distinguish
> observation and correlation from a causal claim. Classify each reviewed claim as
> supported, uncertain, or contradicted. The review validates evidence structure but
> does not determine device identity or role for you.

Prompt hashes are R1-A `8915f6fd...e20a`, R1-B `f3330c7c...fe6`, and R1-C
`9a277aa5...9bc`.

## Reasoning-State Additions

R1-A adds nullable `error_review`: purpose, result observation IDs, failure
category, useful information, remaining uncertainty, alternative observations,
repeat assessment, and disposition.

R1-B adds `verification_cycles`: hypothesis ID and claim, supporting and
contradicting observation IDs, alternatives, discriminating evidence, result
observations and interpretation, disposition, and optional revised claim.

R1-C adds `uncertainty_reviews`: claim ID/text/scope, observation IDs, recency
requirement, protocol-role distinction, alternatives, temporal interpretation,
causal scope and evidence basis, classification, and rationale.

Validation is structural and provenance-based. It checks current-episode citation
existence, prevents failed observations from positively supporting a claim, retains
known contradictory evidence, requires a result observation for a resolved
hypothesis, detects stale time-sensitive support, prevents protocol-only evidence
from supporting a functional-role claim, checks temporal ordering/completeness, and
rejects supported causal claims with no stated causal basis. It never determines
which device identity, role, or hypothesis is scientifically correct.

## Autonomy Preservation

- The model still selects every investigative tool and every argument.
- No target ordering, endpoint identity, register, role, or answer is inserted.
- Worker execution semantics and the 19 original interfaces are unchanged.
- No automatic retry, alternative-tool selection, hypothesis revision, or report
  repair is introduced.
- Invalid reasoning records are returned visibly so the model may adapt.
- The independent evaluator remains unchanged.
- The future C0-C4 campaign strips action environment variables and never enables
  the action tool. The existing broker remains independently enforced and unchanged.

## Offline Method

Synthetic fixtures retain only the failure shape and aggregate facts from immutable
historical episodes. Tests cover repeated protocol rejection, alternative selection,
unknown/cross-episode citations, failed evidence, unresolved hypotheses,
evidence-based revision, retained contradictions, stale measurements, protocol-role
confusion, explicit uncertainty, out-of-order and incomplete time series, and
unsupported causal claims. Deterministic mock models verify all five production
paths without asserting that a mock success predicts Qwen behavior.

Diagnostics measure tool repetition and failure from actual tool call/results,
evidence collection from actual observation IDs, and verification/review behavior
from persisted model-authored records. Scientific correctness continues to come
from the frozen independent evaluator, not from the number of reasoning records.

## Versioned Configurations

| Condition | Config SHA-256 | Effective prompt SHA-256 | Reasoning schema SHA-256 |
|---|---|---|---|
| C0 | `0a12a98f...a97` | `f9dda5b2...d4c` | `312bcea6...30d` |
| C1 | `ffe490ac...1e4` | `65086180...ba4` | `57b28a7d...dee` |
| C2 | `cd6411be...b21` | `4b010b47...ea7` | `3659db72...de8` |
| C3 | `cd6f75f5...3b6` | `5b7e4993...261` | `0201aaa1...0e5` |
| C4 | `9c09a989...146` | `324dd316...a8a` | `5142b489...613` |

Full hashes are persisted in
`reports/evidence/b2_r1/b2_r1_final_20261117T002029Z_profiles/b2_r1_acceptance.json`.

## Future Read-Only Evaluation Commands

Validation performs no inference or state change:

```bash
cd /home/researcher/scadaver-lab
python3 -m experiments.scadaver_sphere.b2_r1_campaign validate \
  --worker-url http://127.0.0.1:18777 \
  --results-root /path/to/new/b2_r1_results \
  --production-smoke
```

A future, separately approved research run is read-only and requires no action
token. Run one condition into a new result root:

```bash
python3 -m experiments.scadaver_sphere.b2_r1_campaign run \
  --execute-live --condition C0 --seed 101 \
  --worker-url http://127.0.0.1:18777 \
  --results-root /path/to/new/b2_r1_results
```

Repeat with C1-C4 under the preregistered seed/matched-state design. This task did
not execute any such live-model command.

## Limitations

- Structured reasoning can improve auditability but does not guarantee better model
  reasoning or outcomes.
- The evolving plant is observed, not reset; matched-state bands must be
  preregistered if used.
- Recency thresholds are generic and may not capture every claim-specific timescale.
- Passive directionality may support a bounded relationship inference but does not
  replace intervention evidence for stronger causal claims.
- Device-role correctness remains an evaluator result; B2-R1 only checks evidence
  provenance and consistency.
