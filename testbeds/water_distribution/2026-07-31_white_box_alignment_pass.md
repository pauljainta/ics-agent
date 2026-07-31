# Water-Distribution White-Box Alignment Pass

Date: `2026-07-31`

## Scope

This pass only strengthened the water-distribution `white_box_attacker__agentic_defender` family. The paper repo was not touched.

## Attacker-Side Strength Gaps Found

Relative to the current water-treatment white-box setup, the water-distribution white-box attacker was weaker in two distinct ways:

1. Runtime strength was lower.
   - Distribution used `qwen_langchain` instead of the stronger treatment-side `qwen_langgraph` path.
   - Distribution used `max_new_tokens: 64` instead of the treatment-side `160`.
   - Distribution used a slower action/reasoning cadence: `action_cooldown_sec: 2.0`, `reasoning_cooldown_sec: 12.0`, `reasoning_refresh_sec: 18.0`, and no explicit `note_cooldown_sec`, while treatment used `0.5`, `1.0`, `2.0`, and `2.0`.
   - Distribution did not explicitly align `attacker_candidate_window` to the treatment-side window of `8`.

2. Operational leverage inside the bounded white-box surface was lower.
   - The distribution white-box prompt was materially weaker than the current treatment white-box prompt.
   - The white-box system-knowledge and prior-generation path in code was still treatment-shaped instead of distribution-shaped.
   - Outside the authored constrained `wd-attack-supply-pump-max` scenario, the distribution white-box candidate path did not generate general `set_actuator` attack candidates, which collapsed most trajectories into sensor-only or nearly one-action behavior.
   - `set_actuator` was not mapped into the white-box attacker surface taxonomy, which weakened surface-memory pressure and follow-on escalation.

## Alignment Changes Made

### Config alignment

Updated `configs/experiments/water_distribution_white_box_vs_agentic_defender.yaml` to match the treatment-side attacker strength more closely:

- `llm.backend: qwen_langgraph`
- `llm.use_langgraph: true`
- `llm.strict_backend: true`
- `llm.allow_langchain_fallback: true`
- `llm.max_new_tokens: 160`
- `action_cooldown_sec: 0.5`
- `note_cooldown_sec: 2.0`
- `reasoning_cooldown_sec: 1.0`
- `reasoning_refresh_sec: 2.0`
- `memory.attacker_candidate_window: 8`

### Prompt alignment

Rewrote `testbeds/water_distribution/prompts/white_box_attacker.md` into a treatment-style bounded white-box prompt adapted to the distribution plant:

- strong but bounded white-box use of topology, control semantics, and process knowledge,
- explicit prohibition on hidden oracle state, future defender actions, and authored attack recipes,
- emphasis on fastest plausible physical-effect paths across supply, grid, and return-water surfaces,
- allowance for stronger but still bounded sensor-write and actuator-write exploitation when exposed.

### Code-path alignment

Updated `ics_agent/agents/agentic.py` so the distribution white-box attacker now uses distribution-specific white-box logic instead of treatment-shaped logic:

- added distribution-specific white-box system knowledge,
- added distribution-specific surface priors,
- added distribution-specific unconstrained white-box candidate generation,
- added fair actuator-write candidates (`set_actuator`) alongside sensor-write candidates where the distribution testbed already exposes them,
- mapped `set_actuator` into attacker surface tracking so follow-on memory and escalation logic can use it.

### Tests

Added focused coverage in `tests/test_water_distribution.py` for:

- distribution white-box actuator candidate generation,
- distribution white-box system knowledge using distribution topology.

Targeted validation passed before the rerun.

## Scenario Constraints

No scenario catalog edits were made.

No authored scenario-level `attacker_constraints` were removed. The special constrained `wd-attack-supply-pump-max` scenario was preserved as authored. The fairness fix in this pass was implementation-level: the unconstrained distribution white-box path now exposes the bounded distribution action surface more fairly.

## Result Handling

Deleted only the prior result family:

- `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/white_box_attacker__agentic_defender`

Kept intact:

- `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/black_box_attacker__agentic_defender`
- `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/gray_box_attacker__agentic_defender`

## Rerun

Ran only the 12-scenario water-distribution `white_box_attacker__agentic_defender` family on the required deployed backend path:

- `backend: sphere_openplc`
- `project/experiment: openplcwater.lgarcia`
- `realization: example-realization-02`

New results were written to:

- `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/white_box_attacker__agentic_defender`

## Immediate Before/After Comparison

Comparison baseline: the immediately prior water-distribution white-box result family preserved before deletion for metric comparison.

| Metric | Prior distribution white-box | Refreshed distribution white-box |
| --- | ---: | ---: |
| Attack runs | 12 | 12 |
| Attacker safety-success rate | 0.0 | 0.0 |
| Attacker action count avg | 1.0 | 9.4167 |
| Attacker recon action count avg | 0.0 | 0.6667 |
| Attacker execution success rate avg | 0.0 | 0.0 |
| Blocked attacker action count avg | 7.1667 | 27.5 |
| Time to detection from attack start avg (sec) | 0.625 | 1.1667 |
| Time to first defender action from attack start avg (sec) | 0.625 | 1.1667 |
| Defender action count avg | 7.25 | 8.1667 |
| Contained-before-violation run count | 12 | 12 |
| Attacker cutoff activation rate | 1.0 | 1.0 |

Operationally, the refreshed attacker is much stronger than the prior distribution white-box baseline even though it still did not achieve safety success:

- average attacker actions increased from `1.0` to `9.4167`,
- average blocked attacker actions increased from `7.1667` to `27.5`,
- `set_actuator` attack attempts appeared in `9/12` refreshed scenarios,
- the defender still detected and contained all `12/12` runs before any safety violation.

White-box safety-success did not increase relative to the immediately prior distribution white-box baseline. It remained `0/12`.

## Secrets

No secrets or credentials were committed in this pass.
