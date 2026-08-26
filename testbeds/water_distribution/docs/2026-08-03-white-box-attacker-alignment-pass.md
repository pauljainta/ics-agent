Water-Distribution White-Box Attacker Alignment Pass

Date: 2026-08-03

Treatment white-box strength knobs standardized into distribution

- Reused the shared white-box breach-profile generator so distribution now works from the same attacker-side notion of low-floor, high-breach, and high-extreme sensor values that treatment uses.
- Reworked distribution white-box surface priors so sensor-manipulation and actuator-write-path leverage are ranked with stronger confidence, explicit preferred target/value hints, alternate directions, breach-capable flags, and faster-path bias.
- Reworked distribution white-box candidate generation so the attacker can escalate from reconnaissance into direct breach-capable sensor commits, sustain those values once selected, and pair sensor and actuator pressure with treatment-style aggressiveness.
- Extended distribution white-box system knowledge and prompt/manual wiring so the attacker is explicitly told that above-bound sensor values are deliberate white-box leverage when the runtime allows them.

Code, prompt, and runtime changes

- `ics_agent/agents/agentic.py`
  - aligned distribution white-box surface priors to treatment-style breach reasoning and stronger direct-commit bias
  - aligned distribution white-box candidate generation to prefer breach-capable sensor paths, stronger escalation, and sustain behavior after commitment
  - exposed breach-capable `set_sensor` semantics and leverage hints in white-box system knowledge
- `testbeds/runtime/water_distribution_runtime.py`
  - changed white-box sensor override normalization to clamp only to the 16-bit runtime register space instead of the nominal safety max
- `testbeds/water_distribution/prompts/white_box_attacker.md`
  - told the white-box attacker to prefer above-bound sensor values when they shorten the path to impact
- `testbeds/water_distribution/knowledge/attacker_manual.md`
  - added breach-capable white-box operating guidance for high-side and low-side spoofing
- `tests/test_water_distribution.py`
  - added focused coverage for breach-capable candidate generation, updated white-box system-knowledge expectations, and verified that above-bound sensor overrides now survive runtime normalization

Breach-capable values and normalization

- Breach-capable white-box sensor values were enabled for distribution by exposing `high_breach` and `high_extreme` candidates above published maxima and by allowing low-side paths to drive signals to the lowest reachable value.
- Runtime normalization changed only for distribution sensor overrides: the runtime now preserves white-box breach intent up to the Modbus register limit instead of collapsing values back into the nominal process envelope.
- Numeric actuator bounds remain unchanged and still clamp to their authored actuator limits.

Validation and rerun scope

- Targeted validation passed before the rerun:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_water_distribution.py`
- Deleted only the prior white-box result family:
  - `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/white_box_attacker__agentic_defender`
- Preserved:
  - `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/black_box_attacker__agentic_defender`
  - `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/gray_box_attacker__agentic_defender`
- Reran only the retained 11-scenario white-box family on the deployed backend path:
  - backend: `sphere_openplc`
  - project: `openplcwater.lgarcia`
  - realization: `example-realization-02`

Results

- New white-box water-distribution results were written to:
  - `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/white_box_attacker__agentic_defender`
- Immediate pre-pass white-box distribution baseline observed before cleanup:
  - attacker safety-success scenario count: `0 / 11`
  - attacker safety-success rate: `0.0`
- Final rerun status: `completed`
- Final rerun completion detail: `11 / 11` scenarios completed, `0` failures, and all `11` runs reached the MITRE `impact` phase.
- Final attacker safety-success scenario count: `11 / 11`
- Final attacker safety-success rate: `1.0`
- Final average time-to-impact metric recorded in the aggregate summary: `0.0455`
- Change versus the immediately prior white-box distribution baseline: `increased from 0 / 11 (0.0) to 11 / 11 (1.0)`

Secrets

No secrets or credentials were committed in this pass.
