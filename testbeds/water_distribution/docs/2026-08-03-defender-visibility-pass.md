Water-Distribution Defender Visibility Realism Pass

Date: 2026-08-03

Bug fixed

The water-distribution defender was receiving passive attacker reconnaissance as directly actionable evidence. In the pre-fix water-distribution white-box runs, the defender could act after only `inspect_surface`, which violated the intended observability model for passive recon.

Code paths changed

- `ics_agent/agents/agentic.py`
  - added a water-distribution-specific defender history sanitization path before online context assembly
  - removed attacker `inspect_surface` actions from defender-visible recent action history
  - removed attacker decision records from defender-visible recent decision history
  - filtered defender/observer discovery-only history items that were derived from passive recon before defender context, incident, MITRE, memory, and candidate-action construction
- `tests/test_water_distribution.py`
  - added focused regression coverage for hidden passive recon and still-visible modifying attacker actions

Fix scope

The behavioral fix is water-distribution-specific. Treatment behavior was not intentionally changed in this pass.

Defender-visible evidence still allowed after the fix

- unsafe or suspicious telemetry
- alarms and explicit alerts
- command/state divergence
- plant state transitions
- modifying attacker command effects
- blocked runtime actions when explicitly modeled by the runtime
- operational control events that are already modeled as defender-visible

Passive reconnaissance visibility

Passive recon via `inspect_surface` is no longer directly visible to the water-distribution defender. Attacker offline reasoning also remains invisible.

Result cleanup and rerun

- deleted old result root: `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc`
- reran all 33 retained-suite water-distribution attack runs:
  - `black_box_attacker__agentic_defender`
  - `gray_box_attacker__agentic_defender`
  - `white_box_attacker__agentic_defender`
- rerun completed on the deployed backend path:
  - backend: `sphere_openplc`
  - merge server: `merge.example.invalid`
  - project: `openplcwater.lgarcia`
  - realization: `example-realization-02`

Result location

New results were written to:

- `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc`

Post-rerun validation

- all 33 runs completed successfully
- all 33 `config_snapshot.yaml` artifacts point to the required deployed backend configuration
- direct scan result: zero runs where the defender first acted after only attacker `inspect_surface`
- the prior water-distribution failure pattern was eliminated in the rerun artifacts

Secrets

No secrets or credentials were committed in this pass.
