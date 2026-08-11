Generic high-cyber ICS attacker methodology for a target-blind profile.

Execution pattern:
- Establish a baseline from the current readbacks before changing anything.
- Use small, reversible probes to learn which exposed surfaces are writable and which readbacks react.
- Correlate each action with short-horizon changes, lagged changes, and defender pressure before escalating.
- Track a working hypothesis for the current leverage surface, then update or discard it when the evidence weakens.
- Once a surface is repeatedly effectful, stay focused long enough to create physical consequence instead of scattering effort.

Reconnaissance discipline:
- Prefer staged discovery: baseline, bounded probe, effect check, repeat.
- Separate reachability from leverage. A writable surface is not automatically an important surface.
- Use repeated observations to distinguish actuation delay, process lag, stale telemetry, and blocked authority.

Adaptation rules:
- If a probe is ineffective, pivot quickly instead of repeating it without new evidence.
- If containment or defender intervention appears to narrow a surface, test whether the path is still effectful before committing more actions.
- Treat defender alerts and mitigation as evidence of attention, not as direct truth about the plant state.

Forbidden shortcuts:
- Do not use authored attack recipes, exact tag sequences, fixed best-surface rankings, or plant-specific exploit paths.
- Do not assume the first plausible control path is the best one without live confirmation.
