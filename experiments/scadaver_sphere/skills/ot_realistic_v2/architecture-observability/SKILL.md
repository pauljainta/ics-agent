# Industrial Architecture and Process Observability

Industrial process physics may be implemented by machinery or process-facing
I/O that is not directly attached to an investigator's network. Do not assume
every physical-process component is a discoverable host.

Use controller input and output telemetry, supervisory observations, historian
records, and synchronized samples to identify process-like behavior. Separate:

- A controller runtime identity.
- A measured process variable.
- A controller output state.
- A monitoring copy of controller-observed data.
- A behavioral association.
- Strong causal evidence from an independently evaluated intervention.

Matching values and timing support association, not packet direction or
causality. Report inaccessible implementation details as unknown.
