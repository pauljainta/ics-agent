# Controller-Mediated Bounded Process Verification

An exposed bounded-action function is a proposal interface, not authorization.
The independent broker determines whether the selected industrial endpoint,
operation, and duration are permitted.

For a controller-mediated transaction, distinguish:

1. Request submitted.
2. Authorization accepted.
3. Supervisory command received.
4. Command applied in controller state.
5. Controller output changed.
6. Independent physical response observed.
7. Numerical objective attained over a completed scan.
8. Command expired or was cleared.
9. Normal closed-loop operation and health recovered.

Do not infer physical success from command acknowledgement. Do not bypass a
rejection, use an undiscovered backend, or issue a second state-changing request.
