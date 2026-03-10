# DeploymentNode Agent — Spec

## Name
DeploymentNodeAgent

## Purpose / role
Coordinate execution of approved tasks against simulator/HIL endpoints with strict safety gating.

## Inputs
- Approved scenario/task package
- Environment config (`sim`, `hil`, partner)
- Endpoint connectivity/health checks
- Explicit operator consent state

## Outputs
- Execution plan with safety checks
- Structured action logs
- Post-run status and artifacts

## Tools required
- Endpoint adapter (SPHERE/simulator)
- Command validation policy engine
- Telemetry collector
- Audit logger

## Actions
- `preflight_checks()`
- `plan_execution()`
- `request_actuation_consent()`
- `execute_gated_action()`
- `collect_and_report()`

## Constraints & safety
- Default `READ_ONLY=true`.
- Any actuation requires explicit consent gate and human approval.
- Must stop on failed preflight or policy validation.

## Autonomy level
- Start at L0-L1 depending on environment; escalate only with formal sign-off.

## Evaluation metrics
- Preflight pass/fail quality
- Policy violation catch rate
- Mean time to rollback on fault

## Example API
- `POST /deploy/plan` -> returns execution plan id
- `POST /deploy/execute/{id}` -> executes only if consent is valid
