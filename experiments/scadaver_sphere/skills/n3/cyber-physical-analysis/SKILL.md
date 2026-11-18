# Cyber-Physical Process Analysis

## Purpose

Teach the agent to interpret process observations, temporal behavior, and potential controller–process relationships.

## When to Use

Consult when the agent needs to understand physical variables, changing measurements, or the difference between controller execution and monitoring.

## Relevant Tools

### `read_modbus_holding`

Arguments:

- `ip`
- `port`
- `start`
- `count`

Reads up to ten holding registers.

A holding-register value is not automatically a command or a physical measurement.

### `read_modbus_input_registers`

Arguments:

- `ip`
- `port`
- `start`
- `count`

Reads up to ten input registers.

The returned values must be interpreted using independent context.

### `read_modbus_coils`

Arguments:

- `ip`
- `port`
- `start`
- `count`

Reads up to ten binary coil values.

A coil can represent a command or another application-level Boolean state.

### `read_modbus_discrete_inputs`

Arguments:

- `ip`
- `port`
- `start`
- `count`

Reads up to ten discrete input values.

Binary status does not automatically imply an actuator.

### `sample_modbus_series`

Arguments:

- `ip`
- `port`
- `register_type`: `holding`, `input`, `coil`, or `discrete`
- `start`
- `count`
- `samples`: one through five
- `interval_seconds`: at least one second

Collects repeated timestamped observations from one endpoint.

Use when a single measurement cannot establish whether the value changes over time.

Interpret the actual timestamps, not merely the requested interval.

### `sample_cross_device_series`

Arguments:

- `targets`: two to four approved Modbus targets.
- `samples`: one through five.
- `interval_seconds`: at least one second.

Each target requires:

- `ip`
- `port`
- `register_type`
- `start`
- `count`

Collects repeated observations across several endpoints.

Measurements within one sweep are sequential, not necessarily simultaneous.

Inspect acquisition timestamps and skew before comparing measurements.

### `get_previous_observations`

Optional argument:

- `ip`

Retrieves observations already collected during the current episode.

Use to review previous evidence rather than blindly repeating network operations.

## Understanding Physical Behavior

Physical processes evolve over time.

A variable may increase, decrease, oscillate, remain constant, or change in response to system operation.

Temporal variation can support a process-dynamics hypothesis.

However, variation alone does not establish the meaning of the variable.

## Measurements Versus Commands

A measurement reports an observed condition.

A command requests or represents a desired control state.

A status may describe the observed state of equipment.

These can have similar representations in an industrial protocol.

Do not classify a variable as a command solely because it is binary.

## Correlation and Causality

Two changing measurements may be correlated because they reflect the same process.

Their correlation does not prove that one endpoint controls the other.

Distinguish:

- Similar values.
- Temporally related values.
- A plausible functional association.
- Independently demonstrated causal influence.

## Timestamp Interpretation

Check:

- Sampling start time.
- Individual measurement timestamps.
- Time between samples.
- Missing measurements.
- Sequential read delay.
- Process scan frequency, when independently known.

Avoid claiming temporal order or causality when sampling resolution is insufficient.

## Static Versus Dynamic Systems

A static response may reflect:

- A constant physical condition.
- An unchanged configuration variable.
- An inactive process.
- An insufficient sampling interval.

Do not infer a functional role from static or dynamic behavior alone.

## Example

Host A exposes a numeric value that changes between repeated observations.

Host B exposes a similar value.

Host C exposes a constant industrial register.

The changing observations may support a shared-process hypothesis.

They do not yet prove that A controls B or establish C's functional role.

Further observations may strengthen or weaken the hypotheses.

## Tool Errors

Invalid Modbus address: the requested data location is unavailable through that operation.

Unsupported function: the endpoint did not support that operation.

Timeout: the observation did not complete.

Incomplete series: only part of the requested temporal evidence was collected.

Clock or sampling discrepancy: comparisons may be unreliable.

## Completion

Provide an evidence-supported characterization of observed process behavior, likely functional roles, and possible system relationships.

State limitations explicitly.
