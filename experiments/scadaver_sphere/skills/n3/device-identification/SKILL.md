# Industrial Device and Runtime Identification

## Purpose

Explain how to investigate industrial protocol identity, software runtime evidence, and functional-role hypotheses.

## When to Use

Consult when industrial services have been discovered but the device identity, software runtime, or operational role remains uncertain.

## Tools

### `probe_device_identity`

Arguments:

- `ip`
- `port`

Requests self-reported Modbus identity metadata when supported.

Reported manufacturer or product information is not proof of hardware authenticity.

Unsupported identification is a valid result.

### `inspect_http_service`

Arguments:

- `ip`
- `port`

Performs bounded read-only inspection of an approved HTTP service root.

A successful HTTP response establishes an accessible HTTP interface, not controller functionality.

This tool does not follow redirects.

### `inspect_http_resource`

Arguments:

- `ip`
- `port`
- `path`
- `method`: `GET` or `HEAD`
- `max_redirects`: 0 through 3

Inspects an agent-selected read-only HTTP resource.

Only same-host redirects are allowed.

A redirect may explain why a root-only inspection returned limited identity information.

Do not attempt credentials, forms, state-changing requests, or unauthorized endpoint access.

### `browse_opcua_nodes`

Arguments:

- `ip`
- `port`

Browses the accessible OPC UA namespace.

Use returned node IDs and metadata as observations.

Do not invent node IDs or treat node labels as proof of control authority.

### `read_opcua_values`

Arguments:

- `ip`
- `port`
- `node_ids`: one to ten selected node identifiers

Reads values from already identified accessible OPC UA nodes.

A value may describe a sensor, a status variable, configuration metadata, or unrelated information.

### `inspect_s7_metadata`

Arguments:

- `ip`
- `port`

Retrieves available S7-compatible self-reported CPU, order-code, or communication metadata.

An S7-compatible response does not independently prove Siemens hardware.

### `read_s7_data`

Arguments:

- `ip`
- `port`
- `db`
- `start`
- `size`: one to 32 bytes

Reads a bounded data-block range through the approved interface.

A failed read may indicate that a requested block or operation is unsupported.

Do not assume the target is nonindustrial based on one failed data-block request.

### `explore_modbus_map`

Arguments:

- `ip`
- `port`
- `start`
- `count`

Investigates a bounded address window across supported Modbus data models.

Results are raw observations without verified semantic labels.

## Identification Principles

A device can support multiple protocols.

A monitoring system can expose industrial metadata.

A protocol gateway may appear to represent downstream industrial systems.

An HMI may expose measurements without implementing primary control logic.

A runtime identity claim should be based on appropriate direct evidence.

Functional-role claims should also consider observed behavior.

## PLC and Controller Fingerprinting Workflow

Use independent evidence layers rather than selecting the host with the most
industrial-looking labels:

1. **Runtime identity:** inspect HTTP and protocol identity evidence for a
   controller runtime. A branded OpenPLC management interface, when directly
   observed, supports OpenPLC software identity without revealing where it is.
2. **Runtime activity:** seek active-program, running-core, scan, or controller
   state. A management page with no active runtime is weaker evidence.
3. **Data-plane behavior:** inspect bounded Modbus, OPC UA, or S7 data and
   distinguish likely inputs, internal state, outputs, and read-only copies.
4. **Temporal behavior:** sample relevant values repeatedly. Look for inputs
   changing over time and outputs changing in a manner consistent with control
   decisions.
5. **Cross-interface corroboration:** determine whether runtime identity and
   cyclic I/O evidence belong to the same host. Do not transfer identity from
   one host to another merely because their values match.
6. **Alternative-role test:** compare the controller hypothesis against HMI,
   historian, engineering-workstation, gateway, remote-I/O, and unknown roles.

### Evidence weights

Strong controller evidence:

- Direct controller-runtime identity plus active runtime/program state.
- Runtime identity corroborated by process inputs and controller outputs on the
  same host.
- Repeated input/output behavior consistent with an executing control loop.

Supporting but insufficient evidence:

- An industrial protocol port.
- CPU-like or vendor-like self-reported metadata.
- Labeled process variables.
- Values that change or match values on another host.
- An engineering or monitoring web interface.

Evidence against a primary-controller classification includes a clearly
read-only monitoring interface, historical-only data, absence of runtime state,
or values that merely lag or mirror another source. Such evidence is not
absolute proof; report uncertainty when direct runtime evidence is unavailable.

## Example

One host reports S7-compatible CPU metadata and labeled process values. Another
host directly identifies a controller runtime, reports an active program, and
exposes process inputs and outputs.

The first host remains a controller candidate, but its labels and metadata do
not prove execution. The second has stronger controller evidence because direct
runtime identity, active state, and data-plane behavior corroborate one another.

## Handling Contradictory Evidence

Suppose one service reports a controller identity but another observation suggests an engineering interface.

Possible explanations include:

- A combined controller and management host.
- A gateway.
- A monitoring or engineering system republishing controller data.
- Misinterpreted metadata.
- Multiple services with unrelated responsibilities.

Do not arbitrarily select one explanation.

Record the uncertainty and identify what evidence would discriminate between the alternatives.

## Common Errors

Unsupported identity request: no confirmed identity metadata was returned.

HTTP redirect: inspect only through an approved read-only path and permitted redirect policy.

OPC UA node unavailable: the requested value could not be read.

S7 data-block error: the requested address or operation was not available through the interface.

Malformed or incomplete metadata: retain only directly supported fields.

## Completion

Return evidence-supported software identity and functional-role hypotheses.

Keep authenticity and causal responsibility uncertain where the observations do not establish them.
