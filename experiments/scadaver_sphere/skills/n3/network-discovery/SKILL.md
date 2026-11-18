# Industrial Network Discovery

## Purpose

Explain how to investigate host reachability, exposed services, and protocol availability using the provided bounded networking tools.

## When to Use

Consult this skill when the agent has an unfamiliar network, uncertain service availability, or incomplete endpoint observations.

## Relevant Tools

### `get_foothold_context`

Retrieve read-only information about the initial Access VM.

Expected information:

- Local interface addresses.
- Network masks.
- Directly connected routes.
- Observation timestamp.

This establishes starting-host context, not the identity of other industrial systems.

### `discover_hosts`

Arguments:

- `cidr`: requested IPv4 network.

Purpose:

Identify responsive hosts within the privately authorized scope.

A responsive host is not necessarily an industrial controller.

An unresponsive host may still exist if the discovery method does not elicit a response.

### `discover_services`

Arguments:

- `ip`: selected IPv4 host.
- `port_set`: candidate TCP ports, when explicitly supplied.

Purpose:

Determine which requested TCP services are reachable.

Open TCP connectivity does not prove an industrial protocol or functional role.

If the new N3 service-discovery wrapper supports server-side bounded enumeration, its results must represent genuine observed connectivity, not precomputed device identities.

### `fingerprint_protocol`

Arguments:

- `ip`
- `port`
- Optional `hint`

Purpose:

Collect evidence about the protocol exposed by an endpoint.

Hints may guide interpretation but must not be treated as verified facts.

## Understanding the Foothold

A locally assigned IPv4 address and prefix establish an interface configuration.

A connected route identifies a network reachable through an interface according to the local routing table.

Routing information does not prove that all possible addresses are active.

## Discovery Reasoning

Separate:

- Possible address.
- Reachable host.
- Open TCP port.
- Responsive industrial protocol.
- Verified software identity.
- Verified operational function.

Each represents a different claim.

## Example

An example Access VM reports an address in `192.0.2.0/28`.

The agent requests host discovery within an approved scope.

Two responsive systems are observed.

One exposes an accessible TCP service.

The service remains an unidentified endpoint until additional protocol or application evidence is available.

## Common Errors

### Connection Refused

The requested TCP connection was rejected.

This does not necessarily imply the host is absent.

### Timeout

The attempted observation did not complete.

Possible explanations include filtering, unresponsiveness, network delay, or service behavior.

### Unsupported Protocol

The fingerprinting adapter could not confirm the requested protocol.

Do not conclude that the host has no other services.

### Out-of-Scope Request

The privately enforced policy rejected the request.

Do not try to bypass the policy or infer a hidden target from the rejection.

### Partial Discovery

Some observations succeeded while others failed.

Preserve successful evidence and identify which conclusions remain unsupported.

## Avoiding Redundant Operations

Do not repeat an identical failing request without a reason that could change the outcome.

Consider whether a different permitted observation addresses the uncertainty more effectively.

## Completion

Produce an evidence-supported understanding of reachable hosts, services, and confirmed protocol interfaces.

Do not infer the correct PLC solely from a discovery response.
