# OT Fundamentals

## Purpose

Provide domain knowledge for reasoning about unfamiliar industrial control systems.

Use this skill to understand common industrial components, communication technologies, and functional relationships.

## Industrial Architecture

Industrial control systems connect computation and communication with physical processes.

A simplified industrial architecture may contain:

1. A physical process.
2. Sensors and actuators.
3. Field I/O interfaces.
4. PLCs or other controllers.
5. Supervisory interfaces.
6. Engineering and monitoring systems.

These roles may be implemented on separate machines or combined within fewer devices.

Do not assume a one-to-one relationship between an IP address and a physical device.

## PLCs and Controllers

A PLC executes a control program.

A typical control cycle involves reading inputs, evaluating control logic, and updating outputs.

Control programs may use timers, counters, internal variables, thresholds, and state machines.

A controller can communicate with remote I/O and process-facing services through industrial protocols.

A controller may expose a management interface in addition to its industrial communication interfaces.

Finding an industrial protocol port is insufficient to prove that controller logic executes on the corresponding host.

## Sensors and Process Variables

Sensors represent physical observations.

Common variables include:

- Liquid level.
- Flow.
- Pressure.
- Temperature.
- Position.
- Speed.
- Binary equipment status.

An application may expose measurements as numeric registers, binary states, or named OPC UA nodes.

Measurement units and meanings must be established through evidence.

A numeric value by itself does not establish a sensor's function.

## Actuators

Actuators influence physical processes.

Examples include pumps, valves, motors, and heaters.

A binary industrial value may represent an actuator command, an actuator status, a mode flag, or unrelated data.

Do not infer that any binary value is an actuator without supporting observations.

## HMI and SCADA Systems

HMIs and SCADA systems frequently display industrial measurements.

They may also expose supervisory controls.

However, a service presenting process information is not necessarily the system executing primary control logic.

## Engineering Workstations

Engineering systems may provide controller configuration, maintenance, diagnostic, and programming interfaces.

The presence of an engineering interface does not prove that the host is the runtime controller.

## Industrial Gateways

Gateways may translate between industrial protocols or relay data between systems.

A gateway can appear to support several protocols without executing the process-control program.

## Identifying the Active Controller

A PLC or controller classification should combine direct runtime identity with
functional evidence. Strong evidence can include an identifiable controller
management interface, active-program or scan state, and industrial I/O that is
consistent with cyclic control behavior.

Common software PLC runtimes, including OpenPLC, may expose a branded web
management interface in addition to Modbus. A directly observed runtime name is
strong software-identity evidence, but controller responsibility should still
be corroborated with active runtime state or control-loop observations.

Monitoring, engineering, HMI, and historian systems may republish the same
measurements with clearer labels than the controller. Clear labels improve
semantic interpretation but do not prove that the host executes control logic.

When comparing controller candidates, ask:

1. Is there direct runtime or active-program evidence?
2. Does the host expose both process-facing inputs and control outputs?
3. Do repeated observations support a cyclic control relationship?
4. Could the values instead be a read-only monitoring copy?
5. What independent observation would reject the leading alternative?

## Modbus Basics

Modbus exposes four common logical data models:

- Coils: binary values.
- Discrete inputs: binary observations.
- Holding registers: word-oriented data.
- Input registers: word-oriented observations.

Conventional Modbus data-model distinctions do not guarantee the application-level semantics of an address.

Register indexing conventions can differ between user documentation and library APIs.

Do not assume that a value belongs to a particular sensor or actuator based only on its numerical address.

## OPC UA Basics

OPC UA presents information through an address space containing nodes, references, and values.

Node names, namespaces, and attributes may provide clues about system function.

They are not independent proof of physical behavior.

## S7-Compatible Communications

S7-compatible services may provide CPU-related metadata or access to supported data blocks.

A compatible response does not automatically establish authentic Siemens hardware.

## Identity Versus Function

Separate three questions:

1. What protocol does this service speak?
2. What software or hardware identity does it report?
3. What function does this system actually appear to perform?

These questions require different evidence.

## Example

An unfamiliar system has three hosts.

Host A exposes changing Modbus values. Host B exposes a controller-runtime
management page, active program state, and Modbus I/O. Host C exposes labeled
OPC UA measurements and historical records.

Host B has the strongest controller evidence because its direct runtime
identity is corroborated by active control state and an industrial data plane.
Host A remains an unidentified process-facing endpoint until its function is
corroborated. Host C is more consistent with monitoring unless additional
evidence shows that it executes control logic.

## Completion

Use this knowledge to classify industrial components conservatively.

Do not invent device roles or assume the existence of exactly one PLC.
