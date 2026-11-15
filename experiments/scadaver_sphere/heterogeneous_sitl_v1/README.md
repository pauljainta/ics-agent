# Heterogeneous SPHERE SITL v1

This directory is an additive deployment bundle for the existing
`scadaverfingerprint.openplcwater.lgarcia` realization.  It does not create,
re-materialize, or modify a SPHERE topology.

## Controller choice

OpenPLC Runtime v4 was evaluated first at source revision
`9fb5e1a29b574806f5aea3a4e78202c7a5d658ee`.  It has native Modbus, OPC UA,
and S7 plugins, but its supported program lifecycle is an authenticated,
OpenPLC-Editor-generated `program.zip` upload.  The existing water simulator
requires a reproducible IEC program with a native Modbus master mapping; the
v4 editor package cannot be deterministically built by this repository.

This bundle therefore uses OpenPLC v3 only for this deployment, pinned to
`b5d41356dab4aeadca0dd7ca64ba542f870b595d`.  The selection is based on a
reproducible, locally compiled IEC ST program and native Modbus-master
configuration, not on tutorial convenience.  The v3 limitation is recorded:
it is end-of-life and must not be treated as an endorsement for production.

## Deployment invariant

`scadaver-legacy-controller.service` and
`scadaver-openplc-controller.service` conflict.  The activation script stops
the legacy controller only after OpenPLC installation and IEC compilation
succeed.  At most one controller can own the simulator command channel.

OpenPLC v3 activation must use its local web UI: upload the IEC source, add it
as a Program, compile it, configure the Modbus TCP slave, and start the PLC.
This is required because the UI creates the database-backed active-program
record and rebuilds the generated I/O binding code.  The deployment script
only installs and starts the web runtime; it intentionally does not bypass
that lifecycle.

The operator must run deterministic validation after deployment.  No LLM
experiment is started by this bundle.
