# Vendor Support

This folder is required for the SPHERE/OpenPLC execution path. It is not an
optional archive.

## Files

- `openplc/scripts/native_plc_runtime.py`: Entry-point wrapper for the native
  PLC runtime. It loads the Modbus support module and starts the selected
  controller or simulator role.
- `openplc/scripts/modbus_bridge.py`: Entry-point wrapper for the controller /
  simulator Modbus bridge. It loads the bridge implementation and starts it.

## Runtime relationship

`ics_agent/paths.py` defines this folder as `OPENPLC_SUPPORT_DIR` and
`OPENPLC_SUPPORT_SCRIPTS`. `ics_agent/environment.py` resolves these scripts,
imports the native runtime, and stages the support payload to remote nodes for
SPHERE/OpenPLC runs. Removing this folder would break that execution and
deployment path.

The runtime may also use generated Python bytecode under
`openplc/scripts/__pycache__/` during local development or payload staging.
Those files are ignored by Git and are generated artifacts, not source files.
They must not be treated as a replacement for the tracked Python wrappers.

This folder contains only the vendored OpenPLC support needed by this project;
experiment-specific PLC programs and mappings live under `testbeds/`.
