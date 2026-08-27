# Vendor Support

This folder is required for the SPHERE/OpenPLC execution path. It is not an
optional archive.

## Files

- `openplc/scripts/native_plc_runtime.py`: Entry-point wrapper for the native
  PLC runtime. It loads the Modbus support module and starts the selected
  controller or simulator role.
- `openplc/scripts/modbus_bridge.py`: Source implementation for the treatment
  controller / simulator Modbus bridge.
- `openplc/scripts/payloads/`: Tracked treatment runtime payloads used by the
  native OpenPLC backend. These files are intentionally outside `__pycache__`
  so routine Python compilation cannot overwrite them.

## Runtime relationship

`ics_agent/paths.py` defines this folder as `OPENPLC_SUPPORT_DIR` and
`OPENPLC_SUPPORT_SCRIPTS`. `ics_agent/environment.py` resolves these scripts,
imports the native runtime, and stages the support payload to remote nodes for
SPHERE/OpenPLC runs. Removing this folder would break that execution and
deployment path.

Generated Python bytecode under `openplc/scripts/__pycache__/` is ignored by Git
and is not a source of truth. Payload staging must use the tracked files under
`openplc/scripts/payloads/` or tracked Python source files.

This folder contains only the vendored OpenPLC support needed by this project;
experiment-specific PLC programs and mappings live under `testbeds/`.
