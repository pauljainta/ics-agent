This directory vendors the minimal OpenPLC support payload that `ics-agent`
needs for local and SPHERE-backed execution.

The runtime currently depends on a small set of compiled helper modules plus
thin source wrappers:

- `scripts/__pycache__/openplc_testbed.cpython-310.pyc`
- `scripts/__pycache__/modbus_raw.cpython-310.pyc`
- `scripts/__pycache__/native_plc_runtime.cpython-310.pyc`
- `scripts/__pycache__/modbus_bridge.cpython-310.pyc`
- `scripts/native_plc_runtime.py`
- `scripts/modbus_bridge.py`

They are vendored here so `ics-agent` no longer requires a sibling
`sphere-usecases` checkout just to run the backend, stage remote services, or
launch the local Modbus bridge.
