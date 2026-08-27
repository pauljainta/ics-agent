# vendor/openplc/scripts/payloads

Tracked treatment runtime payloads.

## Files

- `modbus_raw.cpython-310.pyc_payload`: Modbus helper payload.
- `native_plc_runtime.cpython-310.pyc_payload`: native controller/simulator runtime payload.
- `openplc_testbed.cpython-310.pyc_payload`: treatment testbed support payload.

These are not normal `__pycache__` files. They are explicit runtime payloads kept outside `__pycache__` so routine Python compilation cannot overwrite them. The long-term ideal is to replace any sourceless payload with audited source-backed implementations.

Do not add generated cache files here.
