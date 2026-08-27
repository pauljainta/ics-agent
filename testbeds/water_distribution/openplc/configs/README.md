# testbeds/water_distribution/openplc/configs

OpenPLC mapping configuration for the distribution testbed.

## Files

- [`openplc_map.yaml`](openplc_map.yaml): maps process variables, Modbus locations, and bridge fields used by the distribution OpenPLC runtime.

The bridge in [`../scripts/modbus_bridge.py`](../scripts/modbus_bridge.py) and validation harness in [`../scripts/validation_harness.py`](../scripts/validation_harness.py) depend on this mapping.
