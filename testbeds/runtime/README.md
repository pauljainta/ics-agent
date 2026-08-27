# testbeds/runtime

Shared runtime registry and testbed runtime helpers.

## Files

- [`registry.py`](registry.py): maps normalized testbed identifiers to scenario roots, support roots, and runtime-specific metadata.
- [`water_distribution_runtime.py`](water_distribution_runtime.py): Python-backed water-distribution runtime helpers.
- [`__init__.py`](__init__.py): package exports.

The environment backend in [`../../ics_agent/environment.py`](../../ics_agent/environment.py) uses this package to select the correct runtime behavior for each testbed.
