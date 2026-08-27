"""Local-only configuration for tests that model remote deployments."""

from __future__ import annotations

import os
from pathlib import Path


def _load_test_env() -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = Path(__file__).with_name("test.env")
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key.startswith("ICS_TEST_")})
    return values


_VALUES = _load_test_env()


def _value(name: str, default: str) -> str:
    return _VALUES.get(name, default)


TEST_MERGE_SERVER = _value("ICS_TEST_MERGE_SERVER", "replace-with-local-merge-server")
TEST_TREATMENT_EXPERIMENT = _value("ICS_TEST_TREATMENT_EXPERIMENT", "replace-with-local-treatment-experiment")
TEST_TREATMENT_REALIZATION = _value("ICS_TEST_TREATMENT_REALIZATION", "replace-with-local-treatment-realization")
TEST_DISTRIBUTION_REALIZATION = _value("ICS_TEST_DISTRIBUTION_REALIZATION", "replace-with-local-distribution-realization")
TEST_REVISION = _value("ICS_TEST_REVISION", "replace-with-local-revision")
TEST_MERGE_USER = _value("ICS_TEST_MERGE_USER", "replace-with-local-user")
TEST_XDC_HOST = _value("ICS_TEST_XDC_HOST", "replace-with-local-xdc-host")
TEST_XDC_REF = _value("ICS_TEST_XDC_REF", "replace-with-local-xdc-ref")
TEST_ACCESS_IP = _value("ICS_TEST_ACCESS_IP", "test-access-node")
TEST_ENGINEERING_IP = _value("ICS_TEST_ENGINEERING_IP", "test-engineering-node")
TEST_CONTROLLER_IP = _value("ICS_TEST_CONTROLLER_IP", "test-controller-node")
TEST_SIMULATOR_IP = _value("ICS_TEST_SIMULATOR_IP", "test-simulator-node")