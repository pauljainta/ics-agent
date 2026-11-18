"""V2 agent worker: the private process network is outside its authority."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

from ..n3_worker import N3ReadOnlyWorker
from ..policy import ReadOnlyPolicy
from ..worker import _run
from .constants import AGENT_PORTS, OT_CIDR


class V2ReadOnlyWorker(N3ReadOnlyWorker):
    """Use the sandbox-safe native adapter for Modbus data retrieval."""

    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        models = {
            "read_modbus_holding": "holding",
            "read_modbus_coils": "coil",
        }
        if name in models:
            return self._native_modbus_read(models[name], arguments)
        return super().execute(name, arguments)

    def _read_data_model(
        self, data_model: str, args: dict[str, object], *, record: bool = True,
        reserve: bool = True,
    ) -> dict[str, object]:
        return self._native_modbus_read(data_model, args, record=record, reserve=reserve)


def _v2_runner(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Perform genuine probes without SCADAver's persistent inventory cache."""
    command = ["--no-cache" if value == "--live" else value for value in argv]
    return _run(command, timeout)


def build_worker(**kwargs: object) -> V2ReadOnlyWorker:
    kwargs.setdefault(
        "scadaver",
        os.environ.get("SCADAVER_BIN", "/home/researcher/.cargo/bin/scadaver"),
    )
    kwargs.setdefault("runner", _v2_runner)
    return V2ReadOnlyWorker(
        ReadOnlyPolicy.build(str(OT_CIDR), AGENT_PORTS, max_hosts=16),
        **kwargs,
    )
