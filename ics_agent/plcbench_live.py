"""Bind a live OpenPLC pair to the PLCBench native runner.

The native actor writes native Modbus to the **controller** (through the broker);
the canonical modeled process state is read from the **simulator** endpoint by a
recorder that is outside the actor's writable path. This module provides that
recorder (a stdlib Modbus reader, so no pymodbus dependency on the runner side),
derives the sensor read list and the process-linked object set from the use
case's openplc_map.yaml, and wires both into `run_native_episode`.

Bringing the OpenPLC containers up is a deployment step (see
docs/baselines/chpc/ for the model side and the use-case openplc/ dir for the
pair); this module assumes the pair is reachable and does admission + reset
around the episode.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .plcbench_broker import _MBAP_LEN
from .plcbench_run import run_native_episode
from .plcbench_scorer import Objective

# register_type -> Modbus read function code
_READ_FC = {"coil": 1, "discrete_input": 2, "holding_register": 3, "input_register": 4}
# writable register_type -> destination space label used in the object set
_WRITE_SPACE = {"coil": "coil", "holding_register": "hr"}


class ModbusReadError(RuntimeError):
    pass


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ModbusReadError("connection closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def _read_response(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, _MBAP_LEN)
    length = struct.unpack(">H", header[4:6])[0]
    return _recv_exact(sock, length - 1)  # the PDU (unit already consumed)


def real_from_words(hi: int, lo: int) -> float:
    """Two 16-bit words (high word first) -> IEEE-754 REAL."""
    return struct.unpack(">f", struct.pack(">HH", hi & 0xFFFF, lo & 0xFFFF))[0]


@dataclass(frozen=True)
class SensorSpec:
    tag: str
    fc: int
    address: int
    words: int
    kind: str  # "real" | "int" | "bool"


def sensor_specs_from_map(mappings: Mapping[str, Any],
                          tags: Sequence[str]) -> list[SensorSpec]:
    """Read specs for the canonical sensor tags (what the recorder samples)."""
    specs: list[SensorSpec] = []
    for tag in tags:
        e = mappings[tag]
        rt = e["register_type"]
        if rt not in _READ_FC:
            raise ValueError(f"{tag}: unreadable register_type {rt!r}")
        words = int(e.get("words", 1))
        kind = ("real" if words == 2 else
                ("bool" if rt in ("coil", "discrete_input") else "int"))
        specs.append(SensorSpec(tag, _READ_FC[rt], int(e["address"]), words, kind))
    return specs


def object_set_from_map(mappings: Mapping[str, Any],
                        tags: Sequence[str]) -> list[str]:
    """Process-linked object set as broker destinations (e.g. 'hr:100', 'coil:40').

    Tags must be writable (coil or holding_register); a 2-word REAL spans a range.
    """
    out: list[str] = []
    for tag in tags:
        e = mappings[tag]
        rt = e["register_type"]
        if rt not in _WRITE_SPACE:
            raise ValueError(f"{tag}: non-writable register_type {rt!r} in object set")
        space, addr, words = _WRITE_SPACE[rt], int(e["address"]), int(e.get("words", 1))
        out.append(f"{space}:{addr}" if words <= 1 else f"{space}:{addr}-{addr + words - 1}")
    return out


class ModbusObserver:
    """Canonical modeled-state recorder: reads sensor tags from the simulator.

    Callable, returning ``(t_seconds, {tag: value})`` for the scorer's trace. It
    opens a short connection per sample so it never shares the actor's socket, and
    is independent of the broker/controller path the actor uses.
    """

    def __init__(self, host: str, port: int, specs: Sequence[SensorSpec], *,
                 unit: int = 1, timeout: float = 3.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.host, self.port = host, port
        self.specs = list(specs)
        self.unit = unit
        self.timeout = timeout
        self._clock = clock
        self._t0: float | None = None
        self._txid = 0

    def _read(self, sock: socket.socket, fc: int, address: int, count: int) -> list[int]:
        self._txid = (self._txid + 1) & 0xFFFF
        pdu = bytes([fc]) + struct.pack(">HH", address, count)
        frame = struct.pack(">HHH", self._txid, 0, len(pdu) + 1) + bytes([self.unit]) + pdu
        sock.sendall(frame)
        resp = _read_response(sock)
        if not resp or resp[0] & 0x80:
            raise ModbusReadError(f"fc{fc}@{address}: exception/empty response")
        if fc in (1, 2):  # packed bits
            nbytes = resp[1]
            bits: list[int] = []
            for b in resp[2:2 + nbytes]:
                bits.extend((b >> i) & 1 for i in range(8))
            return bits[:count]
        return list(struct.unpack(">" + "H" * count, resp[2:2 + 2 * count]))

    def sample(self) -> "tuple[float, dict[str, Any]]":
        if self._t0 is None:
            self._t0 = self._clock()
        values: dict[str, Any] = {}
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        try:
            for s in self.specs:
                regs = self._read(sock, s.fc, s.address, s.words)
                if s.kind == "real":
                    values[s.tag] = real_from_words(regs[0], regs[1])
                elif s.kind == "bool":
                    values[s.tag] = int(regs[0])
                else:
                    values[s.tag] = regs[0]
        finally:
            sock.close()
        return self._clock() - self._t0, values

    def __call__(self) -> "tuple[float, dict[str, Any]]":
        return self.sample()


class ContinuousTraceRecorder:
    """Sample the simulator independently while an actor waits for model replies.

    A bounded action loop alone leaves multi-second blind spots in a physical
    trace. This recorder owns its own observer and preserves read failures so a
    later score cannot mistake missing evidence for a negative outcome.
    """

    def __init__(self, observer: ModbusObserver, *, interval_sec: float = 1.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if interval_sec <= 0:
            raise ValueError("recording interval must be positive")
        self.observer = observer
        self.interval_sec = interval_sec
        self.clock = clock
        self.started: float | None = None
        self.trace: list[tuple[float, dict[str, Any]]] = []
        self.errors: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.started is not None:
            raise RuntimeError("recorder already started")
        self.started = self.clock()
        # A failed first read is an admission failure, before model execution.
        t, values = self.observer()
        self.trace.append((t, values))
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()

    def _record(self) -> None:
        assert self.started is not None
        next_sample = self.started + self.interval_sec
        while not self._stop.is_set():
            if self._stop.wait(max(0.0, next_sample - self.clock())):
                return
            try:
                t, values = self.observer()
                self.trace.append((t, values))
            except Exception as exc:
                self.errors.append({"t_sec": self.clock() - self.started,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc)})
            next_sample += self.interval_sec

    def wait_until(self, duration_sec: float) -> None:
        if self.started is None:
            raise RuntimeError("recorder not started")
        remaining = self.started + duration_sec - self.clock()
        if remaining > 0:
            self._stop.wait(remaining)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.observer.timeout + 1.0))
            if self._thread.is_alive():
                raise RuntimeError("simulator recorder did not stop")


def admission_probe(host: str, port: int, specs: Sequence[SensorSpec],
                    timeout: float = 3.0) -> dict[str, Any]:
    """One sensor read to confirm the simulator endpoint is reachable (admission)."""
    t, values = ModbusObserver(host, port, specs, timeout=timeout).sample()
    return {"reachable": True, "sample": values}


def run_live_native_episode(
    *,
    controller_endpoint: tuple[str, int],
    simulator_endpoint: tuple[str, int],
    mappings: Mapping[str, Any],
    sensor_tags: Sequence[str],
    object_tags: Sequence[str],
    objective: Objective,
    generate: Callable[[str], str],
    workload: str,
    profile: str,
    episode_id: str,
    run_dir: Path,
    reset: Callable[[], Any] | None = None,
    deployment: Mapping[str, Any] | None = None,
    observer_clock: Callable[[], float] = time.monotonic,
    status_view: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    **run_kwargs: Any,
):
    """Run a native episode against a live OpenPLC pair, scoring simulator state.

    The broker fronts the controller; the observer reads the simulator's canonical
    modeled sensors; the object set is the controller-owned command registers.
    Admission (a sensor read) runs first, and ``reset`` (if given) before the run.
    """
    specs = sensor_specs_from_map(mappings, sensor_tags)
    object_set = object_set_from_map(mappings, object_tags)
    admission_probe(*simulator_endpoint, specs)
    if reset is not None:
        reset()
    observer = ModbusObserver(*simulator_endpoint, specs, clock=observer_clock)
    # The recorder owns a separate simulator connection and continues sampling
    # while model inference or sandbox commands take tens of seconds.
    recorder = ContinuousTraceRecorder(
        ModbusObserver(*simulator_endpoint, specs, clock=observer_clock), interval_sec=1.0,
        clock=observer_clock,
    )
    recorder.start()
    try:
        return run_native_episode(
            endpoint=controller_endpoint,
            generate=generate,
            observe=observer,
            objective=objective,
            object_set=object_set,
            workload=workload,
            profile=profile,
            episode_id=episode_id,
            run_dir=run_dir,
            deployment=deployment,
            status_view=status_view,
            trace_provider=lambda: list(recorder.trace),
            **run_kwargs,
        )
    finally:
        recorder.stop()
