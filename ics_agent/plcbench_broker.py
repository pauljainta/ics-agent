"""Native-path broker for the PLCBench native actor (OpenPLC/Modbus-TCP phase).

The broker is a localhost TCP proxy in front of the *controller* endpoint. It is
the single native path the sandboxed actor is given, and it is where the
security-relevant properties the sandbox design promises are actually enforced:

- **Allowlist**: it only ever connects upstream to the one controller endpoint it
  was constructed with. The actor is handed the broker's local address, never the
  controller's, the simulator's, or any control-plane host.
- **Rate caps**: an attempted-operation ceiling and a successful-ops-per-second
  cap, so one script cannot emit unbounded transactions on shared hardware.
- **Transaction log**: every request/response pair is decoded and recorded at
  this boundary (outside the actor's writable evidence path), in the shape the
  offline scorer consumes for discover/read/write/manipulate.
- **Enforcement point for defended runs**: a deny rule applied here contains an
  actor holding native sockets, which an API/menu filter cannot.

Modbus/TCP only here; a CIP/Rockwell broker is a later, separately-gated variant
behind the same interface. Stdlib-only, so it unit-tests without the agent stack.

Scope note: this process enforces allowlist/caps/logging at the proxy. The
versioned batch runner combines it with a networkless Linux child and a Unix
socket relay. Legacy subprocess runs remain loopback-only and lack egress
isolation; non-loopback execution is closed pending independent review.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Modbus function codes grouped by intent.
_READ_FCS = {1: "coil", 2: "discrete_input", 3: "hr", 4: "input_register"}
_WRITE_FCS = {5: "coil", 6: "hr", 15: "coil", 16: "hr"}
_MBAP_LEN = 7  # transaction(2) protocol(2) length(2) unit(1)


@dataclass
class DecodedRequest:
    op: str                  # "read" | "write" | "probe"
    function_code: int
    space: str | None        # coil | hr | discrete_input | input_register
    address: int | None
    count: int               # registers/coils touched (1 for single)
    unit: int

    def destination(self) -> str | None:
        """Decoded destination for `manipulate` matching, e.g. 'hr:2066' or 'hr:2066-2067'."""
        if self.space is None or self.address is None or self.op != "write":
            return None
        if self.count <= 1:
            return f"{self.space}:{self.address}"
        return f"{self.space}:{self.address}-{self.address + self.count - 1}"


def decode_request_pdu(pdu: bytes, unit: int = 0) -> DecodedRequest:
    """Classify a Modbus request PDU (function code + data)."""
    if not pdu:
        return DecodedRequest("probe", -1, None, None, 0, unit)
    fc = pdu[0]
    if fc in _READ_FCS and len(pdu) >= 5:
        start, qty = struct.unpack(">HH", pdu[1:5])
        return DecodedRequest("read", fc, _READ_FCS[fc], start, qty, unit)
    if fc in (5, 6) and len(pdu) >= 5:
        addr = struct.unpack(">H", pdu[1:3])[0]
        return DecodedRequest("write", fc, _WRITE_FCS[fc], addr, 1, unit)
    if fc in (15, 16) and len(pdu) >= 5:
        addr, qty = struct.unpack(">HH", pdu[1:5])
        return DecodedRequest("write", fc, _WRITE_FCS[fc], addr, qty, unit)
    # Diagnostics / device-id / unknown: treat as a probe that may identify the service.
    return DecodedRequest("probe", fc, None, None, 0, unit)


def decode_response(req: DecodedRequest, resp_pdu: bytes) -> tuple[bool, bool]:
    """Return (accepted, nonempty) for a response PDU against its request."""
    if not resp_pdu:
        return False, False
    resp_fc = resp_pdu[0]
    if resp_fc & 0x80:  # exception response
        return False, False
    accepted = resp_fc == req.function_code
    nonempty = False
    if accepted and req.op == "read" and len(resp_pdu) >= 2:
        nonempty = resp_pdu[1] > 0  # byte count
    return accepted, nonempty


@dataclass
class NativeTxnRecord:
    """A logged native transaction; maps 1:1 to scorer.NativeTransaction."""

    t: float
    op: str
    accepted: bool = False
    nonempty: bool = False
    destination: str | None = None
    identifies_service: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 4),
            "op": self.op,
            "accepted": self.accepted,
            "nonempty": self.nonempty,
            "destination": self.destination,
            "identifies_service": self.identifies_service,
            "detail": self.detail,
        }


class RateLimiter:
    """Attempted-op ceiling (hard) + successful-ops-per-second sliding window."""

    def __init__(self, *, attempted_ceiling: int, success_per_sec: float,
                 clock=time.monotonic) -> None:
        self.attempted_ceiling = attempted_ceiling
        self.success_per_sec = success_per_sec
        self._clock = clock
        self._attempted = 0
        self._success_times: deque[float] = deque()

    def allow_attempt(self) -> bool:
        """Count an attempt; False once the attempted ceiling is exceeded."""
        self._attempted += 1
        return self._attempted <= self.attempted_ceiling

    def allow_success(self) -> bool:
        """Whether another successful op is within the per-second window now."""
        now = self._clock()
        while self._success_times and now - self._success_times[0] >= 1.0:
            self._success_times.popleft()
        if len(self._success_times) >= self.success_per_sec:
            return False
        self._success_times.append(now)
        return True

    @property
    def attempted(self) -> int:
        return self._attempted


class ModbusBroker:
    """Single-upstream Modbus/TCP proxy that logs + caps native transactions.

    Episodes are serialized, so one client connection at a time is sufficient.
    Start it, hand the actor ``local_addr``, and read ``transactions`` after.
    """

    def __init__(self, upstream_host: str, upstream_port: int, *,
                 success_ops_per_sec: float = 10.0, attempted_ceiling: int = 1000,
                 bind_host: str = "127.0.0.1", bind_port: int = 0,
                 clock=time.monotonic) -> None:
        self.upstream = (upstream_host, upstream_port)
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.limiter = RateLimiter(
            attempted_ceiling=attempted_ceiling,
            success_per_sec=success_ops_per_sec,
            clock=clock,
        )
        self._clock = clock
        self.transactions: list[NativeTxnRecord] = []
        self._t0 = clock()
        self._srv: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    @property
    def local_addr(self) -> tuple[str, int]:
        assert self._srv is not None, "broker not started"
        return self._srv.getsockname()

    def start(self) -> tuple[str, int]:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.bind_host, self.bind_port))
        self._srv.listen(1)
        self._srv.settimeout(0.5)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self.local_addr

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self._srv:
            self._srv.close()

    # -- framing -------------------------------------------------------
    @staticmethod
    def _recv_frame(sock: socket.socket) -> bytes | None:
        """Read one complete Modbus/TCP ADU using the MBAP length field."""
        header = _recv_exact(sock, _MBAP_LEN)
        if header is None:
            return None
        length = struct.unpack(">H", header[4:6])[0]  # unit + PDU
        body = _recv_exact(sock, length - 1)
        if body is None:
            return None
        return header + body

    def _record(self, rec: NativeTxnRecord) -> None:
        with self._lock:
            self.transactions.append(rec)

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._srv.accept()  # type: ignore[union-attr]
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handle(client)
            except OSError:
                # A client/upstream reset is a clean end of session, not an error.
                pass
            finally:
                client.close()

    def _handle(self, client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            upstream = socket.create_connection(self.upstream, timeout=3)
        except OSError:
            # Upstream (allowlisted controller) unreachable; nothing to proxy.
            return
        client.settimeout(1.0)
        upstream.settimeout(3.0)
        try:
            while not self._stop.is_set():
                try:
                    req = self._recv_frame(client)
                except socket.timeout:
                    continue
                if req is None:
                    break
                self._proxy_one(client, upstream, req)
        finally:
            upstream.close()

    def _proxy_one(self, client: socket.socket, upstream: socket.socket,
                   req_frame: bytes) -> None:
        unit = req_frame[6]
        decoded = decode_request_pdu(req_frame[_MBAP_LEN:], unit)
        t = self._clock() - self._t0

        if not self.limiter.allow_attempt():
            self._record(NativeTxnRecord(
                t=t, op=decoded.op, accepted=False, destination=decoded.destination(),
                detail={"capped": "attempted_ceiling", "fc": decoded.function_code}))
            return
        if decoded.op == "write" and not self.limiter.allow_success():
            self._record(NativeTxnRecord(
                t=t, op=decoded.op, accepted=False, destination=decoded.destination(),
                detail={"capped": "success_rate", "fc": decoded.function_code}))
            return

        upstream.sendall(req_frame)
        resp = self._recv_frame(upstream)
        if resp is None:
            self._record(NativeTxnRecord(t=t, op=decoded.op, accepted=False,
                                         detail={"error": "no_response"}))
            return
        accepted, nonempty = decode_response(decoded, resp[_MBAP_LEN:])
        if decoded.op in ("read", "probe") and accepted and not self.limiter.allow_success():
            accepted = False  # over the success cap; do not count it as a success
        self._record(NativeTxnRecord(
            t=t, op=decoded.op, accepted=accepted, nonempty=nonempty,
            destination=decoded.destination(),
            identifies_service=accepted,  # a successful native exchange identifies the service
            detail={"fc": decoded.function_code, "unit": unit,
                    "space": decoded.space, "address": decoded.address,
                    "count": decoded.count}))
        client.sendall(resp)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            if buf:
                continue
            return None
        except OSError:
            # Peer reset/closed the connection: treat as end of stream.
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
