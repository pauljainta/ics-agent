"""Tests for the native-path Modbus broker (decoder, rate caps, proxy+logging)."""

from __future__ import annotations

import socket
import struct
import threading

from ics_agent.plcbench_broker import (
    _MBAP_LEN,
    ModbusBroker,
    RateLimiter,
    decode_request_pdu,
    decode_response,
)


# --- framing helpers for crafting test traffic ----------------------------
def mbap(pdu: bytes, *, txid: int = 1, unit: int = 1) -> bytes:
    return struct.pack(">HHH", txid, 0, len(pdu) + 1) + bytes([unit]) + pdu


def read_holding_req(start: int, qty: int) -> bytes:
    return mbap(bytes([3]) + struct.pack(">HH", start, qty))


def read_holding_resp(values: list[int]) -> bytes:
    data = b"".join(struct.pack(">H", v) for v in values)
    return mbap(bytes([3, len(data)]) + data)


def write_multi_req(addr: int, values: list[int]) -> bytes:
    data = b"".join(struct.pack(">H", v) for v in values)
    pdu = bytes([16]) + struct.pack(">HH", addr, len(values)) + bytes([len(data)]) + data
    return mbap(pdu)


def write_multi_resp(addr: int, qty: int) -> bytes:
    return mbap(bytes([16]) + struct.pack(">HH", addr, qty))


# --- decoder --------------------------------------------------------------
def test_decode_read_and_write_requests():
    r = decode_request_pdu(read_holding_req(2058, 2)[_MBAP_LEN:])
    assert (r.op, r.space, r.address, r.count) == ("read", "hr", 2058, 2)
    assert r.destination() is None  # reads have no manipulate destination

    w = decode_request_pdu(write_multi_req(2066, [0x4270, 0x0000])[_MBAP_LEN:])
    assert (w.op, w.space, w.address, w.count) == ("write", "hr", 2066, 2)
    assert w.destination() == "hr:2066-2067"

    single = decode_request_pdu(bytes([6]) + struct.pack(">HH", 40, 1))
    assert single.op == "write" and single.destination() == "hr:40"
    coil = decode_request_pdu(bytes([5]) + struct.pack(">HH", 40, 0xFF00))
    assert coil.destination() == "coil:40"


def test_decode_probe_and_exception():
    probe = decode_request_pdu(bytes([43, 14, 1, 0]))  # read device id (FC43)
    assert probe.op == "probe"
    req = decode_request_pdu(read_holding_req(10, 1)[_MBAP_LEN:])
    accepted, nonempty = decode_response(req, bytes([0x83, 0x02]))  # exception
    assert accepted is False and nonempty is False
    accepted, nonempty = decode_response(req, bytes([3, 2, 0x00, 0x05]))
    assert accepted is True and nonempty is True


# --- rate limiter ---------------------------------------------------------
def test_attempted_ceiling():
    rl = RateLimiter(attempted_ceiling=3, success_per_sec=100)
    assert [rl.allow_attempt() for _ in range(4)] == [True, True, True, False]
    assert rl.attempted == 4


def test_success_rate_window_with_fake_clock():
    now = {"t": 0.0}
    rl = RateLimiter(attempted_ceiling=1000, success_per_sec=2, clock=lambda: now["t"])
    assert rl.allow_success() is True   # 1st in this second
    assert rl.allow_success() is True   # 2nd
    assert rl.allow_success() is False  # 3rd within the same second -> capped
    now["t"] = 1.01                      # window advances
    assert rl.allow_success() is True


# --- end-to-end proxy + logging ------------------------------------------
class _FakeController:
    """Minimal upstream: answers FC3 reads and FC16 writes with canned frames."""

    def __init__(self) -> None:
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.addr = self.srv.getsockname()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        self.srv.settimeout(0.5)
        try:
            conn, _ = self.srv.accept()
        except socket.timeout:
            return
        conn.settimeout(1.0)
        with conn:
            while not self._stop.is_set():
                header = _recv(conn, _MBAP_LEN)
                if header is None:
                    return
                length = struct.unpack(">H", header[4:6])[0]
                body = _recv(conn, length - 1)
                if body is None:
                    return
                fc = body[0]  # body is the PDU: [fc, ...]
                if fc == 3:
                    conn.sendall(read_holding_resp([0x4270, 0x0000]))
                elif fc == 16:
                    addr, qty = struct.unpack(">HH", body[1:5])
                    conn.sendall(write_multi_resp(addr, qty))

    def stop(self) -> None:
        self._stop.set()
        self.srv.close()


def _recv(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def test_proxy_forwards_and_logs_transactions():
    upstream = _FakeController()
    broker = ModbusBroker(*upstream.addr, success_ops_per_sec=100, attempted_ceiling=1000)
    host, port = broker.start()
    try:
        c = socket.create_connection((host, port), timeout=2)
        c.settimeout(2.0)
        c.sendall(read_holding_req(2058, 2))
        assert _recv(c, _MBAP_LEN) is not None  # response relayed back
        # drain the read response body
        c.sendall(write_multi_req(2066, [0x4270, 0x0000]))
        # give the broker a moment to log both
        deadline = __import__("time").monotonic() + 2.0
        while len(broker.transactions) < 2 and __import__("time").monotonic() < deadline:
            __import__("time").sleep(0.02)
        c.close()
    finally:
        broker.stop()
        upstream.stop()

    ops = [(t.op, t.accepted, t.destination) for t in broker.transactions]
    assert ("read", True, None) in ops
    assert ("write", True, "hr:2066-2067") in ops
    # A successful native exchange identifies the service (discover evidence).
    assert any(t.identifies_service for t in broker.transactions)


def test_attempted_ceiling_blocks_before_upstream():
    upstream = _FakeController()
    broker = ModbusBroker(*upstream.addr, success_ops_per_sec=100, attempted_ceiling=1)
    host, port = broker.start()
    try:
        c = socket.create_connection((host, port), timeout=2)
        c.settimeout(2.0)
        c.sendall(read_holding_req(10, 1))
        _recv(c, _MBAP_LEN)
        c.sendall(read_holding_req(11, 1))  # 2nd attempt exceeds ceiling=1
        import time as _t
        deadline = _t.monotonic() + 2.0
        while len(broker.transactions) < 2 and _t.monotonic() < deadline:
            _t.sleep(0.02)
        c.close()
    finally:
        broker.stop()
        upstream.stop()
    capped = [t for t in broker.transactions if t.detail.get("capped")]
    assert capped and capped[0].detail["capped"] == "attempted_ceiling"
    assert capped[0].accepted is False
