"""Opt-in Docker proof: broker succeeds/logs; direct egress is unreachable."""

import os
import tempfile
from pathlib import Path

import pytest

from ics_agent.plcbench_broker import ModbusBroker
from ics_agent.plcbench_container import LinuxContainerSandbox
from ics_agent.plcbench_run import _LoopbackEndpoint


@pytest.mark.skipif(os.environ.get("PLCBENCH_DOCKER_TEST") != "1", reason="opt-in Docker integration")
def test_networkless_child_broker_and_egress():
    endpoint = _LoopbackEndpoint()
    broker = ModbusBroker(*endpoint.addr)
    broker.start()
    script = """
import os, socket, struct
s = socket.create_connection((os.environ['PLCBENCH_BROKER_HOST'], int(os.environ['PLCBENCH_BROKER_PORT'])), 3)
pdu = bytes([3]) + struct.pack('>HH', 0, 1)
s.sendall(struct.pack('>HHH', 1, 0, len(pdu)+1) + bytes([1]) + pdu)
reply = s.recv(32)
s.close()
assert len(reply) >= 9 and reply[7] == 3, reply
print('BROKER_OK')
try:
    socket.create_connection(('1.1.1.1', 80), 2)
except OSError:
    print('EGRESS_BLOCKED')
else:
    raise AssertionError('direct egress succeeded')
"""
    try:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = LinuxContainerSandbox(Path(directory), broker.local_addr,
                                            image="plcbench-native:py311-pymodbus386", wall_timeout_sec=20)
            try:
                result = sandbox.run("python", script)
                assert result.ok, result.stderr
                assert "BROKER_OK" in result.stdout and "EGRESS_BLOCKED" in result.stdout
                assert len(broker.transactions) == 1
                assert broker.transactions[0].op == "read"
                assert broker.transactions[0].accepted
            finally:
                sandbox.cleanup()
    finally:
        broker.stop()
        endpoint.stop()
