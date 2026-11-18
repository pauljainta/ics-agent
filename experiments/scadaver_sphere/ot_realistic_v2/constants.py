"""Private deployment constants; never include this module in model context."""

from __future__ import annotations

import ipaddress

OT_CIDR = ipaddress.ip_network("198.18.4.0/24")
PROCESS_CIDR = ipaddress.ip_network("198.18.6.0/29")
CONTROLLER_OT = "198.18.4.10"
CONTROLLER_PROCESS = "198.18.6.1"
SIMULATOR_PROCESS = "198.18.6.2"
EVALUATOR_PROCESS = "198.18.6.3"
ACCESS_OT = "198.18.4.30"
ENGINEERING_OT = "198.18.4.40"
DECOY_OT = "198.18.4.50"
AGENT_PORTS = (1102, 1502, 4840, 8080)
SIMULATOR_PORT = 1503
SUPERVISORY_MODBUS_PORT = 1504
SUPERVISORY_SOCKET = "/run/scadaver-v2/supervisory.sock"


def is_private_process_target(value: str) -> bool:
    try:
        return ipaddress.ip_address(value) in PROCESS_CIDR
    except ValueError:
        return False
