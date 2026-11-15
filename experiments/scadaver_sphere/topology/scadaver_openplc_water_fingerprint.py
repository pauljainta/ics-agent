"""Additive OpenPLC realization for runtime/process fingerprinting.

The legacy ``scadaver-water-fingerprint`` realization remains the Python
controller baseline. Materialize this as a distinct realization only after an
operator reviews the deployment runbook.
"""

from mergexp import *


net = Network("scadaver-openplc-water-fingerprint", addressing == ipv4, routing == static)

def node(name):
    return net.node(name, image == "2204", proc.cores >= 2, memory.capacity >= gb(4), disk.capacity == gb(10))

controller, simulator, access, engineering, decoy = (node(name) for name in ("controller", "simulator", "access", "engineering", "decoy"))
engineering.ingress(http, 1881)
link = net.connect([controller, simulator, access, engineering, decoy])
for target, address in ((controller, "198.18.4.10/24"), (simulator, "198.18.4.20/24"), (access, "198.18.4.30/24"), (engineering, "198.18.4.40/24"), (decoy, "198.18.4.50/24")):
    link[target].socket.addrs = ip4(address)

experiment(net)
