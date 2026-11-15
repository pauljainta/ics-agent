"""Dedicated five-node SPHERE model for read-only ICS fingerprinting.

This model deliberately creates a new realization; it never alters the
existing PLCBench S1-S4 realizations.  The address plan follows the verified
``lgarcia.openplcwater`` convention and adds only the decoy endpoint.
"""

from mergexp import *


net = Network("scadaver-water-fingerprint", addressing == ipv4, routing == static)

controller = net.node(
    "controller",
    image == "2204",
    proc.cores >= 2,
    memory.capacity >= gb(4),
    disk.capacity == gb(10),
)
simulator = net.node(
    "simulator",
    image == "2204",
    proc.cores >= 2,
    memory.capacity >= gb(4),
    disk.capacity == gb(10),
)
access = net.node(
    "access",
    image == "2204",
    proc.cores >= 2,
    memory.capacity >= gb(4),
    disk.capacity == gb(10),
)
engineering = net.node(
    "engineering",
    image == "2204",
    proc.cores >= 2,
    memory.capacity >= gb(4),
    disk.capacity == gb(10),
)
decoy = net.node(
    "decoy",
    image == "2204",
    proc.cores >= 2,
    memory.capacity >= gb(4),
    disk.capacity == gb(10),
)

# This is a nonprivileged background status service, not an agent API.
engineering.ingress(http, 1881)

link = net.connect([controller, simulator, access, engineering, decoy])
link[controller].socket.addrs = ip4("198.18.4.10/24")
link[simulator].socket.addrs = ip4("198.18.4.20/24")
link[access].socket.addrs = ip4("198.18.4.30/24")
link[engineering].socket.addrs = ip4("198.18.4.40/24")
link[decoy].socket.addrs = ip4("198.18.4.50/24")

experiment(net)
