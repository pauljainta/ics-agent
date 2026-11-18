"""Versioned SPHERE model for the isolated-process SCADAver laboratory.

Materialize as a new realization only after an operator approves its name.
The original ``scadaver.lgarcia`` realization is intentionally untouched.
"""

from mergexp import *


# Addresses are fully explicit. Do not enable the static-routing reticulator:
# it would add cross-segment routes through the dual-homed controller.
net = Network("scadaver-ot-realistic-v2", experimentnetresolution == False)


def node(name):
    return net.node(
        name,
        image == "2204",
        proc.cores >= 2,
        memory.capacity >= gb(4),
        disk.capacity == gb(10),
    )


controller = node("controller")
access = node("access")
engineering = node("engineering")
decoy = node("decoy")
simulator = net.node(
    "simulator", image == "2204", proc.cores >= 2,
    memory.capacity >= gb(4), disk.capacity == gb(10),
)
evaluator = net.node(
    "evaluator", image == "2204", proc.cores >= 2,
    memory.capacity >= gb(4), disk.capacity == gb(10),
)

# Only the engineering UI is intentionally exposed through SPHERE ingress.
engineering.ingress(http, 1881)

ot_link = net.connect([controller, access, engineering, decoy])
for target, address in (
    (controller, "198.18.4.10/24"),
    (access, "198.18.4.30/24"),
    (engineering, "198.18.4.40/24"),
    (decoy, "198.18.4.50/24"),
):
    ot_link[target].socket.addrs = ip4(address)

process_link = net.connect([controller, simulator, evaluator])
for target, address in (
    (controller, "198.18.6.1/29"),
    (simulator, "198.18.6.2/29"),
    (evaluator, "198.18.6.3/29"),
):
    process_link[target].socket.addrs = ip4(address)

experiment(net)
