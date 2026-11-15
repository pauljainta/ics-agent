"""Five-node Merge experiment model. Review image names before realization."""

from mergexp import Network, experiment, ip4


LAB = {
    "access": "198.18.2.1/29",
    "controller": "198.18.2.2/29",
    "simulator": "198.18.2.3/29",
    "engineering": "198.18.2.4/29",
    "decoy": "198.18.2.5/29",
}


def build() -> None:
    net = Network("scadaver-sphere-readonly")
    nodes = {name: net.node(name) for name in LAB}
    link = net.connect(list(nodes.values()))
    for name, node in nodes.items():
        link[node].socket.addrs = ip4(LAB[name])
    experiment(net)


if __name__ == "__main__":
    build()
