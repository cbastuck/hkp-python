"""Every way out of this server goes through the egress guard (``hkp/egress.py``).

Services reach a board's destinations with ``client_session`` (aiohttp) and
``urlopen`` (urllib) from there; this reads the sources for any other way of
opening a connection and fails on one it was not told about, saying what to use
instead. It reads the code rather than running it, so it is a tripwire, not a
proof: the backstop on a hosted machine is still its firewall. The same check
for hkp-node is ``hkp-node/tests/egress-sources.test.ts``.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "hkp"

# Ways to open a connection that the guard does not see.
RAW = {
    "an aiohttp ClientSession": re.compile(r"\bClientSession\("),
    "an aiohttp TCPConnector": re.compile(r"\bTCPConnector\("),
    "an aiohttp WebSocket client": re.compile(r"\bws_connect\("),
    "urllib's urlopen": re.compile(r"urllib\.request\.urlopen\(|\bbuild_opener\("),
    "urllib's urlretrieve": re.compile(r"\burlretrieve\("),
    "an http.client connection": re.compile(r"\bHTTPS?Connection\("),
    "a raw socket": re.compile(
        r"\bsocket\.(socket|create_connection)\(|\bopen_connection\(|\bcreate_connection\("
    ),
}

# Where a raw connection is not a board's destination, and why. Each entry is
# a decision; add one only with its reason.
EXEMPT: dict[str, dict[str, str]] = {
    "egress.py": {what: "the guard itself" for what in RAW},
    "coordinator_links.py": {
        "an aiohttp ClientSession": "the link to a coordinator, whose address the operator gave",
        "an aiohttp WebSocket client": "the link to a coordinator, whose address the operator gave",
    },
    "services/text_to_speech.py": {
        "urllib's urlretrieve": "the model download, from a fixed release address no board names",
    },
}


def _code(path: Path) -> str:
    """The source with comments taken out, so that describing a call is not
    making it."""
    return re.sub(r"#.*$", "", path.read_text(), flags=re.MULTILINE)


def test_every_way_out_is_known_to_the_guard() -> None:
    found = []
    for path in sorted(SRC.rglob("*.py")):
        name = path.relative_to(SRC).as_posix()
        text = _code(path)
        for what, pattern in RAW.items():
            if pattern.search(text) and what not in EXEMPT.get(name, {}):
                found.append(
                    f"{name} opens {what}. Use client_session() or urlopen() from "
                    "hkp/egress.py, which check where the connection goes"
                )
    assert found == []


def test_no_exemption_outlives_its_reason() -> None:
    for name, exempt in EXEMPT.items():
        if name == "egress.py":
            continue
        text = _code(SRC / name)
        for what in exempt:
            assert RAW[what].search(text), f"{name} no longer opens {what}"
