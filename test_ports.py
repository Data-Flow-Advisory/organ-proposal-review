"""Bind the connection-standard ports check into the pytest suite.

The conformance Action runs ``python check_ports.py`` directly, but binding it
here too means a local ``pytest -v`` (and the Action's existing test step)
catches a ports.json / decide() drift without a separate invocation.
"""

import json
import os

import check_ports

_HERE = os.path.dirname(__file__)


def test_check_ports_passes():
    # check_ports.main() exits 0 on success, raises SystemExit(1) on violation.
    assert check_ports.main() == 0


def test_ports_json_parses_and_shaped():
    with open(os.path.join(_HERE, "ports.json")) as fh:
        ports = json.load(fh)
    assert isinstance(ports["inputs"], list) and ports["inputs"]
    assert isinstance(ports["outputs"], list) and ports["outputs"]
    # exactly one universally-required input: the 'kind' discriminator
    required = [p["name"] for p in ports["inputs"] if p["required"]]
    assert required == ["kind"], required


def test_every_port_type_in_vocabulary():
    with open(os.path.join(_HERE, "ports.json")) as fh:
        ports = json.load(fh)
    with open(os.path.join(_HERE, "types.json")) as fh:
        vocab = set(json.load(fh)["types"].keys())
    for spec in ports["inputs"] + ports["outputs"]:
        assert spec["type"] in vocab, (spec["name"], spec["type"])
