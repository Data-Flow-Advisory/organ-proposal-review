#!/usr/bin/env python3
"""
check_ports.py — connection-standard conformance check for the
proposal-review organ.

Enforces the orchestrator connection standard (CONNECTORS.md) against this
organ's `ports.json`:

  1. ports.json parses and has the required shape
     ({"inputs":[{name,type,required}], "outputs":[{name,type}]}).
  2. Every declared `type` exists in the type vocabulary (types.json).
  3. decide() READS every declared input: each input `name` appears as a
     literal `state.get("<name>")` in organ.py (AST scan — the static
     reads-declared rule; no declared input is dead, no read key undeclared).
  4. decide() WRITES exactly the declared outputs: running decide() over the
     committed samples (which cover all dispatched kinds), the UNION of the
     top-level keys under the result `output` dict equals the declared output
     set (behavioural union-equality — no produced port is undeclared, and no
     declared port is never produced).

This organ is multi-operation (dispatched on `state.kind`), so outputs are
checked as a union across the sample set rather than per-call exact match:
each kind emits a different subset, and together the samples must exercise
every declared output exactly once-or-more with nothing left over.

Exit 0 = conformant, non-zero = violation (with a diagnostic on stderr).
Pure stdlib; no network, no third-party deps.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PORTS = os.path.join(_HERE, "ports.json")
_TYPES = os.path.join(_HERE, "types.json")
_ORGAN = os.path.join(_HERE, "organ.py")
_SAMPLES = os.path.join(_HERE, "samples")


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"check_ports: FAIL — {msg}", file=sys.stderr)
    raise SystemExit(1)


def _load_json(path: str, label: str):
    if not os.path.exists(path):
        _fail(f"{label} not found at {path}")
    try:
        with open(path) as fh:
            return json.load(fh)
    except (ValueError, json.JSONDecodeError) as e:
        _fail(f"{label} is not valid JSON: {e}")


def _vocabulary() -> set:
    types = _load_json(_TYPES, "types.json")
    if not isinstance(types, dict) or not isinstance(types.get("types"), dict):
        _fail("types.json must be an object with a 'types' object")
    vocab = set(types["types"].keys())
    if not vocab:
        _fail("types.json vocabulary is empty")
    return vocab


def _validate_port_shape(ports: dict) -> None:
    if not isinstance(ports, dict):
        _fail("ports.json must be a JSON object")
    for side, required_field in (("inputs", True), ("outputs", False)):
        if side not in ports or not isinstance(ports[side], list):
            _fail(f"ports.json missing list '{side}'")
        seen = set()
        for i, port in enumerate(ports[side]):
            if not isinstance(port, dict):
                _fail(f"{side}[{i}] is not an object")
            if not isinstance(port.get("name"), str) or not port["name"]:
                _fail(f"{side}[{i}] missing non-empty string 'name'")
            if not isinstance(port.get("type"), str) or not port["type"]:
                _fail(f"{side}[{i}] ({port.get('name')}) missing string 'type'")
            if required_field and not isinstance(port.get("required"), bool):
                _fail(f"inputs[{i}] ({port['name']}) missing boolean 'required'")
            if port["name"] in seen:
                _fail(f"{side} has duplicate port name {port['name']!r}")
            seen.add(port["name"])


def _check_types_in_vocab(ports: dict, vocab: set) -> None:
    for side in ("inputs", "outputs"):
        for port in ports[side]:
            if port["type"] not in vocab:
                _fail(
                    f"{side} port {port['name']!r} declares type {port['type']!r} "
                    f"which is not in the vocabulary {sorted(vocab)}"
                )


def _state_get_literals() -> set:
    """Every literal key passed to state.get(...) in organ.py."""
    with open(_ORGAN) as fh:
        tree = ast.parse(fh.read())
    keys = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "state"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
    return keys


def _check_reads_declared(ports: dict) -> None:
    declared = {p["name"] for p in ports["inputs"]}
    read = _state_get_literals()
    missing = declared - read
    if missing:
        _fail(
            "declared input(s) never read via state.get(...) in organ.py: "
            f"{sorted(missing)}"
        )
    undeclared = read - declared
    if undeclared:
        _fail(
            "organ.py reads state key(s) not declared as inputs: "
            f"{sorted(undeclared)}"
        )


def _produced_output_union() -> set:
    """Union of top-level output keys produced across all committed samples."""
    sys.path.insert(0, _HERE)
    import organ  # noqa: E402 — imported after path setup, intentionally

    sample_files = sorted(glob.glob(os.path.join(_SAMPLES, "*.json")))
    if not sample_files:
        _fail("no samples/*.json to verify outputs against")
    union = set()
    for path in sample_files:
        payload = _load_json(path, os.path.basename(path))
        if not isinstance(payload, dict) or "state" not in payload:
            _fail(f"sample {os.path.basename(path)} missing 'state'")
        result = organ.decide(payload["state"], payload.get("context"))
        out = result.get("output")
        if not isinstance(out, dict):
            _fail(f"sample {os.path.basename(path)} produced non-dict output")
        union |= set(out.keys())
    return union


def _check_writes_declared(ports: dict) -> None:
    declared = {p["name"] for p in ports["outputs"]}
    produced = _produced_output_union()
    undeclared = produced - declared
    if undeclared:
        _fail(
            "samples produce output key(s) not declared as outputs: "
            f"{sorted(undeclared)}"
        )
    never = declared - produced
    if never:
        _fail(
            "declared output(s) never produced by any sample (dead port): "
            f"{sorted(never)}"
        )


def main() -> int:
    ports = _load_json(_PORTS, "ports.json")
    vocab = _vocabulary()
    _validate_port_shape(ports)
    _check_types_in_vocab(ports, vocab)
    _check_reads_declared(ports)
    _check_writes_declared(ports)
    print(
        "check_ports: OK — "
        f"{len(ports['inputs'])} inputs, {len(ports['outputs'])} outputs; "
        "types in vocabulary; reads + writes match declared names."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
