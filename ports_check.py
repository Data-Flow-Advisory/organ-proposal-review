#!/usr/bin/env python3
"""Ports checker for the organ-proposal-review organ (the connection-standard gate).

This is the port half of conformance, per CONNECTORS.md ("Conformance gains a
port check"). It asserts, with NO arguments:

  1. ``ports.json`` parses and has the standard shape — ``inputs`` is a list of
     ``{name, type, required}`` and ``outputs`` is a list of ``{name, type}``.
  2. Every ``type`` named in ports.json exists in the shared type vocabulary
     (``types.json``, a vendored snapshot of the orchestrator vocabulary plus
     this organ's proposed additions).
  3. ``decide`` actually READS each declared input name and WRITES each declared
     output name — sampled against the organ's own ``samples/*.json``.

Because organ-proposal-review is a MULTI-KIND DISPATCHER (``state.kind`` selects
one of five behaviours), the read/write check is KIND-AWARE: each port carries a
``kind`` annotation naming the dispatch branch it belongs to, and the check
validates each port only against samples of that kind:

  * read-check: for each declared input name, there is at least one sample whose
    ``state.kind`` matches the port's kind AND whose ``state`` carries the name —
    proving decide() reads it on the branch that uses it.
  * write-check: running ``decide`` on every sample of the port's kind, the
    declared output name is a top-level key of the returned ``output`` dict —
    proving decide() writes it (the declared name is a "witness" key the kind
    emits on both its normal and fail-open paths).

Exits non-zero with a clear message on any violation, so the conformance
workflow goes RED if the organ's ports ever drift from its real decide() I/O or
reference a type outside the vocabulary.

Usage (in CI):  python3 ports_check.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import organ

_HERE = pathlib.Path(__file__).resolve().parent


def _fail(msg: str) -> int:
    print(f"ports violation: {msg}")
    return 1


def _load_json(name: str):
    path = _HERE / name
    if not path.exists():
        raise FileNotFoundError(f"{name} not found beside organ.py")
    with path.open() as fh:
        return json.load(fh)


def _port_kinds(port: dict) -> set:
    """Normalise a port's `kind` annotation (str or list) to a set of kinds."""
    k = port.get("kind")
    if isinstance(k, str):
        return {k}
    if isinstance(k, list):
        return {x for x in k if isinstance(x, str)}
    return set()


def main() -> int:
    # --- 1. ports.json parses + has the standard shape -----------------------
    try:
        ports = _load_json("ports.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return _fail(f"ports.json did not parse: {exc}")
    if not isinstance(ports, dict):
        return _fail("ports.json top-level is not an object")

    inputs = ports.get("inputs")
    outputs = ports.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return _fail("ports.json must carry list `inputs` and list `outputs`")

    for port in inputs:
        if not isinstance(port, dict) or "name" not in port or "type" not in port:
            return _fail(f"input port missing name/type: {port!r}")
        if "required" not in port:
            return _fail(f"input port {port.get('name')!r} missing `required`")
        if not isinstance(port["required"], bool):
            return _fail(f"input port {port['name']!r} `required` must be bool")
        if not _port_kinds(port):
            return _fail(f"input port {port['name']!r} missing a `kind` annotation")
    for port in outputs:
        if not isinstance(port, dict) or "name" not in port or "type" not in port:
            return _fail(f"output port missing name/type: {port!r}")
        if not _port_kinds(port):
            return _fail(f"output port {port['name']!r} missing a `kind` annotation")

    # --- 2. every referenced type exists in the vocabulary -------------------
    try:
        vocab = _load_json("types.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return _fail(f"types.json did not parse: {exc}")
    known_types = set((vocab.get("types") or {}).keys())
    if not known_types:
        return _fail("types.json carries no `types`")
    for port in inputs + outputs:
        t = port["type"]
        if t not in known_types:
            return _fail(
                f"port {port['name']!r} references type {t!r} "
                f"which is not in the vocabulary (types.json)"
            )

    # --- 3. decide reads/writes the declared names (kind-aware, sampled) -----
    sample_paths = sorted((_HERE / "samples").glob("*.json"))
    samples = []  # list of (name, state, context)
    for p in sample_paths:
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            return _fail(f"sample {p.name} did not parse: {exc}")
        st = doc.get("state")
        if isinstance(st, dict):
            samples.append((p.name, st, doc.get("context") or {}))
    if not samples:
        return _fail("no samples with a `state` dict to validate ports against")

    def _samples_of_kind(kinds: set):
        return [(n, st, ctx) for (n, st, ctx) in samples if st.get("kind") in kinds]

    # read-check: each declared input name is present under `state` in at least
    # one sample of its own kind (proving decide() reads it on that branch).
    for port in inputs:
        kinds = _port_kinds(port)
        of_kind = _samples_of_kind(kinds)
        if not of_kind:
            return _fail(
                f"declared input {port['name']!r} has kind(s) {sorted(kinds)} but "
                f"no sample exercises that kind — cannot prove decide() reads it"
            )
        if not any(port["name"] in st for (_n, st, _c) in of_kind):
            return _fail(
                f"declared input {port['name']!r} is never present under `state` "
                f"in any {sorted(kinds)} sample — cannot prove decide() reads it"
            )

    # write-check: decide()'s output carries each declared output name on every
    # sample of that output's kind (the name is a witness key the kind emits).
    for port in outputs:
        kinds = _port_kinds(port)
        of_kind = _samples_of_kind(kinds)
        if not of_kind:
            return _fail(
                f"declared output {port['name']!r} has kind(s) {sorted(kinds)} but "
                f"no sample exercises that kind — cannot prove decide() writes it"
            )
        for sname, st, ctx in of_kind:
            result = organ.decide(st, ctx)
            out = result.get("output")
            if not isinstance(out, dict):
                return _fail(f"decide() on {sname} returned no `output` dict")
            if port["name"] not in out:
                return _fail(
                    f"declared output {port['name']!r} absent from decide() "
                    f"output on {sorted(kinds)} sample {sname} — "
                    f"cannot prove decide() writes it"
                )

    print(
        f"  ports OK: {len(inputs)} input(s) + {len(outputs)} output(s), "
        f"all types in vocabulary, all names read/written across "
        f"{len(samples)} sample(s) (kind-aware)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
