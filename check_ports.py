#!/usr/bin/env python3
"""Connection-standard ports conformance check for organ-proposal-review.

Asserts, with stdlib only:

  1. ``ports.json`` parses and has the declared ``{inputs, outputs}`` shape
     (every input carries name/type/boolean-required; every output name/type).
  2. ``types.json`` parses and declares a non-empty type vocabulary.
  3. Every ``type`` named in ports.json exists in the types.json vocabulary.
  4. ``decide()`` READS every declared input name from ``state`` — verified by a
     static scan of ``organ.py`` for ``state.get("<name>")``.
  5. ``decide()`` WRITES exactly the declared output names — verified
     BEHAVIOURALLY: across a representative run of every kind, the union of
     ``output`` keys produced must equal the declared output set (no undeclared
     key is ever emitted on the canonical fixtures, and every declared output is
     reachable from some kind).

This organ is multi-op (``decide`` dispatches on ``state.kind``); ports.json
declares the UNION of inputs/outputs over all kinds, so the output check is a
union-equality, not a per-call equality. NOTE: the normalization kinds pass a
caller-supplied ``parsed`` dict through, so production output MAY carry extra
keys beyond the declared (canonical) set — the committed fixtures pin the
canonical set, which is what this check verifies.

Exit 0 on success, non-zero (with a diagnostic) on any violation.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _fail(msg: str) -> "None":
    print(f"check_ports: FAIL — {msg}", file=sys.stderr)
    raise SystemExit(1)


def _load(name: str) -> dict:
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        _fail(f"{name} is missing")
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as e:  # noqa: BLE001
        _fail(f"{name} does not parse as JSON: {e}")


def main() -> int:
    ports = _load("ports.json")
    types = _load("types.json")

    # --- 1. ports.json shape -------------------------------------------------
    if not isinstance(ports, dict):
        _fail("ports.json must be a JSON object")
    for key in ("inputs", "outputs"):
        if not isinstance(ports.get(key), list):
            _fail(f"ports.json must declare a list '{key}'")
    for spec in ports["inputs"]:
        if not isinstance(spec, dict) or "name" not in spec or "type" not in spec:
            _fail(f"each input needs 'name' and 'type': {spec!r}")
        if "required" not in spec:
            _fail(f"each input needs a 'required' flag: {spec!r}")
        if not isinstance(spec["required"], bool):
            _fail(f"input 'required' must be boolean: {spec!r}")
    for spec in ports["outputs"]:
        if not isinstance(spec, dict) or "name" not in spec or "type" not in spec:
            _fail(f"each output needs 'name' and 'type': {spec!r}")

    # --- 2. types.json vocabulary -------------------------------------------
    vocab = types.get("types")
    if not isinstance(vocab, dict) or not vocab:
        _fail("types.json must declare a non-empty 'types' object")
    vocab_names = set(vocab.keys())

    # --- 3. every declared type exists in the vocabulary ---------------------
    for spec in ports["inputs"] + ports["outputs"]:
        if spec["type"] not in vocab_names:
            _fail(
                f"port {spec['name']!r} uses type {spec['type']!r} which is not "
                f"in the types.json vocabulary {sorted(vocab_names)}"
            )

    # --- 4. decide() reads every declared input from state -------------------
    with open(os.path.join(HERE, "organ.py")) as fh:
        source = fh.read()
    for spec in ports["inputs"]:
        name = spec["name"]
        pat = re.compile(r"""state\.get\(\s*['"]""" + re.escape(name) + r"""['"]""")
        if not pat.search(source):
            _fail(
                f"declared input {name!r} is never read via state.get({name!r}) "
                f"in organ.py — ports.json and decide() disagree"
            )

    # --- 5. decide() writes exactly the declared outputs (behavioural) -------
    sys.path.insert(0, HERE)
    from organ import decide  # noqa: E402

    # Representative input per kind: the committed samples (cover all 5 kinds)
    # plus synthetic fail-open / not_found runs so every code path is exercised.
    runs = []
    for fname in sorted(os.listdir(os.path.join(HERE, "samples"))):
        if fname.endswith(".json"):
            with open(os.path.join(HERE, "samples", fname)) as fh:
                payload = json.load(fh)
            runs.append((fname, payload.get("state"), payload.get("context")))
    runs.append(("synthetic:critique_failopen", {"kind": "critique", "persona_name": "Tim"}, {}))
    runs.append(("synthetic:synthesis_failopen", {"kind": "synthesis"}, {}))
    runs.append(("synthetic:status_not_found", {"kind": "status", "found": False}, {}))

    declared_outputs = {spec["name"] for spec in ports["outputs"]}
    observed_outputs: set = set()
    for label, state, context in runs:
        res = decide(state, context)
        out = res.get("output")
        if not isinstance(out, dict):
            _fail(f"run {label!r} output is not an object: {out!r}")
        undeclared = set(out.keys()) - declared_outputs
        if undeclared:
            _fail(
                f"run {label!r} emitted undeclared output key(s) {sorted(undeclared)}; "
                f"declared outputs are {sorted(declared_outputs)}"
            )
        observed_outputs |= set(out.keys())

    unreachable = declared_outputs - observed_outputs
    if unreachable:
        _fail(
            f"declared output(s) {sorted(unreachable)} are never produced by any kind — "
            f"ports.json over-declares outputs"
        )

    print(
        "check_ports: OK — ports.json parses, "
        f"{len(ports['inputs'])} input(s)/{len(ports['outputs'])} output(s) "
        "use known types, decide() reads every declared input, and the union of "
        "produced outputs over all kinds equals the declared output set."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
