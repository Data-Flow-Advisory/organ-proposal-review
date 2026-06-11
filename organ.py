#!/usr/bin/env python3
"""
Proposal Review Organ — extracted decision logic from discovery-engine.

A pure decider for the multi-persona proposal-review pipeline
(`app/services/proposal_review.py`, Stream 39 Stage 3). It folds the
DETERMINISTIC parts of that pipeline — output normalization, EU AI Act
Article 13 source attribution, and review-job status derivation — into a
single `decide(state, context)` entry point dispatched by `state.kind`.

What it does NOT do: it never calls an LLM, the DB, or the network. Every
LLM/DB side effect in the original (the Claude critique calls, the
`record_api_call` cost ledger, `job_queue.update_progress`, the
`PendingWidgetAction` creation) lives OUTSIDE the organ. The organ receives
the ALREADY-PARSED model output (or the raw session-result text for the
team-review JSON-extraction case) and applies the pure normalization +
attribution + status rules.

Contract:
  INPUT state: {"kind": <discriminator>, ...kind-specific fields}

  Dispatched kinds:
    - "team_review"        normalize a team-persona session result
                           (parse_team_persona_result)
    - "critique"           normalize a Haiku persona critique + resolve role
                           (generate_persona_critique post-parse)
    - "interviewee_review" normalize an AI-reconstructed interviewee review
                           (run_interviewee_review post-parse)
    - "synthesis"          normalize a synthesis result + stamp source
                           attribution defaults (synthesise_fast post-parse)
    - "status"             derive the overall review status string
                           (get_review_status)

  OUTPUT: {
    "output": <normalized dict | status dict>,
    "rationale": str,
    "self_metric": {"confidence": float, "decision_path": str, ...}
  }

The organ is pure:
  - Takes all inputs via JSON.
  - Makes no DB / network / LLM calls.
  - Returns only computed advice.
  - Never raises on bad input (fail-open to an empty-but-valid shape).
"""

from __future__ import annotations

import json
import re
import sys
import os

# Canonical roster + roles, mirrored from app/services/proposal_review.py.
DEFAULT_TEAM_PERSONAS = ["Tim", "Matt", "Sam", "Jordan", "Priya"]

PERSONA_ROLES = {
    "Tim": "Head of Sales",
    "Matt": "CTO",
    "Sam": "Head of Product",
    "Jordan": "Finance Director",
    "Priya": "Head of Compliance",
}

# Job states that count as "still working" when deriving overall status.
_PENDING_OR_RUNNING = ("pending", "running")


# ---------------------------------------------------------------------------
# Helpers (pure)
# ---------------------------------------------------------------------------

def _as_list(value):
    """Return value if it is already a list, else an empty list."""
    return value if isinstance(value, list) else []


def _coerce_number_or_none(value):
    """int/float passes through; everything else becomes None.

    Mirrors the `overall_strength` / `overall_readiness` coercion in the
    original. ``bool`` is a subclass of ``int`` in Python; the source code
    used ``isinstance(x, (int, float))`` which accepts bools, so we keep that
    behaviour for faithfulness.
    """
    if isinstance(value, (int, float)):
        return value
    return None


def _extract_json_from_text(session_result: dict):
    """Replicate parse_team_persona_result's JSON-extraction heuristic.

    Returns ``(parsed_dict, decision_path, parse_ok)``. Never raises.
    """
    raw = ""
    summary = session_result.get("summary") or ""
    findings = session_result.get("findings") or []
    if findings:
        first = findings[0]
        raw = json.dumps(first) if isinstance(first, dict) else str(first)
    if not raw:
        raw = summary

    decision_path = "raw"
    match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        raw = match.group(1)
        decision_path = "json_block"
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
            decision_path = "bare_braces"
        else:
            decision_path = "no_json"

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}, decision_path, False
        return parsed, decision_path, True
    except (ValueError, json.JSONDecodeError):
        return {}, decision_path, False


# ---------------------------------------------------------------------------
# Per-kind deciders (pure)
# ---------------------------------------------------------------------------

def _decide_team_review(state: dict) -> dict:
    """Normalize a team-persona session result into the canonical review.

    Faithful to ``parse_team_persona_result(session_result, persona_name)``.
    """
    persona_name = state.get("persona_name") or "Unknown persona"
    session_result = state.get("session_result") or {}

    parsed, path, parse_ok = _extract_json_from_text(session_result)

    parsed.setdefault("persona_name", persona_name)
    parsed.setdefault("perspective", f"{persona_name}'s perspective")
    for key in ("strong_points", "concerns", "recommended_changes"):
        parsed[key] = _as_list(parsed.get(key))
    parsed["overall_strength"] = _coerce_number_or_none(parsed.get("overall_strength"))
    # Team persona reviews are always AI-generated.
    parsed["ai_generated"] = True

    return {
        "output": parsed,
        "rationale": (
            f"Normalized team review for '{persona_name}' "
            f"(JSON extracted via {path}, parse_ok={parse_ok})."
        ),
        "self_metric": {
            "confidence": 1.0 if parse_ok else 0.5,
            "decision_path": f"team_review:{path}",
            "parse_ok": parse_ok,
        },
    }


def _decide_critique(state: dict, context: dict) -> dict:
    """Normalize a Haiku persona critique + resolve the persona role.

    Faithful to the post-parse path of ``generate_persona_critique``. The
    role map may be overridden via ``context["persona_roles"]`` or
    ``state["persona_roles"]`` (state wins) so callers can extend the roster
    without a code change.
    """
    persona_name = state.get("persona_name") or "Unknown persona"
    roles = state.get("persona_roles") or context.get("persona_roles") or PERSONA_ROLES
    role = roles.get(persona_name, "Advisor")

    parsed = state.get("parsed")
    parse_ok = isinstance(parsed, dict)
    if not parse_ok:
        # Fail-open: LLM/parse failure → empty-but-valid critique (the
        # original returned exactly this shape from its except branch).
        out = {
            "persona_name": persona_name,
            "role": role,
            "strong_points": [],
            "concerns": [],
            "critical_changes": [],
        }
        if state.get("error"):
            out["error"] = str(state["error"])
        return {
            "output": out,
            "rationale": f"Critique parse unavailable for '{persona_name}'; emitting empty critique (fail-open).",
            "self_metric": {
                "confidence": 0.0,
                "decision_path": "critique:fail_open",
                "parse_ok": False,
            },
        }

    parsed.setdefault("persona_name", persona_name)
    parsed.setdefault("role", role)
    for key in ("strong_points", "concerns", "critical_changes"):
        parsed[key] = _as_list(parsed.get(key))

    return {
        "output": parsed,
        "rationale": f"Normalized critique for '{persona_name}' ({role}).",
        "self_metric": {
            "confidence": 1.0,
            "decision_path": "critique:normalized",
            "parse_ok": True,
        },
    }


def _decide_interviewee_review(state: dict) -> dict:
    """Normalize an AI-reconstructed interviewee review.

    Faithful to the post-parse path of ``run_interviewee_review``. Always
    stamps ``ai_generated=True`` — these are reconstructed perspectives, not
    real statements, which the EU AI Act Article 13 attribution depends on.
    """
    person_name = state.get("person_name") or "Unknown participant"
    role = state.get("role") or "Unknown role"

    parsed = state.get("parsed")
    parse_ok = isinstance(parsed, dict)
    if not parse_ok:
        out = {
            "person_name": person_name,
            "role": role,
            "agrees": [],
            "disagrees": [],
            "missing": [],
            "quotes": [],
            "ai_generated": True,
        }
        if state.get("error"):
            out["error"] = str(state["error"])
        return {
            "output": out,
            "rationale": f"Interviewee review parse unavailable for '{person_name}'; emitting empty review (fail-open).",
            "self_metric": {
                "confidence": 0.0,
                "decision_path": "interviewee_review:fail_open",
                "parse_ok": False,
            },
        }

    parsed.setdefault("person_name", person_name)
    parsed.setdefault("role", role)
    for key in ("agrees", "disagrees", "missing", "quotes"):
        parsed[key] = _as_list(parsed.get(key))
    parsed["ai_generated"] = True

    return {
        "output": parsed,
        "rationale": f"Normalized AI-reconstructed interviewee review for '{person_name}'.",
        "self_metric": {
            "confidence": 1.0,
            "decision_path": "interviewee_review:normalized",
            "parse_ok": True,
        },
    }


def _decide_synthesis(state: dict) -> dict:
    """Normalize a synthesis result + stamp source-attribution defaults.

    Faithful to the post-parse path of ``synthesise_fast``. Every
    ``change_list`` item must carry ``source_type`` + ``source_label`` for EU
    AI Act Article 13 attribution; missing ones default to "unknown"/"".
    """
    parsed = state.get("parsed")
    parse_ok = isinstance(parsed, dict)
    if not parse_ok:
        out = {
            "dfa_team_perspectives": [],
            "interviewee_perspectives": [],
            "change_list": [],
            "quote_weaving_map": [],
            "overall_readiness": None,
        }
        if state.get("error"):
            out["error"] = str(state["error"])
        return {
            "output": out,
            "rationale": "Synthesis parse unavailable; emitting empty synthesis (fail-open).",
            "self_metric": {
                "confidence": 0.0,
                "decision_path": "synthesis:fail_open",
                "parse_ok": False,
            },
        }

    for key in ("change_list", "quote_weaving_map"):
        parsed[key] = _as_list(parsed.get(key))
    unlabeled = 0
    for item in parsed["change_list"]:
        if not isinstance(item, dict):
            continue
        if not item.get("source_type"):
            unlabeled += 1
        item.setdefault("source_type", "unknown")
        item.setdefault("source_label", "")
    parsed["overall_readiness"] = _coerce_number_or_none(parsed.get("overall_readiness"))
    parsed.setdefault("dfa_team_perspectives", [])
    parsed.setdefault("interviewee_perspectives", [])

    return {
        "output": parsed,
        "rationale": (
            f"Normalized synthesis: {len(parsed['change_list'])} change(s), "
            f"{unlabeled} needed a source_type default."
        ),
        "self_metric": {
            "confidence": 1.0,
            "decision_path": "synthesis:normalized",
            "parse_ok": True,
            "change_count": len(parsed["change_list"]),
            "unlabeled_changes": unlabeled,
        },
    }


def _decide_status(state: dict) -> dict:
    """Derive the overall review status string.

    Faithful to ``get_review_status``. The caller resolves child job statuses
    (missing children default to "pending"); we pad ``child_statuses`` up to
    ``expected_child_count`` with "pending" to reproduce the original
    ``child_statuses.get(jid, "pending")`` default.
    """
    found = state.get("found", True)
    if not found:
        return {
            "output": {"overall": "not_found"},
            "rationale": "Synthesis job not found or tenant mismatch.",
            "self_metric": {"confidence": 1.0, "decision_path": "status:not_found"},
        }

    synth_status = state.get("synth_status") or "pending"
    child_statuses = list(_as_list(state.get("child_statuses")))
    expected = state.get("expected_child_count")
    if isinstance(expected, int) and expected > len(child_statuses):
        child_statuses += ["pending"] * (expected - len(child_statuses))

    children_pending = any(s in _PENDING_OR_RUNNING for s in child_statuses)

    if synth_status == "completed":
        overall = "complete"
        path = "synth_completed"
    elif synth_status == "failed":
        overall = "failed"
        path = "synth_failed"
    elif synth_status == "running" or children_pending:
        overall = "running"
        path = "children_pending" if children_pending and synth_status != "running" else "synth_running"
    else:
        overall = "pending"
        path = "default_pending"

    return {
        "output": {"overall": overall, "children_pending": children_pending},
        "rationale": (
            f"synth_status={synth_status!r}, {len(child_statuses)} child(ren), "
            f"children_pending={children_pending} → {overall}."
        ),
        "self_metric": {
            "confidence": 1.0,
            "decision_path": f"status:{path}",
        },
    }


_DISPATCH = {
    "team_review": lambda s, c: _decide_team_review(s),
    "critique": _decide_critique,
    "interviewee_review": lambda s, c: _decide_interviewee_review(s),
    "synthesis": lambda s, c: _decide_synthesis(s),
    "status": lambda s, c: _decide_status(s),
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def decide(state: dict, context: dict | None = None) -> dict:
    """Dispatch on ``state['kind']`` and return the normalized decision.

    Args:
        state: {"kind": <discriminator>, ...kind-specific fields}
        context: optional overrides (e.g. {"persona_roles": {...}})

    Returns:
        {"output": ..., "rationale": str, "self_metric": {...}}

    Fail-open: unknown/missing kind or any internal error returns a valid
    envelope with confidence 0.0 rather than raising.
    """
    context = context or {}
    try:
        if not isinstance(state, dict):
            raise TypeError("state must be a dict")
        kind = state.get("kind")
        handler = _DISPATCH.get(kind)
        if handler is None:
            return {
                "output": {},
                "rationale": f"Unknown kind {kind!r}; expected one of {sorted(_DISPATCH)}.",
                "self_metric": {"confidence": 0.0, "decision_path": "unknown_kind"},
            }
        return handler(state, context)
    except Exception as e:  # noqa: BLE001 — fail-open contract
        return {
            "output": {},
            "rationale": f"Decision logic error (fail-open): {e}",
            "self_metric": {"confidence": 0.0, "decision_path": "error_fallback"},
        }


def main() -> int:
    path = os.environ.get("ORGAN_INPUT")
    raw = open(path).read() if path else sys.stdin.read()
    try:
        payload = json.loads(raw)
        state = payload["state"]
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"invalid input: {e}"}), file=sys.stderr)
        return 1
    print(json.dumps(decide(state, payload.get("context")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
