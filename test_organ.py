"""
Pytest suite for the proposal-review organ.

Covers each dispatched kind plus the cross-cutting fail-open contract:
  - team_review     JSON extraction + normalization
  - critique        role resolution + normalization + fail-open
  - interviewee_review  AI-reconstruction stamping + fail-open
  - synthesis       source-attribution defaulting + fail-open
  - status          overall-status derivation
  - dispatch        unknown/missing kind, bad input
"""

import json
import os
import sys

import pytest

from organ import decide, PERSONA_ROLES, DEFAULT_TEAM_PERSONAS

_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


def _run_sample(filename):
    """Load a committed sample file and run it through ``decide`` exactly as
    the conformance Action does (ORGAN_INPUT → {state, context})."""
    with open(os.path.join(_SAMPLES_DIR, filename)) as fh:
        payload = json.load(fh)
    return decide(payload["state"], payload.get("context"))


# ---------------------------------------------------------------------------
# Envelope contract
# ---------------------------------------------------------------------------

class TestEnvelope:
    def test_all_kinds_return_three_keys(self):
        for kind in ("team_review", "critique", "interviewee_review", "synthesis", "status"):
            r = decide({"kind": kind})
            assert set(r) == {"output", "rationale", "self_metric"}
            assert isinstance(r["self_metric"]["confidence"], float)
            assert "decision_path" in r["self_metric"]

    def test_unknown_kind_fail_open(self):
        r = decide({"kind": "nope"})
        assert r["output"] == {}
        assert r["self_metric"]["confidence"] == 0.0
        assert r["self_metric"]["decision_path"] == "unknown_kind"

    def test_missing_kind_fail_open(self):
        r = decide({})
        assert r["self_metric"]["decision_path"] == "unknown_kind"

    def test_non_dict_state_fail_open(self):
        r = decide("not a dict")
        assert r["output"] == {}
        assert r["self_metric"]["decision_path"] == "error_fallback"

    def test_none_state_fail_open(self):
        r = decide(None)
        assert r["self_metric"]["confidence"] == 0.0


# ---------------------------------------------------------------------------
# team_review
# ---------------------------------------------------------------------------

class TestTeamReview:
    def test_json_block_extraction(self):
        body = '{"perspective": "CTO lens", "overall_strength": 7, "concerns": ["scope"]}'
        state = {
            "kind": "team_review",
            "persona_name": "Matt",
            "session_result": {"summary": f"Here is my review:\n```json\n{body}\n```\nDone."},
        }
        r = decide(state)
        out = r["output"]
        assert out["persona_name"] == "Matt"
        assert out["perspective"] == "CTO lens"
        assert out["overall_strength"] == 7
        assert out["concerns"] == ["scope"]
        assert out["strong_points"] == []
        assert out["recommended_changes"] == []
        assert out["ai_generated"] is True
        assert r["self_metric"]["parse_ok"] is True
        assert r["self_metric"]["decision_path"] == "team_review:json_block"

    def test_bare_braces_extraction(self):
        state = {
            "kind": "team_review",
            "persona_name": "Sam",
            "session_result": {"summary": 'Prefix {"overall_strength": 5} suffix'},
        }
        r = decide(state)
        assert r["output"]["overall_strength"] == 5
        assert r["self_metric"]["decision_path"] == "team_review:bare_braces"

    def test_findings_dict_preferred_over_summary(self):
        state = {
            "kind": "team_review",
            "persona_name": "Tim",
            "session_result": {
                "summary": '{"perspective": "from summary"}',
                "findings": [{"perspective": "from findings", "overall_strength": 9}],
            },
        }
        r = decide(state)
        assert r["output"]["perspective"] == "from findings"
        assert r["output"]["overall_strength"] == 9

    def test_unparseable_text_fail_open_defaults(self):
        state = {
            "kind": "team_review",
            "persona_name": "Priya",
            "session_result": {"summary": "no json here at all"},
        }
        r = decide(state)
        out = r["output"]
        assert out["persona_name"] == "Priya"
        assert out["perspective"] == "Priya's perspective"
        assert out["overall_strength"] is None
        assert out["strong_points"] == []
        assert out["ai_generated"] is True
        assert r["self_metric"]["parse_ok"] is False
        assert r["self_metric"]["confidence"] == 0.5

    def test_missing_persona_name_defaults(self):
        r = decide({"kind": "team_review", "session_result": {"summary": "{}"}})
        assert r["output"]["persona_name"] == "Unknown persona"

    def test_overall_strength_string_coerced_to_none(self):
        state = {
            "kind": "team_review",
            "persona_name": "Jordan",
            "session_result": {"summary": '{"overall_strength": "high"}'},
        }
        assert decide(state)["output"]["overall_strength"] is None

    def test_empty_session_result(self):
        r = decide({"kind": "team_review", "persona_name": "Matt", "session_result": {}})
        assert r["output"]["ai_generated"] is True
        assert r["self_metric"]["parse_ok"] is False

    def test_non_list_lists_replaced(self):
        state = {
            "kind": "team_review",
            "persona_name": "Matt",
            "session_result": {"summary": '{"concerns": "a single string"}'},
        }
        assert decide(state)["output"]["concerns"] == []

    def test_overall_strength_float_passes(self):
        state = {
            "kind": "team_review",
            "persona_name": "Matt",
            "session_result": {"summary": '{"overall_strength": 6.5}'},
        }
        assert decide(state)["output"]["overall_strength"] == 6.5


# ---------------------------------------------------------------------------
# critique
# ---------------------------------------------------------------------------

class TestCritique:
    def test_role_resolved_from_roster(self):
        r = decide({"kind": "critique", "persona_name": "Jordan", "parsed": {}})
        assert r["output"]["role"] == "Finance Director"
        assert r["output"]["persona_name"] == "Jordan"

    def test_unknown_persona_defaults_to_advisor(self):
        r = decide({"kind": "critique", "persona_name": "Zoe", "parsed": {}})
        assert r["output"]["role"] == "Advisor"

    def test_lists_normalized(self):
        parsed = {"strong_points": ["a"], "concerns": None, "critical_changes": "x"}
        r = decide({"kind": "critique", "persona_name": "Tim", "parsed": parsed})
        assert r["output"]["strong_points"] == ["a"]
        assert r["output"]["concerns"] == []
        assert r["output"]["critical_changes"] == []

    def test_parsed_none_fail_open(self):
        r = decide({"kind": "critique", "persona_name": "Priya", "parsed": None})
        out = r["output"]
        assert out["role"] == "Head of Compliance"
        assert out["strong_points"] == []
        assert out["concerns"] == []
        assert out["critical_changes"] == []
        assert r["self_metric"]["confidence"] == 0.0
        assert r["self_metric"]["decision_path"] == "critique:fail_open"

    def test_error_passthrough_on_fail_open(self):
        r = decide({"kind": "critique", "persona_name": "Tim", "parsed": None, "error": "429"})
        assert r["output"]["error"] == "429"

    def test_role_override_via_state(self):
        r = decide({
            "kind": "critique",
            "persona_name": "Morgan",
            "persona_roles": {"Morgan": "Researcher"},
            "parsed": {},
        })
        assert r["output"]["role"] == "Researcher"

    def test_role_override_via_context(self):
        r = decide(
            {"kind": "critique", "persona_name": "Ada", "parsed": {}},
            {"persona_roles": {"Ada": "Orchestrator"}},
        )
        assert r["output"]["role"] == "Orchestrator"

    def test_state_roles_win_over_context(self):
        r = decide(
            {"kind": "critique", "persona_name": "X", "persona_roles": {"X": "from_state"}, "parsed": {}},
            {"persona_roles": {"X": "from_context"}},
        )
        assert r["output"]["role"] == "from_state"

    def test_existing_role_not_overwritten(self):
        r = decide({"kind": "critique", "persona_name": "Tim", "parsed": {"role": "Custom"}})
        assert r["output"]["role"] == "Custom"


# ---------------------------------------------------------------------------
# interviewee_review
# ---------------------------------------------------------------------------

class TestIntervieweeReview:
    def test_always_ai_generated(self):
        r = decide({"kind": "interviewee_review", "person_name": "Dana", "role": "Ops", "parsed": {}})
        assert r["output"]["ai_generated"] is True

    def test_lists_normalized(self):
        parsed = {"agrees": ["x"], "disagrees": "no", "missing": None, "quotes": ["q"]}
        r = decide({"kind": "interviewee_review", "person_name": "Dana", "parsed": parsed})
        assert r["output"]["agrees"] == ["x"]
        assert r["output"]["disagrees"] == []
        assert r["output"]["missing"] == []
        assert r["output"]["quotes"] == ["q"]

    def test_name_role_defaults(self):
        r = decide({"kind": "interviewee_review", "parsed": {}})
        assert r["output"]["person_name"] == "Unknown participant"
        assert r["output"]["role"] == "Unknown role"

    def test_parsed_none_fail_open(self):
        r = decide({"kind": "interviewee_review", "person_name": "Dana", "role": "Ops", "parsed": None})
        out = r["output"]
        assert out["person_name"] == "Dana"
        assert out["role"] == "Ops"
        assert out["agrees"] == [] and out["quotes"] == []
        assert out["ai_generated"] is True
        assert r["self_metric"]["confidence"] == 0.0

    def test_supplied_name_not_overwritten(self):
        r = decide({
            "kind": "interviewee_review",
            "person_name": "Dana",
            "parsed": {"person_name": "Dana Smith"},
        })
        assert r["output"]["person_name"] == "Dana Smith"


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------

class TestSynthesis:
    def test_change_list_source_attribution_defaults(self):
        parsed = {"change_list": [{"action": "tighten ROI"}]}
        r = decide({"kind": "synthesis", "parsed": parsed})
        item = r["output"]["change_list"][0]
        assert item["source_type"] == "unknown"
        assert item["source_label"] == ""
        assert r["self_metric"]["unlabeled_changes"] == 1

    def test_existing_source_attribution_preserved(self):
        parsed = {"change_list": [{"action": "x", "source_type": "dfa_team", "source_label": "[DFA expert review: Tim]"}]}
        r = decide({"kind": "synthesis", "parsed": parsed})
        item = r["output"]["change_list"][0]
        assert item["source_type"] == "dfa_team"
        assert item["source_label"] == "[DFA expert review: Tim]"
        assert r["self_metric"]["unlabeled_changes"] == 0

    def test_overall_readiness_coercion(self):
        assert decide({"kind": "synthesis", "parsed": {"overall_readiness": 80}})["output"]["overall_readiness"] == 80
        assert decide({"kind": "synthesis", "parsed": {"overall_readiness": "high"}})["output"]["overall_readiness"] is None

    def test_perspective_defaults(self):
        r = decide({"kind": "synthesis", "parsed": {}})
        assert r["output"]["dfa_team_perspectives"] == []
        assert r["output"]["interviewee_perspectives"] == []
        assert r["output"]["quote_weaving_map"] == []
        assert r["output"]["change_list"] == []

    def test_non_list_change_list_replaced(self):
        r = decide({"kind": "synthesis", "parsed": {"change_list": "oops"}})
        assert r["output"]["change_list"] == []

    def test_parsed_none_fail_open(self):
        r = decide({"kind": "synthesis", "parsed": None})
        out = r["output"]
        assert out["change_list"] == []
        assert out["overall_readiness"] is None
        assert r["self_metric"]["confidence"] == 0.0
        assert r["self_metric"]["decision_path"] == "synthesis:fail_open"

    def test_change_count_metric(self):
        parsed = {"change_list": [{"action": "a"}, {"action": "b", "source_type": "dfa_team"}]}
        r = decide({"kind": "synthesis", "parsed": parsed})
        assert r["self_metric"]["change_count"] == 2
        assert r["self_metric"]["unlabeled_changes"] == 1

    def test_non_dict_change_item_skipped(self):
        parsed = {"change_list": ["bad", {"action": "ok"}]}
        r = decide({"kind": "synthesis", "parsed": parsed})
        # The dict item still gets defaulted; the string is left as-is.
        assert r["output"]["change_list"][1]["source_type"] == "unknown"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_not_found(self):
        r = decide({"kind": "status", "found": False})
        assert r["output"]["overall"] == "not_found"

    def test_completed(self):
        r = decide({"kind": "status", "synth_status": "completed", "child_statuses": ["completed"]})
        assert r["output"]["overall"] == "complete"

    def test_failed(self):
        r = decide({"kind": "status", "synth_status": "failed"})
        assert r["output"]["overall"] == "failed"

    def test_running_when_synth_running(self):
        r = decide({"kind": "status", "synth_status": "running", "child_statuses": ["completed"]})
        assert r["output"]["overall"] == "running"

    def test_running_when_children_pending(self):
        r = decide({"kind": "status", "synth_status": "pending", "child_statuses": ["completed", "running"]})
        assert r["output"]["overall"] == "running"
        assert r["output"]["children_pending"] is True

    def test_pending_default(self):
        r = decide({"kind": "status", "synth_status": "pending", "child_statuses": ["completed"]})
        assert r["output"]["overall"] == "pending"

    def test_missing_children_padded_to_pending(self):
        # 2 expected, only 1 reported completed → 1 padded pending → running.
        r = decide({
            "kind": "status",
            "synth_status": "pending",
            "child_statuses": ["completed"],
            "expected_child_count": 2,
        })
        assert r["output"]["overall"] == "running"

    def test_no_children_pending_path(self):
        r = decide({"kind": "status", "synth_status": "pending", "child_statuses": []})
        assert r["output"]["overall"] == "pending"
        assert r["output"]["children_pending"] is False

    def test_completed_wins_over_pending_children(self):
        r = decide({"kind": "status", "synth_status": "completed", "child_statuses": ["pending"]})
        assert r["output"]["overall"] == "complete"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_roster_matches_roles(self):
        for name in DEFAULT_TEAM_PERSONAS:
            assert name in PERSONA_ROLES


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_calls_identical(self):
        state = {
            "kind": "synthesis",
            "parsed": {"change_list": [{"action": "x"}], "overall_readiness": 50},
        }
        a = json.dumps(decide(json.loads(json.dumps(state))), sort_keys=True)
        b = json.dumps(decide(json.loads(json.dumps(state))), sort_keys=True)
        assert a == b


# ---------------------------------------------------------------------------
# Committed-sample conformance (verdict pins, not just shape)
#
# The conformance Action only SHADOW-PRINTS each sample's output to the job
# summary — a verdict flip on a sample file would slip through CI green. These
# tests ASSERT the verdict each committed sample must produce, so any change
# that alters a sample's decision is caught. The drift guard keeps the pin set
# and the on-disk sample set in lockstep: add/remove a sample without updating
# the pins and CI goes red.
# ---------------------------------------------------------------------------

# filename -> the verdict fields that sample must produce.
_SAMPLE_EXPECTATIONS = {
    "critique_normalize.json": {
        "decision_path": "critique:normalized",
        "confidence": 1.0,
        "output": {
            "persona_name": "Jordan",
            "role": "Finance Director",  # resolved from PERSONA_ROLES
            "strong_points": ["ROI framed against current spend"],
            "concerns": [],  # string in input coerced to empty list
            "critical_changes": ["quantify year-1 savings"],
        },
    },
    "interviewee_failopen.json": {
        "decision_path": "interviewee_review:fail_open",
        "confidence": 0.0,
        "output": {
            "person_name": "Dana Okoro",
            "role": "Operations Manager",
            "agrees": [],
            "disagrees": [],
            "missing": [],
            "quotes": [],
            "ai_generated": True,
            "error": "model returned malformed JSON",
        },
    },
    "status_running.json": {
        # synth pending, 2 completed children padded to 5 with pending →
        # children_pending=True → overall running via the children path.
        "decision_path": "status:children_pending",
        "confidence": 1.0,
        "output": {"overall": "running", "children_pending": True},
    },
    "synthesis_attribution.json": {
        "decision_path": "synthesis:normalized",
        "confidence": 1.0,
        "self_metric_extra": {"change_count": 2, "unlabeled_changes": 1},
    },
    "team_review_json_block.json": {
        # JSON fenced in the summary, no findings → json_block extraction.
        "decision_path": "team_review:json_block",
        "confidence": 1.0,
        "output": {
            "persona_name": "Matt",
            "perspective": "CTO / technical risk",
            "overall_strength": 7,
            "ai_generated": True,
        },
    },
}


class TestSamplesConform:
    @pytest.mark.parametrize("filename", sorted(_SAMPLE_EXPECTATIONS))
    def test_sample_verdict(self, filename):
        exp = _SAMPLE_EXPECTATIONS[filename]
        r = _run_sample(filename)

        assert r["self_metric"]["decision_path"] == exp["decision_path"], (
            f"{filename}: decision_path drifted"
        )
        assert r["self_metric"]["confidence"] == exp["confidence"], (
            f"{filename}: confidence drifted"
        )

        # Every pinned output field must match exactly.
        for key, want in exp.get("output", {}).items():
            assert r["output"].get(key) == want, (
                f"{filename}: output[{key!r}] = {r['output'].get(key)!r}, want {want!r}"
            )

        # Optional extra self_metric pins (e.g. synthesis counts).
        for key, want in exp.get("self_metric_extra", {}).items():
            assert r["self_metric"].get(key) == want, (
                f"{filename}: self_metric[{key!r}] = {r['self_metric'].get(key)!r}, want {want!r}"
            )

    def test_synthesis_defaults_unlabeled_change_attribution(self):
        # Second change_list item carries no source_type in the sample; the
        # organ must stamp the EU AI Act fallback attribution on it.
        r = _run_sample("synthesis_attribution.json")
        second = r["output"]["change_list"][1]
        assert second["source_type"] == "unknown"
        assert second["source_label"] == ""
        # First item's explicit attribution must be preserved untouched.
        assert r["output"]["change_list"][0]["source_type"] == "dfa_team"

    def test_every_sample_is_pinned(self):
        """Drift guard: the on-disk sample set and the pinned set must match
        exactly, so a new/removed sample can't silently go unasserted."""
        on_disk = {
            f for f in os.listdir(_SAMPLES_DIR) if f.endswith(".json")
        }
        assert on_disk == set(_SAMPLE_EXPECTATIONS), (
            "samples/ and _SAMPLE_EXPECTATIONS diverged — pin every sample's "
            f"verdict. on_disk-only={on_disk - set(_SAMPLE_EXPECTATIONS)}, "
            f"pins-only={set(_SAMPLE_EXPECTATIONS) - on_disk}"
        )


class TestPortsContract:
    """The connection-standard ports.json + check_ports.py guard.

    Binds check_ports.py into the pytest run (it also runs as a dedicated
    conformance step) so a ports/organ drift fails the suite locally too.
    """

    def test_ports_json_parses_and_has_shape(self):
        with open(os.path.join(os.path.dirname(__file__), "ports.json")) as fh:
            ports = json.load(fh)
        assert isinstance(ports["inputs"], list) and ports["inputs"]
        assert isinstance(ports["outputs"], list) and ports["outputs"]
        for port in ports["inputs"]:
            assert isinstance(port["name"], str) and port["name"]
            assert isinstance(port["type"], str) and port["type"]
            assert isinstance(port["required"], bool)
        for port in ports["outputs"]:
            assert isinstance(port["name"], str) and port["name"]
            assert isinstance(port["type"], str) and port["type"]

    def test_declared_types_in_vocabulary(self):
        here = os.path.dirname(__file__)
        with open(os.path.join(here, "ports.json")) as fh:
            ports = json.load(fh)
        with open(os.path.join(here, "types.json")) as fh:
            vocab = set(json.load(fh)["types"].keys())
        for side in ("inputs", "outputs"):
            for port in ports[side]:
                assert port["type"] in vocab, f"{port['name']}:{port['type']}"

    def test_check_ports_passes(self):
        import subprocess

        proc = subprocess.run(
            [sys.executable, "check_ports.py"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
