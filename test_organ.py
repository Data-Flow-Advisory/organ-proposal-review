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
import pytest

from organ import decide, PERSONA_ROLES, DEFAULT_TEAM_PERSONAS


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
