# Proposal Review Organ

A pure decider for the multi-persona **proposal-review pipeline**, extracted
from discovery-engine's `app/services/proposal_review.py` (Stream 39, Stage 3).

It folds the *deterministic* parts of that pipeline — output normalization,
EU AI Act Article 13 source attribution, and review-job status derivation —
into a single `decide(state, context)` entry point dispatched by `state.kind`.

## What it does (and does not)

The original pipeline interleaves LLM calls, a cost ledger, job-queue
progress writes, and a `PendingWidgetAction` create with a layer of
deterministic normalization logic. This organ extracts **only** the
deterministic layer:

| Pure decision (in the organ)                                   | Side effect (stays outside) |
|----------------------------------------------------------------|-----------------------------|
| Extract JSON from a team-persona session result                | The Claude critique call    |
| Default missing review keys; coerce score fields               | `record_api_call` ledger    |
| Resolve a persona's role from the roster                       | `job_queue.update_progress` |
| Stamp `ai_generated` on reconstructed perspectives             | `create_action` (widget)    |
| Default `source_type` / `source_label` on every change         | DB reads of `BackgroundJob` |
| Derive the overall review status from job + child statuses     | network / Flask `g`         |

The organ never calls an LLM, the DB, or the network, and never raises on bad
input — it fails open to an empty-but-valid shape.

## Input contract

```json
{
  "state": { "kind": "<discriminator>", "...": "kind-specific fields" },
  "context": { "persona_roles": { "Morgan": "Researcher" } }
}
```

`context` is optional. Run it:

```bash
ORGAN_INPUT=samples/synthesis_attribution.json python3 organ.py
# or
echo '{"state": {"kind": "status", "synth_status": "completed"}}' | python3 organ.py
```

### Dispatched kinds

#### `team_review` — normalize a team-persona session result
Faithful to `parse_team_persona_result`. Extracts JSON from
`session_result.findings[0]` (preferred) or `session_result.summary`, via a
` ```json ` fenced block or bare `{...}` braces, then defaults
`persona_name`, `perspective`, the `strong_points` / `concerns` /
`recommended_changes` lists, coerces `overall_strength` to a number-or-null,
and stamps `ai_generated: true`.

```json
{"kind": "team_review", "persona_name": "Matt",
 "session_result": {"summary": "...```json {...} ```...", "findings": []}}
```

#### `critique` — normalize a Haiku persona critique
Faithful to `generate_persona_critique` post-parse. Resolves the persona's
role from the roster (override via `state.persona_roles` or
`context.persona_roles`; state wins), defaults the
`strong_points` / `concerns` / `critical_changes` lists. `parsed: null`
(LLM/parse failure) fails open to an empty critique.

```json
{"kind": "critique", "persona_name": "Jordan", "parsed": {"concerns": "vague payback"}}
```

#### `interviewee_review` — normalize an AI-reconstructed interviewee review
Faithful to `run_interviewee_review` post-parse. Defaults `person_name` /
`role` and the `agrees` / `disagrees` / `missing` / `quotes` lists, and
**always** stamps `ai_generated: true` — these are reconstructed
perspectives, never real statements (the Article 13 attribution depends on
it). `parsed: null` fails open.

```json
{"kind": "interviewee_review", "person_name": "Dana", "role": "Ops", "parsed": {...}}
```

#### `synthesis` — normalize a synthesis result + stamp attribution
Faithful to `synthesise_fast` post-parse. Defaults the `change_list` /
`quote_weaving_map` / `dfa_team_perspectives` / `interviewee_perspectives`
lists, coerces `overall_readiness`, and ensures **every** `change_list` item
carries `source_type` (default `"unknown"`) and `source_label` (default `""`)
for EU AI Act Article 13 attribution. `self_metric.unlabeled_changes` reports
how many items needed the default.

```json
{"kind": "synthesis", "parsed": {"change_list": [{"action": "..."}], "overall_readiness": 72}}
```

#### `status` — derive overall review status
Faithful to `get_review_status`. Given `found`, `synth_status`,
`child_statuses`, and optional `expected_child_count` (missing children pad to
`"pending"`), returns `overall` ∈
`not_found | complete | failed | running | pending`. A `completed` synth job
wins over still-pending children; otherwise pending/running children keep the
job `running`.

```json
{"kind": "status", "found": true, "synth_status": "pending",
 "child_statuses": ["completed", "completed"], "expected_child_count": 5}
```

## Output contract

```json
{
  "output": { "...": "the normalized review / status dict" },
  "rationale": "human-readable explanation of what was decided",
  "self_metric": {
    "confidence": 1.0,
    "decision_path": "synthesis:normalized",
    "parse_ok": true
  }
}
```

`confidence` is `1.0` on a clean normalization, `0.5` when team-review JSON
could not be parsed (defaults emitted), and `0.0` on a fail-open path
(`parsed: null`, unknown kind, or internal error).

## EU AI Act Article 13 note

The proposal-review pipeline blends two source classes — **AI-reconstructed
interviewee perspectives** and **DFA expert (team) reviews** — and Article 13
requires the two be clearly distinguished downstream. This organ encodes that
guarantee deterministically: `interviewee_review` and `team_review` always
stamp `ai_generated: true`, and `synthesis` guarantees every actionable change
carries a `source_type` / `source_label` so attribution can never silently go
missing.

## Ports contract (connection standard)

`ports.json` declares the organ's wiring surface per the orchestrator
connection standard (`CONNECTORS.md`): `inputs` are the top-level keys
`decide()` reads from `state`, `outputs` are the top-level keys it writes
under the result `output` dict. Each name maps to a type from the type
vocabulary in `types.json`.

This is a multi-operation organ (dispatched on `state.kind`), so a given
call exercises only a subset of these ports; the contract is the **union**
across all five kinds. `kind` is the required discriminator; every other
input is kind-specific and optional. `context` overrides (e.g.
`persona_roles`) and nested fields (`session_result.summary`, per-change
attribution keys) are *not* ports.

`check_ports.py` enforces the standard in CI: it asserts `ports.json` parses,
every declared type exists in the vocabulary, `decide()` reads exactly the
declared inputs (AST scan of `state.get(...)` literals), and the union of
outputs produced across the committed samples equals the declared output set.

> **Note:** the canonical standard
> (`Data-Flow-Advisory/orchestrator@feat/drift-gate` `CONNECTORS.md` +
> `types.json`) was unreachable when this contract was authored, so
> `types.json` vendors the JSON-primitive vocabulary locally. Type *names*
> should be reconciled with the canonical `types.json` once it is readable.

## Tests & conformance

```bash
python3 -m pytest -v      # organ + ports-contract tests
python3 check_ports.py    # connection-standard port check (also a CI step)
```

CI (`.github/workflows/conformance.yml`) verifies the ports contract, then
shadow-runs the organ on every file in `samples/` and prints each verdict +
`self_metric` to the job summary, then runs the test suite — report-only, no
action taken.

## Provenance

Extracted from `Data-Flow-Advisory/discovery-engine`
`app/services/proposal_review.py`. The organ is a faithful, side-effect-free
re-expression of that module's normalization, attribution, and status logic —
it does not change any decision the original made.
