# Kasparro Auditor

Backend service that turns 50 prompts × 3 AI engines (150 archived answers + 2,439 citations) into transparent visibility scores, three evidence-backed diagnostics, and three prioritised remedies — with a human-review workflow that controls which version can be exported.

**Assignment**: Kasparro Fresher AI Engineer (Pack v1.0, 27 October 2026).
**Budget**: 6 active hours, 24-hour submission deadline.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Brief preference; pydantic-settings requires it |
| Web framework | FastAPI 0.120+ | Auto OpenAPI docs, Pydantic-native validation, async |
| ORM/DB | SQLAlchemy 2.0 (async) + SQLite | Sufficient per brief; SQLAlchemy 2.0 gives typed models |
| Validation | Pydantic v2 | Same models at API boundary + LLM structured output |
| Settings | pydantic-settings 2.x | Required `.env.example` comes for free |
| LLM provider | **Groq** | Production GA, $0.24 cost for the full 150-call run |
| Primary model | `llama-3.3-70b-versatile` | Best reasoning among Groq production models |
| Fallback model | `llama-3.1-8b-instant` | Cheaper/faster; used if primary fails |
| Structured outputs | Instructor 1.x | Pydantic-validated responses with auto-retry |
| Tests | pytest + httpx AsyncClient | FastAPI-native async testing |
| Lint | ruff + mypy | Modern, fast |

---

## Quick start (≤ 10 minutes after prerequisites)

```bash
# 1. Clone / extract the project
cd kasparro-auditor

# 2. Create venv + install deps (Python 3.11+ required)
make setup
source .venv/bin/activate

# 3. Copy env template (no edits needed for stub/replay mode)
cp .env.example .env

# 4. Run the offline input pack checker (verifies data integrity)
python3 data_pack/tools/check_pack.py    # if data_pack is the assignment dir

# 5. Initialize the SQLite database
make init

# 6. Run the full pipeline in replay mode (no API key needed)
make replay
#   → writes outputs/analysis_output.json

# 7. (Optional) Run the API server
make run
#   → http://127.0.0.1:8000/docs

# 8. Run the test suite
make test

# 9. Run the evaluation harness
make eval
#   → writes eval/results/comparison_report.md
```

---

## Project layout

```
kasparro-auditor/
├── src/kasparro_auditor/
│   ├── config.py              # pydantic-settings; two-token auth model
│   ├── main.py                # FastAPI app + exception handlers
│   ├── pipeline.py            # end-to-end runner
│   ├── db/                    # SQLAlchemy 2.0 entities
│   ├── schemas/               # Pydantic: inputs, results, API
│   ├── ingest/                # JSONL → SQLite + URL normalization
│   ├── llm/                   # Groq + Replay + Stub adapters (one Protocol)
│   ├── scoring/               # classify, counts, score, competitors
│   ├── reports/               # state machine + diagnostics/remedies
│   └── api/                   # FastAPI routers + auth deps + errors
├── tests/                     # pytest; failure_scenarios.json mapped 1:1
├── eval/                      # cases.jsonl + grader + baseline + run_eval
├── scripts/                   # init_db, run_analysis, replay_capture
├── data_pack/                 # supplied assignment data (not committed)
├── fixtures/                  # failure_scenarios.json
├── DECISION_NOTE.md           # ~1-page decision note (required)
├── TIMELOG.md                 # active-time log (required)
└── Makefile
```

---

## API endpoints (exact contract from the brief)

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/health` | either | Liveness probe |
| POST | `/datasets` | analyst | Import JSONL pack (idempotent on `response_id`) |
| GET | `/datasets/{id}` | either | Inspect imported dataset |
| POST | `/analyses` | analyst | Run classification on a dataset |
| GET | `/analyses/{id}` | either | Inspect run status + counts |
| POST | `/analyses/{id}/reports` | analyst | Create draft from completed run |
| PATCH | `/reports/{id}` | analyst | Edit a **draft** (non-draft → 409) |
| POST | `/reports/{id}/submit` | analyst | Freeze version → submitted |
| POST | `/reports/{id}/decision` | reviewer | Approve or reject + comment |
| POST | `/reports/{id}/fork` | analyst | New draft from a submitted/approved version |
| GET | `/reports/{id}` | either | Inspect any version |
| GET | `/reports/{id}/export` | either | Export approved-only (others → 409) |
| GET | `/docs` | — | Auto-generated OpenAPI |

---

## Auth model

- Two static tokens in `.env`: `ANALYST_TOKEN`, `REVIEWER_TOKEN`.
- Identity is derived **only** from the bearer token. A caller-supplied role or actor name is **never** authorisation.
- `/submit` and `/patch` require the analyst role.
- `/decision` requires the reviewer role.
- `/export` accepts either role but only on an approved version.

---

## LLM usage

| Step | When | Cost |
|---|---|---|
| Classification (150 calls) | Per response, via `classify_response` | ~$0.24 total |
| Diagnostic synthesis (1 call) | After analysis completes | ~$0.01 |
| Remedy drafting (1 call) | After diagnostics | ~$0.01 |

**Three execution modes:**
- `live` — real Groq calls with `GROQ_API_KEY`
- `replay` — returns cached JSON for known `response_id`; falls back to stub
- `stub` — deterministic regex-based; for tests and offline eval

Every output row is labelled with its `execution_mode` and `model_name`.

---

## Citation dedup rule (disclosed)

- **Raw storage:** every citation occurrence is stored verbatim (preserves order, type, position, url).
- **Per-response uniqueness:** `DISTINCT (response_id, normalized_url)` where `normalized_url` strips the URL fragment + tracking params (`utm_*`, `gclid`, `fbclid`, `mc_cid`, `mc_eid`, `ref`, `source`, etc.) and sorts remaining query params.
- **Cross-response recurrence:** same `normalized_url` appearing in >1 response is counted as a recurrence; raw occurrences preserved.
- URL fragments and tracking parameters are preserved in the raw `url` column.

See `src/kasparro_auditor/ingest/normalize.py` for the canonical implementation.

---

## Visibility score formula

```
composite = 100 * (
    0.30 * mention_rate
  + 0.25 * recommendation_rate
  + 0.15 * (1 - negative_rate)
  + 0.15 * citation_rate
  + 0.15 * position_score
)
```

- Each rate uses `valid_responses` as denominator (excludes failed/skipped).
- `position_score`: 1.0 for rank #1, decaying by 0.2 per rank down to 0.2 floor.
- Bootstrap 95% CI over the per-response grid (1000 samples, seed=42).
- Every component is persisted per `ResponseResult` row for traceability.

A **mention is NOT a recommendation**. A **failed analysis is NOT a successful zero-score observation**.

---

## Evaluation harness

```bash
make eval
```

Outputs:
- `eval/results/comparison_report.md` — human-readable
- `eval/results/comparison_report.json` — machine-readable

Compares:
1. The candidate classifier (live Groq / replay / stub depending on `EXECUTION_MODE`)
2. A pure rule-based baseline (`RuleBaseline`)

Per-label precision/recall/F1 on:
- `mention`
- `recommendation`
- `negative_characterisation`

Plus:
- **Unsupported-claim test:** every classifier output's `evidence_excerpt` is verified to be a verbatim substring of `answer_text`. Failures are listed explicitly.
- **Deterministic score math:** verified separately in `tests/test_scoring.py` with synthetic data.
- **Tuning vs final cases:** `is_tuning=true` cases are reported separately; only `is_tuning=false` cases count toward final-check metrics.

---

## Test coverage

```bash
make test
```

The test suite is mapped 1:1 to `fixtures/failure_scenarios.json`:

| Scenario | Test |
|---|---|
| `transient_timeout` | `test_transient_timeout` |
| `permanent_timeout` | `test_permanent_timeout` |
| `empty` | `test_empty` |
| `malformed` | `test_malformed` |
| `duplicate_request` | `test_duplicate_request` |
| `wrong_role` | `test_wrong_role` |
| `rejected_review` | `test_rejected_review` |
| `review_no_response` | `test_review_no_response` |
| `change_after_approval` | `test_change_after_approval` |

Plus:
- `test_scoring.py` — deterministic score math + failed-result exclusion
- `test_ingest.py` — URL normalization + idempotent import
- `test_stub_adapter.py` — evidence verification + classification

---

## Example end-to-end flow (curl)

```bash
# 1. Import the pack
curl -X POST http://127.0.0.1:8000/datasets \
  -H "Authorization: Bearer $ANALYST_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"source_dir": "./data_pack"}'
# → {"id": "ds-...", ...}

# 2. Run analysis (stub mode for offline testing)
curl -X POST http://127.0.0.1:8000/analyses \
  -H "Authorization: Bearer $ANALYST_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "ds-...", "execution_mode": "stub"}'
# → {"id": "run-...", "status": "completed", ...}

# 3. Create a draft report
curl -X POST http://127.0.0.1:8000/analyses/run-.../reports \
  -H "Authorization: Bearer $ANALYST_TOKEN"
# → {"id": "ver-...", "status": "draft", ...}

# 4. Edit the draft
curl -X PATCH http://127.0.0.1:8000/reports/ver-... \
  -H "Authorization: Bearer $ANALYST_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Kasparro AI Visibility Report v1"}'

# 5. Submit for review
curl -X POST http://127.0.0.1:8000/reports/ver-.../submit \
  -H "Authorization: Bearer $ANALYST_TOKEN"

# 6. Reviewer approves with a comment
curl -X POST http://127.0.0.1:8000/reports/ver-.../decision \
  -H "Authorization: Bearer $REVIEWER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"decision": "approve", "comment": "Looks good"}'

# 7. Export the approved version
curl http://127.0.0.1:8000/reports/ver-.../export \
  -H "Authorization: Bearer $ANALYST_TOKEN"
```

---

## Decision note, time log

- See `DECISION_NOTE.md` for scoring rationale, alternatives, top remedy, failure/fix story, risks, omissions, and AI assistance disclosure.
- See `TIMELOG.md` for active-time tracking.

---

## Reproducibility

```bash
make setup && make init && make replay && make test && make eval
```

All outputs are reproducible in `stub` or `replay` mode with no API key. To run with the live model, set `GROQ_API_KEY` and `EXECUTION_MODE=live`.

---

## Notes

- A bounded synchronous flow is used for the small assignment dataset (150 responses, bounded concurrency). The brief explicitly allows this.
- All secrets stay in environment variables. The `.env.example` documents every required variable.
- The supplied pack (`data_pack/`) is NOT committed to git (see `.gitignore`). Reviewers must place it locally.
