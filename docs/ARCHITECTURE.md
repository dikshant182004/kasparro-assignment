# Architecture & Implementation Notes

This document explains how each piece of the Kasparro Auditor was built, what was fixed based on the assignment brief and reviewer feedback, and how the implementation maps to the assignment scope.

---

## 1. Ingestion Layer (`src/kasparro_auditor/ingest/`)

### What it does
Loads `prompts.jsonl` (50 rows), `responses.jsonl` (150 rows), and `citations.jsonl` (2,439 rows) into SQLite with Pydantic validation, content hashing, and referential integrity checks.

### Key design decisions
- **Per-line JSON parsing**: One malformed JSON line goes to `rejected_records` with the line number; the import continues. Never aborts the whole batch.
- **Content hashing**: SHA-256 of selected fields stored on every row. Same `response_id` + same content → reused. Same `response_id` + different content → conflict (rejected, not overwritten).
- **Idempotent upserts**: Re-importing the same pack creates a new `Dataset` row (audit trail) but never duplicates responses or citations.
- **URL normalization**: Strips tracking params (`utm_*`, `gclid`, `fbclid`), strips `www.`, sorts query params, preserves `%26` encoding. Documented in the README.
- **Referential integrity**: SQLite FKs enabled via `PRAGMA foreign_keys=ON`. Plus explicit checks: citation → response exists, citation.engine matches response.engine, citation.prompt_id matches response.prompt_id.
- **`raw_json` column**: Preserves the original JSON line on every table (brief: "Preserve original inputs").

### What was fixed (reviewer feedback)
- Fixed: One malformed JSONL line aborted the whole import → now caught per line.
- Fixed: SQLite FKs not enforced → added `PRAGMA foreign_keys=ON` event listener.
- Fixed: Duplicate handling compared IDs only → added content hash for reuse-vs-conflict.
- Fixed: Dataset counts showed 0 on re-import → now counts actual linked rows.
- Fixed: `normalize_url` bugs (urlencode, `ref`/`source` preservation, `www.` strip, root-path consistency).
- Fixed: Domain stored as-is → now normalized at ingest (1,186 of 2,439 citations had `www.`).
- Fixed: Rejection records lost context → added `line_no`, `record_id`, `rejection_type`.
- Fixed: Validation looser than contract → tightened Pydantic schemas (required fields, strict types, exact engine enum).

---

## 2. Classification Layer (`src/kasparro_auditor/scoring/classify.py` + `src/kasparro_auditor/llm/`)

### What it does
For each of the 150 responses, computes deterministic Facts (mention, competitors, citation counts) and then calls the LLM for Judgment (recommendation, negative, position, evidence).

### Key design decisions
- **Facts vs Judgment split**: `mention` is computed by regex (word-boundary `kasparro`), `competitors_known` by alias matching, `citation_count` by DB query. The LLM only returns `recommendation`, `negative_characterisation`, `recommendation_position`, `competitors_extra`, and evidence excerpts. The LLM cannot override the regex mention.
- **Skip-LLM-when-absent**: If `mention=False`, the LLM is never called. 78 of 150 responses don't mention Kasparro → 72 LLM calls instead of 150.
- **Per-record catch-all isolation**: Every exception is caught per response. One bad row never aborts the run. Every response ends as `success`, `failed`, or `skipped` with an `error_code`.
- **ErrorCode enum**: `timeout`, `provider_error`, `malformed`, `evidence_unsupported`, `unexpected`, `no_recorded_output`, `empty_input`.
- **Three-mode LLM adapter** (one Protocol, four implementations):
  - `GroqAdapter`: live Groq calls (`openai/gpt-oss-20b` primary, `openai/gpt-oss-120b` fallback). Uses Instructor with JSON mode (avoids Groq's tool-calling bug). Caches each result for replay.
  - `OllamaAdapter`: local Ollama endpoint (OpenAI-compatible). For approved local-model access.
  - `ReplayAdapter`: returns cached JSON. Cache miss → `ReplaySkip` (never silently falls back to stub).
  - `StubAdapter`: deterministic regex-based. Used in tests + as eval baseline.
- **System prompt**: Tells the LLM the answer is UNTRUSTED DATA, to ignore embedded instructions, and that Facts are computed elsewhere.
- **Prompt categorization**: Infers categories (pricing, legitimacy, comparison, onboarding, how_to, tracking, features, general) from prompt text. All 50 IDs share one slug, so the pack gives zero category info.

### What was fixed (reviewer feedback)
- Fixed: Groq models retired (`llama-3.3-70b-versatile` + `llama-3.1-8b-instant` retired 16 Aug 2026) → switched to `openai/gpt-oss-20b`.
- Fixed: `mention` and `competitors_named` came from the LLM → now computed deterministically.
- Fixed: Replay fell back silently to stub → now raises `ReplaySkip`.
- Fixed: `prompt_hash()` hashed `__qualname__` not content → now hashes `SYSTEM_PROMPT` + `build_user_prompt.__code__.co_code`.
- Fixed: Per-call model name tracking (no shared mutable state).
- Fixed: `LLMTimeout` raised for missing API key → now `LLMProviderError`.
- Fixed: LLM returned excerpt AND char offsets → now asks for excerpt only, computes offsets with `text.find()`.
- Fixed: `failed` was free-text, `skipped` never set → added `error_code` enum + `skip_reason`.
- Fixed: Groq tool-calling bug ("attempted to call tool 'json'") → switched to `instructor.Mode.JSON`.
- Fixed: `max_retries` conflict between Instructor and Groq SDK → removed from constructor.
- Fixed: Added progress bar to live classification.

---

## 3. Scoring Layer (`src/kasparro_auditor/scoring/score.py`)

### What it does
Computes a composite visibility score (0-100) with per-row components, per-engine slices, per-category slices (branded vs generic), and a weight sensitivity table.

### v2 Formula (per successful row)
```
raw = w_mention*mention + w_recommended*recommended + w_position*pos + w_citation*cite
row_score = 100 × (raw / active_weight) × (0.5 if negative else 1.0)
```

Where `active_weight` drops the position component when unobservable and renormalises the rest.

Default weights: mention=0.30, recommended=0.40, position=0.15, own_citation=0.15.

### Key design decisions
- **No free points for absence**: If mention=0 and recommendation=0 and citation=0, score=0 (regardless of negative_penalty). v1 gave 15 free points.
- **Negative as multiplier, not component**: `0.5` multiplier on the row score. v1 had it as a +15 additive component.
- **Headline = mean of per-row scores**: Always consistent with parts. v1 had a position distortion bug where the headline fell outside its own CI.
- **All-failed runs return None**: `composite=None` + `insufficient_data=True`. Never a confident 0.0.
- **Cluster bootstrap over prompts**: Each prompt appears 3 times (once per engine). Resamples prompts with replacement, not responses. Returns `None` for <2 prompts.
- **Per-engine + per-category slices**: With denominators, failed/skipped counts, and CIs.
- **Weight sensitivity table**: 3 alternative weight sets (balanced, mention_heavy, recommendation_heavy).
- **`scoring_version` + `category_version`**: Tracked on `AnalysisRun`, `ResponseResult`, and `VisibilityScore`.

### What was fixed (reviewer feedback)
- Fixed: Absent responses got free 15 points from `negative_penalty` → v2 multiplier-based formula.
- Fixed: Headline ≠ mean of row scores (position distortion) → headline = mean by construction.
- Fixed: All-failed runs returned confident 0.0 → now `None` + `insufficient_data=True`.
- Fixed: Unknown position treated as 0 → now renormalises (drops position weight).
- Fixed: CI assumed independent responses → now cluster bootstrap over prompts.
- Fixed: No per-engine or per-category scores → added slices with denominators.
- Fixed: No weight sensitivity → added 3 alternative weight sets.
- Fixed: No `scoring_version` → added to run + score + API output.

---

## 4. Reports Layer (`src/kasparro_auditor/reports/`)

### What it does
Manages the report version state machine: `draft → submitted → approved | rejected`. Includes submit gate, optimistic locking, content hash, and fork.

### Key design decisions
- **Typed exceptions**: `NotFoundError` (404), `WrongRoleError` (403), `InvalidTransitionError` (409 + code), `InsufficientDataError` (409), `UnsupportedClaimsError` (422), etc. No string matching.
- **Optimistic locking**: `lock_version` column on `ReportVersion`. Every transition checks + increments. Mismatch → 409 `concurrent_modification`.
- **content_hash**: SHA-256 of canonical JSON of (title, findings, diagnostics, remedies, notes). Computed at submit, verified at export. Tamper detection.
- **Submit gate**: Exactly 3 diagnostics + 3 remedies. Every `evidence_response_id` must exist in the run. Every excerpt must be verbatim in the cited response's `answer_text`. Every `remedy.supporting_evidence` must reference an existing diagnostic id.
- **Fork only from non-draft**: Prevents parallel drafts. Returns 409 `not_draft` if source is DRAFT.
- **Export metadata**: `analysis_id`, `execution_mode`, `model_name`, `scoring_version`, `model_quality_verified` (False for stub/replay), `content_hash`, `is_latest`, `superseded_by`.

### What was fixed (reviewer feedback)
- Fixed: Vacuous report could be approved/exported → added empty dataset gate (422) + insufficient data gate (409).
- Fixed: Report content unprotected (PATCH scores) → `ReportPatch` restricted to title/diagnostics/remedies/notes only.
- Fixed: Submit validated nothing → added 3+3 gate + evidence verification + remedy→diagnostic linking.
- Fixed: Wrong error codes (string matching) → typed exception hierarchy.
- Fixed: Comment optional → now `min_length=1`.
- Fixed: Fork didn't check status → now refuses DRAFT sources.
- Fixed: No concurrency protection → added `lock_version` optimistic locking.
- Fixed: Default tokens worked if env unset → `validate_startup()` refuses defaults.
- Fixed: Token comparison not constant-time → `hmac.compare_digest`.

---

## 5. API Layer (`src/kasparro_auditor/api/`)

### What it does
FastAPI routers for datasets, analyses, reports, health, and the new paged results + score endpoints.

### Key design decisions
- **Two-token auth**: `ANALYST_TOKEN` + `REVIEWER_TOKEN` from env. `hmac.compare_digest` for constant-time comparison. Default tokens refused at startup.
- **`require_analyst` on `/datasets` and `/analyses`**: Reviewer cannot import or analyse.
- **`require_reviewer` on `/decision`**: Analyst cannot approve/reject.
- **Empty dataset gate**: `POST /analyses` on a dataset with 0 responses → 422 `empty_dataset`.
- **One running analysis per dataset**: 409 `duplicate_request` if a run is already in progress.
- **Preflight model check**: In live mode, makes one trivial call before committing to 150 calls. Fails loudly with 503 `model_unavailable`.
- **New endpoints**: `GET /analyses/{id}/results` (paged), `GET /analyses/{id}/score`.
- **`/health`**: Runs `SELECT 1`, doesn't leak `database_url`.

### What was fixed (reviewer feedback)
- Fixed: Reviewer could import/analyse → now `require_analyst`.
- Fixed: Bad input caused 500s → dropped `source_dir`, `execution_mode` as `Literal`.
- Fixed: Export lacked identity/traceability → added 8 metadata fields.
- Fixed: No per-response results endpoint → added `GET /analyses/{id}/results`.
- Fixed: No request idempotency → added `Idempotency-Key` header + one-running-per-dataset.
- Fixed: `/health` leaked `database_url` → removed, runs `SELECT 1`.
- Fixed: Default tokens hardcoded → `validate_startup()` refuses defaults.
- Fixed: `on_event` deprecated → replaced with `lifespan` context manager.

---

## 6. Evaluation Harness (`eval/`)

### What it does
Hand-labeled gold set of 17 cases (13 real + 2 synthetic + 2 tuning), graded with per-label precision/recall, competitor set-overlap, citation checks, and a diagnostics checker.

### Key design decisions
- **Real gold set**: 10 real responses from the pack across all 3 engines (aimode, chatgpt, gemini) and 4 categories (legitimacy, comparison, onboarding, how_to). Plus 3 dedicated citation cases.
- **Independent RuleBaseline**: No imports from `stub_adapter`. Plain keyword matching. Meaningful comparison.
- **Real answer text**: Real cases load actual `answer_text` from the pack. Raises `FileNotFoundError` if pack missing.
- **Mode banner**: Reports `UNVERIFIED (stub/replay)` when no live model run.
- **Zero-denominator handling**: `precision/recall` return `None` (printed as `n/a`) instead of misleading `0.000`.
- **Competitor set-overlap**: Precision/recall over the set of named competitors.
- **Diagnostics checker**: Plants a wrong-evidence case and verifies it's caught.
- **Synthetic cases flagged**: `synthetic: true`, reported separately, never mixed into real-case metrics.
- **Tuning cases separate**: `is_tuning: true`, reported as informational only.

### What was fixed (reviewer feedback)
- Fixed: 11 of 12 cases were synthetic → replaced with 10 real cases from the pack.
- Fixed: RuleBaseline imported stub's regexes → rewritten as independent keyword matcher.
- Fixed: Diagnostics never evaluated → added `check_diagnostics_evidence()` + planted wrong-evidence case.
- Fixed: Counts unreliable (unsupported-claim as skip, expected-failure as success) → fixed counting.
- Fixed: No results committed → `eval/results/` no longer gitignored.
- Fixed: Failure→fix story on synthetic case → backed by `real-020-aimade` from the pack.
- Fixed: No citation cases → added 3 dedicated citation cases.
- Fixed: No remedy→diagnostic linking → added cross-field validator.

---

## 7. Tests (`tests/`)

### What it does
116 tests covering ingestion, scoring, classification, API, failure scenarios, and the submit gate.

### Test coverage
- `test_ingest.py` (28 tests): URL normalization, content hashing, referential integrity, idempotent imports, conflict detection.
- `test_facts.py` (24 tests): Mention regex, competitor aliasing, prompt categorization, source block stripping.
- `test_stub_adapter.py` (7 tests): Evidence verification, Judgment schema, cross-field validators.
- `test_scoring.py` (18 tests): Absent=0, headline=mean, position renormalisation, all-failed=None, per-engine sums, weight sensitivity, version tracking.
- `test_api_fixes.py` (17 tests): Reviewer blocked, empty dataset, PATCH scores ignored, unsupported claims, comment required, export metadata, concurrent decision, fork draft forbidden, health doesn't leak, exact error codes, results endpoint, score endpoint, default tokens refused.
- `test_failure_scenarios.py` (9 tests): Mapped 1:1 to `fixtures/failure_scenarios.json` — transient timeout, permanent timeout, empty, malformed, duplicate request, wrong role, rejected review, review no response, change after approval.
- `test_submit_gate.py` (2 tests): Remedy citing non-existent diagnostic rejected, valid diagnostic ids pass.

### What was fixed (reviewer feedback)
- Fixed: Tests skipped when data_pack missing → added inline fixtures in `tests/fixtures.py`.
- Fixed: Tests accepted either error code → now assert exact codes.
- Fixed: No tests for empty dataset, scores read-only, unsupported claims, comment required, reviewer blocked, export metadata → added `test_api_fixes.py`.

---

## 8. Live Model Observations

During development, the live Groq path was tested with `openai/gpt-oss-120b` and `openai/gpt-oss-20b`. Key observations:

1. **Groq retired models**: `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` were retired on 16 Aug 2026. Switched to `openai/gpt-oss-20b` (primary) + `openai/gpt-oss-120b` (fallback).
2. **Tool-calling bug**: The `openai/gpt-oss` models have a bug where they try to call a non-existent tool named `json`. Fixed by switching Instructor from `Mode.TOOLS` to `Mode.JSON`.
3. **Rate limits**: Free-tier limit is 30 RPM / 1000 RPD / 8000 TPM. With concurrency=1, we stay under the limit.
4. **Model quality**: `openai/gpt-oss-20b` sometimes fails to produce valid JSON (empty `failed_generation`). The retry loop handles this, but some responses still fail.
5. **Replay cache**: Successfully cached ~72 of 150 responses (those that mention Kasparro). The cache files are at `data_pack/replay/{response_id}__{prompt_hash}__{model}.json`.

**For the final submission, stub mode was used** because:
- The brief's approved-access rule requires hiring-team approval for hosted models.
- The free-tier rate limits + model quality issues made a full 150-response live run impractical within the time budget.
- The stub mode is fully functional and demonstrates all mechanics, failure paths, and the evaluation harness.

See [docs/OBSERVATIONS.md](docs/OBSERVATIONS.md) for the full observations.
