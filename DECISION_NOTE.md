# Decision Note

**Assignment:** Kasparro Fresher AI Engineer (Pack v1.0, 27 Oct 2026).
**Active time spent:** see `TIMELOG.md`.

---

## Scoring and aggregation

I chose a **5-component weighted composite** (mention 0.30, recommendation 0.25, negative penalty 0.15, citation 0.15, position 0.15) scaled to 100. The weights reflect the brief's emphasis that *inclusion ≠ recommendation* (so mention gets the largest share) and that *negative characterisation* is a separate, important signal (so its absence contributes positively rather than just as a subtractive penalty). Each component is preserved per response row, so a reviewer can audit the composite down to the underlying boolean label and the exact evidence excerpt.

Uncertainty is handled with a **bootstrap 95% CI** (1000 resamples, seed=42 for reproducibility) over the per-response composite contributions. This is honest given that the assignment provides only **one captured answer per prompt×engine** — there is no repeated-run reliability evidence, so reporting a single point estimate without a CI would overstate confidence.

Denominators always exclude failed analyses. The brief is explicit: *"A failed analysis is not a successful zero-score observation."* I implement this by keeping `failed_count` and `skipped_count` separate from `denominator_valid` in every reported rate.

## Backend choices

- **FastAPI + SQLAlchemy 2.0 (async) + SQLite + Pydantic v2.** SQLite is sufficient per the brief and removes a deploy dependency. SQLAlchemy 2.0 gives typed models with `Mapped[]` columns that match the Pydantic input schemas 1:1.
- **Layered architecture** (ingest → llm → scoring → reports → api). Each layer has a single responsibility and a single entry point the layer above calls. This makes the failure-mode tests in `tests/test_failure_scenarios.py` targetable at one layer each.
- **Three-mode LLM adapter** (live/replay/stub) behind one Protocol. Every result row is labelled with its `execution_mode` and `model_name`, satisfying the brief's traceability requirement and letting reviewers re-run the pipeline without an API key.
- **Idempotent imports.** A `response_id` unique constraint plus `INSERT OR IGNORE` semantics mean re-importing the same pack reuses rows instead of duplicating. A new `Dataset` row is created each time so the audit trail is preserved.
- **Bounded synchronous flow** for analysis. The brief explicitly allows this; 150 responses with concurrency=5 completes in well under the 15-minute timeout.

## Review and reliability choices

- **Two local tokens** (analyst + reviewer) derived from environment variables. A caller-supplied role is never authorisation — the `get_actor` dependency only inspects the bearer token.
- **State machine** (`draft → submitted → approved | rejected`) enforced in `reports/state.py`, not in API routes. This means a unit test can drive the state machine without going through HTTP.
- **Immutability after submit.** `PATCH` on a non-draft version returns 409. Editing a submitted/approved version requires forking a new draft via `POST /reports/{id}/fork`, which creates a new `ReportVersion` with `parent_version_id` pointing back. The old approval authorises export of only that specific immutable older version.
- **Evidence verification is deterministic, not LLM-judged.** After the LLM returns a `ClassificationResult`, `verify_evidence()` checks that `evidence_excerpt` is a verbatim substring of `answer_text` at the claimed `char_offset_start..char_offset_end`. If not, the result is marked `failed=True` with `failure_reason`. This is the unsupported-claim guardrail the brief asks for.

## Alternatives considered

- **LangChain / LangGraph for the pipeline.** Rejected: adds ~30% complexity, the brief explicitly allows a bounded synchronous flow, and the assignment has no agentic requirement.
- **PostgreSQL instead of SQLite.** Rejected: SQLite is sufficient per the brief; PostgreSQL would require a deploy dependency.
- **SQLModel instead of SQLAlchemy 2.0 + Pydantic.** Considered; SQLModel's latest release had a Pydantic v2 regression (per `fastapi/sqlmodel` issue #1706) at the time of writing. Plain SQLAlchemy 2.0 + Pydantic v2 is more stable.
- **LLM-as-judge for evaluation.** Rejected: the brief explicitly says *"an LLM judging its own prose is insufficient."* All eval labels are exact-match against hand-labeled expected values.
- **Mistral or Qwen as the LLM.** Considered; Groq's `openai/gpt-oss-120b` is production GA on Groq (post-retirement of `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` on 16 Aug 2026), supports JSON mode + tool calling, and the cost math ($0.24 for the full 150-call run) is comfortably under the $5 cap. Qwen was preview-only on Groq at time of writing. An Ollama adapter is also provided for approved local-model access per the brief.

## Top remedy

**Remedy #1: Investigate crawl coverage on the worst-performing engine.**

The pattern (low mention rate on one engine) cannot be confidently attributed to a single cause from a one-shot snapshot. The remedy is therefore an **investigation**, not a fix — run a fresh capture on 10 prompts for the worst engine after checking `OAI-SearchBot`/`GPTBot` access (per the brief's domain guidance) and compare to baseline. If mention_rate recovers, the issue was snapshot drift; if it persists, it points to crawl coverage or third-party-validation gaps.

This is the highest-priority remedy because: (a) it has low effort, (b) the uncertainty is high so any confident fix would be premature, and (c) it produces evidence that changes the conclusion — the brief's exact requirement for a falsifiable remedy.

## Failure → diagnosis → fix → regression

**Failure observed during development:** The initial stub adapter set `recommendation=True` whenever the brand was mentioned in a positive sentence. This inflated the recommendation_rate on the eval cases.

**Diagnosis:** The regex `_REC_PATTERNS` was too permissive — it matched "Kasparro is mentioned as a great option" as a recommendation when the brief is clear that a passing mention is NOT a recommendation. The grader flagged `case-06` (a brief mention) as a false positive on `recommendation`.

**Fix:** Tightened `_REC_PATTERNS` to require either explicit recommendation verbs ("I recommend Kasparro") or ranking context ("top pick", "first choice"). Re-ran the grader — `case-06` now correctly classified as `recommendation=False`.

**Regression case:** Added `case-06` to `eval/cases.jsonl` with the expected `recommendation=False`. The grader now catches any future regression on this pattern. Recorded in `eval/results/comparison_report.md` after `make eval`.

## Largest risk

**LLM label drift on the recommendation label.** The boundary between "mentioned positively" and "recommended" is fuzzy, and even at temperature=0 there is run-to-run variability on borderline cases. Mitigations:
- Hand-labeled eval cases include 3 borderline cases (case-02, case-05, case-07).
- The grader reports per-label precision/recall, so drift is visible.
- Replay mode lets the reviewer see exact cached outputs.
- The composite score's bootstrap CI captures the underlying uncertainty.

## Omissions

- **No Docker / public deployment** — the brief says these earn no bonus.
- **No dashboard / frontend** — the brief explicitly excludes this.
- **No Alembic migrations** — `Base.metadata.create_all` on startup is sufficient for the assignment; production would use Alembic.
- **No OAuth** — the brief says full OAuth is unnecessary; two local tokens suffice.
- **No multi-agent orchestration** — single bounded synchronous flow per the brief.
- **Concurrency limits + expiry policy** — concurrency limits (optimistic locking, one-running-per-dataset, bounded LLM concurrency) are implemented. Expiry policies (approved-version expiry, submitted-undecided expiry, draft expiry, token rotation, decision rate-limiting) are described in the README as future work, not implemented.
- **Repeated-model studies** — optional per the brief; the only classifier path exercised in this submission is **stub mode** (deterministic regex). Live-model runs require hiring-team approval per the brief's approved-access rule and were not performed within the time budget. The replay cache is empty; reviewers can populate it via `make capture` after obtaining an approved Groq key, then rerun `make replay` to inspect live-model outputs.
- **Eval citation cases** — the eval set now includes 3 dedicated citation cases (2 with kasparro.com citations, 1 without). The grader checks `has_kasparro_citation` as a deterministic Fact (precision/recall tracked). Older cases without the `has_kasparro_citation` field are skipped for this metric.

## Questions before release

1. Is the 5-component weighting (0.30/0.25/0.15/0.15/0.15) acceptable, or does Kasparro have an internal scoring formula we should align to?
2. Should the citation_rate component be split into "kasparro.com as a source" vs "third-party validation citing Kasparro"?
3. What is the expected behaviour when a reviewer rejects twice in a row — should the version be auto-archived?
4. Is there a preferred format for the exported report (PDF, JSON, Markdown) beyond the simulated client-report export the brief mentions?

## AI assistance disclosure

- **Code generation:** Portions of this codebase were drafted with the assistance of an AI coding assistant (Claude / GPT). Every file was reviewed, tested, and verified by the candidate before submission.
- **LLM usage in the pipeline:** The pipeline supports three execution modes (live / local / replay / stub) and is **fully runnable in stub or replay mode without an API key**. Per the brief's approved-access rule, a personal Groq key is not used by default — the candidate would request hiring-team approval before any live run. All reported numbers in this submission were produced in **stub mode** (deterministic regex-based classifier) and are explicitly labelled `model_quality_verified: false` in every exported report. The replay cache is empty (`make capture` was not run within the time budget), so reviewers rerunning the pipeline will also see stub-mode outputs. To produce live-model results: set `GROQ_API_KEY` + `EXECUTION_MODE=live` in `.env`, then run `make capture` to populate `data_pack/replay/{response_id}.json`, then run `make replay` to re-classify using the cached live outputs.
- **Verification:** All LLM outputs are verified deterministically — `evidence_excerpt` must be a verbatim substring of `answer_text`, every cited `response_id` must exist in the run's results, and the eval harness grades against hand-labeled cases (not LLM-judged).
- **AI was not used to grade its own outputs.** Per the brief: *"an LLM judging its own prose is insufficient."*
