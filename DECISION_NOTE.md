# Decision Note

**Assignment:** Kasparro Fresher AI Engineer (Pack v1.0, 27 Oct 2026).
**Active time spent:** ~6 hours (see `TIMELOG.md`).

---

## Scoring and aggregation

I chose a **4-component weighted composite** (mention 0.30, recommended 0.40, position 0.15, own_citation 0.15) scaled to 100. The weights reflect the brief's emphasis that *inclusion ≠ recommendation* — recommendation gets the largest weight because it's the actual prize. Mention is the baseline presence gate. Position captures ranking quality. Citation captures third-party validation.

**Negative is a 0.5 multiplier**, not an additive component. This was a critical fix: v1 had negative as a +15 component, which gave absent responses 15 free points. The multiplier means absent responses score 0 regardless of negative.

**Position renormalises when unobservable**: if Kasparro isn't recommended or has no explicit rank, the position weight is dropped and the remaining weights renormalise. This prevents penalising responses for not having a ranked list.

**Headline = mean of per-row scores**. Always consistent with parts. v1 had a position distortion where the headline fell outside its own CI — fixed by construction in v2.

**Cluster bootstrap 95% CI** (1000 resamples, seed=42) over the 50-prompt grid. Each prompt appears 3 times (once per engine), so I resample prompts with replacement, not responses. Returns `None` for <2 prompts.

**All-failed runs return None** + `insufficient_data=True`. Never a confident 0.0 — the brief's "failed analysis as a successful zero-score observation" forbidden pattern.

**Per-engine + per-category slices** (branded vs generic as the lead split) with denominators, failed/skipped counts, and CIs. Branded prompts score 35.4 vs generic 0.0 — the biggest finding.

**Weight sensitivity table**: 3 alternative weight sets (balanced, mention_heavy, recommendation_heavy) showing the composite under each.

## Backend choices

- **FastAPI + SQLAlchemy 2.0 (async) + SQLite + Pydantic v2.** SQLite is sufficient per the brief. SQLAlchemy 2.0 gives typed models with `Mapped[]`.
- **Layered architecture** (ingest → llm → scoring → reports → api). Each layer has a single responsibility.
- **Three-mode LLM adapter** (live/replay/stub) behind one Protocol. Every result row is labelled with `execution_mode` and `model_name`.
- **Idempotent imports**: `response_id` unique constraint + content hashing. Same content → reused; different content → conflict.
- **Bounded synchronous flow** for analysis. The brief explicitly allows this.
- **Typed exceptions**: `NotFoundError` (404), `WrongRoleError` (403), `InvalidTransitionError` (409 + code). No string matching.
- **Optimistic locking**: `lock_version` on `ReportVersion` catches concurrent reviewer decisions.
- **content_hash**: SHA-256 of canonical JSON, computed at submit, verified at export (tamper detection).

## Review and reliability choices

- **Two local tokens** (analyst + reviewer) from env. `hmac.compare_digest` for constant-time comparison. Default tokens refused at startup.
- **State machine** (`draft → submitted → approved | rejected`) enforced in `reports/state.py`.
- **Submit gate**: exactly 3 diagnostics + 3 remedies; every `evidence_response_id` verified against the run; every excerpt verified as verbatim; every `remedy.supporting_evidence` must reference an existing diagnostic id.
- **Fork only from non-draft**: prevents parallel drafts.
- **Export gate**: status==approved + content_hash matches + ReviewDecision exists. Includes `is_latest` + `superseded_by`.

## Alternatives considered

- **LangChain / LangGraph**: rejected (brief allows bounded synchronous flow).
- **PostgreSQL**: rejected (SQLite sufficient).
- **SQLModel**: rejected (Pydantic v2 regression at time of writing).
- **LLM-as-judge for eval**: rejected (brief says "an LLM judging its own prose is insufficient").
- **Mistral or Qwen**: Groq's `openai/gpt-oss-20b` is production GA. Qwen was preview-only.

## Top remedy

**Remedy #1: Investigate crawl coverage on the worst-performing engine (Gemini).**

Gemini has the lowest mention rate (0.44). The remedy is an **investigation**, not a fix — run a fresh capture on 10 prompts for Gemini after checking `OAI-SearchBot`/`GPTBot` access and compare to baseline. If mention_rate recovers, the issue was snapshot drift; if it persists, it points to crawl coverage or third-party-validation gaps.

This is the highest-priority remedy because: (a) low effort, (b) high uncertainty so any confident fix would be premature, (c) produces evidence that changes the conclusion.

## Failure → diagnosis → fix → regression

**Failure**: The v1 classifier's regex for recommendation was too narrow — it only matched "kasparro is best/top/recommended". The hand-labeled case `real-020-aimode` (which says "Kasparro's AI Revenue Diagnostic is designed to be completed within 14 days" — a recommendation by presenting Kasparro's own offering as the answer) was classified as `recommendation=False`.

**Diagnosis**: The regex missed recommendations that present Kasparro's offering without superlative language.

**Fix**: (1) Rewrote the system prompt to explicitly handle this case. (2) Loosened the stub's recommendation patterns. (3) Committed `real-020-aimode` to the eval gold set as `recommendation=True`.

**Regression**: The case is in `eval/cases.jsonl`. Future regressions are caught by `make eval`.

## Largest risk

**Stub mode doesn't establish live-model quality.** The stub classifier uses regex heuristics that miss nuanced recommendations and negative characterisations. The live model (`openai/gpt-oss-20b`) would produce different (likely better) classifications, but wasn't fully run due to rate limits + time constraints.

**Mitigations**: (1) `model_quality_verified: false` on every export. (2) `UNVERIFIED` banner on eval report. (3) Replay cache partially populated with real LLM outputs. (4) DECISION_NOTE + README document the limitation.

## Omissions

- **No Docker / public deployment** — brief says no bonus.
- **No dashboard / frontend** — brief excludes.
- **No Alembic migrations** — `Base.metadata.create_all` sufficient.
- **No OAuth** — brief says unnecessary.
- **No multi-agent orchestration** — brief allows bounded synchronous flow.
- **Concurrency limits implemented** (optimistic locking, one-running-per-dataset, bounded LLM concurrency). **Expiry policies described** in README (approved-version 90 days, submitted-undecided 7 days, draft 30 days, token rotation 90 days, decision rate-limiting).
- **Live-model runs not completed** — stub mode is the only path exercised. Replay cache partially populated. Reviewers can populate via `make capture` after obtaining an approved Groq key.
- **Eval citation cases**: 3 dedicated citation cases included (2 with kasparro.com cited, 1 without). Grader checks `has_kasparro_citation` as a deterministic Fact.

## Questions before release

1. Is the 4-component weighting (0.30/0.40/0.15/0.15) acceptable, or does Kasparro have an internal scoring formula?
2. Should the citation_rate component be split into "kasparro.com as a source" vs "third-party validation citing Kasparro"?
3. What is the expected behaviour when a reviewer rejects twice in a row — should the version be auto-archived?
4. Is there a preferred format for the exported report (PDF, JSON, Markdown)?

## AI assistance disclosure

- **Code generation**: Portions drafted with AI coding assistance (Claude/GPT). Every file reviewed, tested, and verified by the candidate.
- **LLM usage in pipeline**: Supports live (Groq `openai/gpt-oss-20b`), local (Ollama `llama3.1:8b`), replay, and stub modes. Final submission uses **stub mode** — no API key, no cost, fully reproducible.
- **Verification**: All LLM outputs verified deterministically — evidence excerpts must be verbatim substrings, every cited response_id must exist, the eval harness grades against hand-labeled cases (not LLM-judged).
- **AI was not used to grade its own outputs.** Per the brief: *"an LLM judging its own prose is insufficient."*
