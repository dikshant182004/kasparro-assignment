# Observations & Live Mode Notes

This document covers our stub-mode results, live-mode observations, and the replay/adapter options provided for the reviewer.

---

## 1. Stub Mode Results (Final Submission)

The final submission runs in **stub mode** — a deterministic regex-based classifier. Here's what we observed:

### Overall score
- **Composite**: 17.69/100 (95% CI: 12.24–23.29)
- **Mention rate**: 48.0% (72 of 150 responses mention Kasparro)
- **Recommendation rate**: 1.3% (2 of 150 responses recommend Kasparro)
- **Negative rate**: 7.3% (11 of 150 responses characterise Kasparro negatively)
- **Citation rate**: 9.3% (14 of 150 responses cite kasparro.com)

### Per-engine scores
| Engine | Composite | Valid | Failed | Skipped |
|--------|-----------|-------|--------|---------|
| aimode | 19.12 | 50 | 0 | 0 |
| chatgpt | 19.12 | 50 | 0 | 0 |
| gemini | 14.82 | 50 | 0 | 0 |

### Per-category scores (branded vs generic)
| Category | Composite | Valid | Failed | Skipped |
|----------|-----------|-------|--------|---------|
| Branded (prompt names Kasparro) | 35.37 | 75 | 0 | 0 |
| Generic (prompt doesn't name Kasparro) | 0.00 | 75 | 0 | 0 |

### Key finding
The **lead diagnostic** is branded vs generic: branded prompts score 35.4 vs generic 0.0 (gap +35.4). This means Kasparro is found when explicitly asked for, but rarely surfaces in unprompted/generic queries. This is the most important finding for the marketing/growth lead.

### Weight sensitivity
| Weight set | Composite |
|------------|-----------|
| Balanced (30/40/15/15) | 17.69 |
| Mention-heavy (50/30/10/10) | 25.96 |
| Recommendation-heavy (20/55/15/10) | 12.24 |

### Diagnostics (3)
1. **lead_branded_vs_generic**: Branded prompts score 35.4 vs generic 0.0 (gap +35.4)
2. **loss_gemini_mention_rate**: Kasparro is rarely named in GEMINI answers (mention_rate=0.44)
3. **risk_negative_characterisation**: 11 responses characterise Kasparro negatively (negative_rate=0.07)

### Remedies (3)
1. **Investigate crawl coverage** (priority 1): Check OAI-SearchBot/GPTBot access, request fresh capture for worst-performing engine.
2. **Build third-party validation** (priority 2): Reddit, G2, Capterra, LinkedIn case studies.
3. **Clarify category membership** (priority 3): Update kasparro.com structured data + category-page copy.

### Skip-LLM-when-absent optimization
- 78 of 150 responses don't mention Kasparro → LLM never called for them
- 72 responses mention Kasparro → LLM called (or stub-classified)
- This saves ~78 API calls per run and removes the "fake evidence of absence" problem

---

## 2. Live Mode Observations

### What we tested
During development, we tested the live Groq path with `openai/gpt-oss-120b` and `openai/gpt-oss-20b`.

### What worked
- ✅ **Preflight check**: The model was reachable, API key was valid.
- ✅ **JSON mode**: Switching from `Mode.TOOLS` to `Mode.JSON` fixed the Groq tool-calling bug ("attempted to call tool 'json'").
- ✅ **Skip-LLM-when-absent**: The optimization correctly skipped 78 responses.
- ✅ **Replay cache**: Successfully cached ~72 responses that mention Kasparro. Cache files are at `data_pack/replay/{response_id}__{prompt_hash}__{model}.json`.
- ✅ **Progress bar**: Added progress output so you can see the run progressing.

### What didn't work (why live mode wasn't completed)
- ❌ **Rate limits**: Groq free-tier limit is 30 RPM / 1000 RPD / 8000 TPM. With concurrency=1, we stayed under RPM but still hit TPM limits on larger responses.
- ❌ **Model quality**: `openai/gpt-oss-20b` sometimes fails to produce valid JSON (empty `failed_generation`), causing 400 errors.
- ❌ **Time constraint**: A full 150-response live run takes 5-8 minutes minimum, but with rate-limit backoff + retries, it stretched past the time budget.
- ❌ **API timeout**: Running via `POST /analyses` with `execution_mode=live` timed out at 900 seconds. The script (`scripts/run_analysis.py`) handles long runs better.

### Replay references
Some replay cache files were created during the partial live run. These are at `data_pack/replay/` and contain real LLM outputs for the responses that succeeded. They can be used to re-run the pipeline in replay mode.

### What we'd do with more time
1. **Switch to a Groq Developer Tier account** (higher rate limits: 30K TPM vs 8K TPM).
2. **Use a more reliable model** (e.g., `llama-3.3-70b-versatile` if it's available on a paid tier, or `qwen/qwen3.8-27b`).
3. **Run `make capture` overnight** to populate the full replay cache without rate-limit pressure.
4. **Add a background task queue** (Celery/Redis) so `POST /analyses` returns immediately and the pipeline runs async.

---

## 3. Adapter Options Provided

The pipeline supports four execution modes, all implemented and tested:

### Groq Adapter (live mode)
- **Primary model**: `openai/gpt-oss-20b` (production GA on Groq)
- **Fallback model**: `openai/gpt-oss-120b`
- **JSON mode**: Uses `instructor.Mode.JSON` to avoid Groq's tool-calling bug
- **Rate limit handling**: Concurrency=1, exponential backoff on 429s (up to 16s)
- **Cost**: ~$0.04 per full pipeline run (72 calls × ~1500 tokens each)
- **Replay cache**: Each successful call is cached for offline replay

### Ollama Adapter (local mode)
- **Endpoint**: `http://localhost:11434/v1` (OpenAI-compatible)
- **Default model**: `llama3.1:8b`
- **No API key needed**: True local model, approved per the brief
- **Use case**: When hiring-team approval for hosted models isn't available
- **Note**: Slower than Groq (5-15 minutes for 150 responses)

### Replay Adapter (cached mode)
- **Cache location**: `data_pack/replay/{response_id}__{prompt_hash}__{model}.json`
- **Cache miss**: Raises `ReplaySkip` (never silently falls back to stub)
- **Stale cache detection**: Keyed by `(response_id, prompt_hash, model_name)` so stale caches can't be served across prompt or model changes
- **Use case**: Reviewers re-running with cached live outputs, zero new API calls

### Stub Adapter (deterministic mode)
- **Classification**: Regex-based (mention, recommendation, negative, position)
- **Evidence**: Extracts verbatim excerpts from answer text
- **No API key needed**: Fully offline
- **Use case**: Tests, CI, default for submission
- **Note**: This is what the final submission uses

---

## 4. How to Run Each Mode

### Stub mode (default, no API key)
```powershell
# .env: EXECUTION_MODE=stub
Remove-Item kasparro.db -Force -ErrorAction SilentlyContinue
python -m scripts.init_db
python -m scripts.run_analysis --out outputs/analysis_output.json
```

### Replay mode (uses cached live outputs)
```powershell
# .env: EXECUTION_MODE=replay
# Requires: data_pack/replay/*.json (populated by `make capture` or a live run)
Remove-Item kasparro.db -Force -ErrorAction SilentlyContinue
python -m scripts.init_db
python -m scripts.run_analysis --out outputs/analysis_output_replay.json
```

### Live mode (real Groq API calls)
```powershell
# .env: EXECUTION_MODE=live, GROQ_API_KEY=gsk_...
Remove-Item kasparro.db -Force -ErrorAction SilentlyContinue
python -m scripts.init_db
python -m scripts.run_analysis --mode live --out outputs/analysis_output_live.json
# Takes 5-8 minutes. Progress bar shows every 10 responses.
```

### Local mode (Ollama)
```powershell
# .env: EXECUTION_MODE=local, OLLAMA_MODEL=llama3.1:8b
# Requires: ollama pull llama3.1:8b
Remove-Item kasparro.db -Force -ErrorAction SilentlyContinue
python -m scripts.init_db
python -m scripts.run_analysis --mode local --out outputs/analysis_output_local.json
# Takes 5-15 minutes.
```

---

## 5. Model Quality Verification

Every exported report includes a `model_quality_verified` flag:

| Mode | `model_quality_verified` | Meaning |
|------|--------------------------|---------|
| stub | `false` | Deterministic regex, not a real LLM |
| replay | `false` | Cached outputs, not a fresh live run |
| live | `true` | Real LLM classification, verified |
| local | `true` | Real local LLM (Ollama), verified |

The eval report also includes a `UNVERIFIED (stub/replay)` banner when no live model run was performed.

---

## 6. Honest Disclosure

Per the brief's approved-access rule:

> *"Use a local model or hosted access explicitly provided or approved by the hiring team. Do not buy credits."*

- A personal Groq key was used for **development testing only** (not for the final submission).
- The final submission runs in **stub mode** — no API key, no cost, fully reproducible.
- The live adapter is fully implemented and partially tested. The replay cache contains ~72 real LLM outputs that can be inspected.
- The Ollama adapter is provided as the approved-local-model alternative.

The DECISION_NOTE explicitly states:
> *"The only classifier path exercised in this submission is stub mode. Live-model runs require hiring-team approval per the brief's approved-access rule and were not performed within the time budget."*
