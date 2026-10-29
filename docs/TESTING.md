# Testing Guide

This document covers every test we ran during development, from unit tests to live API tests.

---

## 1. Full Test Suite (116 tests)

### Command

**PowerShell:**
```powershell
$env:DATABASE_URL="sqlite+aiosqlite:///:memory:"; $env:EXECUTION_MODE="stub"; python -m pytest tests/ -v
```

**macOS/Linux:**
```bash
DATABASE_URL="sqlite+aiosqlite:///:memory:" EXECUTION_MODE=stub python -m pytest tests/ -v
```

**Expected output:**
```
======================== 116 passed, 1 skipped, 17 warnings in 68.59s ========================
```

### What it covers

| Test file | Count | What it tests |
|-----------|-------|---------------|
| `test_ingest.py` | 28 | URL normalization, content hashing, referential integrity, idempotent imports, conflict detection |
| `test_facts.py` | 24 | Mention regex, competitor aliasing, prompt categorization, source block stripping |
| `test_stub_adapter.py` | 7 | Evidence verification, Judgment schema, cross-field validators |
| `test_scoring.py` | 18 | Absent=0, headline=mean, position renormalisation, all-failed=None, per-engine sums, weight sensitivity |
| `test_api_fixes.py` | 17 | Reviewer blocked, empty dataset, PATCH scores ignored, unsupported claims, comment required, export metadata, concurrent decision, fork draft forbidden |
| `test_failure_scenarios.py` | 9 | Mapped 1:1 to `fixtures/failure_scenarios.json` |
| `test_submit_gate.py` | 2 | Remedy citing non-existent diagnostic rejected, valid diagnostic ids pass |

---

## 2. Pipeline End-to-End Test

### Command

```powershell
# Clean slate (MUST do this before each run)
Remove-Item kasparro.db -Force -ErrorAction SilentlyContinue
python -m scripts.init_db

# Run the pipeline in stub mode
python -m scripts.run_analysis --out outputs/analysis_output.json
```

**Expected output:**
```
=== Analysis summary ===
Mode:           stub
Model:          stub-v1
Counts:         prompts=50 responses(new=150, reused=0) citations=2439 rejected=0
Results:        successful=150 failed=0 skipped=0
Score:          composite=17.69/100 (95% CI: 12.24-23.29)

Diagnostics (3):
  - [lead_branded_vs_generic] Branded prompts score 35.4 vs generic 0.0 (gap +35.4)
  - [loss_gemini_mention_rate] Kasparro is rarely named in GEMINI answers
  - [risk_negative_characterisation] 11 responses characterise Kasparro negatively

Full output written to: outputs/analysis_output.json
```

### Verifying the SQLite DB

```powershell
python -c @"
import sqlite3
conn = sqlite3.connect('kasparro.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()
print('Total results:', cur.execute('SELECT COUNT(*) FROM response_results').fetchone()[0])
print('Skipped LLM calls:', cur.execute('SELECT SUM(CASE WHEN mention = 0 AND attempts = 0 THEN 1 ELSE 0 END) FROM response_results').fetchone()[0])
print('Mention rows:', cur.execute('SELECT SUM(CASE WHEN mention = 1 THEN 1 ELSE 0 END) FROM response_results').fetchone()[0])
conn.close()
"@
```

**Expected:**
```
Total results: 150
Skipped LLM calls: 78
Mention rows: 72
```

---

## 3. Evaluation Harness

### Command

```powershell
$env:DATABASE_URL="sqlite+aiosqlite:///:memory:"; $env:EXECUTION_MODE="stub"; python -m eval.run_eval --out eval/results/comparison_report.md
```

**Expected output:**
```
Loaded 17 cases:
  - 13 real final-check cases
  - 2 synthetic final-check cases
  - 2 tuning cases
  - Engine spread (final, real): ['aimode', 'chatgpt', 'gemini']
  - Category spread (final, real): ['comparison', 'features', 'how_to', 'legitimacy', 'onboarding']

Tuning-case results (informational only):
  candidate-tuning (stub): successful=2/2
  rule-baseline-tuning: successful=2/2

=== Diagnostics evidence checker ===
  ✅ Planted wrong-evidence case correctly caught (2 problems):
     - Diagnostic 'planted_wrong_evidence' cites response_id=nonexistent-response-id-12345 that doesn't exist in the run.
     - Diagnostic 'planted_wrong_evidence' cites an excerpt that is not a verbatim substring of response_id=nonexistent-response-id-12345's answer_text.

Comparison report written to: eval/results/comparison_report.md
```

### Inspecting the eval report

```powershell
cat eval/results/comparison_report.md
```

The report includes:
- **UNVERIFIED (stub/replay) banner** at the top
- **Deterministic Facts metrics** (mention, prompt_names_brand, has_kasparro_citation)
- **LLM Judgment metrics** (recommendation, negative_characterisation)
- **Competitor set-overlap** (precision/recall/F1 + intersection/FP/FN)
- **Grader limits** section

---

## 4. API Server Tests

### Start the server

```powershell
# Make sure stub mode is set in .env
# EXECUTION_MODE=stub

# Start the server
python -m uvicorn kasparro_auditor.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs for OpenAPI docs.

### Set up test variables (in a new terminal)

```powershell
$ANALYST = "real-analyst-token-change-this"      # your .env value
$REVIEWER = "real-reviewer-token-change-this"    # your .env value
$BASE = "http://127.0.0.1:8000"
```

### Test 4.1 — Health check

```powershell
curl.exe -s $BASE/health
```
**Expected:** `{"status":"ok","version":"0.1.0","execution_mode":"stub","database_healthy":true}`

### Test 4.2 — Auth failure (no token)

```powershell
curl.exe -s -o NUL -w "%{http_code}" -X POST $BASE/datasets -H "Content-Type: application/json" -d "{}"
```
**Expected:** `401`

### Test 4.3 — Reviewer blocked from import

```powershell
curl.exe -s -X POST $BASE/datasets -H "Authorization: Bearer $REVIEWER" -H "Content-Type: application/json" -d "{}"
```
**Expected:** `{"error":{"code":"wrong_role",...}}`

### Test 4.4 — Import the pack (analyst)

```powershell
$RESP = curl.exe -s -X POST $BASE/datasets -H "Authorization: Bearer $ANALYST" -H "Content-Type: application/json" -d "{}"
$RESP
$DATASET_ID = ($RESP | ConvertFrom-Json).id
```
**Expected:** `prompt_count:50, response_count:150, citation_count:2439`

### Test 4.5 — Run analysis

```powershell
$body = @{ dataset_id = $DATASET_ID; execution_mode = "stub" } | ConvertTo-Json
$body | Out-File -FilePath body.json -Encoding utf8
$RESP = curl.exe -s -X POST $BASE/analyses -H "Authorization: Bearer $ANALYST" -H "Content-Type: application/json" --data "@body.json"
$RESP
$RUN_ID = ($RESP | ConvertFrom-Json).id
Remove-Item body.json
```
**Expected:** `status:completed, successful:150, score.composite:17.69`

### Test 4.6 — Get results (paged)

```powershell
curl.exe -s "$BASE/analyses/$RUN_ID/results?limit=5&offset=0" -H "Authorization: Bearer $ANALYST"
```
**Expected:** 5 results with all fields (mention, prompt_names_brand, evidence_mention, etc.)

### Test 4.7 — Get score

```powershell
curl.exe -s "$BASE/analyses/$RUN_ID/score" -H "Authorization: Bearer $ANALYST"
```
**Expected:** Full VisibilityScore v2 structure with per_engine, per_category, weight_sensitivity

### Test 4.8 — Create draft report

```powershell
$RESP = curl.exe -s -X POST "$BASE/analyses/$RUN_ID/reports" -H "Authorization: Bearer $ANALYST"
$VERSION_ID = ($RESP | ConvertFrom-Json).id
```
**Expected:** `status:draft, lock_version:0, content_hash:null`

### Test 4.9 — Export draft → 409 not_approved

```powershell
curl.exe -s -o NUL -w "%{http_code}" "$BASE/reports/$VERSION_ID/export" -H "Authorization: Bearer $ANALYST"
```
**Expected:** `409`

### Test 4.10 — Submit the draft

```powershell
curl.exe -s -X POST "$BASE/reports/$VERSION_ID/submit" -H "Authorization: Bearer $ANALYST"
```
**Expected:** `status:submitted, content_hash populated, lock_version:1`

### Test 4.11 — Reviewer approves with lock_version

```powershell
$body = @{ decision = "approve"; comment = "LGTM"; lock_version = 1 } | ConvertTo-Json
$body | Out-File -FilePath body.json -Encoding utf8
curl.exe -s -X POST "$BASE/reports/$VERSION_ID/decision" -H "Authorization: Bearer $REVIEWER" -H "Content-Type: application/json" --data "@body.json"
Remove-Item body.json
```
**Expected:** `decision:approve, actor_role:reviewer`

### Test 4.12 — Concurrent decision → 409 concurrent_modification

```powershell
$body = @{ decision = "reject"; comment = "second"; lock_version = 1 } | ConvertTo-Json
$body | Out-File -FilePath body.json -Encoding utf8
curl.exe -s -X POST "$BASE/reports/$VERSION_ID/decision" -H "Authorization: Bearer $REVIEWER" -H "Content-Type: application/json" --data "@body.json"
Remove-Item body.json
```
**Expected:** `409 concurrent_modification`

### Test 4.13 — Export approved version

```powershell
curl.exe -s "$BASE/reports/$VERSION_ID/export" -H "Authorization: Bearer $ANALYST"
```
**Expected:** 200 with metadata: `analysis_id, execution_mode, model_name, scoring_version, model_quality_verified:false, content_hash, is_latest:true, superseded_by:null`

### Test 4.14 — Fork edit-after-approval

```powershell
$RESP = curl.exe -s -X POST "$BASE/reports/$VERSION_ID/fork" -H "Authorization: Bearer $ANALYST"
$NEW_DRAFT_ID = ($RESP | ConvertFrom-Json).id
```
**Expected:** `status:draft, parent_version_id:$VERSION_ID, version_number:2`

### Test 4.15 — Fork a draft → 409 not_draft

```powershell
curl.exe -s -X POST "$BASE/reports/$NEW_DRAFT_ID/fork" -H "Authorization: Bearer $ANALYST"
```
**Expected:** `409 not_draft`

### Test 4.16 — Wrong role: analyst trying to decide

```powershell
$body = @{ decision = "approve"; comment = "trying as analyst"; lock_version = 1 } | ConvertTo-Json
$body | Out-File -FilePath body.json -Encoding utf8
curl.exe -s -X POST "$BASE/reports/$VERSION_ID/decision" -H "Authorization: Bearer $ANALYST" -H "Content-Type: application/json" --data "@body.json"
Remove-Item body.json
```
**Expected:** `403 wrong_role`

### Test 4.17 — Empty comment → 422

```powershell
# Create + submit a fresh draft first, then:
$body = @{ decision = "approve"; comment = ""; lock_version = 1 } | ConvertTo-Json
$body | Out-File -FilePath body.json -Encoding utf8
curl.exe -s -o NUL -w "%{http_code}" -X POST "$BASE/reports/$FRESH_VERSION_ID/decision" -H "Authorization: Bearer $REVIEWER" -H "Content-Type: application/json" --data "@body.json"
Remove-Item body.json
```
**Expected:** `422`

### Test 4.18 — Unknown ID → 404

```powershell
curl.exe -s -o NUL -w "%{http_code}" "$BASE/reports/nonexistent-id" -H "Authorization: Bearer $ANALYST"
```
**Expected:** `404`

---

## 5. Live Mode Tests (Groq)

### Prerequisites
- Groq API key from https://console.groq.com/keys
- `.env` with `GROQ_API_KEY=gsk_...` and `EXECUTION_MODE=live`

### Test 5.1 — Run the live pipeline

```powershell
Remove-Item kasparro.db -Force -ErrorAction SilentlyContinue
python -m scripts.init_db
python -m scripts.run_analysis --mode live --out outputs/analysis_output_live.json
```

**Expected:** Takes 3-8 minutes. Progress bar shows every 10 responses. Final output shows `Mode: live, Model: openai/gpt-oss-20b`.

### Test 5.2 — Capture live responses for replay

```powershell
python -m scripts.replay_capture
```

**Expected:** `Capture complete: 150/150 succeeded` (or fewer if rate-limited)

### Test 5.3 — Re-run in replay mode

```powershell
# Edit .env: EXECUTION_MODE=replay
Remove-Item kasparro.db -Force -ErrorAction SilentlyContinue
python -m scripts.init_db
python -m scripts.run_analysis --out outputs/analysis_output_replay.json
```

**Expected:** Finishes in ~5 seconds. `Mode: replay, Model: openai/gpt-oss-20b`. Uses cached live outputs, zero new API calls.

---

## 6. Local Mode Tests (Ollama)

### Prerequisites
- Ollama installed: https://ollama.com/download
- Model pulled: `ollama pull llama3.1:8b`
- `.env` with `EXECUTION_MODE=local` and `OLLAMA_MODEL=llama3.1:8b`

### Test 6.1 — Run the local pipeline

```powershell
Remove-Item kasparro.db -Force -ErrorAction SilentlyContinue
python -m scripts.init_db
python -m scripts.run_analysis --mode local --out outputs/analysis_output_local.json
```

**Expected:** Takes 5-15 minutes (Ollama is slower than Groq). `Mode: local, Model: llama3.1:8b`.

---

## 7. Test Summary

| Test Category | Count | Status |
|---------------|-------|--------|
| Unit tests (ingest, facts, stub, scoring) | 77 | ✅ All pass |
| API tests (auth, state machine, concurrency, export) | 17 | ✅ All pass |
| Failure scenarios (1:1 with fixtures) | 9 | ✅ All pass (9 skip without data_pack) |
| Submit gate tests | 2 | ✅ All pass |
| Eval harness | 17 cases | ✅ Citation metrics 1.000, diagnostics checker passes |
| Pipeline end-to-end | 1 | ✅ 150/150 successful |
| Live mode (Groq) | Partial | ⚠️ Preflight passed, ~72 calls succeeded, rate-limited |
| Local mode (Ollama) | Not run | ⚠️ Adapter implemented, not tested live |
