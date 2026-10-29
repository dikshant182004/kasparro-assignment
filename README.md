# Kasparro Auditor

Backend service that turns 50 prompts × 3 AI engines (150 archived answers + 2,439 citations) into transparent visibility scores, three evidence-backed diagnostics, and three prioritised remedies — with a human-review workflow that controls which version can be exported.

**Assignment**: Kasparro Fresher AI Engineer (Pack v1.0, 27 October 2026).
**Budget**: 6 active hours, 24-hour submission deadline.

---

## 📐 Architecture Diagram

```
                ┌─────────────────────────────────────────────────────────────┐
                │                  DATA PACK (gitignored)                     │
                │  prompts.jsonl (50)  responses.jsonl (150)  citations (2439)│
                └────────────────────────┬────────────────────────────────────┘
                                         │
                                         ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  1. INGESTION  (src/kasparro_auditor/ingest/)                           │
   │  JSONL → Pydantic validation → SQLite (idempotent upsert)                │
   │  - Per-line JSON parsing (one bad row never kills the batch)             │
   │  - Content hashing for reuse-vs-conflict detection                      │
   │  - URL normalization + referential integrity checks                     │
   │  - raw_json column preserves original inputs                             │
   └────────────────────────┬────────────────────────────────────────────────┘
                            │
                            ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  2. CLASSIFICATION  (src/kasparro_auditor/scoring/classify.py)          │
   │  - Facts (deterministic): mention regex, competitors, citation counts   │
   │  - Judgment (LLM): recommendation, negative, position, evidence          │
   │  - Skip-LLM-when-absent optimization (78/150 calls saved)                │
   │  - Per-record catch-all isolation (one bad row never aborts run)         │
   │  - ErrorCode enum (timeout/malformed/evidence_unsupported/...)           │
   │  - Adapters: Groq (live) | Ollama (local) | Replay (cached) | Stub     │
   └────────────────────────┬────────────────────────────────────────────────┘
                            │
                            ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  3. SCORING  (src/kasparro_auditor/scoring/score.py)                    │
   │  - v2 formula: multiplier-based negative (no free points for absent)    │
   │  - Position renormalises when unobservable                               │
   │  - Headline = mean of per-row scores (always consistent with parts)     │
   │  - All-failed runs return None + insufficient_data=True                  │
   │  - Cluster bootstrap over prompts (not responses)                        │
   │  - Per-engine + per-category slices with denominators                   │
   │  - Weight sensitivity table (3 alternative weight sets)                 │
   └────────────────────────┬────────────────────────────────────────────────┘
                            │
                            ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  4. REPORTS  (src/kasparro_auditor/reports/)                            │
   │  - State machine: draft → submitted → approved | rejected               │
   │  - Optimistic locking (lock_version) for concurrent decisions            │
   │  - content_hash at submit, verified at export (tamper detection)        │
   │  - Submit gate: 3 diagnostics + 3 remedies + evidence verification      │
   │  - Fork only from non-draft (prevents parallel drafts)                   │
   └────────────────────────┬────────────────────────────────────────────────┘
                            │
                            ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  5. API  (src/kasparro_auditor/api/)                                    │
   │  - FastAPI + typed exception hierarchy (no string matching)              │
   │  - Two-token auth (analyst + reviewer) with hmac.compare_digest         │
   │  - Default tokens refused at startup                                     │
   │  - Endpoints: /datasets, /analyses, /reports, /health, /docs             │
   │  - New: /analyses/{id}/results (paged), /analyses/{id}/score            │
   │  - Export includes: analysis_id, execution_mode, model_name,             │
   │    scoring_version, model_quality_verified, content_hash, is_latest     │
   └─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (≤ 10 minutes after prerequisites)

### Prerequisites

- **Python 3.11+** (3.12 or 3.13 recommended)
- **Git** (to clone the repo)
- **The assignment data pack** (place it locally — it's gitignored)

### Step 1: Clone the repository

```powershell
git clone https://github.com/dikshant182004/kasparro-assignment.git
cd kasparro-assignment
git checkout master
```

### Step 2: Create + activate a virtual environment

**PowerShell (Windows):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```powershell
pip install -e ".[dev]"
```

This installs FastAPI, SQLAlchemy 2.0, Pydantic v2, Groq SDK, Instructor, pytest, and all other dependencies defined in `pyproject.toml`.

### Step 4: Place the assignment data

The supplied data pack is **gitignored** (per the brief: *"Keep supplied data and private outputs out of public Git history."*). You need to place it locally:

```powershell
# Create the data_pack directory
mkdir data_pack

# Copy the assignment's data/ folder into data_pack/
# (Replace with the actual path to your extracted assignment)
Copy-Item -Recurse C:\path\to\extracted\assignment\data data_pack\

# Verify the structure
ls data_pack\data\
# Expected: prompts.jsonl  responses.jsonl  citations.jsonl  run.json  READ_ME.txt
```

**Linux/macOS:**
```bash
mkdir -p data_pack
cp -r /path/to/extracted/assignment/data data_pack/
ls data_pack/data/
# Expected: prompts.jsonl  responses.jsonl  citations.jsonl  run.json  READ_ME.txt
```

### Step 5: Configure environment variables

```powershell
Copy-Item .env.example .env
```

Edit `.env` and change these values:

```env
# Use stub mode for offline runs (no API key needed)
EXECUTION_MODE=stub

# Change these from the defaults (the app refuses to start with defaults)
ANALYST_TOKEN=real-analyst-token-change-this
REVIEWER_TOKEN=real-reviewer-token-change-this

# Database (SQLite file)
DATABASE_URL=sqlite+aiosqlite:///./kasparro.db

# Data pack location
DATA_PACK_DIR=./data_pack
```

### Step 6: Initialize the database

> **⚠️ IMPORTANT: You must re-initialize the database for every fresh pipeline run.**
> If you run the pipeline script and then start the API server (or vice versa) on the same DB, you'll get stale data. Delete `kasparro.db` and re-init before each service run.

```powershell
python -m scripts.init_db
```

**Expected**: `Database initialised at: sqlite+aiosqlite:///./kasparro.db`

### Step 7: Run the pipeline

```powershell
python -m scripts.run_analysis --out outputs/analysis_output.json
```

**Expected output**:
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

### Step 8: Start the API server

```powershell
python -m uvicorn kasparro_auditor.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs in your browser for the auto-generated OpenAPI docs.

---

## 🛠️ Running with Make (if installed)

If you have `make` installed (via Chocolatey on Windows, or natively on macOS/Linux):

```bash
make setup    # Create venv + install deps
make init     # Initialize the database
make replay   # Run the pipeline in stub mode
make test     # Run all 116 tests
make eval     # Run the evaluation harness
make run      # Start the API server
```

> **Note**: On Windows, `make` is not installed by default. Use the PowerShell commands above, or install make via `scoop install make` or `choco install make`.

---

## 📊 Execution Modes

The pipeline supports four execution modes, controlled by the `EXECUTION_MODE` env var:

| Mode | Description | API Key Needed | Use Case |
|------|-------------|----------------|----------|
| `stub` | Deterministic regex-based classifier | No | Tests, offline runs, default for submission |
| `replay` | Returns cached LLM outputs from `data_pack/replay/` | No | Reviewers re-running with cached live outputs |
| `live` | Real Groq API calls | Yes (`GROQ_API_KEY`) | Final live-model run |
| `local` | Ollama (local LLM, OpenAI-compatible) | No | Approved local-model access per brief |

### Switching to live mode (Groq)

1. Get a Groq API key from https://console.groq.com/keys
2. Edit `.env`:
   ```env
   EXECUTION_MODE=live
   GROQ_API_KEY=gsk_your_real_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   ```
3. Re-initialize the DB and run:
   ```powershell
   Remove-Item kasparro.db -Force -ErrorAction SilentlyContinue
   python -m scripts.init_db
   python -m scripts.run_analysis --mode live --out outputs/analysis_output_live.json
   ```

### Switching to local mode (Ollama)

1. Install Ollama: https://ollama.com/download
2. Pull a model:
   ```powershell
   ollama pull llama3.1:8b
   ```
3. Edit `.env`:
   ```env
   EXECUTION_MODE=local
   OLLAMA_MODEL=llama3.1:8b
   ```
4. Re-initialize the DB and run:
   ```powershell
   Remove-Item kasparro.db -Force -ErrorAction SilentlyContinue
   python -m scripts.init_db
   python -m scripts.run_analysis --mode local --out outputs/analysis_output_local.json
   ```

---

## 🧪 Running Tests

### Full test suite (116 tests)

```powershell
$env:DATABASE_URL="sqlite+aiosqlite:///:memory:"; $env:EXECUTION_MODE="stub"; python -m pytest tests/ -v
```

**Expected**: `116 passed, 1 skipped`

### Evaluation harness

```powershell
$env:DATABASE_URL="sqlite+aiosqlite:///:memory:"; $env:EXECUTION_MODE="stub"; python -m eval.run_eval --out eval/results/comparison_report.md
```

**Expected**: 17 cases loaded, citation metrics, diagnostics checker passes.

See [docs/TESTING.md](docs/TESTING.md) for the full testing guide.

---

## 📁 Project Layout

```
kasparro-auditor/
├── src/kasparro_auditor/
│   ├── config.py              # pydantic-settings; two-token auth model
│   ├── main.py                # FastAPI app + exception handlers
│   ├── pipeline.py            # end-to-end runner
│   ├── db/                    # SQLAlchemy 2.0 entities
│   ├── schemas/               # Pydantic: inputs, results, API
│   ├── ingest/                # JSONL → SQLite + URL normalization
│   ├── llm/                   # Groq + Ollama + Replay + Stub adapters
│   ├── scoring/               # classify, counts, score, competitors, facts
│   ├── reports/               # state machine + diagnostics/remedies
│   └── api/                   # FastAPI routers + auth deps + errors
├── tests/                     # pytest; 116 tests
├── eval/                      # cases.jsonl + grader + baseline + run_eval
├── scripts/                   # init_db, run_analysis, replay_capture
├── docs/                      # Architecture, Testing, Observations docs
├── data_pack/                 # supplied assignment data (gitignored)
├── DECISION_NOTE.md           # ~1-page decision note
├── TIMELOG.md                 # active-time log
├── outputs/                   # analysis output JSON (gitignored)
└── Makefile
```

---

## 🔑 Key Design Decisions

1. **Facts vs Judgment split**: mention, competitors, and citation counts are computed deterministically in code. The LLM only returns recommendation, negative, position, and evidence.
2. **Skip-LLM-when-absent**: 78 of 150 responses don't mention Kasparro → LLM is never called for them → saves cost + removes fake "evidence of absence".
3. **v2 scoring formula**: multiplier-based negative (no free 15 points for absent responses). Headline = mean of per-row scores (always consistent with parts).
4. **Optimistic locking**: `lock_version` on `ReportVersion` catches concurrent reviewer decisions.
5. **content_hash**: computed at submit, verified at export (tamper detection).
6. **Typed exceptions**: no string matching for error codes — `NotFoundError` → 404, `WrongRoleError` → 403, etc.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture walkthrough.

---

## 📝 Notes

- A bounded synchronous flow is used for the small assignment dataset (150 responses). The brief explicitly allows this.
- All secrets stay in environment variables. The `.env.example` documents every required variable.
- The supplied pack (`data_pack/`) is NOT committed to git. Reviewers must place it locally.
- **You must re-initialize the database (`python -m scripts.init_db`) for every fresh pipeline run.** If you reuse a stale DB, the pipeline will report `reused=150` and `successful=0`.

## Concurrency limits and expiry policy (future work)

The brief says *"Describe an expiry policy and concurrency limits as future work; implementing them is optional."*

### Implemented concurrency limits
- **Optimistic locking** on `ReportVersion` via the `lock_version` column.
- **One running analysis per dataset**: `POST /analyses` checks for existing running analysis.
- **Bounded LLM concurrency**: `max_concurrent_llm_calls=1` (default) for Groq free-tier.

### Future work — expiry policy (described, not implemented)
- **Approved-version expiry**: 90 days, then re-review required.
- **Submitted-but-undecided expiry**: 7 days, then auto-transition to `expired`.
- **Draft expiry**: 30 days of inactivity, then auto-archive.
- **Token rotation**: 90 days, tracked via `last_rotated_at`.
- **Decision rate-limiting**: max 5 decisions per actor per version per hour.

---

## 📚 Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — How each piece is made, what was fixed, assignment scope
- [docs/TESTING.md](docs/TESTING.md) — How to test everything (pipeline, API, live, eval)
- [docs/OBSERVATIONS.md](docs/OBSERVATIONS.md) — Stub results, live mode observations, replay mode
- [DECISION_NOTE.md](DECISION_NOTE.md) — ~1-page decision note
- [TIMELOG.md](TIMELOG.md) — Active-time log
