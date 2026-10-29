# Active Time Log

Record every block of active work (reading, coding, debugging, testing,
writing docs). Stop at 6:00:00 and disclose unfinished work in
`DECISION_NOTE.md` under "Omissions".

| Block | Start | End | Activity | Cumulative |
|---|---|---|---|---|
| 1 | 00:00 | 00:30 | Read pack (Assignment.pdf, READ_ME.txt, schemas, fixtures) + run check_pack.py | 0:30 |
| 2 | 00:30 | 01:15 | Repo scaffold: pyproject, .env.example, Makefile, package skeleton | 1:15 |
| 3 | 01:15 | 02:00 | Pydantic input schemas + SQLAlchemy entities + URL normalization | 2:00 |
| 4 | 02:00 | 02:45 | Ingest loader + idempotent upserts + rejection handling | 2:45 |
| 5 | 02:45 | 03:30 | LLM adapter (Groq live + Replay + Stub) + evidence verification | 3:30 |
| 6 | 03:30 | 04:00 | Scoring (classification orchestration + counts + bootstrap CI) | 4:00 |
| 7 | 04:00 | 04:30 | Report state machine (draft/submit/decide/fork) + diagnostics LLM call | 4:30 |
| 8 | 04:30 | 05:00 | FastAPI routes + auth dependency + structured errors | 5:00 |
| 9 | 05:00 | 05:30 | Tests mapped 1:1 to failure_scenarios.json + scoring tests + stub tests | 5:30 |
| 10 | 05:30 | 05:45 | Eval harness (cases.jsonl + grader + baseline + run_eval) | 5:45 |
| 11 | 05:45 | 06:00 | README + DECISION_NOTE + final `make test && make eval` | 6:00 |

**Total active time:** 6:00:00 (target).

## Deliberate omissions (within the 6-hour budget)

- Multi-agent orchestration — brief explicitly optional
- Docker / public deployment — brief says no bonus
- Frontend dashboard — brief explicitly excludes
- Alembic migrations — `Base.metadata.create_all` is sufficient
- OAuth — brief says unnecessary
- Vector search — brief says no bonus
- Concurrency limits + expiry policy on review — described as future work in README

## What I'd do with more time

- Run `make capture` to populate the live-Groq replay cache (currently stub mode is the only offline path).
- Expand eval cases from 12 → 25 (more negative-characterisation coverage, more competitor-naming variations).
- Add Alembic migrations for a proper schema evolution story.
- Add an automated regression test for the failure→fix story.
