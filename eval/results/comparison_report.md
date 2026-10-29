# Evaluation Comparison Report (v2)

> ⚠️ **UNVERIFIED (stub/replay)** — no live-model run was performed. > Per the brief, stub/replay outputs demonstrate mechanics, not > live-model correctness, variability, cost or latency. To produce > verified metrics: set GROQ_API_KEY + EXECUTION_MODE=live, run `make capture`, then re-run `make eval`.

## Architecture
- **Facts** (mention, prompt_names_brand, competitors_known) computed deterministically in code.
- **Judgment** (recommendation, negative, position, evidence) returned by the LLM.
- **LLM calls are skipped** when the deterministic mention check returns False.
- **Real cases** load actual answer_text from the pack.
- **Synthetic cases** are flagged `synthetic: true` and reported separately.
- **Tuning cases** kept separate from final-check metrics.

## Classifier: candidate (stub) (mode=stub)

- Total cases: 15
- Successful: 13
- Failed: 0
- Skipped: 2
- Unsupported-claim failures: 0

### Deterministic Facts metrics

| Fact | Precision | Recall | F1 |
|---|---|---|---|
| mention | 1.000 | 1.000 | 1.000 |
| prompt_names_brand | n/a | n/a | n/a |
| has_kasparro_citation | 1.000 | 1.000 | 1.000 |

### LLM Judgment metrics

| Label | Precision | Recall | F1 |
|---|---|---|---|
| recommendation | n/a | 0.000 | n/a |
| negative_characterisation | 0.000 | 0.000 | n/a |

### Competitor set overlap

| Metric | Value |
|---|---|
| Precision | 0.333 |
| Recall | 1.000 |
| F1 | 0.500 |
| Intersection | 3 |
| Predicted-only (FP) | 6 |
| Expected-only (FN) | 0 |

## Classifier: rule-baseline-v2 (mode=stub)

- Total cases: 15
- Successful: 13
- Failed: 0
- Skipped: 2
- Unsupported-claim failures: 0

### Deterministic Facts metrics

| Fact | Precision | Recall | F1 |
|---|---|---|---|
| mention | 1.000 | 1.000 | 1.000 |
| prompt_names_brand | n/a | n/a | n/a |
| has_kasparro_citation | 1.000 | 1.000 | 1.000 |

### LLM Judgment metrics

| Label | Precision | Recall | F1 |
|---|---|---|---|
| recommendation | n/a | 0.000 | n/a |
| negative_characterisation | 0.000 | 0.000 | n/a |

### Competitor set overlap

| Metric | Value |
|---|---|
| Precision | 0.333 |
| Recall | 1.000 |
| F1 | 0.500 |
| Intersection | 3 |
| Predicted-only (FP) | 6 |
| Expected-only (FN) | 0 |


## Grader limits

- Facts (mention, prompt_names_brand, competitors_known) are exact-match.
- LLM Judgment labels (recommendation, negative) are exact-match.
- Competitor extraction is graded as set-overlap (precision/recall).
- An LLM judging its own prose is explicitly insufficient (brief requirement).
- Evidence excerpts verified as verbatim substrings of answer_text.
- Score-component arithmetic verified separately in tests/test_scoring.py.
- Zero-denominator precision/recall reported as 'n/a' (not 0.000).
- Unsupported-claim cases counted as failed (not skipped, not success).
- Real cases load actual answer_text from the pack; missing pack → skipped.

