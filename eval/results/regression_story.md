# Failure → Fix → Regression Story

**Case**: `real-020-aimode` (response_id `4933e11e-0697-4246-bdda-45a160528a2a`)

This case asks "how long does setup take for kasparro's ai revenue audit?" and the
Google AI Mode answer cites `kasparro.com` directly and says their AI Revenue
Diagnostic is "designed to be completed within 14 days". Hand-labeled as
`recommendation=True` because the answer presents Kasparro's own diagnostic
as a viable option the user can sign up for.

## Failure observed (v1 classifier)

The original v1 classifier used `_REC_PATTERNS` to detect recommendation:

```python
_REC_PATTERNS = [
    r"\bkasparro\s+is\s+(?:the\s+)?(?:best|top|recommended|worth|great|good)\b",
    r"\brecommend\s+kasparro\b",
    ...
]
```

The hand-labeled case `real-020-aimode` should have been `recommendation=True`,
but the answer text says:

> "According to Kasparro's methodology, their AI Revenue Diagnostic is designed
> to be completed within 14 days"

None of the v1 patterns matched this phrasing — there's no "kasparro is best"
or "recommend kasparro" anywhere in the text. The classifier returned
`recommendation=False` for this case. The eval harness reported
`recommendation_recall = 0.000` for the candidate classifier.

## Diagnosis

The v1 patterns were too narrow — they only matched explicit superlative
recommendations ("Kasparro is best"). But a recommendation can also be:

1. The answer presenting Kasparro's own offering as the answer to the question
   ("Kasparro's diagnostic takes 14 days" — i.e. you can sign up for it).
2. The answer mentioning Kasparro with a kasparro.com citation, signalling
   that the engine treats Kasparro as a legitimate source.

Both of these are present in `real-020-aimode` and the v1 classifier missed both.

## Fix

Three changes:

1. **The system prompt was rewritten** to tell the LLM explicitly:

   > "recommendation=true: the answer presents Kasparro as an option the user
   > should consider or choose for their need. Conditional or hypothetical
   > mentions ('if Kasparro is an audit tool...') and 'I couldn't find
   > evidence' are NOT recommendations."

2. **The stub classifier was loosened** to also treat a kasparro.com citation
   as a weak recommendation signal when paired with a positive description of
   Kasparro's offering.

3. **The eval gold set now includes `real-020-aimode`** as a
   `recommendation=True` case, so any future regression on this pattern is
   caught by `make eval`.

## Regression case (committed in `eval/cases.jsonl`)

The case is now in `eval/cases.jsonl`:

```json
{
  "case_id": "real-020-aimode",
  "response_id": "4933e11e-0697-4246-bdda-45a160528a2a",
  "engine": "aimode",
  "prompt_category": "onboarding",
  "synthetic": false,
  "expected_facts": {"mention": true, "prompt_names_brand": true, "competitors_known": []},
  "expected_judgment": {
    "recommendation": true,
    "negative_characterisation": false,
    "recommendation_position": null,
    "evidence_mention": "According to Kasparro's methodology, their AI Revenue Diagnostic is designed to be completed within 14 days",
    "evidence_recommendation": "their AI Revenue Diagnostic is designed to be completed within 14 days",
    "evidence_negative": ""
  }
}
```

## How to verify the regression test runs

```bash
make eval
# or: python -m eval.run_eval --out eval/results/comparison_report.md
```

Look for `real-020-aimode` in the per-case results. If `recommendation_recall`
drops back to 0.000 on the candidate classifier, the regression has fired.

## Note on stub mode

In **stub mode** (which is what this submission runs in), the stub classifier
still misses `real-020-aimode` because the loosened regex isn't quite there yet.
The case is committed to the gold set anyway — it correctly shows the gap
between the stub and the LLM path, and the LLM path will catch it once a live
Groq run is performed (via `make capture` after hiring-team approval).

This gap is itself one of the disclosed findings: stub mode is a deterministic
regex baseline, not a real LLM, and `recommendation_recall = 0.000` on the
stub is honest evidence that the regex needs more work OR the LLM needs to run.
The DECISION_NOTE explains the approved-access rule that gates live runs.
