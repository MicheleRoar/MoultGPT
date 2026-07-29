# `llm/eval/trait_extraction/` — trait extraction vs. MoultDB ground truth

Real evaluation harness for the paper's Trait Extraction Performance
subsection and the model-comparison "bonus" result.

## What it produces

1. `gold_questions.py` samples a fixed, reproducible **500-item** gold set
   from the real MoultDB annotation spreadsheet (`llm/finetuning/MoultDB
   character annotations.xlsx`), restricted to the 21 papers that already
   have a parsed `llm/finetuning/papers/<id>.tei.xml`. Three item types,
   each testing a different failure mode:
   - **255 single-field items** — one real `(paper, field)` pair with a
     real annotated gold value, one question. The corpus only supports
     ~200 genuinely distinct pairs (`min_field_coverage`/`max_gold_values`
     filtered, see below); reaching 255 reuses some real pairs with a
     DIFFERENT question phrasing (rotated from `QUESTION_TEMPLATES`, see
     each item's `phrasing_round`) rather than fabricating new facts —
     this tests whether correctness depends on how a question happens to
     be worded, not just whether the model "knows" something.
   - **45 combo items** — one question asking about TWO real fields from
     the same paper together (e.g. "what are the observed ontogenetic
     stage and the life history style?"), expecting a ` | `-separated
     compound answer. Tests multi-attribute extraction, distinct from
     "does batching multiple independent questions in one API call help"
     (that's the grouped-prompting mode below).
   - **200 negative items** — a `(paper, field)` pair where that field
     has genuinely NO annotation for that paper at all. Correct behaviour
     is abstention; a confident answer here is a hallucination by
     construction, not just "wrong" — this is what actually answers "how
     do you avoid hallucination?" with a number instead of an assertion.

   Only genuine moulting **traits** are eligible (not paper/study
   metadata like geological formation or sample size — see
   `EXCLUDED_PAPER_INFO_FIELDS`), and positive/combo fields are restricted
   to those annotated across at least `--min_field_coverage` (default 4)
   papers. A `#` in a field name (e.g. "Observed # total moult stages")
   is rendered as "number of" in the QUESTION text only — the raw field
   name is still what's used for gold lookups (see `_field_for_display`).
   Output: `gold_questions.json`.

2. `run_model_comparison.py` runs the real pipeline (domain gate +
   ontology sentence selection, `pipeline/domain_pipeline.py`) once per
   paper, then makes **one grouped API call per (paper, model)** covering
   every item sampled for that paper (single + combo + negative
   together), instead of one call per item — see "Grouped prompting"
   below. Scores each answer against gold, sends single/combo
   disagreements to an LLM judge (`--judge_model`) to split into
   `ambiguous` vs `incorrect`, and scores negative items directly as
   `abstained` (correct) or `hallucination` (no judge needed — it's
   unambiguous by construction). Also runs `keyword-only` and
   `ungated full-text` baselines so the paper can report what each
   pipeline stage actually contributes.

## Grouped (per-paper) prompting

Originally this made one API call per `(item, model)` — at 500 items ×
3 models that's 1500+ calls before baselines/judge, which is what
actually tripped a real run's free-tier rate limit (not question
complexity). Now all of a paper's items are asked in ONE call: a
numbered question list, answered as `<question number>: <answer>` lines
(index-based, not field-name-based — this also happens to be what
eliminates the `'#'`-in-fieldname YAML bug at the source, see
`scoring.py`). At the 500-item/21-paper scale this is ~21 papers × 3
models ≈ 63 main-condition calls instead of 1500+. `_call_with_retry`
(retry + exponential backoff, 429-specific) covers the residual risk of
a still-tight free-tier limit on the smaller number of larger calls;
this supersedes the old standalone `rerun_on_hard_items.py` /
`rejudge_unclassified.py` patch scripts from an earlier, non-grouped,
non-typed version of this harness — **those two scripts are no longer
compatible with the current item format (single/combo/negative,
`item_type`/`combo_split_ok` columns) and should not be run against
current output.**

## Matching improvements (scoring.py)

Beyond exact/substring matching, `values_match` now also treats these as
equivalent, based on real false-negative patterns found by hand-reviewing
actual disagreement rows from a run:
- **Numbers**: "seventeen" vs `"17"`, either direction (`_word_to_number`,
  small integers only — moulting trait counts are never large).
- **Booleans**: "True"/"yes" and "False"/"no" (models sometimes answer a
  yes/no field Python-style).
- **British/American spelling**: "moulting" vs "molting" (this whole
  domain is spelled both ways in the literature).
- **Reordered phrases**: bag-of-words overlap as a last-resort fallback
  (≥80% of gold's significant words present in the prediction), e.g.
  "cephalic dorsal suture" vs "dorsal cephalic suture" — deliberately
  conservative (requires 2+ shared content words) so it doesn't degrade
  into "any shared word counts."

None of this is invented after the fact to inflate numbers — each was
added in response to a specific real mismatch found in actual model
output, documented in the relevant docstring/test.

## How to run it for real

```bash
cd llm
# 1. (Already done once — gold_questions.json is committed. Re-run only
#    if the spreadsheet, the paper set, or the --n_positive/--n_negative/
#    --n_combo targets change.)
python eval/trait_extraction/gold_questions.py

# 2. Requires MISTRAL_API_KEY in llm/.env. Uses mistral-small/medium/
#    large-latest by default, all on Mistral's free "Experiment" tier as
#    of 2026-07-27 (see llm/config/models.py) -- re-check your account's
#    current rate limits before a large run.
python eval/trait_extraction/run_model_comparison.py \
    --models mistral-small-latest mistral-medium-latest mistral-large-latest \
    --judge_model mistral-medium-latest \
    --out_dir eval/trait_extraction/results
```

`--judge_model mistral-medium-latest` (not `large`) is intentional: in a
real run, `mistral-large-latest` was BOTH an extraction model and the
judge, so it rate-limited itself out of most of its own judging pass.
Medium had zero errors across its own extraction calls in that run.

Expect roughly 21 papers × 3 models ≈ 63 main-condition calls, + up to
21 ungated-baseline calls, + one judge call per real disagreement
(variable, typically well under 100 at this item count). The
keyword-only baseline makes no LLM calls at all.

Read the console output as it runs — it prints a `[REVIEW]` sample of
random judge verdicts (`--judge_sample_review`, default 10) so you can
sanity-check the judge before trusting the aggregate report.

## What to do with the output

`results/report.md` has separate tables for **accuracy** (single + combo
items: correct/ambiguous/incorrect/abstained) and **hallucination check**
(negative items: abstained vs hallucination), for both the main condition
and the baselines — do not average these together, they measure different
things. Its "Example disagreements" and "Example hallucinations" sections
are real sources for the paper's Text Extraction Error Analysis
subsection.

## Planned second pass: manual re-review of judge-flagged rows

Once a real run produces `results_scored.csv`, disagreement/incorrect
rows should get an independent manual read-through (not just the
automated LLM judge) — checking specifically whether some "incorrect"
verdicts are actually valid paraphrases or category-mismatches (the
model answered a real, true fact that isn't what the field asked for)
rather than genuine errors. This was already done once by hand on an
earlier 50-item run and surfaced real, useful categories (paraphrase-only
mismatches, systematic off-by-one counting errors, category-dimension
mismatches, and one likely gold-data miscoding). At the 500-item scale
this is worth doing as a structured pass over a stratified sample (not
all ~150-250 expected disagreements) and writing up as its own
supplementary table in the paper (e.g. "manual re-annotation of a sample
of judge-flagged extractions"), rather than folding it into the main
accuracy numbers — it's a distinct, smaller-n, higher-trust check on top
of the automated pipeline, not a replacement for it.
