#!/usr/bin/env python3
# llm/eval/trait_extraction/run_model_comparison.py
"""
THE script this repo's paper (paper/moultgpt.tex, Trait Extraction
Performance / baseline-comparison placeholders) needs real numbers from.

Why this exists
----------------
The vision module has real, executed benchmark results (Section 5.1 of
the paper: YOLO architecture comparison, classifier comparison, confusion
matrix). The text/LLM module did not -- the paper's Trait Extraction
Performance subsection stated specific numbers (~33 correct traits/paper,
70%/50% accuracy) with no script behind them. This module is that script.

It CANNOT be run inside the sandboxed environment this repo's other
recent additions (llm/retrieval/, llm/finetuning/feedback_to_preferences.py,
etc.) were built and executed in -- that sandbox's outbound network is
proxied and blocks api.mistral.ai entirely (confirmed: even a bare `curl
https://api.mistral.ai/v1/models` gets a 403 from the proxy, not from
Mistral). Run it wherever you have real internet access and the API keys
in llm/.env, and its output (results_scored.csv + report.md) is what
should replace the placeholder tables in the paper -- do not write
numbers into the paper without actually running this first.

What it does
------------
1. Loads gold_questions.json (see gold_questions.py -- 500 items by
   default: 255 single-field + 45 combo (two-field) + 200 negative
   (should-abstain) items sampled from real MoultDB annotations, fixed
   BEFORE any model is called -- see gold_questions.py's module docstring
   and README.md for the full design).
2. For each of the (up to 21) papers referenced by those items: parses
   the paper's real extracted text (pipeline.parser.tei_to_text, no
   GROBID service needed -- the .tei.xml files are already in
   llm/finetuning/papers/), then runs the production domain gate +
   ontology sentence-selection ONCE per paper (pipeline.domain_pipeline.
   run_domain_pipeline) -- this step doesn't depend on which trait is
   being asked about, so it's shared across all of that paper's items
   and across every model being compared.
3. For each (paper, model): ONE grouped API call covering every item
   sampled for that paper together (see scoring.py's GROUPED_SYSTEM_PROMPT
   / _run_grouped_call below) -- each question is still asked narrowly
   and independently (not the deployed default full_traits mode's "give
   me all 55 fields" prompt, which biases toward abstaining on anything
   not front-and-center), just batched into fewer, larger calls instead
   of one call per item. This is what keeps request volume manageable at
   500 items (~21 papers x len(models) calls, not 500 x len(models)).
4. Scores each answer against the item's gold value(s) (scoring.py):
   abstained / correct / disagreement for single+combo items;
   abstained / hallucination for negative items (no "correct" state is
   possible for a negative item -- there's no real value to match).
5. For every single/combo "disagreement", asks a judge model
   (--judge_model, default mistral-medium-latest -- see below for why not
   large) to decide "ambiguous" vs "incorrect", with a short
   justification. This is LLM-as-judge, a real and accepted but
   IMPERFECT technique -- the judge is the same family of model being
   evaluated, not a domain expert. --judge_sample_review lets you print a
   random subsample of judge verdicts for a human sanity-check before
   trusting the aggregate numbers. Negative items are NOT judged --
   hallucination vs. correct abstention doesn't need a judge call, it's
   unambiguous by construction.
6. Runs two baselines for comparison, controlling for one pipeline stage
   each (both also grouped per paper):
     - keyword: no LLM at all. A field's enum-hint values (config/
       trait_schema.py) are searched as literal substrings in the
       selected evidence sentences; first match wins, else abstain.
     - ungated_fulltext: same LLM + same grouped prompt, but skips BOTH
       the domain gate and the ontology sentence-selection -- answered
       against the first --ungated_char_budget characters of the raw
       paper text instead of the selected summary. Isolates the
       contribution of evidence selection specifically (fixed
       --baseline_model, so the comparison is same-model,
       different-context).
7. Writes:
     - results_raw.jsonl  -- every single grouped API call, verbatim.
     - results_scored.csv -- one row per (item[, sub-field], model):
       item_type, label, cleaned prediction, judge verdict if applicable.
     - report.md           -- separate accuracy tables (single+combo) and
       hallucination-check tables (negative items), for main condition and
       baselines, plus real example disagreements/hallucinations.

Usage
-----
    cd llm
    python eval/trait_extraction/gold_questions.py   # if not already run
    python eval/trait_extraction/run_model_comparison.py \\
        --models mistral-small-latest mistral-medium-latest mistral-large-latest \\
        --judge_model mistral-medium-latest \\
        --out_dir eval/trait_extraction/results

`--judge_model mistral-medium-latest` (not large) is deliberate: in a
real run, mistral-large-latest was both an extraction model AND the
judge, so it rate-limited itself out of most of its own judging pass --
see README.md.

Cost: every default model above is on Mistral's free "Experiment" tier as
of 2026-07-27 (see llm/config/models.py). At the 500-item/21-paper scale,
expect roughly 21 x len(models) grouped main-condition calls, + up to
21 grouped ungated-baseline calls, + one judge call per real single/combo
disagreement -- a few hundred requests total (down from 1500+ under the
original one-call-per-item design), well inside typical free-tier daily
limits, but re-check your account's current rate limits before a run.
_call_with_retry adds retry+backoff on 429s as a safety net regardless.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

EVAL_ROOT = Path(__file__).resolve().parent
LLM_ROOT = EVAL_ROOT.parents[1]
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(LLM_ROOT / ".env")
except ImportError:
    pass

from pipeline.parser import tei_to_text  # type: ignore  # noqa: E402
from pipeline.domain_pipeline import run_domain_pipeline  # type: ignore  # noqa: E402
from domain.optimized_taxonomy_lookup import TaxonomyLookup  # type: ignore  # noqa: E402
from domain.moulting_ontology_gate import MoultingOntologyGate  # type: ignore  # noqa: E402
from backend.providers import generate as llm_generate, ProviderError  # type: ignore  # noqa: E402
from config.trait_schema import ENUM_HINTS  # type: ignore  # noqa: E402
from scoring import (  # type: ignore  # noqa: E402
    GROUPED_SYSTEM_PROMPT,
    build_grouped_user_content,
    classify_prediction,
    parse_grouped_response,
)

DEFAULT_PAPERS_DIR = LLM_ROOT / "finetuning" / "papers"
DEFAULT_GOLD_QUESTIONS = EVAL_ROOT / "gold_questions.json"
DEFAULT_MODELS = ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]
DEFAULT_JUDGE_MODEL = "mistral-medium-latest"  # not large -- see module docstring
UNGATED_CHAR_BUDGET = 6000  # ~ a few pages; matches roughly what 20 selected sentences would cost in tokens
GROUPED_TOKENS_PER_ITEM = 70   # budget scales with how many questions are batched into one call
GROUPED_MIN_TOKENS = 300
GROUPED_MAX_TOKENS = 4000


def load_gold_questions(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("items"):
        raise SystemExit(
            f"{path} has no items. Run gold_questions.py first: "
            f"python eval/trait_extraction/gold_questions.py"
        )
    return data


def get_paper_text(paper_id: int) -> Optional[str]:
    tei_path = DEFAULT_PAPERS_DIR / f"{paper_id}.tei.xml"
    if not tei_path.exists():
        return None
    return tei_to_text(str(tei_path))


def build_judge_prompt(field: str, evidence: str, gold_values: List[str], predicted: str) -> tuple[str, str]:
    system_prompt = (
        "You are an impartial scientific reviewer checking a trait-extraction system's output "
        "against evidence sentences and an expert-curated reference value.\n"
        "You will be given: the trait field name, the evidence sentences the extraction system "
        "was shown, the expert reference value(s), and the system's predicted value.\n"
        "Decide whether the system's prediction, although different from the reference value, is "
        "still PLAUSIBLY SUPPORTED by the evidence sentences (label: ambiguous) or whether it "
        "CONTRADICTS the evidence / reference (label: incorrect).\n"
        "Respond with exactly two lines:\n"
        "label: ambiguous OR incorrect\n"
        "reason: <one short sentence>\n"
    )
    user_content = (
        f"Trait field: {field}\n\n"
        f"Evidence sentences shown to the system:\n{evidence.strip()}\n\n"
        f"Reference value(s): {', '.join(gold_values)}\n"
        f"System's predicted value: {predicted}\n"
    )
    return system_prompt, user_content


def parse_judge_verdict(raw_text: str) -> tuple[str, str]:
    label = "incorrect"  # conservative default if parsing fails
    reason = raw_text.strip()
    for line in raw_text.splitlines():
        line = line.strip()
        if line.lower().startswith("label:"):
            value = line.split(":", 1)[1].strip().lower()
            if "ambig" in value:
                label = "ambiguous"
            elif "incorrect" in value or "wrong" in value:
                label = "incorrect"
        if line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    return label, reason


def keyword_only_predict(field: str, evidence_text: str) -> str:
    """No-LLM baseline: does any controlled-vocabulary value for this
    field appear literally in the evidence text?"""
    candidates = ENUM_HINTS.get(field, [])
    text_lower = evidence_text.lower()
    for value in candidates:
        if value.lower() in text_lower:
            return value
    return ""  # abstain


def _score_item(item: Dict[str, Any], answer_text: Optional[str], model_id: Optional[str],
                 condition: str) -> List[Dict[str, Any]]:
    """
    Scores one gold_questions.json item against a (possibly missing)
    parsed answer, returning one scored row for a "single"/"negative"
    item or TWO rows (one per sub-field) for a "combo" item.

    item_type-specific scoring:
      - "single": ordinary classify_prediction against the item's own
        gold_values.
      - "negative": the field has NO real annotation for this paper, so
        the only correct behaviour is abstention. classify_prediction is
        called with gold_values=[] (nothing can ever "match"), and a
        resulting "disagreement" is relabeled "hallucination" -- the
        model produced a confident-looking answer for a fact that, per
        MoultDB's own curators, isn't in the paper. This is a stronger
        and more specific claim than ordinary "wrong", which is exactly
        why it gets its own label and its own report table rather than
        being folded into the single/combo accuracy numbers.
      - "combo": the answer is expected to contain both sub-answers
        separated by " | " (see scoring.GROUPED_SYSTEM_PROMPT), in the
        same order as item["fields"]. If the split doesn't cleanly
        produce exactly as many parts as fields, we score the whole
        answer text against EACH field's gold as a fallback (better than
        discarding the item) but flag combo_split_ok=False so this
        doesn't silently masquerade as a clean compound answer.

    `answer_text=None` means no line was found for this item's index in
    the grouped response (a genuine parse failure, not an abstention) --
    scored as "parse_failed" (or "error" if the whole call failed, which
    the caller relabels).
    """
    item_type = item.get("type", "single")

    if item_type == "combo":
        fields = item["fields"]
        gold_by_field = item["gold_values_by_field"]
        if answer_text is None:
            return [{
                "item_id": item["item_id"], "paper_id": item["paper_id"], "field": f,
                "gold_values": gold_by_field[f], "model_id": model_id, "condition": condition,
                "raw_answer": None, "label": "parse_failed", "cleaned_prediction": None,
                "item_type": item_type, "combo_split_ok": None,
            } for f in fields]

        parts = re.split(r"\s*\|\s*", answer_text.strip())
        split_ok = len(parts) == len(fields)
        if not split_ok:
            parts = [answer_text] * len(fields)
        rows = []
        for f, part in zip(fields, parts):
            gold_values = gold_by_field[f]
            label, cleaned = classify_prediction(part, gold_values)
            rows.append({
                "item_id": item["item_id"], "paper_id": item["paper_id"], "field": f,
                "gold_values": gold_values, "model_id": model_id, "condition": condition,
                "raw_answer": answer_text, "label": label, "cleaned_prediction": cleaned,
                "item_type": item_type, "combo_split_ok": split_ok,
            })
        return rows

    gold_values = item.get("gold_values", [])
    if answer_text is None:
        return [{
            "item_id": item["item_id"], "paper_id": item["paper_id"], "field": item.get("field"),
            "gold_values": gold_values, "model_id": model_id, "condition": condition,
            "raw_answer": None, "label": "parse_failed", "cleaned_prediction": None, "item_type": item_type,
        }]

    label, cleaned = classify_prediction(answer_text, gold_values)
    if item_type == "negative" and label == "disagreement":
        label = "hallucination"
    return [{
        "item_id": item["item_id"], "paper_id": item["paper_id"], "field": item.get("field"),
        "gold_values": gold_values, "model_id": model_id, "condition": condition,
        "raw_answer": answer_text, "label": label, "cleaned_prediction": cleaned, "item_type": item_type,
    }]


GROUPED_MAX_RETRIES = 4
GROUPED_BASE_BACKOFF_SEC = 5.0


def _call_with_retry(system_prompt: str, user_content: str, model_id: str, max_new_tokens: int) -> Dict[str, Any]:
    """
    Thin retry wrapper around backend.providers.generate, specifically for
    429 rate-limit errors -- other ProviderErrors (bad model id, auth
    failure, malformed response) are NOT retried, since retrying those
    just reproduces the same failure slower. Grouping calls per-paper
    already cut request VOLUME by roughly the average number of items per
    paper (~24x at the 500-item/21-paper scale this module is designed
    for); this handles the residual case where a model's free-tier limit
    is still tight enough to trip on the smaller number of larger calls.
    Supersedes the old standalone rerun_on_hard_items.py /
    rejudge_unclassified.py patch scripts, which predate the grouped +
    typed (single/combo/negative) item format and are no longer
    compatible with it -- see README.md.
    """
    attempt = 0
    while True:
        try:
            return llm_generate(system_prompt=system_prompt, user_content=user_content,
                                 model_id=model_id, max_new_tokens=max_new_tokens)
        except ProviderError as e:
            is_rate_limit = "429" in str(e) or "rate_limited" in str(e).lower()
            if not is_rate_limit or attempt >= GROUPED_MAX_RETRIES:
                raise
            wait = GROUPED_BASE_BACKOFF_SEC * (2 ** attempt)
            print(f"      [RATE LIMIT] retry {attempt + 1}/{GROUPED_MAX_RETRIES} for {model_id} in {wait:.1f}s...")
            time.sleep(wait)
            attempt += 1


def _run_grouped_call(evidence_text: str, paper_items: List[Dict[str, Any]], model_id: str,
                       condition: str, raw_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Makes ONE API call for every item in `paper_items` (all from the same
    paper), instead of one call per item -- see scoring.py's grouped-mode
    docstring for why (rate-limit volume was the actual cause of a real
    run's failures, not question complexity). Returns the scored rows for
    every item, whether the call succeeded, partially parsed, or errored
    outright.
    """
    paper_id = paper_items[0]["paper_id"]
    questions = [it["question"] for it in paper_items]
    user_content = build_grouped_user_content(evidence_text, questions)
    max_tokens = min(GROUPED_MAX_TOKENS, max(GROUPED_MIN_TOKENS, GROUPED_TOKENS_PER_ITEM * len(paper_items)))

    try:
        result = _call_with_retry(GROUPED_SYSTEM_PROMPT, user_content, model_id, max_tokens)
        raw_answer = result["text"]
        error = None
    except ProviderError as e:
        raw_answer = None
        error = str(e)

    raw_calls.append({
        "paper_id": paper_id, "condition": condition, "model_id": model_id,
        "n_items": len(paper_items), "raw_answer": raw_answer, "error": error,
    })

    scored_rows: List[Dict[str, Any]] = []
    if error is not None:
        for it in paper_items:
            for row in _score_item(it, None, model_id, condition):
                row["label"] = "error"  # distinct from parse_failed: the whole call never returned
                scored_rows.append(row)
        print(f"    [grouped] paper {paper_id} x {model_id} ({condition}): ERROR {error}")
        return scored_rows

    parsed = parse_grouped_response(raw_answer, len(paper_items))
    for idx, it in enumerate(paper_items, start=1):
        answer_text = parsed.get(idx)
        rows = _score_item(it, answer_text, model_id, condition)
        scored_rows.extend(rows)
        for row in rows:
            print(f"    item {row['item_id']} ({row['field']}) x {model_id} [{condition}]: "
                  f"{row['label']} ({row['cleaned_prediction']!r})")
    return scored_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold_questions", type=str, default=str(DEFAULT_GOLD_QUESTIONS))
    parser.add_argument("--models", type=str, nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--baseline_model", type=str, default=None,
                         help="Model used for the ungated_fulltext baseline (default: first entry in --models).")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_judge", action="store_true", help="Leave disagreements unclassified (faster/cheaper smoke test).")
    parser.add_argument("--judge_sample_review", type=int, default=10,
                         help="Print this many random judge verdicts to stdout for a human sanity-check.")
    parser.add_argument("--taxonomy_csv", type=str, default=str(LLM_ROOT / "data" / "arthropod_taxonomy.csv"))
    parser.add_argument("--taxonomy_pickle", type=str, default=str(LLM_ROOT / "data" / "taxonomy_lookup.pkl"))
    parser.add_argument("--ontology_owl", type=str, default=str(LLM_ROOT / "data" / "moultdb_moulting_ontology_v3_8.owl"))
    parser.add_argument("--out_dir", type=str, default=str(EVAL_ROOT / "results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_model = args.baseline_model or args.models[0]
    args.baseline_model = baseline_model  # resolved value also used by _write_report

    gold = load_gold_questions(Path(args.gold_questions))
    items = gold["items"]
    items_by_paper: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        items_by_paper[it["paper_id"]].append(it)

    print(f"[INFO] {len(items)} gold items across {len(items_by_paper)} papers.")
    print(f"[INFO] Models: {args.models} | judge: {args.judge_model} | baseline model: {baseline_model}")

    print("[INFO] Loading taxonomy lookup + ontology gate...")
    taxonomy_lookup = TaxonomyLookup(csv_path=args.taxonomy_csv, pickle_path=args.taxonomy_pickle, rebuild=False)
    ontology_gate = MoultingOntologyGate(owl_path=args.ontology_owl)

    raw_calls: List[Dict[str, Any]] = []
    scored_rows: List[Dict[str, Any]] = []
    n_papers_gated_out = 0

    def _gated_out_rows(paper_id: int, paper_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for it in paper_items:
            if it["type"] == "combo":
                for f in it["fields"]:
                    rows.append({
                        "item_id": it["item_id"], "paper_id": paper_id, "field": f,
                        "gold_values": it["gold_values_by_field"][f], "model_id": None, "condition": "gated_out",
                        "raw_answer": None, "label": "gated_out", "cleaned_prediction": None, "item_type": "combo",
                    })
            else:
                rows.append({
                    "item_id": it["item_id"], "paper_id": paper_id, "field": it.get("field"),
                    "gold_values": it.get("gold_values", []), "model_id": None, "condition": "gated_out",
                    "raw_answer": None, "label": "gated_out", "cleaned_prediction": None, "item_type": it["type"],
                })
        return rows

    for paper_id, paper_items in items_by_paper.items():
        print(f"\n=== Paper {paper_id} ({len(paper_items)} items: "
              f"{sum(1 for it in paper_items if it['type'] == 'single')} single, "
              f"{sum(1 for it in paper_items if it['type'] == 'combo')} combo, "
              f"{sum(1 for it in paper_items if it['type'] == 'negative')} negative) ===")
        full_text = get_paper_text(paper_id)
        if not full_text or len(full_text.strip()) < 100:
            print(f"[WARN] Could not extract text for paper {paper_id}, skipping its {len(paper_items)} items.")
            continue

        probe_query = "Extract moulting-related traits from this paper."
        pipeline_result = run_domain_pipeline(
            full_text=full_text,
            user_query=probe_query,
            taxonomy_lookup=taxonomy_lookup,
            ontology_gate=ontology_gate,
        )
        summary = pipeline_result["summary"]
        decision = pipeline_result["decision"]
        if not decision["allow"] or not summary.strip():
            print(f"[WARN] Paper {paper_id} was gated out ({decision['final_label']}) or produced an empty "
                  f"summary -- recording as gated_out for its {len(paper_items)} items rather than skipping silently.")
            n_papers_gated_out += 1
            scored_rows.extend(_gated_out_rows(paper_id, paper_items))
            continue

        ungated_text = full_text[:UNGATED_CHAR_BUDGET]

        # Main condition: ONE grouped call per (paper, model), covering every
        # single/combo/negative item sampled for this paper at once -- see
        # _run_grouped_call / scoring.py's grouped-mode docstring for why.
        for model_id in args.models:
            scored_rows.extend(_run_grouped_call(summary, paper_items, model_id, "main", raw_calls))

        if not args.skip_baselines:
            # keyword-only baseline (no LLM) -- free, still done per item.
            for it in paper_items:
                if it["type"] == "combo":
                    for f in it["fields"]:
                        gold_values = it["gold_values_by_field"][f]
                        kw_pred = keyword_only_predict(f, summary)
                        kw_label, kw_cleaned = classify_prediction(kw_pred, gold_values)
                        scored_rows.append({
                            "item_id": it["item_id"], "paper_id": paper_id, "field": f,
                            "gold_values": gold_values, "model_id": "keyword_only", "condition": "baseline_keyword",
                            "raw_answer": kw_pred, "label": kw_label, "cleaned_prediction": kw_cleaned,
                            "item_type": "combo",
                        })
                else:
                    field = it.get("field")
                    gold_values = it.get("gold_values", [])
                    kw_pred = keyword_only_predict(field, summary) if field else ""
                    kw_label, kw_cleaned = classify_prediction(kw_pred, gold_values)
                    if it["type"] == "negative" and kw_label == "disagreement":
                        kw_label = "hallucination"
                    scored_rows.append({
                        "item_id": it["item_id"], "paper_id": paper_id, "field": field,
                        "gold_values": gold_values, "model_id": "keyword_only", "condition": "baseline_keyword",
                        "raw_answer": kw_pred, "label": kw_label, "cleaned_prediction": kw_cleaned,
                        "item_type": it["type"],
                    })

            # ungated_fulltext baseline: same model + same grouped format, no
            # gate/selection -- one grouped call per paper, isolating the
            # evidence-selection stage's contribution the same way the
            # original per-item design did, just batched.
            scored_rows.extend(_run_grouped_call(ungated_text, paper_items, baseline_model,
                                                  "baseline_ungated_fulltext", raw_calls))

    # ── Judge pass over every "disagreement" row from the main condition ──
    judge_verdicts: Dict[int, Dict[str, Any]] = {}
    if not args.skip_judge:
        disagreements = [r for r in scored_rows if r["label"] == "disagreement" and r["condition"] == "main"]
        print(f"\n[INFO] Sending {len(disagreements)} disagreements to judge model {args.judge_model}...")

        # Rebuild each paper's summary once for judge context (cheap: pure text, no LLM call).
        evidence_by_paper: Dict[int, str] = {}
        for row in disagreements:
            pid = row["paper_id"]
            if pid not in evidence_by_paper:
                full_text = get_paper_text(pid)
                pr = run_domain_pipeline(
                    full_text=full_text,
                    user_query="Extract moulting-related traits from this paper.",
                    taxonomy_lookup=taxonomy_lookup,
                    ontology_gate=ontology_gate,
                )
                evidence_by_paper[pid] = pr["summary"]

        for idx, row in enumerate(disagreements):
            evidence = evidence_by_paper.get(row["paper_id"], "")
            sys_p, user_p = build_judge_prompt(row["field"], evidence, row["gold_values"], row["cleaned_prediction"])
            try:
                result = _call_with_retry(sys_p, user_p, args.judge_model, 128)
                verdict_label, verdict_reason = parse_judge_verdict(result["text"])
            except ProviderError as e:
                verdict_label, verdict_reason = "unclassified", str(e)
            key = (row["item_id"], row["model_id"])
            judge_verdicts[key] = {"verdict": verdict_label, "reason": verdict_reason}
            row["judge_verdict"] = verdict_label
            row["judge_reason"] = verdict_reason
            print(f"    [{idx + 1}/{len(disagreements)}] item {row['item_id']} x {row['model_id']}: {verdict_label} -- {verdict_reason}")

        if args.judge_sample_review and disagreements:
            rng = random.Random(args.seed)
            sample = rng.sample(disagreements, k=min(args.judge_sample_review, len(disagreements)))
            print(f"\n[REVIEW] Random sample of {len(sample)} judge verdicts for a human sanity-check:")
            for row in sample:
                print(f"  paper={row['paper_id']} field={row['field']!r} gold={row['gold_values']} "
                      f"predicted={row['cleaned_prediction']!r} -> {row.get('judge_verdict')}: {row.get('judge_reason')}")

    # ── Write outputs ──
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "results_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in raw_calls:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    import csv
    csv_path = out_dir / "results_scored.csv"
    fieldnames = ["item_id", "item_type", "paper_id", "field", "gold_values", "model_id", "condition",
                  "raw_answer", "label", "cleaned_prediction", "combo_split_ok", "judge_verdict", "judge_reason"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scored_rows:
            out_row = {k: row.get(k) for k in fieldnames}
            out_row["gold_values"] = "; ".join(row.get("gold_values") or [])
            writer.writerow(out_row)

    report_path = out_dir / "report.md"
    _write_report(report_path, scored_rows, args, n_papers_gated_out, len(items_by_paper))

    print(f"\n[DONE] Wrote {raw_path}, {csv_path}, {report_path}")


def _write_report(path: Path, scored_rows: List[Dict[str, Any]], args, n_gated_out: int, n_papers: int) -> None:
    lines: List[str] = []
    lines.append("# Trait extraction — model comparison vs. MoultDB ground truth\n")
    lines.append(
        f"Generated by `run_model_comparison.py`. {n_papers} papers, "
        f"{n_gated_out} gated out before extraction. Models compared: {args.models}. "
        f"Judge model: {args.judge_model if not args.skip_judge else '(skipped)'}.\n"
    )
    lines.append(
        "**Read this before the numbers below**: `disagreement` rows are split into `ambiguous` "
        "(judge model decided the prediction is still supported by the evidence) and `incorrect` "
        "(judge decided it contradicts the evidence/gold) only if the judge pass ran. The judge is "
        "an LLM (same family being evaluated, not a MoultDB domain expert) — treat "
        "ambiguous/incorrect splits as a documented heuristic, and see the "
        "`--judge_sample_review` console output from the run that produced this report for a "
        "human spot-check of a random subsample of its verdicts.\n"
    )

    def label_counts(rows):
        counts = defaultdict(int)
        for r in rows:
            label = r["label"]
            if label == "disagreement":
                verdict = r.get("judge_verdict")
                # Two distinct ways a disagreement ends up unclassified:
                # (a) the judge pass never ran / was skipped -> verdict is None/"".
                # (b) the judge pass ran but the judge model itself errored
                #     (e.g. a 429) -> verdict is the literal string
                #     "unclassified" (set in the except branch around the judge
                #     call), which is truthy and previously fell through to
                #     `counts["unclassified"]` instead of the bucket the report
                #     table actually reads from ("disagreement (unclassified)"),
                #     silently undercounting real disagreements as if they'd
                #     never occurred. Both cases now map to the same bucket.
                if verdict in (None, "", "unclassified"):
                    label = "disagreement (unclassified)"
                else:
                    label = verdict
            counts[label] += 1
        return counts

    is_accuracy_row = lambda r: r.get("item_type") in ("single", "combo")  # noqa: E731
    is_negative_row = lambda r: r.get("item_type") == "negative"  # noqa: E731

    lines.append("## Main condition — accuracy (single + combo items, gated + selected evidence)\n")
    lines.append(
        "Combo items contribute one row per sub-field (see `combo_split_ok` in results_scored.csv — "
        "`False` means the model didn't cleanly return two ` | `-separated answers for that compound "
        "question, so both sub-fields were scored against the whole answer text as a fallback).\n"
    )
    lines.append("| Model | n | correct | ambiguous | incorrect | unclassified disagreement | abstained | error/gated_out |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for model_id in args.models:
        rows = [r for r in scored_rows if r["condition"] == "main" and r["model_id"] == model_id and is_accuracy_row(r)]
        n = len(rows)
        c = label_counts(rows)
        other = c.get("error", 0) + c.get("gated_out", 0) + c.get("parse_failed", 0)
        lines.append(
            f"| {model_id} | {n} | {c.get('correct', 0)} | {c.get('ambiguous', 0)} | "
            f"{c.get('incorrect', 0)} | {c.get('disagreement (unclassified)', 0)} | "
            f"{c.get('abstained', 0)} | {other} |"
        )
    gated = [r for r in scored_rows if r["condition"] == "gated_out"]
    if gated:
        lines.append(f"\n> {len(gated)} item-rows came from papers gated out before extraction (see n_papers above).\n")

    lines.append(
        "\n## Main condition — hallucination check (negative items: field genuinely unannotated "
        "for that paper)\n"
    )
    lines.append(
        "The only correct behaviour on a negative item is abstention — MoultDB's own curators found "
        "nothing to record for this (paper, field) pair either. `hallucination` means the model gave a "
        "confident-looking answer anyway; this is a direct, judge-free measurement, not a heuristic.\n"
    )
    lines.append("| Model | n | abstained (correct) | hallucination | error/gated_out/parse_failed |")
    lines.append("|---|---|---|---|---|")
    for model_id in args.models:
        rows = [r for r in scored_rows if r["condition"] == "main" and r["model_id"] == model_id and is_negative_row(r)]
        n = len(rows)
        c = label_counts(rows)
        other = c.get("error", 0) + c.get("gated_out", 0) + c.get("parse_failed", 0)
        hallucination_rate = f"{100 * c.get('hallucination', 0) / n:.1f}%" if n else "n/a"
        lines.append(
            f"| {model_id} | {n} | {c.get('abstained', 0)} | {c.get('hallucination', 0)} ({hallucination_rate}) | {other} |"
        )

    if not args.skip_baselines:
        lines.append("\n## Baselines — accuracy (single + combo items)\n")
        lines.append("| Condition | n | correct | ambiguous/incorrect (unclassified) | abstained |")
        lines.append("|---|---|---|---|---|")
        for condition, label in [("baseline_keyword", "keyword-only (no LLM)"),
                                  ("baseline_ungated_fulltext", f"ungated full-text ({args.baseline_model})")]:
            rows = [r for r in scored_rows if r["condition"] == condition and is_accuracy_row(r)]
            n = len(rows)
            c = label_counts(rows)
            disagreement_n = c.get('disagreement (unclassified)', 0) + c.get('ambiguous', 0) + c.get('incorrect', 0)
            lines.append(f"| {label} | {n} | {c.get('correct', 0)} | {disagreement_n} | {c.get('abstained', 0)} |")
        lines.append(
            "\n> Baselines are not sent through the judge pass (judge budget is spent on the main "
            "condition only) — their disagreements are reported as a single unclassified count, not "
            "split into ambiguous/incorrect.\n"
        )

        lines.append("\n## Baselines — hallucination check (negative items)\n")
        lines.append("| Condition | n | abstained (correct) | hallucination |")
        lines.append("|---|---|---|---|")
        for condition, label in [("baseline_keyword", "keyword-only (no LLM)"),
                                  ("baseline_ungated_fulltext", f"ungated full-text ({args.baseline_model})")]:
            rows = [r for r in scored_rows if r["condition"] == condition and is_negative_row(r)]
            n = len(rows)
            c = label_counts(rows)
            lines.append(f"| {label} | {n} | {c.get('abstained', 0)} | {c.get('hallucination', 0)} |")

    lines.append("\n## Example disagreements\n")
    examples = [r for r in scored_rows if r["condition"] == "main" and r["label"] == "disagreement"][:8]
    for r in examples:
        lines.append(
            f"- **paper {r['paper_id']}, field \"{r['field']}\"** ({r['model_id']}): "
            f"gold=`{r['gold_values']}`, predicted=`{r['cleaned_prediction']}`, "
            f"judge=`{r.get('judge_verdict', 'n/a')}` — {r.get('judge_reason', '')}"
        )

    lines.append("\n## Example hallucinations (negative items)\n")
    hallucinations = [r for r in scored_rows if r["condition"] == "main" and r["label"] == "hallucination"][:8]
    if hallucinations:
        for r in hallucinations:
            lines.append(
                f"- **paper {r['paper_id']}, field \"{r['field']}\"** ({r['model_id']}): field has NO real "
                f"annotation for this paper (correct answer: abstain), model answered `{r['cleaned_prediction']}`"
            )
    else:
        lines.append("(none in this run)\n")

    n_single = sum(1 for r in scored_rows if r.get("item_type") == "single" and r["condition"] == "main"
                   and r["model_id"] == args.models[0])
    n_combo_items = len({r["item_id"] for r in scored_rows if r.get("item_type") == "combo"})
    n_negative = sum(1 for r in scored_rows if r.get("item_type") == "negative" and r["condition"] == "main"
                      and r["model_id"] == args.models[0])
    lines.append(
        "\n## Limitations (stated explicitly, not hidden)\n"
        f"- This run's gold set: {n_single} single-field items, {n_combo_items} compound "
        f"(two-field) items, {n_negative} negative (should-abstain) items across {n_papers} papers. "
        "Real but still modest for claiming statistically robust superiority between models — enough "
        "to catch gross pipeline failures and produce genuine numbers.\n"
        "- Positive items beyond the corpus's ~200 unique (paper, field) pairs are the SAME pairs "
        "asked again with a different phrasing (see `phrasing_round` in gold_questions.json), not new "
        "independent facts — this tests phrasing robustness, not additional knowledge coverage. Rows "
        "sharing a (paper, field) but different phrasing_round are not independent samples for "
        "significance testing.\n"
        "- The judge model is an LLM, not a MoultDB domain expert; ambiguous/incorrect splits should "
        "be read as a documented heuristic (see `--judge_sample_review` output for a spot-check), not "
        "ground truth. Negative items are NOT judged (hallucination vs. correct abstention is "
        "unambiguous by construction, no judge needed).\n"
        "- Gold values come from a single expert-curated spreadsheet; MoultDB annotation "
        "guidelines/conventions were not independently re-verified against the source papers as "
        "part of this evaluation. A handful of gold cells look potentially miscoded on manual "
        "inspection (see the paper's Discussion) — treat isolated gold/model disagreements with that "
        "in mind, not as automatic proof the model is wrong.\n"
        "- Combo items ask two fields in one question and expect a ` | `-separated answer; "
        "`combo_split_ok=False` rows fall back to scoring the whole answer against each field "
        "independently, which is more forgiving than a clean split and should be read as such.\n"
        "- Grouped-per-paper prompting (used here) differs from both the deployed default full_traits "
        "mode AND from a literal one-question-per-call design — see this script's module docstring "
        "and the paper's Discussion for what this does and doesn't tell us about the deployed "
        "system's behaviour.\n"
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
