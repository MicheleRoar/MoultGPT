#!/usr/bin/env python3
# llm/eval/trait_extraction/gold_questions.py
"""
Builds a gold-standard evaluation set of (paper, trait field, expected
value) items from the real MoultDB ground truth, for
run_model_comparison.py to score model output against.

Why this exists
----------------
The paper this repo accompanies claims quantitative trait-extraction
results (accuracy, correct/incorrect/ambiguous/unknown breakdown) that,
before this module, were not backed by any actual computed evaluation --
see the module docstring of run_model_comparison.py for the full context.
This script is step 1: decide, once and deterministically, WHICH
(paper, field) pairs we are going to check, before any model is ever
called. Fixing the question set up front (and writing it to a file) means
the evaluation can't be quietly cherry-picked after seeing model outputs.

Source of truth
----------------
`llm/finetuning/MoultDB character annotations.xlsx`, sheet "data" -- the
same expert-curated annotation table `llm/config/trait_schema.py` derives
its 55-field prompt schema from (see llm/data/moultdb_trait_schema.json).
Restricted to the 21 "Paper ID" values that also have a pre-parsed
`llm/finetuning/papers/<id>.tei.xml` (verified overlap: all 21 papers with
a .tei.xml have at least one matching annotation row -- see this repo's
session notes / git history if that ever needs re-checking after the
Excel or the papers/ folder changes).

Sampling method
----------------
For each (paper_id, field) where at least one annotation row for that
paper has a real (non-missing, non-"NA"/"unknown") value, we have one
candidate item. A single paper is usually annotated with several rows
(different taxa/specimens from the same publication), so a field's
"gold value(s)" for a paper is the SET of distinct real values seen
across all of that paper's rows -- a model answer is scored as correct
if it matches ANY of them, not just the first row encountered.

We deterministically sample --n_items (default 50) candidates, round-
robin across papers (so no single well-annotated paper dominates the
set) with a fixed random seed, after excluding a short blocklist of
fields that are annotation provenance/metadata rather than biological
moulting traits recoverable from the paper's text (see
EXCLUDED_NON_TRAIT_FIELDS below) -- asking a language model to "extract"
e.g. the annotator's own confidence code from the paper text is not a
meaningful text-extraction test.

Usage
-----
    cd llm
    python eval/trait_extraction/gold_questions.py \\
        --out eval/trait_extraction/gold_questions.json --n_items 50
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

EVAL_ROOT = Path(__file__).resolve().parent
LLM_ROOT = EVAL_ROOT.parents[1]
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

DEFAULT_XLSX = LLM_ROOT / "finetuning" / "MoultDB character annotations.xlsx"
DEFAULT_PAPERS_DIR = LLM_ROOT / "finetuning" / "papers"
DEFAULT_SCHEMA_JSON = LLM_ROOT / "data" / "moultdb_trait_schema.json"

# Administrative / provenance fields that ARE in the 55-field prompt schema
# (config/trait_schema.py) but are not meaningful things to ask a language
# model to "find in the paper text" for a QA-style accuracy check -- e.g.
# "Confidence" here is the MoultDB annotator's own confidence code
# (a CIO ontology term describing THEIR annotation), not a fact stated in
# the source article.
EXCLUDED_NON_TRAIT_FIELDS = {
    "Determined by",
    "Contributor",
    "Museum collection",
    "Museum accession",
    "Location: GPS coordinates",
    "Evidence code",
    "Confidence",
    "General Comments",
}

# Fields that ARE real annotation columns with real values, but describe the
# STUDY/SPECIMEN rather than the organism's moulting biology -- locality,
# geological context, sample size, specimen provenance. Excluded on request:
# the evaluation should test whether the pipeline recovers actual moulting
# TRAITS, not incidental paper metadata that happens to share the same
# spreadsheet. (These also turned out to be exactly the highest-coverage
# fields in the raw annotation table -- "Type of specimens of interest" and
# "Environment" are each annotated for 20/21 papers, "Number of specimens
# in the sample" for 18/21 -- so leaving them in would have skewed a
# coverage-weighted sample toward non-trait fields.)
EXCLUDED_PAPER_INFO_FIELDS = {
    "Type of specimens of interest",
    "Environment",
    "Number of specimens in the sample",
    "Number of specimens for this annotation",
    "Geological formation",
    "Biozone",
    "Geological age",
    "Geological age.1",
}

# The values MoultDB annotators use to mean "not applicable / not known" --
# these must NOT count as a real gold value, or every model that abstains
# on a genuinely-unannotated field would look "correct" by accident.
NA_LIKE_VALUES = {
    "", "na", "n/a", "unknown", "?", "none", "nan",
}

# Paper IDs (in the annotation spreadsheet) that have a matching pre-parsed
# llm/finetuning/papers/<id>.tei.xml -- computed once (see module
# docstring) and hard-coded here so this script doesn't silently drift if
# someone adds a stray .tei.xml without a matching annotation row, or vice
# versa. Regenerate by intersecting the "Paper ID" column of the "data"
# sheet with the *.tei.xml stems in llm/finetuning/papers/ if either
# changes.
AVAILABLE_PAPER_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26, 39, 44, 45, 46]


def _is_real_value(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    if not s:
        return False
    return s.lower() not in NA_LIKE_VALUES


def load_annotation_rows(xlsx_path: Path, sheet_name: str = "data") -> List[Dict[str, Any]]:
    """Returns one dict per annotation row, keyed by column header."""
    import openpyxl  # local import: only needed for this offline data-prep step

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else None for h in next(rows_iter)]

    rows: List[Dict[str, Any]] = []
    for raw_row in rows_iter:
        row = {}
        for col_name, value in zip(header, raw_row):
            if col_name is None:
                continue
            row[col_name] = value
        if row.get("Paper ID") is not None:
            rows.append(row)
    return rows


def build_candidates(
    rows: List[Dict[str, Any]],
    trait_columns: List[str],
    available_paper_ids: List[int],
    max_gold_values: int = 6,
) -> Dict[int, Dict[str, Set[str]]]:
    """
    Returns {paper_id: {field: {distinct real values across all rows}}}.

    A paper with many annotated specimens/rows (e.g. a large ontogenetic
    series) can accumulate a long list of distinct values for a
    per-specimen field like "Previous ontogenetic stage" -- if left in,
    "matches ANY gold value" degrades into "mentions any plausible value
    at all", which is not a meaningful single-answer QA target. Fields
    with more than `max_gold_values` distinct real values for a given
    paper are dropped as candidates for THAT paper (the field can still
    be sampled for other, less fragmented papers).
    """
    excluded = EXCLUDED_NON_TRAIT_FIELDS | EXCLUDED_PAPER_INFO_FIELDS
    wanted_fields = [c for c in trait_columns if c not in excluded]
    available = set(available_paper_ids)

    by_paper: Dict[int, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        try:
            paper_id = int(row["Paper ID"])
        except (TypeError, ValueError):
            continue
        if paper_id not in available:
            continue
        for field in wanted_fields:
            value = row.get(field)
            if _is_real_value(value):
                by_paper[paper_id][field].add(str(value).strip())

    filtered: Dict[int, Dict[str, Set[str]]] = {}
    n_dropped = 0
    for pid, fields in by_paper.items():
        kept = {}
        for field, values in fields.items():
            if len(values) > max_gold_values:
                n_dropped += 1
                continue
            kept[field] = values
        filtered[pid] = kept
    if n_dropped:
        print(f"[INFO] Dropped {n_dropped} (paper, field) candidates with more than "
              f"{max_gold_values} distinct gold values (too fragmented for single-answer QA).")

    return filtered


def field_coverage(by_paper: Dict[int, Dict[str, Set[str]]]) -> Dict[str, int]:
    """Number of distinct papers (out of those passed in) that have at
    least one real value for each field -- used to prioritize
    well-annotated trait fields over ones with a single stray value."""
    coverage: Dict[str, int] = defaultdict(int)
    for fields in by_paper.values():
        for field in fields:
            coverage[field] += 1
    return dict(coverage)


def filter_by_min_coverage(
    by_paper: Dict[int, Dict[str, Set[str]]],
    min_field_coverage: int,
) -> Dict[int, Dict[str, Set[str]]]:
    """Drops fields annotated in fewer than `min_field_coverage` of the
    available papers -- keeps the candidate pool focused on trait fields
    that are actually recoverable across a meaningful slice of the
    corpus, rather than a field with a single lucky annotation."""
    coverage = field_coverage(by_paper)
    kept_fields = {f for f, n in coverage.items() if n >= min_field_coverage}
    dropped = sorted(f for f in coverage if f not in kept_fields)
    if dropped:
        print(f"[INFO] Dropped {len(dropped)} fields annotated in fewer than "
              f"{min_field_coverage} papers: {dropped}")

    result: Dict[int, Dict[str, Set[str]]] = {}
    for pid, fields in by_paper.items():
        result[pid] = {f: v for f, v in fields.items() if f in kept_fields}
    return result


QUESTION_TEMPLATES = [
    "According to this paper, what is the value for \"{field}\"?",
    "Based on the evidence above, what does the paper report for {field}?",
    "Determine the {field} described in this paper.",
    "From the given evidence, what is the {field}?",
    "Extract the value of \"{field}\" from the text above.",
]

COMBO_QUESTION_TEMPLATES = [
    "According to this paper, what are the values for \"{field_a}\" and \"{field_b}\"?",
    "Based on the evidence above, what does the paper report for {field_a}, and separately for {field_b}?",
]


def _field_for_display(field: str) -> str:
    """
    Renders a MoultDB field name for use INSIDE a question string only --
    the raw `field` value (with '#' intact) is still what's used for gold
    lookups/scoring, this only changes what the model is shown.

    Several real field names contain a literal '#' (e.g. "Observed #
    total moult stages"). Beyond the YAML-parsing bug this already caused
    once (see scoring.py's _strip_hash_from_yaml_keys), putting a bare
    '#' in front of a language model is just asking for trouble -- some
    tokenizers/chat templates treat leading '#' as a markdown heading
    marker, and it's not how a human would ask the question anyway.
    "Observed # total moult stages" reads naturally as "Observed number
    of total moult stages"; "# body segments in adult individuals" reads
    naturally as "number of body segments in adult individuals".
    """
    display = field.replace("#", "number of")
    display = re.sub(r"\s+", " ", display).strip()
    return display


def humanize_question(field: str, template: Optional[str] = None,
                       rng: Optional[random.Random] = None) -> str:
    """
    Deliberately simple, templated question generation -- not hand-curated
    per field, not fancy NLG. The point of this evaluation is what the
    extraction pipeline can recover given the MoultDB field name (which is
    exactly what's injected into the real system prompt, see
    pipeline/prompting.py), not how well a hand-crafted natural-language
    question can be phrased.

    Pass an explicit `template` when the caller needs a SPECIFIC phrasing
    (e.g. sample_positive_items cycling through QUESTION_TEMPLATES in
    fixed rounds, so the same (paper, field) pair asked a second time
    reliably gets a DIFFERENT template, not a random one that might repeat).
    Pass `rng` instead when any deterministic-but-unpredictable choice is
    fine. If neither is given, falls back to the first template.

    Rotating templates at all (rather than one fixed sentence reused for
    every item) was flagged as a real risk, not a stylistic nitpick: with
    identical phrasing on every call, any systematic behaviour tied to
    that specific phrasing (e.g. the model echoing the field name
    verbatim into its answer -- which is exactly what caused the
    '#'-in-fieldname YAML-parsing bug found in a real run, see
    scoring.py) gets baked into every single item instead of being one
    phrasing-specific failure mode among several.
    """
    if template is None:
        template = rng.choice(QUESTION_TEMPLATES) if rng is not None else QUESTION_TEMPLATES[0]
    return template.format(field=_field_for_display(field))


def _round_robin_pairs(
    per_paper_fields: Dict[int, List[str]],
    paper_ids_order: List[int],
    max_items_per_field: int,
) -> List[Tuple[int, str]]:
    """
    Shared core of the positive/negative samplers: round-robin across
    papers so no single paper dominates, walking each paper's
    (already-ordered) field list and capping how many times the same
    field name can be picked overall -- once a field hits its cap it's
    skipped in favour of that paper's next field. Returns ALL pairs this
    process yields (i.e. one full pass), not just up to some n_items --
    callers that need more items than this produces reuse the same pairs
    with a different question phrasing rather than inventing new pairs.
    """
    pairs: List[Tuple[int, str]] = []
    field_used_count: Dict[str, int] = defaultdict(int)
    cursor = {pid: 0 for pid in paper_ids_order}
    progressed = True
    while progressed:
        progressed = False
        for paper_id in paper_ids_order:
            fields = per_paper_fields[paper_id]
            i = cursor[paper_id]
            while i < len(fields) and field_used_count[fields[i]] >= max_items_per_field:
                i += 1
            cursor[paper_id] = i
            if i >= len(fields):
                continue
            field = fields[i]
            cursor[paper_id] = i + 1
            field_used_count[field] += 1
            progressed = True
            pairs.append((paper_id, field))
    return pairs


def sample_positive_items(
    by_paper: Dict[int, Dict[str, Set[str]]],
    n_items: int,
    seed: int,
    max_items_per_field: int = 3,
) -> List[Dict[str, Any]]:
    """
    Positive items: real (paper, field) pairs with a real annotated gold
    value. Round-robin across papers, preferring each paper's most
    globally-annotated fields first, capped at `max_items_per_field` uses
    of the same field PER PASS.

    Scaling beyond one pass: the corpus only supports a limited number of
    genuinely distinct (paper, field) pairs (roughly 200, see the
    n_candidates log line) -- reaching a larger --n_positive than that
    without fabricating data means asking the SAME real pairs again with
    a DIFFERENT phrasing (see QUESTION_TEMPLATES / humanize_question).
    This is a real, legitimate technique (paraphrase-robustness
    augmentation), not padding: it directly tests whether a model's
    correctness on a given fact depends on how the question happens to be
    worded, which single-fixed-phrasing evaluation can't distinguish from
    genuine knowledge. Each item records `phrasing_round` (0 = the
    original single pass, 1+ = a reused pair with the next template) so
    this can be analyzed as its own axis in the paper, not hidden.
    """
    rng = random.Random(seed)
    coverage = field_coverage(by_paper)

    per_paper_fields: Dict[int, List[str]] = {}
    for paper_id, fields in by_paper.items():
        field_list = list(fields.keys())
        jitter = {f: rng.random() for f in field_list}
        field_list.sort(key=lambda f: (-coverage.get(f, 0), jitter[f]))
        per_paper_fields[paper_id] = field_list

    paper_ids_order = list(per_paper_fields.keys())
    rng.shuffle(paper_ids_order)

    base_pairs = _round_robin_pairs(per_paper_fields, paper_ids_order, max_items_per_field)
    if not base_pairs:
        return []

    items: List[Dict[str, Any]] = []
    round_idx = 0
    while len(items) < n_items and round_idx < len(QUESTION_TEMPLATES):
        template = QUESTION_TEMPLATES[round_idx]
        for paper_id, field in base_pairs:
            if len(items) >= n_items:
                break
            gold_values = sorted(by_paper[paper_id][field])
            items.append({
                "item_id": None,  # assigned after merging with negatives/combos
                "type": "single",
                "is_negative": False,
                "paper_id": paper_id,
                "field": field,
                "gold_values": gold_values,
                "question": humanize_question(field, template=template),
                "phrasing_round": round_idx,
            })
        round_idx += 1

    if len(items) < n_items:
        print(f"[WARN] Only {len(items)} positive items possible from {len(base_pairs)} unique "
              f"(paper, field) pairs x {len(QUESTION_TEMPLATES)} phrasings (requested {n_items}). "
              f"Writing all available rather than fabricating more.")

    return items


def build_negative_candidates(
    by_paper_unfiltered: Dict[int, Dict[str, Set[str]]],
    trait_columns: List[str],
    available_paper_ids: List[int],
) -> Dict[int, List[str]]:
    """
    For each available paper, the trait fields with NO real annotated
    value for that paper at all. These become "negative" items: the
    biologically correct behaviour is to abstain ("not mentioned"),
    because MoultDB's own curators found nothing to record for that
    (paper, field) pair either. A model that fabricates a confident
    answer here is hallucinating, not just wrong on a hard question --
    that's what the negative set measures that the positive set can't.

    IMPORTANT: takes `by_paper_unfiltered` -- the output of
    build_candidates() BEFORE filter_by_min_coverage/max_gold_values are
    applied to build the POSITIVE set. If we used the filtered dict here
    instead, a field dropped from positives purely for being
    over-fragmented (>max_gold_values distinct values) would incorrectly
    look "missing" and get sampled as a negative, even though the paper
    genuinely has real content for it -- which would corrupt the
    hallucination-rate measurement by mislabeling a real-content case as
    a should-abstain case.
    """
    excluded = EXCLUDED_NON_TRAIT_FIELDS | EXCLUDED_PAPER_INFO_FIELDS
    wanted_fields = [c for c in trait_columns if c not in excluded]

    negatives: Dict[int, List[str]] = {}
    for paper_id in available_paper_ids:
        annotated = set(by_paper_unfiltered.get(paper_id, {}).keys())
        negatives[paper_id] = [f for f in wanted_fields if f not in annotated]
    return negatives


def sample_negative_items(
    negatives_by_paper: Dict[int, List[str]],
    n_items: int,
    seed: int,
    max_items_per_field: int = 15,
) -> List[Dict[str, Any]]:
    """
    Same round-robin-across-papers, cap-per-field, multi-round-reuse
    pattern as sample_positive_items, applied to the "genuinely
    unannotated" pairs from build_negative_candidates instead. gold_values
    is always [] -- correct behaviour is abstention, scored specially by
    run_model_comparison.py (label "hallucination" if the model answers
    with a real-looking value instead of abstaining).
    """
    rng = random.Random(seed + 97)  # offset so negatives don't retrace positives' exact draws
    per_paper_fields: Dict[int, List[str]] = {}
    for paper_id, fields in negatives_by_paper.items():
        field_list = list(fields)
        rng.shuffle(field_list)
        per_paper_fields[paper_id] = field_list

    paper_ids_order = [pid for pid in negatives_by_paper if negatives_by_paper[pid]]
    rng.shuffle(paper_ids_order)

    base_pairs = _round_robin_pairs(per_paper_fields, paper_ids_order, max_items_per_field)
    if not base_pairs:
        return []

    items: List[Dict[str, Any]] = []
    round_idx = 0
    while len(items) < n_items and round_idx < len(QUESTION_TEMPLATES):
        template = QUESTION_TEMPLATES[round_idx]
        for paper_id, field in base_pairs:
            if len(items) >= n_items:
                break
            items.append({
                "item_id": None,
                "type": "negative",
                "is_negative": True,
                "paper_id": paper_id,
                "field": field,
                "gold_values": [],
                "question": humanize_question(field, template=template),
                "phrasing_round": round_idx,
            })
        round_idx += 1

    if len(items) < n_items:
        print(f"[WARN] Only {len(items)} negative items possible from {len(base_pairs)} unique "
              f"unannotated (paper, field) pairs (requested {n_items}).")

    return items


def sample_combo_items(
    by_paper: Dict[int, Dict[str, Set[str]]],
    n_items: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Compound items: ONE question asking about two distinct real fields
    from the SAME paper together, e.g. "what are the observed
    ontogenetic stage and the life mode reported in this paper?". Tests
    multi-attribute extraction in a single question, as opposed to
    sample_positive_items' one-field-per-question design -- a real,
    distinct axis from "does batching multiple questions in one API call
    help/hurt" (that's the grouped-prompting mode in
    run_model_comparison.py, which batches already-independent
    single-field questions; this is a genuinely COMPOUND question, one
    unit the model must parse two answers out of).

    Each combo item carries "fields": [a, b] and
    "gold_values_by_field": {a: [...], b: [...]}, scored as two
    independent sub-answers by run_model_comparison.py.
    """
    rng = random.Random(seed + 331)
    paper_ids = [pid for pid, fields in by_paper.items() if len(fields) >= 2]
    if not paper_ids:
        return []
    rng.shuffle(paper_ids)

    items: List[Dict[str, Any]] = []
    pair_used: Set[Tuple[int, Tuple[str, str]]] = set()
    idx = 0
    max_attempts = max(n_items * 50, 200)
    attempts = 0
    while len(items) < n_items and attempts < max_attempts:
        attempts += 1
        paper_id = paper_ids[idx % len(paper_ids)]
        idx += 1
        fields = list(by_paper[paper_id].keys())
        field_a, field_b = rng.sample(fields, 2)
        key = (paper_id, tuple(sorted([field_a, field_b])))
        if key in pair_used:
            continue
        pair_used.add(key)
        template = COMBO_QUESTION_TEMPLATES[len(items) % len(COMBO_QUESTION_TEMPLATES)]
        question = template.format(field_a=_field_for_display(field_a), field_b=_field_for_display(field_b))
        items.append({
            "item_id": None,
            "type": "combo",
            "is_negative": False,
            "paper_id": paper_id,
            "fields": [field_a, field_b],
            "gold_values_by_field": {
                field_a: sorted(by_paper[paper_id][field_a]),
                field_b: sorted(by_paper[paper_id][field_b]),
            },
            "question": question,
        })

    if len(items) < n_items:
        print(f"[WARN] Only {len(items)} combo items possible (requested {n_items}).")

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", type=str, default=str(DEFAULT_XLSX))
    parser.add_argument("--sheet", type=str, default="data")
    parser.add_argument("--schema_json", type=str, default=str(DEFAULT_SCHEMA_JSON))
    parser.add_argument("--out", type=str, default=str(EVAL_ROOT / "gold_questions.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_gold_values", type=int, default=6,
                         help="Drop (paper, field) candidates with more than this many distinct "
                              "gold values for that paper (too fragmented for single-answer QA).")
    parser.add_argument("--min_field_coverage", type=int, default=4,
                         help="Drop fields annotated in fewer than this many of the available "
                              "papers -- keeps the sample on well-annotated moulting traits.")
    parser.add_argument("--max_items_per_field", type=int, default=3,
                         help="Cap on how many times the same field can appear in one phrasing round.")
    parser.add_argument("--n_positive", type=int, default=300,
                         help="Total positive items (real gold value), single + combo combined.")
    parser.add_argument("--n_combo", type=int, default=45,
                         help="Of --n_positive, how many are compound two-field questions "
                              "(the rest are single-field). ~15%% by default.")
    parser.add_argument("--n_negative", type=int, default=200,
                         help="Items where the field has no annotation for that paper at all -- "
                              "correct behaviour is abstention; measures hallucination rate directly.")
    args = parser.parse_args()

    schema = json.loads(Path(args.schema_json).read_text(encoding="utf-8"))
    trait_columns = schema["trait_columns"]

    print(f"[INFO] Loading annotation rows from {args.xlsx} (sheet={args.sheet})...")
    rows = load_annotation_rows(Path(args.xlsx), args.sheet)
    print(f"[INFO] {len(rows)} annotation rows loaded.")

    # Unfiltered view (only max_gold_values=infinite-ish, i.e. no fragmentation
    # drop) -- used to know what's genuinely UNANNOTATED for negatives, so a
    # field dropped from positives for being over-fragmented doesn't get
    # mislabeled as "missing".
    by_paper_unfiltered = build_candidates(rows, trait_columns, AVAILABLE_PAPER_IDS, max_gold_values=10**9)

    by_paper = build_candidates(rows, trait_columns, AVAILABLE_PAPER_IDS, max_gold_values=args.max_gold_values)
    by_paper = filter_by_min_coverage(by_paper, args.min_field_coverage)
    n_candidates = sum(len(fields) for fields in by_paper.values())
    print(f"[INFO] {n_candidates} candidate (paper, field) items across {len(by_paper)} papers "
          f"(of {len(AVAILABLE_PAPER_IDS)} available).")

    n_single = args.n_positive - args.n_combo
    single_items = sample_positive_items(by_paper, n_single, args.seed, max_items_per_field=args.max_items_per_field)
    combo_items = sample_combo_items(by_paper, args.n_combo, args.seed)

    negatives_by_paper = build_negative_candidates(by_paper_unfiltered, trait_columns, AVAILABLE_PAPER_IDS)
    negative_items = sample_negative_items(negatives_by_paper, args.n_negative, args.seed)

    all_items = single_items + combo_items + negative_items
    for i, it in enumerate(all_items):
        it["item_id"] = i

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_items": len(all_items),
        "n_single": len(single_items),
        "n_combo": len(combo_items),
        "n_negative": len(negative_items),
        "seed": args.seed,
        "available_paper_ids": AVAILABLE_PAPER_IDS,
        "excluded_non_trait_fields": sorted(EXCLUDED_NON_TRAIT_FIELDS),
        "question_templates": QUESTION_TEMPLATES,
        "combo_question_templates": COMBO_QUESTION_TEMPLATES,
        "items": all_items,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    papers_touched = sorted({it["paper_id"] for it in all_items})
    print(f"[DONE] Wrote {len(all_items)} items ({len(single_items)} single + {len(combo_items)} combo + "
          f"{len(negative_items)} negative) covering {len(papers_touched)} papers to {out_path}")
    print(f"       Papers covered: {papers_touched}")


if __name__ == "__main__":
    main()
