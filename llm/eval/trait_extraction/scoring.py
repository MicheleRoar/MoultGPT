#!/usr/bin/env python3
# llm/eval/trait_extraction/scoring.py
"""
Parsing and matching utilities for run_model_comparison.py, split into
their own module so they're independently unit-testable (see
llm/tests/test_trait_extraction_scoring.py) without needing network
access, an API key, or the xlsx/tei.xml corpus -- unlike the rest of this
package, everything here is pure string logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - PyYAML is in llm/requirements.txt
    yaml = None  # noqa: N816

ABSTAIN_PATTERNS = [
    r"\bnot mentioned\b",
    r"\bnot stated\b",
    r"\bnot specified\b",
    r"\bnot reported\b",
    r"\bnot available\b",
    r"\bnot provided\b",
    r"\bnot given\b",
    r"\bnot found\b",
    r"\bno information\b",
    r"\bno evidence\b",
    r"\bcannot be determined\b",
    r"\bcan.?t be determined\b",
    r"\bunable to determine\b",
    r"\bunclear\b",
    r"\bunknown\b",
    r"^n\s*/\s*a$",
    r"^na$",
    r"^none$",
    r"^-+$",
]
_ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.IGNORECASE)

NA_LIKE_VALUES = {"", "na", "n/a", "unknown", "?", "none", "nan", "null"}


def is_abstention(raw_answer: str) -> bool:
    """
    True if the model's raw text is best read as "I don't know / not
    stated", rather than an actual attempted value. This is a heuristic
    regex match, not a certainty -- borderline cases are exactly the kind
    of thing the manual/LLM-judge review pass (see run_model_comparison.py)
    exists to catch, not something this function is expected to get
    perfectly right on its own.
    """
    s = (raw_answer or "").strip()
    if not s or s.lower() in NA_LIKE_VALUES:
        return True
    return bool(_ABSTAIN_RE.search(s))


def _strip_hash_from_yaml_keys(text: str) -> str:
    """
    single_trait mode doesn't pin down a YAML key name, so the model
    commonly echoes the MoultDB field name verbatim as its key. Several
    real field names contain a literal '#' (e.g. "Observed # total
    moult stages", "Estimated # moult stages", "# body segments in
    adult individuals"). YAML treats a '#' preceded by whitespace as
    the start of a comment, so a line like
    "Observed # total moult stages: Not mentioned" parses not as a
    mapping but as the bare scalar "Observed" -- everything from " #"
    onward, INCLUDING the real value after the colon, is discarded as
    a comment.

    This was found by tracing a real, reproducible pattern in an actual
    run's output: every item tied to a '#'-containing field name came
    back as the single truncated word "Observed" or "Estimated" across
    every model (small/medium/large) -- not a coincidence, and not a
    model reasoning failure. The LLM judge scored these as "incorrect"
    without knowing it was judging a parsing artifact rather than the
    model's real answer (which, in the cases checked, was actually an
    abstention like "not mentioned" / "null" -- so the fix mostly
    reclassifies bogus "disagreement/incorrect" rows as "abstained",
    not "correct"; it removes a false hallucination signal, it doesn't
    manufacture accuracy that wasn't there).

    Fix: strip '#' characters that appear in the KEY portion (text
    before the first ':' on a line) -- a '#' in a MoultDB field name
    carries no information worth preserving. Separately, if the VALUE
    portion (after the colon) contains a '#' that YAML would itself
    read as a comment marker, quote the whole value so YAML treats it
    as a literal string instead of truncating it there too.
    """
    fixed_lines = []
    for line in text.split("\n"):
        colon_idx = line.find(":")
        if colon_idx == -1:
            fixed_lines.append(line)
            continue
        key_part, value_part = line[:colon_idx], line[colon_idx + 1:]
        key_part = key_part.replace("#", "")
        if re.search(r"(^|\s)#", value_part):
            value_part = ' "' + value_part.strip().replace('"', '\\"') + '"'
        fixed_lines.append(key_part + ":" + value_part)
    return "\n".join(fixed_lines)


def extract_yaml_value(raw_text: str) -> str:
    """
    single_trait mode (pipeline/prompting.py) asks the model for "CLEAN
    YAML only" but does not pin down a specific key name, since the
    question changes every time. Rather than assuming a fixed schema,
    this tries, in order:
      1. Strip a ```yaml ... ``` / ``` ... ``` code fence if present.
      2. Strip stray '#' characters from the key portion only (see
         `_strip_hash_from_yaml_keys`) -- otherwise field names like
         "Observed # total moult stages" get comment-truncated by the
         YAML parser before we ever see the value.
      3. Parse as YAML. If the result is a dict, take the first value
         (whatever key the model chose to use). If it's a plain scalar
         (a bare string/number), use it directly -- this is the common
         case, since "the trait's value" often isn't naturally a mapping.
      4. If YAML parsing fails outright (model didn't actually return
         valid YAML despite being asked to), fall back to the raw
         stripped text -- still scored, not silently dropped, so a
         formatting failure shows up as a real (likely "incorrect" or
         "abstain") data point rather than disappearing from the results.
    """
    text = (raw_text or "").strip()
    fence_match = re.match(r"^```(?:yaml)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    if not text:
        return ""

    text = _strip_hash_from_yaml_keys(text)

    if yaml is not None:
        try:
            parsed = yaml.safe_load(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and parsed:
            first_value = next(iter(parsed.values()))
            if isinstance(first_value, (list, tuple)):
                return ", ".join(str(v) for v in first_value)
            return "" if first_value is None else str(first_value).strip()
        if isinstance(parsed, (str, int, float)):
            return str(parsed).strip()
        if isinstance(parsed, list):
            return ", ".join(str(v) for v in parsed)

    return text


_BRITISH_MOULT_RE = re.compile(r"\bmoult(ing|ed|s)?\b")


def normalize_value(value: str) -> str:
    v = (value or "").strip().lower()
    v = v.replace("_", " ").replace("-", " ")
    # British/American spelling: this whole domain is "moulting", and
    # models mix both spellings freely (confirmed in real output --
    # "molting_en_masse" vs gold "mass moulting"). Normalize to the
    # American form on both sides so the same word doesn't count as a
    # mismatch purely for spelling.
    v = _BRITISH_MOULT_RE.sub(lambda m: "molt" + (m.group(1) or ""), v)
    v = re.sub(r"\s+", " ", v)
    v = v.strip(" .;:")
    return v


# Tokens that mean "affirmative"/"negative" for boolean-style fields.
# Real gap found in run output: models sometimes answer a yes/no MoultDB
# field with the literal word "True"/"False" (Python-flavored) instead of
# "yes"/"no", which previously scored as a hard mismatch even though it's
# the same answer.
_AFFIRMATIVE_TOKENS = {"true", "yes", "y"}
_NEGATIVE_TOKENS = {"false", "no", "n"}

_STOPWORDS_FOR_OVERLAP = {"the", "a", "an", "of", "in", "on", "at", "to", "for",
                           "and", "or", "is", "are", "with", "as", "by"}


def _bag_of_words_overlap(pred_norm: str, gold_norm: str) -> bool:
    """
    Last-resort fallback: treat predicted/gold as sets of significant
    words and match if gold's words are (almost) fully covered by
    predicted's words, regardless of order -- catches reordered/
    reworded phrases like predicted "en masse molting" vs gold "mass
    moulting" partially, and fully catches simple reorderings like
    "dorsal cephalic suture" vs "cephalic suture dorsal". Deliberately
    requires most of gold's words to be covered (not just any overlap)
    so this doesn't turn into a near-meaningless fuzzy match -- a
    single shared common word between two otherwise-unrelated phrases
    should NOT count as a match.
    """
    gold_tokens = {t for t in gold_norm.split() if t not in _STOPWORDS_FOR_OVERLAP}
    pred_tokens = {t for t in pred_norm.split() if t not in _STOPWORDS_FOR_OVERLAP}
    if len(gold_tokens) < 2 or not pred_tokens:
        return False  # too short to safely fuzzy-match without false positives
    overlap = gold_tokens & pred_tokens
    return len(overlap) / len(gold_tokens) >= 0.8


# Small-number word forms only -- moulting trait counts (# instars, #
# body segments, # moult stages) are, in every real MoultDB field this
# eval samples, well under 100. Not a general-purpose text2num library;
# just enough to stop "seventeen" != "17" from counting as a wrong
# answer when it's the same number spelled out.
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _word_to_number(text: str) -> Optional[int]:
    """
    Parses a spelled-out small integer ("seventeen", "twenty-three",
    "twenty three") into an int, or returns None if `text` isn't
    (purely) a number word. Deliberately conservative -- if there's any
    token that isn't part of a number word, this returns None rather
    than guessing at a partial match.
    """
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return None
    total = 0
    matched_any = False
    i = 0
    while i < len(words):
        w = words[i]
        if w == "and":
            i += 1
            continue
        if w in _TENS:
            total += _TENS[w]
            matched_any = True
        elif w in _ONES:
            total += _ONES[w]
            matched_any = True
        else:
            return None  # a non-number-word token -> not a pure number-word value
        i += 1
    return total if matched_any else None


def _numbers_equivalent(a: str, b: str) -> bool:
    """
    True if `a` and `b` denote the same number, whether written as a
    digit string ("17") or spelled out ("seventeen") -- in either
    direction, since either the gold value or the prediction could be
    in either form.
    """
    def as_number(s: str) -> Optional[float]:
        s = s.strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return float(s)
        word_val = _word_to_number(s)
        return float(word_val) if word_val is not None else None

    na, nb = as_number(a), as_number(b)
    return na is not None and nb is not None and na == nb


def values_match(predicted: str, gold_values: List[str]) -> bool:
    """
    Tolerant comparison: exact match after normalization, OR predicted
    text contains one gold value as a substring (handles the model
    answering in a slightly longer phrase, e.g. predicted="the cephalic
    suture" vs gold="cephalic suture"), OR the two denote the same
    number in different forms ("seventeen" vs "17"), OR (for comma/
    semicolon-separated multi-value gold entries) any individual gold
    token matches under the same rules.
    """
    pred_norm = normalize_value(predicted)
    if not pred_norm:
        return False

    for gold in gold_values:
        gold_norm = normalize_value(gold)
        if not gold_norm:
            continue
        if pred_norm == gold_norm:
            return True
        if gold_norm in pred_norm or pred_norm in gold_norm:
            return True
        if _numbers_equivalent(pred_norm, gold_norm):
            return True
        if pred_norm in _AFFIRMATIVE_TOKENS and gold_norm in _AFFIRMATIVE_TOKENS:
            return True
        if pred_norm in _NEGATIVE_TOKENS and gold_norm in _NEGATIVE_TOKENS:
            return True
        if _bag_of_words_overlap(pred_norm, gold_norm):
            return True
        # multi-value gold cell, e.g. "dorsal; cephalothoracic joint"
        for part in re.split(r"[;,]", gold_norm):
            part = part.strip()
            if part and (part == pred_norm or part in pred_norm or _numbers_equivalent(pred_norm, part)
                          or _bag_of_words_overlap(pred_norm, part)):
                return True
    return False


def classify_prediction(predicted_raw: str, gold_values: List[str]) -> Tuple[str, str]:
    """
    Returns (label, cleaned_predicted_value), where label is one of:
      "abstained" -- model declined / said not stated
      "correct"   -- matches a gold value (tolerant match)
      "disagreement" -- model gave a value, but it doesn't match gold;
                        NOT the same as "incorrect" -- see
                        run_model_comparison.py's judge pass, which is
                        what actually decides incorrect vs ambiguous.
    """
    cleaned = extract_yaml_value(predicted_raw)
    if is_abstention(cleaned):
        return "abstained", cleaned
    if values_match(cleaned, gold_values):
        return "correct", cleaned
    return "disagreement", cleaned


# ── Grouped (per-paper) single-trait prompting ──────────────────────────
#
# The original design asked ONE (paper, field) question per API call:
# 37 non-gated items x 3 models = 111 calls for the main condition alone,
# before baselines/judge -- this volume is what actually tripped the free
# tier's rate limit on mistral-large-latest (26/37 calls errored in a
# real run), not the complexity of any single question.
#
# This groups all of one paper's sampled questions into a SINGLE call per
# (paper, model), each question still asked narrowly and independently
# (not the deployed full_traits mode's "give me all 55 fields" prompt,
# which was deliberately avoided -- see run_model_comparison.py's module
# docstring for why). Answers come back as "<question number>: <value>"
# lines -- using the question's 1-based INDEX as the key, not the field
# name, is deliberate: it's what stops the model from ever echoing a
# field name (like "Observed # total moult stages") back into its own
# answer format, which is exactly what caused the '#'-in-fieldname YAML
# parsing bug this module already works around elsewhere. Grouping by
# paper also means the same evidence context is reused across all of a
# paper's questions in one shot, rather than re-sent per question.

GROUPED_SYSTEM_PROMPT = (
    "You are a scientific assistant specialized in arthropod moulting.\n"
    "You will be given evidence sentences extracted from a paper, and a "
    "numbered list of independent questions, each asking about a single "
    "moulting-related trait.\n"
    "Answer EACH question independently, using ONLY the evidence provided. "
    "Do not let your answer to one question influence another, and do not "
    "infer or guess beyond what the evidence explicitly supports.\n"
    "If the evidence does not support an answer to a question, answer "
    "exactly \"Not mentioned\" for that question -- do not omit the line.\n"
    "Output EXACTLY one line per question, in this format and nothing "
    "else -- no YAML, no markdown, no repeating the question or field "
    "name, no extra commentary:\n"
    "<question number>: <answer>\n"
)


def build_grouped_user_content(summary: str, questions: List[str]) -> str:
    """
    `questions` is an ordered list of already-phrased question strings
    (see gold_questions.py's humanize_question) for one paper; the
    returned index (1-based, matching list order) is what
    parse_grouped_response uses to map answers back to items -- the
    field name itself is never part of the expected answer format.
    """
    lines = [f"Context:\n{summary.strip()}\n", "Questions:"]
    for idx, question in enumerate(questions, start=1):
        lines.append(f"{idx}. {question}")
    return "\n".join(lines)


def parse_grouped_response(raw_text: str, n_questions: int) -> Dict[int, str]:
    """
    Maps a grouped response's "<number>: <answer>" lines back to 1-based
    question indices. Tolerant of minor formatting noise (a stray
    leading '-'/'*' bullet, a period or ')' instead of ':' after the
    number) but does NOT try to recover a line that's missing entirely --
    a missing index simply isn't a key in the returned dict, and the
    caller (run_model_comparison.py) records that specific item as a
    parse failure rather than silently guessing or dropping it.
    """
    answers: Dict[int, str] = {}
    for line in (raw_text or "").splitlines():
        line = line.strip().lstrip("-*").strip()
        m = re.match(r"^(\d+)\s*[.):]\s*(.*)$", line)
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= n_questions:
            answers[idx] = m.group(2).strip()
    return answers
