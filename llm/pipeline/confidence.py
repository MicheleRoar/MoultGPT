# llm/pipeline/confidence.py
"""
Post-hoc confidence scoring for extracted trait values, filling in what
was previously a placeholder in the paper (and, before that, code that
didn't exist at all -- see this repo's development notes: there was no
confidence-scoring implementation anywhere in llm/ prior to this module).

What this IS
-------------
A heuristic estimate of how well an extracted `field: value` line is
textually grounded in the evidence sentences the model was actually
shown (the selected-sentence summary from pipeline/summarization.py).
Combines:
  1. Exact-phrase groundedness: does the (normalized) value appear
     verbatim as a substring of the evidence text?
  2. Token-overlap groundedness: what fraction of the value's own
     content words (stopwords excluded) also appear somewhere in the
     evidence text?
This answers "does this answer look like it was lifted from the
evidence, or does it look invented" -- a real, if narrow, question. It
does NOT require ground truth, which is what makes it usable in
production (unlike the accuracy scoring in eval/trait_extraction/, which
needs MoultDB annotations to compare against).

What this is NOT
------------------
- Not a probability. Nothing here is calibrated against how often a
  "high confidence" answer is actually correct -- that calibration
  question is exactly what eval/trait_extraction/'s gold comparison is
  for, and the two are complementary, not the same thing.
- Not a hallucination detector in the strong sense. A model can echo
  evidence text while still misinterpreting it (e.g. negating a
  statement, misattributing a value to the wrong entity) and still score
  high here, because this only checks lexical presence, not semantic
  correctness. This limitation is stated in the paper, not hidden.
- Not the same computation as eval/trait_extraction/scoring.py's
  `values_match` -- that function checks a predicted value against a
  KNOWN CORRECT gold value; this module checks a predicted value against
  the EVIDENCE TEXT it was generated from, which is available in
  production even when no gold value exists. They happen to share the
  same "normalize + tolerant substring/token overlap" flavor of string
  matching because that flavor is the right tool for both jobs, not
  because they're doing the same comparison.

Usage
-----
    from pipeline.confidence import score_confidence

    result = score_confidence("cephalothoracic joint; dorsal", evidence_text)
    result.score        # float in [0, 1]
    result.exact_phrase  # bool
    result.token_overlap # float in [0, 1]
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Small, deliberately short stopword list -- just enough to stop function
# words from inflating token overlap (e.g. "the", "of", "and" appearing
# in both a predicted value and any evidence text, regardless of topic).
# Not a general-purpose NLP stopword list; domain terms are all kept.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "with", "as",
    "by", "this", "that", "these", "those", "it", "its", "from",
}

# Weights for combining the two signals into one score. Exact-phrase
# match is weighted higher because it's a stronger signal (the model
# reproduced a specific phrase from the evidence) than partial token
# overlap (which can happen somewhat by chance on short values).
EXACT_PHRASE_WEIGHT = 0.6
TOKEN_OVERLAP_WEIGHT = 0.4


@dataclass
class ConfidenceResult:
    score: float
    exact_phrase: bool
    token_overlap: float
    value_tokens: int
    matched_tokens: int


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _content_tokens(text: str) -> set:
    normalized = _normalize(text)
    words = re.findall(r"[a-z0-9]+", normalized)
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def score_confidence(value: str, evidence_text: str) -> ConfidenceResult:
    """
    Scores how well `value` (an extracted field's value, e.g. "dorsal
    cephalothoracic joint") is textually grounded in `evidence_text` (the
    selected-sentence summary the model was shown when producing it).

    Returns a ConfidenceResult with score in [0, 1]; 0.0 for an empty
    value (nothing to ground) or empty evidence (nothing to ground it
    in).
    """
    value_norm = _normalize(value)
    evidence_norm = _normalize(evidence_text)

    if not value_norm or not evidence_norm:
        return ConfidenceResult(score=0.0, exact_phrase=False, token_overlap=0.0,
                                 value_tokens=0, matched_tokens=0)

    exact_phrase = value_norm in evidence_norm

    value_tokens = _content_tokens(value)
    evidence_tokens = _content_tokens(evidence_text)
    if value_tokens:
        matched = value_tokens & evidence_tokens
        token_overlap = len(matched) / len(value_tokens)
    else:
        # Value had no content tokens at all (e.g. purely numeric, "4"):
        # fall back to whether that literal appears in the evidence, so
        # a numeric value isn't unfairly scored 0 just for lacking words.
        matched = set()
        token_overlap = 1.0 if value_norm in evidence_norm else 0.0

    score = EXACT_PHRASE_WEIGHT * (1.0 if exact_phrase else 0.0) + TOKEN_OVERLAP_WEIGHT * token_overlap
    score = round(min(1.0, max(0.0, score)), 4)

    return ConfidenceResult(
        score=score,
        exact_phrase=exact_phrase,
        token_overlap=round(token_overlap, 4),
        value_tokens=len(value_tokens),
        matched_tokens=len(matched),
    )


def confidence_label(score: float) -> str:
    """Coarse bucket for display purposes (e.g. a UI badge) -- thresholds
    are round numbers chosen for interpretability, not fit to data."""
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"
