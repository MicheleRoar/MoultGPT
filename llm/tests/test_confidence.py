# llm/tests/test_confidence.py
"""
Unit tests for pipeline/confidence.py -- pure string logic, no
network/API key/corpus needed, same dependency-free bar as the rest of
llm/tests/.
"""

from pipeline.confidence import confidence_label, score_confidence


def test_exact_phrase_match_scores_high():
    result = score_confidence("dorsal cephalothoracic joint",
                               "The suture ruptures at the dorsal cephalothoracic joint during ecdysis.")
    assert result.exact_phrase is True
    assert result.score >= 0.6  # at least the exact-phrase weight alone


def test_no_overlap_scores_zero():
    result = score_confidence("marine", "The species was collected from a freshwater lake in Bavaria.")
    assert result.exact_phrase is False
    assert result.score == 0.0


def test_partial_token_overlap_scores_between_zero_and_exact():
    result = score_confidence("terrestrial isopod habitat",
                               "This terrestrial species was observed in leaf litter habitat.")
    assert result.exact_phrase is False
    assert 0.0 < result.score < 1.0


def test_empty_value_scores_zero():
    result = score_confidence("", "Some evidence text here.")
    assert result.score == 0.0


def test_empty_evidence_scores_zero():
    result = score_confidence("marine", "")
    assert result.score == 0.0


def test_numeric_value_falls_back_to_substring_check():
    result = score_confidence("4", "The species undergoes 4 juvenile moults before reaching adulthood.")
    assert result.token_overlap == 1.0


def test_numeric_value_not_present_scores_zero_overlap():
    result = score_confidence("7", "The species undergoes 4 juvenile moults before reaching adulthood.")
    assert result.token_overlap == 0.0


def test_confidence_label_buckets():
    assert confidence_label(0.9) == "high"
    assert confidence_label(0.75) == "high"
    assert confidence_label(0.5) == "medium"
    assert confidence_label(0.4) == "medium"
    assert confidence_label(0.1) == "low"
    assert confidence_label(0.0) == "low"


def test_score_is_bounded_zero_to_one():
    result = score_confidence("dorsal cephalothoracic joint dorsal cephalothoracic joint",
                               "dorsal cephalothoracic joint " * 20)
    assert 0.0 <= result.score <= 1.0
