# llm/tests/test_feedback_to_preferences.py
"""
Unit tests for finetuning/feedback_to_preferences.py — the logic that
turns raw 👍/👎 feedback (from the unified demo's feedback UI, via
POST /feedback) into DPO-style {prompt, chosen, rejected} preference pairs.

Pure in-memory data (no real feedback.jsonl needed) so these run in CI
without any training infra — same "dependency-free, fast" bar as the rest
of llm/tests/. This is the one piece of the RAG/DPO additions that's
genuinely unit-testable without a GPU or an API key; train_dpo.py and the
retrieval embedders are exercised for real instead via
llm/retrieval/eval_results.md and manual runs (see llm/finetuning's and
llm/retrieval's README/docstrings for what's actually been executed).
"""

from finetuning.feedback_to_preferences import (
    build_preference_pairs,
    load_feedback,
    normalize_prompt,
    summarize,
)


def test_normalize_prompt_collapses_whitespace_and_case():
    assert normalize_prompt("  How Many   Instars?  ") == "how many instars?"


def test_pair_built_from_one_positive_one_negative_same_prompt():
    entries = [
        {"query": "How many instars?", "response": "Five.", "rating": 1},
        {"query": "How many instars?", "response": "Unclear.", "rating": -1},
    ]
    pairs = build_preference_pairs(entries)
    assert pairs == [{"prompt": "How many instars?", "chosen": "Five.", "rejected": "Unclear."}]


def test_no_pair_when_only_positive_feedback():
    entries = [
        {"query": "How many instars?", "response": "Five.", "rating": 1},
        {"query": "How many instars?", "response": "Also five.", "rating": 1},
    ]
    assert build_preference_pairs(entries) == []


def test_no_pair_when_only_negative_feedback():
    entries = [{"query": "How many instars?", "response": "Unclear.", "rating": -1}]
    assert build_preference_pairs(entries) == []


def test_neutral_and_missing_ratings_are_ignored():
    entries = [
        {"query": "Q", "response": "A", "rating": 0},
        {"query": "Q", "response": "B"},  # no rating key at all
        {"query": "Q", "response": "C", "rating": 1},
    ]
    # Only one signed rating (C) survives filtering -> no opposite side, no pair.
    assert build_preference_pairs(entries) == []


def test_cross_product_within_a_group_excludes_identical_text():
    entries = [
        {"query": "Q", "response": "Good answer", "rating": 1},
        {"query": "Q", "response": "Great answer", "rating": 1},
        {"query": "Q", "response": "Bad answer", "rating": -1},
        {"query": "Q", "response": "Good answer", "rating": -1},  # same text as a chosen -> must not pair with itself
    ]
    pairs = build_preference_pairs(entries)
    pair_tuples = {(p["chosen"], p["rejected"]) for p in pairs}
    assert ("Good answer", "Good answer") not in pair_tuples
    assert ("Good answer", "Bad answer") in pair_tuples
    assert ("Great answer", "Bad answer") in pair_tuples
    assert ("Great answer", "Good answer") in pair_tuples


def test_max_pairs_per_prompt_caps_combinatorial_blowup():
    entries = [{"query": "Q", "response": f"chosen-{i}", "rating": 1} for i in range(10)]
    entries += [{"query": "Q", "response": f"rejected-{i}", "rating": -1} for i in range(10)]
    pairs = build_preference_pairs(entries, max_pairs_per_prompt=3)
    assert len(pairs) == 3


def test_prompt_grouping_is_whitespace_and_case_insensitive():
    entries = [
        {"query": "How many instars?", "response": "Five.", "rating": 1},
        {"query": "  HOW MANY INSTARS?  ", "response": "Unclear.", "rating": -1},
    ]
    pairs = build_preference_pairs(entries)
    assert len(pairs) == 1
    # Output keeps the first-seen original text, not a normalized version.
    assert pairs[0]["prompt"] == "How many instars?"


def test_load_feedback_skips_malformed_lines(tmp_path):
    f = tmp_path / "feedback.jsonl"
    f.write_text(
        '{"query": "Q1", "response": "A1", "rating": 1}\n'
        "not valid json\n"
        '{"query": "Q2", "response": "A2", "rating": -1}\n'
        "\n"  # blank line, also skipped
    )
    entries = load_feedback(f)
    assert len(entries) == 2
    assert entries[0]["query"] == "Q1"
    assert entries[1]["query"] == "Q2"


def test_load_feedback_missing_file_returns_empty_list(tmp_path):
    assert load_feedback(tmp_path / "does_not_exist.jsonl") == []


def test_summarize_reports_one_sided_prompts():
    entries = [
        {"query": "Q1", "response": "A", "rating": 1},
        {"query": "Q1", "response": "B", "rating": -1},
        {"query": "Q2 (one-sided)", "response": "C", "rating": 1},
    ]
    pairs = build_preference_pairs(entries)
    text = summarize(entries, pairs)
    assert "1 prompt(s) have feedback on only one side" in text
    assert "Built 1 preference pair(s)" in text
