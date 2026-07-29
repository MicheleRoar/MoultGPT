# llm/tests/test_gold_questions.py
"""
Unit tests for eval/trait_extraction/gold_questions.py's sampling logic,
using small synthetic by_paper dicts -- no real xlsx needed, same
dependency-free bar as the rest of llm/tests/. Covers the scale-up from
a single fixed-phrasing 50-item positive-only sample to a mixed
positive/negative/combo set with rotating question phrasing.
"""

from eval.trait_extraction.gold_questions import (
    QUESTION_TEMPLATES,
    _field_for_display,
    build_negative_candidates,
    humanize_question,
    sample_combo_items,
    sample_negative_items,
    sample_positive_items,
)


def _toy_by_paper():
    return {
        1: {"Life mode": {"marine"}, "Reproductive state": {"adult"}, "Moulting phase": {"biphasic"}},
        2: {"Life mode": {"terrestrial"}, "Reproductive state": {"juvenile"}},
        3: {"Life mode": {"marine"}, "Moulting phase": {"monophasic"}},
    }


def test_field_for_display_replaces_hash_with_number_of():
    assert _field_for_display("Observed # total moult stages") == "Observed number of total moult stages"
    assert _field_for_display("# body segments in adult individuals") == "number of body segments in adult individuals"
    assert "#" not in _field_for_display("Estimated # moult stages")


def test_field_for_display_leaves_hashless_fields_unchanged():
    assert _field_for_display("Life mode") == "Life mode"


def test_humanize_question_uses_given_template_exactly():
    q = humanize_question("Life mode", template=QUESTION_TEMPLATES[2])
    assert q == QUESTION_TEMPLATES[2].format(field="Life mode")


def test_humanize_question_default_is_first_template():
    assert humanize_question("Life mode") == QUESTION_TEMPLATES[0].format(field="Life mode")


def test_humanize_question_substitutes_hash_field_into_question_text():
    q = humanize_question("Observed # total moult stages", template=QUESTION_TEMPLATES[0])
    assert "#" not in q
    assert "number of" in q


def test_sample_positive_items_one_round_covers_unique_pairs_without_repeats():
    by_paper = _toy_by_paper()
    n_unique_pairs = sum(len(f) for f in by_paper.values())  # 3 + 2 + 2 = 7
    items = sample_positive_items(by_paper, n_items=n_unique_pairs, seed=1, max_items_per_field=10)
    assert len(items) == n_unique_pairs
    pairs = {(it["paper_id"], it["field"]) for it in items}
    assert len(pairs) == n_unique_pairs  # no (paper, field) pair repeated in round 0
    assert all(it["phrasing_round"] == 0 for it in items)
    assert all(it["type"] == "single" and it["is_negative"] is False for it in items)


def test_sample_positive_items_scales_beyond_unique_pairs_via_phrasing_rounds():
    by_paper = _toy_by_paper()
    n_unique_pairs = sum(len(f) for f in by_paper.values())
    n_requested = n_unique_pairs * 2 + 1  # force reuse into round 1 and partway into round 2
    items = sample_positive_items(by_paper, n_items=n_requested, seed=1, max_items_per_field=10)
    assert len(items) == n_requested
    rounds_seen = {it["phrasing_round"] for it in items}
    assert rounds_seen == {0, 1, 2}
    # a (paper, field) pair reused across rounds must get a DIFFERENT phrasing
    round0 = next(it for it in items if it["phrasing_round"] == 0)
    round1_same_pair = next(it for it in items if it["phrasing_round"] == 1
                             and it["paper_id"] == round0["paper_id"] and it["field"] == round0["field"])
    assert round0["question"] != round1_same_pair["question"]


def test_sample_positive_items_caps_beyond_available_phrasings_without_fabricating():
    by_paper = {1: {"Life mode": {"marine"}}}  # exactly one unique pair
    items = sample_positive_items(by_paper, n_items=1000, seed=1, max_items_per_field=10)
    assert len(items) == len(QUESTION_TEMPLATES)  # one item per template, then stops


def test_build_negative_candidates_uses_unfiltered_view():
    by_paper_unfiltered = _toy_by_paper()
    trait_columns = ["Life mode", "Reproductive state", "Moulting phase", "Consumption of exuviae"]
    negatives = build_negative_candidates(by_paper_unfiltered, trait_columns, available_paper_ids=[1, 2, 3])
    # paper 1 has Life mode, Reproductive state, Moulting phase annotated -> only "Consumption of exuviae" missing
    assert negatives[1] == ["Consumption of exuviae"]
    # paper 2 has Life mode, Reproductive state -> missing Moulting phase + Consumption of exuviae
    assert set(negatives[2]) == {"Moulting phase", "Consumption of exuviae"}


def test_sample_negative_items_marks_type_and_empty_gold():
    negatives_by_paper = {1: ["Consumption of exuviae"], 2: ["Moulting phase", "Consumption of exuviae"]}
    items = sample_negative_items(negatives_by_paper, n_items=3, seed=1)
    assert len(items) == 3
    assert all(it["type"] == "negative" and it["is_negative"] is True for it in items)
    assert all(it["gold_values"] == [] for it in items)


def test_sample_negative_items_empty_pool_returns_empty():
    assert sample_negative_items({1: []}, n_items=5, seed=1) == []


def test_sample_combo_items_pairs_two_distinct_fields_same_paper():
    by_paper = _toy_by_paper()
    items = sample_combo_items(by_paper, n_items=2, seed=1)
    assert len(items) == 2
    for it in items:
        assert it["type"] == "combo"
        assert len(it["fields"]) == 2
        assert it["fields"][0] != it["fields"][1]
        assert set(it["gold_values_by_field"].keys()) == set(it["fields"])
        assert "|" not in it["question"]  # the '|' convention is for the ANSWER, not the question


def test_sample_combo_items_skips_papers_with_fewer_than_two_fields():
    by_paper = {1: {"Life mode": {"marine"}}}  # only one field -- no valid combo
    assert sample_combo_items(by_paper, n_items=5, seed=1) == []
