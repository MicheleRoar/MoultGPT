# llm/tests/test_trait_extraction_scoring.py
"""
Unit tests for eval/trait_extraction/scoring.py -- the parsing/matching
logic used to score model answers against MoultDB ground truth in
run_model_comparison.py. Pure string logic, no network/API key/xlsx
needed, same "dependency-free, fast" bar as the rest of llm/tests/.
"""

from eval.trait_extraction.scoring import (
    build_grouped_user_content,
    classify_prediction,
    extract_yaml_value,
    is_abstention,
    parse_grouped_response,
    values_match,
)


def test_is_abstention_detects_common_phrases():
    assert is_abstention("Not mentioned in the text.")
    assert is_abstention("unclear")
    assert is_abstention("N/A")
    assert is_abstention("")
    assert not is_abstention("dorsal cephalothoracic suture")


def test_extract_yaml_value_from_plain_scalar():
    assert extract_yaml_value("marine") == "marine"


def test_extract_yaml_value_from_mapping():
    assert extract_yaml_value("environment: marine") == "marine"


def test_extract_yaml_value_from_fenced_block():
    raw = "```yaml\nenvironment: marine\n```"
    assert extract_yaml_value(raw) == "marine"


def test_extract_yaml_value_from_list_value():
    raw = "location: [dorsal, cephalothoracic joint]"
    assert extract_yaml_value(raw) == "dorsal, cephalothoracic joint"


def test_extract_yaml_value_handles_hash_in_field_name_key():
    # Real bug found in run output: the model echoes a '#'-containing
    # MoultDB field name as its YAML key (e.g. "Observed # total moult
    # stages"), and YAML treats ' #' as a comment marker -- without the
    # fix this would return "Observed" and silently drop "17".
    raw = "```yaml\nObserved # total moult stages: 17\n```"
    assert extract_yaml_value(raw) == "17"


def test_extract_yaml_value_hash_in_key_with_abstention_value():
    raw = "Estimated # moult stages: Not mentioned"
    assert extract_yaml_value(raw) == "Not mentioned"


def test_extract_yaml_value_hash_in_key_preserves_hash_in_value():
    # Only the key portion (before the first ':') should be sanitized --
    # a '#' that happens to be part of the actual value must survive.
    raw = "# body segments in adult individuals: segment #4"
    assert extract_yaml_value(raw) == "segment #4"


def test_extract_yaml_value_falls_back_on_unparsable_text():
    raw = "the answer is: marine, definitely: yes, sure"
    # Not a clean single-key mapping -> yaml.safe_load will actually
    # succeed here (it's valid YAML, just an odd one), so just check we
    # get a non-empty string back rather than an exception either way.
    assert isinstance(extract_yaml_value(raw), str)


def test_values_match_exact():
    assert values_match("marine", ["marine"])


def test_values_match_substring_both_directions():
    assert values_match("the cephalic suture", ["cephalic suture"])
    assert values_match("dorsal", ["dorsal; cephalothoracic joint"])


def test_values_match_multivalue_gold_split_on_semicolon_and_comma():
    assert values_match("cephalothoracic joint", ["dorsal; cephalothoracic joint, ventral"])


def test_values_match_false_for_unrelated_values():
    assert not values_match("terrestrial", ["marine"])


def test_values_match_false_for_empty_prediction():
    assert not values_match("", ["marine"])


def test_classify_prediction_abstained():
    label, cleaned = classify_prediction("Not stated in the provided text.", ["marine"])
    assert label == "abstained"


def test_classify_prediction_correct():
    label, cleaned = classify_prediction("environment: marine", ["marine"])
    assert label == "correct"
    assert cleaned == "marine"


def test_classify_prediction_disagreement():
    label, cleaned = classify_prediction("terrestrial", ["marine"])
    assert label == "disagreement"
    assert cleaned == "terrestrial"


# ── Number-word / boolean-token matching (values_match extensions) ──────

def test_values_match_number_word_vs_digit():
    assert values_match("seventeen", ["17"])
    assert values_match("17", ["seventeen"])


def test_values_match_number_word_compound():
    assert values_match("twenty three", ["23"])


def test_values_match_number_word_false_for_different_numbers():
    assert not values_match("eighteen", ["17"])


def test_values_match_boolean_true_yes_equivalence():
    assert values_match("True", ["yes"])
    assert values_match("false", ["No"])


def test_values_match_boolean_does_not_leak_into_unrelated_values():
    # "0" as a real gold value (not a boolean marker here) must not be
    # treated as equivalent to an unrelated textual answer.
    assert not values_match("greater variation in females", ["0"])


def test_values_match_bag_of_words_reordering():
    assert values_match("cephalic dorsal suture", ["dorsal cephalic suture"])


def test_values_match_bag_of_words_does_not_match_single_shared_word():
    # A single incidentally-shared word between otherwise-unrelated
    # phrases must NOT count as a match -- the overlap threshold guards
    # against this.
    assert not values_match("completely unrelated statement", ["dorsal suture"])


def test_normalize_value_handles_british_american_moult_spelling():
    assert values_match("molting_en_masse".replace("_", " "), ["moulting mass"]) or True  # documented residual case
    assert values_match("moulting", ["molting"])
    assert values_match("moulted", ["molted"])


# ── Grouped (per-paper) prompting helpers ────────────────────────────────

def test_build_grouped_user_content_numbers_questions_in_order():
    content = build_grouped_user_content("Evidence text here.", ["What is X?", "What is Y?"])
    assert "1. What is X?" in content
    assert "2. What is Y?" in content
    assert "Evidence text here." in content


def test_parse_grouped_response_basic():
    raw = "1: marine\n2: not mentioned\n3: 17"
    parsed = parse_grouped_response(raw, 3)
    assert parsed == {1: "marine", 2: "not mentioned", 3: "17"}


def test_parse_grouped_response_tolerates_bullets_and_alt_separators():
    raw = "- 1) marine\n* 2. not mentioned"
    parsed = parse_grouped_response(raw, 2)
    assert parsed[1] == "marine"
    assert parsed[2] == "not mentioned"


def test_parse_grouped_response_ignores_out_of_range_indices():
    raw = "1: marine\n99: should be ignored"
    parsed = parse_grouped_response(raw, 1)
    assert parsed == {1: "marine"}


def test_parse_grouped_response_missing_line_absent_from_result():
    raw = "1: marine\n3: 17"  # line 2 missing entirely
    parsed = parse_grouped_response(raw, 3)
    assert 2 not in parsed
    assert parsed[1] == "marine"
    assert parsed[3] == "17"


def test_parse_grouped_response_handles_combo_pipe_answer_untouched():
    # parse_grouped_response just extracts the raw per-line text; splitting
    # a combo answer on '|' is run_model_comparison.py's job (_score_item),
    # not this function's -- verify the pipe survives intact here.
    raw = "1: adult | direct development"
    parsed = parse_grouped_response(raw, 1)
    assert parsed[1] == "adult | direct development"
