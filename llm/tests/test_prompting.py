# llm/tests/test_prompting.py
"""
Unit tests for pipeline/prompting.py — the single prompt-construction
module shared by the live backend (backend/app.py) and the benchmarking
script (eval/compare_models.py). Any regression here would silently change
what every model is asked, in a way that's easy to miss without a test
that pins the observable contract (which mode returns which format).
"""

import pytest

from pipeline.prompting import (
    MODE_FULL_TRAITS,
    MODE_SINGLE_TRAIT,
    VALID_MODES,
    build_system_prompt,
    build_user_content,
)


def test_valid_modes_contains_both_modes():
    assert set(VALID_MODES) == {MODE_SINGLE_TRAIT, MODE_FULL_TRAITS}


def test_single_trait_system_prompt_demands_yaml_only():
    prompt = build_system_prompt(MODE_SINGLE_TRAIT)
    assert "YAML" in prompt
    assert "Extract ONLY the trait" in prompt


def test_full_traits_system_prompt_includes_trait_schema():
    prompt = build_system_prompt(MODE_FULL_TRAITS)
    # The 55-field MoultDB schema block must actually be injected — that's
    # the entire point of full_traits mode vs. single_trait.
    assert "Fields:" in prompt
    assert "MoultDB" in prompt


def test_build_system_prompt_defaults_to_full_traits_for_unknown_mode():
    # build_system_prompt() falls back to full_traits for anything that
    # isn't exactly "single_trait" — pin that behaviour explicitly.
    assert build_system_prompt("not_a_real_mode") == build_system_prompt(MODE_FULL_TRAITS)


def test_build_user_content_single_trait_includes_context_and_query():
    content = build_user_content("summary sentence", "how many instars?", MODE_SINGLE_TRAIT)
    assert "summary sentence" in content
    assert "how many instars?" in content
    assert "Context:" in content


def test_build_user_content_full_traits_labels_query_as_optional_focus():
    content = build_user_content("summary sentence", "how many instars?", MODE_FULL_TRAITS)
    assert "summary sentence" in content
    assert "Optional user focus" in content


@pytest.mark.parametrize("mode", [MODE_SINGLE_TRAIT, MODE_FULL_TRAITS])
def test_build_user_content_strips_whitespace(mode):
    content = build_user_content("  padded summary  \n", "  padded query  ", mode)
    assert "  padded summary  " not in content
    assert "padded summary" in content
