# llm/tests/test_domain_gate.py
"""
Unit tests for domain/domain_gate.py — the combined "is this request in
scope for MoultGPT" decision that gates every /query call before it is
allowed to reach a (paid, remote) LLM provider.

Real TaxonomyLookup / MoultingOntologyGate instances load an OWL ontology
and a taxonomy CSV, so tests use minimal fakes exposing just the two
methods domain_gate.py actually calls on them. This keeps the suite fast
and dependency-free while still covering every branch of the combined
decision table.
"""

from domain.domain_gate import (
    _find_non_arthropod_query_hits,
    analyze_paper_and_query_domain,
)


class FakeTaxonomyLookup:
    def __init__(self, has_signal: bool):
        self._has_signal = has_signal

    def summarize_taxonomic_signal(self, paper_text):
        return {
            "has_taxonomic_signal": self._has_signal,
            "n_direct_matches": 1 if self._has_signal else 0,
            "n_propagated_matches": 0,
            "direct_matches": ["Insecta"] if self._has_signal else [],
            "propagated_matches": [],
        }


class FakeOntologyGate:
    def __init__(self, allow: bool):
        self._allow = allow

    def analyze_text(self, text, min_hits=1, min_score=2.5):
        return {
            "allow": self._allow,
            "n_hits": 3 if self._allow else 0,
            "score": 5.0 if self._allow else 0.0,
            "hits": ["moult", "instar"] if self._allow else [],
            "summary_hits": [],
        }


SUFFICIENT_SUMMARY = "\n".join([f"Sentence {i} about moulting." for i in range(6)])
INSUFFICIENT_SUMMARY = "Only one sentence."


def test_in_scope_when_everything_checks_out():
    result = analyze_paper_and_query_domain(
        paper_text="An arthropod paper.",
        user_query="How many instars before the final moult?",
        taxonomy_lookup=FakeTaxonomyLookup(has_signal=True),
        ontology_gate=FakeOntologyGate(allow=True),
        summary_text=SUFFICIENT_SUMMARY,
    )
    assert result["allow"] is True
    assert result["final_label"] == "in_scope"


def test_paper_out_of_scope_when_no_taxonomic_signal():
    result = analyze_paper_and_query_domain(
        paper_text="A paper about something else entirely.",
        user_query="How many instars before the final moult?",
        taxonomy_lookup=FakeTaxonomyLookup(has_signal=False),
        ontology_gate=FakeOntologyGate(allow=True),
        summary_text=SUFFICIENT_SUMMARY,
    )
    assert result["allow"] is False
    assert result["final_label"] == "paper_out_of_scope"


def test_paper_not_relevant_when_summary_too_short():
    result = analyze_paper_and_query_domain(
        paper_text="An arthropod paper.",
        user_query="How many instars before the final moult?",
        taxonomy_lookup=FakeTaxonomyLookup(has_signal=True),
        ontology_gate=FakeOntologyGate(allow=True),
        summary_text=INSUFFICIENT_SUMMARY,
    )
    assert result["allow"] is False
    assert result["final_label"] == "paper_not_relevant"


def test_query_out_of_scope_when_query_not_about_moulting():
    result = analyze_paper_and_query_domain(
        paper_text="An arthropod paper.",
        user_query="What is the capital of France?",
        taxonomy_lookup=FakeTaxonomyLookup(has_signal=True),
        ontology_gate=FakeOntologyGate(allow=False),
        summary_text=SUFFICIENT_SUMMARY,
    )
    assert result["allow"] is False
    assert result["final_label"] == "query_out_of_scope"


def test_non_arthropod_query_overrides_everything_else():
    # Even if the paper is a perfect arthropod match, a query explicitly
    # about e.g. bird moulting must still be rejected.
    result = analyze_paper_and_query_domain(
        paper_text="An arthropod paper.",
        user_query="Describe feather moulting in birds.",
        taxonomy_lookup=FakeTaxonomyLookup(has_signal=True),
        ontology_gate=FakeOntologyGate(allow=True),
        summary_text=SUFFICIENT_SUMMARY,
    )
    assert result["allow"] is False
    assert result["final_label"] == "query_out_of_scope"
    assert "bird" in result["query_gate"]["non_arthropod_hits"]


def test_find_non_arthropod_query_hits_is_case_insensitive():
    # "snake" is a substring of "snakes", so both terms match — the function
    # does substring matching on purpose (simple and fast), not word-boundary
    # matching, so this test also documents that behaviour.
    hits = _find_non_arthropod_query_hits("Do Snakes shed their SKIN like arthropods?")
    assert set(hits) == {"snake", "snakes", "skin"}


def test_find_non_arthropod_query_hits_empty_when_no_match():
    assert _find_non_arthropod_query_hits("How many instars does this beetle have?") == []
