from __future__ import annotations

from typing import Dict, Any, Optional, List


NON_ARTHROPOD_QUERY_TERMS = {
    # birds
    "bird", "birds", "feather", "feathers", "plumage",
    # mammals / humans
    "mammal", "mammals", "human", "humans", "fur", "hair", "coat",
    # reptiles / amphibians / fish
    "snake", "snakes", "skin",
    "lizard", "lizards", "reptile", "reptiles",
    "frog", "frogs", "amphibian", "amphibians",
    "fish", "fishes",
}


def _paper_gate_from_taxonomy(
    paper_text: str,
    taxonomy_lookup,
) -> Dict[str, Any]:
    tax_signal = taxonomy_lookup.summarize_taxonomic_signal(paper_text)

    allow = bool(tax_signal.get("has_taxonomic_signal", False))
    label = "arthropod_detected" if allow else "no_arthropod_signal"

    return {
        "allow": allow,
        "label": label,
        "n_direct_matches": tax_signal.get("n_direct_matches", 0),
        "n_propagated_matches": tax_signal.get("n_propagated_matches", 0),
        "direct_matches": tax_signal.get("direct_matches", []),
        "propagated_matches": tax_signal.get("propagated_matches", []),
    }


def _paper_gate_from_summary(
    summary_text: Optional[str],
    min_summary_sentences: int = 5,
) -> Dict[str, Any]:
    lines = [line.strip() for line in (summary_text or "").splitlines() if line.strip()]
    n_summary_sentences = len(lines)

    allow = n_summary_sentences >= min_summary_sentences
    label = (
        "moulting_content_detected"
        if allow
        else "insufficient_moulting_content"
    )

    return {
        "allow": allow,
        "label": label,
        "n_summary_sentences": n_summary_sentences,
        "min_summary_sentences": min_summary_sentences,
        "summary_preview": lines[:10],
    }


def _find_non_arthropod_query_hits(user_query: str) -> List[str]:
    q = user_query.lower()
    hits = sorted({term for term in NON_ARTHROPOD_QUERY_TERMS if term in q})
    return hits


def _query_gate_from_ontology(
    user_query: str,
    ontology_gate,
    min_hits: int = 1,
    min_score: float = 2.5,
) -> Dict[str, Any]:
    query_signal = ontology_gate.analyze_text(
        user_query,
        min_hits=min_hits,
        min_score=min_score,
    )

    non_arthropod_hits = _find_non_arthropod_query_hits(user_query)

    ontology_allow = bool(query_signal.get("allow", False))
    non_arthropod_detected = len(non_arthropod_hits) > 0

    if non_arthropod_detected:
        allow = False
        label = "non_arthropod_moulting_query_detected"
    else:
        allow = ontology_allow
        label = "moulting_query_detected" if allow else "no_moulting_signal_in_query"

    return {
        "allow": allow,
        "label": label,
        "n_hits": query_signal.get("n_hits", 0),
        "score": query_signal.get("score", 0.0),
        "hits": query_signal.get("hits", []),
        "summary_hits": query_signal.get("summary_hits", []),
        "non_arthropod_hits": non_arthropod_hits,
    }


def analyze_paper_and_query_domain(
    paper_text: str,
    user_query: str,
    taxonomy_lookup,
    ontology_gate,
    summary_text: Optional[str] = None,
    min_summary_sentences: int = 5,
    min_query_hits: int = 1,
    min_query_score: float = 2.5,
    paper_text_for_taxonomy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Combined domain gate for MoultGPT.

    Logic
    -----
    1. Paper must contain arthropod taxonomic signal.
    2. Paper must contain enough moulting-related summary sentences.
    3. User query must contain moulting-related signal from ontology.
    4. Query must not explicitly concern non-arthropod moulting.
    5. Only if all are true, request is in scope.
    """

    taxonomy_text = paper_text_for_taxonomy if paper_text_for_taxonomy is not None else paper_text

    paper_taxonomy_gate = _paper_gate_from_taxonomy(
        paper_text=taxonomy_text,
        taxonomy_lookup=taxonomy_lookup,
    )

    paper_summary_gate = _paper_gate_from_summary(
        summary_text=summary_text,
        min_summary_sentences=min_summary_sentences,
    )

    query_gate = _query_gate_from_ontology(
        user_query=user_query,
        ontology_gate=ontology_gate,
        min_hits=min_query_hits,
        min_score=min_query_score,
    )

    tax_ok = paper_taxonomy_gate["allow"]
    summary_ok = paper_summary_gate["allow"]
    query_ok = query_gate["allow"]
    non_arthropod_query = len(query_gate.get("non_arthropod_hits", [])) > 0

    if non_arthropod_query:
        allow = False
        final_label = "query_out_of_scope"
        message = (
            "The query appears to concern moulting in non-arthropods "
            f"({', '.join(query_gate['non_arthropod_hits'])})."
        )
    elif tax_ok and summary_ok and query_ok:
        allow = True
        final_label = "in_scope"
        message = (
            "Paper contains arthropod signal, sufficient moulting content, "
            "and the query concerns moulting."
        )
    elif not tax_ok and query_ok:
        allow = False
        final_label = "paper_out_of_scope"
        message = (
            "Query concerns moulting, but the paper does not show arthropod signal."
        )
    elif tax_ok and not summary_ok and query_ok:
        allow = False
        final_label = "paper_not_relevant"
        message = (
            "Paper contains arthropod signal, but too little moulting-related "
            "content was detected in the summary."
        )
    elif tax_ok and summary_ok and not query_ok:
        allow = False
        final_label = "query_out_of_scope"
        message = (
            "Paper appears relevant, but the query does not concern moulting."
        )
    elif not tax_ok and not summary_ok and query_ok:
        allow = False
        final_label = "paper_out_of_scope"
        message = (
            "Query concerns moulting, but the paper lacks arthropod signal and "
            "does not contain enough moulting-related content."
        )
    elif tax_ok and not summary_ok and not query_ok:
        allow = False
        final_label = "out_of_scope"
        message = (
            "Paper has arthropod signal, but it lacks enough moulting-related "
            "content and the query is not a moulting query."
        )
    else:
        allow = False
        final_label = "out_of_scope"
        message = "The request did not pass the combined domain gates."

    return {
        "allow": allow,
        "final_label": final_label,
        "message": message,
        "paper_taxonomy_gate": paper_taxonomy_gate,
        "paper_summary_gate": paper_summary_gate,
        "query_gate": query_gate,
    }