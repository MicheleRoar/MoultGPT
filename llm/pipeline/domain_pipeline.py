# llm/pipeline/domain_pipeline.py
"""
Shared "paper -> ontology summary -> combined domain gate" step.

This is the one place where the summarization + gating logic lives, so that
llm/backend/app.py (the live Flask API) and llm/eval/compare_models.py (the
model-comparison script used for benchmarking/publication) run the *exact
same* pipeline before handing the result off to whichever remote LLM they
call. If this step changes, both callers pick it up automatically instead of
silently drifting apart from each other.
"""

from __future__ import annotations

from typing import Any, Dict

from pipeline.summarization import select_relevant_sentences_detailed
from domain.domain_gate import analyze_paper_and_query_domain


def run_domain_pipeline(
    full_text: str,
    user_query: str,
    taxonomy_lookup,
    ontology_gate,
    num_summary_sentences: int = 20,
    min_summary_sentences: int = 5,
    min_total_score: float = 2.5,
    min_substring_len: int = 5,
    min_query_hits: int = 1,
    min_query_score: float = 2.5,
    taxonomy_snippet_chars: int = 10000,
) -> Dict[str, Any]:
    """
    Build the ontology-driven summary of `full_text`, then run the combined
    domain gate (paper taxonomy signal + paper summary sufficiency + query
    moulting signal) against `user_query`.

    Returns:
        {
            "summary": str,
            "decision": <analyze_paper_and_query_domain(...) result dict>,
            "evidence": <select_relevant_sentences_detailed(...) result dict>,
        }
    `evidence` is the same selection `summary` was built from (see
    select_relevant_sentences_detailed's docstring) -- exposed so callers
    (llm/backend/app.py's /preprocess and /query) can show a reviewer exactly
    which sentences were kept, their score/position, and whether the
    num_summary_sentences cap actually dropped anything for this paper.
    """
    paper_taxonomy_text = full_text[:taxonomy_snippet_chars]

    evidence = select_relevant_sentences_detailed(
        full_text=full_text,
        ontology_gate=ontology_gate,
        num_sentences=num_summary_sentences,
        min_total_score=min_total_score,
        min_substring_len=min_substring_len,
    )
    summary = "\n".join(d["sentence"] for d in evidence["selected"])

    decision = analyze_paper_and_query_domain(
        paper_text=full_text,
        paper_text_for_taxonomy=paper_taxonomy_text,
        user_query=user_query,
        taxonomy_lookup=taxonomy_lookup,
        ontology_gate=ontology_gate,
        summary_text=summary,
        min_summary_sentences=min_summary_sentences,
        min_query_hits=min_query_hits,
        min_query_score=min_query_score,
    )

    return {"summary": summary, "decision": decision, "evidence": evidence}
