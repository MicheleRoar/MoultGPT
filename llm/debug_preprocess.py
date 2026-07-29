# llm/debug_preprocess.py

import argparse
import sys
from pathlib import Path

LLM_ROOT = Path(__file__).resolve().parent
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from pipeline.processor import input_to_text  # type: ignore
from pipeline.summarization import extract_relevant_sentences  # type: ignore

from domain.optimized_taxonomy_lookup import TaxonomyLookup  # type: ignore
from domain.moulting_ontology_gate import MoultingOntologyGate  # type: ignore
from domain.domain_gate import analyze_paper_and_query_domain  # type: ignore


TEST_QUERIES = [
    "What moulting traits are reported for Hurdiidae and Kerygmachela in this paper?",
    "Extract all information related to moulting of the species in this paper.",
    "How often do birds moult their feathers?",
    "Which species is described in this paper?",
    "Summarise all moulting traits of the spider described in this paper.",
    "What is the GDP of France?",
    "What evidence of ecdysis is reported in this paper?",
    "Describe the moulting process in this species.",
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Debug pipeline: DOI/PDF -> text -> ontology-based summarization -> "
            "taxonomy/summary/query gates -> final decision."
        )
    )
    parser.add_argument("--doi", type=str, default=None, help="DOI of the paper")
    parser.add_argument("--pdf", type=str, default=None, help="Path to local PDF")
    parser.add_argument("--email", type=str, default=None, help="Email for Unpaywall")

    parser.add_argument(
        "--taxonomy_csv",
        type=str,
        default=str(LLM_ROOT / "data" / "arthropod_taxonomy.csv"),
        help="Path to arthropod taxonomy CSV",
    )
    parser.add_argument(
        "--taxonomy_pickle",
        type=str,
        default=str(LLM_ROOT / "data" / "taxonomy_lookup.pkl"),
        help="Path to taxonomy lookup pickle",
    )
    parser.add_argument(
        "--ontology_owl",
        type=str,
        # v3_8 is the current, most complete ontology version (337 classes vs
        # 138 in the older v3_2 this script used to default to). See
        # llm/data/ for other historical versions.
        default=str(LLM_ROOT / "data" / "moultdb_moulting_ontology_v3_8.owl"),
        help="Path to moulting ontology OWL file",
    )
    parser.add_argument(
        "--rebuild_taxonomy",
        action="store_true",
        help="Rebuild taxonomy lookup pickle from CSV",
    )

    parser.add_argument(
        "--min_query_hits",
        type=int,
        default=1,
        help="Minimum ontology hits required in query",
    )
    parser.add_argument(
        "--min_query_score",
        type=float,
        default=2.5,
        help="Minimum ontology score required in query",
    )

    # summarization params
    parser.add_argument(
        "--num_sentences",
        type=int,
        default=20,
        help="Maximum number of summary sentences",
    )
    parser.add_argument(
        "--min_summary_sentences",
        type=int,
        default=5,
        help="Minimum number of summary sentences required for paper relevance",
    )
    parser.add_argument(
        "--min_total_score",
        type=float,
        default=2.5,
        help="Minimum ontology score required for a sentence to enter the summary",
    )
    parser.add_argument(
        "--min_substring_len",
        type=int,
        default=5,
        help="Minimum ontology surface length for substring matching",
    )

    # taxonomy gate optimization
    parser.add_argument(
        "--taxonomy_snippet_chars",
        type=int,
        default=10000,
        help="Number of initial characters used for taxonomy paper gate",
    )

    args = parser.parse_args()

    if not args.doi and not args.pdf:
        print("ERROR: Provide either --doi or --pdf")
        sys.exit(1)

    # STEP 1
    print("\n[STEP 1] Extracting full text...")
    full_text = input_to_text(doi=args.doi, pdf_path=args.pdf, email=args.email)

    if not full_text or len(full_text.strip()) < 100:
        print("[ERROR] Extracted text is empty or too short.")
        sys.exit(1)

    print(f"[INFO] Extracted text length: {len(full_text)} characters")
    print("\n[PREVIEW] First 1000 characters:\n")
    print(full_text[:1000])
    print("\n" + "-" * 100 + "\n")

    # Precompute snippet for taxonomy gate
    paper_taxonomy_text = full_text[:args.taxonomy_snippet_chars]
    print(
        f"[INFO] Taxonomy gate snippet length: {len(paper_taxonomy_text)} characters "
        f"(from first {args.taxonomy_snippet_chars} chars)"
    )
    print("\n" + "-" * 100 + "\n")

    # STEP 2
    print("[STEP 2] Initializing ontology + taxonomy...")
    taxonomy_lookup = TaxonomyLookup(
        csv_path=args.taxonomy_csv,
        pickle_path=args.taxonomy_pickle,
        rebuild=args.rebuild_taxonomy,
    )
    ontology_gate = MoultingOntologyGate(
        owl_path=args.ontology_owl,
    )

    print(f"[INFO] Ontology concepts indexed: {len(ontology_gate.concepts)}")
    print(f"[INFO] Ontology surfaces indexed: {len(ontology_gate.surface_index)}")
    print("\n" + "-" * 100 + "\n")

    # STEP 3
    print("[STEP 3] Ontology-driven summarization...\n")

    summary = extract_relevant_sentences(
        full_text=full_text,
        ontology_gate=ontology_gate,
        num_sentences=args.num_sentences,
        min_total_score=args.min_total_score,
        min_substring_len=args.min_substring_len,
    )

    summary_lines = [s for s in summary.splitlines() if s.strip()]
    n_summary = len(summary_lines)

    print(f"[INFO] n_summary_sentences={n_summary}")
    print(f"[INFO] min_summary_sentences={args.min_summary_sentences}")

    print("\n[SUMMARY OUTPUT]\n")
    print(summary if summary else "[EMPTY SUMMARY]")
    print("\n" + "-" * 100 + "\n")

    # STEP 4
    print("[STEP 4] Query tests with combined domain gate...\n")

    for q in TEST_QUERIES:
        print("=" * 100)
        print(f"QUERY: {q}")

        result = analyze_paper_and_query_domain(
            paper_text=full_text,
            paper_text_for_taxonomy=paper_taxonomy_text,
            user_query=q,
            taxonomy_lookup=taxonomy_lookup,
            ontology_gate=ontology_gate,
            summary_text=summary,
            min_summary_sentences=args.min_summary_sentences,
            min_query_hits=args.min_query_hits,
            min_query_score=args.min_query_score,
        )

        print(f"FINAL LABEL: {result['final_label']}")
        print(f"ALLOW: {result['allow']}")
        print(f"MESSAGE: {result['message']}")

        paper_taxonomy_gate = result["paper_taxonomy_gate"]
        paper_summary_gate = result["paper_summary_gate"]
        query_gate = result["query_gate"]

        print(
            f"PAPER TAXONOMY: {paper_taxonomy_gate['label']} | "
            f"direct={paper_taxonomy_gate['n_direct_matches']} | "
            f"propagated={paper_taxonomy_gate['n_propagated_matches']}"
        )

        print(
            f"PAPER SUMMARY: {paper_summary_gate['label']} | "
            f"n_summary={paper_summary_gate['n_summary_sentences']} | "
            f"min_required={paper_summary_gate['min_summary_sentences']}"
        )

        print(
            f"QUERY ONTOLOGY: {query_gate['label']} | "
            f"hits={query_gate['n_hits']} | "
            f"score={query_gate['score']}"
        )

        non_arthropod_hits = query_gate.get("non_arthropod_hits", [])
        if non_arthropod_hits:
            print(
                f"QUERY NEGATIVE SIGNAL: non_arthropod_hits={non_arthropod_hits}"
            )

        if paper_taxonomy_gate["direct_matches"]:
            print("[DIRECT TAXON MATCHES]")
            for m in paper_taxonomy_gate["direct_matches"][:10]:
                print(
                    f"  - canonical='{m['canonical_name']}' | "
                    f"matched='{m['matched_name']}' | "
                    f"path={m['path']} | depth={m['depth']}"
                )

        if query_gate["summary_hits"]:
            print("[QUERY ONTOLOGY HITS]")
            for h in query_gate["summary_hits"][:10]:
                print(
                    f"  - matched='{h['matched_surface']}' | "
                    f"canonical='{h['canonical']}' | "
                    f"role={h['role']} | weight={h['weight']} | "
                    f"score={h['score']}"
                )

        print()

    print("-" * 100)
    print("[DONE] Full debug pipeline completed.")


if __name__ == "__main__":
    main()