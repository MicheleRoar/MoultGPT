# llm/pipeline/summarization.py

import re
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def _simple_sentence_split(text: str) -> List[str]:
    """Very simple sentence splitter: dot/question/exclamation + capital letter."""
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.?!])\s+(?=[A-Z])", text)


def _normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    replacements = {
        "molting": "moulting",
        "molt": "moult",
        "premoult": "pre moult",
        "postmoult": "post moult",
        "intermoult": "intermoult",
        "instars": "instar",
        "exuviae": "exuvia",
    }

    for src, tgt in replacements.items():
        text = re.sub(rf"\b{re.escape(src)}\b", tgt, text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _surface_matches_sentence(surface: str, sent_norm: str, min_substring_len: int = 5) -> bool:
    """
    Matching strategy:
    1. Exact whole-surface match first.
    2. If the ontology surface is long enough, allow substring match.
       This helps recover forms like:
         moult -> moulting
         ecdysi -> ecdysis / ecdysial
         exuvia -> exuvial
    """
    surface = surface.strip()
    if not surface:
        return False

    # exact phrase match
    if re.search(rf"\b{re.escape(surface)}\b", sent_norm):
        return True

    # controlled substring match for longer surfaces only
    if len(surface) >= min_substring_len and surface in sent_norm:
        return True

    return False


def _ontology_sentence_signal(
    sentence: str,
    ontology_gate,
    min_total_score: float = 2.5,
    min_substring_len: int = 5,
) -> Dict[str, Any]:
    """
    Score a sentence against ontology concepts.

    Returns:
        {
            "allow": bool,
            "score": float,
            "n_hits": int,
            "hits": [...]
        }
    """
    s = sentence.strip()
    if len(s) < 30:
        return {"allow": False, "score": 0.0, "n_hits": 0, "hits": []}
    if len(s.split()) < 5:
        return {"allow": False, "score": 0.0, "n_hits": 0, "hits": []}
    if "®" in s or "fig." in s.lower():
        return {"allow": False, "score": 0.0, "n_hits": 0, "hits": []}

    sent_norm = _normalize_text(s)

    hits = []
    seen = set()

    for meta in ontology_gate.concepts.values():
        for surface in meta["surfaces"]:
            if not surface:
                continue

            if not _surface_matches_sentence(
                surface=surface,
                sent_norm=sent_norm,
                min_substring_len=min_substring_len,
            ):
                continue

            key = (surface, meta["canonical"])
            if key in seen:
                continue

            weight_score = {
                "strong": 3.0,
                "medium": 2.0,
                "weak": 1.0,
                None: 1.0,
            }.get(meta["weight"], 1.0)

            role_bonus = {
                "process": 1.0,
                "event": 1.0,
                "phase": 0.8,
                "evidence": 0.8,
                "trait": 0.4,
                "value": 0.2,
                "anatomy": 0.0,
                None: 0.0,
            }.get(meta["role"], 0.0)

            lexical_bonus = 0.3 if len(surface.split()) > 1 else 0.0
            substring_penalty = -0.2 if surface not in sent_norm.split() else 0.0
            context_penalty = -0.8 if meta["requires_context"] else 0.0

            score = weight_score + role_bonus + lexical_bonus + substring_penalty + context_penalty

            hits.append({
                "matched_surface": surface,
                "canonical": meta["canonical"],
                "role": meta["role"],
                "weight": meta["weight"],
                "requires_context": meta["requires_context"],
                "score": score,
            })
            seen.add(key)

    if not hits:
        return {"allow": False, "score": 0.0, "n_hits": 0, "hits": []}

    total_score = sum(h["score"] for h in hits)
    strong_hits = sum(1 for h in hits if h["weight"] == "strong")
    process_hits = sum(1 for h in hits if h["role"] in {"process", "event", "phase", "evidence"})
    unique_canonicals = len({h["canonical"] for h in hits})

    # slight diversity bonus
    total_score += min(unique_canonicals * 0.3, 1.5)

    allow = (
        total_score >= min_total_score
        and (
            strong_hits >= 1
            or process_hits >= 1
            or unique_canonicals >= 2
        )
    )

    hits = sorted(
        hits,
        key=lambda x: (-x["score"], x["canonical"])
    )

    return {
        "allow": allow,
        "score": round(total_score, 2),
        "n_hits": len(hits),
        "hits": hits,
    }


def select_relevant_sentences_detailed(
    full_text: str,
    ontology_gate,
    num_sentences: int = 20,
    min_total_score: float = 2.5,
    min_substring_len: int = 5,
) -> Dict[str, Any]:
    """
    Same ontology-filter + TF-IDF/KMeans clustering as extract_relevant_sentences,
    but returns the full selection detail instead of just a joined string --
    for the debug/evidence UI (see llm/backend/app.py's /preprocess and /query
    "evidence" field), so a reviewer can see exactly which sentences were kept,
    their ontology score, their position in the (whitespace-normalized) full
    text, and -- critically for deciding whether to raise num_sentences --
    how many sentences passed the ontology filter in total vs. how many the
    num_sentences cap actually let through.

    extract_relevant_sentences() below calls this and returns only the joined
    "selected" sentence text, so the two never diverge: the evidence shown in
    the UI is guaranteed to be exactly what the LLM prompt contained, not a
    separately-recomputed approximation that could disagree with it.

    Returns:
        {
            "normalized_full_text": str,   # whitespace-collapsed full_text;
                                            # char_start/char_end below are
                                            # offsets into THIS string, not
                                            # the original full_text, since
                                            # that is what sentence-splitting
                                            # actually operated on
            "total_sentences": int,        # sentences after splitting, before any filtering
            "n_filtered": int,             # sentences that passed the ontology gate
                                            # (i.e. would be kept with an unlimited cap)
            "cap": int,                    # num_sentences argument actually used
            "capped": bool,                # True iff n_filtered > cap, i.e. raising
                                            # num_sentences would include more evidence
                                            # for THIS paper
            "selected": [
                {"sentence": str, "score": float, "original_index": int,
                 "char_start": int, "char_end": int, "top_concepts": [str, ...]},
                ...
            ],  # sorted by original_index, i.e. reading order
        }
    """
    normalized_full_text = re.sub(r"\s+", " ", full_text)
    all_sentences = [s.strip() for s in _simple_sentence_split(normalized_full_text) if s.strip()]

    # Recover each sentence's char offset in normalized_full_text by scanning
    # forward with a cursor -- robust here because normalized_full_text has
    # already been whitespace-collapsed to single spaces, so sentences appear
    # verbatim and in order (this would NOT be safe against the raw,
    # un-normalized full_text, which can have arbitrary internal whitespace).
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for sent in all_sentences:
        start = normalized_full_text.find(sent, cursor)
        if start == -1:
            start = cursor  # pathological fallback; keeps offsets monotonic
        end = start + len(sent)
        spans.append((start, end))
        cursor = end

    filtered_scored: List[Tuple[int, str, float, List[str]]] = []
    for idx, sent in enumerate(all_sentences):
        signal = _ontology_sentence_signal(
            sentence=sent,
            ontology_gate=ontology_gate,
            min_total_score=min_total_score,
            min_substring_len=min_substring_len,
        )
        if signal["allow"]:
            top_concepts = [h["canonical"] for h in signal["hits"][:5]]
            filtered_scored.append((idx, sent, signal["score"], top_concepts))

    result: Dict[str, Any] = {
        "normalized_full_text": normalized_full_text,
        "total_sentences": len(all_sentences),
        "n_filtered": len(filtered_scored),
        "cap": num_sentences,
        "capped": len(filtered_scored) > num_sentences,
        "selected": [],
    }

    if not filtered_scored:
        return result

    def _to_entry(idx: int, sent: str, score: float, concepts: List[str]) -> Dict[str, Any]:
        start, end = spans[idx]
        return {
            "sentence": sent,
            "score": score,
            "original_index": idx,
            "char_start": start,
            "char_end": end,
            "top_concepts": concepts,
        }

    # If few sentences survive, keep them all -- no clustering needed.
    if len(filtered_scored) <= num_sentences:
        result["selected"] = [_to_entry(idx, sent, sc, tc) for idx, sent, sc, tc in filtered_scored]
        return result

    # TF-IDF + KMeans clustering over ontology-filtered sentences
    filtered = [s for _, s, _, _ in filtered_scored]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(filtered)

    k = min(num_sentences, len(filtered))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)

    by_sentence = {sent: (idx, sc, tc) for idx, sent, sc, tc in filtered_scored}
    chosen: List[Dict[str, Any]] = []
    for i in range(k):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if not cluster_indices.size:
            continue

        center = kmeans.cluster_centers_[i]
        scores = X[cluster_indices] @ center.T
        closest_idx = cluster_indices[np.argmax(scores)]
        chosen_sentence = filtered[closest_idx]
        orig_idx, sc, tc = by_sentence[chosen_sentence]
        chosen.append(_to_entry(orig_idx, chosen_sentence, sc, tc))

    chosen.sort(key=lambda d: d["original_index"])
    result["selected"] = chosen
    return result


def extract_relevant_sentences(
    full_text: str,
    ontology_gate,
    num_sentences: int = 20,
    min_total_score: float = 2.5,
    min_substring_len: int = 5,
) -> str:
    """
    Extract biologically relevant sentences about arthropod moulting
    using ontology-based matching instead of hardcoded keyword lists.

    Logic:
      - split full text into sentences
      - keep only sentences with sufficient ontology signal
      - cluster kept sentences with TF-IDF + KMeans
      - return representative sentences in original order

    Thin wrapper around select_relevant_sentences_detailed() -- see that
    function's docstring for why the two never diverge.
    """
    detailed = select_relevant_sentences_detailed(
        full_text=full_text,
        ontology_gate=ontology_gate,
        num_sentences=num_sentences,
        min_total_score=min_total_score,
        min_substring_len=min_substring_len,
    )
    return "\n".join(d["sentence"] for d in detailed["selected"])