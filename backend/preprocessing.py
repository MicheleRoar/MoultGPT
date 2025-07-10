import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# === Parameters ===
NUM_SENTENCES = 12
KEYWORDS = [
    "moult", "moulting", "instar", "stadium", "cuticle", "ecdysis", "shed",
    "exuviae", "aestivation", "desiccation", "survival", "water loss",
    "reabsorption", "calcification", "growth", "resistance"
]

def simple_sentence_split(text: str) -> list[str]:
    """
    Splits text into sentences using punctuation and capitalization heuristics.
    """
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.?!])\s+(?=[A-Z])", text)

def is_biologically_relevant(sentence: str) -> bool:
    """
    Checks if a sentence is long enough, contains at least one keyword,
    and excludes unwanted patterns like figure references or trademarks.
    """
    return (
        len(sentence) > 30 and
        any(k in sentence.lower() for k in KEYWORDS) and
        len(sentence.split()) >= 5 and
        "®" not in sentence and
        "fig." not in sentence.lower() and
        not sentence.lower().endswith("fig.") and
        not re.search(r"\bfig\b", sentence.lower())
    )

def extract_relevant_sentences(full_text: str, num_sentences: int = NUM_SENTENCES) -> str:
    """
    Extracts a set of biologically relevant sentences from the full article text.
    Uses TF-IDF + KMeans clustering to select diverse representative sentences.
    
    Returns:
        A string of newline-separated sentences.
    
    Raises:
        ValueError: if no relevant sentences are found.
    """
    all_sentences = [s.strip() for s in simple_sentence_split(full_text)]
    filtered = [s for s in all_sentences if is_biologically_relevant(s)]

    if not filtered:
        raise ValueError("No biologically relevant sentences found in the text.")

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(filtered)

    k = min(num_sentences, len(filtered))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

    summary = []
    for i in range(k):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) == 0:
            continue
        center = kmeans.cluster_centers_[i]
        distances = X[cluster_indices] @ center.T
        closest_idx = cluster_indices[np.argmax(distances)]
        summary.append((closest_idx, filtered[closest_idx]))

    summary.sort(key=lambda x: x[0])
    return "\n".join(s for _, s in summary)
