import re
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def summarize_relevant_sentences(full_text: str, num_sentences: int = 20) -> str:
    def simple_sentence_split(text):
        text = re.sub(r"\s+", " ", text)
        return re.split(r"(?<=[.?!])\s+(?=[A-Z])", text)

    keywords = [
        "moult", "moulting", "instar", "stadium", "cuticle", "ecdysis", "shed",
        "exuviae", "aestivation", "desiccation", "survival", "water loss",
        "reabsorption", "calcification", "growth", "resistance"
    ]

    def is_relevant(s):
        return (
            len(s) > 30 and
            any(k in s.lower() for k in keywords) and
            len(s.split()) >= 5 and
            "®" not in s and
            "fig." not in s.lower()
        )

    all_sentences = [s.strip() for s in simple_sentence_split(full_text)]
    filtered = [s for s in all_sentences if is_relevant(s)]
    if not filtered:
        return ""

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(filtered)
    k = min(num_sentences, len(filtered))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)

    summary = []
    for i in range(k):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if not cluster_indices.size:
            continue
        center = kmeans.cluster_centers_[i]
        distances = X[cluster_indices] @ center.T
        closest_idx = cluster_indices[np.argmax(distances)]
        summary.append((closest_idx, filtered[closest_idx]))

    summary.sort(key=lambda x: x[0])
    return "\n".join(s for _, s in summary)

def pdf_to_summary(paper_path, grobid_url="http://localhost:8070/api/processFulltextDocument") -> str:
    with open(paper_path, "rb") as f:
        response = requests.post(grobid_url, files={"input": f}, data={"consolidateHeader": 1})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "xml")
    full_text = "\n".join(p.get_text() for p in soup.find_all("p"))
    return summarize_relevant_sentences(full_text)