import os
import re
import pickle
from typing import List, Dict, Any, Optional

import pandas as pd


DEFAULT_PICKLE_PATH = "data/taxonomy_lookup.pkl"


class TaxonomyLookup:
    """
    Fast lexical taxonomy lookup for MoultGPT.

    Goals
    -----
    - expensive build only once
    - fast repeated lookups afterward
    - detect whether a text contains names present in the taxonomy file
    - optionally propagate matched taxa to all ancestors via dotted path

    Expected CSV columns
    --------------------
    id, path,
    ncbi_canonical_name, inat_canonical_name, gbif_canonical_name,
    gbif_synonyms_names, ncbi_synonyms_names
    """

    def __init__(
        self,
        csv_path: str = "data/arthropod_taxonomy.csv",
        pickle_path: str = DEFAULT_PICKLE_PATH,
        rebuild: bool = False,
    ):
        self.csv_path = csv_path
        self.pickle_path = pickle_path

        if os.path.exists(self.pickle_path) and not rebuild:
            print("[TaxonomyLookup] Loading precomputed lookup...")
            self._load()
        else:
            print("[TaxonomyLookup] Building lookup from CSV (slow, one-time)...")
            self._build()
            self._save()

    # ------------------------------------------------------------------
    # Normalization / filtering
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        text = str(text).strip().lower()
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[.,;:()\[\]{}]+$", "", text)
        return text

    def _normalize_runtime_text(self, text: str) -> str:
        """
        Normalize free text a bit more aggressively than names.
        Keeps letters, spaces and hyphens.
        """
        text = self._normalize(text)
        text = re.sub(r"[^a-z\s\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_valid_name(self, name: str) -> bool:
        if not isinstance(name, str):
            return False

        name = name.strip()
        if len(name) < 4:
            return False

        if re.search(r"[^A-Za-z\s\-]", name):
            return False

        nlow = name.lower()

        banned = {
            "data", "area", "areas", "region", "regions", "record", "records",
            "number", "sample", "samples", "information", "dataset", "table",
            "appendix", "thorax", "abdomen", "segment", "segments", "growth",
            "development", "formation", "layer", "species", "organism",
            "organisms", "material", "materials", "result", "results",
        }
        if nlow in banned:
            return False

        return True

    def _split_synonyms(self, value: Any) -> List[str]:
        if not isinstance(value, str) or not value.strip():
            return []
        return [x.strip() for x in value.split(";") if x.strip()]

    # ------------------------------------------------------------------
    # Row parsing helpers
    # ------------------------------------------------------------------

    def _pick_canonical(self, row: pd.Series) -> str:
        for col in [
            "ncbi_canonical_name",
            "inat_canonical_name",
            "gbif_canonical_name",
        ]:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def _collect_names(self, row: pd.Series) -> List[str]:
        names: List[str] = []

        for col in [
            "ncbi_canonical_name",
            "inat_canonical_name",
            "gbif_canonical_name",
        ]:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                names.append(val.strip())

        for syn in self._split_synonyms(row.get("gbif_synonyms_names")):
            names.append(syn)

        for syn in self._split_synonyms(row.get("ncbi_synonyms_names")):
            names.append(syn)

        out = []
        seen = set()
        for n in names:
            if not self._is_valid_name(n):
                continue
            norm = self._normalize(n)
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)

        return out

    # ------------------------------------------------------------------
    # Build / save / load
    # ------------------------------------------------------------------
    #
    # NOTE on matching strategy: earlier versions compiled one regex per
    # taxon name and, on every lookup, ran ALL of them against the input
    # text sequentially (O(n_patterns x text_length)). With ~2M+ names in
    # the real arthropod_taxonomy.csv, that made a single paper-length gate
    # check take minutes. Matching is now done the other way around: the
    # input text is tokenized once into word n-grams (O(text_length)), and
    # each n-gram is looked up in `name_to_entry` (an O(1) dict lookup).
    # Same whole-word/whole-phrase semantics as the old \b...\b regexes,
    # several orders of magnitude faster at this taxonomy size.

    def _build(self):
        df = pd.read_csv(self.csv_path)

        self.name_to_entry: Dict[str, Dict[str, Any]] = {}
        self.path_to_entry: Dict[str, Dict[str, Any]] = {}

        total_rows = 0
        total_names = 0

        for _, row in df.iterrows():
            total_rows += 1

            canonical = self._pick_canonical(row)
            if not canonical:
                continue

            canonical_norm = self._normalize(canonical)

            path = str(row.get("path", "")).strip()
            taxon_id = row.get("id", None)
            depth = path.count(".") + 1 if path else 0

            canonical_entry = {
                "matched_name": canonical_norm,
                "canonical_name": canonical_norm,
                "taxon_id": taxon_id,
                "path": path,
                "depth": depth,
            }

            if path and path not in self.path_to_entry:
                self.path_to_entry[path] = canonical_entry

            all_names = self._collect_names(row)

            for norm_name in all_names:
                if norm_name not in self.name_to_entry:
                    self.name_to_entry[norm_name] = {
                        "matched_name": norm_name,
                        "canonical_name": canonical_norm,
                        "taxon_id": taxon_id,
                        "path": path,
                        "depth": depth,
                    }
                    total_names += 1

        # Longest taxon name in words, capped so a single pathological
        # multi-word entry can't blow up n-gram generation at lookup time.
        self.max_name_words = min(
            8, max((len(n.split()) for n in self.name_to_entry.keys()), default=1)
        )

        print(f"[TaxonomyLookup] Rows read: {total_rows}")
        print(f"[TaxonomyLookup] Unique normalized names: {len(self.name_to_entry)}")
        print(f"[TaxonomyLookup] Paths indexed: {len(self.path_to_entry)}")
        print(f"[TaxonomyLookup] Max name length (words): {self.max_name_words}")

    def _save(self):
        payload = {
            "name_to_entry": self.name_to_entry,
            "path_to_entry": self.path_to_entry,
            "max_name_words": self.max_name_words,
        }
        with open(self.pickle_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[TaxonomyLookup] Saved compiled lookup to: {self.pickle_path}")

    def _load(self):
        with open(self.pickle_path, "rb") as f:
            data = pickle.load(f)

        if "max_name_words" not in data:
            # Pickle built by the old (regex-based) version — its schema
            # doesn't have what the new lookup needs. Rebuild once instead
            # of crashing; the fresh pickle will load instantly next time.
            print(
                "[TaxonomyLookup] Cached pickle is from an older format "
                "(pre n-gram lookup). Rebuilding once from CSV..."
            )
            self._build()
            self._save()
            return

        self.name_to_entry = data["name_to_entry"]
        self.path_to_entry = data["path_to_entry"]
        self.max_name_words = data["max_name_words"]

        print(f"[TaxonomyLookup] Names loaded: {len(self.name_to_entry)}")
        print(f"[TaxonomyLookup] Paths loaded: {len(self.path_to_entry)}")
        print(f"[TaxonomyLookup] Max name length (words): {self.max_name_words}")

    def _ngrams_from_text(self, normalized_text: str) -> List[str]:
        words = normalized_text.split()
        n_words = len(words)
        max_n = min(self.max_name_words, n_words)
        grams: List[str] = []
        for n in range(1, max_n + 1):
            for i in range(n_words - n + 1):
                grams.append(" ".join(words[i:i + n]))
        return grams

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _deduplicate(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique = []

        for e in entries:
            key = (e["canonical_name"], e["taxon_id"])
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_any_match(self, text: str) -> bool:
        text = self._normalize_runtime_text(text)
        if not text:
            return False

        for gram in self._ngrams_from_text(text):
            if gram in self.name_to_entry:
                return True
        return False

    def find_taxa_in_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Direct matches only: taxa explicitly found in text.
        """
        text = self._normalize_runtime_text(text)
        if not text:
            return []

        found = []
        seen_names = set()

        for gram in self._ngrams_from_text(text):
            if gram in seen_names:
                continue
            entry = self.name_to_entry.get(gram)
            if entry is not None:
                found.append(entry)
                seen_names.add(gram)

        found = self._deduplicate(found)
        found.sort(key=lambda x: x["depth"], reverse=True)
        return found

    def propagate_taxa(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add all ancestors for each matched entry using dotted path.
        Example:
            1.172.50.16.16 -> 1.172.50.16 -> 1.172.50 -> 1.172 -> 1
        """
        propagated = []
        seen = set()

        for e in entries:
            path = e.get("path", "")
            if not path:
                continue

            parts = path.split(".")
            ancestor_paths = [
                ".".join(parts[:i]) for i in range(len(parts), 0, -1)
            ]

            for ap in ancestor_paths:
                anc = self.path_to_entry.get(ap)
                if anc is None:
                    continue

                key = (anc["canonical_name"], anc["taxon_id"])
                if key not in seen:
                    seen.add(key)
                    propagated.append(anc)

        propagated.sort(key=lambda x: x["depth"], reverse=True)
        return propagated

    def find_taxa_with_ancestors(self, text: str) -> List[Dict[str, Any]]:
        """
        Direct matches + all ancestors from the taxonomy path.
        """
        direct = self.find_taxa_in_text(text)
        return self.propagate_taxa(direct)

    def summarize_taxonomic_signal(self, text: str) -> Dict[str, Any]:
        """
        Lightweight summary useful for gating/debugging.
        """
        direct = self.find_taxa_in_text(text)
        propagated = self.propagate_taxa(direct)

        return {
            "has_taxonomic_signal": len(direct) > 0,
            "n_direct_matches": len(direct),
            "n_propagated_matches": len(propagated),
            "direct_matches": direct,
            "propagated_matches": propagated,
        }


if __name__ == "__main__":
    lookup = TaxonomyLookup(
        csv_path="data/arthropod_taxonomy.csv",
        pickle_path="data/taxonomy_lookup.pkl",
        rebuild=False,
    )

    sample = """
    Trilobite exuviae record the development of individual trilobites and their molting process.
    The study focuses on Omegops sp. A, a phacopid trilobite from the Hongguleleng Formation.
    """

    print("\nHAS MATCH:", lookup.has_any_match(sample))

    print("\nDIRECT MATCHES")
    for m in lookup.find_taxa_in_text(sample):
        print(m)

    print("\nWITH ANCESTORS")
    for m in lookup.find_taxa_with_ancestors(sample):
        print(m)