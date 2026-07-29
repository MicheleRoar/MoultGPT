from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple, Any, Optional

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL, SKOS


MOULT = Namespace("http://purl.org/moultdb/ontology/")


STOPWORDS_LIGHT = {
    "a", "an", "the",
    "its", "their", "his", "her",
    "this", "that", "these", "those",
    "of", "during", "in", "on", "at", "to", "for",
    "is", "are", "was", "were", "be", "been",
    "what", "which", "who", "whom", "whose",
    "how", "when", "where", "why",
    "do", "does", "did",
    "can", "could", "would", "should",
    "there", "about",
}

NOISE_CANONICALS = {
    "yes", "no", "none"
}

WEIGHT_SCORES = {
    "strong": 3.0,
    "medium": 2.0,
    "weak": 1.0,
    None: 1.0,
}

ROLE_BONUS = {
    "process": 1.0,
    "event": 1.0,
    "phase": 0.8,
    "evidence": 0.8,
    "trait": 0.4,
    "value": 0.2,
    "anatomy": 0.0,
    None: 0.0,
}

DEFAULT_ROOT_ONTO_CLASSES = [
    "MoultingProcess",
    "MoultingEvent",
    "MoultingPhase",
    "Trait",
    "TraitValue",
    "Context",
    "Instar",
    "Exuvia",
]


def uri_fragment(uri: URIRef | str) -> str:
    s = str(uri)
    if "#" in s:
        return s.split("#")[-1]
    return s.rsplit("/", 1)[-1]


def normalize_surface(text: str) -> str:
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
        "intermolt": "intermoult",
        "instars": "instar",
        "exuviae": "exuvia",
    }

    for src, tgt in replacements.items():
        text = re.sub(rf"\b{re.escape(src)}\b", tgt, text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_text_variants(text: str) -> Set[str]:
    base = normalize_surface(text)
    variants = {base}

    toks = base.split()
    compact = " ".join(t for t in toks if t not in STOPWORDS_LIGHT)
    compact = re.sub(r"\s+", " ", compact).strip()
    if compact:
        variants.add(compact)

    return variants


def literal_values(graph: Graph, node: URIRef, predicate: URIRef) -> List[str]:
    vals = []
    for _, _, obj in graph.triples((node, predicate, None)):
        if isinstance(obj, Literal):
            txt = str(obj).strip()
            if txt:
                vals.append(txt)
    return vals


def get_first_literal(graph: Graph, node: URIRef, predicate: URIRef) -> Optional[str]:
    vals = literal_values(graph, node, predicate)
    return vals[0] if vals else None


def bool_from_literal(text: Optional[str], default: bool = False) -> bool:
    if text is None:
        return default
    return str(text).strip().lower() in {"true", "1", "yes"}


def canonical_surface_choice(surfaces: Set[str]) -> str:
    if not surfaces:
        return ""
    ordered = sorted(
        surfaces,
        key=lambda x: (
            0 if len(x.split()) > 1 else 1,
            len(x),
            x,
        )
    )
    return ordered[0]


def get_all_surface_forms(graph: Graph, node: URIRef) -> Set[str]:
    surfaces: Set[str] = set()

    for v in literal_values(graph, node, RDFS.label):
        surfaces.add(v)
    for v in literal_values(graph, node, SKOS.altLabel):
        surfaces.add(v)
    for v in literal_values(graph, node, SKOS.prefLabel):
        surfaces.add(v)
    for v in literal_values(graph, node, MOULT.hasLexicalCue):
        surfaces.add(v)

    frag = uri_fragment(node)
    if frag:
        surfaces.add(frag)

    normed: Set[str] = set()
    for s in surfaces:
        s2 = normalize_surface(s)
        if s2:
            normed.add(s2)

    return normed


def find_class_by_fragment(graph: Graph, fragment_name: str) -> Optional[URIRef]:
    target = normalize_surface(fragment_name)

    for subj in graph.subjects(RDF.type, OWL.Class):
        frag = normalize_surface(uri_fragment(subj))
        if frag == target:
            return subj

    for subj in graph.subjects(RDF.type, OWL.Class):
        surfaces = get_all_surface_forms(graph, subj)
        if target in surfaces:
            return subj

    return None


def get_subclasses(graph: Graph, parent: URIRef) -> Set[URIRef]:
    return set(subj for subj in graph.subjects(RDFS.subClassOf, parent))


def get_all_descendants(graph: Graph, parent: URIRef) -> Set[URIRef]:
    visited: Set[URIRef] = set()
    stack = [parent]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        children = get_subclasses(graph, current)
        for ch in children:
            if ch not in visited:
                stack.append(ch)

    return visited


def get_instances_of_classes(graph: Graph, class_nodes: Set[URIRef]) -> Set[URIRef]:
    instances: Set[URIRef] = set()
    for cls in class_nodes:
        for inst in graph.subjects(RDF.type, cls):
            instances.add(inst)
    return instances


def should_exclude_concept(meta: Dict[str, Any]) -> bool:
    canonical = meta["canonical"]
    role = meta["role"]
    requires_context = meta["requires_context"]
    kind = meta["kind"]
    weight = meta["weight"]

    if canonical in NOISE_CANONICALS:
        return True

    if kind == "individual" and role is None and weight is None:
        return True

    if requires_context and role == "anatomy":
        return True

    return False


def enrich_surfaces_with_manual_bridges(meta: Dict[str, Any]) -> None:
    surfaces = set(meta["surfaces"])
    canonical = meta["canonical"]

    if canonical in {"instar", "instar transition"} or "instar" in surfaces:
        surfaces.update({
            "multiple instar",
            "developmental stage between moult",
        })

    if {"ecdysis", "moult", "moulting"} & surfaces:
        surfaces.update({
            "shedding exoskeleton",
            "shed exoskeleton",
        })

    meta["surfaces"] = sorted(normalize_surface(s) for s in surfaces if s)


def build_ngrams(tokens: List[str], max_n: int = 5) -> Set[str]:
    grams: Set[str] = set()
    for n in range(1, max_n + 1):
        if len(tokens) >= n:
            grams |= {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}
    return grams


def concept_match_score(meta: Dict[str, Any], matched_surface: str) -> float:
    weight_score = WEIGHT_SCORES.get(meta["weight"], 1.0)
    role_bonus = ROLE_BONUS.get(meta["role"], 0.0)
    lexical_bonus = 0.3 if len(matched_surface.split()) > 1 else 0.0
    context_penalty = -0.8 if meta["requires_context"] else 0.0
    return weight_score + role_bonus + lexical_bonus + context_penalty


class MoultingOntologyGate:
    def __init__(
        self,
        owl_path: str,
        root_class_names: Optional[List[str]] = None,
    ):
        self.owl_path = owl_path
        self.root_class_names = root_class_names or DEFAULT_ROOT_ONTO_CLASSES

        self.graph = Graph()
        self.graph.parse(self.owl_path)

        (
            self.surface_index,
            self.concepts,
            self.selected_classes,
            self.selected_instances,
        ) = self._build_moulting_ontology_terms()

    def _build_moulting_ontology_terms(self):
        selected_classes: Set[URIRef] = set()

        for name in self.root_class_names:
            cls = find_class_by_fragment(self.graph, name)
            if cls is None:
                print(f"[WARN] root class not found: {name}")
                continue
            selected_classes |= get_all_descendants(self.graph, cls)

        selected_instances = get_instances_of_classes(self.graph, selected_classes)
        concepts: Dict[str, Dict[str, Any]] = {}

        def register_node(node: URIRef, kind: str) -> None:
            uri = str(node)
            surfaces = get_all_surface_forms(self.graph, node)
            if not surfaces:
                return

            role = get_first_literal(self.graph, node, MOULT.hasSemanticRole)
            weight = get_first_literal(self.graph, node, MOULT.hasGatingWeight)
            requires_context_raw = get_first_literal(self.graph, node, MOULT.requiresContext)

            meta = {
                "uri": uri,
                "kind": kind,
                "surfaces": sorted(surfaces),
                "role": normalize_surface(role) if role else None,
                "weight": normalize_surface(weight) if weight else None,
                "requires_context": bool_from_literal(requires_context_raw, default=False),
                "canonical": canonical_surface_choice(surfaces),
            }

            enrich_surfaces_with_manual_bridges(meta)

            if should_exclude_concept(meta):
                return

            concepts[uri] = meta

        for node in selected_classes:
            register_node(node, "class")

        for node in selected_instances:
            register_node(node, "individual")

        surface_index: Dict[str, List[Dict[str, Any]]] = {}
        for meta in concepts.values():
            for surf in meta["surfaces"]:
                surface_index.setdefault(surf, []).append(meta)

        return surface_index, concepts, selected_classes, selected_instances

    def match_terms(self, text: str) -> List[Dict[str, Any]]:
        hits = []
        seen: Set[Tuple[str, str]] = set()

        for text_n in generate_text_variants(text):
            tokens = text_n.split()
            grams = build_ngrams(tokens, max_n=5)

            for gram in grams:
                if gram not in self.surface_index:
                    continue

                for meta in self.surface_index[gram]:
                    key = (gram, meta["canonical"])
                    if key in seen:
                        continue

                    hit = {
                        "matched_surface": gram,
                        "canonical": meta["canonical"],
                        "uri": meta["uri"],
                        "kind": meta["kind"],
                        "role": meta["role"],
                        "weight": meta["weight"],
                        "requires_context": meta["requires_context"],
                        "score": concept_match_score(meta, gram),
                    }
                    hits.append(hit)
                    seen.add(key)

        hits = sorted(
            hits,
            key=lambda h: (
                -h["score"],
                -len(h["matched_surface"].split()),
                h["matched_surface"],
            )
        )
        return hits

    def summarize_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "matched_surface": h["matched_surface"],
                "canonical": h["canonical"],
                "role": h["role"],
                "weight": h["weight"],
                "requires_context": h["requires_context"],
                "score": round(h["score"], 2),
            }
            for h in hits
        ]

    def analyze_text(
        self,
        text: str,
        min_hits: int = 1,
        min_score: float = 2.5,
    ) -> Dict[str, Any]:
        hits = self.match_terms(text)
        total_score = sum(h["score"] for h in hits)

        allow = (len(hits) >= min_hits) and (total_score >= min_score)

        return {
            "allow": allow,
            "n_hits": len(hits),
            "score": round(total_score, 2),
            "hits": hits,
            "summary_hits": self.summarize_hits(hits),
        }