# llm/backend/app.py
"""
MoultGPT — LLM backend

- Accepts DOI, PDF upload or raw text + user prompt.
- Uses the LLM pipeline modules to:
  1) Convert DOI/PDF to text (Unpaywall + GROBID).
  2) Run the ontology-driven pipeline:
     - build a moulting-focused summary of the paper (TF-IDF/KMeans over
       ontology-scored sentences),
     - check arthropod taxonomic signal in the paper,
     - check the user query is a moulting question (and not about
       non-arthropod moulting or an off-topic question).
  3) Route or block the request according to the combined domain gate.
  4) Call a REMOTE LLM (no local GPU / no local weights) and return its
     reply (expected to be clean YAML per the system prompt).

Any of several free-tier remote models can be used — see
llm/config/models.py for the registry and llm/backend/providers.py for the
provider adapters (OpenRouter / Mistral / Gemini). The active model is
selected per-request via the "model" form field (defaults to
config.models.DEFAULT_MODEL_ID), so the same pipeline can be used to
benchmark several models on the same paper/query (see
llm/eval/compare_models.py).

Image prediction is NOT handled here (that's in the vision backend).

Run locally:
    cd llm/backend
    python app.py

Then open http://localhost:5002/ui for a small built-in web UI (DOI/PDF/text
input, multi-model selection, side-by-side results) — see llm/frontend/.

Environment variables:
    PORT                (default: 5002)
    TAXONOMY_CSV_PATH    (default: llm/data/arthropod_taxonomy.csv)
    TAXONOMY_PICKLE_PATH (default: llm/data/taxonomy_lookup.pkl)
    ONTOLOGY_OWL_PATH    (default: llm/data/moultdb_moulting_ontology_v3_8.owl)
    MAX_NEW_TOKENS_FULL   (default: 1024, used when mode=full_traits)
    MAX_NEW_TOKENS_SINGLE (default: 256, used when mode=single_trait)
    OPENROUTER_API_KEY / MISTRAL_API_KEY / GEMINI_API_KEY
        (only the ones needed for the models you actually use — see
        llm/config/models.py)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, Tuple

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[LLM] NOTE: 'python-dotenv' not installed; relying on real env vars only.")

# ────────────────────── Path setup ──────────────────────

# This file lives in llm/backend/app.py
# We add the llm/ root so we can import from llm/pipeline/*, llm/domain/*,
# llm/config/* and llm/backend/* (this module's own package) consistently.
LLM_ROOT = Path(__file__).resolve().parents[1]
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

FRONTEND_DIR = LLM_ROOT / "frontend"

from pipeline.processor import input_to_text  # type: ignore  # noqa: E402
from pipeline.domain_pipeline import run_domain_pipeline  # type: ignore  # noqa: E402
from pipeline.parser import grobid_is_alive, GROBID_ISALIVE_URL  # type: ignore  # noqa: E402

from domain.optimized_taxonomy_lookup import TaxonomyLookup  # type: ignore  # noqa: E402
from domain.moulting_ontology_gate import MoultingOntologyGate  # type: ignore  # noqa: E402

from backend.providers import generate as llm_generate, ProviderError  # type: ignore  # noqa: E402
from config.models import list_models, get_model, is_known_model, DEFAULT_MODEL_ID  # type: ignore  # noqa: E402
from pipeline.prompting import (  # type: ignore  # noqa: E402
    MODE_SINGLE_TRAIT,
    MODE_FULL_TRAITS,
    DEFAULT_MODE,
    VALID_MODES,
    build_system_prompt,
    build_user_content,
)

# ────────────────────── Config ──────────────────────

PORT = int(os.getenv("PORT", 5002))

TAXONOMY_CSV_PATH = os.getenv("TAXONOMY_CSV_PATH", str(LLM_ROOT / "data" / "arthropod_taxonomy.csv"))
TAXONOMY_PICKLE_PATH = os.getenv("TAXONOMY_PICKLE_PATH", str(LLM_ROOT / "data" / "taxonomy_lookup.pkl"))
ONTOLOGY_OWL_PATH = os.getenv("ONTOLOGY_OWL_PATH", str(LLM_ROOT / "data" / "moultdb_moulting_ontology_v3_8.owl"))

# full_traits mode can legitimately emit ~50 field lines, single_trait mode
# answers one question — different token budgets by default so full_traits
# responses don't get silently truncated mid-schema.
MAX_NEW_TOKENS_FULL = int(os.getenv("MAX_NEW_TOKENS_FULL", 1024))
MAX_NEW_TOKENS_SINGLE = int(os.getenv("MAX_NEW_TOKENS_SINGLE", 256))

# Summarization / gating thresholds (same defaults as debug_preprocess.py,
# kept here so the Flask backend and the CLI debug script stay in sync).
NUM_SUMMARY_SENTENCES = int(os.getenv("NUM_SUMMARY_SENTENCES", 20))
MIN_SUMMARY_SENTENCES = int(os.getenv("MIN_SUMMARY_SENTENCES", 5))
MIN_TOTAL_SCORE = float(os.getenv("MIN_TOTAL_SCORE", 2.5))
MIN_SUBSTRING_LEN = int(os.getenv("MIN_SUBSTRING_LEN", 5))
MIN_QUERY_HITS = int(os.getenv("MIN_QUERY_HITS", 1))
MIN_QUERY_SCORE = float(os.getenv("MIN_QUERY_SCORE", 2.5))
TAXONOMY_SNIPPET_CHARS = int(os.getenv("TAXONOMY_SNIPPET_CHARS", 10000))


# ────────────────────── Flask app ──────────────────────

print("[BOOT] Initializing MoultGPT LLM backend...")
app = Flask(__name__)
CORS(app)

# ────────────────────── Warm-up: taxonomy lookup + ontology gate ──────────────────────
#
# These replace the two local models MoultGPT used to depend on: they are
# lightweight (CSV lookup + OWL graph in memory), so this backend no longer
# needs a GPU or any downloaded model weights to start.

print("[BOOT] Loading taxonomy lookup + moulting ontology gate...")

READY = False
BOOT_ERROR: Optional[str] = None
TAXONOMY_LOOKUP: Optional[TaxonomyLookup] = None
ONTOLOGY_GATE: Optional[MoultingOntologyGate] = None

try:
    t0_boot = time.time()
    TAXONOMY_LOOKUP = TaxonomyLookup(
        csv_path=TAXONOMY_CSV_PATH,
        pickle_path=TAXONOMY_PICKLE_PATH,
        rebuild=False,
    )
    ONTOLOGY_GATE = MoultingOntologyGate(owl_path=ONTOLOGY_OWL_PATH)
    READY = True
    print(
        f"[BOOT] Ready. Taxonomy names: {len(TAXONOMY_LOOKUP.name_to_entry)} | "
        f"Ontology concepts: {len(ONTOLOGY_GATE.concepts)} | "
        f"warm-up time: {time.time() - t0_boot:.2f}s"
    )
except Exception as e:
    BOOT_ERROR = str(e)
    print(f"[BOOT] ERROR loading taxonomy/ontology resources: {e}")


def _ensure_ready():
    if not READY:
        raise RuntimeError(
            f"Taxonomy/ontology resources are not loaded; check server logs. "
            f"({BOOT_ERROR})"
        )


# ────────────────────── Helpers ──────────────────────


@app.route("/", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok" if READY else "degraded",
            "module": "llm",
            "ready": READY,
            "boot_error": BOOT_ERROR,
            "default_model": DEFAULT_MODEL_ID,
            "available_models": [m["id"] for m in list_models()],
        }
    ), (200 if READY else 500)


@app.route("/models", methods=["GET"])
def models():
    """List candidate models (for a frontend dropdown / eval script)."""
    return jsonify({"models": list_models(), "default_model": DEFAULT_MODEL_ID})


@app.route("/ui", methods=["GET"])
@app.route("/ui/", methods=["GET"])
def ui():
    """
    Minimal built-in web UI: pick a DOI/PDF/text source, a query, one or
    several models, and compare responses side by side. Served from the
    same Flask app (same origin as /query etc.), so no CORS setup needed.
    Source: llm/frontend/index.html.
    """
    return send_from_directory(FRONTEND_DIR, "index.html")


def _extract_full_text_from_request():
    """
    Resolve DOI / file / raw text into a single full_text string.

    Returns:
        full_text: str or None
        source: "doi" | "file" | "text" | None
        saved_pdf_path: str or None
        used_doi: bool
        used_pdf: bool
        used_raw_text: bool
    """
    doi = request.form.get("doi", "").strip()
    raw_text = request.form.get("text", "").strip()
    file = request.files.get("file")

    full_text: Optional[str] = None
    source: Optional[str] = None
    saved_pdf_path: Optional[str] = None

    if doi:
        print(f"[LLM] Using DOI={doi}")
        full_text = input_to_text(doi=doi)
        source = "doi"

    elif file:
        tmp_dir = LLM_ROOT / "data" / "papers_pdf"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / file.filename
        file.save(tmp_path)
        print(f"[LLM] Saved uploaded PDF to {tmp_path}")
        full_text = input_to_text(pdf_path=str(tmp_path))
        source = "file"
        saved_pdf_path = str(tmp_path)

    elif raw_text:
        print("[LLM] Using raw text from request.")
        full_text = raw_text
        source = "text"

    return full_text, source, saved_pdf_path, bool(doi), bool(file), bool(raw_text)


def _extraction_error_payload(source: Optional[str]) -> dict:
    """
    Build a clear error payload when text extraction produced nothing/too
    little. For "doi"/"file" sources this depends on GROBID, which is a
    separate long-running service (see llm/tools/grobid) — not something
    this backend starts on its own. If GROBID is unreachable, say so
    explicitly instead of the generic "text too short" message, since that
    is by far the most common cause of this failure.
    """
    payload = {
        "error": "Could not extract meaningful content from the input.",
        "details": "Text too short or empty after extraction.",
    }
    if source in ("doi", "file"):
        alive = grobid_is_alive()
        payload["grobid_reachable"] = alive
        if not alive:
            payload["hint"] = (
                f"GROBID does not appear to be running at {GROBID_ISALIVE_URL}. "
                "Start it separately (it is not launched by this backend): "
                "cd llm/tools/grobid && ./gradlew run — then retry once "
                "`curl http://localhost:8070/api/isalive` returns true."
            )
    return payload


# Upper bound on a per-request num_sentences override (see _resolve_num_sentences
# below) -- KMeans over a few hundred clusters on a single paper is still fast,
# but there's no reason to let a request ask for an unbounded number.
MAX_NUM_SENTENCES_OVERRIDE = 300


def _resolve_num_sentences() -> int:
    """
    Sentence-cap for THIS request: the server default (NUM_SUMMARY_SENTENCES),
    unless the request form includes a valid `num_sentences` override -- lets
    the /ui demo test "what if we raised the cap?" live, on a real paper,
    without redeploying. Silently falls back to the default on a missing/
    invalid value rather than erroring, since this is a debug convenience,
    not a required parameter.
    """
    raw = request.form.get("num_sentences", "").strip()
    if not raw:
        return NUM_SUMMARY_SENTENCES
    try:
        n = int(raw)
    except ValueError:
        return NUM_SUMMARY_SENTENCES
    return max(1, min(n, MAX_NUM_SENTENCES_OVERRIDE))


def _run_domain_pipeline(full_text: str, user_prompt: str, num_summary_sentences: int = None) -> dict:
    """
    Thin wrapper around pipeline.domain_pipeline.run_domain_pipeline, bound
    to this backend's config + loaded taxonomy/ontology resources. The same
    underlying function is used by llm/eval/compare_models.py, so the live
    API and the benchmarking script always run identical logic up to the
    point where the LLM is called (as long as num_summary_sentences isn't
    overridden -- see _resolve_num_sentences -- which only happens from a
    live /ui request, never from the eval scripts).
    """
    return run_domain_pipeline(
        full_text=full_text,
        user_query=user_prompt,
        taxonomy_lookup=TAXONOMY_LOOKUP,
        ontology_gate=ONTOLOGY_GATE,
        num_summary_sentences=num_summary_sentences if num_summary_sentences is not None else NUM_SUMMARY_SENTENCES,
        min_summary_sentences=MIN_SUMMARY_SENTENCES,
        min_total_score=MIN_TOTAL_SCORE,
        min_substring_len=MIN_SUBSTRING_LEN,
        min_query_hits=MIN_QUERY_HITS,
        min_query_score=MIN_QUERY_SCORE,
        taxonomy_snippet_chars=TAXONOMY_SNIPPET_CHARS,
    )


def _evidence_payload(evidence: dict) -> dict:
    """
    Passes through select_relevant_sentences_detailed()'s full output,
    including normalized_full_text (the whole paper, whitespace-collapsed --
    typically tens of KB for these papers), because the /ui evidence view
    highlights selected sentences in place in that text (a lightweight
    saliency-style view), which needs the full text to render against.
    """
    return {
        "cap": evidence["cap"],
        "total_sentences": evidence["total_sentences"],
        "n_filtered": evidence["n_filtered"],
        "n_selected": len(evidence["selected"]),
        "capped": evidence["capped"],
        "selected": evidence["selected"],
        "normalized_full_text": evidence["normalized_full_text"],
    }


# ────────────────────── Preprocess endpoint ──────────────────────


@app.route("/preprocess", methods=["POST"])
def preprocess():
    """
    Debug / helper endpoint: run only the pre-processing pipeline
    (ontology-driven summary + combined domain gate) without calling any LLM.

    Form fields:
        - doi       (optional)
        - text      (optional, raw text)
        - file      (optional, PDF upload)
        - prompt    (optional; if provided, the query gate is evaluated too;
                     otherwise a generic placeholder query is used just to
                     surface the paper-level gates)
    """
    try:
        _ensure_ready()
        full_text, source, _, used_doi, used_pdf, used_raw_text = _extract_full_text_from_request()

        if not full_text or len(full_text.strip()) < 100:
            return jsonify(_extraction_error_payload(source)), 500

        probe_query = request.form.get("prompt", "Describe the moulting process in this species.")
        num_sentences = _resolve_num_sentences()
        result = _run_domain_pipeline(full_text, probe_query, num_summary_sentences=num_sentences)
        decision = result["decision"]

        return jsonify(
            {
                "source": source,
                "full_text_chars": len(full_text),
                "full_text_preview": full_text[:1000],
                "summary": result["summary"],
                "n_summary": decision["paper_summary_gate"]["n_summary_sentences"],
                "paper_taxonomy_gate": decision["paper_taxonomy_gate"],
                "paper_summary_gate": decision["paper_summary_gate"],
                "paper_is_relevant": (
                    decision["paper_taxonomy_gate"]["allow"]
                    and decision["paper_summary_gate"]["allow"]
                ),
                "used_doi": used_doi,
                "used_pdf": used_pdf,
                "used_raw_text": used_raw_text and not used_doi and not used_pdf,
                "evidence": _evidence_payload(result["evidence"]),
            }
        )

    except Exception as e:
        print(f"[LLM] ERROR in /preprocess: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


# ────────────────────── Main LLM endpoint ──────────────────────


@app.route("/query", methods=["POST"])
def query():
    """
    Main LLM endpoint.

    Form fields:
        - doi       (optional, string)
        - text      (optional, raw text)
        - prompt    (required, user query — still used for the query-gate
                     ontology check either way; in full_traits mode it also
                     acts as an optional focus hint, see build_user_content)
        - file      (optional, PDF upload)
        - model     (optional, one of the ids in GET /models;
                     defaults to config.models.DEFAULT_MODEL_ID)
        - mode      (optional, "full_traits" [default] or "single_trait")
                     - full_traits: asks for every one of the 55 MoultDB
                       trait fields the text supports evidence for (mirrors
                       the original local fine-tuned model's behaviour —
                       see config/trait_schema.py).
                     - single_trait: answers only the exact question in
                       `prompt`, ignores everything else (old default).

    Priority for text source:
        1) doi
        2) file (PDF)
        3) raw text

    Routing logic (domain.domain_gate.analyze_paper_and_query_domain):
        - Paper must show arthropod taxonomic signal.
        - Paper must contain enough moulting-related summary sentences.
        - Query must carry a moulting signal (ontology-based) and must not
          concern non-arthropod moulting.
    """
    t0 = time.time()

    user_prompt = request.form.get("prompt", "").strip()
    if not user_prompt:
        return jsonify({"error": "Missing 'prompt' in form data."}), 400

    mode = request.form.get("mode", DEFAULT_MODE).strip()
    if mode not in VALID_MODES:
        return (
            jsonify(
                {
                    "error": "unknown_mode",
                    "message": f"'{mode}' is not a valid mode.",
                    "available_modes": list(VALID_MODES),
                }
            ),
            400,
        )

    model_id = request.form.get("model", DEFAULT_MODEL_ID).strip()
    if not is_known_model(model_id):
        return (
            jsonify(
                {
                    "error": "unknown_model",
                    "message": f"'{model_id}' is not in the model registry.",
                    "available_models": [m["id"] for m in list_models()],
                }
            ),
            400,
        )

    try:
        _ensure_ready()
        full_text, source, saved_pdf_path, used_doi, used_pdf, used_raw_text = _extract_full_text_from_request()

        if not full_text or len(full_text.strip()) < 100:
            return jsonify(_extraction_error_payload(source)), 500

        num_sentences = _resolve_num_sentences()
        result = _run_domain_pipeline(full_text, user_prompt, num_summary_sentences=num_sentences)
        summary = result["summary"]
        decision = result["decision"]
        evidence = result["evidence"]

        print(
            f"[ROUTING] allow={decision['allow']} label={decision['final_label']} "
            f"model={model_id}"
        )

        if not decision["allow"]:
            return (
                jsonify(
                    {
                        "error": "out_of_scope",
                        "message": decision["message"],
                        "reason": decision["final_label"],
                        "paper_taxonomy_gate": decision["paper_taxonomy_gate"],
                        "paper_summary_gate": decision["paper_summary_gate"],
                        "query_gate": decision["query_gate"],
                        "evidence": _evidence_payload(evidence),
                    }
                ),
                400,
            )

        if not summary or len(summary.strip()) == 0:
            return (
                jsonify(
                    {
                        "error": "No relevant sentences could be extracted.",
                        "details": "The article may not contain moulting-related content.",
                    }
                ),
                500,
            )

        # ── Remote LLM call (no local GPU / weights) ──
        system_prompt = build_system_prompt(mode)
        user_content = build_user_content(summary, user_prompt, mode)
        max_new_tokens = MAX_NEW_TOKENS_SINGLE if mode == MODE_SINGLE_TRAIT else MAX_NEW_TOKENS_FULL

        try:
            llm_result = llm_generate(
                system_prompt=system_prompt,
                user_content=user_content,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
            )
        except ProviderError as e:
            print(f"[LLM] ERROR calling provider for model={model_id}: {e}")
            return (
                jsonify(
                    {
                        "error": "provider_error",
                        "message": str(e),
                        "model_id": model_id,
                    }
                ),
                502,
            )

        dt = time.time() - t0
        print(
            f"[LLM] /query completed in {dt:.2f}s "
            f"(source={source}, pdf={saved_pdf_path}, model={model_id})"
        )

        return jsonify(
            {
                "response": llm_result["text"],
                "mode": mode,
                "provider": llm_result["provider"],
                "model_id": llm_result["model_id"],
                "resolved_model_id": llm_result["resolved_model_id"],
                "model_label": llm_result["model_label"],
                "model_latency_sec": llm_result["latency_sec"],
                "total_latency_sec": dt,
                "used_doi": used_doi,
                "used_pdf": used_pdf,
                "used_raw_text": used_raw_text and not used_doi and not used_pdf,
                "routing_label": decision["final_label"],
                "n_relevant_sentences": decision["paper_summary_gate"]["n_summary_sentences"],
                "evidence": _evidence_payload(evidence),
            }
        )

    except Exception as e:
        print(f"[LLM] ERROR in /query: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


# ────────────────────── Feedback endpoint ──────────────────────


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Store user feedback for later analysis / model comparison.

    Expected JSON body:
        {
            "query": "...",
            "response": "...",
            "model_id": "...",       # which model produced the response
            "rating": int (e.g. 1-5 or -1/1),
            "comment": "optional free text"
        }
    """
    data = request.get_json(silent=True) or {}
    feedback_dir = LLM_ROOT / "backend" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_file = feedback_dir / "feedback.jsonl"

    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    print(f"[LLM] Starting LLM backend on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
