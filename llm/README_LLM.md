# 🐛 MoultGPT — LLM module (`llm/`)

Quickstart for this module only. For the full project (vision + LLM), repo
structure, and detailed API docs, see the top-level `README_.md`.

> **No GPU, no local model weights.** The LLM call goes to a remote
> provider's API (OpenRouter / Mistral / Gemini — see `config/models.py`),
> not to a model loaded in this process. This module only needs Python, a
> reachable GROBID service (for DOI/PDF input), and at least one API key.

---

## Setup

```bash
cd llm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: at least one of OPENROUTER_API_KEY / MISTRAL_API_KEY / GEMINI_API_KEY,
# plus UNPAYWALL_EMAIL if you'll resolve DOIs.
```

GROBID (only needed for DOI/PDF input, not for the LLM call itself):

```bash
cd tools/grobid
./gradlew run
```

## Run the backend

```bash
cd backend
python app.py
```

Endpoints: `GET /` (health), `GET /models` (registry), `POST /preprocess`
(summary + gating, no LLM call), `POST /query` (full pipeline), `POST
/feedback`. Full request/response examples are in the top-level
`README_.md`.

## Web UI

Open **http://localhost:5002/ui** (same Flask app, no separate server) for a
small built-in tester: pick a DOI / PDF / raw text source, write a query,
select one or several models from the registry, and see their responses
side by side — including latency and the gate decision if a request is
rejected. Source: `frontend/index.html` (single static file, no build step,
no npm).

## Compare models

```bash
python eval/compare_models.py --dataset eval/dataset_example.json --out eval/results.csv
```

Runs the same paper/query pairs through every model in `config/models.py`
and writes one CSV row per (paper, query, model) — see that script's
docstring for the dataset format and the `--models` flag to run a subset.

## Module layout

```
llm/
├── backend/
│   ├── app.py          # Flask API
│   ├── providers.py    # OpenRouter / Mistral / Gemini adapters
│   └── feedback/
├── config/
│   └── models.py       # candidate model registry
├── pipeline/
│   ├── downloader.py       # DOI -> PDF (Unpaywall)
│   ├── parser.py           # PDF -> TEI (GROBID) -> text
│   ├── processor.py        # input_to_text(...)
│   ├── summarization.py    # extract_relevant_sentences(...)
│   └── domain_pipeline.py  # run_domain_pipeline(...): summary + gate, shared by app.py and eval/
├── domain/
│   ├── optimized_taxonomy_lookup.py  # TaxonomyLookup
│   ├── moulting_ontology_gate.py     # MoultingOntologyGate (OWL-based)
│   └── domain_gate.py                # analyze_paper_and_query_domain(...)
├── eval/
│   └── compare_models.py
├── data/
│   ├── arthropod_taxonomy.csv
│   └── moultdb_moulting_ontology_v3_8.owl
├── finetuning/          # legacy local LoRA scripts, not on the current path
├── debug_preprocess.py  # CLI: DOI/PDF -> summary -> gates, no server needed
└── requirements.txt
```

## Debugging without the server

```bash
python debug_preprocess.py --pdf test/paper.pdf
# or
python debug_preprocess.py --doi 10.1038/s41598-022-18146-3
```

Runs summarization + the combined domain gate against a fixed list of test
queries and prints the decision for each — useful for checking the
taxonomy/ontology setup before touching the API.

---

## Acknowledgements

- [Mistral AI](https://docs.mistral.ai/) / [OpenRouter](https://openrouter.ai/docs) / [Google Gemini API](https://ai.google.dev/)
- [GROBID](https://github.com/kermitt2/grobid)
- [Unpaywall API](https://unpaywall.org/products/api)
- [rdflib](https://rdflib.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)

## Contact

Project lead: **Michele Leone** — https://www.moulting.org
