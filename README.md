# 🐛 MoultGPT

[![CI](https://github.com/MicheleRoar/MoultGPT/actions/workflows/ci.yml/badge.svg)](https://github.com/MicheleRoar/MoultGPT/actions/workflows/ci.yml)

Extract moulting information about arthropods from **scientific papers** and from **photographs**, in one demo.

![MoultGPT: viewer with example outputs on the left, controls on the right](output/ui_overview.png)

![MoultGPT: extracted traits as cards, with the evidence sentence](output/ui_traits.png)

## What it does

| | Papers (`llm/`) | Photographs (`vision/`) |
|---|---|---|
| **Input** | DOI, PDF or text, plus a question | An arthropod photo |
| **Output** | `Field: value` traits with the evidence sentence | Boxes and the moulting stage with a confidence |

It only answers about moulting in arthropods, and prefers saying nothing to guessing.

## Under the hood

**Papers**
- DOI → open-access PDF (Unpaywall) → TEI text (GROBID).
- Scope gates: a taxonomy lookup (regex index over GBIF/NCBI/iNaturalist names) and the MoultDB moulting ontology (OWL) must find arthropods and moulting content, and the question must not target non-arthropods. A failed gate means no LLM call.
- Sentence selection: ontology-scored sentences, diversified with TF-IDF + K-Means (about 20 sentences).
- Inference: remote models through a provider-agnostic layer (Mistral, OpenRouter, Gemini), temperature 0, no GPU. Default output is one `Field: value` line per supported trait, unsupported fields skipped.
- Also in the repo: a model-comparison harness, a trait-extraction benchmark against MoultDB annotations, a query-aware retrieval module (`llm/retrieval/`), and a LoRA/QLoRA + DPO fine-tuning track served with vLLM, kept separate from the live path.

**Photographs**
- Detector: YOLO11n fine-tuned on two classes, `organism` and `exuviae`. If a whole-image pass finds nothing, one SAHI tiled pass runs (confidence 0.10, overlap 0.35, 900 px slices).
- Features: box IoU, centroid distance, box positions, exuvia height, mean colour of the organism region, clade one-hot, and indicators for which of the two objects was detected.
- Stage classifier: XGBoost, `moulting` vs. `post-moult`. Training rows are expanded into full / organism-masked / exuvia-masked variants so one model handles partial detections; `scale_pos_weight` is computed from the data.
- Evaluation uses an observation-grouped split (`StratifiedGroupKFold` on the iNaturalist observation id), so photos of one observation never fall on both sides.
- Data are iNaturalist images (CC0, CC-BY, CC-BY-NC), turned into YOLO labels and feature tables by the scripts in `vision/utility/` and `vision/scripts/pipeline/`.

## Quick start

Needs Docker and one LLM provider key (free tiers work).

```bash
git clone https://github.com/MicheleRoar/MoultGPT.git && cd MoultGPT
cp llm/.env.example llm/.env      # add e.g. MISTRAL_API_KEY=...
docker compose up --build         # then open http://localhost:8080
```

Four services start: GROBID, the LLM backend (:5002), the vision backend (:5001) and a small gateway (:8080) that serves the page and proxies both. To run on a server (HTTPS, rate limits, upload cap) see [DEPLOY.md](DEPLOY.md). Tests: `cd llm && python -m pytest tests -q`.

## Status

Research code with a manuscript in preparation.

- The served vision backend uses an older three-class model (`post-moult`, `moulting`, `exuviae`). The unified binary classifier above is evaluated in `vision/scripts/pipeline/unified_classifier_eval.py` but not yet served.
- Vision accuracy is not quoted: the train/evaluation split is being re-audited for overlap.
- Text results are in `llm/eval/trait_extraction/results/report.md`, with their limitations (small gold set, LLM judge).

## Repository

```text
llm/        text pipeline, gates, evaluation, fine-tuning track
vision/     detector, stage classifier, training and evaluation scripts
frontend/   demo page and gateway
docs/       full technical reference
```

More: [technical reference](docs/DETAILS.md) · [LLM backend](llm/README_LLM.md) · [deployment](DEPLOY.md)

## Contact

Michele Leone · https://www.moulting.org
