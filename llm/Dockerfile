# MoultGPT — llm/ backend image
#
# Lightweight on purpose: this backend calls REMOTE LLM APIs (OpenRouter /
# Mistral / Gemini), so there is no torch/transformers/GPU in this image.
# Domain data (taxonomy CSV, ontology OWL, taxonomy pickle cache) is NOT
# baked in here — mount it as a volume (see docker-compose.yml) so the
# image stays small and the data can be updated without a rebuild.

FROM python:3.10-slim AS base

# System deps: lxml/rdflib need libxml2/libxslt at build time; curl is
# handy for the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the code the backend actually needs at runtime.
# llm/data, llm/finetuning, llm/test, llm/tools/grobid are intentionally
# excluded (see .dockerignore) — data is volume-mounted, finetuning/tools
# are dev-only.
COPY backend/ backend/
COPY pipeline/ pipeline/
COPY domain/ domain/
COPY config/ config/
COPY eval/ eval/
COPY frontend/ frontend/

# Default paths match docker-compose's volume mount point for ./llm/data.
# PYTHONUNBUFFERED: without this, Python buffers stdout when it isn't a TTY
# (i.e. always, inside a container), so this app's print()-based debug trail
# ([DEBUG]/[UNPAYWALL]/[!] lines in pipeline/processor.py and
# pipeline/downloader.py) sits in a buffer and never reaches `docker compose
# logs` -- only Werkzeug's own request-logging (flushed separately) shows
# up, which is why "POST /query 500" appeared with no explanation above it.
ENV TAXONOMY_CSV_PATH=/app/data/arthropod_taxonomy.csv \
    TAXONOMY_PICKLE_PATH=/app/data/taxonomy_lookup.pkl \
    ONTOLOGY_OWL_PATH=/app/data/moultdb_moulting_ontology_v3_8.owl \
    PORT=5002 \
    PYTHONUNBUFFERED=1

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5002/ || exit 1

WORKDIR /app/backend
CMD ["python", "app.py"]
