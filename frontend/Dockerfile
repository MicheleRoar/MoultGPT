# MoultGPT — frontend/ (unified demo gateway) image
#
# The smallest image in this repo on purpose: just Flask + requests to
# serve index.html and reverse-proxy to the two real backends (see
# app.py's module docstring). No torch, no ML deps — this is not where
# any inference happens.

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py index.html ./

ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["python", "app.py"]
