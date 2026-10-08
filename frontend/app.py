# frontend/app.py
"""
MoultGPT — unified demo gateway.

Deliberately the *only* new "third process" in the stack, and deliberately
tiny (Flask + requests, nothing else). It does two things:

  1. Serves frontend/index.html — the single-page demo that lets a visitor
     scan either a paper (text pipeline) or an image (vision pipeline)
     from one interface.
  2. Reverse-proxies /api/llm/<path> and /api/vision/<path> to the two real
     backends (llm/backend/app.py, vision/backend/app.py), so the browser
     only ever talks to one origin — no CORS juggling, no "which port was
     vision on again" for whoever is driving the demo.

Why a proxy instead of importing both backends into one process: llm/
and vision/ have deliberately disjoint dependency footprints (llm/ has no
torch/GPU deps at all; vision/ needs torch+ultralytics+xgboost — see the
header comments in llm/Dockerfile and vision/Dockerfile). Merging them
into a single Python process would mean a single environment needs both
sets of dependencies installed together, undoing that split for no benefit
other than "one file" — this gateway gets the same "one command, one URL"
result without that cost, and without touching either backend's code.

Run locally (after the two backends are already running on their usual
ports):
    cd frontend
    pip install -r requirements.txt
    python app.py
    # open http://localhost:8080

Run via Docker: this is the `gateway` service in the root docker-compose.yml
— it starts after llm-backend and vision-backend and talks to them by their
service names on the internal compose network.

Environment variables:
    PORT               (default: 8080)
    LLM_BACKEND_URL     (default: http://localhost:5002)
    VISION_BACKEND_URL  (default: http://localhost:5001)
    PROXY_TIMEOUT_SEC   (default: 120 — text/vision inference can be slow)
"""

import os
from pathlib import Path

import requests
from flask import Flask, Response, request, send_from_directory

FRONTEND_DIR = Path(__file__).resolve().parent

PORT = int(os.getenv("PORT", 8080))
LLM_BACKEND_URL = os.getenv("LLM_BACKEND_URL", "http://localhost:5002").rstrip("/")
VISION_BACKEND_URL = os.getenv("VISION_BACKEND_URL", "http://localhost:5001").rstrip("/")
PROXY_TIMEOUT_SEC = float(os.getenv("PROXY_TIMEOUT_SEC", 120))

app = Flask(__name__)

# --- Production hardening (all opt-out via env, safe defaults for local dev) ---
# Upload cap: PDFs / photos. Requests above this get a 413 before they are
# proxied anywhere.
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", 20)) * 1024 * 1024

# Behind a reverse proxy (Caddy / Cloudflare Tunnel) the real client IP is in
# X-Forwarded-For. Only trust it when explicitly told we are behind one,
# otherwise anyone could spoof their IP and dodge the rate limiter.
if os.getenv("TRUST_PROXY", "0") == "1":
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Per-IP rate limiting. The expensive endpoints (remote LLM call = money,
# YOLO inference = CPU) get tight limits. In-memory storage is fine because
# the gateway runs as a single gunicorn worker (see docker-compose.prod.yml).
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "0") == "1"
LIMIT_LLM_QUERY = os.getenv("LIMIT_LLM_QUERY", "5 per minute;40 per day")
LIMIT_VISION = os.getenv("LIMIT_VISION", "10 per minute;100 per day")
LIMIT_DEFAULT = os.getenv("LIMIT_DEFAULT", "120 per minute")

if RATE_LIMIT_ENABLED:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app, default_limits=[LIMIT_DEFAULT],
                      storage_uri="memory://")
else:
    class _NoLimiter:
        def limit(self, *_a, **_k):
            return lambda f: f
    limiter = _NoLimiter()

# Only these backend paths are reachable through the public gateway. Anything
# else (e.g. the LLM tester /ui, debug routes) stays internal-only.
ALLOWED_LLM_PATHS = {"", "models", "preprocess", "query", "feedback"}
ALLOWED_VISION_PATHS = {"healthz", "predict_image"}
ENFORCE_ALLOWLIST = os.getenv("ENFORCE_ALLOWLIST", "0") == "1"

# Hop-by-hop headers that must not be forwarded verbatim between proxy hops
# (RFC 7230 §6.1) — forwarding these breaks chunked responses / connection
# handling on either side of the proxy.
HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
    "content-length",
}


def _proxy(target_base: str, path: str) -> Response:
    """Forward the current Flask request to target_base/path and relay the response."""
    url = f"{target_base}/{path}"

    # Whenever the ORIGINAL request was form-encoded (multipart, with or
    # without actual files, or a plain form post), `requests` re-encodes
    # the body itself from `data=`/`files=` -- as multipart with a brand
    # new boundary if `files` is non-empty, or as
    # application/x-www-form-urlencoded if there are no files at all. In
    # either case the resulting body's real encoding no longer matches
    # whatever the *original* incoming Content-Type header declared (old
    # boundary, or "multipart" when the new body is actually urlencoded),
    # so that header must be dropped and left for `requests` to set
    # correctly -- otherwise the upstream backend fails to parse the body
    # and request.form/request.files come back empty even though the
    # bytes were sent (this bit both the image-upload and the DOI/prompt
    # text-only form paths, for the two different reasons above). Only
    # for a genuine raw body (JSON etc, the request.get_data() branch) do
    # we keep forwarding the original Content-Type, since that body passes
    # through completely unchanged.
    is_form_request = bool(request.form) or bool(request.files)
    excluded_headers = {"host", "content-length"} | ({"content-type"} if is_form_request else set())

    try:
        upstream = requests.request(
            method=request.method,
            url=url,
            params=request.args,
            # Forward the raw body for JSON/text requests, and the parsed
            # multipart form/files separately so file uploads (PDFs,
            # images) survive the hop intact rather than being re-encoded.
            data=request.form if request.form else request.get_data(),
            files={k: (f.filename, f.stream, f.mimetype) for k, f in request.files.items()} or None,
            headers={k: v for k, v in request.headers if k.lower() not in excluded_headers},
            timeout=PROXY_TIMEOUT_SEC,
            stream=True,
        )
    except requests.exceptions.ConnectionError:
        return Response(
            f'{{"error":"gateway_upstream_unreachable","target":"{target_base}",'
            f'"hint":"Is this backend running? (see README for how to start it)"}}',
            status=502,
            mimetype="application/json",
        )
    except requests.exceptions.Timeout:
        return Response(
            f'{{"error":"gateway_upstream_timeout","target":"{target_base}"}}',
            status=504,
            mimetype="application/json",
        )

    response_headers = [
        (k, v) for k, v in upstream.raw.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS
    ]
    return Response(upstream.content, status=upstream.status_code, headers=response_headers)


def _not_allowed():
    return Response('{"error":"not_found"}', status=404, mimetype="application/json")


# Only /query and /preprocess are expensive on the LLM side (remote model call
# / full pipeline); everything else falls under the default limit.
_EXPENSIVE_LLM_PATHS = ("query", "preprocess")


@app.route("/api/llm/", defaults={"path": ""}, methods=["GET", "POST"])
@app.route("/api/llm/<path:path>", methods=["GET", "POST"])
@limiter.limit(LIMIT_LLM_QUERY, methods=["POST"],
               exempt_when=lambda: (request.view_args or {}).get("path") not in _EXPENSIVE_LLM_PATHS)
def proxy_llm(path):
    if ENFORCE_ALLOWLIST and path not in ALLOWED_LLM_PATHS:
        return _not_allowed()
    return _proxy(LLM_BACKEND_URL, path)


@app.route("/api/vision/", defaults={"path": ""}, methods=["GET", "POST"])
@app.route("/api/vision/<path:path>", methods=["GET", "POST"])
@limiter.limit(LIMIT_VISION, methods=["POST"])
def proxy_vision(path):
    if ENFORCE_ALLOWLIST and path not in ALLOWED_VISION_PATHS:
        return _not_allowed()
    return _proxy(VISION_BACKEND_URL, path)


@app.route("/healthz")
def healthz():
    """Gateway's own health check — separate from the two backends' /healthz and /."""
    return {"ok": True, "llm_backend_url": LLM_BACKEND_URL, "vision_backend_url": VISION_BACKEND_URL}


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    print(f"[GATEWAY] Serving unified demo on http://0.0.0.0:{PORT}")
    print(f"[GATEWAY] Proxying /api/llm/*    -> {LLM_BACKEND_URL}")
    print(f"[GATEWAY] Proxying /api/vision/* -> {VISION_BACKEND_URL}")
    app.run(host="0.0.0.0", port=PORT)
