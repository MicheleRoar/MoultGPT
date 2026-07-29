# MoultGPT — vision/ backend image (YOLO + XGBoost + FastSAM, CPU inference)
#
# Model weights (vision/models/*.pt, *.pkl) are NOT baked into the image —
# mount them as a volume (see docker-compose.yml) so the image stays
# reasonably sized and weights can be swapped without a rebuild.

FROM python:3.10-slim AS base

# libgl1/libglib2.0-0: required by opencv (pulled in transitively by
# ultralytics) even in headless deployments — omitting them causes an
# ImportError at container start, not at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install the CPU-only torch wheel explicitly first (the default PyPI wheel
# pulls in ~2GB of CUDA/nvidia-* deps that are useless without a GPU); pip
# then sees torch==2.3.0 already satisfied when installing the rest.
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/

ENV MODEL_PATH=/app/models/yolo_detect.pt \
    STAGE_MODEL_PATH=/app/models/xgboost_stage.pkl \
    LABEL_ENCODER_PATH=/app/models/label_encoder.pkl \
    FASTSAM_WEIGHTS=/app/models/FastSAM-x.pt \
    USE_CUDA=false \
    USE_MPS=false \
    PORT=5001

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5001/healthz || exit 1

WORKDIR /app/backend
CMD ["python", "app.py"]
