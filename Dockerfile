# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY scripts/export_onnx.py ./scripts/
RUN pip install --no-cache-dir -r requirements.txt "optimum[onnxruntime]" onnx onnxscript \
    && python ./scripts/export_onnx.py \
    && find /build/onnx-model -type f -printf '%p\n' | head -20

FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DATA_DIR=/app/data \
    SEEDS_PATH=/app/seeds/sales-faq.jsonl \
    ONNX_MODEL_DIR=/app/onnx-model

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash chatbrain

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
    && find /usr/local/lib/python3.11/site-packages/transformers/models -mindepth 1 -maxdepth 1 \
       ! -name 'bert' ! -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.11/site-packages -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /usr/local/lib/python3.11/site-packages/sympy \
              /usr/local/lib/python3.11/site-packages/mpmath \
              /usr/local/lib/python3.11/site-packages/pip \
              /usr/local/lib/python3.11/site-packages/setuptools \
              /usr/local/lib/python3.11/site-packages/hf_xet

COPY --from=builder /build/onnx-model /app/onnx-model
COPY app ./app
COPY seeds ./seeds

RUN mkdir -p /app/data && chown -R chatbrain:chatbrain /app

USER chatbrain

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]