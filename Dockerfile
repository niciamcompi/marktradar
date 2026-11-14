# syntax=docker/dockerfile:1
# Multi-stage build:
#   1. "base":    Python environment with all dependencies (exactly pinned by uv.lock)
#   2. "train":   downloads the BAG data, validates it and trains the model
#   3. "runtime": slim image with only code, model and processed data, non-root
# Result: one image = one reproducible model version.
#
# Corporate networks (internal registry, TLS proxy with its own certificate):
#   docker build \
#     --build-arg PYTHON_IMAGE=registry.example.ch/python:3.12-slim \
#     --secret id=ca_bundle,src=/path/to/corporate-ca.pem \
#     -t marktradar .
# The certificate is only mounted during the build and never ends up in the image.

ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS base
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app
# Helper: applies the optional corporate CA certificate for pip, uv and httpx
RUN printf '#!/bin/sh\nif [ -f /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle PIP_CERT=/run/secrets/ca_bundle; fi\nexec "$@"\n' \
      > /usr/local/bin/with-ca && chmod +x /usr/local/bin/with-ca
RUN --mount=type=secret,id=ca_bundle,required=false \
    with-ca pip install --no-cache-dir uv==0.11.32
# Dependencies first -> the Docker cache stays valid as long as uv.lock is unchanged
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=secret,id=ca_bundle,required=false \
    with-ca uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY config ./config
RUN --mount=type=secret,id=ca_bundle,required=false \
    with-ca uv sync --frozen --no-dev

FROM base AS train
RUN --mount=type=secret,id=ca_bundle,required=false \
    with-ca uv run --no-sync marktradar download \
 && uv run --no-sync marktradar build \
 && uv run --no-sync marktradar train

FROM ${PYTHON_IMAGE} AS runtime
WORKDIR /app
RUN useradd --create-home --uid 10001 app
COPY --from=base /app /app
COPY --from=train /app/models ./models
COPY --from=train /app/data/processed ./data/processed
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["marktradar", "serve", "--host", "0.0.0.0", "--port", "8000"]
