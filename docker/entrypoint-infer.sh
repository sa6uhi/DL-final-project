#!/usr/bin/env sh
# Inference container entrypoint.
# Launches uvicorn with an env-overridable worker count (default 1, since
# inference is CPU-bound and each worker loads its own model copy).
set -eu

WORKERS="${UVICORN_WORKERS:-1}"

exec uvicorn src.serving.api:app \
    --host "${UVICORN_HOST:-0.0.0.0}" \
    --port "${UVICORN_PORT:-8000}" \
    --workers "${WORKERS}"
