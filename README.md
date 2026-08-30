# Real-Time Financial Fraud Detection, Temporal Cross-Attention, and Calibrated Uncertainty Triage Engine

**Deep Learning Cohort I 2026 (DLE-AI-202)** · AI Academy · National Artificial Intelligence Center

End-to-end deep learning platform for real-time financial fraud detection combining:

- **FT-CAT:** Feature-Tokenizer Cross-Attention Transformer contextualizing current transactions against historical cardholder sequences ($K=5$).
- **Deep DAE:** Semi-supervised Deep Denoising Autoencoder for zero-day anomaly detection.
- **Conformal Prediction:** Finite-sample 99% coverage guarantees with Auto-Approve / Auto-Block / Analyst Triage decision policies.
- **High-Throughput Serving:** FastAPI microservice with TorchScript / ONNX serialization ($<15$ ms P99 on CPU), Docker Compose stack, and Streamlit triage dashboard.

## Repository Layout

```
src/
├── utils/          Centralized config loader, structured logger, deterministic seeder
├── models/         FT-CAT transformer, deep denoising autoencoder, hybrid gating scorer
├── training/       Training entry points
├── serving/        FastAPI microservice, Pydantic schemas, model serialization
├── uncertainty/    Conformal prediction engine
├── explainability/ SHAP explainer
├── dashboard/      Streamlit analyst triage UI
├── evaluation/     Metrics, cost matrix, latency benchmarks
└── data/           Temporal split, preprocessing, sequence extraction

config/config.yaml   Centralized configuration (all hyperparameters, paths, seeds)
docker/              Multi-stage Dockerfile and docker-compose stack
report/              IEEE two-column paper
presentation/        Defense slide deck
tests/               Unit test suite (>= 80% coverage enforced)
experiments/         Exp 1-7 experiment scripts
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Full Reproduction

```bash
chmod +x run_all.sh
./run_all.sh
```

`run_all.sh` runs the deterministic pipeline from a clean checkout: data prep
(if raw data is mounted) → baselines → FT-CAT → DAE → serialization → latency
benchmark → anomaly evaluation → unit-test coverage gate. Steps whose inputs
are absent are skipped with a clear log line. Full output is captured to
`logs/run_all.log`.

## Model Serialization (EXIR + ONNX)

The serving stack ships both an EXIR graph and a standard ONNX model, each
verified for numerical parity against the source model:

```bash
python src/serving/model_serializer.py \
    --input models/checkpoints/ \
    --output models/checkpoints/ \
    --config config/config.yaml
```

If no trained checkpoint is present, a deterministic reference scoring model is
exported so the stack is exercisable before training artifacts exist.

## Serving

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
streamlit run src/dashboard/app.py --server.port 8501
```

## Docker

Two images are provided under `docker/`:

- **`Dockerfile.infer`** — lean multi-stage inference image (venv built in a
  builder stage, only the venv + code shipped to a non-root `slim` runtime).
  CPU-only PyTorch; models are role-mounted at `/app/models/checkpoints`.
  Exposes `:8000` with a `/health`-based `HEALTHCHECK`.
- **`Dockerfile.train`** — full training stack.

```bash
# Start the API service
docker compose -f docker/docker-compose.yml up --build

# Also run the training container
docker compose --profile train -f docker/docker-compose.yml up --build
```

Mount your `models/checkpoints/` volume (with `autoencoder.pt` / `autoencoder.onnx`)
at deploy time so the API serves your trained artifacts.

## CI

`.github/workflows/ci.yml` runs two jobs:

- **`lint`** — black + flake8 (no torch needed).
- **`test`** — `pytest --cov=src --cov-fail-under=80 tests/`.

Both use `uv` with caching across runs; requirements are split into
`requirements.txt` (runtime), `requirements-format.txt` (lint), and
`requirements-ci.txt` (test). Branch protection should require both `lint`
and `test`.
