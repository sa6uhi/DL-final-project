# Real-Time Financial Fraud Detection and Uncertainty Triage Engine

**Track 3: Industry Product** · DLE-AI-202 (Deep Learning Cohort I 2026) · AI Academy

A high-throughput, production-grade fraud detection platform combining a semi-supervised Deep Denoising Autoencoder (zero-day anomaly detection via reconstruction residuals) with a calibrated hybrid gating engine, served via a sub-15ms CPU FastAPI microservice.

---

## Architecture Overview

```
                        [ Inbound Transaction ]
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
       [ 800-dim Numeric Features ]      [ Cardholder Sequence ]
                  │                                 │
                  ▼                                 ▼
      [ Denoising Autoencoder ]          [ FT-CAT Cross-Attention ]
      Reconstruction Error r(x)          Supervised Posterior P(y|x)
                  │                                 │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                      [ Learned Hybrid Gating ]
                                   │
                                   ▼
                     [ Calibrated Decision Policy ]
                     ├── Auto-Approve  (p < 0.10)
                     ├── Auto-Block    (p > 0.85)
                     └── Escalate to Analyst Triage
```

* **Deep Denoising Autoencoder (DAE):** 5-layer bottleneck network ($800 \to 256 \to 128 \to 32 \to 128 \to 256 \to 800$) with LeakyReLU ($\alpha=0.2$) and batch normalization. Trained exclusively on non-fraudulent transactions with 10% feature corruption to identify zero-day structural anomalies via reconstruction residuals.
* **Calibrated Hybrid Gating:** Combines supervised sequence posteriors with unsupervised DAE anomaly percentiles, delegating borderline or high-uncertainty transactions to human analysts.
* **Optimized Serving Layer:** Compiled via ONNX Runtime and PyTorch EXIR, achieving sub-2ms P99 inference on standard commodity CPU.

---

## Production Latency Benchmarks (CPU)

Measured on standard CPU across 200 timed evaluation runs on the 800-dimensional production checkpoint:

| Backend | Batch Size | P50 Latency | P90 Latency | P95 Latency | P99 Latency (SLA) | Throughput | SLA Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Eager DAE** | 1 | 0.75 ms | 0.78 ms | 0.79 ms | 0.80 ms | 1,320 req/s | PASS (< 15 ms) |
| **PyTorch Eager DAE** | 8 | 0.29 ms | 0.43 ms | 0.45 ms | 0.50 ms | 24,175 req/s | PASS (< 15 ms) |
| **PyTorch Eager DAE** | 32 | 0.44 ms | 0.55 ms | 0.58 ms | 0.63 ms | 69,318 req/s | PASS (< 15 ms) |
| **PyTorch Eager DAE** | 64 | 0.63 ms | 0.69 ms | 0.72 ms | 0.79 ms | 99,380 req/s | PASS (< 15 ms) |
| **PyTorch EXIR (`.pt2`)** | 1 | 0.87 ms | 0.90 ms | 0.91 ms | 0.99 ms | 1,156 req/s | PASS (< 15 ms) |
| **PyTorch EXIR (`.pt2`)** | 8 | 0.80 ms | 0.81 ms | 0.83 ms | 0.85 ms | 9,937 req/s | PASS (< 15 ms) |
| **PyTorch EXIR (`.pt2`)** | 32 | 0.98 ms | 0.98 ms | 0.99 ms | 1.02 ms | 32,672 req/s | PASS (< 15 ms) |
| **PyTorch EXIR (`.pt2`)** | 64 | 1.24 ms | 1.25 ms | 1.28 ms | 1.32 ms | 51,544 req/s | PASS (< 15 ms) |
| **ONNX Runtime** | 1 | 0.06 ms | 0.09 ms | 0.11 ms | 1.27 ms | 10,259 req/s | PASS (12x faster) |
| **ONNX Runtime** | 8 | 0.09 ms | 0.11 ms | 0.11 ms | 0.13 ms | 85,042 req/s | PASS (< 15 ms) |
| **ONNX Runtime** | 32 | 0.26 ms | 0.27 ms | 0.27 ms | 0.28 ms | 123,795 req/s | PASS (< 15 ms) |
| **ONNX Runtime** | 64 | 0.40 ms | 0.45 ms | 0.48 ms | 0.52 ms | 154,868 req/s | PASS (< 15 ms) |

### End-to-End FastAPI Microservice Latency

Measured via 500 timed sequential HTTP POST requests to `/predict` (including Pydantic V2 schema validation, ONNX scoring, learned gating, and JSON serialization):

| Metric | Measured Value | SLA Target | Compliance |
| :--- | :---: | :---: | :---: |
| **P50 Latency** | 5.17 ms | — | Fast path |
| **P90 Latency** | 5.77 ms | — | Stable distribution |
| **P95 Latency** | 5.95 ms | — | Bounded tail |
| **P99 Latency** | **6.21 ms** | **< 15.00 ms** | **PASS (58.6% margin)** |
| **Mean Latency** | 5.25 ms | — | Sub-6ms average |
| **Numerical Parity** | $\le 1.22 \times 10^{-4}$ ($\le 1\text{ ULP}$) | $< 2.0 \times 10^{-3}$ | **EXACT MATCH** |

---

## Quickstart & Deployment

### Option A: Docker Compose (Prebuilt Images)

Docker automatically pulls prebuilt images from the GitHub Container Registry (`ghcr.io/sa6uhi/dl-final-project-api:v1.0-final`):

```bash
# Start the FastAPI inference microservice on port 8000
docker compose -f docker/docker-compose.yml up
```

To build locally from source instead of pulling from GHCR:

```bash
docker compose -f docker/docker-compose.yml up --build
```

To run the containerized training pipeline:

```bash
docker compose --profile train -f docker/docker-compose.yml up
```

### Option B: Local Environment Reproduction

```bash
# 1. Environment setup
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. End-to-end automated pipeline
./run_all.sh
```

Or execute individual pipeline steps sequentially:

```bash
# Data download from IEEE-CIS mirror & temporal split (70/15/15)
python -m src.data.download_data
python -m src.data.prepare_data

# Train Semi-Supervised DAE
python src/training/train_autoencoder.py --config config/config.yaml

# Model Serialization (Torch EXIR + ONNX)
python src/serving/model_serializer.py --input models/checkpoints/ --output models/artifacts/

# Latency and SLA Benchmarks
python src/evaluation/latency_benchmark.py --config config/config.yaml

# Launch FastAPI Microservice
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

---

## API Specification

The service exposes high-performance REST endpoints on `http://localhost:8000`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Evaluates a single transaction; returns anomaly score, calibrated fraud probability, and policy decision (`APPROVE`, `BLOCK`, `ESCALATE`). |
| `POST` | `/stream` | Batched scoring for streaming ingestion (up to 256 transactions per payload). |
| `POST` | `/explain` | Feature attribution explaining top risk drivers for a flagged transaction. |
| `GET` | `/health` | Service health, uptime, and checkpoint load verification. |
| `GET` | `/metrics` | Operational self-monitoring: request volume, error counts, and latency percentiles (P50, P99). |

### Example Request (`POST /predict`):

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "transaction_id": "tx_10492",
       "features": [0.12, -0.45, 1.89, ...],
       "ft_probability": 0.04
     }'
```

### Example Response:

```json
{
  "transaction_id": "tx_10492",
  "is_fraud": false,
  "fraud_probability": 0.038,
  "anomaly_score": 0.114,
  "decision": "APPROVE",
  "gate_used": true,
  "latency_ms": 1.12
}
```

---

## Testing & Quality Gates

The test suite enforces zero lookahead leakage, numerical stability, schema validation, and minimum 80% line coverage:

```bash
# Run complete test suite with coverage enforcement
pytest --cov=src --cov-report=term-missing --cov-fail-under=80 tests/

# Code formatting and linting
black --check src tests
flake8 src tests
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every pull request and push to `master`:
* **Linting Job:** Validates formatting (`black`) and PEP 8 compliance (`flake8`).
* **Testing Job:** Runs the complete test suite ensuring $\ge 80\%$ test coverage.
* **Docker Packaging (Manual Trigger):** A `workflow_dispatch` stage builds and publishes multi-stage OCI images to GitHub Container Registry (`ghcr.io/sa6uhi/dl-final-project-api:v1.0-final` and `ghcr.io/sa6uhi/dl-final-project-train:v1.0-final`). Trigger manually via GitHub CLI (`gh workflow run ci.yml`) or the Actions tab.
