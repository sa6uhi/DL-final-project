# Real-Time Financial Fraud Detection — Triage Engine

**Track 3: Industry Product** · DLE-AI-202 (Deep Learning Cohort I 2026) · AI Academy

Real-time fraud detection platform combining a semi-supervised denoising autoencoder
(zero-day anomaly detection) with a calibrated hybrid gating scorer, served through a
high-throughput FastAPI microservice.

## How to run

### Without Docker

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Train, serialize, benchmark, and verify the whole pipeline:

```bash
./run_all.sh
```

Or run steps individually:

```bash
python src/training/train_autoencoder.py --config config/config.yaml   # train DAE
python src/serving/model_serializer.py --input models/checkpoints/ --output models/artifacts/  # EXIR + ONNX
python src/evaluation/latency_benchmark.py --config config/config.yaml # latency benchmark

uvicorn src.serving.api:app --host 0.0.0.0 --port 8000                  # serve
```

Endpoints: `POST /predict`, `POST /stream`, `GET /health`, `GET /metrics`.

Run tests (>= 80% coverage enforced):

```bash
pytest --cov=src --cov-fail-under=80 tests/
```

### With Docker

```bash
# Start the API
docker compose -f docker/docker-compose.yml up --build

# Also run the training container
docker compose --profile train -f docker/docker-compose.yml up --build
```

Inference is CPU-bound, so the API service defaults to 1 uvicorn worker (each
worker loads its own model copy). Adjust the worker count via
`UVICORN_WORKERS` in `docker/docker-compose.yml`.

## CI

`.github/workflows/ci.yml` runs two jobs (`lint`, `test`) using cached `uv` installs:
black + flake8, and `pytest --cov=src --cov-fail-under=80 tests/`.
