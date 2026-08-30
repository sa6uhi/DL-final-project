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

## Serving

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
streamlit run src/dashboard/app.py --server.port 8501
```
