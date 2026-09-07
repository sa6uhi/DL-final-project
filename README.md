# Real-Time Financial Fraud Detection and Uncertainty Triage Engine

**Track 3: Industry Product** · DLE-AI-202 (Deep Learning Cohort I 2026) · AI Academy

A production-oriented fraud detection platform combining a semi-supervised Deep Denoising Autoencoder (DAE) for zero-day anomaly detection, an FT-CAT sequence model for supervised fraud probability estimation, a learned hybrid gating model, conformal uncertainty triage, explainability, and a FastAPI serving layer.

---

## Architecture Overview

```text
                        Inbound Transaction
                                 |
                +----------------+----------------+
                |                                 |
                v                                 v
       800-dim Numeric Features        Cardholder Sequence
                |                                 |
                v                                 v
      Denoising Autoencoder             FT-CAT Cross-Attention
      Reconstruction Error r(x)         Supervised Posterior P(y|x)
                |                                 |
                +----------------+----------------+
                                 |
                                 v
                      Learned Hybrid Gating
                                 |
                                 v
                    Calibrated Decision Policy
                 +---------------+---------------+
                 |               |               |
                 v               v               v
             APPROVE          BLOCK          ESCALATE
```

### Main components

* **Deep Denoising Autoencoder (DAE)**
  Semi-supervised anomaly detector trained on non-fraudulent transactions with feature corruption to identify structural anomalies through reconstruction residuals.

* **FT-CAT Transformer**
  Supervised transaction classifier that combines continuous, categorical, and historical sequence features through attention.

* **Learned Hybrid Gate**
  Learns when to trust the supervised model and when anomaly evidence should influence the final fraud probability.

* **Conformal Triage**
  Uses uncertainty information to identify borderline transactions that should be escalated for analyst review.

* **Explainability**
  SHAP-based explanations with DAE reconstruction-residual fallback for highlighting the most influential features.

* **Serving Layer**
  FastAPI application with `/predict`, `/stream`, `/explain`, `/health`, and `/metrics` endpoints.

---

## Repository Structure

```text
DL-final-project/
├── config/
├── data/
├── docker/
├── experiments/
├── figures/
├── models/
├── results/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── models/
│   ├── serving/
│   ├── training/
│   └── utils/
├── tests/
└── run_all.sh
```

---

## Experiments

The repository contains experiments covering the major parts of the system.

| Experiment                     | Purpose                                                     |
| ------------------------------ | ----------------------------------------------------------- |
| Baseline benchmark             | Compare classical fraud detection baselines using PR curves |
| Sequence window ablation       | Evaluate different transaction history lengths              |
| Hybrid gating ablation         | Compare gating strategies                                   |
| Hybrid gating seed robustness  | Measure robustness across random seeds                      |
| Conformal triage evaluation    | Evaluate uncertainty-based escalation                       |
| Autoencoder anomaly evaluation | Evaluate DAE anomaly detection                              |
| Latency benchmark              | Measure inference latency and throughput                    |
| SHAP consistency               | Validate explanation stability                              |

---

## Inference Benchmark

Inference benchmarks were run on the production checkpoints using warm-up iterations followed by repeated timed runs.

### Batch size 1

| Component                  | Mean latency | Throughput |
| -------------------------- | -----------: | ---------: |
| DAE                        |     0.465 ms | 2,151 tx/s |
| FT-CAT                     |     4.895 ms |   204 tx/s |
| Learned Gate               |     0.149 ms | 6,694 tx/s |
| End-to-End Neural Pipeline |     7.806 ms |   128 tx/s |

### Batch size 32

| Component                  | Mean latency |   Throughput |
| -------------------------- | -----------: | -----------: |
| DAE                        |     1.682 ms |  19,020 tx/s |
| FT-CAT                     |    15.890 ms |   2,014 tx/s |
| Learned Gate               |     0.179 ms | 178,763 tx/s |
| End-to-End Neural Pipeline |    15.297 ms |   2,092 tx/s |

### Batch size 128

| Component                  | Mean latency |   Throughput |
| -------------------------- | -----------: | -----------: |
| DAE                        |     2.471 ms |  51,805 tx/s |
| FT-CAT                     |    40.881 ms |   3,131 tx/s |
| Learned Gate               |     0.140 ms | 914,743 tx/s |
| End-to-End Neural Pipeline |    46.621 ms |   2,746 tx/s |

### Batch size 512

| Component                  | Mean latency |     Throughput |
| -------------------------- | -----------: | -------------: |
| DAE                        |     5.594 ms |    91,523 tx/s |
| FT-CAT                     |   234.346 ms |     2,185 tx/s |
| Learned Gate               |     0.224 ms | 2,288,902 tx/s |
| End-to-End Neural Pipeline |   190.157 ms |     2,693 tx/s |

The learned gate adds very little computational overhead, while the full neural pipeline remains suitable for real-time transaction scoring on CPU.

Benchmark artifacts are stored in:

```text
results/benchmark/inference_benchmark.csv
results/benchmark/inference_benchmark.json
```

---

## Model Serialization

The trained neural models are exported to both PyTorch EXIR (`.pt2`) and ONNX formats.

Serialized artifacts include:

```text
models/serialized/
├── autoencoder.pt2
├── autoencoder.onnx
├── ft_transformer.pt2
├── ft_transformer.onnx
├── hybrid_gating.pt2
└── hybrid_gating.onnx
```

Numerical parity tests verify that exported models produce outputs within the configured tolerance of the original PyTorch models.

---

## API Endpoints

The FastAPI service exposes the following endpoints.

| Method | Endpoint   | Description                           |
| ------ | ---------- | ------------------------------------- |
| POST   | `/predict` | Score a single transaction            |
| POST   | `/stream`  | Score a batch of transactions         |
| POST   | `/explain` | Return top risk drivers               |
| GET    | `/health`  | Health and model status               |
| GET    | `/metrics` | Request counts and latency statistics |

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_10492",
    "features": [0.12, -0.45, 1.89],
    "ft_probability": 0.04
  }'
```

### Example response

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

## Quickstart

### Local environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run the full pipeline:

```bash
./run_all.sh
```

Or execute the major stages individually:

```bash
python -m src.data.download_data
python -m src.data.prepare_data
python -m src.training.train_autoencoder --config config/config.yaml
python -m src.serving.model_serializer \
  --input models/checkpoints \
  --output models/serialized \
  --config config/config.yaml
python -m experiments.inference_benchmark
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

---

## Testing and Quality Gates

The project includes tests for:

* model serialization and numerical parity
* conformal prediction
* hybrid gating
* dashboard and explainability integration
* SHAP explanations
* serving API behavior
* experiment reproducibility

Run the focused quality suite:

```bash
python -m pytest \
  tests/test_shap_explainer.py \
  tests/test_dashboard_xai.py \
  tests/test_train_hybrid_gating.py \
  tests/test_hybrid_gating.py \
  tests/test_hybrid_gating_ablation.py \
  tests/test_conformal.py \
  tests/test_conformal_triage_eval.py \
  tests/test_serialization.py \
  -q
```

Current result:

```text
148 passed
```

Formatting and linting:

```bash
python -m black --check src tests experiments
python -m flake8 src tests experiments
```

---

## CI/CD

The GitHub Actions workflow performs:

* Black formatting checks
* Flake8 linting
* automated test execution
* Docker image packaging when manually triggered

---

## Key Outputs

```text
results/
├── benchmark/
│   ├── inference_benchmark.csv
│   └── inference_benchmark.json
├── conformal/
├── hybrid_gating/
└── explainability/

figures/
├── inference_benchmark/
├── conformal/
├── hybrid_gating/
└── explainability/
```

The repository is designed to provide reproducible training, evaluation, uncertainty triage, explainability, model serialization, and production-style serving for a modern fraud detection workflow.
