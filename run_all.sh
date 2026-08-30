#!/usr/bin/env bash
# One-command deterministic reproduction pipeline.
# Expected to run from a clean checkout: data ingestion -> training ->
# evaluation -> serialization -> latency benchmark -> unit test gate.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON=${PYTHON:-}
if [ -z "$PYTHON" ] && [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
fi
PYTHON=${PYTHON:-python}
CONFIG=${CONFIG:-config/config.yaml}

if [ ! -f "$CONFIG" ]; then
    echo "error: config not found: $CONFIG" >&2
    exit 1
fi

run() {
    echo "==> ${1}"
    shift
    "$PYTHON" "$@"
}

MAIN_LOG="logs/run_all.log"
mkdir -p logs

{
echo "=== Pipeline started $(date -u +%F\ %T) ==="

# 0) Deterministic environment
export PYTHONHASHSEED=42
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 1) Data preparation & temporal split (Member A) - skipped when raw data absent
if [ -d "data/raw" ] && [ "$(find data/raw -type f | wc -l)" -gt 0 ]; then
    run "Data preparation step" src/data/temporal_split.py --input data/raw/ --output data/processed/
else
    echo "==> Data prep: no raw data mounted, skipping (run with data/raw mounted)."
fi

# 2) Baselines (Member A) - optional until raw data present
if [ -f "data/processed/features.npz" ]; then
    run "Baseline training" src/training/train_baselines.py --config "$CONFIG"
else
    echo "==> Baselines: processed features absent, skipping."
fi

# 3) Supervised FT-CAT transformer (Member B) - guarded by the same inputs
if [ -f "data/processed/features.npz" ]; then
    run "FT-CAT transformer training" src/training/train_transformer.py --config "$CONFIG"
else
    echo "==> Transformer: processed features absent, skipping."
fi

# 4) Semi-supervised DAE autoencoder (Member C)
run "DAE autoencoder training" src/training/train_autoencoder.py --config "$CONFIG"

# 5) Serialize model artifacts (EXIR + ONNX) with parity verification
run "Model serialization" src/serving/model_serializer.py --input models/checkpoints/ --output models/artifacts/

# 6) Exp-5 latency & throughput benchmark
run "Latency benchmark (Exp-5)" src/evaluation/latency_benchmark.py --config "$CONFIG"

# 7) Exp-6 anomaly evaluation when the scored archive is present
if [ -f "data/processed/anomaly_eval.npz" ]; then
    run "Anomaly evaluation (Exp-6)" src/evaluation/dae_anomaly_eval.py --config "$CONFIG" --archive data/processed/anomaly_eval.npz
else
    echo "==> Anomaly eval: scored archive absent, skipping."
fi

# 8) Unit test gate (>=80% coverage enforced by pytest)
run "Unit test gate" -m pytest --cov=src --cov-fail-under=80 tests/ -q

} 2>&1 | tee "$MAIN_LOG"

echo "=== Pipeline finished OK - full log: $MAIN_LOG ==="
