"""Streamlit dashboard for the real-time fraud triage engine.

The dashboard surfaces real model checkpoints, experiment artifacts, and
evaluation outputs. Missing artifacts are reported honestly rather than
replaced with fabricated operational metrics.

Run with:

    python -m streamlit run src/dashboard/app.py
"""

# Import necessary modules and libraries
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final
from urllib import error, request

import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR: Final[Path] = PROJECT_ROOT / "models" / "checkpoints"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "figures"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"

API_BASE_URL: Final[str] = os.getenv(
    "FRAUD_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")
API_TIMEOUT_SECONDS: Final[float] = 10.0

DAE_CHECKPOINT: Final[Path] = CHECKPOINT_DIR / "autoencoder.pt"
FT_CHECKPOINT: Final[Path] = CHECKPOINT_DIR / "ft_transformer.pt"
GATE_CHECKPOINT: Final[Path] = CHECKPOINT_DIR / "hybrid_gating.pt"

TSNE_FIGURE: Final[Path] = FIGURES_DIR / "autoencoder_latent" / "tsne_latent_space.png"

SHAP_GLOBAL_FIGURE: Final[Path] = FIGURES_DIR / "explainability" / "dae_shap_global_importance.png"

SHAP_LOCAL_FIGURE: Final[Path] = FIGURES_DIR / "explainability" / "dae_shap_waterfall.png"

SHAP_CONSISTENCY_FIGURE: Final[Path] = FIGURES_DIR / "explainability" / "dae_shap_consistency.png"

SHAP_CONSISTENCY_RESULTS: Final[Path] = RESULTS_DIR / "explainability" / "dae_shap_consistency.json"

TEST_DATA: Final[Path] = PROJECT_ROOT / "data" / "processed" / "test.parquet"
FT_FEATURE_SPEC: Final[Path] = PROJECT_ROOT / "data" / "processed" / "ft_cat_features.json"
CONFIG_PATH: Final[Path] = PROJECT_ROOT / "config" / "config.yaml"

CONFORMAL_COVERAGE_FIGURE: Final[Path] = FIGURES_DIR / "conformal" / "coverage_vs_workload.png"

CONFORMAL_THRESHOLD_FIGURE: Final[Path] = FIGURES_DIR / "conformal" / "threshold_sensitivity.png"

CONFORMAL_RESULTS: Final[Path] = RESULTS_DIR / "conformal" / "conformal_metrics.json"

CONFORMAL_METADATA: Final[Path] = RESULTS_DIR / "conformal" / "metadata.json"

GATE_DISAGREEMENT_FIGURE: Final[Path] = FIGURES_DIR / "hybrid_gating" / "gate_disagreement.png"


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
NAV_ITEMS: Final[tuple[str, ...]] = (
    "Command",
    "Prediction",
    "Batch Analysis",
    "Model Insights",
    "Conformal Triage",
    "Explainability",
    "Settings",
    "About",
)


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------
def artifact_exists(path: Path) -> bool:
    """Return whether a model or experiment artifact exists."""
    return path.is_file()


@st.cache_data(ttl=60, show_spinner=False)
def load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    """Safely load a JSON object or list of objects.

    Missing or malformed files return an empty dictionary so unfinished
    experiment artifacts do not crash the dashboard.
    """
    if not path.is_file():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError:
        return {}
    except json.JSONDecodeError:
        return {}

    if isinstance(data, dict):
        return data

    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data

    return {}


def select_conformal_result(
    data: dict[str, Any] | list[dict[str, Any]],
    alpha: float = 0.01,
) -> dict[str, Any]:
    """Select the conformal result row matching the deployed alpha."""
    if isinstance(data, dict):
        return data

    for row in data:
        row_alpha = row.get("alpha")

        if isinstance(row_alpha, (int, float)) and abs(float(row_alpha) - alpha) < 1e-12:
            return row

    return {}


def status_label(available: bool) -> str:
    """Return dashboard readiness text."""
    return "READY" if available else "PENDING"


def readiness_count() -> int:
    """Count available core model checkpoints."""
    return sum(
        path.is_file()
        for path in (
            DAE_CHECKPOINT,
            FT_CHECKPOINT,
            GATE_CHECKPOINT,
        )
    )


def safe_metric(
    data: dict[str, Any],
    *keys: str,
) -> float | None:
    """Extract the first numeric metric available from candidate keys."""
    for key in keys:
        value = data.get(key)

        if isinstance(value, (int, float)):
            return float(value)

    return None


def api_url(path: str) -> str:
    """Build one versioned FastAPI endpoint URL."""
    return f"{API_BASE_URL}/api/v1/{path.lstrip('/')}"


def api_get(path: str) -> dict[str, Any]:
    """Fetch JSON from the FastAPI service."""
    endpoint = api_url(path)

    try:
        with request.urlopen(
            endpoint,
            timeout=API_TIMEOUT_SECONDS,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Unable to reach fraud API at {API_BASE_URL}: {exc.reason}") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid API response from {endpoint}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected API response from {endpoint}")

    return payload


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON to the FastAPI service and return its JSON response."""
    endpoint = api_url(path)
    body = json.dumps(payload).encode("utf-8")

    http_request = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(
            http_request,
            timeout=API_TIMEOUT_SECONDS,
        ) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Unable to reach fraud API at {API_BASE_URL}: {exc.reason}") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid API response from {endpoint}: {exc}") from exc

    if not isinstance(response_payload, dict):
        raise RuntimeError(f"Unexpected API response from {endpoint}")

    return response_payload


@st.cache_data(show_spinner=False)
def load_demo_transactions() -> Any:
    """Load the processed held-out test split for live demo scoring."""
    import pandas as pd

    if not TEST_DATA.is_file():
        raise RuntimeError(f"Held-out test data not found: {TEST_DATA}")

    return pd.read_parquet(TEST_DATA)


@st.cache_data(show_spinner=False)
def load_ft_feature_spec() -> dict[str, Any]:
    """Load the FT-CAT feature contract used by the trained checkpoint."""
    data = load_json(FT_FEATURE_SPEC)

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid FT-CAT feature specification: {FT_FEATURE_SPEC}")

    return data


@st.cache_data(show_spinner=False)
def load_dae_feature_columns() -> list[str]:
    """Resolve the ordered 800-feature DAE serving contract."""
    import pandas as pd

    from src.training.dae_features import resolve_dae_feature_columns
    from src.utils.config import load_config

    config = load_config(CONFIG_PATH)

    frame = pd.read_parquet(TEST_DATA)

    return resolve_dae_feature_columns(
        frame,
        non_feature_cols=list(config.data.non_feature_cols),
        expected_dim=int(config.autoencoder.input_dim),
    )


def build_prediction_payload(row: Any) -> dict[str, Any]:
    """Build one real FastAPI prediction payload from a held-out row."""
    import numpy as np

    spec = load_ft_feature_spec()
    dae_columns = load_dae_feature_columns()

    dae_features = row.loc[dae_columns].to_numpy(dtype=np.float32)
    ft_continuous = row.loc[spec["continuous_cols"]].to_numpy(dtype=np.float32)
    ft_categorical = row.loc[spec["categorical_cols"]].to_numpy(dtype=np.int64)
    ft_sequence = np.stack(row["sequence_array"]).astype(np.float32)

    return {
        "transaction_id": str(int(row["TransactionID"])),
        "features": dae_features.astype(float).tolist(),
        "ft_continuous": ft_continuous.astype(float).tolist(),
        "ft_categorical": ft_categorical.astype(int).tolist(),
        "ft_sequence": ft_sequence.astype(float).tolist(),
    }


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def inject_styles() -> None:
    """Inject command-center-inspired graphite styling.

    Neutral graphite/white/gray carries most of the interface; a small set
    of semantic accents (blue/cyan/purple for model identity, green/amber/
    red for ready/review/fraud states) is layered on top only where the
    color itself communicates something, not as blanket decoration.
    """
    st.markdown(
        (
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            '<link href="https://fonts.googleapis.com/css2?family='
            'IBM+Plex+Mono:wght@500;600;700&display=swap" '
            'rel="stylesheet">'
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        :root {
            --bg: #0D1117;
            --sidebar: #11161D;
            --surface: #171C24;
            --surface-2: #1C222C;
            --surface-3: #12171E;
            --border: #2A303B;
            --border-soft: #222832;
            --text: #F1F3F5;
            --muted: #929AA6;
            --blue: #6C8FF8;
            --cyan: #62C3D0;
            --green: #55C995;
            --amber: #E8B86D;
            --red: #E87979;
            --purple: #A78BDA;
            --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
        }

        html, body, [class*="css"] {
            font-family:
                Inter,
                ui-sans-serif,
                system-ui,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid var(--border-soft);
        }

        [data-testid="stSidebar"] * {
            color: var(--muted);
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--text);
        }

        [data-testid="stSidebar"] .stRadio > div {
            gap: 0.35rem;
        }

        [data-testid="stSidebar"] .stRadio label {
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
            transition: all 0.15s ease;
            border-left: 3px solid transparent;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: var(--surface);
        }

        [data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background: var(--surface-2);
            border-left: 3px solid var(--blue);
        }

        [data-testid="stSidebar"] .stRadio label:has(input:checked) p {
            color: var(--text) !important;
            font-weight: 600;
        }

        h1, h2, h3 {
            color: var(--text) !important;
            letter-spacing: -0.01em;
        }

        p, label {
            color: var(--muted);
        }

        .command-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.4rem;
            padding-bottom: 1.1rem;
            border-bottom: 1px solid var(--border-soft);
        }

        .header-title {
            font-size: 1.28rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.02em;
        }

        .header-subtitle {
            color: var(--muted);
            margin-top: 0.18rem;
            font-size: 0.8rem;
        }

        .system-live {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            border: 1px solid var(--border);
            background: var(--surface-2);
            color: var(--green);
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }

        .live-dot {
            width: 8px;
            height: 8px;
            background: var(--green);
            border-radius: 50%;
            box-shadow: 0 0 9px rgba(85, 201, 149, 0.9);
        }

        .section-title {
            color: var(--text);
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-bottom: 0.7rem;
            margin-top: 0.6rem;
        }

        .section-caption {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: -0.3rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            min-height: 138px;
            padding: 1.15rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 6px 22px rgba(0, 0, 0, 0.18);
        }

        .metric-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            font-size: 0.66rem;
        }

        .metric-value {
            font-family: var(--mono);
            color: var(--text);
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 0.35rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.74rem;
            margin-top: 0.25rem;
            line-height: 1.4;
        }

        .metric-value.accent-green { color: var(--green); }
        .metric-value.accent-blue { color: var(--blue); }
        .metric-value.accent-cyan { color: var(--cyan); }
        .metric-value.accent-purple { color: var(--purple); }
        .metric-value.accent-amber { color: var(--amber); }
        .metric-value.accent-red { color: var(--red); }

        .pipeline-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.2rem 1.35rem;
            color: var(--muted);
            box-shadow: 0 6px 22px rgba(0, 0, 0, 0.16);
            line-height: 2.2;
        }

        .pipeline-node {
            display: inline-block;
            color: var(--text);
            background: var(--surface-3);
            border: 1px solid var(--border);
            padding: 0.3rem 0.6rem;
            border-radius: 8px;
            margin: 0.15rem;
            font-size: 0.78rem;
            font-family: var(--mono);
        }

        .pipeline-ftcat {
            color: var(--blue);
            border-color: rgba(108, 143, 248, 0.4);
        }

        .pipeline-dae {
            color: var(--cyan);
            border-color: rgba(98, 195, 208, 0.4);
        }

        .pipeline-gate {
            color: var(--purple);
            border-color: rgba(167, 139, 218, 0.4);
        }

        .pipeline-conformal {
            color: var(--blue);
            border-color: rgba(108, 143, 248, 0.4);
        }

        .pipeline-arrow {
            color: var(--muted);
            font-weight: 700;
            margin: 0 0.1rem;
        }

        .pipeline-join {
            color: var(--muted);
            font-size: 0.72rem;
            margin: 0 0.05rem;
        }

        .pipeline-approve {
            color: var(--green);
            border-color: rgba(85, 201, 149, 0.4);
        }

        .pipeline-review {
            color: var(--amber);
            border-color: rgba(232, 184, 109, 0.4);
        }

        .pipeline-block {
            color: var(--red);
            border-color: rgba(232, 121, 121, 0.4);
        }

        .status-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.4rem 1rem;
            box-shadow: 0 6px 22px rgba(0, 0, 0, 0.14);
        }

        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            min-height: 46px;
            border-bottom: 1px solid var(--border-soft);
            color: var(--muted);
            font-size: 0.78rem;
        }

        .status-row:last-child {
            border-bottom: none;
        }

        .pending-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(232, 184, 109, 0.08);
            border: 1px solid rgba(232, 184, 109, 0.20);
            color: var(--amber);
            border-radius: 999px;
            padding: 0.22rem 0.5rem;
            font-size: 0.64rem;
            font-weight: 700;
        }

        .small-dot-ready {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: var(--green);
            box-shadow: 0 0 5px var(--green);
            display: inline-block;
        }

        .small-dot-pending {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: var(--amber);
            display: inline-block;
        }

        .ready-quiet {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: var(--muted);
            font-size: 0.66rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }

        .artifact-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
        }

        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
        }

        [data-testid="stFileUploader"] {
            background: var(--surface);
            border-radius: 14px;
        }

        .stSelectbox > div > div {
            background: var(--surface);
            border-color: var(--border);
        }

        .sidebar-logo {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: var(--green);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--bg);
            font-weight: 900;
            font-size: 1.1rem;
            box-shadow: 0 0 18px rgba(85, 201, 149, 0.28);
            margin-bottom: 0.8rem;
        }

        .sidebar-readiness {
            margin-top: 1rem;
            border-top: 1px solid var(--border-soft);
            padding-top: 1rem;
            color: var(--muted);
            font-size: 0.72rem;
        }

        .tiny-mono {
            color: var(--muted);
            font-family: var(--mono);
            font-size: 0.72rem;
        }

        /* ------------------------------------------------------------------
           Prediction experience
           ------------------------------------------------------------------ */

        .prediction-hero {
            position: relative;
            overflow: hidden;
            min-height: 245px;
            margin: 0.2rem 0 1.35rem;
            padding: 2.15rem 2.35rem;
            border: 1px solid var(--border);
            border-radius: 22px;
            background:
                radial-gradient(
                    circle at 78% 34%,
                    rgba(167, 139, 218, 0.16),
                    transparent 23%
                ),
                radial-gradient(
                    circle at 68% 45%,
                    rgba(108, 143, 248, 0.15),
                    transparent 29%
                ),
                radial-gradient(
                    circle at 90% 55%,
                    rgba(98, 195, 208, 0.09),
                    transparent 20%
                ),
                linear-gradient(
                    120deg,
                    #111720 0%,
                    #111721 46%,
                    #121827 100%
                );
        }

        .prediction-hero::after {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            opacity: 0.16;
            background-image:
                linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
            background-size: 42px 42px;
            mask-image: linear-gradient(
                to right,
                transparent 20%,
                black 75%
            );
        }

        .prediction-kicker {
            position: relative;
            z-index: 1;
            color: var(--blue);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .prediction-title {
            position: relative;
            z-index: 1;
            margin: 0;
            color: var(--text);
            font-size: clamp(2.4rem, 5vw, 4.25rem);
            font-weight: 800;
            letter-spacing: -0.055em;
            line-height: 0.98;
        }

        .prediction-title-accent {
            background: linear-gradient(
                90deg,
                var(--blue),
                var(--purple)
            );
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .prediction-tagline {
            position: relative;
            z-index: 1;
            color: #D8DEE8;
            margin-top: 0.8rem;
            font-size: 1.05rem;
            font-weight: 500;
        }

        .prediction-description {
            position: relative;
            z-index: 1;
            max-width: 620px;
            margin-top: 0.4rem;
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.65;
        }

        .prediction-capabilities {
            position: relative;
            z-index: 1;
            display: flex;
            gap: 2rem;
            flex-wrap: wrap;
            margin-top: 1.45rem;
        }

        .prediction-capability {
            display: flex;
            align-items: flex-start;
            gap: 0.65rem;
        }

        .capability-icon {
            display: flex;
            width: 34px;
            height: 34px;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            background: rgba(108, 143, 248, 0.11);
            border: 1px solid rgba(108, 143, 248, 0.24);
            color: var(--blue);
            font-weight: 800;
        }

        .capability-copy strong {
            display: block;
            color: var(--text);
            font-size: 0.79rem;
            margin-bottom: 0.12rem;
        }

        .capability-copy span {
            color: var(--muted);
            font-size: 0.69rem;
        }

        .prediction-selection-shell {
            padding: 1.15rem 1.35rem 0.3rem;
            margin-bottom: 1rem;
            background: linear-gradient(
                145deg,
                rgba(23, 28, 36, 0.94),
                rgba(18, 23, 31, 0.96)
            );
            border: 1px solid var(--border);
            border-radius: 18px;
        }

        .prediction-selection-title {
            color: var(--text);
            font-size: 0.92rem;
            font-weight: 700;
        }

        .prediction-selection-caption {
            color: var(--muted);
            font-size: 0.75rem;
            margin-top: 0.16rem;
            margin-bottom: 0.85rem;
        }

        /* Primary prediction CTA */
        div[data-testid="stButton"] > button[kind="primary"] {
            min-height: 58px;
            border: 1px solid rgba(108, 143, 248, 0.75) !important;
            border-radius: 14px !important;
            background:
                linear-gradient(
                    100deg,
                    #2697F3 0%,
                    #5577F6 48%,
                    #A72BF2 100%
                ) !important;
            color: #FFFFFF !important;
            font-weight: 750 !important;
            font-size: 0.95rem !important;
            letter-spacing: -0.01em;
            box-shadow:
                0 8px 28px rgba(73, 104, 245, 0.20),
                0 0 32px rgba(167, 43, 242, 0.08);
            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease,
                filter 0.15s ease;
        }

        div[data-testid="stButton"] > button[kind="primary"]:hover {
            transform: translateY(-1px);
            filter: brightness(1.08);
            box-shadow:
                0 10px 34px rgba(73, 104, 245, 0.28),
                0 0 38px rgba(167, 43, 242, 0.12);
        }

        div[data-testid="stButton"] > button[kind="primary"] p {
            color: #FFFFFF !important;
            font-weight: 750 !important;
        }

        /* Prediction workflow */
        .workflow-card {
            margin-top: 1.15rem;
            padding: 1.15rem 1.35rem;
            background:
                linear-gradient(
                    110deg,
                    rgba(98, 195, 208, 0.035),
                    rgba(108, 143, 248, 0.045),
                    rgba(167, 139, 218, 0.035)
                ),
                var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
        }

        .workflow-title {
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .workflow-steps {
            display: grid;
            grid-template-columns:
                minmax(0, 1fr) auto
                minmax(0, 1fr) auto
                minmax(0, 1fr) auto
                minmax(0, 1fr);
            gap: 0.75rem;
            align-items: center;
        }

        .workflow-step {
            display: flex;
            gap: 0.7rem;
            align-items: center;
        }

        .workflow-number {
            flex: 0 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: linear-gradient(
                135deg,
                rgba(38, 151, 243, 0.88),
                rgba(100, 80, 225, 0.88)
            );
            border: 1px solid rgba(108, 143, 248, 0.55);
            color: white;
            font-family: var(--mono);
            font-weight: 700;
        }

        .workflow-copy strong {
            display: block;
            color: var(--text);
            font-size: 0.73rem;
        }

        .workflow-copy span {
            display: block;
            color: var(--muted);
            font-size: 0.64rem;
            margin-top: 0.1rem;
        }

        .workflow-arrow {
            color: #596473;
            font-size: 1rem;
        }

        @media (max-width: 900px) {
            .prediction-hero {
                padding: 1.5rem;
            }

            .prediction-capabilities {
                gap: 1rem;
            }

            .workflow-steps {
                grid-template-columns: 1fr;
            }

            .workflow-arrow {
                display: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Reusable components
# ---------------------------------------------------------------------------
def render_header(
    title: str,
    subtitle: str,
) -> None:
    """Render command-center page header."""
    ready = readiness_count()

    system_text = "CORE MODELS READY" if ready == 3 else f"{ready}/3 CORE MODELS READY"

    st.markdown(
        f"""
        <div class="command-header">
            <div>
                <div class="header-title">{title}</div>
                <div class="header-subtitle">{subtitle}</div>
            </div>
            <div class="system-live">
                <span class="live-dot"></span>
                {system_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    note: str,
    *,
    accent: str = "neutral",
) -> None:
    """Render one command-center metric card.

    Values default to a neutral white/gray so a card is not colored just
    because it exists. Pass a semantic accent (green/amber/red for a
    ready/review/fraud state, blue/cyan/purple for FT-CAT/DAE/Gate
    identity) only when the value itself carries that meaning.
    """
    value_class = "metric-value" if accent == "neutral" else f"metric-value accent-{accent}"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="{value_class}">
                {value}
            </div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_row(
    label: str,
    available: bool,
) -> None:
    """Render one artifact-readiness row.

    Ready is the expected, common state and stays quiet (a small
    checkmark-style dot); only a genuinely pending artifact gets the
    bordered attention-pill, so the exception is what stands out
    instead of every row shouting the same badge.
    """
    if available:
        st.markdown(
            f"""
            <div class="status-row">
                <span>{label}</span>
                <span class="ready-quiet">
                    <span class="small-dot-ready"></span>
                    Ready
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-row">
                <span>{label}</span>
                <span class="pending-pill">
                    <span class="small-dot-pending"></span>
                    PENDING
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_artifact_image(
    path: Path,
    title: str,
    description: str,
    *,
    caption: str | None = None,
) -> None:
    """Render a saved experiment artifact or a truthful placeholder."""
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-caption">{description}</div>',
        unsafe_allow_html=True,
    )

    if path.is_file():
        st.image(
            str(path),
            caption=caption,
            width="stretch",
        )
    else:
        st.info(
            "Artifact pending. It will appear automatically once the "
            "corresponding experiment has generated it."
        )


# ---------------------------------------------------------------------------
# Command page
# ---------------------------------------------------------------------------
def command_page() -> None:
    """Render the main fraud-command overview.

    One pass over the live API, then three non-repeating sections: the
    headline conformal numbers, the decision pipeline, and readiness —
    each concept (model checkpoints vs. generated artifacts) shown once
    rather than the same "ready" state echoed across several card styles.
    """
    render_header(
        "Fraud Triage Engine",
        "FT-CAT • DAE • Learned Gate • Split Conformal Prediction",
    )

    try:
        api_get("health")
    except RuntimeError as exc:
        st.error(str(exc))

    conformal_data = select_conformal_result(
        load_json(CONFORMAL_RESULTS),
        alpha=0.01,
    )

    coverage = safe_metric(
        conformal_data,
        "coverage",
        "empirical_coverage",
    )

    review_rate = safe_metric(
        conformal_data,
        "review_rate",
        "review_fraction",
    )

    singleton_rate = safe_metric(
        conformal_data,
        "singleton_rate",
    )

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True,
    )

    overview_columns = st.columns(3)

    with overview_columns[0]:
        if coverage is None:
            value = "PENDING"
            note = "Waiting for final held-out conformal evaluation"
            accent = "amber"
        else:
            value = f"{coverage * 100:.2f}%"
            note = "Empirical held-out conformal coverage"
            accent = "neutral"

        render_metric_card("Conformal Coverage", value, note, accent=accent)

    with overview_columns[1]:
        if review_rate is None:
            value = "PENDING"
            note = "Analyst workload will populate from real evaluation"
            accent = "amber"
        else:
            value = f"{review_rate * 100:.2f}%"
            note = "Transactions routed to human review"
            accent = "neutral"

        render_metric_card("Analyst Review Workload", value, note, accent=accent)

    with overview_columns[2]:
        if singleton_rate is None:
            value = "PENDING"
            note = "Singleton prediction-set rate not generated yet"
            accent = "amber"
        else:
            value = f"{singleton_rate * 100:.2f}%"
            note = "Confident singleton conformal decisions"
            accent = "neutral"

        render_metric_card("Singleton Decisions", value, note, accent=accent)

    st.markdown(
        '<div class="section-title">Decision Pipeline</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pipeline-card">
            <span class="pipeline-node">Transaction</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-node pipeline-ftcat">FT-CAT Probability</span>
            <span class="pipeline-join">+</span>
            <span class="pipeline-node pipeline-dae">DAE Anomaly Score</span>
            <span class="pipeline-join">+</span>
            <span class="pipeline-node">History Context</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-node pipeline-gate">
                Learned Gate
            </span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-node">Fraud Probability</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-node pipeline-conformal">
                Conformal Prediction
            </span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-node pipeline-approve">Approve</span>
            <span class="pipeline-join">/</span>
            <span class="pipeline-node pipeline-review">Review</span>
            <span class="pipeline-join">/</span>
            <span class="pipeline-node pipeline-block">Block</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Model Status</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="status-panel">', unsafe_allow_html=True)

    render_status_row("FT-CAT Transformer", FT_CHECKPOINT.is_file())
    render_status_row("Denoising Autoencoder", DAE_CHECKPOINT.is_file())
    render_status_row("Learned Hybrid Gate", GATE_CHECKPOINT.is_file())
    render_status_row("Conformal Calibration", CONFORMAL_RESULTS.is_file())

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Artifacts</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="status-panel">', unsafe_allow_html=True)

    render_status_row("DAE latent-space visualization", TSNE_FIGURE.is_file())
    render_status_row("SHAP explainability artifacts", SHAP_GLOBAL_FIGURE.is_file())
    render_status_row("Conformal evaluation artifacts", CONFORMAL_RESULTS.is_file())

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def prediction_page() -> None:
    """Run live fraud scoring for a real held-out transaction."""
    st.markdown(
        """
        <div class="prediction-hero">
            <div class="prediction-kicker">Prediction</div>
            <div class="prediction-title">
                Fraud <span class="prediction-title-accent">Triage</span>
            </div>
            <div class="prediction-tagline">
                From data to decisions.
            </div>
            <div class="prediction-description">
                Analyze a real held-out transaction using FT-CAT, the
                denoising autoencoder, learned hybrid gating, and
                uncertainty-aware conformal prediction.
            </div>
            <div class="prediction-capabilities">
                <div class="prediction-capability">
                    <div class="capability-icon">&check;</div>
                    <div class="capability-copy">
                        <strong>Detect</strong>
                        <span>Identify fraud patterns</span>
                    </div>
                </div>
                <div class="prediction-capability">
                    <div class="capability-icon">&#8599;</div>
                    <div class="capability-copy">
                        <strong>Explain</strong>
                        <span>Understand risk signals</span>
                    </div>
                </div>
                <div class="prediction-capability">
                    <div class="capability-icon">&#9678;</div>
                    <div class="capability-copy">
                        <strong>Decide</strong>
                        <span>Route with confidence</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        frame = load_demo_transactions()
    except Exception as exc:
        st.error(f"Unable to load held-out test transactions: {exc}")
        return

    if frame.empty:
        st.warning("Held-out test dataset contains no transactions.")
        return

    st.markdown(
        """
        <div class="prediction-selection-shell">
            <div class="prediction-selection-title">
                Transaction Selection
            </div>
            <div class="prediction-selection-caption">
                Choose a real chronological held-out example to run
                through the production fraud pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selection_mode = st.radio(
        "Example type",
        ("Fraud", "Legitimate", "Random"),
        horizontal=True,
    )

    if selection_mode == "Fraud":
        candidates = frame[frame["isFraud"] == 1]
    elif selection_mode == "Legitimate":
        candidates = frame[frame["isFraud"] == 0]
    else:
        candidates = frame

    if candidates.empty:
        st.warning(f"No {selection_mode.lower()} transactions are available.")
        return

    max_index = min(len(candidates) - 1, 1000)

    example_index = st.number_input(
        "Example index",
        min_value=0,
        max_value=max_index,
        value=0,
        step=1,
    )

    if selection_mode == "Random":
        selected_row = candidates.sample(
            n=1,
            random_state=int(example_index),
        ).iloc[0]
    else:
        selected_row = candidates.iloc[int(example_index)]

    transaction_id = int(selected_row["TransactionID"])
    ground_truth = int(selected_row["isFraud"])
    processed_amount = float(selected_row["TransactionAmt"])

    info_columns = st.columns(3)

    with info_columns[0]:
        render_metric_card(
            "Transaction ID",
            str(transaction_id),
            "Real chronological held-out transaction",
            accent="neutral",
        )

    with info_columns[1]:
        render_metric_card(
            "Ground Truth",
            "FRAUD" if ground_truth == 1 else "LEGITIMATE",
            "Held-out label used only after prediction",
            accent="red" if ground_truth == 1 else "green",
        )

    with info_columns[2]:
        render_metric_card(
            "Processed Amount",
            f"{processed_amount:.4f}",
            "Scaled model input, not raw currency value",
            accent="neutral",
        )

    st.markdown("")

    run_analysis = st.button(
        "▶  Run Fraud Analysis  →",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="workflow-card">
            <div class="workflow-title">What happens next?</div>
            <div class="workflow-steps">
                <div class="workflow-step">
                    <div class="workflow-number">1</div>
                    <div class="workflow-copy">
                        <strong>Analyze</strong>
                        <span>FT-CAT + DAE + Gate</span>
                    </div>
                </div>
                <div class="workflow-arrow">&rarr;</div>
                <div class="workflow-step">
                    <div class="workflow-number">2</div>
                    <div class="workflow-copy">
                        <strong>Calibrate</strong>
                        <span>Conformal prediction</span>
                    </div>
                </div>
                <div class="workflow-arrow">&rarr;</div>
                <div class="workflow-step">
                    <div class="workflow-number">3</div>
                    <div class="workflow-copy">
                        <strong>Decide</strong>
                        <span>Approve / Review / Block</span>
                    </div>
                </div>
                <div class="workflow-arrow">&rarr;</div>
                <div class="workflow-step">
                    <div class="workflow-number">4</div>
                    <div class="workflow-copy">
                        <strong>Explain</strong>
                        <span>Model signals &amp; insights</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not run_analysis:
        st.info(
            "Select a held-out transaction and run the full "
            "DAE + FT-CAT + gate + conformal pipeline."
        )
        return

    try:
        payload = build_prediction_payload(selected_row)

        with st.spinner("Running production fraud models..."):
            result = api_post("predict", payload)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.markdown(
        '<div class="section-title">Live Model Output</div>',
        unsafe_allow_html=True,
    )

    ft_probability = result.get("ft_probability")
    anomaly_score = result.get("anomaly_score")
    fraud_probability = result.get("fraud_probability")
    decision = result.get("decision")
    conformal_set = result.get("conformal_set")
    latency_ms = result.get("latency_ms")

    metric_columns = st.columns(4)

    with metric_columns[0]:
        render_metric_card(
            "FT-CAT Probability",
            (
                f"{float(ft_probability) * 100:.2f}%"
                if isinstance(ft_probability, (int, float))
                else "N/A"
            ),
            "Supervised fraud probability",
            accent="blue",
        )

    with metric_columns[1]:
        render_metric_card(
            "DAE Anomaly Score",
            (f"{float(anomaly_score):.4f}" if isinstance(anomaly_score, (int, float)) else "N/A"),
            "Reconstruction-based anomaly signal",
            accent="cyan",
        )

    with metric_columns[2]:
        render_metric_card(
            "Fused Probability",
            (
                f"{float(fraud_probability) * 100:.2f}%"
                if isinstance(fraud_probability, (int, float))
                else "N/A"
            ),
            "Learned hybrid gate output",
            accent="purple",
        )

    with metric_columns[3]:
        render_metric_card(
            "Latency",
            (f"{float(latency_ms):.2f} ms" if isinstance(latency_ms, (int, float)) else "N/A"),
            "FastAPI inference latency",
            accent="neutral",
        )

    decision_columns = st.columns(2)

    with decision_columns[0]:
        st.markdown(
            '<div class="section-title">Conformal Prediction</div>',
            unsafe_allow_html=True,
        )

        st.code(
            str(conformal_set) if conformal_set is not None else "Conformal prediction unavailable"
        )

    with decision_columns[1]:
        st.markdown(
            '<div class="section-title">Operational Decision</div>',
            unsafe_allow_html=True,
        )

        decision_labels = {
            "auto_approve": "AUTO-APPROVE",
            "auto_block": "AUTO-BLOCK",
            "human_review": "HUMAN REVIEW",
            "escalate": "ESCALATE",
        }

        decision_text = decision_labels.get(
            str(decision),
            str(decision).upper(),
        )

        if decision == "auto_block":
            st.error(decision_text)
        elif decision in {"human_review", "escalate"}:
            st.warning(decision_text)
        else:
            st.success(decision_text)

    st.caption(
        "Pipeline engaged — "
        f"FT-CAT model: {'yes' if result.get('ft_model_used') else 'no'} · "
        f"Learned gate fusion: {'yes' if result.get('gate_used') else 'no'} · "
        f"Conformal calibration: {'yes' if result.get('conformal_used') else 'no'}"
    )

    predicted_fraud = bool(result.get("is_fraud"))

    if predicted_fraud == bool(ground_truth):
        st.success("Point prediction matches the held-out ground-truth label.")
    else:
        st.warning("Point prediction differs from the held-out ground-truth label.")

    with st.expander("API response"):
        st.json(result)


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------
def batch_analysis_page() -> None:
    """Render result-file batch inspection."""
    render_header(
        "Batch Analysis",
        "Inspect prepared transaction-level model outputs.",
    )

    uploaded_file = st.file_uploader(
        "Upload prepared results CSV",
        type=["csv"],
        help=("Upload experiment/model output rather than raw IEEE-CIS " "transactions."),
    )

    if uploaded_file is None:
        st.info(
            "Upload a prepared prediction/result CSV to inspect rows, "
            "columns, fraud scores, triage outputs, or evaluation fields."
        )
        return

    try:
        import pandas as pd

        frame = pd.read_csv(uploaded_file)

    except Exception as exc:
        st.error(f"Unable to read uploaded CSV: {exc}")
        return

    if frame.empty:
        st.warning("The uploaded CSV contains no rows.")
        return

    col1, col2 = st.columns(2)

    with col1:
        render_metric_card(
            "Transactions",
            f"{len(frame):,}",
            "Rows in uploaded result file",
            accent="neutral",
        )

    with col2:
        render_metric_card(
            "Available Fields",
            f"{len(frame.columns):,}",
            "Columns available for inspection",
            accent="neutral",
        )

    st.markdown(
        '<div class="section-title">Evaluation Transactions</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        frame,
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Model insights
# ---------------------------------------------------------------------------
def model_insights_page() -> None:
    """Render model diagnostics."""
    render_header(
        "Model Insights",
        "Latent structure, learned-gate behavior, and calibration diagnostics.",
    )

    selected = st.selectbox(
        "Analysis",
        (
            "DAE Latent Space",
            "Gate Disagreement",
            "Coverage vs Workload",
            "Threshold Sensitivity",
        ),
        label_visibility="collapsed",
    )

    if selected == "DAE Latent Space":
        render_artifact_image(
            TSNE_FIGURE,
            "DAE Bottleneck — t-SNE",
            (
                "Held-out transaction representations projected from the "
                "autoencoder's 32-dimensional latent space."
            ),
            caption=(
                "The class-separation statistic is computed in the original "
                "latent space, not the two-dimensional t-SNE projection."
            ),
        )

        st.info(
            "The DAE representation shows weak but non-zero fraud-vs-"
            "legitimate separation, supporting its role as an unsupervised "
            "anomaly component rather than a standalone classifier."
        )

    elif selected == "Gate Disagreement":
        render_artifact_image(
            GATE_DISAGREEMENT_FIGURE,
            "Learned Gate vs Fixed-Alpha Fusion",
            (
                "Highlights transactions where learned dynamic fusion "
                "diverges from the fixed-alpha baseline."
            ),
        )

    elif selected == "Coverage vs Workload":
        render_artifact_image(
            CONFORMAL_COVERAGE_FIGURE,
            "Coverage vs Analyst Workload",
            (
                "Empirical conformal coverage relative to the fraction "
                "of transactions routed for human review."
            ),
        )

    else:
        render_artifact_image(
            CONFORMAL_THRESHOLD_FIGURE,
            "Threshold Sensitivity",
            (
                "Sensitivity of empirical coverage, review workload, and "
                "prediction-set behavior across calibration thresholds."
            ),
        )


# ---------------------------------------------------------------------------
# Conformal triage
# ---------------------------------------------------------------------------
def conformal_page() -> None:
    """Render conformal evaluation."""
    render_header(
        "Conformal Triage",
        "Finite-sample uncertainty quantification for fraud decisions.",
    )

    results = select_conformal_result(
        load_json(CONFORMAL_RESULTS),
        alpha=0.01,
    )

    coverage = safe_metric(
        results,
        "coverage",
        "empirical_coverage",
    )

    review_rate = safe_metric(
        results,
        "review_rate",
        "review_fraction",
    )

    avg_set_size = safe_metric(
        results,
        "average_set_size",
        "avg_set_size",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        render_metric_card(
            "Empirical Coverage",
            "PENDING" if coverage is None else f"{coverage * 100:.2f}%",
            (
                "Final chronological test evaluation"
                if coverage is not None
                else "Waiting for real conformal evaluation"
            ),
            accent="amber" if coverage is None else "neutral",
        )

    with col2:
        render_metric_card(
            "Review Workload",
            ("PENDING" if review_rate is None else f"{review_rate * 100:.2f}%"),
            "Fraction routed to human analyst review",
            accent="amber" if review_rate is None else "neutral",
        )

    with col3:
        render_metric_card(
            "Average Set Size",
            ("PENDING" if avg_set_size is None else f"{avg_set_size:.3f}"),
            "Mean conformal prediction-set cardinality",
            accent="amber" if avg_set_size is None else "neutral",
        )

    render_artifact_image(
        CONFORMAL_COVERAGE_FIGURE,
        "Coverage vs Workload",
        ("Operational trade-off between empirical label coverage " "and human-review workload."),
    )

    render_artifact_image(
        CONFORMAL_THRESHOLD_FIGURE,
        "Threshold Sensitivity",
        ("Review rate and prediction-set behavior as the conformal " "threshold changes."),
    )

    st.info(
        "The 99% split-conformal finite-sample guarantee applies under "
        "exchangeability. The final test split is chronological, so actual "
        "held-out empirical coverage must also be reported under possible "
        "temporal distribution shift."
    )


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def explainability_page() -> None:
    """Render DAE-component SHAP artifacts."""
    render_header(
        "Explainability",
        "Feature-level interpretation of the DAE anomaly-score component.",
    )

    st.warning(
        "Current SHAP artifacts explain the DAE anomaly component only. "
        "They are not explanations of the final learned hybrid-gating "
        "decision."
    )

    selected = st.selectbox(
        "Explanation",
        (
            "Global Importance",
            "Local Waterfall",
            "Sampling Consistency",
        ),
        label_visibility="collapsed",
    )

    if selected == "Global Importance":
        render_artifact_image(
            SHAP_GLOBAL_FIGURE,
            "Global DAE SHAP Importance",
            ("Mean absolute SHAP contribution across sampled held-out " "transactions."),
        )

    elif selected == "Local Waterfall":
        render_artifact_image(
            SHAP_LOCAL_FIGURE,
            "Local DAE SHAP Waterfall",
            (
                "Feature contribution breakdown for a selected "
                "high-anomaly fraudulent transaction."
            ),
        )

    else:
        consistency = load_json(SHAP_CONSISTENCY_RESULTS)

        render_artifact_image(
            SHAP_CONSISTENCY_FIGURE,
            "SHAP Sampling Consistency",
            ("Stability of global DAE feature importance across " "independent samples."),
        )

        summary = consistency.get("summary", consistency)

        metrics = (
            (
                "Mean Spearman",
                summary.get("mean_spearman_correlation"),
            ),
            (
                "Min Spearman",
                summary.get("min_spearman_correlation"),
            ),
            (
                "Mean Top-K Jaccard",
                summary.get("mean_top_k_jaccard"),
            ),
            (
                "Min Top-K Jaccard",
                summary.get("min_top_k_jaccard"),
            ),
        )

        columns = st.columns(4)

        for column, (label, value) in zip(
            columns,
            metrics,
            strict=True,
        ):
            with column:
                if isinstance(value, (int, float)):
                    st.metric(label, f"{value:.3f}")
                else:
                    st.metric(label, "—")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def settings_page() -> None:
    """Render project artifact status."""
    render_header(
        "Settings",
        "Runtime artifacts and model-readiness configuration.",
    )

    st.markdown(
        '<div class="section-title">Core Models</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="status-panel">',
        unsafe_allow_html=True,
    )

    render_status_row(
        "Denoising Autoencoder",
        DAE_CHECKPOINT.is_file(),
    )

    render_status_row(
        "FT-CAT Transformer",
        FT_CHECKPOINT.is_file(),
    )

    render_status_row(
        "Learned Hybrid Gate",
        GATE_CHECKPOINT.is_file(),
    )

    render_status_row(
        "Conformal Evaluation",
        CONFORMAL_RESULTS.is_file(),
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Artifact Locations</div>',
        unsafe_allow_html=True,
    )

    st.code(
        "\n".join(
            (
                f"Models:  {CHECKPOINT_DIR.relative_to(PROJECT_ROOT)}",
                f"Figures: {FIGURES_DIR.relative_to(PROJECT_ROOT)}",
                f"Results: {RESULTS_DIR.relative_to(PROJECT_ROOT)}",
            )
        )
    )


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
def about_page() -> None:
    """Render technical architecture information."""
    render_header(
        "About",
        "Real-Time Financial Fraud Detection, Explainability, and Triage.",
    )

    st.markdown("""
        ### FT-CAT Transformer

        Provides the supervised fraud-probability signal using continuous,
        categorical, and historical transaction inputs.

        ### Denoising Autoencoder

        Learns legitimate transaction structure and provides an unsupervised
        reconstruction-based anomaly signal.

        ### Learned Hybrid Gate

        Dynamically combines:

        - normalized DAE anomaly score,
        - FT-CAT fraud probability,
        - transaction-history activity,
        - transaction-history amount intensity.

        ### Split Conformal Prediction

        Converts fused fraud probabilities into prediction sets:

        - `{0}` → Auto-Approve
        - `{1}` → Auto-Block
        - `{0,1}` → Human Review
        - `{}` → Human Review

        ### SHAP Explainability

        Current SHAP artifacts interpret the DAE anomaly-score component,
        not the final learned hybrid decision.
        """)

    st.info(
        "The conformal finite-sample guarantee is conditional on "
        "exchangeability. Chronological test coverage is therefore "
        "reported empirically to expose possible temporal drift."
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    """Render command-center navigation."""
    ready = readiness_count()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <svg
                    width="20" height="20" viewBox="0 0 24 24"
                    fill="none" xmlns="http://www.w3.org/2000/svg"
                >
                    <path
                        d="M12 2L4 5v6c0 5.25 3.4 9.74 8 11 4.6-1.26 8-5.75 8-11V5l-8-3z"
                        fill="var(--bg)"
                    />
                    <path
                        d="M9 12.3l2 2 4-4.6"
                        stroke="var(--text)"
                        stroke-width="1.7"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    />
                </svg>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Fraud Triage")

        st.markdown(
            '<div class="tiny-mono">' "FT-CAT • DAE • Gate • Conformal" "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("")

        selected_page = st.radio(
            "Navigation",
            NAV_ITEMS,
            label_visibility="collapsed",
        )

        st.markdown(
            f"""
            <div class="sidebar-readiness">
                <span class="small-dot-ready"></span>
                <span class="tiny-mono">{ready}/3 models ready</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return selected_page


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    """Run Streamlit dashboard."""
    st.set_page_config(
        page_title="Fraud Triage Engine",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()

    selected_page = render_sidebar()

    pages = {
        "Command": command_page,
        "Prediction": prediction_page,
        "Batch Analysis": batch_analysis_page,
        "Model Insights": model_insights_page,
        "Conformal Triage": conformal_page,
        "Explainability": explainability_page,
        "Settings": settings_page,
        "About": about_page,
    }

    pages[selected_page]()


if __name__ == "__main__":
    main()
