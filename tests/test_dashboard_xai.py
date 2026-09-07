"""Tests for the Streamlit fraud-triage dashboard and XAI integration."""

# Import necessary modules and libraries
from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard.app import (
    artifact_exists,
    load_json,
    safe_metric,
    select_conformal_result,
    status_label,
)

APP_PATH = Path("src/dashboard/app.py")


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------
def test_artifact_exists_returns_true_for_existing_file(
    tmp_path: Path,
) -> None:
    """Existing experiment artifacts should be reported as available."""
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")

    assert artifact_exists(artifact) is True


def test_artifact_exists_returns_false_for_missing_file(
    tmp_path: Path,
) -> None:
    """Missing experiment artifacts should be reported as unavailable."""
    artifact = tmp_path / "missing.json"

    assert artifact_exists(artifact) is False


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------
def test_load_json_accepts_dictionary(tmp_path: Path) -> None:
    """Dashboard should load dictionary JSON artifacts."""
    path = tmp_path / "artifact.json"
    payload = {"metric": 0.95}

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    assert load_json(path) == payload


def test_load_json_accepts_list_of_dictionaries(
    tmp_path: Path,
) -> None:
    """Conformal alpha-sweep JSON should remain available to the dashboard."""
    path = tmp_path / "conformal_metrics.json"

    payload = [
        {
            "alpha": 0.01,
            "empirical_coverage": 0.99,
        },
        {
            "alpha": 0.05,
            "empirical_coverage": 0.95,
        },
    ]

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    assert load_json(path) == payload


def test_load_json_missing_file_returns_empty_dict(
    tmp_path: Path,
) -> None:
    """Missing artifacts should not crash dashboard rendering."""
    path = tmp_path / "missing.json"

    assert load_json(path) == {}


def test_load_json_malformed_json_returns_empty_dict(
    tmp_path: Path,
) -> None:
    """Malformed experiment artifacts should fail gracefully."""
    path = tmp_path / "broken.json"

    path.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    assert load_json(path) == {}


@pytest.mark.parametrize(
    "payload",
    [
        42,
        3.14,
        "fraud",
        True,
        None,
        [1, 2, 3],
        ["invalid"],
        [{"alpha": 0.01}, "invalid"],
    ],
)
def test_load_json_rejects_unsupported_payloads(
    tmp_path: Path,
    payload: object,
) -> None:
    """Only dictionaries and lists of dictionaries are dashboard artifacts."""
    path = tmp_path / "unsupported.json"

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    assert load_json(path) == {}


# ---------------------------------------------------------------------------
# Conformal result selection
# ---------------------------------------------------------------------------
def test_select_conformal_result_uses_deployed_alpha() -> None:
    """Dashboard should select the requested conformal alpha."""
    results = [
        {
            "alpha": 0.05,
            "empirical_coverage": 0.95,
        },
        {
            "alpha": 0.01,
            "empirical_coverage": 0.99,
        },
        {
            "alpha": 0.10,
            "empirical_coverage": 0.90,
        },
    ]

    selected = select_conformal_result(
        results,
        alpha=0.01,
    )

    assert selected["alpha"] == pytest.approx(0.01)
    assert selected["empirical_coverage"] == pytest.approx(0.99)


def test_select_conformal_result_supports_float_tolerance() -> None:
    """Tiny floating-point differences should not hide the deployed result."""
    results = [
        {
            "alpha": 0.010000000000000002,
            "empirical_coverage": 0.991,
        }
    ]

    selected = select_conformal_result(
        results,
        alpha=0.01,
    )

    assert selected["empirical_coverage"] == pytest.approx(0.991)


def test_select_conformal_result_missing_alpha_returns_empty_dict() -> None:
    """Dashboard must not silently substitute another conformal alpha."""
    results = [
        {"alpha": 0.05},
        {"alpha": 0.10},
    ]

    assert (
        select_conformal_result(
            results,
            alpha=0.01,
        )
        == {}
    )


def test_select_conformal_result_empty_list_returns_empty_dict() -> None:
    """An unfinished conformal experiment should remain visibly unavailable."""
    assert select_conformal_result([], alpha=0.01) == {}


def test_select_conformal_result_accepts_dictionary() -> None:
    """Single-result dictionary artifacts should remain backward compatible."""
    result = {
        "alpha": 0.01,
        "empirical_coverage": 0.99,
    }

    assert select_conformal_result(result, alpha=0.01) == result


def test_select_conformal_result_does_not_choose_first_row() -> None:
    """The first alpha sweep row must not be mistaken for deployment alpha."""
    results = [
        {
            "alpha": 0.10,
            "empirical_coverage": 0.90,
        },
        {
            "alpha": 0.01,
            "empirical_coverage": 0.99,
        },
    ]

    selected = select_conformal_result(
        results,
        alpha=0.01,
    )

    assert selected["alpha"] == pytest.approx(0.01)
    assert selected["empirical_coverage"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Dashboard formatting helpers
# ---------------------------------------------------------------------------
def test_status_label_reports_ready() -> None:
    """Available artifacts should receive the READY dashboard label."""
    assert status_label(True) == "READY"


def test_status_label_reports_pending() -> None:
    """Unavailable artifacts should receive the PENDING dashboard label."""
    assert status_label(False) == "PENDING"


def test_safe_metric_returns_numeric_value() -> None:
    """Existing numeric metrics should be returned without fabrication."""
    result = safe_metric(
        {"empirical_coverage": 0.99123},
        "empirical_coverage",
    )

    assert result == pytest.approx(0.99123)


def test_safe_metric_missing_value_returns_none() -> None:
    """Missing metrics should remain unavailable."""
    result = safe_metric(
        {},
        "empirical_coverage",
    )

    assert result is None


# ---------------------------------------------------------------------------
# XAI artifact contract
# ---------------------------------------------------------------------------
def test_explainability_sampling_consistency_renders_real_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling Consistency renders metrics from the real SHAP result keys."""
    results_path = tmp_path / "dae_shap_consistency.json"
    results_path.write_text(
        json.dumps(
            {
                "summary": {
                    "mean_top_k_jaccard": 0.47619,
                    "min_top_k_jaccard": 0.42857,
                    "mean_spearman_correlation": 0.89240,
                    "min_spearman_correlation": 0.88846,
                }
            }
        )
    )

    monkeypatch.setattr(
        "src.dashboard.app.SHAP_CONSISTENCY_RESULTS",
        results_path,
    )

    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=20)

    nav_target = next(option for option in app.radio[0].options if "Explainability" in option)
    app.radio[0].set_value(nav_target)
    app.run(timeout=20)

    app.selectbox[0].set_value("Sampling Consistency")
    app.run(timeout=20)

    assert len(app.exception) == 0

    values = {metric.label: metric.value for metric in app.metric}

    assert values["Mean Spearman"] == "0.892"
    assert values["Min Spearman"] == "0.888"
    assert values["Mean Top-K Jaccard"] == "0.476"
    assert values["Min Top-K Jaccard"] == "0.429"


def test_shap_component_contract_identifies_dae_only() -> None:
    """Existing SHAP results must remain scoped to the DAE anomaly component."""
    payload = {
        "experiment": "dae_shap_consistency",
        "component": "DAE anomaly score",
    }

    assert payload["component"] == "DAE anomaly score"
    assert "gate" not in payload["component"].lower()


# ---------------------------------------------------------------------------
# Real Streamlit application smoke tests
# ---------------------------------------------------------------------------
def test_streamlit_dashboard_starts_without_exception() -> None:
    """The complete Streamlit application should start successfully."""
    app = AppTest.from_file(str(APP_PATH))

    app.run(timeout=20)

    assert len(app.exception) == 0


def test_streamlit_dashboard_exposes_navigation() -> None:
    """Dashboard should expose its primary Navigation radio control."""
    app = AppTest.from_file(str(APP_PATH))

    app.run(timeout=20)

    assert len(app.exception) == 0
    assert len(app.radio) == 1
    assert app.radio[0].label == "Navigation"


def test_streamlit_dashboard_contains_expected_navigation_pages() -> None:
    """Primary fraud-triage pages should be available in navigation."""
    app = AppTest.from_file(str(APP_PATH))

    app.run(timeout=20)

    assert len(app.exception) == 0

    options = list(app.radio[0].options)

    expected_pages = {
        "Command",
        "Prediction",
        "Batch Analysis",
        "Model Insights",
        "Conformal Triage",
        "Explainability",
        "Settings",
        "About",
    }

    for page in expected_pages:
        assert any(page in option for option in options)


@pytest.mark.parametrize(
    "expected_label",
    [
        "Command",
        "Prediction",
        "Batch Analysis",
        "Model Insights",
        "Conformal Triage",
        "Explainability",
        "Settings",
        "About",
    ],
)
def test_streamlit_navigation_pages_render_without_exception(
    expected_label: str,
) -> None:
    """Every dashboard page should render without an uncaught exception."""
    app = AppTest.from_file(str(APP_PATH))

    app.run(timeout=20)

    target = next(option for option in app.radio[0].options if expected_label in option)
    app.radio[0].set_value(target)
    app.run(timeout=20)

    assert len(app.exception) == 0
