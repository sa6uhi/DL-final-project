# Import necessary libraries for testing
import numpy as np
import pytest
import torch

from src.explainability.shap_explainer import (
    DAEAnomalyScoreWrapper,
    compute_shap_values,
    explain_transaction,
    rank_feature_drivers,
)


# Test cases for the SHAP explainer module
def test_explain_transaction_rejects_empty_features() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        explain_transaction([])


def test_explain_transaction_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        explain_transaction([1.0, 2.0], top_k=0)


def test_explain_transaction_rejects_non_finite_features() -> None:
    with pytest.raises(ValueError, match="finite"):
        explain_transaction([1.0, float("nan")])


def test_explain_transaction_rejects_invalid_ft_probability() -> None:
    with pytest.raises(ValueError, match="ft_probability"):
        explain_transaction([1.0, 2.0], ft_probability=1.5)


def test_explain_transaction_requires_model() -> None:
    with pytest.raises(ValueError, match="model is required"):
        explain_transaction([1.0, 2.0])


def test_dae_wrapper_matches_anomaly_score() -> None:
    class IdentityModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 0.5

    model = IdentityModel()
    wrapper = DAEAnomalyScoreWrapper(model, l1_gamma=0.4)

    x = torch.tensor([[1.0, 2.0]])

    wrapped_score = wrapper(x)

    reconstruction = model(x)
    residual = x - reconstruction
    expected = residual.pow(2).sum(dim=-1)
    expected += 0.4 * residual.abs().sum(dim=-1)

    assert wrapped_score.shape == (1, 1)
    assert torch.allclose(wrapped_score.squeeze(-1), expected)


def test_dae_wrapper_rejects_negative_l1_gamma() -> None:
    model = torch.nn.Identity()

    with pytest.raises(ValueError, match="non-negative"):
        DAEAnomalyScoreWrapper(model, l1_gamma=-0.1)


def test_compute_shap_values_returns_expected_shape() -> None:
    class SimpleModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 0.5

    model = SimpleModel()

    background = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.2],
            [0.4, 0.4],
        ],
        dtype=torch.float32,
    )

    samples = torch.tensor(
        [[1.0, 2.0]],
        dtype=torch.float32,
    )

    values = compute_shap_values(
        model=model,
        background=background,
        samples=samples,
    )

    assert values.shape == (1, 2)
    assert np.isfinite(values).all()


def test_rank_feature_drivers_orders_by_absolute_attribution() -> None:
    features = [10.0, 20.0, 30.0]
    shap_values = np.array([0.1, -0.8, 0.4])

    drivers = rank_feature_drivers(
        features=features,
        shap_values=shap_values,
        top_k=2,
    )

    assert len(drivers) == 2
    assert drivers[0]["feature_name"] == "feature_1"
    assert drivers[0]["attribution"] == pytest.approx(-0.8)
    assert drivers[0]["value"] == pytest.approx(20.0)

    assert drivers[1]["feature_name"] == "feature_2"


def test_rank_feature_drivers_supports_feature_names() -> None:
    drivers = rank_feature_drivers(
        features=[100.0, 2.0],
        shap_values=np.array([0.7, 0.1]),
        top_k=1,
        feature_names=["TransactionAmt", "D1"],
    )

    assert drivers == [
        {
            "feature_name": "TransactionAmt",
            "attribution": pytest.approx(0.7),
            "value": pytest.approx(100.0),
        }
    ]


def test_explain_transaction_returns_ranked_drivers() -> None:
    class SimpleModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 0.5

    model = SimpleModel()

    background = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.2],
            [0.4, 0.4],
        ],
        dtype=torch.float32,
    )

    drivers = explain_transaction(
        features=[1.0, 2.0],
        model=model,
        background=background,
        top_k=2,
    )

    assert len(drivers) == 2
    assert drivers[0]["feature_name"] in {"feature_0", "feature_1"}
    assert "attribution" in drivers[0]
    assert "value" in drivers[0]
