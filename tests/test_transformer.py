"""Unit tests for the FT-CAT architecture (Member B).

Verifies forward shapes across all three ablation variants, the temporal
cross-attention weights that drive the analyst heatmaps, gradient flow,
checkpoint metadata round-tripping, and the defensive shape validation.
Everything runs on tiny synthetic tensors so the suite stays fast.
"""

from __future__ import annotations

import pytest
import torch

from src.models.ft_transformer import (
    VARIANTS,
    AttentionMaps,
    FeatureTokenizer,
    FTCATransformer,
    HistoryEncoder,
    TabularMLP,
    build_ft_model,
    count_parameters,
    validate_batch,
)
from src.utils.config import Config

BATCH = 6
N_CONT = 6
CARDS = [3, 4]
SEQ_LEN = 5
SEQ_DIM = 4
D_MODEL = 8
N_HEADS = 2
N_TOKENS = 1 + N_CONT + len(CARDS)


@pytest.fixture()
def tiny_config() -> Config:
    """A small transformer configuration that trains in milliseconds."""
    return Config(
        {
            "transformer": {
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "dim_feedforward": 16,
                "n_layers": 1,
                "dropout": 0.0,
                "activation": "gelu",
                "norm_first": True,
                "seq_len": SEQ_LEN,
                "n_continuous": N_CONT,
                "categorical_cardinalities": CARDS,
            },
            "sequence": {"feature_cols": ["a", "b", "c", "d"]},
        }
    )


@pytest.fixture()
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A synthetic ``(x_cont, x_cat, seq)`` batch matching ``tiny_config``."""
    generator = torch.Generator().manual_seed(42)
    x_cont = torch.randn(BATCH, N_CONT, generator=generator)
    x_cat = torch.stack(
        [torch.randint(0, card, (BATCH,), generator=generator) for card in CARDS], dim=1
    )
    seq = torch.randn(BATCH, SEQ_LEN, SEQ_DIM, generator=generator)
    return x_cont, x_cat, seq


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_returns_one_logit_per_row(
    tiny_config: Config, batch: tuple[torch.Tensor, ...], variant: str
) -> None:
    """Every variant emits a flat ``(batch,)`` logit vector."""
    model = build_ft_model(tiny_config, variant)
    logits = model(*batch)
    assert logits.shape == (BATCH,)
    assert torch.isfinite(logits).all()


def test_build_ft_model_rejects_unknown_variant(tiny_config: Config) -> None:
    """A typo in the variant name fails loudly rather than silently."""
    with pytest.raises(ValueError, match="unknown variant"):
        build_ft_model(tiny_config, "ft_kat")


def test_build_ft_model_honours_dimension_overrides(tiny_config: Config) -> None:
    """A resolved FeatureSpec can override the config mirrors."""
    model = build_ft_model(
        tiny_config,
        "ft_cat",
        n_continuous=3,
        categorical_cardinalities=[9],
        seq_dim=2,
    )
    meta = model.state_meta()
    assert meta["n_continuous"] == 3
    assert meta["categorical_cardinalities"] == [9]
    assert meta["seq_dim"] == 2


def test_variants_differ_in_capacity(tiny_config: Config) -> None:
    """Dropping cross-attention removes parameters; the MLP control has more."""
    sizes = {v: count_parameters(build_ft_model(tiny_config, v)) for v in VARIANTS}
    assert sizes["ft_self_only"] < sizes["ft_cat"]
    assert sizes["mlp"] > sizes["ft_cat"]


def test_checkpoint_stays_within_the_size_budget() -> None:
    """The production-sized model honours the sub-1 MB deployment claim."""
    model = FTCATransformer(
        n_continuous=100,
        categorical_cardinalities=[6] * 30 + [1548],
        seq_len=5,
        seq_dim=10,
    )
    assert count_parameters(model) * 4 < 1_000_000


def test_cross_attention_weights_have_expected_shape(
    tiny_config: Config, batch: tuple[torch.Tensor, ...]
) -> None:
    """Weights are per-head and map every token onto every history slot."""
    model = build_ft_model(tiny_config, "ft_cat").eval()
    with torch.no_grad():
        logits, maps = model(*batch, return_attention=True)
    assert isinstance(maps, AttentionMaps)
    assert maps.cross_attn is not None
    assert maps.cross_attn.shape == (BATCH, N_HEADS, N_TOKENS, SEQ_LEN)
    assert logits.shape == (BATCH,)


def test_cross_attention_rows_are_distributions(
    tiny_config: Config, batch: tuple[torch.Tensor, ...]
) -> None:
    """Each token's attention over history sums to one in eval mode."""
    model = build_ft_model(tiny_config, "ft_cat").eval()
    with torch.no_grad():
        _, maps = model(*batch, return_attention=True)
    totals = maps.cross_attn.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


@pytest.mark.parametrize("variant", ["ft_self_only", "mlp"])
def test_variants_without_cross_attention_report_no_maps(
    tiny_config: Config, batch: tuple[torch.Tensor, ...], variant: str
) -> None:
    """Ablation arms keep the interface but carry no attention weights."""
    model = build_ft_model(tiny_config, variant).eval()
    with torch.no_grad():
        _, maps = model(*batch, return_attention=True)
    assert maps.cross_attn is None


def test_cross_attention_makes_history_matter(
    tiny_config: Config, batch: tuple[torch.Tensor, ...]
) -> None:
    """FT-CAT responds to history; the self-attention-only arm cannot.

    This is the behavioural claim the whole architecture rests on, so it is
    asserted directly rather than inferred from a metric.
    """
    x_cont, x_cat, seq = batch
    other = torch.randn(BATCH, SEQ_LEN, SEQ_DIM, generator=torch.Generator().manual_seed(7))

    ft_cat = build_ft_model(tiny_config, "ft_cat").eval()
    self_only = build_ft_model(tiny_config, "ft_self_only").eval()
    with torch.no_grad():
        assert not torch.allclose(ft_cat(x_cont, x_cat, seq), ft_cat(x_cont, x_cat, other))
        assert torch.allclose(self_only(x_cont, x_cat, seq), self_only(x_cont, x_cat, other))


def test_padding_mask_flags_zero_filled_slots() -> None:
    """Zero rows are padding; a row of real values is not."""
    seq = torch.zeros(1, 3, 2)
    seq[0, 2] = torch.tensor([1.0, -1.0])
    mask = HistoryEncoder.padding_mask(seq)
    assert mask.tolist() == [[True, True, False]]


def test_padding_mask_keeps_history_free_rows_unmasked() -> None:
    """A fully padded window stays unmasked so attention cannot emit NaN."""
    mask = HistoryEncoder.padding_mask(torch.zeros(1, 4, 2))
    assert mask.tolist() == [[False, False, False, False]]


def test_forward_is_finite_without_any_history(
    tiny_config: Config, batch: tuple[torch.Tensor, ...]
) -> None:
    """A cardholder with no prior transactions scores without NaNs."""
    x_cont, x_cat, _ = batch
    model = build_ft_model(tiny_config, "ft_cat").eval()
    with torch.no_grad():
        logits = model(x_cont, x_cat, torch.zeros(BATCH, SEQ_LEN, SEQ_DIM))
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("kwargs", [{"seq_len": 0}, {"seq_dim": 0}])
def test_history_encoder_rejects_degenerate_shapes(kwargs: dict) -> None:
    """A zero-length or zero-width history window is a configuration error."""
    base = {"seq_len": 4, "seq_dim": 2, "d_model": 8, "dropout": 0.0}
    base.update(kwargs)
    with pytest.raises(ValueError, match="must be positive"):
        HistoryEncoder(**base)


@pytest.mark.parametrize("variant", VARIANTS)
def test_gradients_are_finite_and_non_zero(
    tiny_config: Config, batch: tuple[torch.Tensor, ...], variant: str
) -> None:
    """Every parameter that should learn receives a usable gradient."""
    model = build_ft_model(tiny_config, variant)
    model(*batch).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    assert torch.isfinite(total_norm)
    assert float(total_norm) > 0.0


def test_gradients_reach_the_feature_tokenizer(
    tiny_config: Config, batch: tuple[torch.Tensor, ...]
) -> None:
    """The per-feature projections train, not just the head."""
    model = build_ft_model(tiny_config, "ft_cat")
    model(*batch).sum().backward()
    grad = model.tokenizer.cont_weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


@pytest.mark.parametrize("variant", VARIANTS)
def test_state_meta_rebuilds_an_identical_model(
    tiny_config: Config, batch: tuple[torch.Tensor, ...], variant: str
) -> None:
    """A checkpoint is self-describing: meta + weights reproduce the logits."""
    model = build_ft_model(tiny_config, variant).eval()
    clone_cls = TabularMLP if variant == "mlp" else FTCATransformer
    clone = clone_cls(**model.state_meta())
    clone.load_state_dict(model.state_dict())
    clone.eval()
    with torch.no_grad():
        assert torch.allclose(model(*batch), clone(*batch), atol=1e-6)


def test_tokenizer_token_count(tiny_config: Config) -> None:
    """Token accounting matches continuous + categorical + [CLS]."""
    assert FeatureTokenizer(N_CONT, CARDS, D_MODEL).n_tokens == N_TOKENS


@pytest.mark.parametrize(
    "args, message",
    [
        ((4, [3], 0), "d_model must be positive"),
        ((-1, [3], 8), "n_continuous must be non-negative"),
        ((4, [0], 8), "cardinalities must be positive"),
        ((0, [], 8), "at least one continuous or categorical"),
    ],
)
def test_tokenizer_rejects_invalid_geometry(args: tuple, message: str) -> None:
    """Impossible tokenizer geometries fail at construction time."""
    with pytest.raises(ValueError, match=message):
        FeatureTokenizer(*args)


def test_tokenizer_rejects_out_of_range_category_code() -> None:
    """A code beyond the table size names the offending feature."""
    tokenizer = FeatureTokenizer(1, [3], D_MODEL)
    with pytest.raises(IndexError, match="exceeds table size"):
        tokenizer(torch.zeros(2, 1), torch.tensor([[0], [7]]))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_layers": -1}, "n_layers must be non-negative"),
        ({"n_heads": 3}, "must be divisible"),
    ],
)
def test_transformer_rejects_invalid_hyperparameters(kwargs: dict, message: str) -> None:
    """Hyperparameter combinations that cannot form a model are rejected."""
    base = {
        "n_continuous": N_CONT,
        "categorical_cardinalities": CARDS,
        "seq_len": SEQ_LEN,
        "seq_dim": SEQ_DIM,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        FTCATransformer(**base)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c, k, s: (c[:, :2], k, s), "expected 6 continuous features"),
        (lambda c, k, s: (c, k[:, :1], s), "expected 2 categorical features"),
        (lambda c, k, s: (c, k, s[:, :2]), "expected history of shape"),
        (lambda c, k, s: (c[:2], k, s), "inconsistent batch sizes"),
        (lambda c, k, s: (c.unsqueeze(-1), k, s), "x_cont must be 2D"),
        (lambda c, k, s: (c, k.unsqueeze(-1), s), "x_cat must be 2D"),
        (lambda c, k, s: (c, k, s[:, 0]), "seq must be 3D"),
    ],
)
def test_forward_rejects_malformed_batches(
    tiny_config: Config, batch: tuple[torch.Tensor, ...], mutate, message: str
) -> None:
    """Shape drift between the data pipeline and the model is caught early."""
    model = build_ft_model(tiny_config, "ft_cat")
    with pytest.raises(ValueError, match=message):
        model(*mutate(*batch))


def test_validate_batch_accepts_a_well_formed_batch(
    batch: tuple[torch.Tensor, ...],
) -> None:
    """The happy path raises nothing."""
    validate_batch(
        *batch,
        n_continuous=N_CONT,
        n_categorical=len(CARDS),
        seq_len=SEQ_LEN,
        seq_dim=SEQ_DIM,
    )
