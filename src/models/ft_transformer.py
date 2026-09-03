"""Feature-Tokenizer Cross-Attention Transformer (FT-CAT) for fraud detection.

The model treats a transaction as a *sequence of feature tokens* rather than a
flat vector. Each continuous feature gets its own learned projection, each
categorical feature its own embedding table, and a learnable ``[CLS]`` token
aggregates the result. Standard self-attention blocks then model interactions
*within* the current transaction.

* ``ft_cat``       -- tokenizer + self-attention + temporal cross-attention.
* ``ft_self_only`` -- tokenizer + self-attention, history ignored.
* ``mlp``          -- flat MLP control over the same inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

from src.utils.logger import get_logger

logger = get_logger(__name__)

VARIANTS: tuple[str, ...] = ("ft_cat", "ft_self_only", "mlp")


@dataclass(frozen=True)
class AttentionMaps:
    """Attention weights captured during a forward pass."""

    cross_attn: torch.Tensor | None = None


def validate_batch(
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    seq: torch.Tensor,
    n_continuous: int,
    n_categorical: int,
    seq_len: int,
    seq_dim: int,
) -> None:
    """Assert that a batch matches the shapes the model was built for."""
    if x_cont.dim() != 2:
        raise ValueError(f"x_cont must be 2D (batch, n_continuous), got {tuple(x_cont.shape)}")
    if x_cat.dim() != 2:
        raise ValueError(f"x_cat must be 2D (batch, n_categorical), got {tuple(x_cat.shape)}")
    if seq.dim() != 3:
        raise ValueError(f"seq must be 3D (batch, seq_len, seq_dim), got {tuple(seq.shape)}")
    if x_cont.size(1) != n_continuous:
        raise ValueError(f"expected {n_continuous} continuous features, got {x_cont.size(1)}")
    if x_cat.size(1) != n_categorical:
        raise ValueError(f"expected {n_categorical} categorical features, got {x_cat.size(1)}")
    if tuple(seq.shape[1:]) != (seq_len, seq_dim):
        raise ValueError(
            f"expected history of shape (*, {seq_len}, {seq_dim}), got {tuple(seq.shape)}"
        )
    batch = x_cont.size(0)
    if x_cat.size(0) != batch or seq.size(0) != batch:
        raise ValueError(
            f"inconsistent batch sizes: x_cont={batch}, x_cat={x_cat.size(0)}, seq={seq.size(0)}"
        )


class FeatureTokenizer(nn.Module):
    """Project heterogeneous tabular features into a shared token space.

    Continuous feature ``j`` becomes ``e_j = x_j * w_j + b_j`` with its own
    ``w_j, b_j`` in ``R^d``; categorical feature ``m`` is looked up in its own
    embedding table. A learnable ``[CLS]`` token is prepended and later read
    out by the classification head.
    """

    def __init__(
        self,
        n_continuous: int,
        categorical_cardinalities: Sequence[int],
        d_model: int,
    ) -> None:
        """Initialize the tokenizer."""
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_continuous < 0:
            raise ValueError(f"n_continuous must be non-negative, got {n_continuous}")
        cards = [int(c) for c in categorical_cardinalities]
        if any(card <= 0 for card in cards):
            raise ValueError(f"cardinalities must be positive, got {cards}")
        if n_continuous == 0 and not cards:
            raise ValueError("tokenizer needs at least one continuous or categorical feature")

        self.n_continuous = int(n_continuous)
        self.cardinalities = cards
        self.d_model = int(d_model)

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        if n_continuous > 0:
            self.cont_weight = nn.Parameter(torch.empty(n_continuous, d_model))
            self.cont_bias = nn.Parameter(torch.empty(n_continuous, d_model))
        else:
            self.register_parameter("cont_weight", None)
            self.register_parameter("cont_bias", None)
        self.cat_embeddings = nn.ModuleList(nn.Embedding(card, d_model) for card in cards)
        self.reset_parameters()

    @property
    def n_tokens(self) -> int:
        """Total tokens emitted, including ``[CLS]``."""
        return 1 + self.n_continuous + len(self.cardinalities)

    def reset_parameters(self) -> None:
        """Initialize every parameter uniformly in +/- ``1/sqrt(d_model)``.

        This is the scheme used by the FT-Transformer paper; it keeps token
        magnitudes comparable across the continuous and categorical paths so
        neither dominates early training.
        """
        bound = 1.0 / math.sqrt(self.d_model)
        nn.init.uniform_(self.cls_token, -bound, bound)
        if self.cont_weight is not None:
            nn.init.uniform_(self.cont_weight, -bound, bound)
            nn.init.uniform_(self.cont_bias, -bound, bound)
        for embedding in self.cat_embeddings:
            nn.init.uniform_(embedding.weight, -bound, bound)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Tokenize one batch of transactions."""
        batch = x_cont.size(0) if self.n_continuous else x_cat.size(0)
        tokens = [self.cls_token.expand(batch, -1, -1)]

        if self.cont_weight is not None:
            # (B, F, 1) * (1, F, d) -> (B, F, d): one projection per feature.
            tokens.append(x_cont.unsqueeze(-1) * self.cont_weight.unsqueeze(0) + self.cont_bias)

        for index, embedding in enumerate(self.cat_embeddings):
            codes = x_cat[:, index]
            if codes.numel() and int(codes.max()) >= embedding.num_embeddings:
                raise IndexError(
                    f"category code {int(codes.max())} exceeds table size "
                    f"{embedding.num_embeddings} for categorical feature {index}"
                )
            tokens.append(embedding(codes).unsqueeze(1))

        return torch.cat(tokens, dim=1)


class HistoryEncoder(nn.Module):
    """Embed the K most recent transactions into cross-attention memory."""

    def __init__(self, seq_len: int, seq_dim: int, d_model: int, dropout: float) -> None:
        """Initialize the history encoder."""
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if seq_dim <= 0:
            raise ValueError(f"seq_dim must be positive, got {seq_dim}")
        self.seq_len = int(seq_len)
        self.seq_dim = int(seq_dim)
        self.projection = nn.Linear(seq_dim, d_model)
        self.position = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.position, -1.0 / math.sqrt(d_model), 1.0 / math.sqrt(d_model))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode a batch of history windows."""
        return self.dropout(self.norm(self.projection(seq) + self.position))

    @staticmethod
    def padding_mask(seq: torch.Tensor) -> torch.Tensor:
        """Flag history slots that are zero padding rather than real events.

        ``build_historical_sequences`` zero-fills slots for cardholders with
        fewer than ``K`` prior transactions, so an all-zero row is padding.
        Rows where *every* slot is padding are deliberately left unmasked:
        masking them entirely would make the attention softmax degenerate and
        emit NaNs, so such transactions instead attend over an all-zero
        history, which is the correct "no history available" signal.
        """
        padded = seq.abs().sum(dim=-1) == 0
        fully_padded = padded.all(dim=1, keepdim=True)
        return padded & ~fully_padded


class TemporalCrossAttention(nn.Module):
    """Multi-head cross-attention from feature tokens onto transaction history.

    Queries are the current transaction feature tokens; keys and values are
    the encoded history. The returned per-head weights are what the analyst
    dashboard and the paper heatmaps visualise.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        """Initialize the cross-attention block."""
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attend from feature tokens onto historical transactions."""
        attended, weights = self.attention(
            query=tokens,
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        return self.norm(tokens + self.dropout(attended)), weights


class FTCATransformer(nn.Module):
    """Feature-Tokenizer Transformer with optional temporal cross-attention.

    Setting ``use_cross_attention=False`` yields the ``ft_self_only`` ablation
    arm: an otherwise identical model that never sees transaction history.
    """

    def __init__(
        self,
        n_continuous: int,
        categorical_cardinalities: Sequence[int],
        seq_len: int,
        seq_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        dim_feedforward: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        use_cross_attention: bool = True,
    ) -> None:
        """Build the FT-CAT classifier."""
        super().__init__()
        if n_layers < 0:
            raise ValueError(f"n_layers must be non-negative, got {n_layers}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.tokenizer = FeatureTokenizer(n_continuous, categorical_cardinalities, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True,
        )
        # Nested tensors are unavailable under pre-norm ordering; disabling the fast path explicitly keeps the constructor from warning on every build.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.use_cross_attention = bool(use_cross_attention)
        if self.use_cross_attention:
            self.history_encoder = HistoryEncoder(seq_len, seq_dim, d_model, dropout)
            self.cross_attention = TemporalCrossAttention(d_model, n_heads, dropout)
        else:
            self.history_encoder = None
            self.cross_attention = None

        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.bias)

        self._meta: dict[str, Any] = {
            "n_continuous": int(n_continuous),
            "categorical_cardinalities": [int(c) for c in categorical_cardinalities],
            "seq_len": int(seq_len),
            "seq_dim": int(seq_dim),
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "dim_feedforward": int(dim_feedforward),
            "n_layers": int(n_layers),
            "dropout": float(dropout),
            "activation": str(activation),
            "norm_first": bool(norm_first),
            "use_cross_attention": self.use_cross_attention,
        }

    def state_meta(self) -> dict[str, Any]:
        """Return the constructor arguments needed to rebuild this model."""
        return dict(self._meta)

    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        seq: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionMaps]:
        """Score a batch of transactions."""
        meta = self._meta
        validate_batch(
            x_cont,
            x_cat,
            seq,
            n_continuous=meta["n_continuous"],
            n_categorical=len(meta["categorical_cardinalities"]),
            seq_len=meta["seq_len"],
            seq_dim=meta["seq_dim"],
        )

        tokens = self.encoder(self.tokenizer(x_cont, x_cat))

        cross_weights: torch.Tensor | None = None
        if self.cross_attention is not None and self.history_encoder is not None:
            memory = self.history_encoder(seq)
            tokens, cross_weights = self.cross_attention(
                tokens,
                memory,
                key_padding_mask=HistoryEncoder.padding_mask(seq),
                need_weights=return_attention,
            )

        logits = self.head(self.head_norm(tokens[:, 0])).squeeze(-1)
        if return_attention:
            return logits, AttentionMaps(cross_attn=cross_weights)
        return logits


class TabularMLP(nn.Module):
    """Flat multi-layer perceptron control arm for the Exp-3 ablation.

    Consumes exactly the same inputs as :class:`FTCATransformer` -- including
    the flattened history window -- but with no attention anywhere, isolating
    how much of the performance comes from the attention structure rather
    than from the features themselves.
    """

    def __init__(
        self,
        n_continuous: int,
        categorical_cardinalities: Sequence[int],
        seq_len: int,
        seq_dim: int,
        d_model: int = 32,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        """Build the MLP control."""
        super().__init__()
        cards = [int(c) for c in categorical_cardinalities]
        self.embeddings = nn.ModuleList(nn.Embedding(card, d_model) for card in cards)
        in_features = n_continuous + len(cards) * d_model + seq_len * seq_dim
        hidden = dim_feedforward * 2
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )
        self._meta: dict[str, Any] = {
            "n_continuous": int(n_continuous),
            "categorical_cardinalities": cards,
            "seq_len": int(seq_len),
            "seq_dim": int(seq_dim),
            "d_model": int(d_model),
            "dim_feedforward": int(dim_feedforward),
            "dropout": float(dropout),
        }

    def state_meta(self) -> dict[str, Any]:
        """Return the constructor arguments needed to rebuild this model."""
        return dict(self._meta)

    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        seq: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionMaps]:
        """Score a batch of transactions."""
        meta = self._meta
        validate_batch(
            x_cont,
            x_cat,
            seq,
            n_continuous=meta["n_continuous"],
            n_categorical=len(meta["categorical_cardinalities"]),
            seq_len=meta["seq_len"],
            seq_dim=meta["seq_dim"],
        )
        parts = [x_cont]
        parts += [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        parts.append(seq.flatten(start_dim=1))
        logits = self.network(torch.cat(parts, dim=1)).squeeze(-1)
        if return_attention:
            return logits, AttentionMaps(cross_attn=None)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_ft_model(
    config: Any,
    variant: str = "ft_cat",
    n_continuous: int | None = None,
    categorical_cardinalities: Sequence[int] | None = None,
    seq_dim: int | None = None,
) -> nn.Module:
    """Construct one ablation variant from the central configuration.

    Dimensions default to the ``transformer`` block of ``config/config.yaml``
    but may be overridden by a resolved
    :class:`~src.training.feature_selection.FeatureSpec`, which is the
    authoritative contract once the real data has been prepared."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")

    tcfg = config.transformer
    cards = (
        list(tcfg.categorical_cardinalities)
        if categorical_cardinalities is None
        else list(categorical_cardinalities)
    )
    kwargs: dict[str, Any] = {
        "n_continuous": int(tcfg.n_continuous if n_continuous is None else n_continuous),
        "categorical_cardinalities": [int(c) for c in cards],
        "seq_len": int(tcfg.seq_len),
        "seq_dim": int(len(config.sequence.feature_cols) if seq_dim is None else seq_dim),
        "d_model": int(tcfg.d_model),
        "n_heads": int(tcfg.n_heads),
        "dim_feedforward": int(tcfg.dim_feedforward),
        "n_layers": int(tcfg.n_layers),
        "dropout": float(tcfg.dropout),
    }

    model: nn.Module
    if variant == "mlp":
        model = TabularMLP(**kwargs)
    else:
        model = FTCATransformer(
            **kwargs,
            activation=str(tcfg.activation),
            norm_first=bool(tcfg.norm_first),
            use_cross_attention=variant == "ft_cat",
        )

    logger.info(
        "Built %r variant: %d trainable parameters (%.0f KB fp32)",
        variant,
        count_parameters(model),
        count_parameters(model) * 4 / 1024,
    )
    return model
