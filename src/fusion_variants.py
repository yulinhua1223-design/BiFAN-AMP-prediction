"""Fusion operators for the BiFAN dual-branch architecture.

Every operator consumes the pooled sequence feature, the pooled handcrafted
feature and (optionally) the token-level ESM-2 embeddings, and returns a fused
representation of width ``d_model``. All operators are followed by the same
output projection inside ``AmpPredictor``, so a head-to-head comparison isolates
the fusion operator itself. Operators that need extra projections carry more
parameters; the per-variant parameter count is reported with the results.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _output_layer(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.LayerNorm(d_model),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class FusionBase(nn.Module):
    """Interface shared by every fusion operator."""

    def forward(self, seq_feat, hand_feat, tokens=None, token_mask=None):
        raise NotImplementedError


class ConcatFusion(FusionBase):
    """Concatenate along the channel axis and project back to ``d_model``."""

    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.combine = nn.Linear(2 * d_model, d_model)
        self.output_layer = _output_layer(d_model, dropout)

    def forward(self, seq_feat, hand_feat, tokens=None, token_mask=None):
        return self.output_layer(self.combine(torch.cat([seq_feat, hand_feat], dim=-1)))


class AddFusion(FusionBase):
    """Element-wise sum of the two branch representations."""

    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.output_layer = _output_layer(d_model, dropout)

    def forward(self, seq_feat, hand_feat, tokens=None, token_mask=None):
        return self.output_layer(seq_feat + hand_feat)


class HadamardFusion(FusionBase):
    """Element-wise product of the two branch representations."""

    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.output_layer = _output_layer(d_model, dropout)

    def forward(self, seq_feat, hand_feat, tokens=None, token_mask=None):
        return self.output_layer(seq_feat * hand_feat)


class BilinearFusion(FusionBase):
    """Bilinear interaction comparison operator."""

    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.proj_seq = nn.Linear(d_model, d_model)
        self.proj_hand = nn.Linear(d_model, d_model)
        self.output_layer = _output_layer(d_model, dropout)

    def forward(self, seq_feat, hand_feat, tokens=None, token_mask=None):
        interacted = self.proj_seq(seq_feat) * self.proj_hand(hand_feat)
        return self.output_layer(interacted + seq_feat)


class GatedBilinearFusion(FusionBase):
    """Bilinear interaction blended with the additive path through a learned gate."""

    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.proj_seq = nn.Linear(d_model, d_model)
        self.proj_hand = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.output_layer = _output_layer(d_model, dropout)

    def forward(self, seq_feat, hand_feat, tokens=None, token_mask=None):
        seq_proj = self.proj_seq(seq_feat)
        hand_proj = self.proj_hand(hand_feat)
        gate = self.gate(torch.cat([seq_feat, hand_feat], dim=-1))
        fused = gate * (seq_proj * hand_proj) + (1.0 - gate) * (seq_proj + hand_proj)
        return self.output_layer(fused + seq_feat)


class CrossAttentionFusion(FusionBase):
    """The handcrafted vector queries the token-level ESM-2 embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, num_heads: int = 4,
                 token_dim: int | None = None, **kwargs) -> None:
        super().__init__()
        token_dim = token_dim or d_model
        self.num_heads = num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(token_dim, d_model)
        self.value = nn.Linear(token_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = _output_layer(d_model, dropout)

    def forward(self, seq_feat, hand_feat, tokens=None, token_mask=None):
        if tokens is None:
            # degenerate case (no token input): fall back to additive fusion
            return self.output_layer(seq_feat + hand_feat)

        batch, length, _ = tokens.shape
        heads = self.num_heads
        width = self.norm.normalized_shape[0]
        head_dim = width // heads

        query = self.query(hand_feat).view(batch, heads, 1, head_dim)
        key = self.key(tokens).view(batch, length, heads, head_dim).transpose(1, 2)
        value = self.value(tokens).view(batch, length, heads, head_dim).transpose(1, 2)

        padding = None
        if token_mask is not None:
            padding = token_mask == 0

        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        ) if padding is None else self._masked_attention(query, key, value, padding, head_dim)

        attended = attended.transpose(1, 2).reshape(batch, 1, width).squeeze(1)
        return self.output_layer(self.norm(attended) + seq_feat)

    @staticmethod
    def _masked_attention(query, key, value, padding, head_dim):
        scale = 1.0 / float(head_dim) ** 0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)


FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "add": AddFusion,
    "hadamard": HadamardFusion,
    "bilinear": BilinearFusion,
    "gated": GatedBilinearFusion,
    "cross_attention": CrossAttentionFusion,
}

FUSION_NAMES = tuple(FUSION_REGISTRY)


def build_fusion(name: str, d_model: int, dropout: float = 0.1, **kwargs) -> FusionBase:
    if name not in FUSION_REGISTRY:
        raise KeyError(f"unknown fusion operator {name!r}; available: {sorted(FUSION_REGISTRY)}")
    return FUSION_REGISTRY[name](d_model=d_model, dropout=dropout, **kwargs)
