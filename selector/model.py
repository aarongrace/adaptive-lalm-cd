"""
Selector architectures.

:class:`SelectorMLP` is the paper's reported method: a plain feed-forward head
over a pooled feature vector, trained with multi-label BCE and routed by
argmax at inference (Section III-C). It is deliberately small. The whole point
of the adaptive selector is that it costs nothing at inference beyond the
clean forward pass the decoder already runs, so anything that needed its own
sizeable encoder would undercut the claim. The head-size sweep (Section VI-A)
found a 3-layer ``[512, 256, 128]`` taper optimal, with accuracy degrading
consistently past 3 layers from overfitting; :data:`SELECTOR_SPEC` is that
configuration.

:class:`CrossAttentionSelector` is the alternative in the last two rows of the
feature ablation (Table VI): instead of pooling audio frames into one vector,
the last-token state attends over them. It reaches 76.4%, essentially matching
the last-token feature alone, which is the paper's evidence that externally
re-injecting audio adds nothing once cross-modal attention has already folded
it into the last-token state during the forward pass.

Both take their inputs as a tuple from ``selector/data.py`` and emit one logit
per candidate branch, so ``selector/train.py`` drives either without knowing
which it has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    hidden_dims: List[int]
    dropout: float = 0.1


# The paper's reported best selector head (Section VI-A): 3-layer taper,
# reaching 76.7% / 76.4% top-1 routing accuracy on Qwen2 / AF3 AH Existence
# with N = 4 candidate branches.
SELECTOR_SPEC = SelectorSpec("mlp_3taper", hidden_dims=[512, 256, 128], dropout=0.1)

# Depth variants used by the head-architecture sweep.
HEAD_VARIANTS: dict[str, SelectorSpec] = {
    "mlp_1": SelectorSpec("mlp_1", hidden_dims=[512]),
    "mlp_2": SelectorSpec("mlp_2", hidden_dims=[512, 256]),
    "mlp_3taper": SELECTOR_SPEC,
    "mlp_4": SelectorSpec("mlp_4", hidden_dims=[512, 256, 128, 64]),
    "mlp_wide": SelectorSpec("mlp_wide", hidden_dims=[1024, 512, 256]),
}


class SelectorMLP(nn.Module):
    """Pooled feature vector -> one logit per candidate branch."""

    def __init__(self, input_dim: int, num_classes: int, spec: SelectorSpec = SELECTOR_SPEC):
        super().__init__()
        self.spec = spec
        layers: list[nn.Module] = []
        prev = input_dim
        for width in spec.hidden_dims:
            layers += [nn.Linear(prev, width), nn.GELU(), nn.Dropout(spec.dropout)]
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionSelector(nn.Module):
    """Unidirectional cross-attention from a pooled text state over audio frames.

        attended = cross_attn(Q=text, K=V=proj(frames))
        out      = MLP(concat(text, LayerNorm(text + attended)))

    The residual keeps the text state on a direct path to the head, so the
    model can fall back to text-only behaviour if attending over the frames
    contributes nothing -- which, per Table VI, is roughly what happens.
    """

    def __init__(self, text_dim: int, frame_dim: int, num_classes: int,
                 n_heads: int = 4, spec: SelectorSpec = SELECTOR_SPEC):
        super().__init__()
        self.text_dim = text_dim
        self.frame_proj = nn.Linear(frame_dim, text_dim) if frame_dim != text_dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(text_dim, n_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(text_dim)
        self.head = SelectorMLP(text_dim * 2, num_classes, spec)

    def forward(self, text: torch.Tensor, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # text: [B, D_text]   frames: [B, T, D_frame]   mask: [B, T] bool, True = valid
        kv = self.frame_proj(frames)
        query = text.unsqueeze(1)
        attended, _ = self.cross_attn(query, kv, kv, key_padding_mask=~mask)
        fused = self.norm(attended.squeeze(1) + text)
        return self.head(torch.cat([text, fused], dim=-1))


def build_selector(kind: str, input_dims: Sequence[int], num_classes: int,
                   spec: SelectorSpec = SELECTOR_SPEC) -> nn.Module:
    """Construct the head matching a dataset's feature kind.

    ``input_dims`` comes from ``SelectorDataset.input_dims``: one entry for a
    pooled vector feature, two (query dim, frame dim) for a cross-attention
    feature. Centralising this keeps the training scripts from having to know
    which architecture a given Table VI row implies.
    """
    if kind == "vector":
        if len(input_dims) != 1:
            raise ValueError(f"A vector selector takes exactly one input dim, got {input_dims}")
        return SelectorMLP(input_dims[0], num_classes, spec)
    if kind == "cross_attention":
        if len(input_dims) != 2:
            raise ValueError(
                f"A cross-attention selector takes (text_dim, frame_dim), got {input_dims}"
            )
        return CrossAttentionSelector(input_dims[0], input_dims[1], num_classes, spec=spec)
    raise ValueError(f"Unknown selector kind {kind!r}; expected 'vector' or 'cross_attention'")
