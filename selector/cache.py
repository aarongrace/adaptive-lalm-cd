"""
Clean-branch feature caching for the selector.

The selector must choose a negative branch *before* any perturbation is
evaluated, so its only legal input is what the model computes from the clean,
unperturbed audio and question -- the same forward pass every decoding run
already performs. Reading anything derived from a perturbed branch would leak
the answer the selector is supposed to predict.

This module runs that one forward pass per example, using the very same
encoding path the decoder uses (``decoding.engine.get_adapter``, not a
reimplementation), and caches the tensors the paper's feature ablation
(Table VI) is built from. Each example is keyed by
``helpers.datasets.example_key`` and stored as one ``.pt`` file:

  ``layers``           ``[L+1, H]``  last real token's hidden vector at every
                       layer, including the input-embedding layer. The paper's
                       reported best feature is the first/middle/last slice of
                       this tensor concatenated.
  ``layer_means``      ``[L+1, H]``  the same layers mean-pooled over all
                       non-padding positions instead. This is what the
                       "mean pool" rows of Table VI ablate, and the contrast
                       between the two is the paper's central feature finding:
                       in a causal decoder only the final position has attended
                       the complete input, so pooling dilutes it.
  ``embed_seq``        ``[<=SEQ_CAP, H]``  strided sample of the LM's combined
                       input-embedding sequence (projected audio frames and
                       text tokens together, before any attention). Used as the
                       key/value sequence by the cross-attention selector.
  ``raw_audio``        ``[<=SEQ_CAP, D]``  strided sample of the audio
                       encoder's own output, before projection into the LM
                       space. Present only when the encoder module was found.
  ``projected_audio``  ``[<=SEQ_CAP, H]``  the same frames after the learned
                       projection into the LM embedding space.

The two audio tensors come from forward hooks on the audio tower and its
projector, so they are the encoder's features in isolation rather than the
combined sequence. Module naming differs across model implementations and
across ``transformers`` releases, so discovery is by candidate name with a
graceful fallback: if a hook target cannot be found, the hidden-state features
are still cached and only the audio-feature rows of the ablation become
unavailable, with an explicit warning rather than a silent omission.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import torch
from tqdm import tqdm

from decoding.engine import get_adapter, load_batch_audio
from helpers.config import DEFAULT_BATCH_SIZE, RunConfig
from helpers.datasets import example_key

# Stored sequence length for the frame-level tensors. Sequences longer than
# this are evenly strided down: the cross-attention selector needs coverage of
# the whole clip, not full temporal resolution, and 64 frames per example keeps
# a 10,800-example cache to a manageable size on disk.
SEQ_CAP = 64

# Candidate attribute paths for the audio encoder and its projection head.
# Tried in order; the first that resolves to a module is hooked.
AUDIO_TOWER_PATHS = (
    "audio_tower",
    "model.audio_tower",
    "audio_encoder",
    "model.audio_encoder",
    "sound_tower",
)
AUDIO_PROJECTOR_PATHS = (
    "multi_modal_projector",
    "model.multi_modal_projector",
    "audio_projector",
    "model.audio_projector",
    "mm_projector",
)


# Fields that are frame sequences over time and may be strided down. The
# per-layer tensors are deliberately excluded: striding them would renumber the
# layer axis and silently corrupt the first/middle/final slice the reported
# feature is built from.
STRIDED_FIELDS = ("embed_seq", "raw_audio", "projected_audio")


def _stride_to(seq: torch.Tensor, cap: int = SEQ_CAP) -> torch.Tensor:
    """Evenly subsample a ``[T, H]`` tensor down to at most ``cap`` positions."""
    length = seq.shape[0]
    if length <= cap:
        return seq
    index = torch.linspace(0, length - 1, steps=cap).round().long()
    return seq[index]


def cache_path(cfg: RunConfig, key: str) -> Path:
    """One file per example, named by a hash of its key.

    The key embeds a full audio path and the question text, neither of which
    is safe to use as a filename; hashing keeps the layout flat and portable.
    """
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
    return cfg.cache_dir / f"{digest}.pt"


def has_cached(cfg: RunConfig, key: str) -> bool:
    return cache_path(cfg, key).exists()


def save_example_cache(cfg: RunConfig, key: str, tensors: dict[str, torch.Tensor]) -> None:
    path = cache_path(cfg, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key": key}
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        if name in STRIDED_FIELDS and tensor.dim() == 2:
            tensor = _stride_to(tensor)
        payload[name] = tensor.cpu()
    torch.save(payload, path)


def load_example_cache(cfg: RunConfig, key: str) -> dict:
    path = cache_path(cfg, key)
    if not path.exists():
        raise FileNotFoundError(f"No cache for example {key!r} at {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Audio-encoder hooks
# ---------------------------------------------------------------------------

def _resolve_module(model, dotted: str):
    """Follow a dotted attribute path, returning None if any step is missing."""
    current = model
    for part in dotted.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current if isinstance(current, torch.nn.Module) else None


def _first_tensor(output):
    """Unwrap the tensor from a module output that may be a tuple or a dataclass."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    return getattr(output, "last_hidden_state", None)


class AudioFeatureHooks:
    """Capture the audio tower's output and its projected form for one batch.

    Used as a context manager so the hooks are always removed, even if the
    forward pass raises: a leaked hook would keep accumulating tensors across
    batches and quietly exhaust GPU memory.
    """

    def __init__(self, model):
        self.raw: Optional[torch.Tensor] = None
        self.projected: Optional[torch.Tensor] = None
        self._handles: list = []
        self.tower = next(
            (m for m in (_resolve_module(model, p) for p in AUDIO_TOWER_PATHS) if m is not None),
            None,
        )
        self.projector = next(
            (m for m in (_resolve_module(model, p) for p in AUDIO_PROJECTOR_PATHS) if m is not None),
            None,
        )

    @property
    def available(self) -> bool:
        return self.tower is not None or self.projector is not None

    def describe(self) -> str:
        found = []
        found.append(f"audio tower={'found' if self.tower is not None else 'not found'}")
        found.append(f"projector={'found' if self.projector is not None else 'not found'}")
        return ", ".join(found)

    def __enter__(self) -> "AudioFeatureHooks":
        def capture(attr: str):
            def hook(_module, _inputs, output):
                setattr(self, attr, _first_tensor(output))
            return hook

        if self.tower is not None:
            self._handles.append(self.tower.register_forward_hook(capture("raw")))
        if self.projector is not None:
            self._handles.append(self.projector.register_forward_hook(capture("projected")))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def row(self, attr: str, index: int) -> Optional[torch.Tensor]:
        """Frames for one batch row, as ``[T, D]`` float32, or None."""
        tensor = getattr(self, attr)
        if tensor is None:
            return None
        if tensor.dim() == 2:            # already [T, D]: a single-example batch
            return tensor.float()
        if tensor.dim() < 3 or index >= tensor.shape[0]:
            return None
        return tensor[index].float()

    def clear(self) -> None:
        self.raw = None
        self.projected = None


# ---------------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------------

def _masked_layer_means(hidden_states: Sequence[torch.Tensor],
                        attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool every layer over non-padding positions -> ``[B, L+1, H]``.

    Padding is excluded explicitly rather than averaged over: with left
    padding a naive mean would be dominated by pad embeddings on short
    sequences, which would make the "mean pool" ablation rows measure padding
    rather than content.

    The sum is accumulated in float32 even for a float16 or bfloat16 model,
    because summing hundreds of positions at half precision loses enough
    accuracy to be visible in the pooled vector.
    """
    mask = attention_mask.unsqueeze(-1).float()
    lengths = mask.sum(dim=1).clamp(min=1.0)
    pooled = [(layer.float() * mask).sum(dim=1) / lengths for layer in hidden_states]
    return torch.stack(pooled, dim=1)


def build_cache(cfg: RunConfig, model, processor, examples: list[dict],
                missing_only: bool = True, capture_audio: bool = True) -> dict:
    """Run the clean-branch forward pass over ``examples`` and cache each one.

    Returns a small summary describing what was written, so the calling script
    can report whether the audio-encoder features are available for the feature
    ablation on this model.
    """
    adapter = get_adapter(cfg.model)
    sr = processor.feature_extractor.sampling_rate

    pending = [ex for ex in examples if not has_cached(cfg, example_key(ex))] if missing_only else list(examples)
    summary = {
        "requested": len(examples),
        "written": 0,
        "skipped_existing": len(examples) - len(pending),
        "audio_features": False,
    }
    if not pending:
        print(f"Cache already complete for {cfg.model}/{cfg.dataset} ({len(examples):,} examples)")
        return summary

    print(f"Caching {len(pending):,} of {len(examples):,} example(s) for "
          f"{cfg.model}/{cfg.dataset} -> {cfg.cache_dir}")

    hooks = AudioFeatureHooks(model) if capture_audio else None
    if capture_audio:
        if hooks.available:
            print(f"Audio-encoder hooks: {hooks.describe()}")
        else:
            print(
                "Audio-encoder hooks: no audio tower or projector module found on this "
                "model. Hidden-state features will still be cached; the raw/projected "
                "audio rows of the feature ablation will be unavailable."
            )
            hooks = None

    context = hooks if hooks is not None else _NullContext()
    with context:
        for offset in tqdm(range(0, len(pending), DEFAULT_BATCH_SIZE), unit="batch"):
            batch = pending[offset:offset + DEFAULT_BATCH_SIZE]
            items, audios = load_batch_audio(batch, sr)
            if not items:
                continue
            if hooks is not None:
                hooks.clear()

            inputs = adapter.encode(model, processor, items, audios, cfg.run_prompt)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, return_dict=True)

            # Left padding puts the last real token at position -1 for every
            # row, regardless of that row's length.
            last_token = torch.stack([h[:, -1, :].float() for h in out.hidden_states], dim=1)
            layer_means = _masked_layer_means(out.hidden_states, inputs["attention_mask"])
            embed_seq = out.hidden_states[0].float()

            for row, item in enumerate(items):
                tensors = {
                    "layers": last_token[row],
                    "layer_means": layer_means[row],
                    "embed_seq": embed_seq[row],
                }
                if hooks is not None:
                    raw = hooks.row("raw", row)
                    projected = hooks.row("projected", row)
                    if raw is not None:
                        tensors["raw_audio"] = raw
                    if projected is not None:
                        tensors["projected_audio"] = projected
                    summary["audio_features"] = summary["audio_features"] or (
                        raw is not None or projected is not None
                    )
                save_example_cache(cfg, example_key(item), tensors)
                summary["written"] += 1

    print(f"Cached {summary['written']:,} example(s)"
          + ("" if summary["audio_features"] else "; audio-encoder features unavailable"))
    return summary


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
