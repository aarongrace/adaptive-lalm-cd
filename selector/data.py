"""
Turns cached forward-pass tensors (``selector/cache.py``) and oracle labels
(``selector/oracle.py``) into the tensors ``selector/train.py`` trains on.

The feature registry below is the paper's input-feature ablation (Table VI)
made runnable. Its central finding is the contrast between the first two
groups: mean-pooled LLM states plateau at 72.3-72.7%, effectively the fixed
no-audio baseline, while the **last-token** state jumps to 76.3%. In a causal
decoder the final non-padding position is the only one to have attended the
complete input -- system prompt, audio frames, and question alike -- so mean
pooling dilutes it by averaging in earlier positions that saw only part of the
context. Concatenating the first, middle, and final layers' last-token states
adds a further 0.4 points to 76.7%, because earlier layers retain information
the final layer has already discarded.

The third finding is negative and equally load-bearing: separately supplied
audio-encoder features add nothing. Raw or projected audio alone matches
mean-pooled LLM states, appending them to any LLM feature yields no reliable
gain, and cross-attending over them essentially matches the last-token feature
by itself. By the time the classifier reads the last-token state, cross-modal
attention has already folded the audio signal into it during the forward pass,
so re-injecting it externally is redundant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from helpers.config import RunConfig
from selector.cache import load_example_cache
from selector.oracle import load_oracle

# ---------------------------------------------------------------------------
# Per-example feature builders
#
# Each takes one cached example dict and returns a tuple of tensors. Vector
# features return a single pooled vector; cross-attention features return
# (query vector, key/value frames), which build_dataset then pads into a batch.
# ---------------------------------------------------------------------------


def _first_mid_last(layers: torch.Tensor) -> torch.Tensor:
    """Concatenate the first, middle, and final layer of a ``[L+1, H]`` stack."""
    num_layers = layers.shape[0]
    return torch.cat([layers[0], layers[num_layers // 2], layers[num_layers - 1]], dim=-1)


def _hidden_fml(cached: dict) -> tuple[torch.Tensor, ...]:
    return (_first_mid_last(cached["layers"]),)


def _last_token(cached: dict) -> tuple[torch.Tensor, ...]:
    return (cached["layers"][-1],)


def _last_token_all_layer_mean(cached: dict) -> tuple[torch.Tensor, ...]:
    layers = cached["layers"]
    return (torch.cat([layers[-1], layers.mean(dim=0)], dim=-1),)


def _mean_pool_last(cached: dict) -> tuple[torch.Tensor, ...]:
    return (cached["layer_means"][-1],)


def _mean_pool_all(cached: dict) -> tuple[torch.Tensor, ...]:
    # Averaged across layers rather than concatenated: concatenating 30-plus
    # layers would make the input dimension dwarf the ~7,500 training examples
    # and turn the ablation into a test of regularisation instead of features.
    return (cached["layer_means"].mean(dim=0),)


def _audio_mean(field: str) -> Callable[[dict], tuple[torch.Tensor, ...]]:
    def build(cached: dict) -> tuple[torch.Tensor, ...]:
        return (cached[field].mean(dim=0),)
    return build


def _concat(*builders: Callable[[dict], tuple[torch.Tensor, ...]]) -> Callable[[dict], tuple[torch.Tensor, ...]]:
    def build(cached: dict) -> tuple[torch.Tensor, ...]:
        parts = [tensor for builder in builders for tensor in builder(cached)]
        return (torch.cat(parts, dim=-1),)
    return build


def _cross_attention(query: Callable[[dict], tuple[torch.Tensor, ...]],
                     frames_field: str) -> Callable[[dict], tuple[torch.Tensor, ...]]:
    def build(cached: dict) -> tuple[torch.Tensor, ...]:
        return (query(cached)[0], cached[frames_field])
    return build


@dataclass(frozen=True)
class FeatureSpec:
    """One row of the paper's input-feature ablation."""

    name: str
    description: str
    requires: tuple[str, ...]                       # cache fields this feature reads
    build: Callable[[dict], tuple[torch.Tensor, ...]]
    kind: str = "vector"                            # "vector" or "cross_attention"


FEATURES: dict[str, FeatureSpec] = {
    # --- LLM hidden states only -------------------------------------------
    "hidden": FeatureSpec(
        "hidden",
        "Last token, first/middle/final layers concatenated (paper's reported best)",
        ("layers",), _hidden_fml,
    ),
    "last_token": FeatureSpec(
        "last_token",
        "Last token, final layer only",
        ("layers",), _last_token,
    ),
    "last_token_all_layer_mean": FeatureSpec(
        "last_token_all_layer_mean",
        "Last token final layer, concatenated with its mean across all layers",
        ("layers",), _last_token_all_layer_mean,
    ),
    "mean_pool_last": FeatureSpec(
        "mean_pool_last",
        "Final layer mean-pooled over all non-padding positions",
        ("layer_means",), _mean_pool_last,
    ),
    "mean_pool_all": FeatureSpec(
        "mean_pool_all",
        "Token-mean-pooled states averaged across all layers",
        ("layer_means",), _mean_pool_all,
    ),
    # --- Audio-encoder features only ---------------------------------------
    "raw_audio_mean": FeatureSpec(
        "raw_audio_mean",
        "Audio encoder output, mean-pooled over frames (before projection)",
        ("raw_audio",), _audio_mean("raw_audio"),
    ),
    "projected_audio_mean": FeatureSpec(
        "projected_audio_mean",
        "Projected audio frames, mean-pooled (after projection into LM space)",
        ("projected_audio",), _audio_mean("projected_audio"),
    ),
    # --- LLM hidden states concatenated with audio -------------------------
    "last_token_raw_audio": FeatureSpec(
        "last_token_raw_audio",
        "Last token, final layer, concatenated with mean raw audio",
        ("layers", "raw_audio"), _concat(_last_token, _audio_mean("raw_audio")),
    ),
    "last_token_projected_audio": FeatureSpec(
        "last_token_projected_audio",
        "Last token, final layer, concatenated with mean projected audio",
        ("layers", "projected_audio"), _concat(_last_token, _audio_mean("projected_audio")),
    ),
    "hidden_projected_audio": FeatureSpec(
        "hidden_projected_audio",
        "First/middle/final last-token concat, plus mean projected audio",
        ("layers", "projected_audio"), _concat(_hidden_fml, _audio_mean("projected_audio")),
    ),
    # --- Cross-attention over audio frames ---------------------------------
    "cross_attention": FeatureSpec(
        "cross_attention",
        "Last-token query cross-attending over projected audio frames",
        ("layers", "projected_audio"),
        _cross_attention(_last_token, "projected_audio"),
        kind="cross_attention",
    ),
    "mean_pool_cross_attention": FeatureSpec(
        "mean_pool_cross_attention",
        "Mean-pooled final-layer query cross-attending over projected audio frames",
        ("layer_means", "projected_audio"),
        _cross_attention(_mean_pool_last, "projected_audio"),
        kind="cross_attention",
    ),
    # --- Exploratory --------------------------------------------------------
    "embedding": FeatureSpec(
        "embedding",
        "Mean of the LM's combined input-embedding sequence (audio and text mixed)",
        ("embed_seq",), lambda cached: (cached["embed_seq"].mean(dim=0),),
    ),
}

DEFAULT_FEATURE = "hidden"

# Features that need the hooked audio-encoder tensors, which are only present
# when scripts/cache_hidden_states.py located the model's audio modules.
AUDIO_FEATURES = tuple(
    name for name, spec in FEATURES.items()
    if any(field in ("raw_audio", "projected_audio") for field in spec.requires)
)


def feature_spec(name: str) -> FeatureSpec:
    try:
        return FEATURES[name]
    except KeyError:
        raise ValueError(
            f"Unknown selector feature {name!r}. Available: {', '.join(sorted(FEATURES))}"
        ) from None


# ---------------------------------------------------------------------------
# Candidate-pool construction
# ---------------------------------------------------------------------------

def rank_specs_by_success(examples: dict, candidate_pool: Optional[Sequence[str]] = None) -> list[str]:
    """Rank candidate branches by global success rate, retaining zero-hit branches.

    Supplying the oracle's ``branch_specs`` as the pool matters: a branch that
    happens to solve nothing must stay visible in a protocol manifest rather
    than silently dropping out of the candidate universe and shifting what
    "top-N" means. Ties are broken by name so the pool is reproducible.
    """
    counts: dict[str, int] = {spec: 0 for spec in (candidate_pool or [])}
    for row in examples.values():
        for spec in row["worked_perturbations"]:
            if candidate_pool is None or spec in candidate_pool:
                counts[spec] = counts.get(spec, 0) + 1
    return sorted(counts, key=lambda spec: (-counts[spec], spec))


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def _pad_frames(frames: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length frame sequences into ``[N, T, D]`` plus a mask."""
    max_len = max(f.shape[0] for f in frames)
    dim = frames[0].shape[1]
    padded = torch.zeros(len(frames), max_len, dim)
    mask = torch.zeros(len(frames), max_len, dtype=torch.bool)
    for row, frame in enumerate(frames):
        length = frame.shape[0]
        padded[row, :length] = frame
        mask[row, :length] = True
    return padded, mask


@dataclass
class SelectorDataset:
    """Model inputs, multi-hot targets, and the example keys they came from.

    ``labels`` is always dense over ``branch_specs`` -- every branch the oracle
    knows about, not just the ones currently being routed among. That makes
    :meth:`restrict` a column selection rather than a rebuild, so sweeping the
    candidate-pool size N (Table V) reads the cache from disk once instead of
    once per pool size.
    """

    inputs: tuple[torch.Tensor, ...]
    labels: torch.Tensor
    keys: list[str]
    feature: str
    branch_specs: list[str]

    @property
    def kind(self) -> str:
        return FEATURES[self.feature].kind

    @property
    def input_dims(self) -> tuple[int, ...]:
        return tuple(t.shape[-1] for t in self.inputs if t.dtype != torch.bool)

    def __len__(self) -> int:
        return len(self.keys)

    def restrict(self, candidate_specs: Sequence[str]) -> "SelectorDataset":
        """Narrow the label columns to one candidate pool, keeping features intact."""
        index = {label: i for i, label in enumerate(self.branch_specs)}
        missing = [label for label in candidate_specs if label not in index]
        if missing:
            raise ValueError(f"Candidate branch(es) absent from the oracle: {missing}")
        columns = torch.as_tensor([index[label] for label in candidate_specs], dtype=torch.long)
        return SelectorDataset(
            inputs=self.inputs,
            labels=self.labels[:, columns],
            keys=self.keys,
            feature=self.feature,
            branch_specs=list(candidate_specs),
        )

    def subset(self, rows: Sequence[int]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        index = torch.as_tensor(list(rows), dtype=torch.long)
        return tuple(tensor[index] for tensor in self.inputs), self.labels[index]


def build_dataset(cfg: RunConfig, feature: str,
                  candidate_specs: Optional[Sequence[str]] = None) -> SelectorDataset:
    """Assemble features and multi-hot labels for every cached oracle example.

    With ``candidate_specs`` omitted the labels span every branch in the
    oracle, which is the form to build once and then :meth:`restrict` per pool.

    Examples without a cache entry are skipped rather than failing the run, so
    a partially built cache still trains; the count is reported so a badly
    incomplete cache is visible rather than silently halving the training set.
    """
    spec = feature_spec(feature)
    oracle = load_oracle(cfg.oracle_file)
    examples = oracle["examples"]
    all_specs = list(oracle.get("branch_specs") or sorted(
        {branch for row in examples.values() for branch in row["worked_perturbations"]}
    ))
    spec_index = {label: i for i, label in enumerate(all_specs)}

    collected: list[tuple[torch.Tensor, ...]] = []
    labels: list[torch.Tensor] = []
    keys: list[str] = []
    missing_cache = 0
    missing_fields: set[str] = set()

    for key, row in examples.items():
        try:
            cached = load_example_cache(cfg, key)
        except FileNotFoundError:
            missing_cache += 1
            continue
        absent = [field for field in spec.requires if field not in cached]
        if absent:
            missing_fields.update(absent)
            continue

        collected.append(spec.build(cached))
        target = torch.zeros(len(all_specs))
        for branch in row["worked_perturbations"]:
            if branch in spec_index:
                target[spec_index[branch]] = 1.0
        labels.append(target)
        keys.append(key)

    if missing_fields:
        raise RuntimeError(
            f"Feature {feature!r} needs cached field(s) {sorted(missing_fields)}, which are "
            f"absent for {cfg.model}/{cfg.dataset}. Re-run scripts.cache_hidden_states; if the "
            "audio-encoder modules could not be hooked on this model, the audio-feature rows "
            "of the ablation are unavailable and only the hidden-state features can be run."
        )
    if not collected:
        raise RuntimeError(
            f"No cached examples found for {cfg.model}/{cfg.dataset}. "
            "Run `python -m scripts.cache_hidden_states` first."
        )
    if missing_cache:
        print(f"  note: {missing_cache:,} oracle example(s) had no cache entry and were skipped")

    if spec.kind == "cross_attention":
        queries = torch.stack([item[0] for item in collected])
        frames, mask = _pad_frames([item[1] for item in collected])
        inputs: tuple[torch.Tensor, ...] = (queries, frames, mask)
    else:
        inputs = (torch.stack([item[0] for item in collected]),)

    dataset = SelectorDataset(
        inputs=inputs,
        labels=torch.stack(labels),
        keys=keys,
        feature=feature,
        branch_specs=all_specs,
    )
    return dataset if candidate_specs is None else dataset.restrict(candidate_specs)


def split_dataset(dataset: SelectorDataset, split: dict) -> dict:
    """Partition a dataset by a split file's example keys.

    ``split`` is the payload from ``helpers.datasets.load_split``. Keys absent
    from the cache are dropped, so a split built against the full manifest
    still applies to a partially cached run.
    """
    key_to_row = {key: row for row, key in enumerate(dataset.keys)}
    parts = {}
    for partition in ("train", "val", "test"):
        rows = [key_to_row[key] for key in split[partition] if key in key_to_row]
        parts[partition] = dataset.subset(rows)
    return parts
