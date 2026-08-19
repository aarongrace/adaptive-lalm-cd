"""
The parts of the selector protocol that are stated in the paper as specific
values rather than derived at runtime.

Two kinds of constant live here.

**Candidate pools.** The paper names its N=4 AH Existence pools in physical
parameters ("Noise (sigma = 1.0)", "High pass (6 kHz)"). Their registry labels
are pinned below so that a later tie in aggregate ranking, or a rerun over a
different subset of decoded branches, cannot silently substitute a different
pool for the reported experiment. ``scripts/train_selector.py --paper-candidates``
uses exactly these.

**Reported reference values.** The oracle/selector/gap grid (Table V) and the
input-feature ablation (Table VI) are recorded so a reproduction can print its
own number beside the published one. They are documentation, never a fallback:
nothing here is substituted for a measurement, and a reproduction that cannot
compute a value reports that it could not, rather than echoing the paper.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Candidate pools (paper Table V footnote)
# ---------------------------------------------------------------------------

PAPER_CANDIDATE_SETS: dict[tuple[str, str, int], list[str]] = {
    ("qwen2", "ah_existence", 4): [
        "NO_AUDIO",
        "NOISE:overwhelming",          # sigma = 1.0
        "HIGH_PASS:ultra_extreme",     # 6 kHz
        "SPECTRAL_BLUR:heavy",         # sigma_time = sigma_freq = 15
    ],
    ("af3", "ah_existence", 4): [
        "GATE_INVERTED:very_extreme",  # threshold = 0.10
        "GATE:ultra_extreme",          # threshold = 0.75
        "ORIGINAL",                    # routing to "no correction" is a valid choice
        "PITCH_SHIFT:extreme_up",      # +24 semitones
    ],
}


def paper_candidate_set(model: str, dataset: str, n: int) -> list[str]:
    """Return a paper-listed candidate pool, or explain which inputs are known."""
    key = (model, dataset, n)
    try:
        return list(PAPER_CANDIDATE_SETS[key])
    except KeyError as exc:
        available = ", ".join(
            f"{m}/{d}/N={size}" for m, d, size in sorted(PAPER_CANDIDATE_SETS)
        )
        raise ValueError(
            f"No explicitly listed paper candidate pool for {model}/{dataset}/N={n}. "
            f"Available: {available}. Use the default aggregate ranking or "
            "--candidate-spec for a separately documented experiment."
        ) from exc


# ---------------------------------------------------------------------------
# Reported results (paper Table V): accuracy (%) by candidate-pool size N on
# AH Existence at alpha = 1.0, averaged over the five balanced splits.
#
# N = 0 is the unmodified model with no contrastive decoding at all; N = 1 is
# the single best fixed branch, where selector and oracle necessarily coincide.
# Oracle rises with N as new transformations expose distinct solvable examples,
# while selector accuracy peaks at N = 4 and then declines: larger pools
# dilute a fixed ~7,500-example training signal across more classes.
# ---------------------------------------------------------------------------

PAPER_N_SWEEP: dict[str, dict[int, dict[str, Optional[float]]]] = {
    "qwen2": {
        0:  {"oracle": 67.8, "selector": None},
        1:  {"oracle": 72.4, "selector": 72.4},
        3:  {"oracle": 82.9, "selector": 76.4},
        4:  {"oracle": 83.5, "selector": 76.7},
        6:  {"oracle": 84.4, "selector": 76.5},
        10: {"oracle": 85.2, "selector": 76.1},
        20: {"oracle": 85.9, "selector": 75.8},
        30: {"oracle": 86.1, "selector": 75.7},
        60: {"oracle": 86.4, "selector": 75.3},
    },
    "af3": {
        0:  {"oracle": 69.5, "selector": None},
        1:  {"oracle": 73.9, "selector": 73.9},
        3:  {"oracle": 80.4, "selector": 76.1},
        4:  {"oracle": 81.3, "selector": 76.4},
        6:  {"oracle": 82.2, "selector": 76.4},
        10: {"oracle": 83.2, "selector": 76.2},
        20: {"oracle": 84.2, "selector": 76.2},
        30: {"oracle": 84.6, "selector": 76.0},
        60: {"oracle": 85.0, "selector": 75.9},
    },
}

# Reported input-feature ablation (paper Table VI): Qwen2, AH Existence, N = 4,
# alpha = 1.0, averaged over the five balanced splits, 3-layer MLP head. Keys
# are the feature names accepted by selector/data.py.
PAPER_FEATURE_ABLATION: dict[str, float] = {
    "mean_pool_last": 72.3,
    "mean_pool_all": 72.7,
    "last_token": 76.3,
    "last_token_all_layer_mean": 76.5,
    "hidden": 76.7,              # last token, first/middle/last layers concatenated
    "projected_audio_mean": 72.5,
    "raw_audio_mean": 72.6,
    "last_token_projected_audio": 76.2,
    "last_token_raw_audio": 76.2,
    "hidden_projected_audio": 76.6,
    "cross_attention": 76.4,
    "mean_pool_cross_attention": 72.4,
}

# Baselines the ablation is read against, on the same setting.
PAPER_AH_EXISTENCE_BASELINES = {
    "qwen2": {"original": 67.8, "best_fixed_branch": 72.4, "oracle": 86.2},
    "af3": {"original": 69.5, "best_fixed_branch": 73.9, "oracle": 85.0},
}


def paper_n_sweep_value(model: str, n: int, key: str) -> Optional[float]:
    """Reported AH Existence accuracy at pool size N, or None if not tabulated."""
    return PAPER_N_SWEEP.get(model, {}).get(n, {}).get(key)


def paper_feature_value(feature: str) -> Optional[float]:
    """Reported Table VI accuracy for a feature configuration, if listed."""
    return PAPER_FEATURE_ABLATION.get(feature)


def format_against_paper(measured: float, reported: Optional[float]) -> str:
    """Render a measured percentage next to its published counterpart.

    ``measured`` is a fraction in [0, 1]; ``reported`` is a percentage as
    printed in the paper. Returns just the measured value when there is no
    published number to compare against, so absence never looks like a match.
    """
    if reported is None:
        return f"{measured:.2%}"
    delta = measured * 100.0 - reported
    return f"{measured:.2%} (paper {reported:.1f}%, {delta:+.1f} pts)"
