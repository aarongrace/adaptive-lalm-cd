#!/usr/bin/env python3
"""Cache the clean-branch forward-pass features the selector trains on.

One forward pass per example, over the unperturbed audio and question, using
the same encoding path the decoder uses. See ``selector/cache.py`` for exactly
which tensors are stored and why each one exists.

This is the only selector stage that needs a GPU, and it is the expensive one:
the run is the same size as a single decoding branch. It is also resumable --
``--missing-only`` skips examples that already have a cache file, so an
interrupted run costs only what was lost.

Whether the raw and projected audio-encoder features can be captured depends
on the model exposing an audio tower and projector this script can hook. When
they are found, the full Table VI feature ablation is runnable; when they are
not, the hidden-state features are still cached and the script says so plainly
rather than leaving a silent gap.

Usage:
    python -m scripts.cache_hidden_states --model qwen2 --dataset ah_existence
    python -m scripts.cache_hidden_states --model af3 --dataset ah_existence --missing-only
    python -m scripts.cache_hidden_states --model qwen2 --dataset clotho_aqa   # all partitions
"""

from __future__ import annotations

import argparse

from decoding.engine import get_adapter
from helpers.config import DATASETS, PAPER_ALPHA, PROMPTS, RunConfig
from helpers.datasets import dataset_summary, load_examples, print_dataset_summary
from helpers.runtime import require_cuda
from selector.cache import build_cache
from selector.data import AUDIO_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", "-m", required=True, choices=["qwen2", "af3"])
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASETS))
    parser.add_argument("--split", default=None, choices=["train", "val", "test"],
                        help="Clotho-AQA partition; omit to cache all three")
    parser.add_argument("--alpha", "-a", type=float, default=PAPER_ALPHA,
                        help="Recorded for consistency; the cache itself is alpha-independent")
    parser.add_argument("--prompt", default="constrained", choices=sorted(PROMPTS),
                        help="Must match the prompt the selector's decoding runs used")
    parser.add_argument("--missing-only", action="store_true",
                        help="Skip examples that already have a cache file")
    parser.add_argument("--no-audio-features", action="store_true",
                        help="Skip the audio-encoder hooks; caches only hidden-state features")
    args = parser.parse_args()

    if args.dataset == "clotho_aqa" and args.split is None:
        splits = ("train", "val", "test")
    else:
        splits = (args.split,)

    require_cuda()
    adapter = get_adapter(args.model)

    # All partitions of a dataset share one model, so it is loaded once and
    # reused; only the per-split cache directory changes.
    configs = [
        RunConfig(model=args.model, dataset=args.dataset, alpha=args.alpha,
                  split=split, prompt_name=args.prompt)
        for split in splits
    ]
    print(f"Loading {adapter.label} ...")
    model, processor = adapter.load(configs[0])

    captured_audio = False
    for cfg in configs:
        print(f"\n=== {cfg.describe()} ===")
        examples = load_examples(cfg)
        print_dataset_summary(dataset_summary(cfg, examples))
        summary = build_cache(
            cfg, model, processor, examples,
            missing_only=args.missing_only,
            capture_audio=not args.no_audio_features,
        )
        captured_audio = captured_audio or summary["audio_features"]

    print(f"\nCache written to {configs[0].cache_dir.parent}")
    if captured_audio:
        print("Audio-encoder features captured; the full feature ablation is available.")
    else:
        print(
            "Audio-encoder features were not captured, so these feature configurations "
            f"cannot be trained: {', '.join(sorted(AUDIO_FEATURES))}. "
            "The reported hidden-state selector ('hidden') is unaffected."
        )


if __name__ == "__main__":
    main()
