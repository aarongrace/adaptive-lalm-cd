#!/usr/bin/env python3
"""Build the oracle: per-example correctness across every decoded branch.

Run this after ``decoding/run_parallel.py`` has produced ``eval_*.json.gz``
files for every branch the oracle and selector should consider. The oracle is
both the paper's upper bound (Eq. 4, the accuracy an ideal per-example router
would reach) and the multi-hot target the selector is trained against, so it
has to be rebuilt whenever the set of decoded branches changes.

Two reports are printed alongside it:

  ``--rank``      per-branch accuracy, which is the raw material for the
                  paper's per-setting rankings (Table IV).
  ``--coverage``  the greedy branch-coverage curve behind the Section VI-A
                  observation that the library is highly redundant: coverage
                  keeps inching up until roughly half of it is included, yet is
                  effectively flat past N = 10. Selector accuracy falls over
                  that same range, which is what makes the bottleneck training
                  data rather than candidate diversity.

Usage:
    python -m scripts.build_oracle --model qwen2 --dataset ah_existence
    python -m scripts.build_oracle --model qwen2 --dataset ah_existence --rank --coverage
    python -m scripts.build_oracle --model qwen2 --dataset clotho_aqa   # all three partitions
"""

from __future__ import annotations

import argparse

from helpers.config import DATASETS, PAPER_ALPHA, PROMPTS, RunConfig
from perturbations import paper_label
from selector.oracle import (
    build_oracle,
    greedy_branch_coverage,
    rank_branches,
    save_oracle,
)

TOP_BRANCHES = 10


def _split_spec(label: str) -> tuple[str, str | None]:
    perturbation_type, _, setting = label.partition(":")
    return perturbation_type, setting or None


def print_branch_ranking(oracle: dict, top: int = TOP_BRANCHES) -> None:
    """Best and worst branches by accuracy, in the paper's table style."""
    ranked = rank_branches(oracle)
    if not ranked:
        return
    original = dict(ranked).get("ORIGINAL")
    print(f"\nBranch ranking ({len(ranked)} decoded branches)")
    if original is not None:
        print(f"  reference: ORIGINAL (no contrastive decoding) at {original:.2%}")

    def show(start: int, stop: int) -> None:
        for position in range(start, stop):
            label, accuracy = ranked[position]
            delta = f"  ({accuracy - original:+.2%} vs original)" if original is not None else ""
            print(f"  #{position + 1:<4} {paper_label(*_split_spec(label)):<38} {accuracy:.2%}{delta}")

    head = min(top, len(ranked))
    show(0, head)
    tail_start = max(head, len(ranked) - top)
    if tail_start > head:
        print(f"  ... {tail_start - head} branch(es) omitted ...")
    show(tail_start, len(ranked))


def print_coverage_curve(oracle: dict, max_n: int | None) -> None:
    """Greedy pool growth: how many branches are actually needed (Section VI-A)."""
    curve = greedy_branch_coverage(oracle, max_n=max_n)
    if not curve:
        return
    peak = curve[-1]["oracle_accuracy"]
    print(f"\nGreedy branch coverage (full-pool oracle {peak:.2%})")
    print(f"  {'N':>4}  {'branch added':<38} {'new':>6}  {'oracle':>8}  {'% of peak':>10}")
    saturated_at = None
    for row in curve:
        share = row["oracle_accuracy"] / peak if peak else 0.0
        if saturated_at is None and share >= 0.99:
            saturated_at = row["n"]
        # Print the head densely and then thin out: the interesting structure
        # is entirely in the first few steps.
        if row["n"] <= 12 or row["n"] % 10 == 0 or row is curve[-1]:
            print(f"  {row['n']:>4}  {paper_label(*_split_spec(row['branch'])):<38} "
                  f"{row['newly_covered']:>6}  {row['oracle_accuracy']:>8.2%}  {share:>9.1%}")
        if row["newly_covered"] == 0 and row["n"] > 1:
            print(f"  (branches beyond N={row['n'] - 1} add no new solvable examples)")
            break
    if saturated_at is not None:
        print(f"  99% of peak oracle coverage is reached at N = {saturated_at}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", "-m", required=True, choices=["qwen2", "af3"])
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASETS))
    parser.add_argument("--split", default=None, choices=["train", "val", "test"],
                        help="Clotho-AQA partition; omit to build all three")
    parser.add_argument("--alpha", "-a", type=float, default=PAPER_ALPHA)
    parser.add_argument("--prompt", default="constrained", choices=sorted(PROMPTS))
    parser.add_argument(
        "--exclude-original", action="store_true",
        help="Drop ORIGINAL from the candidate labels. Exploratory only: the paper's "
             "AF3 N=4 pool contains ORIGINAL, so excluding it makes that pool unbuildable.",
    )
    parser.add_argument("--rank", action="store_true", help="Print the per-branch accuracy ranking")
    parser.add_argument("--coverage", action="store_true",
                        help="Print the greedy branch-coverage curve (Section VI-A)")
    parser.add_argument("--max-coverage-n", type=int, default=None,
                        help="Stop the coverage curve at this pool size")
    args = parser.parse_args()

    if args.dataset == "clotho_aqa" and args.split is None:
        splits = ("train", "val", "test")
    else:
        splits = (args.split,)

    for split in splits:
        cfg = RunConfig(model=args.model, dataset=args.dataset, alpha=args.alpha,
                        split=split, prompt_name=args.prompt)
        print(f"\n=== {cfg.describe()} ===")
        oracle = build_oracle(cfg.results_dir, include_original=not args.exclude_original)
        save_oracle(oracle, cfg.oracle_file)
        if args.rank:
            print_branch_ranking(oracle)
        if args.coverage:
            print_coverage_curve(oracle, args.max_coverage_n)
        print(f"Saved oracle -> {cfg.oracle_file}")


if __name__ == "__main__":
    main()
