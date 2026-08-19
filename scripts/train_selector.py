#!/usr/bin/env python3
"""Train the adaptive selector under the paper's partition protocol.

AH results are reported as the arithmetic mean across five balanced,
composition-aware 70/15/15 splits, and this script trains all five by default.
A single ``--split-name`` run is useful while debugging but is not a paper
result and is labelled as such in its output. Clotho-AQA instead uses its
official train/validation/test files and is trained once on that partition.

Three modes:

  default             one candidate-pool size, one feature configuration.
  ``--sweep-n``       the oracle/selector/gap grid over pool sizes (Table V).
                      Oracle rises with N as new branches expose distinct
                      solvable examples, while selector accuracy peaks at N = 4
                      and then declines: a fixed ~7,500-example training signal
                      spread across more classes. The widening gap is the
                      paper's evidence that the method is data-limited rather
                      than candidate-limited.
  ``--feature-ablation``  every input-feature configuration (Table VI), which
                      is where the last-token-versus-mean-pooling result comes
                      from. Configurations needing audio-encoder features are
                      skipped with a note when the cache does not have them.

Where the paper reports a number for what is being run, it is printed beside
the measured value. That is a reading aid, never a substitute: nothing is ever
filled in from the table.

Usage:
    python -m scripts.train_selector --model qwen2 --dataset ah_existence --n 4 \
        --paper-candidates --save
    python -m scripts.train_selector --model af3 --dataset ah_existence --n 4 \
        --paper-candidates --save
    python -m scripts.train_selector --model qwen2 --dataset ah_existence --sweep-n
    python -m scripts.train_selector --model qwen2 --dataset ah_existence --feature-ablation
    python -m scripts.train_selector --model qwen2 --dataset clotho_aqa --n 4 --save
"""

from __future__ import annotations

import argparse
import json
from statistics import mean, pstdev
from typing import Optional, Sequence

import torch

from helpers.config import CHECKPOINTS_DIR, DATASETS, N_PAPER_SPLITS, PAPER_ALPHA, PROMPTS, RunConfig
from helpers.datasets import list_splits, load_split
from helpers.runtime import set_random_seed
from selector.data import (
    AUDIO_FEATURES,
    DEFAULT_FEATURE,
    FEATURES,
    SelectorDataset,
    build_dataset,
    rank_specs_by_success,
    split_dataset,
)
from selector.model import HEAD_VARIANTS, SELECTOR_SPEC, build_selector
from selector.oracle import branch_accuracies, load_oracle, oracle_accuracy
from selector.protocol import (
    format_against_paper,
    paper_candidate_set,
    paper_feature_value,
    paper_n_sweep_value,
)
from selector.train import TrainConfig, evaluate, train_selector

PAPER_N_VALUES = (1, 3, 4, 6, 10, 20, 30, 60)


# ---------------------------------------------------------------------------
# Candidate pools
# ---------------------------------------------------------------------------

def available_branches(oracles: Sequence[dict]) -> set[str]:
    """Branches decoded in every oracle involved in this run.

    Clotho-AQA trains across three separately decoded partitions, so a branch
    is only usable if all three have it; intersecting avoids a candidate that
    exists in train but not test.
    """
    available = set(oracles[0].get("branch_specs", []))
    if not available:
        available = {spec for row in oracles[0]["examples"].values() for spec in row["worked_perturbations"]}
    for oracle in oracles[1:]:
        branch_specs = set(oracle.get("branch_specs", []))
        if branch_specs:
            available &= branch_specs
    return available


def choose_candidates(args: argparse.Namespace, oracles: Sequence[dict], n: int) -> list[str]:
    """Resolve the candidate pool for one pool size, in priority order."""
    available = available_branches(oracles)

    if args.paper_candidates:
        candidates = paper_candidate_set(args.model, args.dataset, n)
    elif args.candidate_spec:
        candidates = list(args.candidate_spec)
    else:
        # Rank by aggregate success across every oracle involved. Keys are
        # namespaced per oracle so identical example keys in different
        # partitions are counted separately rather than colliding.
        combined: dict[str, dict] = {}
        for number, oracle in enumerate(oracles):
            combined.update({f"{number}:{key}": row for key, row in oracle["examples"].items()})
        candidates = rank_specs_by_success(combined, sorted(available))[:n]

    if len(candidates) != n:
        raise ValueError(f"Candidate selection produced {len(candidates)} branches for N={n}: {candidates}")
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"Candidate pool contains duplicates: {candidates}")
    missing = sorted(set(candidates) - available)
    if missing:
        raise ValueError(
            f"Candidate branch(es) missing from the oracle: {missing}. "
            "Rebuild the oracle after decoding every requested branch."
        )
    return candidates


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_once(parts: dict, num_classes: int, kind: str, input_dims: tuple[int, ...],
                cfg: TrainConfig, spec, seed: int) -> tuple[torch.nn.Module, dict]:
    empty = [part for part, (inputs, _) in parts.items() if inputs[0].shape[0] == 0]
    if empty:
        raise ValueError(
            f"No cached examples in {', '.join(empty)}; rebuild the cache and splits "
            "from the same manifest."
        )
    set_random_seed(seed)
    model = build_selector(kind, input_dims, num_classes, spec)
    model, summary = train_selector(model, parts["train"], parts["val"], cfg=cfg)
    return model, {
        "best_epoch": summary["best_epoch"],
        "epochs_run": summary["epochs_run"],
        "best_val_loss": summary["best_val_loss"],
        "val_top1_accuracy": evaluate(model, parts["val"]),
        "test_top1_accuracy": evaluate(model, parts["test"]),
    }


def train_across_splits(dataset: SelectorDataset, cfg: RunConfig, candidates: Sequence[str],
                        train_cfg: TrainConfig, spec, split_names: Sequence[str],
                        save_fn=None, quiet: bool = False) -> dict:
    """Train one configuration on each split and average the test scores."""
    restricted = dataset.restrict(candidates)
    records = []
    for index, name in enumerate(split_names):
        parts = split_dataset(restricted, load_split(cfg.splits_dir, name))
        model, metrics = _train_once(
            parts, len(candidates), restricted.kind, restricted.input_dims,
            train_cfg, spec, seed=42 + index,
        )
        records.append({"split": name, **metrics})
        if not quiet:
            print(f"  {name}: best epoch {metrics['best_epoch']:>3}  "
                  f"val {metrics['val_top1_accuracy']:.2%}  test {metrics['test_top1_accuracy']:.2%}")
        if save_fn is not None:
            save_fn(model, name, records[-1])

    scores = [record["test_top1_accuracy"] for record in records]
    return {
        "splits": records,
        "mean_test_top1_accuracy": mean(scores),
        "population_std_test_top1_accuracy": pstdev(scores) if len(scores) > 1 else 0.0,
    }


def resolve_split_names(args: argparse.Namespace, cfg: RunConfig) -> list[str]:
    names = [args.split_name] if args.split_name else list_splits(cfg.splits_dir)
    if not names:
        raise FileNotFoundError(
            f"No balanced split files in {cfg.splits_dir}. Run `python -m scripts.prepare_splits` first."
        )
    if args.split_name is None and len(names) != N_PAPER_SPLITS:
        raise ValueError(
            f"Paper aggregation requires exactly {N_PAPER_SPLITS} balanced splits, "
            f"found {len(names)}: {names}. Use --split-name only for an explicit debug run."
        )
    return names


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_single(args, cfg, dataset, oracles, train_cfg, spec) -> dict:
    candidates = choose_candidates(args, oracles, args.n)
    names = resolve_split_names(args, cfg)
    print(f"Candidate pool (N={args.n}): {', '.join(candidates)}")
    print(f"Feature: {args.feature} ({FEATURES[args.feature].description})")

    def save_fn(model, name, record):
        if args.save:
            _save_checkpoint(model, args, candidates, name, {"candidate_specs": candidates, **record})

    result = train_across_splits(dataset, cfg, candidates, train_cfg, spec, names, save_fn=save_fn)
    oracle_at_n = oracle_accuracy(oracles[0], candidates)
    measured = result["mean_test_top1_accuracy"]
    reported = paper_n_sweep_value(args.model, args.n, "selector") if args.dataset == "ah_existence" else None

    print(f"\nSelector test accuracy: {format_against_paper(measured, reported)}"
          f"  (std {result['population_std_test_top1_accuracy']:.2%} across {len(names)} split(s))")
    print(f"Oracle at N={args.n}: {oracle_at_n:.2%}   remaining headroom {oracle_at_n - measured:+.2%}")
    return {
        "mode": "single",
        "n": args.n,
        "candidate_specs": candidates,
        "oracle_accuracy": oracle_at_n,
        "aggregation": "mean_across_five_balanced_splits" if args.split_name is None else "single_debug_split",
        **result,
    }


def run_sweep_n(args, cfg, dataset, oracles, train_cfg, spec) -> dict:
    """Reproduce Table V: oracle, selector, and gap by candidate-pool size."""
    names = resolve_split_names(args, cfg)
    values = args.sweep_n or list(PAPER_N_VALUES)
    available = available_branches(oracles)
    # N=0 in Table V is the unmodified model: no contrastive correction at all,
    # which is exactly the accuracy of the ORIGINAL branch.
    baseline = branch_accuracies(oracles[0]).get("ORIGINAL")

    print(f"Candidate-pool sweep over N = {values} ({len(names)} split(s) each)")
    print(f"\n{'N':>4}  {'oracle':>18}  {'selector':>18}  {'gap':>7}")
    rows = []
    for n in values:
        if n > len(available):
            print(f"{n:>4}  only {len(available)} branches decoded; skipped")
            continue
        sweep_args = argparse.Namespace(**vars(args))
        # The paper pins the N=4 pool explicitly; every other size uses the
        # top-N aggregate ranking, so those flags apply only at their own size.
        sweep_args.paper_candidates = args.paper_candidates and n == 4
        sweep_args.candidate_spec = (
            args.candidate_spec if args.candidate_spec and n == len(args.candidate_spec) else None
        )
        candidates = choose_candidates(sweep_args, oracles, n)

        result = train_across_splits(dataset, cfg, candidates, train_cfg, spec, names, quiet=True)
        selector = result["mean_test_top1_accuracy"]
        oracle_at_n = oracle_accuracy(oracles[0], candidates)
        paper_oracle = paper_n_sweep_value(args.model, n, "oracle") if args.dataset == "ah_existence" else None
        paper_selector = paper_n_sweep_value(args.model, n, "selector") if args.dataset == "ah_existence" else None

        print(f"{n:>4}  {format_against_paper(oracle_at_n, paper_oracle):>18}  "
              f"{format_against_paper(selector, paper_selector):>18}  "
              f"{oracle_at_n - selector:>6.2%}")
        rows.append({
            "n": n,
            "candidate_specs": candidates,
            "oracle_accuracy": oracle_at_n,
            "selector_accuracy": selector,
            "gap": oracle_at_n - selector,
            "population_std": result["population_std_test_top1_accuracy"],
            "splits": result["splits"],
        })

    if rows:
        best = max(rows, key=lambda row: row["selector_accuracy"])
        if baseline is not None:
            print(f"\nNo contrastive decoding (N=0): {baseline:.2%}")
            print(f"Best selector at N={best['n']}: {best['selector_accuracy']:.2%} "
                  f"({best['selector_accuracy'] - baseline:+.2%} over the unmodified model)")
        else:
            print(f"\nBest selector at N={best['n']}: {best['selector_accuracy']:.2%} "
                  "(ORIGINAL was not decoded, so the N=0 reference is unavailable)")
    return {"mode": "sweep_n", "n_values": values, "baseline_accuracy": baseline, "rows": rows}


def run_feature_ablation(args, cfg, oracles, train_cfg, spec) -> dict:
    """Reproduce Table VI: routing accuracy by input-feature configuration."""
    candidates = choose_candidates(args, oracles, args.n)
    names = resolve_split_names(args, cfg)
    features = args.features or list(FEATURES)
    print(f"Feature ablation at N={args.n}: {', '.join(candidates)}")
    print(f"\n{'feature':<28}  {'accuracy':>26}  description")

    rows = []
    for feature in features:
        try:
            dataset = build_dataset(cfg, feature)
        except RuntimeError as exc:
            reason = "audio-encoder features not cached" if feature in AUDIO_FEATURES else str(exc)
            print(f"{feature:<28}  {'skipped':>26}  {reason}")
            rows.append({"feature": feature, "skipped": True, "reason": reason})
            continue

        result = train_across_splits(dataset, cfg, candidates, train_cfg, spec, names, quiet=True)
        measured = result["mean_test_top1_accuracy"]
        reported = paper_feature_value(feature) if args.dataset == "ah_existence" and args.model == "qwen2" else None
        print(f"{feature:<28}  {format_against_paper(measured, reported):>26}  "
              f"{FEATURES[feature].description}")
        rows.append({
            "feature": feature,
            "accuracy": measured,
            "population_std": result["population_std_test_top1_accuracy"],
            "splits": result["splits"],
        })

    scored = [row for row in rows if not row.get("skipped")]
    if scored:
        best = max(scored, key=lambda row: row["accuracy"])
        print(f"\nBest feature: {best['feature']} at {best['accuracy']:.2%}")
    return {"mode": "feature_ablation", "n": args.n, "candidate_specs": candidates, "rows": rows}


def run_clotho(args, oracles, train_cfg, spec) -> dict:
    """Clotho-AQA uses its official partition rather than generated splits."""
    candidates = choose_candidates(args, oracles, args.n)
    print(f"Candidate pool (N={args.n}): {', '.join(candidates)}")
    parts = {}
    dataset: Optional[SelectorDataset] = None
    for partition in ("train", "val", "test"):
        cfg = RunConfig(model=args.model, dataset=args.dataset, alpha=args.alpha,
                        split=partition, prompt_name=args.prompt)
        dataset = build_dataset(cfg, args.feature, candidates)
        parts[partition] = dataset.subset(range(len(dataset)))
        print(f"  {partition}: {len(dataset):,} cached example(s)")
    assert dataset is not None
    model, metrics = _train_once(
        parts, len(candidates), dataset.kind, dataset.input_dims, train_cfg, spec, seed=42
    )
    print(f"Clotho selector: best epoch {metrics['best_epoch']}  "
          f"val {metrics['val_top1_accuracy']:.2%}  test {metrics['test_top1_accuracy']:.2%}")
    if args.save:
        _save_checkpoint(model, args, candidates, "official", {"candidate_specs": candidates, **metrics})
    return {"mode": "clotho_official", "n": args.n, "candidate_specs": candidates,
            "aggregation": "official_train_val_test", **metrics}


def _save_checkpoint(model: torch.nn.Module, args: argparse.Namespace,
                     candidates: Sequence[str], label: str, summary: dict) -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"selector_{args.model}_{args.dataset}_n{args.n}_{args.feature}_{label}"
    checkpoint = CHECKPOINTS_DIR / f"{stem}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "candidate_specs": list(candidates),
        "feature": args.feature,
        "head": args.head,
    }, checkpoint)
    checkpoint.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  saved checkpoint -> {checkpoint}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", "-m", required=True, choices=["qwen2", "af3"])
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASETS))
    parser.add_argument("--alpha", "-a", type=float, default=PAPER_ALPHA)
    parser.add_argument("--prompt", default="constrained", choices=sorted(PROMPTS))
    parser.add_argument("--n", type=int, default=4, help="Candidate-pool size")
    parser.add_argument("--feature", choices=sorted(FEATURES), default=DEFAULT_FEATURE)
    parser.add_argument("--head", choices=sorted(HEAD_VARIANTS), default=SELECTOR_SPEC.name,
                        help="MLP head variant from the architecture sweep")

    candidate_group = parser.add_mutually_exclusive_group()
    candidate_group.add_argument("--paper-candidates", action="store_true",
                                 help="Use the paper's listed N=4 candidate labels exactly")
    candidate_group.add_argument("--candidate-spec", action="append", metavar="TYPE[:SETTING]",
                                 help="Explicit candidate label; repeat once per branch")

    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--all-splits", action="store_true",
                             help="Explicitly request five-split AH aggregation (the default)")
    split_group.add_argument("--split-name", help="Run one named AH split only; diagnostic, not a paper result")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--sweep-n", type=int, nargs="*", metavar="N",
                            help=f"Sweep candidate-pool size (Table V); default {PAPER_N_VALUES}")
    mode_group.add_argument("--feature-ablation", action="store_true",
                            help="Sweep input-feature configurations (Table VI)")
    parser.add_argument("--features", nargs="+", choices=sorted(FEATURES),
                        help="Restrict --feature-ablation to these configurations")

    parser.add_argument("--unregularized", action="store_true",
                        help="Train with every regulariser off (the Section VI-D baseline)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--feature-noise", type=float, default=None)
    parser.add_argument("--input-dropout", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.candidate_spec and len(args.candidate_spec) != args.n:
        parser.error("Pass --candidate-spec exactly N times")
    if args.dataset == "clotho_aqa" and args.split_name:
        parser.error("Clotho-AQA uses its official partition; do not pass --split-name")
    if args.dataset == "clotho_aqa" and (args.sweep_n is not None or args.feature_ablation):
        parser.error("--sweep-n and --feature-ablation apply to the AH benchmarks only")
    if args.features and not args.feature_ablation:
        parser.error("--features only applies together with --feature-ablation")

    base = TrainConfig.unregularized() if args.unregularized else TrainConfig()
    train_cfg = base.with_overrides(
        epochs=args.epochs, label_smoothing=args.label_smoothing,
        feature_noise_std=args.feature_noise, input_dropout=args.input_dropout,
        mixup_alpha=args.mixup,
    )
    spec = HEAD_VARIANTS[args.head]

    if args.dataset == "clotho_aqa":
        oracles = [
            load_oracle(RunConfig(model=args.model, dataset=args.dataset, alpha=args.alpha,
                                  split=split, prompt_name=args.prompt).oracle_file)
            for split in ("train", "val", "test")
        ]
        print(f"{args.model} | clotho_aqa | official partition | head={args.head}")
        result = run_clotho(args, oracles, train_cfg, spec)
    else:
        cfg = RunConfig(model=args.model, dataset=args.dataset, alpha=args.alpha,
                        prompt_name=args.prompt)
        oracles = [load_oracle(cfg.oracle_file)]
        print(f"{cfg.describe()} | head={args.head} | "
              f"regularisation={'off' if args.unregularized else 'paper defaults'}")

        if args.feature_ablation:
            result = run_feature_ablation(args, cfg, oracles, train_cfg, spec)
        else:
            dataset = build_dataset(cfg, args.feature)
            print(f"Loaded {len(dataset):,} cached example(s); feature dims {dataset.input_dims}")
            result = (run_sweep_n(args, cfg, dataset, oracles, train_cfg, spec)
                      if args.sweep_n is not None
                      else run_single(args, cfg, dataset, oracles, train_cfg, spec))

    result.update({
        "model": args.model, "dataset": args.dataset, "alpha": args.alpha,
        "prompt": args.prompt, "feature": args.feature, "head": args.head,
    })
    if args.save:
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        summary = CHECKPOINTS_DIR / f"selector_{args.model}_{args.dataset}_{result['mode']}_summary.json"
        summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved selector summary -> {summary}")


if __name__ == "__main__":
    main()
