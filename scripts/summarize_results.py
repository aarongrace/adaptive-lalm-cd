#!/usr/bin/env python3
"""Turn a directory of decoding results into the paper's tables and figures.

Everything here is post-processing over ``eval_*.json.gz`` files. No model is
loaded and no GPU is needed, because each result file already stores the
step-0 clean and negative logits for every example.

  ``--table``            per-branch accuracy, affirmative bias, and mean
                         softmax distance, written as CSV.
  ``--ranking``          the per-setting ranking in the paper's Table IV
                         style: best and worst branches per task, against the
                         unmodified baseline and the oracle.
  ``--prompt-table``     Table II: the open AAD prompt against this paper's
                         constrained yes/no prompt, with and without
                         contrastive decoding, plus the affirmative bias that
                         explains the difference. Needs both prompts decoded.
  ``--alpha-figure``     accuracy against contrastive strength alpha
                         (Section V-B, Fig. 2), recomputed from stored logits.
  ``--distance-figure``  accuracy against candidate-pool size N under
                         VACoDe-style maximum-divergence branch selection
                         (Section V-D, Fig. 4), one line per distance metric.

The alpha figure deserves a note: because Eq. (3) combines two fixed logit
vectors and the constrained prompt makes the answer a single token, sweeping
alpha over an existing run is exact rather than approximate. It reproduces
what the decoder would have emitted at each alpha, with no re-inference.

Usage:
    python -m scripts.summarize_results --model qwen2 --dataset ah_existence --table --ranking
    python -m scripts.summarize_results --model qwen2 --dataset ah_existence --prompt-table
    python -m scripts.summarize_results --model qwen2 --dataset ah_existence --alpha-figure \
        --perturbations NO_AUDIO NOISE:extreme BANDPASS:bass_only HARMONIC_REMOVE:full \
                        REPEAT_SEGMENT:repeat_middle
    python -m scripts.summarize_results --model qwen2 --dataset ah_existence --distance-figure
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional

from decoding.contrastive import SOFTMAX_DISTANCE_KEYS, predict_yes_no_at_alpha
from helpers.config import DATASETS, OUTPUTS_DIR, PAPER_ALPHA, PROMPTS, RunConfig
from helpers.datasets import example_key
from helpers.results import load_run_result, prediction_bias
from perturbations import family_of, paper_label

ALPHA_GRID = [round(step * 0.1, 1) for step in range(0, 21)]  # 0.0 .. 2.0
TOP_ROWS = 4
BOTTOM_ROWS = 2


def _split_spec(label: str) -> tuple[str, Optional[str]]:
    perturbation_type, _, setting = label.partition(":")
    return perturbation_type, setting or None


def _result_files(cfg: RunConfig) -> list[Path]:
    files = sorted(cfg.results_dir.glob("eval_*.json*"))
    if not files:
        raise FileNotFoundError(
            f"No decoding results in {cfg.results_dir}\n"
            "Run `python -m decoding.run_parallel` for this model/dataset/prompt first."
        )
    return files


def _load_runs(cfg: RunConfig) -> list:
    return [load_run_result(path) for path in _result_files(cfg)]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_table(cfg: RunConfig) -> Path:
    """One CSV row per decoded branch: accuracy, bias, and mean distances."""
    runs = _load_runs(cfg)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"table_{cfg.model}_{cfg.dataset}_{cfg.prompt_tag}_alpha{cfg.alpha}.csv"

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "perturbation_type", "perturbation_setting", "family", "paper_label",
            "accuracy", "correct", "total", "predicted_yes_rate", "affirmative_bias",
            *[f"avg_{key}" for key in SOFTMAX_DISTANCE_KEYS],
        ])
        for run in sorted(runs, key=lambda r: -r.accuracy):
            distances = run.avg_softmax_distance or {}
            bias = prediction_bias(run.results)
            writer.writerow([
                run.perturbation_type, run.perturbation_setting or "",
                family_of(run.perturbation_type) or "",
                paper_label(run.perturbation_type, run.perturbation_setting),
                f"{run.accuracy:.4f}", run.correct, run.total,
                bias["predicted_yes_rate"] if bias["predicted_yes_rate"] is not None else "",
                bias["affirmative_bias"] if bias["affirmative_bias"] is not None else "",
                *[distances.get(key, "") for key in SOFTMAX_DISTANCE_KEYS],
            ])
    print(f"Wrote {len(runs)} rows -> {out_path}")
    return out_path


def print_ranking(cfg: RunConfig, top: int = TOP_ROWS, bottom: int = BOTTOM_ROWS) -> None:
    """Best and worst branches for one setting, in the paper's Table IV layout."""
    runs = _load_runs(cfg)
    by_label = {run.spec_label: run for run in runs}
    original = by_label.get("ORIGINAL")
    contrastive = sorted(
        (run for run in runs if run.perturbation_type != "ORIGINAL"),
        key=lambda run: -run.accuracy,
    )
    if not contrastive:
        print("No contrastive branches decoded; nothing to rank.")
        return

    # Oracle here is over the decoded branches present in this directory.
    keys = {example_key({"path": s.audio_file, "Q": s.question}) for s in contrastive[0].results}
    solved = {
        key for run in runs for s, key in
        ((s, example_key({"path": s.audio_file, "Q": s.question})) for s in run.results)
        if s.is_correct
    }
    oracle = len(solved & keys) / len(keys) if keys else 0.0

    print(f"\n{cfg.model_label} / {cfg.dataset} (alpha={cfg.alpha}, prompt={cfg.prompt_tag})")
    if original is not None:
        print(f"  Original (no contrastive decoding)   {original.accuracy:.1%}")
    print(f"  Oracle over {len(runs)} decoded branches  {oracle:.1%}")
    print(f"  {'rank':>5}  {'branch':<40} {'acc':>7}  {'vs original':>12}  family")

    def show(position: int, run) -> None:
        delta = f"{run.accuracy - original.accuracy:+.1%}" if original is not None else "n/a"
        print(f"  {position:>5}  {paper_label(run.perturbation_type, run.perturbation_setting):<40} "
              f"{run.accuracy:>7.1%}  {delta:>12}  {family_of(run.perturbation_type) or '-'}")

    for position, run in enumerate(contrastive[:top], start=1):
        show(position, run)
    if len(contrastive) > top + bottom:
        print(f"  {'...':>5}  {len(contrastive) - top - bottom} branch(es) omitted")
    for offset, run in enumerate(contrastive[-bottom:]):
        show(len(contrastive) - bottom + offset + 1, run)


def print_prompt_table(args: argparse.Namespace) -> None:
    """Table II: prompt calibration, with and without no-audio contrastive decoding.

    The paper's point is that constraining the output to a single yes/no token
    is not cosmetic. It removes a large share of the model's affirmative bias
    before any contrastive decoding runs, and contrastive decoding then removes
    most of what remains -- the two are largely additive.
    """
    print(f"\nPrompt calibration -- {args.model} / {args.dataset} (alpha={args.alpha})")
    header = f"  {'prompt':<14} {'branch':<12} {'accuracy':>9}  {'yes rate':>9}  {'affirmative bias':>17}"
    print(header)

    for prompt_name in sorted(PROMPTS):
        cfg = RunConfig(model=args.model, dataset=args.dataset, alpha=args.alpha,
                        split=args.split, prompt_name=prompt_name)
        if not cfg.results_dir.exists():
            print(f"  {prompt_name:<14} not decoded ({cfg.results_dir} is missing)")
            continue
        by_label = {run.spec_label: run for run in _load_runs(cfg)}
        for branch in ("ORIGINAL", "NO_AUDIO"):
            run = by_label.get(branch)
            if run is None:
                print(f"  {prompt_name:<14} {branch:<12} {'not decoded':>9}")
                continue
            bias = prediction_bias(run.results)
            print(f"  {prompt_name:<14} {branch:<12} {run.accuracy:>9.1%}  "
                  f"{bias['predicted_yes_rate']:>9.1%}  {bias['affirmative_bias']:>+17.1%}")
    print("  (ORIGINAL is the model with no contrastive decoding; NO_AUDIO is AAD's branch.)")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def accuracy_at_alphas(result_path: Path, alphas: Iterable[float]) -> dict[float, float]:
    """Re-score one stored run across a grid of alpha values."""
    run = load_run_result(result_path)
    pairs = []
    for sample in run.results:
        yes_no = (sample.logit_trace or {}).get("yes_no_logits", {})
        clean, negative = yes_no.get("original"), yes_no.get("negative")
        if clean and negative:
            pairs.append((clean, negative, sample.ground_truth))
    if not pairs:
        raise ValueError(
            f"{result_path.name} stores no negative-branch logits. The ORIGINAL branch "
            "runs no negative pass, so it has no alpha to sweep."
        )
    curve = {}
    for alpha in alphas:
        correct = sum(
            1 for clean, negative, truth in pairs
            if predict_yes_no_at_alpha(clean, negative, alpha) == truth
        )
        curve[alpha] = correct / len(pairs)
    return curve


def write_alpha_figure(cfg: RunConfig, labels: list[str]) -> None:
    """Accuracy against alpha for a set of branches (Section V-B, Fig. 2).

    The paper's reading of this figure: helpful branches improve with alpha and
    plateau around 1.0, while harmful ones degrade monotonically from the
    outset, which is why alpha is fixed at 1.0 rather than tuned per branch.
    """
    import matplotlib.pyplot as plt

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 5))
    rows = [["perturbation", *[str(alpha) for alpha in ALPHA_GRID]]]

    for label in labels:
        perturbation_type, setting = _split_spec(label)
        stem = perturbation_type.lower() + (f"_{setting.lower()}" if setting else "")
        matches = sorted(cfg.results_dir.glob(f"eval_{stem}_alpha_*.json*"))
        if not matches:
            print(f"  skip {label}: no result file matching eval_{stem}_alpha_*.json* "
                  f"in {cfg.results_dir}")
            continue
        curve = accuracy_at_alphas(matches[0], ALPHA_GRID)
        axes.plot(ALPHA_GRID, [curve[a] for a in ALPHA_GRID], marker="o", markersize=3,
                  label=paper_label(perturbation_type, setting))
        rows.append([label, *[f"{curve[a]:.4f}" for a in ALPHA_GRID]])

    if len(rows) == 1:
        print("No branches could be plotted; nothing written.")
        return

    axes.axvline(PAPER_ALPHA, color="0.6", linestyle="--", linewidth=1)
    axes.annotate(f"alpha={PAPER_ALPHA}", xy=(PAPER_ALPHA, axes.get_ylim()[0]),
                  xytext=(4, 6), textcoords="offset points", fontsize=8, color="0.4")
    axes.set_xlabel("alpha (contrastive strength)")
    axes.set_ylabel("accuracy")
    axes.set_title(f"Alpha sensitivity -- {cfg.model_label} / {cfg.dataset}")
    axes.legend(fontsize=8)
    figure.tight_layout()

    png_path = OUTPUTS_DIR / f"alpha_sweep_{cfg.model}_{cfg.dataset}.png"
    csv_path = png_path.with_suffix(".csv")
    figure.savefig(png_path, dpi=150)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    print(f"Wrote {png_path} and {csv_path}")


def write_distance_figure(cfg: RunConfig, metrics: list[str], max_n: Optional[int] = None) -> None:
    """VACoDe-style distance selection swept over pool size N (Section V-D, Fig. 4).

    For each pool size the branch with the largest clean-versus-negative
    softmax divergence is chosen per example. The paper's finding is that this
    is unstable across the full pool and beats the fixed no-audio baseline only
    marginally at best, which is why the learned selector is needed: divergence
    turns out to be a poor proxy for contrastive utility, since a branch can
    diverge simply by triggering a different hallucination.
    """
    import matplotlib.pyplot as plt

    runs = [run for run in _load_runs(cfg) if run.perturbation_type != "ORIGINAL"]
    if not runs:
        raise ValueError("No contrastive branches decoded; nothing to select among.")

    # Rank branches by aggregate accuracy so the sweep grows a top-N pool, the
    # strongest variant the paper reports.
    runs.sort(key=lambda run: -run.accuracy)
    per_branch = []
    for run in runs:
        by_key = {}
        for sample in run.results:
            if not sample.softmax_distance:
                continue
            by_key[example_key({"path": sample.audio_file, "Q": sample.question})] = (
                sample.softmax_distance, sample.is_correct,
            )
        if by_key:
            per_branch.append((run.spec_label, run.accuracy, by_key))

    if not per_branch:
        raise ValueError("Decoded results carry no softmax distances to select on.")

    baseline = next((run.accuracy for run in runs if run.perturbation_type == "NO_AUDIO"), None)
    limit = min(max_n or len(per_branch), len(per_branch))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 5))
    rows = [["N", *metrics]]
    curves: dict[str, list[float]] = {metric: [] for metric in metrics}
    sizes = list(range(1, limit + 1))

    # Growing the pool one branch at a time lets the per-example argmax be
    # maintained incrementally: each step only compares the newly added branch
    # against the running best, instead of rescanning the whole pool.
    best: dict[str, dict[str, tuple[float, bool]]] = {metric: {} for metric in metrics}
    for n in sizes:
        _, _, added = per_branch[n - 1]
        for key, (distances, is_correct) in added.items():
            for metric in metrics:
                distance = distances.get(metric)
                if distance is None:
                    continue
                current = best[metric].get(key)
                if current is None or distance > current[0]:
                    best[metric][key] = (distance, is_correct)
        for metric in metrics:
            chosen = best[metric]
            correct = sum(1 for _, is_correct in chosen.values() if is_correct)
            curves[metric].append(correct / len(chosen) if chosen else 0.0)
        rows.append([n, *[f"{curves[metric][-1]:.4f}" for metric in metrics]])

    for metric in metrics:
        axes.plot(sizes, curves[metric], linewidth=1.2, label=metric)
    if baseline is not None:
        axes.axhline(baseline, color="0.4", linestyle="--", linewidth=1,
                     label=f"fixed no-audio ({baseline:.1%})")

    axes.set_xlabel("N (candidate pool size, branches ranked by accuracy)")
    axes.set_ylabel("accuracy")
    axes.set_title(f"Distance-based branch selection -- {cfg.model_label} / {cfg.dataset}")
    axes.legend(fontsize=8, ncol=2)
    figure.tight_layout()

    png_path = OUTPUTS_DIR / f"distance_selection_{cfg.model}_{cfg.dataset}.png"
    csv_path = png_path.with_suffix(".csv")
    figure.savefig(png_path, dpi=150)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)

    best_metric = max(metrics, key=lambda metric: max(curves[metric]))
    best_n = 1 + curves[best_metric].index(max(curves[best_metric]))
    print(f"Wrote {png_path} and {csv_path}")
    print(f"Best distance selection: {best_metric} at N={best_n}, "
          f"{max(curves[best_metric]):.2%}"
          + (f" (fixed no-audio {baseline:.2%})" if baseline is not None else ""))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", "-m", required=True, choices=["qwen2", "af3"])
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASETS))
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--alpha", "-a", type=float, default=PAPER_ALPHA)
    parser.add_argument("--prompt", default="constrained", choices=sorted(PROMPTS))
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--ranking", action="store_true")
    parser.add_argument("--prompt-table", action="store_true")
    parser.add_argument("--alpha-figure", action="store_true")
    parser.add_argument("--perturbations", nargs="+", default=["NO_AUDIO"],
                        help="TYPE or TYPE:setting labels to include in the alpha figure")
    parser.add_argument("--distance-figure", action="store_true")
    parser.add_argument("--metric", nargs="+", default=list(SOFTMAX_DISTANCE_KEYS),
                        choices=list(SOFTMAX_DISTANCE_KEYS),
                        help="Distance metrics to plot; the paper's figure shows all six")
    parser.add_argument("--max-n", type=int, default=None)
    args = parser.parse_args()

    modes = (args.table, args.ranking, args.prompt_table, args.alpha_figure, args.distance_figure)
    if not any(modes):
        parser.error("choose at least one of --table / --ranking / --prompt-table / "
                     "--alpha-figure / --distance-figure")

    cfg = RunConfig(model=args.model, dataset=args.dataset, alpha=args.alpha,
                    split=args.split, prompt_name=args.prompt)

    if args.table:
        write_table(cfg)
    if args.ranking:
        print_ranking(cfg)
    if args.prompt_table:
        print_prompt_table(args)
    if args.alpha_figure:
        write_alpha_figure(cfg, args.perturbations)
    if args.distance_figure:
        write_distance_figure(cfg, args.metric, args.max_n)


if __name__ == "__main__":
    main()
