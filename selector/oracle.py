"""
Oracle construction: turns a directory of per-perturbation decoding results
into a per-example, multi-label correctness target.

For every example ``i`` and branch ``s``, the paper defines
``M[i, s] = 1[prediction == label]``, and the oracle accuracy

    A_oracle = (1/N) * sum_i  max_s M[i, s]                        (Eq. 4)

is the accuracy an ideal per-example router would reach -- the fraction of
examples that *some* branch in the pool answers correctly. It is the upper
bound the paper measures selector headroom against (86.2% against a 72.4% best
fixed branch on Qwen2 AH Existence), and no fixed or learned single choice can
exceed it. The same ``M[i]`` row is the multi-hot BCE target the selector
trains on (``selector/train.py``).

``ORIGINAL`` is retained by default. It is the unperturbed reference rather
than a contrastive transformation, but it is a legitimate routing decision --
"apply no correction to this example" -- and the paper's AF3 AH Existence N=4
candidate pool explicitly contains it. Excluding it must therefore be a
deliberate, recorded choice rather than a silent default.

This module also implements the greedy branch-coverage search behind the
paper's observation that the library is far more redundant than its size
suggests: roughly half of it (N ~ 51) is needed to reach maximum oracle
coverage, but the coverage curve is effectively flat after N = 10. That is the
evidence that the selector is data-limited rather than candidate-limited.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

from helpers.datasets import example_key
from helpers.results import load_run_result
from perturbations import spec_label  # re-exported: the canonical branch label

__all__ = [
    "spec_label",
    "build_oracle",
    "save_oracle",
    "load_oracle",
    "oracle_accuracy",
    "branch_accuracies",
    "greedy_branch_coverage",
]


def build_oracle(results_dir: Path, *, include_original: bool = True) -> dict:
    """Aggregate every ``eval_*.json(.gz)`` in ``results_dir`` into correctness rows.

    Returns ``{"metadata": ..., "branch_specs": [...], "examples": {key: {...}}}``
    where each example carries ``worked_perturbations``: the branches that
    answered it correctly.

    The consistency checks here are deliberately strict. The oracle is a dense
    matrix over (example, branch), so a branch decoded over a different example
    set, a duplicated row, or a disagreeing ground-truth label would silently
    misalign every downstream split, cache, and selector target. Failing loudly
    at build time is much cheaper than discovering it in a trained selector.
    """
    result_files = sorted(results_dir.glob("eval_*.json*"))
    if not result_files:
        raise FileNotFoundError(
            f"No eval_*.json(.gz) result files found in {results_dir}\n"
            "Run `python -m decoding.run_parallel` for this model/dataset first."
        )

    examples: dict[str, dict] = {}
    expected_keys: Optional[set[str]] = None
    branch_specs: list[str] = []

    for path in result_files:
        run = load_run_result(path)
        if run.perturbation_type == "ORIGINAL" and not include_original:
            continue
        label = run.spec_label
        if label in branch_specs:
            raise ValueError(f"Duplicate decoded branch label {label!r}: {path}")
        branch_specs.append(label)

        keys: set[str] = set()
        for sample in run.results:
            key = example_key({"path": sample.audio_file, "Q": sample.question})
            if key in keys:
                raise ValueError(f"Duplicate example {key!r} in {path}")
            keys.add(key)

        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(
                f"Result files do not cover the same examples: {path.name} has "
                f"{len(expected_keys - keys)} missing and {len(keys - expected_keys)} "
                "unexpected example(s). Complete or repair the decoding sweep first."
            )

        for sample in run.results:
            key = example_key({"path": sample.audio_file, "Q": sample.question})
            entry = examples.setdefault(key, {
                "audio_file": sample.audio_file,
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "worked_perturbations": [],
            })
            if entry["ground_truth"] != sample.ground_truth:
                raise ValueError(
                    f"Ground-truth disagreement for {key!r} in {path}; "
                    "repair the decoding sweep before building an oracle."
                )
            if sample.is_correct:
                entry["worked_perturbations"].append(label)

    if not branch_specs or not examples:
        raise ValueError(
            "No usable result rows were found. Decode at least one branch "
            "(and more than ORIGINAL alone, if --exclude-original was passed)."
        )

    accuracy = oracle_accuracy(examples)
    per_branch = branch_accuracies(examples, branch_specs)
    best_label, best_accuracy = max(per_branch.items(), key=lambda kv: (kv[1], kv[0]))

    print(
        f"Oracle: {len(examples):,} examples x {len(branch_specs)} branches | "
        f"oracle accuracy {accuracy:.2%} (Eq. 4) | "
        f"best fixed branch {best_label} at {best_accuracy:.2%} | "
        f"headroom {accuracy - best_accuracy:+.2%}"
    )

    return {
        "metadata": {
            "include_original": include_original,
            "num_branches": len(branch_specs),
            "num_examples": len(examples),
            "oracle_accuracy": round(accuracy, 6),
            "best_fixed_branch": best_label,
            "best_fixed_accuracy": round(best_accuracy, 6),
        },
        "branch_specs": branch_specs,
        "branch_accuracies": {k: round(v, 6) for k, v in per_branch.items()},
        "examples": examples,
    }


def save_oracle(oracle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(oracle, f)


def load_oracle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Oracle not found: {path}\nRun `python -m scripts.build_oracle` first."
        )
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def _as_examples(oracle_or_examples: dict) -> dict:
    """Accept either a full oracle payload or its ``examples`` mapping."""
    return oracle_or_examples.get("examples", oracle_or_examples)


def oracle_accuracy(oracle_or_examples: dict,
                    candidate_specs: Optional[Sequence[str]] = None) -> float:
    """Eq. (4): fraction of examples solved by at least one branch in the pool.

    With ``candidate_specs`` given, the maximum is taken over that pool only,
    which is how the paper's oracle column is computed at each candidate-pool
    size N (Table V).
    """
    examples = _as_examples(oracle_or_examples)
    if not examples:
        return 0.0
    pool = set(candidate_specs) if candidate_specs is not None else None
    solved = 0
    for row in examples.values():
        worked = row["worked_perturbations"]
        if pool is None:
            solved += bool(worked)
        else:
            solved += any(spec in pool for spec in worked)
    return solved / len(examples)


def branch_accuracies(oracle_or_examples: dict,
                      branch_specs: Optional[Iterable[str]] = None) -> dict[str, float]:
    """Per-branch accuracy over the oracle's examples.

    Branches are seeded from ``branch_specs`` so that one which happens to
    solve nothing still appears at 0.0 rather than vanishing from the ranking.
    """
    examples = _as_examples(oracle_or_examples)
    if branch_specs is None and isinstance(oracle_or_examples, dict):
        branch_specs = oracle_or_examples.get("branch_specs")
    counts: dict[str, int] = {spec: 0 for spec in (branch_specs or [])}
    for row in examples.values():
        for spec in row["worked_perturbations"]:
            counts[spec] = counts.get(spec, 0) + 1
    total = len(examples) or 1
    return {spec: count / total for spec, count in counts.items()}


def rank_branches(oracle_or_examples: dict,
                  branch_specs: Optional[Iterable[str]] = None) -> list[tuple[str, float]]:
    """Branches ordered by accuracy, descending; ties broken by name for stability."""
    accuracies = branch_accuracies(oracle_or_examples, branch_specs)
    return sorted(accuracies.items(), key=lambda kv: (-kv[1], kv[0]))


def greedy_branch_coverage(oracle_or_examples: dict,
                           branch_specs: Optional[Iterable[str]] = None,
                           max_n: Optional[int] = None) -> list[dict]:
    """Greedily grow a candidate pool, maximising oracle coverage at each step.

    At every step this adds the branch that newly solves the largest number of
    still-unsolved examples. The resulting curve is the paper's evidence
    (Section VI-A) that the 105-branch library is highly redundant: coverage
    keeps creeping up until roughly half the library is included, yet is
    effectively flat past N = 10. Since selector accuracy *falls* over that
    same range as the training signal is diluted across more classes, the
    bottleneck is training data rather than candidate diversity.

    Returns one row per step with the branch chosen, how many examples it
    newly covered, and the cumulative oracle accuracy at that pool size.
    """
    examples = _as_examples(oracle_or_examples)
    if branch_specs is None and isinstance(oracle_or_examples, dict):
        branch_specs = oracle_or_examples.get("branch_specs")
    specs = list(branch_specs or sorted(
        {spec for row in examples.values() for spec in row["worked_perturbations"]}
    ))
    if not specs or not examples:
        return []

    solved_by: dict[str, set[str]] = {spec: set() for spec in specs}
    for key, row in examples.items():
        for spec in row["worked_perturbations"]:
            if spec in solved_by:
                solved_by[spec].add(key)

    total = len(examples)
    limit = min(max_n or len(specs), len(specs))
    covered: set[str] = set()
    remaining = set(specs)
    curve: list[dict] = []

    for step in range(1, limit + 1):
        gains = {spec: len(solved_by[spec] - covered) for spec in remaining}
        gain = max(gains.values())
        # Ties are broken by branch name, so the curve is reproducible.
        best = min(spec for spec, value in gains.items() if value == gain)
        covered |= solved_by[best]
        remaining.discard(best)
        curve.append({
            "n": step,
            "branch": best,
            "newly_covered": gain,
            "oracle_accuracy": len(covered) / total,
        })
        if not remaining:
            break
    return curve
