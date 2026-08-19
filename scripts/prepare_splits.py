#!/usr/bin/env python3
"""Create the paper's composition-aware AH selector splits.

The AH benchmarks ship without standard splits, so the paper constructs five
balanced 70/15/15 replicates per task and averages results across them. These
are emphatically **not** five random record-level shuffles, and the difference
matters enough to be worth stating plainly.

Every AH audio file is a rendered mixture of a background track and several
foreground events, and each mixture carries many questions. Splitting at the
record level would scatter questions about the same audio across train and
test, so a selector could memorise a clip during training and be rewarded for
recognising it at test time. This script therefore partitions at the
**audio-composition** level: all questions derived from one mixture move
together.

For AH Order and AH Attribute that is sufficient -- no audio file appears in
two partitions. For AH Existence it is not, and the paper says so: the
benchmark builds its 10,800 mixtures by recombining a much smaller pool of
source recordings, so most source clips appear in many compositions. Zero
train/held-out source overlap is therefore *structurally impossible* at a
70/15/15 ratio, and the search minimises overlap as a penalty instead of
pretending to eliminate it. ``--report`` prints the inventory that makes this
concrete: how many distinct sources exist, how many compositions each appears
in, and how low the overlap can actually be driven.

The search runs in three phases plus a refinement stage:

  Phase 1  Score ``--phase1`` composition-level permutations by difficulty
           balance -- how evenly hard examples are spread across the three
           partitions -- and keep the best ``--phase2``.
  Phase 2  Re-score those by weighted per-branch accuracy variance (so no
           partition is unusually favourable to a particular negative branch)
           plus source-clip overlap; keep the best ``--phase3``.
  Phase 3  Refine the top ``--refine-pool`` candidates by partition-preserving
           swaps, ramping the overlap penalty through several stages. This
           stage is what actually reaches low overlap: random permutations
           cluster tightly around one value, so sampling alone can only return
           the best of a pool that never contained a good candidate.
  Select   Choose the final five for uniform held-out coverage, so every
           example appears in validation or test a similar number of times and
           contributes roughly equally to the cross-split mean.

The default budget matches the experiment implementation. Smaller ``--phase*``
values are for smoke-testing a new manifest; their outputs are recorded as not
paper-compatible in each split's metadata.

Usage:
    python -m scripts.prepare_splits --model qwen2 --dataset ah_existence --report
    python -m scripts.prepare_splits --model af3 --dataset ah_order
    python -m scripts.prepare_splits --model qwen2 --dataset ah_existence --verify
    python -m scripts.prepare_splits --model qwen2 --dataset ah_existence \
        --phase1 100 --phase2 20 --phase3 10 --no-refine   # smoke test only
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm

from helpers.config import AH_DATASETS, N_PAPER_SPLITS, PAPER_ALPHA, PROMPTS, RunConfig
from helpers.datasets import example_key, list_splits, load_examples, load_split, save_split
from selector.oracle import load_oracle

RATIOS = (0.70, 0.15, 0.15)
PAPER_PHASE1 = 5_000_000
PAPER_PHASE2 = 500_000
PAPER_PHASE3 = 5_000
PAPER_REFINE_POOL = 24
PAPER_REFINE_SWEEPS = 40
REFINE_WEIGHT_STAGES = (0.5, 2.0, 5.0, 12.0, 30.0, 80.0)
REFINE_SAMPLE_SIZE = 48
RANDOM_SEED = 42


# ===========================================================================
# Composition grouping
# ===========================================================================

def _path_stem(item: dict) -> str:
    return Path(item["path"]).stem


def build_composition_groups(
    examples: list[dict], dataset: str
) -> tuple[list[tuple], dict[tuple, list[int]], dict[tuple, frozenset[str]], str]:
    """Return group IDs per row, member rows per group, source IDs, and the basis.

    For paper-compatible AH Existence manifests either supply
    ``composition_id`` plus ``source_clip_ids`` on every row, or preserve the
    raw BEAF-style ``background@event_a@event_b@event_c`` filename convention.
    Two-event "pair" recordings are mapped back to the parent three-event
    combination they were derived from, exactly as the experiment split builder
    does; without that step a pair and its parent would be treated as unrelated
    compositions and could land on opposite sides of the boundary.
    """
    if not examples:
        raise ValueError("Cannot split an empty manifest")

    if dataset != "ah_existence":
        # Order and Attribute have one composition per audio file, so grouping
        # by explicit id (when present) or by audio path is exact.
        group_ids = [
            (str(item["composition_id"]),) if item.get("composition_id") else (_path_stem(item),)
            for item in examples
        ]
        source_ids: dict[tuple, frozenset[str]] = {}
        for item, group in zip(examples, group_ids):
            sources = frozenset(str(value) for value in item.get("source_clip_ids", [_path_stem(item)]))
            previous = source_ids.setdefault(group, sources)
            if previous != sources:
                raise ValueError(f"Inconsistent source_clip_ids within composition {group[0]!r}")
        basis = "composition_id" if all(item.get("composition_id") for item in examples) else "audio_path"
        return _group_rows(group_ids, source_ids, basis)

    # Explicit metadata is preferred: it survives any filename convention.
    if all(item.get("composition_id") for item in examples):
        missing_sources = [i for i, item in enumerate(examples) if not item.get("source_clip_ids")]
        if missing_sources:
            raise ValueError(
                "AH Existence paper splits require source_clip_ids when composition_id is supplied; "
                f"missing on {len(missing_sources)} row(s), first row {missing_sources[0]}."
            )
        group_ids = [(str(item["composition_id"]),) for item in examples]
        source_ids = {}
        for item, group in zip(examples, group_ids):
            clips = frozenset(str(clip) for clip in item["source_clip_ids"])
            if not clips:
                raise ValueError(f"Empty source_clip_ids for composition {group[0]!r}")
            previous = source_ids.setdefault(group, clips)
            if previous != clips:
                raise ValueError(f"Inconsistent source_clip_ids within composition {group[0]!r}")
        return _group_rows(group_ids, source_ids, "composition_id+source_clip_ids")

    stems = [_path_stem(item) for item in examples]
    if not all("@" in stem for stem in stems):
        raise ValueError(
            "AH Existence cannot be split paper-faithfully from paths alone. Supply "
            "composition_id and source_clip_ids in every manifest row, or preserve the raw "
            "BEAF background@event_a@event_b@event_c filename convention. Grouping by the "
            "rendered mixture file alone implements neither the composition split nor its "
            "train/held-out source-overlap objective."
        )

    # Pass 1: collect every three-event combination that exists per background.
    background_to_triples: dict[str, set[frozenset[str]]] = defaultdict(set)
    for stem in stems:
        parts = stem.split("@")
        if len(parts) == 4:
            background_to_triples[parts[0]].add(frozenset(parts[1:]))

    # Pass 2: map each two-event subset back to its parent triple.
    pair_parent: dict[tuple[str, frozenset[str]], tuple] = {}
    for background, triples in background_to_triples.items():
        for triple in triples:
            for omitted in triple:
                pair_parent[(background, triple - {omitted})] = (background, triple)

    group_ids = []
    source_ids = {}
    for stem in stems:
        parts = stem.split("@")
        background, events = parts[0], frozenset(parts[1:])
        if len(events) == 3:
            group = (background, events)
        elif len(events) == 2:
            group = pair_parent.get((background, events), (background, events))
        else:
            raise ValueError(f"Unrecognised AH Existence composition stem {stem!r}")
        group_ids.append(group)
        source_ids[group] = frozenset({background}) | group[1]
    return _group_rows(group_ids, source_ids, "BEAF filename composition")


def _group_rows(
    group_ids: list[tuple], source_ids: dict[tuple, frozenset[str]], basis: str
) -> tuple[list[tuple], dict[tuple, list[int]], dict[tuple, frozenset[str]], str]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        groups[group].append(index)
    for group in groups:
        source_ids.setdefault(group, frozenset(str(value) for value in group))
    return group_ids, dict(groups), source_ids, basis


# ===========================================================================
# Oracle alignment and scoring inputs
# ===========================================================================

def _require_oracle_alignment(examples: list[dict], oracle: dict) -> list[dict]:
    """Confirm the manifest and oracle describe exactly the same example set."""
    manifest_keys = [example_key(item) for item in examples]
    if len(set(manifest_keys)) != len(manifest_keys):
        raise ValueError("Manifest has duplicate (path, Q) keys; resolve them before building splits.")
    oracle_examples = oracle.get("examples", {})
    if set(manifest_keys) != set(oracle_examples):
        missing = len(set(manifest_keys) - set(oracle_examples))
        extra = len(set(oracle_examples) - set(manifest_keys))
        raise ValueError(
            "Manifest and oracle do not describe exactly the same examples "
            f"({missing} missing oracle rows, {extra} extra oracle rows). "
            "Rebuild the decoding sweep and oracle first."
        )
    return [oracle_examples[key] for key in manifest_keys]


def _oracle_matrix(oracle_rows: list[dict], branch_specs: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    """Dense ``[n_examples, n_branches]`` correctness matrix ``M`` from Eq. (4)."""
    specs = list(branch_specs)
    if not specs:
        specs = sorted({spec for row in oracle_rows for spec in row["worked_perturbations"]})
    if not specs:
        raise ValueError("Oracle has no branch specifications")
    index = {spec: i for i, spec in enumerate(specs)}
    matrix = np.zeros((len(oracle_rows), len(specs)), dtype=np.float64)
    for row, item in enumerate(oracle_rows):
        for spec in item["worked_perturbations"]:
            if spec in index:
                matrix[row, index[spec]] = 1.0
    return matrix, specs


def _perm_from_seed(seed: int, n_groups: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n_groups)


def _partition_counts(n_groups: int) -> tuple[int, int]:
    train = int(n_groups * RATIOS[0])
    val = int(n_groups * RATIOS[1])
    if train == 0 or val == 0 or n_groups - train - val == 0:
        raise ValueError(f"Need enough composition groups for a 70/15/15 split, found {n_groups}")
    return train, val


def _expand_perm(perm: np.ndarray, train_groups: int, val_groups: int,
                 group_keys: list[tuple], group_rows: dict[tuple, list[int]]
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a composition permutation into example-level row indices."""
    return (
        np.concatenate([group_rows[group_keys[int(i)]] for i in perm[:train_groups]]),
        np.concatenate([group_rows[group_keys[int(i)]] for i in perm[train_groups:train_groups + val_groups]]),
        np.concatenate([group_rows[group_keys[int(i)]] for i in perm[train_groups + val_groups:]]),
    )


def _difficulty_score(perm: np.ndarray, train_groups: int, val_groups: int,
                      group_difficulty: np.ndarray, global_mean: float) -> float:
    """Total deviation of each partition's mean difficulty from the global mean."""
    parts = (perm[:train_groups], perm[train_groups:train_groups + val_groups], perm[train_groups + val_groups:])
    return float(sum(abs(group_difficulty[part].mean() - global_mean) for part in parts))


def _variance_score(perm: np.ndarray, train_groups: int, val_groups: int,
                    group_matrix: np.ndarray, weights: np.ndarray) -> float:
    """Rarity-weighted variance of per-branch accuracy across the three partitions.

    Branches the model rarely gets right carry the most weight, because those
    are the ones a lopsided split would most distort.
    """
    parts = (perm[:train_groups], perm[train_groups:train_groups + val_groups], perm[train_groups + val_groups:])
    accuracies = np.stack([group_matrix[part].mean(axis=0) for part in parts])
    return float((weights * accuracies.var(axis=0)).sum())


def _clip_overlap(perm: np.ndarray, train_groups: int, group_keys: list[tuple],
                  group_sources: dict[tuple, frozenset[str]]) -> float:
    """Jaccard similarity of the train and held-out source-ID sets.

    0.0 means the two sides share no source recording; 1.0 means every source
    appears on both. Held-out is validation union test, matching the selector's
    evaluation boundary.
    """
    train_sources: set[str] = set()
    held_sources: set[str] = set()
    for index in perm[:train_groups]:
        train_sources.update(group_sources[group_keys[int(index)]])
    for index in perm[train_groups:]:
        held_sources.update(group_sources[group_keys[int(index)]])
    return len(train_sources & held_sources) / max(1, len(train_sources | held_sources))


# ===========================================================================
# Phase 3: partition-preserving swap refinement
# ===========================================================================

def _refine_by_swaps(
    perm: np.ndarray,
    train_groups: int,
    val_groups: int,
    group_keys: list[tuple],
    group_sources: dict[tuple, frozenset[str]],
    group_difficulty: np.ndarray,
    group_matrix: np.ndarray,
    weights: np.ndarray,
    global_difficulty: float,
    sweeps_per_stage: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Lower source overlap by swapping compositions between partitions.

    Every move is a swap, so partition cardinalities are preserved exactly and
    the 70/15/15 ratio survives the search. The objective is

        variance_score + difficulty_score + w * clip_overlap

    with ``w`` ramped through :data:`REFINE_WEIGHT_STAGES`: early stages settle
    the balance terms, later ones trade a little balance for a large drop in
    overlap. All three terms are maintained incrementally -- partition sums for
    the two balance scores, per-source train/held counts for the overlap -- so
    a candidate swap costs time proportional to the sources it touches rather
    than a full rescore. That is what makes several thousand sweeps practical.

    Note that validation-test swaps leave train/held overlap untouched, so the
    search can repair difficulty balance for free once the overlap weight is
    high.
    """
    n_groups = len(perm)
    assignment = np.empty(n_groups, dtype=np.int8)
    assignment[perm[:train_groups]] = 0
    assignment[perm[train_groups:train_groups + val_groups]] = 1
    assignment[perm[train_groups + val_groups:]] = 2

    part_size = np.array([train_groups, val_groups, n_groups - train_groups - val_groups], dtype=np.float64)
    matrix_sums = np.zeros((3, group_matrix.shape[1]), dtype=np.float64)
    difficulty_sums = np.zeros(3, dtype=np.float64)
    source_lists = [tuple(group_sources[key]) for key in group_keys]
    train_count: dict[str, int] = defaultdict(int)
    held_count: dict[str, int] = defaultdict(int)

    for group_index in range(n_groups):
        part = assignment[group_index]
        matrix_sums[part] += group_matrix[group_index]
        difficulty_sums[part] += group_difficulty[group_index]
        target = train_count if part == 0 else held_count
        for source in source_lists[group_index]:
            target[source] += 1

    all_sources = set(train_count) | set(held_count)
    shared = sum(train_count[s] > 0 and held_count[s] > 0 for s in all_sources)

    def move(group_index: int, source_part: int, target_part: int) -> None:
        nonlocal shared
        matrix_sums[source_part] -= group_matrix[group_index]
        matrix_sums[target_part] += group_matrix[group_index]
        difficulty_sums[source_part] -= group_difficulty[group_index]
        difficulty_sums[target_part] += group_difficulty[group_index]
        assignment[group_index] = target_part
        if (source_part == 0) == (target_part == 0):
            return  # a val<->test move cannot change train/held overlap
        delta = 1 if target_part == 0 else -1
        for clip in source_lists[group_index]:
            before = train_count[clip] > 0 and held_count[clip] > 0
            train_count[clip] += delta
            held_count[clip] -= delta
            after = train_count[clip] > 0 and held_count[clip] > 0
            shared += int(after) - int(before)

    def scores() -> tuple[float, float, float]:
        variance = float((weights * (matrix_sums / part_size[:, None]).var(axis=0)).sum())
        difficulty = float(np.abs(difficulty_sums / part_size - global_difficulty).sum())
        return variance, difficulty, shared / max(1, len(all_sources))

    trajectory: list[dict[str, float]] = []
    for weight in REFINE_WEIGHT_STAGES:
        variance, difficulty, overlap = scores()
        objective = variance + difficulty + weight * overlap
        for _ in range(sweeps_per_stage):
            accepted = 0
            for first in rng.permutation(n_groups):
                first = int(first)
                first_part = int(assignment[first])
                sample = rng.choice(n_groups, size=min(REFINE_SAMPLE_SIZE, n_groups), replace=False)
                for second in sample:
                    second = int(second)
                    second_part = int(assignment[second])
                    if first_part == second_part:
                        continue
                    move(first, first_part, second_part)
                    move(second, second_part, first_part)
                    var_trial, diff_trial, overlap_trial = scores()
                    if var_trial + diff_trial + weight * overlap_trial < objective - 1e-15:
                        objective = var_trial + diff_trial + weight * overlap_trial
                        accepted += 1
                        break  # keep this swap and move to the next composition
                    move(first, second_part, first_part)   # revert
                    move(second, first_part, second_part)
            if not accepted:
                break  # this stage has converged
        variance, difficulty, overlap = scores()
        trajectory.append({
            "weight": float(weight),
            "variance_score": variance,
            "difficulty_score": difficulty,
            "clip_overlap": overlap,
        })

    refined = np.concatenate([np.flatnonzero(assignment == part) for part in range(3)])
    return refined, trajectory


def _greedy_heldout_coverage(candidates: list[dict], n_splits: int, n_examples: int) -> list[dict]:
    """Pick the final replicates for the most uniform per-example held-out coverage.

    Starting from the best-scoring candidate, each step adds the remaining
    candidate that minimises the variance of the running per-example coverage
    count. After all five are chosen, every example has appeared in validation
    or test a similar number of times, so the cross-split mean weights examples
    roughly equally instead of over-representing whichever ones happen to sit
    in held-out partitions repeatedly. The variance update is computed
    analytically, so each step is linear in the held-out size.
    """
    if not candidates:
        return []
    selected = [candidates[0]]
    coverage = np.zeros(n_examples, dtype=np.float64)
    coverage[np.concatenate([candidates[0]["val"], candidates[0]["test"]])] += 1.0
    remaining = list(range(1, len(candidates)))

    for _ in range(min(n_splits, len(candidates)) - 1):
        if not remaining:
            break
        sum_sq, sum_value = float((coverage ** 2).sum()), float(coverage.sum())
        best_index, best_variance = None, float("inf")
        for candidate_index in remaining:
            held = np.concatenate([candidates[candidate_index]["val"], candidates[candidate_index]["test"]])
            next_sum_sq = sum_sq + 2.0 * float(coverage[held].sum()) + len(held)
            next_mean = (sum_value + len(held)) / n_examples
            variance = next_sum_sq / n_examples - next_mean ** 2
            if variance < best_variance:
                best_index, best_variance = candidate_index, variance
        assert best_index is not None
        held = np.concatenate([candidates[best_index]["val"], candidates[best_index]["test"]])
        coverage[held] += 1.0
        selected.append(candidates[best_index])
        remaining.remove(best_index)
    return selected


# ===========================================================================
# Search driver
# ===========================================================================

def create_paper_splits(
    examples: list[dict],
    oracle: dict,
    dataset: str,
    *,
    candidate_specs: Optional[list[str]] = None,
    phase1: int = PAPER_PHASE1,
    phase2: int = PAPER_PHASE2,
    phase3: int = PAPER_PHASE3,
    refine_pool: int = PAPER_REFINE_POOL,
    refine_sweeps: int = PAPER_REFINE_SWEEPS,
    n_splits: int = N_PAPER_SPLITS,
) -> tuple[list[dict], dict]:
    """Run the full search and return the selected splits plus shared metadata."""
    oracle_rows = _require_oracle_alignment(examples, oracle)
    matrix, all_specs = _oracle_matrix(oracle_rows, oracle.get("branch_specs", []))
    if candidate_specs:
        unknown = sorted(set(candidate_specs) - set(all_specs))
        if unknown:
            raise ValueError(f"Requested scoring branch(es) absent from oracle: {unknown}")
        matrix = matrix[:, [all_specs.index(spec) for spec in candidate_specs]]
        scoring_specs = list(candidate_specs)
    else:
        scoring_specs = all_specs

    group_ids, group_rows, group_sources, grouping_basis = build_composition_groups(examples, dataset)
    group_keys = list(group_rows)
    train_groups, val_groups = _partition_counts(len(group_keys))

    difficulty = 1.0 - matrix.mean(axis=1)          # per example: fraction of branches that fail it
    global_difficulty = float(difficulty.mean())
    weights = 1.0 - matrix.mean(axis=0)             # per branch: rarity of its successes
    group_difficulty = np.array([difficulty[group_rows[key]].mean() for key in group_keys])
    group_matrix = np.array([matrix[group_rows[key]].mean(axis=0) for key in group_keys])

    phase1 = max(1, phase1)
    phase2 = min(max(1, phase2), phase1)
    phase3 = min(max(n_splits * 20, phase3), phase2)

    print(f"{dataset}: {len(examples):,} examples in {len(group_keys):,} composition groups "
          f"(grouping basis: {grouping_basis})")
    print(f"Composition partition: {train_groups}/{val_groups}/"
          f"{len(group_keys) - train_groups - val_groups} train/val/test")
    print(f"Scoring against {len(scoring_specs)} branch(es); mean example difficulty "
          f"{global_difficulty:.4f}")

    print(f"Phase 1: scoring {phase1:,} composition permutations for difficulty balance")
    difficulty_scores = np.empty(phase1, dtype=np.float64)
    for seed in tqdm(range(phase1), desc="phase 1", unit="split"):
        difficulty_scores[seed] = _difficulty_score(
            _perm_from_seed(seed, len(group_keys)), train_groups, val_groups,
            group_difficulty, global_difficulty,
        )
    phase2_seeds = np.argsort(difficulty_scores)[:phase2]

    print(f"Phase 2: scoring {phase2:,} candidates for branch balance plus source overlap")
    combined = np.empty(phase2, dtype=np.float64)
    variance_scores = np.empty(phase2, dtype=np.float64)
    overlap_scores = np.empty(phase2, dtype=np.float64)
    for rank, seed in enumerate(tqdm(phase2_seeds, desc="phase 2", unit="split")):
        perm = _perm_from_seed(int(seed), len(group_keys))
        variance_scores[rank] = _variance_score(perm, train_groups, val_groups, group_matrix, weights)
        overlap_scores[rank] = _clip_overlap(perm, train_groups, group_keys, group_sources)
        combined[rank] = variance_scores[rank] + overlap_scores[rank]
    print(f"  sampled source overlap: min {overlap_scores.min():.4f}  "
          f"median {np.median(overlap_scores):.4f}  max {overlap_scores.max():.4f}")

    candidates: list[dict] = []
    for rank in np.argsort(combined)[:phase3]:
        seed = int(phase2_seeds[int(rank)])
        train, val, test = _expand_perm(
            _perm_from_seed(seed, len(group_keys)), train_groups, val_groups, group_keys, group_rows
        )
        candidates.append({
            "seed": seed, "train": train, "val": val, "test": test,
            "variance_score": float(variance_scores[int(rank)]),
            "difficulty_score": float(difficulty_scores[seed]),
            "clip_overlap": float(overlap_scores[int(rank)]),
        })

    if refine_pool:
        pool = candidates[:refine_pool]
        print(f"Phase 3: refining the best {len(pool)} candidates by partition-preserving swaps "
              f"({len(REFINE_WEIGHT_STAGES)} weight stages, {refine_sweeps} sweeps each)")
        before = float(np.mean([c["clip_overlap"] for c in pool]))
        rng = np.random.default_rng(RANDOM_SEED)
        for candidate in tqdm(pool, desc="phase 3", unit="split"):
            refined, trajectory = _refine_by_swaps(
                _perm_from_seed(candidate["seed"], len(group_keys)),
                train_groups, val_groups, group_keys, group_sources, group_difficulty,
                group_matrix, weights, global_difficulty, refine_sweeps, rng,
            )
            train, val, test = _expand_perm(refined, train_groups, val_groups, group_keys, group_rows)
            final = trajectory[-1]
            candidate.update({
                "train": train, "val": val, "test": test,
                "variance_score": final["variance_score"],
                "difficulty_score": final["difficulty_score"],
                "clip_overlap": final["clip_overlap"],
                "refinement": trajectory,
            })
        after = float(np.mean([c["clip_overlap"] for c in pool]))
        print(f"  mean source overlap {before:.4f} -> {after:.4f} ({after - before:+.4f})")
        candidates = sorted(pool, key=lambda item: item["variance_score"] + item["clip_overlap"])

    selected = _greedy_heldout_coverage(candidates, n_splits, len(examples))
    if len(selected) < n_splits:
        # The coverage step can only choose from the candidates it is given, so
        # a refinement pool smaller than the requested replicate count silently
        # yields fewer splits -- and train_selector then refuses to aggregate.
        raise ValueError(
            f"Only {len(selected)} split(s) could be selected from a pool of {len(candidates)}; "
            f"{n_splits} were requested. Raise --refine-pool (or --phase3 when refinement is "
            "disabled) to at least the number of splits."
        )
    paper_compatible = (
        n_splits == N_PAPER_SPLITS
        and phase1 == PAPER_PHASE1 and phase2 == PAPER_PHASE2 and phase3 == PAPER_PHASE3
        and refine_pool == PAPER_REFINE_POOL and refine_sweeps == PAPER_REFINE_SWEEPS
    )
    common_meta = {
        "paper_compatible": paper_compatible,
        "dataset": dataset,
        "grouping_basis": grouping_basis,
        "split_unit": "audio_composition",
        "ratios": {"train": RATIOS[0], "val": RATIOS[1], "test": RATIOS[2]},
        "n_examples": len(examples),
        "n_compositions": len(group_keys),
        "n_scoring_specs": len(scoring_specs),
        "scoring_specs": scoring_specs,
        "search": {"phase1": phase1, "phase2": phase2, "phase3": phase3,
                   "refine_pool": refine_pool, "refine_sweeps": refine_sweeps},
        "source_overlap_definition": "Jaccard(train source IDs, validation union test source IDs)",
    }
    if not paper_compatible:
        common_meta["warning"] = (
            "Search budget differs from the paper default; this is a smoke-test artifact, "
            "not a paper-compatible split."
        )
    return selected, common_meta


# ===========================================================================
# Diagnostics and verification
# ===========================================================================

def _keys(examples: list[dict], indices: np.ndarray) -> list[str]:
    return [example_key(examples[int(index)]) for index in indices]


def _overlap_from_indices(indices: dict[str, np.ndarray], group_ids: list[tuple],
                          group_sources: dict[tuple, frozenset[str]]) -> float:
    """Recompute overlap from final row indices, independent of the search state."""
    train_sources: set[str] = set()
    held_sources: set[str] = set()
    for index in indices["train"]:
        train_sources.update(group_sources[group_ids[int(index)]])
    for part in ("val", "test"):
        for index in indices[part]:
            held_sources.update(group_sources[group_ids[int(index)]])
    return len(train_sources & held_sources) / max(1, len(train_sources | held_sources))


def print_composition_report(examples: list[dict], dataset: str) -> None:
    """Print the structural inventory behind the paper's overlap claim.

    This is what turns "we minimise overlap instead of eliminating it" from an
    assertion into something a reader can check: if a source recording appears
    in many compositions, it cannot be confined to one side of a 70/15/15
    boundary, and the degree histogram shows exactly how many are in that
    position.
    """
    group_ids, group_rows, group_sources, basis = build_composition_groups(examples, dataset)
    group_keys = list(group_rows)
    train_groups, val_groups = _partition_counts(len(group_keys))
    sizes = np.array([len(rows) for rows in group_rows.values()])

    source_to_groups: dict[str, set[int]] = defaultdict(set)
    for index, key in enumerate(group_keys):
        for source in group_sources[key]:
            source_to_groups[source].add(index)
    degrees = Counter(len(groups) for groups in source_to_groups.values())
    total_sources = len(source_to_groups)
    singletons = degrees.get(1, 0)

    print("\nComposition inventory")
    print(f"  grouping basis          {basis}")
    print(f"  examples                {len(examples):,}")
    print(f"  compositions            {len(group_keys):,} "
          f"(questions per composition: min {sizes.min()}, max {sizes.max()}, mean {sizes.mean():.2f})")
    print(f"  composition partition   {train_groups}/{val_groups}/"
          f"{len(group_keys) - train_groups - val_groups} train/val/test")
    print(f"  distinct source IDs     {total_sources:,}")
    print(f"  sources in one group    {singletons:,} "
          f"({singletons / max(1, total_sources):.1%}; these can always be kept on one side)")
    print(f"  sources in many groups  {total_sources - singletons:,} "
          f"({(total_sources - singletons) / max(1, total_sources):.1%}; "
          "each is shared whenever its compositions straddle the boundary)")
    histogram = "  ".join(f"{degree}:{count}" for degree, count in sorted(degrees.items())[:12])
    print(f"  source degree histogram {histogram}")
    if total_sources == singletons:
        print("  -> every source belongs to exactly one composition, so zero train/held-out "
              "overlap is attainable and the search should reach it.")
    else:
        print("  -> re-used sources make zero train/held-out overlap structurally impossible "
              "at this ratio; the search minimises it as a penalty instead.")


def verify_splits(cfg: RunConfig, examples: list[dict], dataset: str) -> bool:
    """Check every split on disk for composition-level integrity and full coverage.

    Three failure modes are worth catching separately: a composition whose
    questions were scattered across partitions (leakage), an example that is
    missing or duplicated (a broken partition), and a split built against a
    different manifest (stale keys). Returns True when every split passes.
    """
    names = list_splits(cfg.splits_dir)
    if not names:
        print(f"No split files found in {cfg.splits_dir}")
        return False

    group_ids, _, group_sources, _ = build_composition_groups(examples, dataset)
    key_to_row = {example_key(item): row for row, item in enumerate(examples)}
    all_ok = True

    print(f"Verifying {len(names)} split file(s) in {cfg.splits_dir}")
    for name in names:
        split = load_split(cfg.splits_dir, name)
        problems: list[str] = []
        assigned: dict[int, str] = {}
        unknown = 0
        for part in ("train", "val", "test"):
            for key in split[part]:
                row = key_to_row.get(key)
                if row is None:
                    unknown += 1
                    continue
                assigned[row] = part
        if unknown:
            problems.append(f"{unknown} key(s) not present in the current manifest")
        if len(assigned) != len(examples):
            problems.append(f"covers {len(assigned):,} of {len(examples):,} examples")

        composition_parts: dict[tuple, set[str]] = defaultdict(set)
        for row, part in assigned.items():
            composition_parts[group_ids[row]].add(part)
        straddling = sum(1 for parts in composition_parts.values() if len(parts) > 1)
        if straddling:
            problems.append(f"{straddling} composition(s) straddle a partition boundary")

        indices = {part: np.array([r for r, p in assigned.items() if p == part], dtype=np.int64)
                   for part in ("train", "val", "test")}
        overlap = _overlap_from_indices(indices, group_ids, group_sources)
        meta = split.get("meta", {})
        recorded = meta.get("clip_overlap")
        if recorded is not None and abs(recorded - overlap) > 1e-9:
            problems.append(f"recorded overlap {recorded:.4f} != recomputed {overlap:.4f}")

        sizes = {part: len(values) for part, values in indices.items()}
        status = "OK" if not problems else "FAILED"
        flag = "" if meta.get("paper_compatible", True) else "  [non-paper search budget]"
        print(f"  {name}: {status}  sizes={sizes}  source overlap={overlap:.4f}{flag}")
        for problem in problems:
            print(f"      - {problem}")
        all_ok = all_ok and not problems

    print("All splits verified." if all_ok else "One or more splits failed verification; regenerate them.")
    return all_ok


def write_random_baselines(cfg: RunConfig, examples: list[dict], dataset: str,
                           n_splits: int, common_meta: dict) -> None:
    """Write composition-level random splits as a comparison point.

    These are the control for the balanced search: same grouping, same ratios,
    no balancing objective. Comparing their source overlap and balance scores
    against the selected splits shows how much of the final quality came from
    the search rather than from grouping alone.
    """
    group_ids, group_rows, group_sources, _ = build_composition_groups(examples, dataset)
    group_keys = list(group_rows)
    train_groups, val_groups = _partition_counts(len(group_keys))
    rng = np.random.default_rng(RANDOM_SEED)

    for number in range(1, n_splits + 1):
        perm = rng.permutation(len(group_keys))
        train, val, test = _expand_perm(perm, train_groups, val_groups, group_keys, group_rows)
        indices = {"train": train, "val": val, "test": test}
        overlap = _overlap_from_indices(indices, group_ids, group_sources)
        meta = dict(common_meta)
        meta.update({
            "type": "random_baseline",
            "paper_compatible": False,
            "note": "Unbalanced composition-level control split; not a paper result.",
            "clip_overlap": overlap,
            "sizes": {part: int(len(values)) for part, values in indices.items()},
        })
        path = save_split(
            cfg.splits_dir, f"random_split_{number}",
            _keys(examples, train), _keys(examples, val), _keys(examples, test), metadata=meta,
        )
        print(f"random_split_{number}: source overlap={overlap:.4f} -> {path}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", "-m", required=True, choices=["qwen2", "af3"],
                        help="Model whose oracle defines difficulty and branch balance")
    parser.add_argument("--dataset", "-d", required=True, choices=list(AH_DATASETS))
    parser.add_argument("--alpha", "-a", type=float, default=PAPER_ALPHA)
    parser.add_argument("--prompt", default="constrained", choices=sorted(PROMPTS))
    parser.add_argument("--candidate-spec", action="append", default=None, metavar="TYPE[:SETTING]",
                        help="Restrict split balancing to a documented candidate pool; repeat per branch")
    parser.add_argument("--num-splits", type=int, default=N_PAPER_SPLITS)
    parser.add_argument("--phase1", type=int, default=PAPER_PHASE1)
    parser.add_argument("--phase2", type=int, default=PAPER_PHASE2)
    parser.add_argument("--phase3", type=int, default=PAPER_PHASE3)
    parser.add_argument("--refine-pool", type=int, default=PAPER_REFINE_POOL)
    parser.add_argument("--refine-sweeps", type=int, default=PAPER_REFINE_SWEEPS)
    parser.add_argument("--no-refine", action="store_true",
                        help="Skip phase 3; sampling alone cannot reach low source overlap")
    parser.add_argument("--random-baselines", action="store_true",
                        help="Also write unbalanced composition-level control splits")
    parser.add_argument("--report", action="store_true",
                        help="Print the composition and source inventory, then continue")
    parser.add_argument("--verify", action="store_true",
                        help="Check the splits already on disk and exit without regenerating")
    args = parser.parse_args()
    if args.num_splits < 1:
        parser.error("--num-splits must be positive")

    cfg = RunConfig(model=args.model, dataset=args.dataset, alpha=args.alpha, prompt_name=args.prompt)
    # Split construction reads only the manifest's structure, so audio files
    # do not need to be present on this machine.
    examples = load_examples(cfg, check_files=False)

    if args.verify:
        raise SystemExit(0 if verify_splits(cfg, examples, args.dataset) else 1)
    if args.report:
        print_composition_report(examples, args.dataset)

    oracle = load_oracle(cfg.oracle_file)
    selected, common_meta = create_paper_splits(
        examples, oracle, args.dataset,
        candidate_specs=args.candidate_spec,
        phase1=args.phase1, phase2=args.phase2, phase3=args.phase3,
        refine_pool=0 if args.no_refine else args.refine_pool,
        refine_sweeps=args.refine_sweeps, n_splits=args.num_splits,
    )

    group_ids, _, group_sources, _ = build_composition_groups(examples, args.dataset)
    for number, candidate in enumerate(selected, start=1):
        indices = {part: candidate[part] for part in ("train", "val", "test")}
        overlap = _overlap_from_indices(indices, group_ids, group_sources)
        meta = dict(common_meta)
        meta.update({
            "type": "balanced",
            "seed": candidate["seed"],
            "variance_score": candidate["variance_score"],
            "difficulty_score": candidate["difficulty_score"],
            "clip_overlap": overlap,
            "sizes": {part: int(len(values)) for part, values in indices.items()},
            "coverage": "complete and composition-disjoint",
        })
        if "refinement" in candidate:
            meta["refinement"] = candidate["refinement"]
        path = save_split(
            cfg.splits_dir, f"balanced_split_{number}",
            _keys(examples, indices["train"]), _keys(examples, indices["val"]),
            _keys(examples, indices["test"]), metadata=meta,
        )
        print(f"balanced_split_{number}: sizes={meta['sizes']} source overlap={overlap:.4f} -> {path}")

    if args.random_baselines:
        print("\nRandom composition-level baselines (control, not paper results)")
        write_random_baselines(cfg, examples, args.dataset, args.num_splits, common_meta)

    print(f"\nVerify what was written with:\n  python -m scripts.prepare_splits "
          f"--model {args.model} --dataset {args.dataset} --verify")


if __name__ == "__main__":
    main()
