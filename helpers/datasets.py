"""Dataset loading, validation, and split I/O.

A dataset item is a plain dict::

    {"path": <audio path>, "Q": <question>, "text": "yes" | "no", ...}

Input is validated before a multi-billion-parameter model is loaded, so a
malformed manifest or a missing audio file cannot silently produce a partial
evaluation. Every dataset in this release is binary yes/no audio QA, and the
validator enforces exactly that: a record whose label is anything other than
``yes`` or ``no`` is reported, not skipped.

Optional provenance fields
--------------------------
Two extra fields carry the composition structure the AH selector splits are
built on (paper Section IV, "we partition at the audio-composition level"):

    ``composition_id``    stable identifier for the background-plus-events
                          mixture an example was rendered from
    ``source_clip_ids``   the individual source recordings that went into it

They are optional for decoding, which only needs path/Q/text, but
``scripts/prepare_splits.py`` requires them for AH Existence unless the raw
BEAF ``background@event_a@event_b@event_c`` filename convention is preserved.
When present they are validated here, so a conversion error surfaces at load
time rather than halfway through a split search.

Splits are stored as ``{"train": [key, ...], "val": [...], "test": [...],
"meta": {...}}`` under a dataset's ``splits_dir``, one file per replicate
(``balanced_split_1.json`` .. ``balanced_split_5.json``), grouped so examples
sharing an audio composition never straddle a partition boundary. Membership
is stored by :func:`example_key` rather than by row index, so a split stays
valid regardless of the order in which oracle and cache files enumerate
examples.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from helpers.config import RunConfig

MAX_REPORTED_ISSUES = 10

REQUIRED_FIELDS = ("path", "Q", "text")
VALID_LABELS = {"yes", "no"}


def _validate_item(index: int, item: object, issues: list[str], *, check_files: bool) -> None:
    """Append every schema problem found in one manifest row."""
    if not isinstance(item, dict):
        issues.append(f"item {index}: expected an object, got {type(item).__name__}")
        return

    missing = [field for field in REQUIRED_FIELDS if field not in item]
    if missing:
        issues.append(f"item {index}: missing {', '.join(missing)}")
        return

    if not isinstance(item["path"], str) or not item["path"].strip():
        issues.append(f"item {index}: path must be a non-empty string")
    elif check_files and not Path(item["path"]).is_file():
        issues.append(f"item {index}: audio file not found: {item['path']}")

    if not isinstance(item["Q"], str) or not item["Q"].strip():
        issues.append(f"item {index}: Q must be a non-empty string")

    if not isinstance(item["text"], str) or item["text"].strip().lower() not in VALID_LABELS:
        issues.append(f"item {index}: text must be exactly 'yes' or 'no', got {item.get('text')!r}")

    # Optional provenance, validated only when supplied.
    composition_id = item.get("composition_id")
    if composition_id is not None and (not isinstance(composition_id, str) or not composition_id.strip()):
        issues.append(f"item {index}: composition_id must be a non-empty string when present")

    source_clip_ids = item.get("source_clip_ids")
    if source_clip_ids is not None:
        if not isinstance(source_clip_ids, (list, tuple)) or not source_clip_ids:
            issues.append(f"item {index}: source_clip_ids must be a non-empty list when present")
        elif not all(isinstance(clip, str) and clip.strip() for clip in source_clip_ids):
            issues.append(f"item {index}: source_clip_ids must contain non-empty strings")


def load_examples(cfg: RunConfig, *, check_files: bool = True) -> list[dict]:
    """Read and validate one dataset manifest.

    ``check_files`` exists for offline analysis passes (split construction,
    table regeneration) that only need the manifest's structure, not the audio
    itself. Decoding and caching always leave it on.
    """
    path = cfg.dataset_json
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Place the licensed {cfg.dataset} data there first "
            "(see the Setup section of README.md)."
        )
    try:
        examples = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset file is not valid JSON: {path}\n{exc}") from exc
    if not isinstance(examples, list):
        raise ValueError(f"Dataset file must contain a JSON list, not {type(examples).__name__}: {path}")
    if not examples:
        raise ValueError(f"Dataset file is empty: {path}")

    issues: list[str] = []
    for index, item in enumerate(examples):
        _validate_item(index, item, issues, check_files=check_files)
        if len(issues) >= MAX_REPORTED_ISSUES:
            break

    if issues:
        truncated = issues[:MAX_REPORTED_ISSUES]
        suffix = f" (showing the first {MAX_REPORTED_ISSUES})" if len(issues) >= MAX_REPORTED_ISSUES else ""
        raise ValueError(f"Invalid dataset manifest{suffix}:\n- " + "\n- ".join(truncated))
    return examples


def dataset_summary(cfg: RunConfig, examples: list[dict]) -> dict:
    """Counts a reviewer wants before trusting an accuracy number.

    A yes/no benchmark that is not close to balanced makes raw accuracy hard
    to read, and the paper's affirmative-bias measurement (Section V-A) is
    defined against a balanced 50% reference, so the label split is reported
    up front rather than assumed.
    """
    labels = Counter(item["text"].strip().lower() for item in examples)
    total = len(examples)
    yes_rate = labels["yes"] / total if total else 0.0
    return {
        "dataset": cfg.dataset,
        "split": cfg.split,
        "task": cfg.spec.task,
        "examples": total,
        "paper_examples": cfg.spec.paper_examples,
        "unique_audio_files": len({item["path"] for item in examples}),
        "yes": labels["yes"],
        "no": labels["no"],
        "yes_rate": yes_rate,
        "has_composition_metadata": all(item.get("composition_id") for item in examples),
        "has_source_clip_ids": all(item.get("source_clip_ids") for item in examples),
    }


def print_dataset_summary(summary: dict) -> None:
    scope = summary["dataset"] + (f"/{summary['split']}" if summary["split"] else "")
    note = ""
    if summary["split"] is None and summary["examples"] != summary["paper_examples"]:
        note = f"  (paper reports {summary['paper_examples']:,} for this setting)"
    print(
        f"{scope}: {summary['examples']:,} examples, "
        f"{summary['unique_audio_files']:,} unique audio files, "
        f"yes={summary['yes']:,} no={summary['no']:,} "
        f"(yes rate {summary['yes_rate']:.1%}){note}"
    )


def example_key(item: dict) -> str:
    """Stable per-example key used to align oracle labels, caches, and splits.

    An audio file appears under several questions, and a question recurs
    across audio files, so neither alone identifies an example; the pair does.
    """
    return f"{item['path']}||{item['Q']}"


def load_split(splits_dir: Path, split_name: str) -> dict:
    """Load one ``balanced_split_N.json``.

    Returns the raw payload: ``train``/``val``/``test`` key lists plus the
    ``meta`` block written by ``scripts/prepare_splits.py``, which records the
    grouping basis, search budget, seed, and measured source-clip overlap.
    """
    path = splits_dir / f"{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\nRun `python -m scripts.prepare_splits` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [part for part in ("train", "val", "test") if part not in payload]
    if missing:
        raise ValueError(f"Split file {path} is missing partition(s): {', '.join(missing)}")

    seen: set[str] = set()
    for part in ("train", "val", "test"):
        keys = payload[part]
        if not isinstance(keys, list):
            raise ValueError(f"Split file {path}: '{part}' must be a list of example keys")
        overlap = seen & set(keys)
        if overlap:
            raise ValueError(
                f"Split file {path}: {len(overlap)} example(s) appear in more than one partition"
            )
        seen.update(keys)
    return payload


def save_split(
    splits_dir: Path,
    split_name: str,
    train: list[str],
    val: list[str],
    test: list[str],
    *,
    metadata: Optional[dict] = None,
) -> Path:
    """Write one split file, with provenance metadata when supplied."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    path = splits_dir / f"{split_name}.json"
    payload: dict = {"train": train, "val": val, "test": test}
    if metadata is not None:
        payload["meta"] = metadata
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def list_splits(splits_dir: Path, prefix: str = "balanced_split_") -> list[str]:
    """Names of the split files on disk, ordered by their replicate number.

    Plain lexicographic ordering would put ``balanced_split_10`` before
    ``balanced_split_2``; the numeric sort keeps replicate order stable if a
    future run produces more than nine.
    """
    if not splits_dir.exists():
        return []
    names = [p.stem for p in splits_dir.glob(f"{prefix}*.json")]

    def order(name: str) -> tuple[int, str]:
        suffix = name[len(prefix):]
        return (int(suffix), name) if suffix.isdigit() else (1 << 30, name)

    return sorted(names, key=order)


def split_coverage(split: dict, keys: Iterable[str]) -> dict:
    """Compare a split's membership against the keys actually available.

    A split built from one manifest and applied to a partially cached or
    partially decoded set silently shrinks the evaluation, so the mismatch is
    surfaced as counts the caller can assert on.
    """
    available = set(keys)
    covered = {part: [k for k in split[part] if k in available] for part in ("train", "val", "test")}
    assigned = set().union(*(set(v) for v in covered.values())) if covered else set()
    return {
        "sizes": {part: len(values) for part, values in covered.items()},
        "requested": {part: len(split[part]) for part in ("train", "val", "test")},
        "missing": {part: len(split[part]) - len(covered[part]) for part in ("train", "val", "test")},
        "unassigned": len(available - assigned),
    }
