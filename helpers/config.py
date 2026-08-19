"""
Central configuration: model registry, dataset registry, prompts, and the
local runtime directory layout.

Every experiment in the paper is identified by the same four coordinates --
model, dataset, prompt, and contrastive strength alpha -- so ``RunConfig``
below is the single object that turns those coordinates into concrete file
locations. Nothing else in the repository hard-codes a path.

``data/``, ``runs/``, ``cache/``, ``checkpoints/``, and ``outputs/`` are
created on demand and ignored by Git (see ``.gitignore``). This module only
decides *where* those directories live; populating them is the job of the
stages in ``scripts/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"                 # licensed datasets you provide
RUNS_DIR = PROJECT_ROOT / "runs"                 # decoding run outputs
CACHE_DIR = PROJECT_ROOT / "cache"               # hidden-state / audio-feature caches
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"   # local selector checkpoints
OUTPUTS_DIR = PROJECT_ROOT / "outputs"           # regenerated tables and figures

# Paper Section IV: Qwen2-Audio pairs a Whisper-large audio encoder with a
# Qwen-7B backbone; AF3 pairs the AF-Whisper encoder with a Qwen2.5-7B
# backbone. Both are ~7B-parameter LALMs evaluated at 16-bit precision.
MODEL_IDS = {
    "qwen2": "Qwen/Qwen2-Audio-7B-Instruct",
    "af3": "nvidia/audio-flamingo-3-hf",
}

MODEL_LABELS = {
    "qwen2": "Qwen2-Audio-7B-Instruct",
    "af3": "Audio Flamingo 3",
}

# Weight dtype per model, matching the reported runs. AF3 is released in
# bfloat16; Qwen2-Audio is evaluated in float16.
MODEL_DTYPES = {"qwen2": "float16", "af3": "bfloat16"}

# ---------------------------------------------------------------------------
# Prompts (paper Section V-A, Table II)
#
# Prior work (AAD, arXiv:2506.07233) prefixes an attention-directing
# instruction. This paper appends an explicit one-word yes/no constraint. The
# constraint alone raises Qwen2 AH Existence accuracy from 56.9% to 67.9%
# before any contrastive decoding, by cutting the model's affirmative bias
# from +40.4 points to +21.0 points. Both prompts are kept selectable so that
# Table II can be regenerated end to end rather than quoted.
# ---------------------------------------------------------------------------

AAD_PROMPT = "Focus on the given audio and answer the following question."
CONSTRAINED_PROMPT = (
    "Focus on the given audio and answer the following question with "
    "exactly one word: yes or no."
)

PROMPTS = {
    "aad": AAD_PROMPT,
    "constrained": CONSTRAINED_PROMPT,
}

DEFAULT_PROMPT_NAME = "constrained"

# The prompt is joined to the question without an added separator, exactly as
# in the reported runs; dataset questions already begin with a space or their
# own capitalisation.
PROMPT_JOINER = ""

# ---------------------------------------------------------------------------
# Decoding defaults
# ---------------------------------------------------------------------------

# Paper Section V-B sweeps alpha over [0, 2] and fixes alpha = 1.0 for every
# main experiment: past 1.0 the correction term outweighs the expert branch.
# Decoding accepts the whole swept range so the sweep is runnable directly;
# ALPHA_RANGE is the guard, PAPER_ALPHA the reported operating point.
ALPHA_RANGE = (0.0, 2.0)
PAPER_ALPHA = 1.0

DEFAULT_BATCH_SIZE = 16
# The constrained prompt makes the answer a single token, so one new token is
# all that is scored. Raise it only to inspect free-form continuations.
DEFAULT_MAX_NEW_TOKENS = 1
TOP_K_LOGGING = 10
CHECKPOINT_INTERVAL = 10  # save a resumable checkpoint every N batches


def resolve_prompt(prompt_name: str) -> str:
    """Map a prompt name from the CLI onto its literal text."""
    try:
        return PROMPTS[prompt_name]
    except KeyError:
        raise ValueError(
            f"prompt={prompt_name!r} must be one of {sorted(PROMPTS)}"
        ) from None


@dataclass(frozen=True)
class DatasetSpec:
    """Static facts about one evaluation setting (paper Table I).

    ``file_stem`` resolves to ``data/{name}/{file_stem}[_{split}].json``.
    ``has_splits`` is False for the AH benchmarks, which ship as a single file
    and are partitioned by ``scripts/prepare_splits.py``, and True for
    Clotho-AQA, which has official train/validation/test files.
    """

    name: str
    file_stem: str
    has_splits: bool
    task: str            # short task label used in regenerated tables
    paper_examples: int  # yes/no subset size reported in Table I
    source: str          # upstream provenance of the audio


DATASETS = {
    "ah_existence": DatasetSpec(
        "ah_existence", "ah_existence", has_splits=False,
        task="Existence Y/N", paper_examples=10_800,
        source="Audio Hallucination benchmark, synthetic composites",
    ),
    "ah_order": DatasetSpec(
        "ah_order", "ah_order", has_splits=False,
        task="Order Y/N", paper_examples=3_078,
        source="Audio Hallucination benchmark, built on CompA",
    ),
    "ah_attribute": DatasetSpec(
        "ah_attribute", "ah_attribute", has_splits=False,
        task="Attribute Y/N", paper_examples=1_599,
        source="Audio Hallucination benchmark, built on CompA",
    ),
    "clotho_aqa": DatasetSpec(
        "clotho_aqa", "clotho_aqa", has_splits=True,
        task="Mixed Y/N", paper_examples=7_959,
        source="Clotho-AQA over FreeSound clips",
    ),
}

# The AH benchmarks are the ones partitioned by the composition-aware split
# search; Clotho-AQA uses its official partition instead.
AH_DATASETS = tuple(name for name, spec in DATASETS.items() if not spec.has_splits)

# Number of balanced replicate splits the paper averages AH results over.
N_PAPER_SPLITS = 5


@dataclass
class RunConfig:
    """Everything needed to place, run, and locate the output of one evaluation."""

    model: str
    dataset: str
    alpha: float = PAPER_ALPHA
    split: Optional[str] = None       # "train"/"val"/"test" for clotho_aqa, else None
    prompt_name: str = DEFAULT_PROMPT_NAME
    prompt: Optional[str] = None      # verbatim override; wins over prompt_name
    max_new_tokens: Optional[int] = None

    def __post_init__(self):
        if self.model not in MODEL_IDS:
            raise ValueError(f"model={self.model!r} must be one of {list(MODEL_IDS)}")
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset={self.dataset!r} must be one of {list(DATASETS)}")
        low, high = ALPHA_RANGE
        if not (low <= self.alpha <= high):
            raise ValueError(f"alpha={self.alpha} must lie in [{low}, {high}]")
        if self.prompt is None and self.prompt_name not in PROMPTS:
            raise ValueError(f"prompt_name={self.prompt_name!r} must be one of {sorted(PROMPTS)}")
        spec = DATASETS[self.dataset]
        if spec.has_splits and self.split not in ("train", "val", "test"):
            raise ValueError(f"dataset={self.dataset!r} requires split in ('train', 'val', 'test')")
        if not spec.has_splits and self.split is not None:
            raise ValueError(f"dataset={self.dataset!r} has no splits; leave split=None")

    # -- identity ---------------------------------------------------------

    @property
    def spec(self) -> DatasetSpec:
        return DATASETS[self.dataset]

    @property
    def model_id(self) -> str:
        return MODEL_IDS[self.model]

    @property
    def model_label(self) -> str:
        return MODEL_LABELS[self.model]

    @property
    def run_prompt(self) -> str:
        """The literal instruction prefixed to every question in this run."""
        return self.prompt if self.prompt is not None else resolve_prompt(self.prompt_name)

    @property
    def prompt_tag(self) -> str:
        """Short name recorded in results and used in non-default output paths."""
        if self.prompt is not None:
            for name, text in PROMPTS.items():
                if text == self.prompt:
                    return name
            return "custom"
        return self.prompt_name

    @property
    def new_tokens(self) -> int:
        return self.max_new_tokens if self.max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS

    def describe(self) -> str:
        parts = [self.model_label, self.dataset]
        if self.split:
            parts.append(self.split)
        parts += [f"alpha={self.alpha}", f"prompt={self.prompt_tag}"]
        return " | ".join(parts)

    # -- filesystem layout -------------------------------------------------
    #
    # The constrained prompt is the paper default and keeps the short path
    # runs/{model}/{dataset}[/{split}]/{alpha}. Any other prompt is written
    # under a prompt-tagged sibling so a Table II comparison sweep cannot
    # overwrite the main results.

    @property
    def dataset_json(self) -> Path:
        stem = self.spec.file_stem if self.split is None else f"{self.spec.file_stem}_{self.split}"
        return DATA_DIR / self.dataset / f"{stem}.json"

    def _scoped(self, root: Path, *, with_alpha: bool) -> Path:
        parts = [self.model, self.dataset]
        if self.split:
            parts.append(self.split)
        if self.prompt_tag != DEFAULT_PROMPT_NAME:
            parts.append(f"prompt_{self.prompt_tag}")
        if with_alpha:
            parts.append(str(self.alpha))
        return root.joinpath(*parts)

    @property
    def results_dir(self) -> Path:
        """Where decoding writes eval_*.json.gz for this run."""
        return self._scoped(RUNS_DIR, with_alpha=True)

    @property
    def cache_dir(self) -> Path:
        """Where the clean-branch feature cache lives (alpha-independent)."""
        return self._scoped(CACHE_DIR, with_alpha=False)

    @property
    def oracle_file(self) -> Path:
        """Per-example multi-hot branch correctness, built from results_dir.

        Clotho-AQA partitions share one directory and are distinguished by the
        filename stem, because the selector loads all three together.
        """
        stem = "oracle" if self.split is None else f"oracle_{self.split}"
        parts = [self.model, self.dataset]
        if self.prompt_tag != DEFAULT_PROMPT_NAME:
            parts.append(f"prompt_{self.prompt_tag}")
        return CACHE_DIR.joinpath(*parts) / f"{stem}.json.gz"

    @property
    def splits_dir(self) -> Path:
        """Where scripts/prepare_splits.py writes balanced_split_*.json."""
        return DATA_DIR / self.dataset / "splits"
