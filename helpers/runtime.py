"""
Reproducibility and resume utilities shared by the decoding runners and the
hidden-state cache: seeding, dtype resolution, checkpoint I/O, and the output
filename convention.

A full perturbation sweep is 105 branches over up to 10,800 examples per
model, which is long enough that a run will be interrupted at some point.
Every branch therefore writes a resumable checkpoint every
``CHECKPOINT_INTERVAL`` batches and skips itself entirely if its final output
already exists, so re-running a sweep costs only the work that was lost.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

DEFAULT_SEED = 42


def require_cuda() -> None:
    """Fail early with an actionable message when a model runner has no GPU."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA-enabled PyTorch installation and a visible NVIDIA GPU are required "
            "for decoding and hidden-state caching. Install the PyTorch build for your "
            "CUDA version, then confirm that torch.cuda.is_available() is True."
        )


def set_random_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy, and Torch, and pin cuDNN to deterministic kernels.

    The stochastic perturbations (see ``perturbations.STOCHASTIC_TYPES``) draw
    from NumPy's global state, so this call plus a fixed manifest order is
    what makes a branch sweep reproducible. Returns the seed for logging.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def resolve_dtype(model: str):
    """Torch dtype for a model key, from the registry in helpers.config.

    Kept here rather than inline in each runner so the decoding path and the
    caching path cannot drift apart: the selector's features must come from
    the same numerical forward pass the decoder scored.
    """
    import torch

    from helpers.config import MODEL_DTYPES

    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    try:
        return dtypes[MODEL_DTYPES[model]]
    except KeyError:
        raise ValueError(f"No dtype registered for model {model!r}") from None


# ---------------------------------------------------------------------------
# Output and checkpoint naming
#
# The enclosing directory already encodes model, dataset, split, prompt, and
# alpha (see RunConfig.results_dir); the filename identifies the branch. Alpha
# is repeated in the filename so a file remains self-describing if it is moved
# or attached to a bug report.
# ---------------------------------------------------------------------------

def _branch_stem(perturbation_type: str, perturbation_setting: Optional[str], alpha: float) -> str:
    name = perturbation_type.lower()
    if perturbation_setting:
        name = f"{name}_{perturbation_setting.lower()}"
    return f"{name}_alpha_{alpha}"


def output_filename(results_dir: Path, perturbation_type: str,
                    perturbation_setting: Optional[str], alpha: float) -> Path:
    return results_dir / f"eval_{_branch_stem(perturbation_type, perturbation_setting, alpha)}.json.gz"


def checkpoint_filename(results_dir: Path, perturbation_type: str,
                        perturbation_setting: Optional[str], alpha: float) -> Path:
    return results_dir / f"checkpoint_{_branch_stem(perturbation_type, perturbation_setting, alpha)}.json"


def results_exist(results_dir: Path, perturbation_type: str,
                  perturbation_setting: Optional[str], alpha: float) -> bool:
    path = output_filename(results_dir, perturbation_type, perturbation_setting, alpha)
    if path.exists():
        print(f"Results already exist: {path} (delete it to re-run)")
        return True
    return False


def load_checkpoint(results_dir: Path, perturbation_type: str,
                    perturbation_setting: Optional[str], alpha: float) -> tuple[list[dict], int]:
    """Return (raw sample dicts, last completed batch index), or ``([], -1)``."""
    path = checkpoint_filename(results_dir, perturbation_type, perturbation_setting, alpha)
    if not path.exists():
        return [], -1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results", [])
        last_batch_idx = data.get("last_batch_idx", -1)
        print(f"Resuming from checkpoint: {path} (batch {last_batch_idx + 1}, {len(results)} samples so far)")
        return results, last_batch_idx
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        print(f"Checkpoint at {path} is unreadable ({exc}); starting this branch fresh.")
        return [], -1


def save_checkpoint(results_dir: Path, perturbation_type: str, perturbation_setting: Optional[str],
                    alpha: float, results: list[dict], last_batch_idx: int) -> None:
    """Persist progress atomically, so a kill during the write is recoverable."""
    path = checkpoint_filename(results_dir, perturbation_type, perturbation_setting, alpha)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"last_batch_idx": last_batch_idx, "results": results}, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def delete_checkpoint(results_dir: Path, perturbation_type: str,
                      perturbation_setting: Optional[str], alpha: float) -> None:
    path = checkpoint_filename(results_dir, perturbation_type, perturbation_setting, alpha)
    if path.exists():
        path.unlink()
