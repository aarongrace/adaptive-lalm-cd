"""
Result schema, answer extraction, and scoring shared by both decoding runners.

Every dataset in this release is binary yes/no audio QA, so a sample's
correctness reduces to comparing one extracted token against the ground truth.
Two extraction paths are supported:

  - **step-0 logit comparison** -- the paper's primary method. Under the
    constrained one-word prompt the model's very first generated token is
    yes/no, so comparing the (possibly contrastive-modified) logits for the
    ``Yes``/``yes`` variants against ``No``/``no`` is exact. It does not
    depend on decoding a token and re-parsing it as text, and it keeps scoring
    identical across the clean and contrastive branches.
  - **text extraction** -- a fallback for prompts or models where the first
    generated token is not reliably yes/no. The open AAD prompt (Table II)
    is exactly that case: it often produces a full sentence.

Beyond accuracy, this module computes the **affirmative bias** the paper uses
to explain *why* the constrained prompt helps (Section V-A): the predicted-yes
rate minus 50%, on a balanced benchmark. Under the AAD prompt Qwen2 answers
"yes" to 90.4% of AH Existence questions (+40.4 points); the constrained
prompt cuts that to +21.0, and no-audio contrastive decoding to +1.8.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

YES_TOKENS = ("Yes", "yes")
NO_TOKENS = ("No", "no")


@dataclass
class SampleResult:
    """One (audio, question) evaluation under one negative branch."""

    audio_file: str
    question: str
    ground_truth: str
    model_response: str
    extracted_answer: str
    is_correct: bool
    perturbation_type: str
    perturbation_setting: Optional[str]
    alpha: float
    # Step-0 top-k tokens and yes/no logits for the clean, negative, and
    # modified distributions, plus the six softmax-distance metrics (see
    # decoding/contrastive.py). The ORIGINAL branch records only the clean
    # side, since it runs no negative pass.
    logit_trace: Optional[dict] = None
    softmax_distance: Optional[dict] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict) -> "SampleResult":
        return SampleResult(**d)


@dataclass
class RunResult:
    """Every sample decoded under one branch, plus that branch's aggregate scores."""

    model: str
    dataset: str
    perturbation_type: str
    perturbation_setting: Optional[str]
    alpha: float
    prompt: str = "constrained"       # prompt tag this run used (paper Table II)
    split: Optional[str] = None       # Clotho-AQA partition, else None
    accuracy: float = 0.0
    correct: int = 0
    total: int = 0
    avg_softmax_distance: Optional[dict] = None
    results: list[SampleResult] = field(default_factory=list)

    @property
    def spec_label(self) -> str:
        if self.perturbation_setting is None:
            return self.perturbation_type
        return f"{self.perturbation_type}:{self.perturbation_setting}"

    def to_dict(self) -> dict:
        return {
            "metadata": {
                "model": self.model,
                "dataset": self.dataset,
                "split": self.split,
                "perturbation_type": self.perturbation_type,
                "perturbation_setting": self.perturbation_setting,
                "alpha": self.alpha,
                "prompt": self.prompt,
            },
            "performance": {
                "accuracy": round(self.accuracy, 4),
                "correct": self.correct,
                "total": self.total,
                **prediction_bias(self.results),
            },
            "avg_softmax_distance": self.avg_softmax_distance,
            "results": [r.to_dict() for r in self.results],
        }

    @staticmethod
    def from_dict(d: dict) -> "RunResult":
        meta = d["metadata"]
        perf = d.get("performance", {})
        return RunResult(
            model=meta["model"],
            dataset=meta["dataset"],
            split=meta.get("split"),
            perturbation_type=meta["perturbation_type"],
            perturbation_setting=meta.get("perturbation_setting"),
            alpha=meta["alpha"],
            # Runs produced before the prompt was recorded were all decoded
            # with the constrained prompt, which is the default here.
            prompt=meta.get("prompt", "constrained"),
            accuracy=perf.get("accuracy", 0.0),
            correct=perf.get("correct", 0),
            total=perf.get("total", 0),
            avg_softmax_distance=d.get("avg_softmax_distance"),
            results=[SampleResult.from_dict(r) for r in d.get("results", [])],
        )


def save_run_result(run: RunResult, path: Path) -> None:
    """Write a run atomically, so an interrupted save cannot leave a half file.

    A truncated ``eval_*.json.gz`` is worse than a missing one: the oracle
    builder would read it as a complete branch with fewer examples and reject
    the whole sweep.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(run.to_dict(), indent=2)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".gz":
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(payload)
    else:
        tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def load_run_result(path: Path) -> RunResult:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return RunResult.from_dict(json.load(f))
        return RunResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"Could not read decoding result {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer_text(response: str) -> str:
    """First-word yes/no extraction, falling back to a yes/no substring search."""
    text = response.lower().strip()
    first_word = "".join(c for c in (text.split() or [""])[0] if c.isalnum())
    if first_word.startswith("yes"):
        return "yes"
    if first_word.startswith("no"):
        return "no"
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return text[:20]


def extract_answer_from_logits(logit_trace: Optional[dict]) -> Optional[str]:
    """Compare step-0 yes/no logits, after the contrastive combination.

    ``modified`` is preferred and ``original`` is the fallback, so an ORIGINAL
    run (which has no negative branch and therefore no modified distribution)
    is scored by the same code path as every contrastive branch. Each side
    takes the max over its capitalisation variants, since the tokenizer treats
    ``Yes`` and ``yes`` as distinct tokens and either can lead.
    """
    if not logit_trace:
        return None
    yes_no = logit_trace.get("yes_no_logits", {})
    scores = yes_no.get("modified") or yes_no.get("original")
    if not scores:
        return None
    yes_logit = max((scores[t]["logit"] for t in YES_TOKENS if t in scores), default=None)
    no_logit = max((scores[t]["logit"] for t in NO_TOKENS if t in scores), default=None)
    if yes_logit is None or no_logit is None:
        return None
    return "yes" if yes_logit > no_logit else "no"


def extract_answer(response: str, logit_trace: Optional[dict] = None) -> str:
    """Prefer exact step-0 logit comparison; fall back to parsing generated text."""
    from_logits = extract_answer_from_logits(logit_trace)
    return from_logits if from_logits is not None else extract_answer_text(response)


def answers_match(extracted: str, ground_truth: str) -> bool:
    return extracted.strip().lower() == ground_truth.strip().lower()


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------

def accuracy_of(results: list[SampleResult]) -> tuple[float, int, int]:
    correct = sum(1 for r in results if r.is_correct)
    total = len(results)
    return (correct / total if total else 0.0), correct, total


def prediction_bias(results: list[SampleResult]) -> dict:
    """Predicted-yes rate and affirmative bias (paper Section V-A).

    ``affirmative_bias`` is the predicted-yes rate minus the ground-truth yes
    rate. On a balanced benchmark that reference is 0.5, which reproduces the
    paper's "+40.4%" figure directly; measuring against the actual label rate
    keeps the number meaningful on a partition that is not exactly balanced.
    """
    total = len(results)
    if not total:
        return {"predicted_yes_rate": None, "ground_truth_yes_rate": None, "affirmative_bias": None}
    predicted_yes = sum(1 for r in results if r.extracted_answer.strip().lower() == "yes")
    actual_yes = sum(1 for r in results if r.ground_truth.strip().lower() == "yes")
    predicted_rate = predicted_yes / total
    actual_rate = actual_yes / total
    return {
        "predicted_yes_rate": round(predicted_rate, 6),
        "ground_truth_yes_rate": round(actual_rate, 6),
        "affirmative_bias": round(predicted_rate - actual_rate, 6),
    }


def average_softmax_distance(results: list[SampleResult]) -> Optional[dict]:
    """Mean of each softmax-distance metric over the samples that recorded one."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in results:
        if not r.softmax_distance:
            continue
        for k, v in r.softmax_distance.items():
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    if not sums:
        return None
    return {k: round(sums[k] / counts[k], 6) for k in sums}
