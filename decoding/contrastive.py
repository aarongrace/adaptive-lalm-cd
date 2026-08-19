"""
Contrastive audio decoding: the combination rule shared by every negative
branch in this repository, the softmax-distance metrics used to score how far
a branch diverges from the clean prediction, and the two logits processors
that apply it during generation.

Audio-Aware Decoding (AAD, arXiv:2506.07233) suppresses hallucinated answers
by running two forward passes at each generation step -- one on the real audio
("clean") and one on a degraded or absent view of it ("negative") -- and
replacing the clean logits with

    modified = (1 + alpha) * clean - alpha * negative

which is Eq. (3) in the paper. Tokens whose probability is elevated
specifically by the presence of real audio are amplified; tokens the model
would predict regardless of audio are penalised. AAD itself evaluates two
negative modes, silence and fully removing the audio (NO_AUDIO, the stronger
of the two). This paper generalises the negative branch to the full library in
``perturbations.py`` -- waveform and spectral transforms that each deny a
different dimension of the acoustic evidence -- and adds a selector (see
``selector/``) that learns which branch to use per example instead of fixing
one for the whole task.

Alpha controls how hard the correction pushes. The paper sweeps it over
[0, 2] and fixes 1.0 for all main experiments: beyond that the correction term
outweighs the expert branch and gains become hard to attribute.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from helpers.config import TOP_K_LOGGING

YES_TOKENS = ("Yes", "yes")
NO_TOKENS = ("No", "no")


def contrastive_logits(clean_logits: torch.Tensor, negative_logits: torch.Tensor,
                       alpha: float) -> torch.Tensor:
    """Eq. (3): ``(1 + alpha) * clean - alpha * negative``, applied every decode step."""
    return (1.0 + alpha) * clean_logits - alpha * negative_logits


def predict_yes_no_at_alpha(clean: Dict[str, dict], negative: Dict[str, dict],
                            alpha: float) -> str:
    """Re-score one stored step-0 trace at an arbitrary alpha.

    ``clean`` and ``negative`` are the per-token ``{"logit": ..., "prob": ...}``
    maps recorded in a result file. Because Eq. (3) is a scalar combination of
    two fixed logit vectors, and the constrained prompt makes the answer a
    single token, an alpha sweep over an existing run needs no re-inference at
    all: this is exactly what the decoder would have produced at that alpha.
    That equivalence is what makes ``scripts/summarize_results.py
    --alpha-figure`` a pure post-processing step (paper Section V-B, Fig. 2).
    """
    scale = 1.0 + alpha

    def best(tokens: Sequence[str]) -> Optional[float]:
        values = [
            scale * clean[t]["logit"] - alpha * negative[t]["logit"]
            for t in tokens if t in clean and t in negative
        ]
        return max(values) if values else None

    yes, no = best(YES_TOKENS), best(NO_TOKENS)
    if yes is None or no is None:
        raise ValueError("Stored trace lacks paired yes/no logits for both branches")
    return "yes" if yes > no else "no"


# ---------------------------------------------------------------------------
# Branch-distance metrics (paper Section V-D)
#
# VACoDe (arXiv:2408.05337) selects the negative branch that maximises L2
# distance between the clean and negative softmax distributions. The paper
# evaluates the analogous criterion for audio across all six metrics below and
# finds KL divergence the strongest of them, though still weaker than the
# learned selector -- suggesting logit divergence is an unreliable proxy for
# contrastive utility, since excessive divergence can trigger new
# hallucinations rather than isolating the intended acoustic cue.
#
# Earth Mover's Distance is deliberately omitted: vocabulary token indices
# carry no meaningful ground distance, so an EMD along that axis would not be
# interpretable.
# ---------------------------------------------------------------------------

SOFTMAX_DISTANCE_KEYS = ("l1", "l2", "l3", "linf", "cosine", "kl")


def compute_softmax_distances(logits_clean: torch.Tensor,
                              logits_negative: torch.Tensor) -> Dict[str, float]:
    """Distances between two next-token distributions.

    Both arguments are 1-D raw logit vectors of shape ``[vocab_size]``. They
    are cast to float32 before softmax so that bfloat16/float16 model outputs
    do not lose small-probability tokens to rounding, which would distort the
    tail-sensitive metrics (``kl`` especially).
    """
    p_clean = torch.softmax(logits_clean.float(), dim=-1)
    p_neg = torch.softmax(logits_negative.float(), dim=-1)
    diff = p_clean - p_neg

    l1 = diff.abs().sum().item()
    l2 = diff.norm(p=2).item()
    l3 = diff.abs().pow(3).sum().pow(1.0 / 3.0).item()
    linf = diff.abs().max().item()
    cosine = 1.0 - F.cosine_similarity(p_clean.unsqueeze(0), p_neg.unsqueeze(0)).item()
    # KL(p_clean || p_neg): nats needed to encode the clean distribution using
    # a code optimised for the negative one. p_neg is clamped away from zero to
    # keep the log finite on tokens the negative branch rules out entirely.
    kl = F.kl_div(p_neg.clamp(min=1e-10).log(), p_clean, reduction="sum").item()

    return {
        "l1": round(l1, 6),
        "l2": round(l2, 6),
        "l3": round(l3, 6),
        "linf": round(linf, 6),
        "cosine": round(cosine, 6),
        "kl": round(kl, 6),
    }


# ---------------------------------------------------------------------------
# LogitsProcessor implementations
#
# Both are pure decoding-time mechanics: they need only a model exposing
# get_input_embeddings() and a standard forward/generate interface, so
# run_qwen.py and run_af3.py share them verbatim. Everything model-specific --
# chat template format, input dtype, how a negative branch's embeddings are
# produced -- stays in the runner that builds embeds_neg/atts_neg before
# handing them to AudioLogitsProcessor.
# ---------------------------------------------------------------------------

def _first_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Could not encode token text: {text!r}")
    return ids[0]


def _yes_no_logits(token_ids: Dict[str, int], logits_row: torch.Tensor) -> dict:
    """Record the raw logit and probability of every yes/no token variant."""
    probs = torch.softmax(logits_row.float(), dim=-1)
    return {
        text: {"logit": round(logits_row[tid].item(), 4), "prob": round(probs[tid].item(), 6)}
        for text, tid in token_ids.items()
    }


def _top_k(tokenizer, logits_row: torch.Tensor, k: int) -> list:
    """Top-k tokens by logit, for inspecting what a branch actually predicted."""
    probs = torch.softmax(logits_row.float(), dim=-1)
    vals, idxs = torch.topk(logits_row, k)
    return [
        {"token": tokenizer.decode([i]), "score": round(v.item(), 4), "prob": round(probs[i].item(), 6)}
        for v, i in zip(vals, idxs)
    ]


class OriginalLogitsCapture(LogitsProcessor):
    """Pass-through processor for the ORIGINAL (no contrastive decoding) branch.

    It still records step-0 yes/no logits, so an ORIGINAL run is scored by the
    same ``extract_answer()`` path as every contrastive branch and its numbers
    are directly comparable. No negative forward pass runs, so there are no
    softmax distances to report.
    """

    def __init__(self, tokenizer, batch_size: int, top_k: int = TOP_K_LOGGING):
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.step0_trace: list = [None] * batch_size
        self.step0_distance: list = [None] * batch_size
        self._token_ids = {t: _first_token_id(tokenizer, t) for t in (*YES_TOKENS, *NO_TOKENS)}
        self._first_call = True

    def __call__(self, input_ids, scores):
        if self._first_call:
            for i in range(scores.shape[0]):
                self.step0_trace[i] = {
                    "original_top_k": _top_k(self.tokenizer, scores[i], self.top_k),
                    "yes_no_logits": {"original": _yes_no_logits(self._token_ids, scores[i])},
                }
            self._first_call = False
        return scores


class AudioLogitsProcessor(LogitsProcessor):
    """Runs the negative branch alongside generation and applies Eq. (3).

    ``embeds_neg``/``atts_neg`` start as the negative branch's full prompt
    (perturbed audio + text, or text-only for NO_AUDIO). At every later step
    the processor appends the just-generated token's embedding to
    ``embeds_neg``, so both branches stay conditioned on the same partial
    answer ``y_{<t}`` -- without that, the subtraction after step 0 would
    compare distributions conditioned on different prefixes.

    Under the constrained prompt only step 0 is scored, but the loop is kept
    general so free-form continuations (``--max-new-tokens`` > 1) remain
    correct.
    """

    def __init__(self, model, tokenizer, embeds_neg: torch.Tensor, atts_neg: torch.Tensor,
                 alpha: float, top_k: int = TOP_K_LOGGING):
        self.model = model
        self.tokenizer = tokenizer
        self.embeds_neg = embeds_neg
        self.atts_neg = atts_neg
        self.alpha = alpha
        self.top_k = top_k
        self.step0_trace: list = [None] * embeds_neg.shape[0]
        self.step0_distance: list = [None] * embeds_neg.shape[0]
        self._token_ids = {t: _first_token_id(tokenizer, t) for t in (*YES_TOKENS, *NO_TOKENS)}
        self._first_call = True

    def __call__(self, input_ids, scores):
        device = scores.device
        with torch.no_grad():
            if self._first_call:
                self.embeds_neg = self.embeds_neg.to(device)
                self.atts_neg = self.atts_neg.to(device)
            else:
                new_tokens = input_ids[:, -1:].to(device)
                new_embeds = self.model.get_input_embeddings()(new_tokens).to(device)
                self.embeds_neg = torch.cat([self.embeds_neg, new_embeds], dim=1)
                new_atts = (new_tokens != self.tokenizer.eos_token_id).to(self.atts_neg.dtype).to(device)
                self.atts_neg = torch.cat([self.atts_neg, new_atts], dim=1)

            logits_neg = self.model(
                inputs_embeds=self.embeds_neg, attention_mask=self.atts_neg
            ).logits[:, -1, :]

        modified = contrastive_logits(scores, logits_neg, self.alpha)

        if self._first_call:
            for i in range(scores.shape[0]):
                self.step0_distance[i] = compute_softmax_distances(scores[i], logits_neg[i])
                self.step0_trace[i] = {
                    "original_top_k": _top_k(self.tokenizer, scores[i], self.top_k),
                    "negative_top_k": _top_k(self.tokenizer, logits_neg[i], self.top_k),
                    "modified_top_k": _top_k(self.tokenizer, modified[i], self.top_k),
                    "yes_no_logits": {
                        "original": _yes_no_logits(self._token_ids, scores[i]),
                        "negative": _yes_no_logits(self._token_ids, logits_neg[i]),
                        "modified": _yes_no_logits(self._token_ids, modified[i]),
                    },
                }
            self._first_call = False

        return modified
