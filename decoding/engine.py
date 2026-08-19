"""
The model-independent half of a decoding run.

``run_qwen.py`` and ``run_af3.py`` differ only in how a batch of waveforms and
questions becomes tokenised model inputs: Qwen2-Audio wants a text-templated
string plus a list of arrays, AF3 wants conversations carrying the arrays
in-line, and the two run at different precisions. Everything after that point
-- perturbing the audio, building the negative branch, applying Eq. (3) during
generation, scoring, checkpointing, and writing results -- is identical.

That shared half lives here, and each runner contributes a :class:`ModelAdapter`
holding its two model-specific pieces. Keeping one loop matters beyond tidiness:
``selector/cache.py`` must produce the selector's features from *the same*
clean forward pass the decoder scored, and it does so by calling the same
adapter rather than by maintaining a parallel copy that can quietly drift.

Negative-branch construction, in the order the paper defines the branches:

  ORIGINAL   no negative pass at all; logits are captured and passed through.
  NO_AUDIO   the prompt is re-encoded with the audio omitted, and the LM's
             input embeddings are taken directly. This is AAD's strongest
             mode and the paper's reference branch.
  otherwise  the waveform is transformed by ``perturbations.py``, re-encoded,
             and pushed through one forward pass with ``output_hidden_states``;
             ``hidden_states[0]`` -- the combined audio+text sequence as the LM
             sees it before any attention -- becomes the negative branch's
             starting point, which the logits processor then extends token by
             token during generation.
"""

from __future__ import annotations

import argparse
import gc
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import librosa
import torch
from tqdm import tqdm

from decoding.contrastive import AudioLogitsProcessor, OriginalLogitsCapture
from helpers.config import (
    CHECKPOINT_INTERVAL,
    DATASETS,
    DEFAULT_BATCH_SIZE,
    PAPER_ALPHA,
    PROMPTS,
    RunConfig,
)
from helpers.datasets import dataset_summary, load_examples, print_dataset_summary
from helpers.results import (
    RunResult,
    SampleResult,
    accuracy_of,
    answers_match,
    average_softmax_distance,
    extract_answer,
    prediction_bias,
    save_run_result,
)
from helpers.runtime import (
    delete_checkpoint,
    load_checkpoint,
    output_filename,
    require_cuda,
    results_exist,
    save_checkpoint,
)
from perturbations import (
    FAMILY_ORDER,
    get_perturbation,
    get_perturbation_class,
    iter_perturbation_configs,
)


@dataclass(frozen=True)
class ModelAdapter:
    """The two model-specific hooks the shared loop needs.

    ``load(cfg) -> (model, processor)`` instantiates the LALM at its reported
    precision with automatic device mapping.

    ``encode(model, processor, items, audios, prompt) -> dict`` turns one
    batch into kwargs for ``model(**inputs)`` and ``model.generate(**inputs)``,
    already moved to the model's device and dtype. ``audios`` is None for the
    text-only NO_AUDIO branch and a list of waveforms otherwise.
    """

    key: str
    label: str
    load: Callable[[RunConfig], tuple]
    encode: Callable[..., dict]


def get_adapter(model_key: str) -> ModelAdapter:
    """Fetch a runner's adapter by model key, importing it lazily.

    The import is deferred because each runner pulls in its own heavyweight
    ``transformers`` model class, and most entry points need only one of them.
    """
    if model_key == "qwen2":
        from decoding.run_qwen import ADAPTER
    elif model_key == "af3":
        from decoding.run_af3 import ADAPTER
    else:
        raise ValueError(f"No decoding adapter for model {model_key!r}; expected 'qwen2' or 'af3'")
    return ADAPTER


def load_batch_audio(batch: Sequence[dict], sr: int) -> tuple[list[dict], list]:
    """Read every waveform in a batch at the model's sampling rate.

    The manifest was validated before the model was loaded, so a file missing
    here means it vanished mid-run; that is a hard error rather than a skip,
    because a silently short branch would corrupt the oracle's example
    alignment across branches.
    """
    items, audios = [], []
    for item in batch:
        if not os.path.exists(item["path"]):
            raise FileNotFoundError(
                f"Audio file disappeared after manifest validation: {item['path']}"
            )
        audio, _ = librosa.load(item["path"], sr=sr, mono=True)
        items.append(item)
        audios.append(audio)
    return items, audios


def build_negative_branch(adapter: ModelAdapter, model, processor, items, audios,
                          perturbation_type: str, perturbation_setting: Optional[str],
                          prompt: str, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(embeds_neg, atts_neg)`` for a non-ORIGINAL branch."""
    if perturbation_type == "NO_AUDIO":
        inputs_neg = adapter.encode(model, processor, items, None, prompt)
        return model.get_input_embeddings()(inputs_neg["input_ids"]), inputs_neg["attention_mask"]

    pert_cls = get_perturbation_class(perturbation_type)
    pert_fn = get_perturbation(pert_cls, perturbation_setting, sr=sr)
    perturbed = [pert_fn(a) for a in audios]
    inputs_neg = adapter.encode(model, processor, items, perturbed, prompt)
    with torch.no_grad():
        out_neg = model(**inputs_neg, output_hidden_states=True, return_dict=True)
    return out_neg.hidden_states[0], inputs_neg["attention_mask"]


def process_batch(batch, model, processor, adapter: ModelAdapter, perturbation_type: str,
                  perturbation_setting: Optional[str], alpha: float, prompt: str,
                  max_new_tokens: int) -> list[SampleResult]:
    """Decode one batch under one negative branch and score it."""
    sr = processor.feature_extractor.sampling_rate
    items, audios = load_batch_audio(batch, sr)
    if not items:
        return []

    inputs_clean = adapter.encode(model, processor, items, audios, prompt)

    if perturbation_type == "ORIGINAL":
        proc = OriginalLogitsCapture(processor.tokenizer, batch_size=len(items))
    else:
        embeds_neg, atts_neg = build_negative_branch(
            adapter, model, processor, items, audios,
            perturbation_type, perturbation_setting, prompt, sr,
        )
        proc = AudioLogitsProcessor(model, processor.tokenizer, embeds_neg, atts_neg, alpha)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs_clean,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=[proc],
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    responses = processor.batch_decode(
        output_ids[:, inputs_clean["input_ids"].size(1):], skip_special_tokens=True
    )

    results = []
    for row, (item, response) in enumerate(zip(items, responses)):
        trace = proc.step0_trace[row]
        distance = proc.step0_distance[row]
        ground_truth = item["text"].strip().lower()
        extracted = extract_answer(response, trace)
        results.append(SampleResult(
            audio_file=item["path"], question=item["Q"], ground_truth=ground_truth,
            model_response=response, extracted_answer=extracted,
            is_correct=answers_match(extracted, ground_truth),
            perturbation_type=perturbation_type, perturbation_setting=perturbation_setting,
            alpha=alpha, logit_trace=trace, softmax_distance=distance,
        ))
    return results


def run_branch(perturbation_type: str, perturbation_setting: Optional[str], cfg: RunConfig,
               adapter: ModelAdapter, model=None, processor=None) -> Optional[RunResult]:
    """Evaluate one negative branch over a whole dataset.

    Returns None when the branch already has results on disk. Pass ``model``
    and ``processor`` to reuse a loaded LALM across several branches in one
    process (what ``--all`` does); leave them None to load and release per
    branch, which is what the per-GPU subprocesses in ``run_parallel.py`` do.
    """
    if results_exist(cfg.results_dir, perturbation_type, perturbation_setting, cfg.alpha):
        return None
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_filename(cfg.results_dir, perturbation_type, perturbation_setting, cfg.alpha)

    checkpoint_rows, last_batch_idx = load_checkpoint(
        cfg.results_dir, perturbation_type, perturbation_setting, cfg.alpha
    )
    all_results = [SampleResult.from_dict(r) for r in checkpoint_rows]
    start_batch = last_batch_idx + 1

    data = load_examples(cfg)
    require_cuda()
    print(f"{adapter.label} | {perturbation_type}:{perturbation_setting} | {cfg.describe()}")
    if start_batch == 0:
        print_dataset_summary(dataset_summary(cfg, data))

    owns_model = model is None
    if owns_model:
        model, processor = adapter.load(cfg)

    try:
        batch_starts = list(range(0, len(data), DEFAULT_BATCH_SIZE))
        for batch_idx, offset in enumerate(
            tqdm(batch_starts, initial=start_batch, total=len(batch_starts))
        ):
            if batch_idx < start_batch:
                continue
            all_results.extend(process_batch(
                data[offset:offset + DEFAULT_BATCH_SIZE], model, processor, adapter,
                perturbation_type, perturbation_setting, cfg.alpha,
                cfg.run_prompt, cfg.new_tokens,
            ))
            if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(cfg.results_dir, perturbation_type, perturbation_setting,
                                cfg.alpha, [r.to_dict() for r in all_results], batch_idx)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        if owns_model:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    accuracy, correct, total = accuracy_of(all_results)
    run_result = RunResult(
        model=cfg.model_id, dataset=cfg.dataset, split=cfg.split,
        perturbation_type=perturbation_type, perturbation_setting=perturbation_setting,
        alpha=cfg.alpha, prompt=cfg.prompt_tag,
        accuracy=accuracy, correct=correct, total=total,
        avg_softmax_distance=average_softmax_distance(all_results),
        results=all_results,
    )
    save_run_result(run_result, out_path)
    delete_checkpoint(cfg.results_dir, perturbation_type, perturbation_setting, cfg.alpha)

    bias = prediction_bias(all_results)["affirmative_bias"]
    bias_text = f", affirmative bias {bias:+.1%}" if bias is not None else ""
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total}){bias_text} -> {out_path}")
    return run_result


def run_many(configs: Sequence[tuple[str, Optional[str]]], cfg: RunConfig,
             adapter: ModelAdapter) -> None:
    """Evaluate several branches in one process, loading the model only once."""
    pending = [
        (pert_type, setting) for pert_type, setting in configs
        if not output_filename(cfg.results_dir, pert_type, setting, cfg.alpha).exists()
    ]
    if not pending:
        print(f"All {len(configs)} requested branch(es) already have results in {cfg.results_dir}")
        return

    require_cuda()
    print(f"{adapter.label}: {len(pending)} of {len(configs)} branch(es) still to decode.")
    model, processor = adapter.load(cfg)
    try:
        for pert_type, setting in pending:
            run_branch(pert_type, setting, cfg, adapter, model=model, processor=processor)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Shared command-line surface
#
# Both runners take the same options, and so does run_parallel.py, which
# forwards them verbatim to the subprocesses it spawns. Defining them once
# keeps the three entry points from drifting apart.
# ---------------------------------------------------------------------------

def add_run_arguments(parser: argparse.ArgumentParser, *, include_branch: bool = True) -> None:
    """Attach the options that identify one decoding run."""
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASETS))
    parser.add_argument("--split", default=None, choices=["train", "val", "test"],
                        help="Required for clotho_aqa, which ships official partitions")
    parser.add_argument("--alpha", "-a", type=float, default=PAPER_ALPHA,
                        help="Contrastive strength in Eq. (3); the paper reports alpha=1.0")
    parser.add_argument("--prompt", default="constrained", choices=sorted(PROMPTS),
                        help="'constrained' is this paper's one-word yes/no prompt; "
                             "'aad' is the open prior-work prompt used in Table II")
    parser.add_argument("--max-new-tokens", type=int, default=1,
                        help="1 scores the single constrained answer token; raise to inspect prose")
    if include_branch:
        parser.add_argument("--perturbation", "-p", default="NO_AUDIO",
                            help="Branch type, e.g. NO_AUDIO, REVERSE, PITCH_SHIFT")
        parser.add_argument("--setting", "-s", default=None,
                            help="Named setting within that type, e.g. extreme_up")
        parser.add_argument("--all", action="store_true",
                            help="Run every branch in the registry instead of a single one")
        parser.add_argument("--family", nargs="+", default=None, choices=list(FAMILY_ORDER),
                            help="With --all, restrict to these perturbation families")


def config_from_args(args: argparse.Namespace, model_key: str) -> RunConfig:
    return RunConfig(
        model=model_key,
        dataset=args.dataset,
        alpha=args.alpha,
        split=args.split,
        prompt_name=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )


def main_from_args(args: argparse.Namespace, adapter: ModelAdapter) -> None:
    """Turn parsed arguments into either one branch run or a whole sweep."""
    cfg = config_from_args(args, adapter.key)
    if args.all:
        families = tuple(args.family) if args.family else None
        run_many(list(iter_perturbation_configs(families=families)), cfg, adapter)
    else:
        if args.family:
            raise SystemExit("--family only applies together with --all")
        run_branch(args.perturbation.upper(), args.setting, cfg, adapter)
