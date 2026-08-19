"""
Audio Flamingo 3 (AF3) evaluation runner.

AF3 pairs the AF-Whisper audio encoder with a Qwen2.5-7B backbone. Functionally
this runner is the twin of ``run_qwen.py`` -- same contrastive mechanics, same
checkpoint/resume behaviour, same result schema, all shared through
``decoding/engine.py`` -- and it exists separately only because AF3 builds its
inputs differently:

  - Its processor consumes conversations directly through
    ``apply_chat_template(..., tokenize=True)``, with NumPy waveforms passed
    in-line via the ``path`` field of an audio content block. There is no
    intermediate text-templating step.
  - It runs in bfloat16 rather than float16, so floating-point inputs are cast
    to the model's dtype while integer ids and masks are left alone.

Because the waveform must be present for the template to emit audio positions
at all, AF3's NO_AUDIO branch omits the audio content block entirely rather
than keeping an unfilled placeholder the way Qwen2-Audio does. Both amount to
the same thing -- a text-only negative branch carrying no acoustic evidence --
but the mechanism differs, which is why encoding stays model-specific.

Usage:
    python -m decoding.run_af3 --dataset ah_order --perturbation REVERSE --setting full
    python -m decoding.run_af3 --dataset ah_existence --all
"""

from __future__ import annotations

import argparse

import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from decoding.engine import ModelAdapter, add_run_arguments, main_from_args
from helpers.config import PROMPT_JOINER, RunConfig
from helpers.runtime import resolve_dtype, set_random_seed

set_random_seed(42)


def _conversation(item: dict, prompt: str, audio=None) -> list:
    """AF3 chat turn: the instructed question, plus the waveform when present."""
    content = [{"type": "text", "text": prompt + PROMPT_JOINER + item["Q"]}]
    if audio is not None:
        content.append({"type": "audio", "path": audio})
    return [{"role": "user", "content": content}]


def _cast_inputs(inputs, model) -> dict:
    """Cast floating-point tensors to the model's dtype; leave ids and masks alone."""
    dtype = next(model.parameters()).dtype
    return {
        key: value.to(dtype) if torch.is_floating_point(value) else value
        for key, value in inputs.items()
    }


def encode(model, processor, items: list[dict], audios, prompt: str) -> dict:
    """Batch of items (+ optional waveforms) -> model kwargs on the right device."""
    if audios is None:
        conversations = [_conversation(item, prompt) for item in items]
    else:
        conversations = [
            _conversation(item, prompt, audio) for item, audio in zip(items, audios)
        ]
    inputs = processor.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True, return_dict=True,
    ).to(model.device)
    return _cast_inputs(inputs, model)


def load(cfg: RunConfig):
    """Load AF3 in bfloat16 with left padding.

    Left padding keeps the last real token at position -1 for every row, which
    is the position ``selector/cache.py`` reads its hidden-state feature from.
    """
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    processor.tokenizer.padding_side = "left"
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        cfg.model_id, device_map="auto", torch_dtype=resolve_dtype("af3"),
    )
    model.eval()
    return model, processor


ADAPTER = ModelAdapter(
    key="af3",
    label="Audio Flamingo 3",
    load=load,
    encode=encode,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio Flamingo 3 adaptive perturbation evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_arguments(parser)
    main_from_args(parser.parse_args(), ADAPTER)


if __name__ == "__main__":
    main()
