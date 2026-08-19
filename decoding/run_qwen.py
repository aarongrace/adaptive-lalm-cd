"""
Qwen2-Audio-7B-Instruct evaluation runner.

Qwen2-Audio (arXiv:2407.10759) pairs a Whisper-large audio encoder with a
Qwen-7B language model. Raw 16 kHz mono waveforms become 128-channel
mel-spectrograms, are encoded by the Whisper stack, and are projected into the
Qwen-7B embedding space by a learned ``multi_modal_projector``; the projected
frames replace the ``<|AUDIO|>`` placeholder positions in the text token
sequence the LM attends over.

This module contributes only the two Qwen2-specific pieces the shared loop in
``decoding/engine.py`` needs: how to load the model, and how to turn a batch
of waveforms and questions into model inputs. The contrastive mechanics, the
checkpoint/resume behaviour, and the result schema are shared verbatim with
``run_af3.py``.

One detail worth stating explicitly, because it defines what "no audio" means
for this model. The NO_AUDIO branch re-encodes the *same* templated text --
placeholder token included -- with ``audio=None``, and then reads the LM's
input embeddings directly. The placeholder therefore keeps its ordinary text
embedding instead of being substituted with projected audio frames, so the
negative branch is the model reasoning from language priors alone at
comparable sequence positions.

Usage:
    python -m decoding.run_qwen --dataset ah_existence --perturbation REVERSE --setting full
    python -m decoding.run_qwen --dataset ah_existence --all
    python -m decoding.run_qwen --dataset ah_existence --perturbation NO_AUDIO --prompt aad
"""

from __future__ import annotations

import argparse

from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from decoding.engine import ModelAdapter, add_run_arguments, main_from_args
from helpers.config import PROMPT_JOINER, RunConfig
from helpers.runtime import resolve_dtype, set_random_seed

set_random_seed(42)


def _conversation(item: dict, prompt: str) -> list:
    """Qwen2-Audio chat turn: one audio block followed by the instructed question.

    ``audio_url`` is consumed by the chat template only, to place the
    ``<|AUDIO|>`` placeholder; the waveform itself is supplied separately to
    the processor, which is what lets the NO_AUDIO branch reuse this template
    unchanged.
    """
    return [{"role": "user", "content": [
        {"type": "audio", "audio_url": item["path"]},
        {"type": "text", "text": prompt + PROMPT_JOINER + item["Q"]},
    ]}]


def encode(model, processor, items: list[dict], audios, prompt: str) -> dict:
    """Batch of items (+ optional waveforms) -> model kwargs on the right device."""
    sr = processor.feature_extractor.sampling_rate
    texts = [
        processor.apply_chat_template(_conversation(item, prompt),
                                      add_generation_prompt=True, tokenize=False)
        for item in items
    ]
    inputs = processor(
        text=texts, audio=audios, return_tensors="pt", padding=True, sampling_rate=sr
    )
    return inputs.to(model.device)


def load(cfg: RunConfig):
    """Load Qwen2-Audio at the reported precision with left padding.

    Left padding matters downstream: it puts the last real token at position
    -1 for every row in a batch, which is exactly the position the selector's
    hidden-state feature is read from (``selector/cache.py``).
    """
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    processor.tokenizer.padding_side = "left"
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        cfg.model_id, device_map="auto", torch_dtype=resolve_dtype("qwen2"),
    )
    model.eval()
    return model, processor


ADAPTER = ModelAdapter(
    key="qwen2",
    label="Qwen2-Audio-7B-Instruct",
    load=load,
    encode=encode,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen2-Audio adaptive perturbation evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_arguments(parser)
    main_from_args(parser.parse_args(), ADAPTER)


if __name__ == "__main__":
    main()
