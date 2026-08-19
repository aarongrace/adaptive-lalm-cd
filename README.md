# Adaptive Perturbation Selection for Contrastive Audio Decoding

Large audio-language models (LALMs) hallucinate by letting language priors
override acoustic evidence. Contrastive decoding pushes back: run the model
twice, once on the real audio and once on a degraded view of it, and subtract
the second from the first, so tokens that stay likely even without the
acoustic evidence get penalised. Prior audio work picks one fixed negative
branch and keeps it — usually no-audio, additive noise, or temporal blurring.

This project asks what else that branch can be, and finds that the answer
depends on the question being asked.

- **A library of 105 perturbations across 38 types** ([`perturbations.py`](perturbations.py))
  spanning six families: temporal, frequency-filter, spectral,
  amplitude/dynamics, environmental, and additive-noise. The best branch is
  strongly task-dependent. Reversing the waveform destroys temporal structure
  while leaving every event audible, which is exactly what a temporal-order
  question needs: it takes Audio Flamingo 3 from 74.7% to 81.4% on AH Order.
  The same branch is useless for an existence question, where a pitch shift
  leads instead (69.5% → 73.9%).
- **A constrained prompt.** Requiring a one-word yes/no answer cuts
  Qwen2-Audio's affirmative bias on AH Existence from +40.4 points to +21.0,
  raising accuracy from 56.9% to 67.9% before any contrastive decoding — over
  four times the prompt-engineering gain reported by prior work on the same
  model.
- **A lightweight adaptive selector** ([`selector/`](selector/)) trained on
  hidden states cached from the clean forward pass, routing each example to
  its own negative branch at no extra inference cost. On Qwen2-Audio AH
  Existence it reaches 76.7% against a 72.4% best fixed branch, with 6.8
  points still separating it from the per-example oracle (83.5%) at the best
  candidate-pool size (N=4).

The selector wins when the candidate pool is diverse *and* the routing signal
is learnable, and the paper is explicit about when it is not. On AF3 AH Order
the top branches are all temporal disruptions that correct heavily overlapping
example sets, so reverse alone is hard to beat. On Clotho-AQA the selector
falls behind the best fixed branch as soon as more than one candidate is in
play. On AH Attribute both models sit near chance regardless of branch, and
contrastive decoding has nothing to work with. Section V of the paper works
through each case.

[Paper (PDF)](AdaptivePerturbation.pdf) &nbsp;|&nbsp; [arXiv:2607.00247](https://arxiv.org/abs/2607.00247)

![Overview of the adaptive perturbation selection pipeline: a selector reads text and audio embeddings, picks a negative branch, and both branches are forwarded through the LALM before contrastive correction at decoding time.](architecture_overview.png)

## Code tour

| Paper component | Entry point | Purpose |
| --- | --- | --- |
| Structured audio perturbations (Sec. III-B) | `perturbations.py` | The 105 negative branches, their six families, and their paper-facing labels |
| Contrastive correction (Eq. 3) and branch distances (Sec. V-D) | `decoding/contrastive.py` | The combination rule and the six softmax-divergence metrics |
| Model execution (Sec. IV) | `decoding/engine.py`, `decoding/run_qwen.py`, `decoding/run_af3.py`, `decoding/run_parallel.py` | One shared evaluation loop, two model adapters, GPU-parallel dispatch |
| Oracle upper bound (Eq. 4) and selector (Sec. III-C) | `selector/` | Correctness targets, cached features, the MLP head, and training |
| Offline experiment stages | `scripts/` | Oracle, splits, cache, training, and table/figure regeneration |
| Research context | `AdaptivePerturbation.pdf` | Method, experiments, figures, and reported results |

Run `python perturbations.py` to print the per-family breakdown and verify the
registry against the counts the paper reports.

## Workflow

Result summarisation and selector training both consume the decoded branch
results and are independent of each other.

```
licensed data + model weights
          |
          v
python -m decoding.run_parallel                 (105 branches x model x dataset)
          |
          ├──► python -m scripts.summarize_results    (tables and figures)
          │
          └──► python -m scripts.build_oracle         (per-example correctness, Eq. 4)
                        |
                        v
               python -m scripts.prepare_splits       (five composition-aware splits)
                        |
                        v
               python -m scripts.cache_hidden_states  (clean-branch features)
                        |
                        v
               python -m scripts.train_selector       (routing head, 5-split mean)
```

Run every command from the project root; all imports are absolute against it.

```bash
# 1. Decode all perturbation branches for one model/dataset.
python -m decoding.run_parallel --model qwen2 --dataset ah_existence --gpus 0 1

# 2. Build the oracle, with the per-branch ranking and coverage curve.
python -m scripts.build_oracle --model qwen2 --dataset ah_existence --rank --coverage

# 3. Generate the five composition-aware balanced 70/15/15 splits
#    (AH benchmarks only; uses the oracle from step 2).
python -m scripts.prepare_splits --model qwen2 --dataset ah_existence --report

# 4. Cache the clean-branch features the selector trains on.
python -m scripts.cache_hidden_states --model qwen2 --dataset ah_existence

# 5. Train and report across all five splits using the paper's exact pool.
python -m scripts.train_selector --model qwen2 --dataset ah_existence \
  --n 4 --paper-candidates --save

# 6. Regenerate tables and figures, any time after step 1.
python -m scripts.summarize_results --model qwen2 --dataset ah_existence --table --ranking
```

Step 1 writes to `runs/qwen2/ah_existence/1.0/eval_*.json.gz`. Branches that
already have results are skipped, so an interrupted sweep resumes by re-running
the same command. Before committing to a full sweep, drive one decoding run
directly as a smoke test:

```bash
python -m decoding.run_qwen --dataset ah_existence --perturbation REVERSE --setting full
python -m decoding.run_parallel --model qwen2 --dataset ah_existence --dry-run
```

## Reproducing specific paper results

Every table and figure has a command. Those marked *derived* are pure
post-processing over decoded results and need no GPU.

| Paper artifact | Command |
| --- | --- |
| Table II — prompt calibration and affirmative bias | decode with `--prompt aad` and `--prompt constrained`, then `scripts.summarize_results --prompt-table` *(derived)* |
| Fig. 2 — accuracy across alpha | `scripts.summarize_results --alpha-figure --perturbations ...` *(derived)* |
| Table IV — per-setting branch rankings | `scripts.summarize_results --table --ranking` *(derived)* |
| Fig. 4 — distance-based selection across N | `scripts.summarize_results --distance-figure` *(derived)* |
| Table V — oracle / selector / gap by pool size N | `scripts.train_selector --sweep-n` |
| Table V footnote — branch-coverage plateau | `scripts.build_oracle --coverage` *(derived)* |
| Table VI — input-feature ablation | `scripts.train_selector --feature-ablation` |
| Sec. VI-A — head architecture sweep | `scripts.train_selector --head mlp_1 \| mlp_2 \| mlp_3taper \| mlp_4 \| mlp_wide` |
| Sec. VI-D — regularisation baseline | `scripts.train_selector --unregularized` |

Where the paper reports a number for exactly what you are running,
`train_selector` prints it beside the measured value with the difference. That
is a reading aid only: no value is ever filled in from the table, and a
configuration that cannot be measured is reported as skipped rather than
quoted.

The alpha sweep deserves a note. Because Eq. (3) combines two fixed logit
vectors and the constrained prompt makes the answer a single token, sweeping
alpha over a stored run reproduces exactly what the decoder would have emitted
at each alpha. It is not an approximation, and it needs no re-inference.

## Setup

### Environment

Decoding and caching require Python 3.10+ and a CUDA GPU; both models are
~7B-parameter LALMs running at 16-bit precision. Install the matching
CUDA-enabled PyTorch build from the
[PyTorch selector](https://pytorch.org/get-started/locally/) first, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last command must print `True`. Transformers is pinned to 5.x because it
is the first series to include Audio Flamingo 3, and `accelerate` is required
for automatic device mapping (`device_map="auto"`).

The offline stages — split construction, table and figure regeneration — need
neither a GPU nor the model packages.

### Models

`helpers/config.py` points at
[`Qwen/Qwen2-Audio-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
and [`nvidia/audio-flamingo-3-hf`](https://huggingface.co/nvidia/audio-flamingo-3-hf).
Both download on first use via `from_pretrained`. Run `huggingface-cli login`
if a model card requires accepting a licence. Weights are never stored in this
repository. Audio Flamingo 3 is licensed for non-commercial research use, so
check each model card before use.

### Data format

Each dataset is a JSON list of
`{"path": <wav path>, "Q": <question>, "text": "yes"|"no"}` records under
`data/<dataset>/`:

| Dataset | Expected path | Source |
| --- | --- | --- |
| AH Existence | `data/ah_existence/ah_existence.json` | [Audio Hallucination benchmark](https://arxiv.org/abs/2410.16130) |
| AH Order | `data/ah_order/ah_order.json` | Audio Hallucination, built on [CompA](https://arxiv.org/abs/2310.08753) |
| AH Attribute | `data/ah_attribute/ah_attribute.json` | Audio Hallucination, built on [CompA](https://arxiv.org/abs/2310.08753) |
| Clotho-AQA | `data/clotho_aqa/clotho_aqa_{train,val,test}.json` | [Clotho-AQA](https://arxiv.org/abs/2204.09634), yes/no subset |

Obtain these third-party datasets under their own terms, convert them to the
schema above, and point `path` at the stored audio. The decoder validates the
manifest and every audio file before loading a model, strictly requires the
binary yes/no schema, and reports malformed records rather than skipping them.

**AH Existence composition provenance.** For a paper-compatible selector
split, the conversion must additionally preserve either the raw BEAF filename
convention `background@event_a@event_b@event_c` (with pair rows using the
corresponding two-event form), or add these fields to every row:

```json
{
  "composition_id": "stable-background-plus-events identifier",
  "source_clip_ids": ["background", "event_a", "event_b", "event_c"]
}
```

`prepare_splits.py` refuses a path-only AH Existence manifest, because
grouping by the rendered mixture file alone implements neither the paper's
composition-level split nor its train/held-out source-overlap objective. AH
Order and AH Attribute use `composition_id` when supplied, and otherwise group
all questions for the same audio path together.

## The split protocol

This is the part most easily got wrong, so it is worth stating directly. The
AH benchmarks ship without standard splits, and the paper constructs five
balanced 70/15/15 replicates per task, averaging results across them.

They are not five random record-level shuffles. Every AH audio file is a
rendered mixture of a background track and several foreground events, and each
mixture carries many questions. A record-level split would scatter questions
about the same audio across train and test, letting a selector be rewarded for
recognising a clip it trained on. Splits are therefore made at the
**audio-composition** level: all questions from one mixture move together.

For AH Order and AH Attribute that is enough — no audio file appears in two
partitions. For AH Existence it is not, and the paper says so. The benchmark
builds its 10,800 mixtures by recombining a much smaller pool of source
recordings, so most sources appear in many compositions and zero train/held-out
source overlap is structurally impossible at a 70/15/15 ratio. The search
minimises overlap as a penalty instead. `--report` prints the inventory that
makes this checkable: how many distinct sources exist, how many compositions
each appears in, and therefore how low overlap can actually be driven.

```bash
python -m scripts.prepare_splits --model qwen2 --dataset ah_existence --report
python -m scripts.prepare_splits --model qwen2 --dataset ah_existence --verify
python -m scripts.prepare_splits --model qwen2 --dataset ah_existence --random-baselines
```

`--verify` re-checks every split on disk for composition integrity, complete
coverage, and agreement between recorded and recomputed overlap.
`--random-baselines` writes unbalanced composition-level control splits, so
the contribution of the balancing search can be separated from that of the
grouping alone. Each split file records its grouping basis, search budget,
seed, partition sizes, and measured overlap, and is explicitly flagged when
the search budget was reduced below the paper default.

## Selector protocol

Build the oracle before constructing splits. It retains every decoded branch
including `ORIGINAL`, which matters because routing to "apply no correction
here" is a legitimate decision and the paper's AF3 AH Existence N=4 pool
contains it.

For the two reported AH Existence N=4 selectors, `--paper-candidates` pins the
exact branch labels. Without it, `train_selector` picks the top-N branches by
aggregate oracle success and records the resulting set; that is a reasonable
new experiment, not a substitute for the listed configuration.

AH selector output is the arithmetic mean across exactly five split files.
Passing `--split-name` is deliberately labelled a single-split debug run.
Clotho-AQA has official train/validation/test files instead: omit `--split`
when building its oracle and cache to process all three, then invoke
`train_selector` once for its official partition.

### Generated artifacts

`data/`, `runs/`, `cache/`, `checkpoints/`, and `outputs/` are created on
demand and git-ignored. Their locations are the only paths `helpers/config.py`
hard-codes; nothing else in the repository assumes a machine-specific path.
Runs under a non-default prompt are written to a prompt-tagged sibling
directory, so a Table II comparison sweep cannot overwrite the main results.

## Reproducibility scope

This repository is a paper-protocol reproduction implementation. It does not
distribute raw audio, model weights, or generated artifacts (anything under
`data/`, `runs/`, `cache/`, `checkpoints/`, or `outputs/`).

The code is self-contained and depends on no file from the authors' private
workspace. Given the original licensed manifests (including AH composition
provenance), complete branch outputs, and model access, it implements the
paper's decoding, oracle, split, cache, and selector protocol. Exact numerical
reproduction additionally depends on dataset conversion, seeds, hardware,
model revisions, and run artifacts that are not distributed here.

Known boundaries:

- Datasets must already be converted into the documented JSON schema. Upstream
  acquisition and conversion are not scripted.
- The raw and projected audio-encoder features used by four rows of the
  feature ablation are captured through forward hooks on the model's audio
  tower and projector. Module naming varies across model implementations and
  `transformers` releases; when a hook target cannot be found, the caching
  script says so and those rows are reported as skipped rather than silently
  omitted. The reported best hidden-state selector does not depend on them.
- Package and model revisions are range-pinned rather than exact, and the
  paper's generated result artifacts are not distributed. See
  [`PAPER_REPRODUCIBILITY_AUDIT.md`](PAPER_REPRODUCIBILITY_AUDIT.md) before
  making any claim of numerical reproduction.

## Citation

```bibtex
@article{grace2026adaptive,
  title   = {Adaptive Perturbation Selection for Contrastive Audio Decoding},
  author  = {Grace, Aaron Isidore and Huo, Zhouyuan and Wang, Weiran},
  journal = {arXiv preprint arXiv:2607.00247},
  year    = {2026}
}
```

## Contact

Aaron Isidore Grace <aaron.grace@uwaterloo.ca> &nbsp;|&nbsp;
Zhouyuan Huo <huozhouyuan@gmail.com> &nbsp;|&nbsp;
Weiran Wang <weiran-wang@uiowa.edu>
