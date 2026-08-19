# Paper-to-release reproducibility audit

## Scope and evidence

This audit compares the public release against the submitted manuscript source
(`paper/submission/main.tex`) and the experiment-side implementations in the
authors' working checkout. It is a **source and offline-behaviour audit, not a
fresh GPU rerun**: licensed datasets, model weights, and the paper's result
artifacts are not part of this repository.

What was executed for this audit is recorded under
[Verification performed](#verification-performed) below. What was not is
recorded under [Verification boundary](#verification-boundary). Neither list
should be read as a claim that the published decimals were reproduced.

The paper-critical contracts the release must honour:

- **Prompts.** Table II compares an open prior-work prompt against this
  paper's constrained one-word yes/no prompt, with and without contrastive
  decoding. Both must be runnable, and the affirmative-bias statistic that
  explains the difference must be measurable, not just quoted.
- **Correction rule.** Decoding applies
  `(1 + alpha) * clean - alpha * negative` at every step, and scores the
  constrained task from step-0 yes/no logits.
- **Perturbation library.** 105 branches across 38 types, in the six families
  of Section III-B, with the parameter settings the manuscript lists.
- **Splits.** AH uses five balanced 70/15/15 replicates, averaged at reporting
  time, partitioned at the audio-composition level. AH Existence re-uses source
  clips across compositions, so zero train/held-out overlap is structurally
  impossible and the search minimises it instead; AH Order and Attribute must
  keep an audio file wholly within one partition. The search balances example
  difficulty and per-branch correctness, not just record counts, then selects
  five splits with balanced held-out coverage.
- **Oracle.** Labels are multi-hot branch-correctness vectors, and the oracle
  accuracy is `(1/N) sum_i max_s M[i,s]`. `ORIGINAL` is a valid selectable
  branch wherever it is in the candidate pool: the paper's AF3 AH Existence
  N=4 pool explicitly contains it.
- **Selector.** A 3-layer `[512, 256, 128]` MLP trained with BCE on
  first/middle/final-layer last-token states, with label smoothing 0.25,
  feature noise 0.10, and input dropout 0.05.

## Findings and disposition

### Resolved in the first repair pass

| Priority | Release behaviour before repair | Paper consequence | Disposition |
| --- | --- | --- | --- |
| Critical | `prepare_splits.py` shuffled `path` groups and greedily matched record counts only. | AH Existence was not split at its composition level; it neither minimised source-clip overlap nor balanced branch utility and difficulty. | Replaced with the paper-compatible three-phase composition-aware search plus swap refinement. |
| Critical | A selector command trained one default split and printed one test score. | The five-split mean in the selector table could not be produced. | Five fixed replicates are trained and averaged by default; a single split must be requested explicitly and is labelled a debug run. |
| Critical | `build_oracle()` discarded `ORIGINAL`. | The AF3 N=4 paper candidate pool was impossible to construct. | `ORIGINAL` is retained by default; exclusion is explicit and recorded in the oracle metadata. |
| High | Split files recorded member keys and no generation evidence. | A reviewer could not establish grouping, score basis, overlap, or coverage. | Split metadata records grouping basis, scoring specs, seed, sizes, search budget, and recomputed overlap. |
| High | The README claimed the full workflow while requiring an unspecified flattened manifest. | A generic JSON conversion could silently produce a non-paper split. | The manifest contract now requires composition metadata or the documented raw BEAF filename convention for paper-compatible AH Existence splits, and `prepare_splits.py` refuses a path-only manifest. |
| High | The Clotho selector was described as unsupported. | The paper's Clotho selector behaviour could not be rerun from release entry points. | The oracle, cache, and selector commands support Clotho's official partition. |

### Resolved in this pass

| Priority | Release behaviour before repair | Paper consequence | Disposition |
| --- | --- | --- | --- |
| Critical | `run_af3.py` did not set `padding_side = "left"`, while `cache_hidden_states.py` did. | Two distinct defects. Batched generation slices continuations at `input_ids.size(1)`, which is only correct under left padding, so batched AF3 decoding was scoring the wrong token positions for short rows. Separately, the selector's cached features came from a differently padded forward pass than the decoder scored, breaking the premise that the selector reads the decoder's own clean pass. | Both runners set left padding at load time, and both decoding and caching now share one encoding path (`decoding/engine.py`), so they cannot diverge again. |
| High | Neither prompt was selectable at the command line; `config.py` held both strings but nothing consumed them. | Table II could not be regenerated at all — only quoted. | `--prompt {constrained,aad}` on both runners and the parallel launcher, with non-default prompts written to a prompt-tagged directory so a comparison sweep cannot overwrite the main results. |
| High | Affirmative bias was not computed anywhere. | The mechanism the paper gives for the prompt result (predicted-yes rate falling from +40.4 to +21.0 to +1.8 points) was unmeasurable from the release. | Predicted-yes rate and affirmative bias are computed per run, stored in result metadata, and reported by `summarize_results --prompt-table`. |
| High | `RunConfig` accepted only alpha in `{0.5, 1.0}`. | The paper sweeps alpha over `[0, 2]`; that sweep could not be decoded directly. | Alpha is validated against the swept range, with 1.0 retained as the reported operating point. |
| High | Only two feature configurations were exposed (`hidden`, `embedding`); audio-encoder features were listed as an unresolved limitation. | Ten of the twelve rows of the input-feature ablation could not be run, including every mean-pooling row that carries the paper's central feature finding. | The cache now stores per-layer token-mean-pooled states and, via forward hooks on the audio tower and projector, the raw and projected audio features. All twelve ablation rows plus both cross-attention variants are runnable through `--feature-ablation`. |
| Medium | Oracle accuracy (Eq. 4) was printed as an aside and never stored; there was no oracle-at-N. | Table V's oracle column and gap could not be produced. | Oracle accuracy, best fixed branch, and per-branch accuracies are computed and stored; `--sweep-n` produces the full oracle/selector/gap grid. |
| Medium | The greedy branch-coverage search described in Section VI-A was not implemented. | The claim that the library saturates well before its full size — the basis for "data-limited, not candidate-limited" — had no supporting output. | `build_oracle --coverage` prints the greedy coverage curve and the pool size at which 99% of peak coverage is reached. |
| Medium | The distance figure plotted a single metric with no reference line. | The paper's figure shows all six metrics against the fixed no-audio baseline; the release could not draw it. | All six metrics are plotted on one axis with the no-audio baseline drawn in, and the per-example argmax is maintained incrementally so the sweep is linear rather than quadratic in pool size. |
| Medium | The perturbation module documented every transform as "deterministic". | Eight of the 36 classes draw from NumPy's global random state. The claim was simply false, and it obscured that reproducibility depends on the process seed *and* a fixed manifest order. | The reproducibility contract is stated accurately, stochastic classes are tagged in `STOCHASTIC_TYPES`, and an explicit `rng` can be threaded through for order-independent draws. The legacy global-state path is byte-identical to the previous implementation, so reported runs are unaffected. |
| Medium | The library carried no family taxonomy or paper-facing labels. | Section III-B organises the library into six families and the tables name branches in physical units; neither was recoverable from the code. | Every class declares its family and a curated label, `paper_label()` renders table-style names, and the module self-check prints the per-family breakdown. |
| Low | `prepare_splits.py` could silently return fewer replicates than requested when the refinement pool was smaller than the split count. | The selector would then refuse to aggregate, with the cause several stages upstream. | The condition is detected at generation time and reported with the flag to raise. |
| Low | Result and checkpoint writes were non-atomic. | An interrupted write leaves a truncated `eval_*.json.gz`, which the oracle builder would read as a complete branch with fewer examples and reject the whole sweep. | Both write to a temporary file and `os.replace` into position. |

## Components that already match the reported method

- The perturbation registry has the same class set and parameter settings as
  the experiment implementation, and retains 105 branches across 38 types
  under the release's counting convention — verified by executing
  `perturbations.py`, whose per-family breakdown (7 / 5 / 8 / 10 / 4 / 2
  classes) matches the manuscript's Section III-B listing exactly.
- The decoding path applies the manuscript's correction and scores the
  constrained yes/no task from step-0 yes/no logits.
- The default selector is the reported `[512, 256, 128]` MLP over
  first/middle/final last-token hidden states, with BCE, label smoothing 0.25,
  feature noise 0.10, and input dropout 0.05.
- The split search reproduces the experiment implementation's three phases,
  budgets, refinement weight ladder, and greedy coverage selection.

## Verification performed

Executed against this checkout on CPU, with no model weights and no licensed
data:

1. All modules byte-compile, and a static pass reports no unused imports or
   undefined names.
2. Every command-line entry point parses (`--help`), except `run_af3.py`,
   which fails to import on `transformers` 4.x exactly as its documented
   `transformers>=5.0` pin predicts.
3. All 104 configured perturbations run on synthetic audio, under both the
   global random state and an explicit generator, producing finite non-empty
   output.
4. The stochastic transforms are byte-identical to the pre-refactor
   implementations under the legacy global-state path, so the refactor cannot
   have shifted any reported number.
5. An end-to-end offline run over synthetic fixtures — 1,008 examples across
   504 BEAF-named compositions and 7 decoded branches — exercised manifest
   validation, the result schema, oracle construction and Eq. (4),
   composition grouping and pair-to-parent recovery, the three-phase split
   search, split verification, feature caching, and selector training across
   all thirteen feature configurations including both cross-attention
   variants.
6. `RunConfig` path scoping was checked directly: the constrained-prompt
   layout is unchanged from the previous release, non-default prompts land in
   a disjoint directory, and invalid alpha, prompt, and split combinations are
   rejected.

## Verification boundary

None of the above touches a GPU, a real LALM, or licensed audio. In
particular, the audio-encoder forward hooks are exercised only against
synthetic cache tensors here; whether the hook targets resolve on a given
`transformers` release is reported at runtime by
`scripts/cache_hidden_states.py`, and the affected ablation rows are reported
as skipped rather than silently omitted if they do not.

A claim about the paper's reported numbers still requires all four of:

1. the original converted manifests, with AH composition provenance,
2. complete decoding outputs for every intended branch and both prompts,
3. the five split artifacts generated at the default paper search budget, and
4. a GPU rerun on the licensed data.

## Reporting language

Until that end-to-end rerun exists, describe this as a **paper-protocol
reproduction implementation**, not a verified reproduction of the paper's
reported numbers. In particular: do not pool a single selector split, or a
path-only AH Existence split, and present it as a manuscript result; and do
not present a reduced-budget split search as paper-compatible — the split
metadata records which it was.
