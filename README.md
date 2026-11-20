# Annotation-Efficient Nested NER for Maintenance Text

How much labelled data does nested named entity recognition on short maintenance records actually need — and does active learning reduce it? This repository answers that question on the public **MaintIE** corpus with a span-based NER model and a pre-registered comparison of sampling strategies.

> Public data only. No AstraZeneca or other private data is used in this repository.

## Overview

- **Task:** predict all entity spans of the five top-level MaintIE types (PhysicalObject, State, Process, Activity, Property), including nested spans.
- **Model:** a pretrained Transformer encoder (`bert-base-uncased`, chosen in a pilot against RoBERTa and ModernBERT) with a span-classification head that scores every contiguous span ([`src/model.py`](src/model.py)).
- **Strategies:** random sampling, uncertainty sampling (length-controlled span entropy), and a hybrid of uncertainty + diversity (farthest-first over static BERT embeddings).
- **Evaluation:** learning curves over a token-budget grid (5–100% of the training pool), 10 seeds, area under the learning curve, budget needed to reach 90 / 95 / 98% of full-data F1, and paired Wilcoxon tests.
- **Rigour:** the split, configuration, and each protocol were frozen and tagged in Git (`v0.0-split` … `v0.9-gpu-protocol`) before the corresponding experiment was run.

## Status

| Stage | Status |
|---|---|
| Data preparation, grouped split (near-duplicates kept in one split) | Done |
| Encoder pilot and frozen configuration | Done |
| Random baseline, uncertainty sampling, hybrid strategy (10 seeds each) | Done — see [uncertainty results](#results-uncertainty-sampling-recorded-2026-10-20) and [hybrid results](#results-hybrid-strategy-recorded-2026-10-21) |
| Independent replication on NAISS GPU hardware | In progress — see the [GPU replication protocol](#gpu-replication-protocol-pre-registered-2026-10-21-before-any-gpu-experiment-run) |

## Repository layout

```text
src/       data loading, span model, budgets and pools, sampling strategies, evaluation, statistics
scripts/   data preparation, split creation, pilot, random / active-learning runs, comparisons
configs/   frozen experiment configuration (base.yaml)
data/      MaintIE gold release, processed records, frozen split
results/   per-seed result files (JSONL) and learning-curve plots
tests/     unit tests for data, budgets, strategies, hybrid selection, evaluation, statistics
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # exact versions: requirements.lock
python -m pytest tests                   # unit tests
python scripts/run_random.py             # random baseline   -> results/random.jsonl
python scripts/run_active.py --strategy uncertainty
python scripts/run_active.py --strategy hybrid
python scripts/compare_all.py            # three-way comparison and plot
```

`scripts/gpu_job.sh` is the SLURM job script used on the NAISS Arrhenius cluster.

## Limitations

- One grouped split with 215 test records; it differs from published MaintIE splits, so scores are not comparable with published benchmarks.
- The Property type has only 35 mentions in total; its per-type results are unreliable.
- Results on Apple MPS and on GPU are not identical for the same seed; the Mac results are the main results.

---

# Research log

The entries below are the chronological, pre-registered record of the project. They are kept unchanged.

## Log 10.11 — Near-Duplicate Grouping

- **Near-duplicate grouping:** Records with a token-set Jaccard similarity ≥ 0.6 are considered near-duplicates (excluding placeholders and hyphens `-`). A Union-Find algorithm is used to group connected records, ensuring that all records in the same group are assigned to the same data split.
- **Rationale:** A threshold of 0.5 resulted in a large cluster of 164 largely unrelated records, whereas 0.6 produced a maximum cluster size of 22 records with consistent content.
- **Remaining limitation:** Near-duplicates with Jaccard similarities between 0.5 and 0.6 may still appear across different splits.

## Pilot Protocol (Frozen Before the 10.12 Experiments)

- **Data split:** Training uses only the `train_pool` from `split_v1` (758 records). Model comparisons are conducted exclusively on the validation set (103 records). The test set is not loaded.
- **Candidate encoders:** `bert-base-uncased`, `roberta-base`, and `answerdotai/ModernBERT-base`.
- **Pilot random seeds:** 101, 102, and 103 (different from those used in the main experiments).
- **Encoder learning rate selection:** The encoder learning rate is selected once from `{2e-5, 5e-5}`. BERT is first evaluated using both learning rates, and the one achieving the higher mean validation micro-F1 is selected. The same learning rate is then applied to all three encoders.
- **Fixed hyperparameters (no further tuning):**
  - `head_lr = 1e-3`
  - `none_weight = 0.2`
  - `head_hidden = 256`
  - `width_dim = 16`
  - `dropout = 0.1`
  - `batch_size = 16`
  - `max_len = 64`
  - `min_steps = 300`
  - `min_epochs = 5`
- **Model selection criterion:** The encoder with the highest mean validation micro-F1 is selected. If the difference between the top two encoders is smaller than one standard deviation, preference is given to the more stable and computationally efficient encoder.

## Log 2026-10-11: near-duplicate grouping
- Records are grouped by word-set Jaccard similarity >= 0.6 (lower-cased; placeholders and "-" ignored). Groups are built with union-find and each group goes entirely into one split.
- Rationale: at 0.5 unrelated "blown ..." records chain into a 164-record group (15.2%); at 0.6 the largest group has 22 records (2.0%) and is a coherent "change out position <num> tyre" template. 0.6 is the lowest threshold without large chaining.
- Chosen using the text only, before any model result. Frozen.
- Limitation: near-duplicates with similarity between 0.5 and 0.6 can still fall on different sides of the split.

## Split v1 (frozen, tag v0.0-split)
- Ratios 70/10/20, seed 13 (1 try), hash 4b5d2d412da3e0b6.
- train_pool 758 records / 4,281 tokens; val 103 / 627; test 215 / 1,211.
- Property is rare (35 mentions in total): 26 / 4 / 5 across train_pool / val / test. Property results are reported separately and marked as unreliable; micro-F1 is the primary metric.

## Pilot rules (pre-registered before running experiments)
- Train on the full train_pool (758 records); compare on val (103 records) only; the test set is never loaded.
- Candidates: bert-base-uncased, roberta-base, answerdotai/ModernBERT-base.
- Pilot seeds: 101, 102, 103 (different from the seeds of the final experiments).
- encoder_lr is chosen once from {2e-5, 5e-5}: run BERT with both values, take the higher mean validation micro-F1, and use that value for all three encoders.
- Fixed and not tuned: head_lr 1e-3, none_weight 0.2, head_hidden 256, width_dim 16, dropout 0.1, batch_size 16, max_len 64, min_steps 300, min_epochs 5.
- Decision: highest mean validation micro-F1 wins; if the gap to the runner-up is smaller than one standard deviation, choose the more stable and cheaper encoder. A failed run is recorded as a failure with its cause.

## Pilot results (validation micro-F1, mean ± std over 3 seeds)
| Encoder | encoder_lr | micro-F1 | Time per run | Runs OK |
|---|---|---|---|---|
| BERT (uncased) | 2e-5 | 0.8948 ± 0.0054 | 152 s | 3/3 |
| BERT (uncased) | 5e-5 | 0.9205 ± 0.0020 | 150 s | 3/3 |
| RoBERTa | 5e-5 | 0.9119 ± 0.0044 | 165 s | 3/3 |
| ModernBERT | 5e-5 | 0.8834 ± 0.0069 | 215 s | 3/3 |

- Decision: bert-base-uncased with encoder_lr 5e-5 (gap to RoBERTa is 0.86 points, about 4x BERT's std). Model and hyper-parameters frozen (tag v0.1-frozen). The test set was never used.
- Limitations: the validation set has only 103 records; encoder_lr was selected on BERT only and not tuned separately for RoBERTa or ModernBERT. Pilot numbers are specific to this split and are not comparable with results from other splits.

## Protocol amendment (2026-10-13, before any experiment was run)
- The budget fill rule is changed from "stop once the budget is reached or exceeded" to the handbook rule: records are added in candidate order, a record that does not fit is skipped, and the total never exceeds the target. This aligns the actual budgets of all strategies almost exactly and supports adding records to an existing labelled set, which active-learning rounds need.
- Random baseline: for each seed the budget points are built incrementally on one random permutation (smaller sets are contained in larger ones). The nested sets are always computed over the full grid, whichever points are run.
- The 5 percent set of a seed is also the initial labelled set of the active-learning strategies with that seed.
- All other items of the previous protocol are unchanged. No experiment result existed at the time of this amendment.

## Protocol amendment 2 (2026-10-13, before any experiment was run)
- Budget grid extended from 7 to 9 points: 5, 10, 15, 20, 30, 40, 60, 80, 100 percent of the train_pool tokens.
- Rationale: a smoke run with seed 1, judged on the validation set only, gave F1 of 0.703 / 0.796 / 0.845 / 0.910 at 5 / 10 / 20 / 100 percent. The 90 percent threshold lies between 10 and 20 percent and the 95 percent threshold between 20 and 40 percent, so points were added at 15 and 30 percent to reduce interpolation error in the budget-to-reach metric.
- No test-set result was used for this decision. No result of the full experiment existed at the time of this amendment.

## Active-learning protocol (pre-registered 2026-10-16, before any active-learning run)
- Rounds follow the budget grid (5, 10, 15, 20, 30, 40, 60, 80 percent). In each round the model trained on the current labelled set is evaluated on the test set (the learning-curve point) and is then used to score the remaining pool; the next batch fills the gap to the next budget point with fill_budget in ranked order. One batch per round, no retraining inside a round.
- Initial labelled set: the 5 percent set of the random baseline with the same seed. The 5 percent point and the 100 percent point are shared with the random baseline.
- Uncertainty: entropy of the 6-class softmax of every candidate span. Record score = mean of the L largest span entropies, where L is the number of words of the record (length-controlled). Ties are broken by the smaller record id.
- Alternatives (maximum span entropy, fixed top-3 mean) are not part of the main experiment. If they are explored later they are reported as an ablation and never used to change the main strategy.
- Seeds 1-10, as in the random baseline.
## Note on the configuration freeze and result files (added 2026-10-17)
- The pilot decision (bert-base-uncased, encoder_lr 5.0e-5) was recorded in README.md and results/pilot.json in commit e5265eb (tag v0.1-frozen, 2026-10-11). The configs/base.yaml in that commit still contained null for encoder and encoder_lr. The filled file was used for all later experiments, but it was committed only on 2026-10-17; tag v0.1b-config marks the commit that contains it. No other value of the configuration changed after the freeze.
- The random baseline (results/random.jsonl) was run on 2026-10-13 and committed on 2026-10-17 (tag v0.3-random-baseline). The file was not modified in between.

## Results: uncertainty sampling (recorded 2026-10-20)

The full uncertainty run (10 seeds, budgets 10-80%; 5% and 100% shared with the random baseline) is finished and is in `results/uncertainty.jsonl`. Comparison script: `scripts/compare.py`, plot: `results/compare_curve.png`. No setting was changed after seeing these numbers.

| Budget | Random | Uncertainty | Diff |
|---|---|---|---|
| 5% | 0.7305 | 0.7305 | 0.0000 |
| 10% | 0.7940 | 0.8120 | +0.0179 |
| 15% | 0.8232 | 0.8429 | +0.0197 |
| 20% | 0.8352 | 0.8544 | +0.0192 |
| 30% | 0.8519 | 0.8705 | +0.0185 |
| 40% | 0.8633 | 0.8799 | +0.0165 |
| 60% | 0.8751 | 0.8874 | +0.0123 |
| 80% | 0.8807 | 0.8886 | +0.0079 |
| 100% | 0.8866 | 0.8866 | 0.0000 |

- AULC (5-60%): random 0.8436 +- 0.0051, uncertainty 0.8597 +- 0.0036; the difference is +0.0162, uncertainty is better in 10 of 10 seeds, Wilcoxon p = 0.0020 (the smallest value possible with 10 seeds).
- Budget to reach 90 / 95 / 98% of the reference F1 (0.8866): random 10.7 / 24.2 / 49.4%, uncertainty 9.1 / 14.9 / 29.0%.
- Caveats: one split, 215 test records, public data only. The value at 80% (0.8886) is slightly above the 100% reference; the gap of 0.002 is within noise and is not interpreted.

## Hybrid strategy protocol (pre-registered 2026-10-20, before any hybrid run)

Everything not listed here is identical to the uncertainty run: same split, same frozen config, same seeds 1-10, same budget grid, same initial 5% random set per seed, 5% and 100% points shared with the random baseline, same evaluation.

1. After each training round, score the remaining pool exactly as in the uncertainty strategy and rank it.
2. Candidate set: take records in uncertainty order until their total tokens reach 3 times the tokens still needed to hit the next budget point.
3. Record embedding: the mean of the static input word-piece embeddings of the pre-trained bert-base-uncased (not fine-tuned, so it never depends on training or test labels), L2-normalised. Distance = 1 - cosine similarity.
4. Selection: farthest-first among the candidates. Repeatedly pick the candidate whose smallest distance to the already labelled records and the records picked in this round is largest. Ties go to the lower record id.
5. A picked record that does not fit in the remaining budget is skipped (same rule as `fill_budget`), and selection continues. If the candidates are used up before the budget is exactly filled, the remaining space is filled from the next records in uncertainty order using `fill_budget`.
6. The factor 3 and the embedding are fixed now. They are not tuned. Only this one configuration is run.
7. Primary comparisons, same as before: AULC over 5-60%, budget to reach 90 / 95 / 98% of the reference, paired Wilcoxon test over 10 seeds, for hybrid vs uncertainty and hybrid vs random. Whatever the outcome, it is reported.

## Results: hybrid strategy (recorded 2026-10-21)

The hybrid run (10 seeds, budgets 10-80%; 5% and 100% shared with the random baseline) is finished and is in `results/hybrid.jsonl`. Three-way comparison: `scripts/compare_all.py`, plot `results/compare_all_curve.png`. The protocol of 2026-10-20 was not changed after seeing these numbers.

| Target (share of reference F1 0.8866) | Random | Uncertainty | Hybrid |
|---|---|---|---|
| 90% | 10.7% | 9.1% | 9.3% |
| 95% | 24.2% | 14.9% | 17.6% |
| 98% | 49.4% | 29.0% | 31.7% |
| AULC (5-60%) | 0.8436 | 0.8597 | 0.8552 |

- Hybrid vs random: AULC +0.0116, better in 10 of 10 seeds, Wilcoxon p = 0.0020.
- Hybrid vs uncertainty: AULC -0.0045, better in only 2 of 10 seeds, Wilcoxon p = 0.0137.
- Result: adding diversity (farthest-first over static BERT embeddings) did not improve on plain uncertainty sampling in this setup. No reason was tested; possible explanations (the static embedding is a poor similarity measure, or the candidate set of 3 times the needed tokens drops some of the most uncertain records) are hypotheses only.
- Caveats as before: one split, 215 test records, public data only, no correction for the three comparisons.

## GPU replication protocol (pre-registered 2026-10-21, before any GPU experiment run)

Purpose: check whether the three-way comparison holds on different hardware and software. This is an independent replication, not a change to the experiment.

- Same code, same frozen config, same split, same seeds 1-10, same budget grid and the same protocols for random, uncertainty and hybrid as before. The only change is the device.
- Environment: NAISS Arrhenius GPU partition, NVIDIA GH200 (aarch64), Python 3.13.5, torch 2.9.1, numpy 2.4.6, transformers 5.19.0. The earlier runs used a Mac (MPS), Python 3.11, torch 2.14.1.
- Results are written to separate files in `results/gpu/` and are never merged with the Mac rows. The Mac results remain the main results. The GPU results are reported next to them.
- Hardware check on 2026-10-21: seed 1, 5% budget, uncertainty. Mac test micro-F1 0.7197, GPU 0.7127. Same-seed results are not identical across hardware, so no row-by-row comparison is made.
- Analysis: the same `compare_all.py`, with the same metrics, tests and thresholds. Agreement or disagreement between the two environments is reported either way. No setting is changed after seeing the GPU results.
- Private data: no AstraZeneca data is used or copied to the cluster.

## Note on the strategy label in hybrid result files (added 2026-10-31)
- Commit 08ec3f3 wrote the literal string "a.strategy" instead of the strategy name into the `strategy` field of every row produced by `scripts/run_active.py`. All 80 rows of `results/hybrid.jsonl` (and the 2 rows of `results/smoke_hybrid.jsonl`) carry this wrong label; all other fields are correct. The uncertainty results were produced before that commit and are labelled correctly.
- The analysis scripts select results by file name, not by this field, so no reported number is affected. The result files are left unchanged; the code is fixed before the GPU replication, so the `results/gpu/` files will carry the correct label.

## Results: GPU replication (run 2026-10-21, recorded 2026-11-16)

The three runs of the GPU replication protocol finished on 2026-10-21 (NAISS Arrhenius, GH200; jobs 4188777-4188779) and are in `results/gpu/`; they were copied back to the repository only on 2026-11-16 and were not changed. Analysis with the same `scripts/compare_all.py` (now taking the result folder as an argument: `python scripts/compare_all.py results/gpu`), plot `results/gpu/compare_all_curve.png`.

| | Mac (main results) | GPU |
|---|---|---|
| Reference F1 (random, 100%) | 0.8866 | 0.8882 |
| AULC random / uncertainty / hybrid | 0.8436 / 0.8597 / 0.8552 | 0.8433 / 0.8571 / 0.8546 |
| Uncertainty - random | +0.0162, 10 of 10 seeds, p = 0.0020 | +0.0138, 9 of 10 seeds, p = 0.0039 |
| Hybrid - random | +0.0116, 10 of 10 seeds, p = 0.0020 | +0.0113, 9 of 10 seeds, p = 0.0039 |
| Hybrid - uncertainty | -0.0045, 2 of 10 seeds, p = 0.0137 | -0.0025, 2 of 10 seeds, p = 0.1934 |
| Budget to 90 / 95 / 98% of reference: random | 10.7 / 24.2 / 49.4% | 11.3 / 24.5 / 52.4% |
| uncertainty | 9.1 / 14.9 / 29.0% | 9.5 / 16.7 / 32.1% |
| hybrid | 9.3 / 17.6 / 31.7% | 9.3 / 17.9 / 34.6% |

- Agreement: on both environments uncertainty and hybrid beat random in AULC with p < 0.01, and both save labelling to reach 95% and 98% of the reference.
- Disagreement: hybrid is again lower than uncertainty and better in only 2 of 10 seeds, but on the GPU the difference is smaller (-0.0025) and not significant (p = 0.19). The Mac finding "hybrid is worse than uncertainty" is therefore not robust to the change of hardware and software; it is reported as a weak difference, not as an established one.
- The GPU runs used the code before the label fix of 2026-10-31, so the `strategy` field of `results/gpu/uncertainty.jsonl` and `results/gpu/hybrid.jsonl` reads "a.strategy"; all other fields are correct.

## LLM annotation protocol (pre-registered 2026-11-16, before any LLM call)

Purpose: measure how well LLMs label the MaintIE entities, whether their errors can be found before a human checks them, and whether active learning still helps when the training labels come from an LLM. Human labels stay the reference throughout: no LLM output is ever used as the evaluation standard. Only MaintIE is used; no AstraZeneca data.

Everything not listed here is unchanged: split_v1, the frozen config, seeds 1-10, the budget grid, the initial 5% set per seed, the selection strategies and the human-labelled test set (215 records). The existing human-label results (`results/random.jsonl`, `uncertainty.jsonl`, `hybrid.jsonl`) are the comparison and are not rerun.

### Models
- Eight models in five families; three families with a large and a small model:

| Family | Large | Small |
|---|---|---|
| OpenAI | current generally available flagship | its mini model |
| Anthropic | Claude Opus 5.5 | Claude Haiku 5.5 |
| Qwen | flagship via API | Qwen3.8-27B, run locally |
| Google | current generally available Pro model | - |
| DeepSeek | deepseek-v4-pro | - |

- The exact model IDs for the OpenAI, Google and Qwen slots are taken from the providers' official model lists (generally available models only, no preview or restricted models) and recorded in an amendment to this section before the first LLM call. The list is not changed afterwards. If a model cannot be run, that is reported with its cause and it is not replaced.
- Settings: temperature 0 where the API allows it, otherwise the provider default (recorded). Reasoning or thinking modes are switched off, or set to the lowest level where they cannot be switched off (recorded). One run per model. Model ID, call date, settings and every raw response are stored.

### Labelling task (step 1)
- Input: the tokens of one record with their indices; output: a list of spans `[start, end, type]` over token indices, type one of PhysicalObject, Activity, State, Process, Property. Same span format as `gold.jsonl`; nested spans are allowed, as in the gold data.
- One prompt for all models, with the five type definitions and a few examples taken from validation records. The prompt is tuned on at most 20 validation records, then frozen before any train_pool record is sent. The test set is never sent to any LLM.
- Validation: a span with indices outside the record, start >= end, an unknown type or an exact duplicate is dropped and counted. A response that cannot be parsed is retried at most twice; after that the record gets no spans and is counted as a failure. Drop and failure counts are reported per model.
- Each model labels all 758 train_pool records. Labels are stored per model in the `gold.jsonl` format.

### RQ1: label quality (step 2)
- Exact-match span micro-F1 against the human labels on train_pool, per model; also per entity type (Property is rare, 26 mentions, and marked unreliable) and per record-length tertile.
- Error types: missed span, extra span, boundary error (same type, overlapping span), type error (same span, other type).
- Shared errors: the share of human spans that all eight models miss, and the share of model spans that at least six models agree on but that are wrong.
- Bootstrap 95% intervals over records (1,000 resamples).

### RQ2: finding LLM errors before review (step 3)
- Unit: one record of one model's labels. A record is in error when that model's spans differ from the human spans in any way.
- Pool: the train_pool records outside the 5% initial set of the seed.
- Signals, computed per model and seed:
  1. model uncertainty: the existing record score (mean of the L largest span entropies) from the NER model trained on the 5% initial set, using that LLM's labels for the 5% set;
  2. LLM disagreement: 1 minus the mean span F1 between this model's labels and each of the other seven models' labels on the record;
  3. combined: the mean of the two signals' ranks.
- Measures: AUROC of each signal for "record in error"; and the share of all span errors of the model that a reviewer would see when checking the top k records by the signal, for k = 50, 100 and 200, compared with the expected share for k random records. Results are averaged over seeds 1-10 and reported per model.

### RQ3: active learning with LLM labels (step 4)
- Two models: the highest and the lowest train_pool micro-F1 in RQ1 (ties: the smaller model, then alphabetical ID).
- For each of the two, random, uncertainty and hybrid are run with that model's labels as the training labels (initial 5% set, budget grid and seeds as before; the selection and the budget use the same records and tokens as before). Evaluation on the human test set.
- Primary comparisons, paired over seeds 1-10 with the Wilcoxon signed-rank test:
  - AULC (5-60%) of uncertainty vs random with LLM labels;
  - AULC of each strategy with LLM labels vs the same strategy with human labels;
  - the active-learning gain (uncertainty minus random AULC) with LLM labels vs with human labels.
- Budget to reach 90 / 95 / 98% of the human-label reference F1 (0.8866), so all curves share one reference.
- Run on the same Mac environment as the human-label results.

### Rules
- No setting in this section is changed after seeing any LLM label or result. Any change before that is recorded as a dated amendment with its reason.
- All outcomes are reported, including null or negative ones. There is no correction for multiple comparisons; the number of comparisons is stated with the results.
- Limitations stated in advance: MaintIE is public, so LLMs may have seen it during training and their scores may be too high; MaintIE has no status labels, so errors such as hedged text labelled as confirmed are not covered; one dataset, short English records.

### Amendment 1 (2026-11-16, before any LLM call): model IDs and settings
Checked on 2026-11-16 against the providers' official pages (OpenAI developers.openai.com/api/docs/models and changelog; Google aistudio.google.com/docs/models and changelog; Alibaba Cloud Model Studio model list; Hugging Face Qwen organisation; DeepSeek API change log).

| Slot | Model ID | Sampling | Reasoning / thinking |
|---|---|---|---|
| OpenAI large | `gpt-6-astra` | no custom temperature (provider default) | `low` (lowest; `none` not supported) |
| OpenAI small | `gpt-6-luna` | temperature 0 if accepted, else default | `none` |
| Anthropic large | `claude-opus-5-5` | temperature 0 | extended thinking off |
| Anthropic small | `claude-haiku-5-5` | temperature 0 | extended thinking off |
| Qwen large | `qwen4-max` (Model Studio, international) | temperature 0 | `enable_thinking` false |
| Qwen small | `Qwen/Qwen3.8-27B`, BF16 weights, self-hosted on the NAISS GPU node | temperature 0 (greedy) | thinking off (`enable_thinking` false) |
| Google | `gemini-4-carbon` | no custom temperature (ignored by the API) | `thinking_level` low (lowest; cannot be switched off) |
| DeepSeek | `deepseek-v4-pro` | temperature 0 | non-thinking mode if offered, else default |

Notes on the choices:
- OpenAI has no model named "mini" in the current list; `gpt-6-luna` is its smallest generally available text model.
- Google has no generally available "Pro" model (Gemini 3.1 Pro is preview only). `gemini-4-carbon` is described as Google's most capable reasoning model and is GA since 2026-11-10.
- `qwen4-max` is the flagship text model on Model Studio; the page does not label it stable or preview. The small Qwen slot is the newest open-weight generation (3.8), so the Qwen large/small pair also differs in generation; this is reported with the size comparison.
- The Qwen small model runs on the NAISS GPU node, not on the Mac (24 GB is too little for BF16 weights, and a quantised model would add a second difference). MaintIE is public data, so this is allowed by the GPU protocol.
- `gpt-6-astra` and `gemini-4-carbon` cannot switch reasoning off or set a custom temperature. They run at their lowest reasoning level and default sampling; this difference from the other models is reported with the results.
- The settings actually sent and the provider's returned model version are logged with every response.

### Amendment 2 (2026-11-16, before any LLM call): training runs on the NAISS GPU
- All NER training in this protocol (the uncertainty signal of RQ2 and all runs of RQ3) runs on the NAISS GPU environment of the GPU replication protocol, not on the Mac. Reason: about 560 training runs, too slow on the Mac.
- Because same-seed results differ between the Mac and the GPU (GPU replication protocol, hardware check), RQ3 compares GPU with GPU only: the human-label reference curves for RQ3 are the GPU replication results in `results/gpu/` (random, uncertainty, hybrid), and budget-to-reach uses the GPU human-label 100% value as reference. The GPU replication is therefore run before RQ3. The Mac human-label results stay the main results of the earlier protocols and are not mixed with GPU rows.
- LLM labelling through provider APIs is run from the Mac (no computation there; compute nodes may have no internet access). The self-hosted `Qwen/Qwen3.8-27B` runs on the NAISS GPU, as in amendment 1.
- Nothing else changes.

### Amendment 3 (2026-11-16, before any LLM call): Claude settings and output limit
- Checked against Anthropic's current API reference: the current Claude models accept no sampling parameters (`temperature`, `top_p`, `top_k`), so amendment 1's "temperature 0" for the two Claude models cannot be sent. Claude Opus 5.5 cannot switch thinking off; it runs at the lowest effort (`output_config.effort` `low`). Claude Haiku 5.5 runs with thinking switched off (`thinking` `disabled`). Both use default sampling. Claude Opus 5.5 therefore joins `gpt-6-astra` and `gemini-4-carbon` in the group that keeps a minimum of reasoning; this is reported with the results.
- No server-side fallback to another model is enabled for any provider: a fallback would mix another model's labels into a model's file. A refusal counts as an unparseable reply (retried, then a failure) and its stop reason is logged.
- The output limit is 16,000 tokens per call for every model, so that reasoning tokens (which count towards the limit) cannot cut an answer short. Only generated tokens are billed.

### Amendment 4 (2026-11-17, before any LLM call): Claude, Gemini and qwen4-max through OpenRouter
- The two Claude models, `gemini-4-carbon` and `qwen4-max` are called through OpenRouter (`https://openrouter.ai/api/v1`, OpenAI-compatible chat API) instead of the Anthropic, Google and Alibaba APIs. Reason: one account instead of three; the model versions are the same (OpenRouter IDs `anthropic/claude-opus-5.5`, `anthropic/claude-haiku-5.5`, `google/gemini-4-carbon`, `qwen/qwen4-max`, checked in OpenRouter's public model list on 2026-11-17). OpenAI and DeepSeek stay on their own APIs; `Qwen/Qwen3.8-27B` stays self-hosted on the NAISS GPU.
- Each routed model is pinned to the model maker's own backend (`anthropic`, `google-ai-studio`, `alibaba`) with `allow_fallbacks` false, so no request is served by another host (for example a cloud reseller or a quantised copy), and `require_parameters` true, so a backend that would ignore a setting is not used. The backend that served each reply (`served_by`) is logged with the reply.
- The settings of amendments 1 and 3 are sent in OpenRouter's form: `reasoning.effort` `low` for Claude Opus 5.5 and `gemini-4-carbon`, `reasoning.enabled` false for Claude Haiku 5.5 and `qwen4-max`, temperature 0 for `qwen4-max` only. The smoke test (one tuning record per model) checks from the returned usage that thinking is off where it should be; if a setting is not honoured, this is recorded in a further amendment before prompt tuning.
- OpenRouter adds a fee on credit purchases, not on tokens; per-token prices are the providers' prices.

### Amendment 5 (2026-11-17, before any labelling): OpenAI and DeepSeek through OpenRouter as well
- The OpenAI and DeepSeek API keys available to the project were not accepted by the official APIs, so `gpt-6-astra`, `gpt-6-luna` and the DeepSeek model are also called through OpenRouter, pinned to the model maker's own backend (`openai`, `deepseek`) in the same way as amendment 4. All seven API models now use one OpenRouter account. The only calls made so far are the one-record smoke tests, which are not used in any result.
- DeepSeek: on OpenRouter only `deepseek/deepseek-v4-pro-0813` (described by OpenRouter as the GA release of DeepSeek V4 Pro) is served by DeepSeek itself; the undated `deepseek/deepseek-v4-pro` entry is served only by third-party hosts, most of them quantised. The DeepSeek slot therefore uses `deepseek/deepseek-v4-pro-0813`. DeepSeek's backend may use inputs for training, so the OpenRouter account allows such backends while this protocol's labelling runs. MaintIE is public data; no other data is sent through this account while the setting is on.
- `gpt-6-luna`: OpenRouter's OpenAI backend accepts no temperature parameter, so it runs at the provider default (amendment 1 allowed this: "temperature 0 if accepted, else default"). `reasoning.effort` is `none` for `gpt-6-luna` and `low` for `gpt-6-astra`; DeepSeek runs with `reasoning.enabled` false and temperature 0.
- Smoke test (2026-11-17, one tuning record each): all models that ran returned valid JSON, were served by the pinned backend, and reported 0 reasoning tokens.

### Amendment 6 (2026-11-17, before labelling train_pool): prompt frozen
- The labelling prompt is frozen as `configs/prompts/label_v2.md` (SHA-1 `58c5057f9899`…), with the 5 fixed val examples (seed 2026). Each raw record also logs the SHA-1 of the full prompt it was sent.
- Tuning log: 2 versions were tried, below the limit of 3. Both runs are kept in `data/llm/tune/label_v1/` and `data/llm/tune/label_v2/`. No reply failed to parse and no span was dropped in either version. v2 adds three rules after reading the v1 errors: a word that only says what kind of object it is is not marked on its own, position words are not part of an object, and words saying that work is needed are State.

| Model | v1 micro-F1 | v2 micro-F1 |
|---|---|---|
| `gpt-6-astra` | 0.841 | 0.870 |
| `gpt-6-luna` | 0.894 | 0.806 |
| `claude-opus-5-5` | 0.853 | 0.891 |
| `claude-haiku-5-5` | 0.820 | 0.885 |
| `gemini-4-carbon` | 0.832 | 0.879 |
| `qwen4-max` | 0.773 | 0.868 |
| `deepseek-v4-pro` | 0.754 | 0.783 |
| mean | 0.824 | 0.855 |

- v2 is better on mean F1 and for six of the seven models; `gpt-6-luna` is worse with v2. All models keep the same prompt, as pre-registered. The 20 records hold about 60 gold spans, so differences of a few points between models are within noise and are not a result of this protocol; RQ1 uses train_pool.
- No third version was tried: most remaining v1 and v2 errors are on nested object names where the gold labels themselves are inconsistent (for example "steering cylinder" has "cylinder" nested, "stick cylinder" does not), which a prompt rule cannot fix without fitting these 20 records.
- `Qwen/Qwen3.8-27B` (self-hosted on NAISS) was not part of tuning; it uses the frozen v2 prompt unchanged.
- Cost of tuning: about USD 0.39 for the seven API models.

### Amendment 7 (2026-11-17, after the train_pool replies were logged, before any analysis): replies with extra text
- In 10 of the 5,306 train_pool labelling jobs (7 models × 758 records) every reply failed the strict JSON format after the two pre-registered retries: Claude Haiku 5.5 3, Claude Opus 5.5 5, `qwen4-max` 2. In each case the reply held a valid spans object, with an explanation written before it (Claude) or an end token `</s>` after it (Qwen).
- Both results are kept and reported. The strict result (`failed` in `data/llm/raw/`) is the format-compliance number of RQ1. For the labels used in RQ1–RQ3, `scripts/recover_lenient.py` parses the logged replies of these records leniently (`parse_reply_lenient`: the last JSON object with a `spans` list, checked with the same span rules), taking the first reply of the record that parses. No model was called again. All 10 records were recovered; they are listed in `data/llm/recovered.jsonl`.
- This rule was decided after seeing that these replies existed, but before computing any agreement with gold. It treats a format slip separately from labelling quality, so that a model is not scored as having found no entities because of a sentence written before its answer.
- Labelling notes: every reply was served by the pinned backend and returned the expected model version. Network and rate-limit errors (new-account limits and two credit stops) were retried or resumed and are not counted as failures. Claude Opus 5.5 used 23,551 reasoning tokens over the 758 records and `gpt-6-astra` 6,268 at their lowest settings; the other five models used none. Cost of the train_pool labelling: USD 15.22 (`gpt-6-astra` 8.07, Claude Opus 5.5 4.75, `gemini-4-carbon` 1.60, others below 0.60).

### Labelling note (2026-11-17): Qwen3.8-27B on the NAISS GPU
- `Qwen/Qwen3.8-27B` labelled the 20 tuning records (job 5417063) and train_pool (job 5417439) on one GH200 with the frozen prompt v2, BF16 weights, greedy decoding and thinking off, via `scripts/gpu_label_job.sh`. The weights (about 56 GB) are kept in the project's storage, not in the 30 GB home. Before the runs, two loading problems were fixed without changing any setting: the model is loaded with its tokenizer only (the full processor needs torchvision for video), and it is moved to the GPU without `device_map` (no accelerate on NAISS).
- train_pool: 758 records, no reply failed the strict format, no span dropped, no retry; no reply contained a think block (logged per reply); at most 67 new tokens per reply. Tuning records: 20 records, no failure, micro-F1 0.807 (the tuning files stay on NAISS and are not used in any result).
- All eight models have now labelled train_pool.

### Amendment 9 (2026-11-17, before any RQ1–RQ3b computation): three definitions left open
The exploratory figures above were drawn before this amendment; the definitions below were fixed before any of the pre-registered RQ1–RQ3b numbers were computed.
1. RQ1 error types. Each model span that is not an exact match of a human span is one error: a **type error** if a human span has the same start and end; otherwise a **boundary error** if it overlaps a human span of the same type; otherwise a **boundary-and-type error** if it overlaps only human spans of other types (this case is not in the pre-registered list and is reported as a fifth type, not merged or dropped); otherwise an **extra span**. Each human span that no model span overlaps is a **missed span**.
2. RQ1 shared errors. Two spans are the same when start, end and type are equal. "Wrong spans that at least six models agree on" is the number of distinct spans produced by at least six of the eight models and not in the human labels, divided by the number of distinct spans produced by at least six models. "Human spans that all eight models miss" counts human spans that no model produces exactly, divided by all human spans in train_pool.
3. RQ2 "record in error" and error counts. A record is in error for a model when its span set differs from the human span set in any way. The number of span errors in a record is the number of the model's spans not in the human set plus the number of human spans not in the model's set.
4. RQ1 length tertiles: train_pool records are split by token count at the 1/3 and 2/3 quantiles (records equal to a cut point go to the lower group).

## Results: RQ1, label quality (computed 2026-11-17)
`scripts/rq1.py` → `results/llm/rq1.json`, `results/figures/rq1_f1.png`. Exact-match span F1 against the human labels on the 758 train_pool records, bootstrap 95% intervals over records (1,000 resamples, seed 0). Definitions as in amendment 9.

| Model | micro-F1 [95% CI] | PhysicalObject | State | Process | Activity | Property* |
|---|---|---|---|---|---|---|
| `gemini-4-carbon` | 0.836 [0.824, 0.848] | 0.771 | 0.913 | 0.910 | 0.965 | 0.765 |
| `gpt-6-astra` | 0.831 [0.819, 0.842] | 0.768 | 0.898 | 0.930 | 0.958 | 0.788 |
| Claude Opus 5.5 | 0.820 [0.808, 0.833] | 0.744 | 0.904 | 0.935 | 0.964 | 0.769 |
| `qwen4-max` | 0.789 [0.774, 0.804] | 0.702 | 0.908 | 0.893 | 0.941 | 0.712 |
| Claude Haiku 5.5 | 0.779 [0.763, 0.795] | 0.704 | 0.874 | 0.860 | 0.907 | 0.706 |
| `gpt-6-luna` | 0.765 [0.749, 0.781] | 0.678 | 0.884 | 0.856 | 0.915 | 0.676 |
| `deepseek-v4-pro` | 0.751 [0.735, 0.769] | 0.651 | 0.848 | 0.834 | 0.924 | 0.453 |
| `qwen3.8-27b` (self-hosted) | 0.714 [0.696, 0.732] | 0.582 | 0.874 | 0.886 | 0.898 | 0.719 |

\* Property: 26 human spans in train_pool; unreliable.

- In every family the large model scores higher than the small one (OpenAI 0.831 vs 0.765, Anthropic 0.820 vs 0.779, Qwen 0.789 vs 0.714). The intervals of the top three models overlap; `gemini-4-carbon` and `gpt-6-astra` are not distinguishable.
- PhysicalObject is the weakest type for every model (0.58–0.77); Activity the strongest (0.90–0.97).
- Length tertiles (373 short, 173 medium, 212 long records): medium-length records score lowest for seven of the eight models; the differences are small (mostly 0.02–0.05). Values per model in `rq1.json`.
- Error types (share of each model's errors): boundary errors are 60–83%; the three best models have 82–83% boundary errors and 2% type errors; Claude Haiku 5.5, DeepSeek and `qwen3.8-27b` miss more spans (15–16%); DeepSeek has the most type errors (9%).
- Shared errors: 1,994 distinct spans are produced by at least six models; 254 of them (12.7%) are not in the human labels. 120 of the 2,398 human spans (5.0%) are produced exactly by none of the eight models.
- Models for RQ3 (highest and lowest micro-F1, by the point estimate as pre-registered): `gemini-4-carbon` and `qwen3.8-27b`.

## Results: RQ2, finding LLM errors before review (computed 2026-11-19)
`scripts/rq2_uncertainty.py` on the NAISS GPU (80 runs, 8 models × seeds 1–10) → `results/gpu/llm/rq2_uncertainty.jsonl`; `scripts/rq2.py` → `results/llm/rq2.json` and `results/llm/rq2_signals.json` (the combined signal used by RQ3b). Pool: 717–721 records per seed. Means over seeds 1–10; AUROC ranges over seeds in brackets.

| Model | records in error | AUROC uncertainty | AUROC disagreement | AUROC combined | top 50 / 100 / 200, disagreement | top 50 / 100 / 200, combined |
|---|---|---|---|---|---|---|
| `gemini-4-carbon` | 412 of 719 | 0.639 [0.608, 0.665] | 0.796 [0.792, 0.800] | 0.769 [0.750, 0.778] | 14% / 28% / 49% | 15% / 27% / 47% |
| `gpt-6-astra` | 423 | 0.642 [0.584, 0.673] | 0.809 [0.806, 0.814] | 0.777 [0.749, 0.793] | 12% / 26% / 48% | 14% / 26% / 46% |
| Claude Opus 5.5 | 425 | 0.643 [0.607, 0.684] | 0.812 [0.805, 0.815] | 0.780 [0.761, 0.806] | 15% / 27% / 48% | 15% / 26% / 47% |
| `qwen4-max` | 426 | 0.633 [0.605, 0.683] | 0.825 [0.821, 0.828] | 0.785 [0.771, 0.810] | 17% / 31% / 55% | 15% / 28% / 50% |
| `deepseek-v4-pro` | 444 | 0.613 [0.570, 0.671] | 0.843 [0.837, 0.846] | 0.785 [0.766, 0.807] | 16% / 28% / 50% | 15% / 25% / 46% |
| Claude Haiku 5.5 | 451 | 0.618 [0.565, 0.653] | 0.840 [0.834, 0.845] | 0.784 [0.754, 0.808] | 16% / 30% / 50% | 15% / 26% / 46% |
| `gpt-6-luna` | 465 | 0.630 [0.591, 0.688] | 0.870 [0.868, 0.872] | 0.809 [0.792, 0.840] | 16% / 30% / 52% | 14% / 26% / 47% |
| `qwen3.8-27b` | 473 | 0.610 [0.571, 0.653] | 0.908 [0.906, 0.912] | 0.827 [0.800, 0.852] | 14% / 28% / 51% | 14% / 26% / 47% |

Checking k random records shows on average k / N of the errors: 7% / 14% / 28% for k = 50 / 100 / 200.

- All three signals find errors better than chance for every model. Checking the top 200 records (about 28% of the pool) by either disagreement or the combined signal shows 46–55% of a model's span errors, against 28% for random records.
- LLM disagreement is the strongest signal (AUROC 0.80–0.91). Model uncertainty is weak (0.61–0.64) and varies more between seeds, because it comes from an NER model trained on only 37–41 records.
- The combined signal has a lower AUROC than disagreement alone for every model and every seed (80 of 80). The weak uncertainty signal pulls it down. For the top-k shares the difference is smaller (0–5 points). RQ3b uses the combined signal as pre-registered (amendment 8); it is not replaced by disagreement.
- Disagreement AUROC is highest for the weakest models (`qwen3.8-27b` 0.91, `gpt-6-luna` 0.87) and lowest for the strongest (`gemini-4-carbon` 0.80). A weak model's mistakes are often ones the other seven do not make; a strong model's remaining mistakes are more often shared with the others (RQ1: 12.7% of the spans produced by at least six models are wrong), so disagreement cannot see them.
- More than half the records are in error for every model (412–473 of about 719), because one wrong span puts a record in error. AUROC at the record level therefore says how well a signal separates records with any error from records with none; the top-k shares count span errors.

## Results: RQ3, active learning with LLM labels (computed 2026-11-20)
NAISS GPU, jobs 5419498 and 5419523–5419527 → `results/gpu/llm/<model>/{random,uncertainty,hybrid}.jsonl`; `scripts/rq3.py` → `results/llm/rq3.json`, `results/figures/rq3_curves.png`. Human-label comparison and reference: the GPU replication runs in `results/gpu/` (amendment 2), reference F1 0.8882; the human-label numbers below are the same as in the GPU replication table. AULC over 5–60% of the training tokens, mean ± sd over seeds 1–10. Test labels are human in every case.

| Training labels | AULC random | AULC uncertainty | AULC hybrid | F1 with all of train_pool |
|---|---|---|---|---|
| human | 0.8433 ± 0.0060 | 0.8571 ± 0.0044 | 0.8546 ± 0.0045 | 0.8882 |
| `gemini-4-carbon` | 0.7696 ± 0.0054 | 0.7759 ± 0.0030 | 0.7712 ± 0.0041 | 0.7956 |
| `qwen3.8-27b` | 0.6724 ± 0.0098 | 0.6726 ± 0.0057 | 0.6655 ± 0.0059 | 0.7079 |

Pre-registered comparisons (paired Wilcoxon over seeds 1–10; 10 tests, no correction for multiple comparisons):

| | `gemini-4-carbon` | `qwen3.8-27b` |
|---|---|---|
| uncertainty − random, LLM labels | +0.0063, 8 of 10 seeds, p = 0.020 | +0.0002, 4 of 10, p = 0.70 |
| random: LLM − human labels | −0.0737, 0 of 10, p = 0.002 | −0.1710, 0 of 10, p = 0.002 |
| uncertainty: LLM − human labels | −0.0812, 0 of 10, p = 0.002 | −0.1845, 0 of 10, p = 0.002 |
| hybrid: LLM − human labels | −0.0834, 0 of 10, p = 0.002 | −0.1891, 0 of 10, p = 0.002 |
| active-learning gain (uncertainty − random): LLM − human | −0.0075, 2 of 10, p = 0.020 | −0.0135, 1 of 10, p = 0.006 |

Not pre-registered: hybrid − random with LLM labels, +0.0016 (6 of 10, p = 0.56) for `gemini-4-carbon` and −0.0069 (3 of 10, p = 0.11) for `qwen3.8-27b`.

Budget to reach 90 / 95 / 98% of the reference (0.799 / 0.844 / 0.870): no curve with LLM labels reaches even 90%, at any budget, for either model. Human labels: random 11.3 / 24.5 / 52.4%, uncertainty 9.5 / 16.7 / 32.1%, hybrid 9.3 / 17.9 / 34.6%.

- Training on LLM labels costs a lot of accuracy at every budget: 0.07–0.08 AULC with the best LLM and 0.17–0.19 with the weakest, in every seed.
- The ceiling is low. With all 758 records labelled by `gemini-4-carbon` the NER model reaches 0.796, about what 10% of the text with human labels gives (random 0.792, about 76 records). With all 758 records labelled by `qwen3.8-27b` it reaches 0.708, below the 5% initial set with human labels (0.731, about 39 records).
- Active learning still helps with the better LLM's labels, but less: uncertainty beats random by 0.006 AULC (human labels 0.014). With the weaker LLM's labels it does not help at all. The gain is smaller with LLM labels than with human labels for both models.
- The NER model trained on `gemini-4-carbon` labels (0.796 on test) scores below the labels themselves on train_pool (0.836 in RQ1). The two are on different records, so this is only a rough sign that the model learns the LLM's boundary habits and does not average them away.

## Results: RQ3b, LLM labels with a small human review (computed 2026-11-20)
NAISS GPU, jobs 5546026 and 5546028 → `results/gpu/llm/review.jsonl` (140 runs); `scripts/rq3b.py` → `results/llm/rq3b.json`, `results/figures/rq3b_review.png`. One NER model on all 758 train_pool records; the k reviewed records get their human labels (simulated full correction, so an upper bound for a real review). Mean ± sd over seeds 1–10. "Errors fixed": the share of the LLM's span errors in train_pool that lie in the reviewed records.

| LLM | k | choice | test micro-F1 | share of reference (0.8882) | tokens reviewed | errors fixed |
|---|---|---|---|---|---|---|
| `gemini-4-carbon` | 0 | – | 0.7981 ± 0.0027 | 89.9% | 0% | 0% |
| | 50 | signal | 0.8082 ± 0.0029 | 91.0% | 7.0% | 14.3% |
| | 50 | random | 0.8047 ± 0.0051 | 90.6% | 6.6% | 7.4% |
| | 100 | signal | 0.8163 ± 0.0028 | 91.9% | 14.1% | 25.6% |
| | 100 | random | 0.8112 ± 0.0040 | 91.3% | 13.1% | 14.1% |
| | 200 | signal | 0.8308 ± 0.0061 | 93.5% | 28.2% | 44.4% |
| | 200 | random | 0.8213 ± 0.0045 | 92.5% | 26.2% | 27.3% |
| `qwen3.8-27b` | 0 | – | 0.7055 ± 0.0075 | 79.4% | 0% | 0% |
| | 50 | signal | 0.7391 ± 0.0070 | 83.2% | 6.9% | 13.6% |
| | 50 | random | 0.7241 ± 0.0074 | 81.5% | 6.6% | 6.8% |
| | 100 | signal | 0.7633 ± 0.0073 | 85.9% | 14.0% | 25.1% |
| | 100 | random | 0.7400 ± 0.0095 | 83.3% | 13.1% | 13.6% |
| | 200 | signal | 0.8025 ± 0.0100 | 90.4% | 27.8% | 44.7% |
| | 200 | random | 0.7778 ± 0.0095 | 87.6% | 26.2% | 27.0% |

Pre-registered comparisons (paired Wilcoxon over seeds 1–10; 18 tests, no correction):
- Signal vs random choice. `gemini-4-carbon`: k = 50 +0.0034 (7 of 10 seeds, p = 0.13), k = 100 +0.0051 (9 of 10, p = 0.004), k = 200 +0.0095 (10 of 10, p = 0.002). `qwen3.8-27b`: k = 50 +0.0150, k = 100 +0.0232, k = 200 +0.0247 (each 10 of 10, p = 0.002).
- Each k > 0 vs k = 0: every one of the 12 comparisons is better in 10 of 10 seeds (p = 0.002). Gains for k = 50 / 100 / 200 with the signal: `gemini-4-carbon` +0.010 / +0.018 / +0.033, `qwen3.8-27b` +0.034 / +0.058 / +0.097.
- Per-type F1 (descriptive, `rq3b.json`): most of the gain is in PhysicalObject (`gemini-4-carbon` 0.730 → 0.774, `qwen3.8-27b` 0.571 → 0.717 at k = 200 with the signal); Activity, State and Process change by 0.05 or less. Property also rises for `gemini-4-carbon` (0.742 → 0.833) but has too few spans to interpret (RQ1).

Reading:
- A review always helps, and choosing the records by the RQ2 signal helps more than choosing them at random: at each k the signal-chosen records hold about twice the errors of random ones (14 / 25 / 44% against 7 / 14 / 27%) for about the same amount of text (0.3–2.0 points more tokens; signal-chosen records are slightly longer). The advantage is larger for the weaker LLM, as RQ2 predicted (disagreement finds its errors more easily).
- The review does not close the gap. Even after 200 records (about 28% of the text) are corrected, the best LLM gives 93.5% of the human-label reference and the weakest 90.4%.
- k = 0 (0.7981 for `gemini-4-carbon`) and RQ3 random at 100% (0.7956) train on the same labels and records; the difference of 0.0025 comes from the order of the records in training and GPU non-determinism.

Not pre-registered, compared by eye with the RQ3 human-label curves: correcting 200 records (about 28% of the text) and keeping the LLM labels on the rest gives 0.831 (`gemini-4-carbon`) and 0.803 (`qwen3.8-27b`). Training on human labels only, on 30% of the text and without any LLM labels, gives 0.854 (random) and 0.869 (uncertainty). On MaintIE, the LLM labels on the unreviewed records therefore seem to lower the result compared with not using them. This was not tested and the records differ (signal-chosen against random or uncertainty-chosen); a direct test would train on the k reviewed records alone.

## Exploratory overview figures (added 2026-11-17, not part of a pre-registered analysis)
`scripts/plot_overview.py` draws three figures into `results/figures/`. They are descriptive and were made after the train_pool labelling; nothing below is a tested result, and the RQ1 analysis will be planned and run separately.

**1. Gold labels** (`1_gold_labels.png`). The types are very unequal: PhysicalObject 59% of all 3,397 labels, Activity 23%, State 13%, Process 4%, Property 1% (35 labels). The micro F1 is therefore mostly a PhysicalObject score, and Property scores rest on very few labels (5 in test). The type mix is about the same in train_pool, val and test (test has slightly more Activity, 26% against 22%). Most labels are one word; PhysicalObject has the most multi-word labels (28% two words), so its boundaries are the hardest to get exactly right.

**2. Test F1 per type in the active-learning runs** (`2_per_type_learning_curves.png`, Mac main results, mean of 10 seeds). Activity is learned fastest (about 0.90 with 10% of train_pool) and the three strategies are almost the same there. PhysicalObject is the hardest type (about 0.86 with all of train_pool). The gap between the active strategies and random is visible mainly for PhysicalObject, State and Process. The Property curve jumps because test holds only 5 Property labels. Reading: the advantage of active learning seen in the overall curves comes from the harder types, not the easy one.

**3. The 7 API LLMs on train_pool, by type** (`3_llm_by_type.png`). Label counts relative to gold: almost every model marks more Property spans than the human annotators (1.5–1.8 times; DeepSeek 1.04), most mark somewhat more PhysicalObject spans (up to 1.21 times, mostly extra nested parts of object names, as seen in prompt tuning), and DeepSeek marks fewer (0.76 times). F1 by type: Activity is closest to the human labels (0.91–0.97), PhysicalObject and Property are furthest (0.65–0.79, DeepSeek Property 0.45). Highest micro F1: `gemini-4-carbon` 0.84 and `gpt-6-astra` 0.83.

**4. Kinds of disagreement with gold** (`4_llm_disagreement_kinds.png`; counted over LLM spans that do not match exactly plus gold spans with nothing marked there). Over the 7 models, 76% of the disagreements keep the type but differ in boundary or nesting, 9% are missed spans, 6% differ in both boundary and type, 6% are extra spans and 3% are the same words with another type. For the strongest models (`gemini-4-carbon`, `gpt-6-astra`, Claude Opus 5.5) boundary differences are 82–83%; Claude Haiku 5.5 and DeepSeek miss more (15–16%), and DeepSeek has the most type errors (9%). By type, PhysicalObject disagreements are 87% boundary; Activity disagreements are mostly missed spans (36%); State is spread out with 18% type errors; Process (42%) and Property (71%) are mostly spans that differ in both boundary and type. So a type definition or ontology in the prompt could address mainly State, Process and Property (together about 11% of all disagreements), while boundary rules or a term list would address PhysicalObject.

**Update (2026-11-17): Qwen3.8-27B added.** Figures 3 and 4 now include the self-hosted `qwen3.8-27b`; the numbers in paragraphs 3 and 4 above are for the 7 API models. `qwen3.8-27b` has the lowest train_pool micro-F1 so far (0.71; PhysicalObject 0.58, Activity 0.90, State 0.87, Process 0.89, Property 0.72). It marks fewer PhysicalObject spans than the human annotators (0.77 times) and 15% of its disagreements are missed spans. Its tuning score on 20 records (0.807) was higher, which shows how little 20 records say.

**Together.** For both the trained NER model and the LLMs, the hardest part is where an object name starts and ends. The gold labels are themselves not fully consistent there (amendment 6), so part of this difficulty belongs to the task and the annotation, not only to the models. Possible uses, to be decided in the analysis plans: in RQ1, separate boundary and nesting differences from wrong types; in RQ2, check whether uncertainty and disagreement point mostly to PhysicalObject spans.

### Amendment 8 (2026-11-17, after the train_pool labels were seen, before any training run): RQ3b, LLM labels with a small human review
Added after the label quality of the API models was seen (exploratory figures above) and before any NER model was trained on LLM labels. RQ1–RQ3 are not changed.

- Question: if an LLM labels all of train_pool and a reviewer corrects only k records, how close does the NER model come to training on human labels only, and does choosing the records by the RQ2 signal beat choosing them at random?
- LLMs: the same two as RQ3 (highest and lowest train_pool micro-F1 in RQ1).
- Review budget k: 0, 50, 100 and 200 records. k = 0 is the LLM labels unchanged.
- Choosing the k records, per model and seed: (a) the top k of the RQ2 pool by the RQ2 combined signal (mean rank of model uncertainty and LLM disagreement); (b) k records drawn at random from the same pool. Only the combined signal is used; it is not replaced by whichever RQ2 signal turns out best.
- Review is simulated: the chosen records get their human labels. This assumes a reviewer who corrects everything, so the results are an upper bound for a real review.
- Training: one NER model on all 758 train_pool records (the k reviewed records with human labels, the rest with the LLM's labels), same model and training settings as before, seeds 1–10, on the NAISS GPU (amendment 2). Evaluation on the human test set.
- Reference: the GPU human-label result with all of train_pool, test micro-F1 0.8882 (mean of 10 seeds, `results/gpu/random.jsonl`).
- Comparisons, paired over seeds 1–10 with the Wilcoxon signed-rank test: signal vs random choice at each k > 0; each k > 0 vs k = 0. Reported for each k: test micro-F1, share of the reference, and the number of records and tokens reviewed. Per-type F1 is reported descriptively.
- Size: 2 LLMs × (1 + 2 choices × 3 budgets) × 10 seeds = 140 training runs.

## Notes for the analysis (added 2026-11-17, to be taken up once all runs are finished)
Questions raised while the runs were going. None of them is a result; each is to be decided in the analysis plan.

1. **LLM against a trained NER model.** A rough comparison: the best LLM on train_pool (`gemini-4-carbon`, micro-F1 0.836) is about where the NER model is with roughly 150 randomly chosen or 100–110 actively chosen human-labelled records (Mac main results, test set: random 0.835 at 152 records; uncertainty 0.812 at 76 and 0.843 at 114). By type the gap is almost all PhysicalObject (NER 0.86 with all of train_pool, best LLM 0.77) and Property; for Activity, Process and State the best LLMs are level with or above the NER model. Limits of this comparison: the two are scored on different records (NER on test, LLMs on train_pool; test is never sent to an LLM); the LLMs had a written type guide, 5 examples and a prompt tuned on 20 val records, not only 5 examples; MaintIE is public, so the LLMs may have seen it. A fair comparison needs both on the same records, for example NER predictions on train_pool by cross-validation.
2. **Reading of the comparison.** The LLMs seem to know what the words mean but not the annotators' conventions for where an object name starts and ends; the NER model learns those conventions from the labels. RQ3 and RQ3b test whether the two can be combined.
3. **Guidance in the prompt.** Since 76% of the disagreements are boundary or nesting differences with the right type and 3% are type errors (figure 4), an ontology or type definitions would address mainly State, Process and Property, while boundary rules (for example the MaintIE annotation guidelines) or a term list would address PhysicalObject. This would be a separate experiment with its own prompt, tuned on val only; any guidance must come from outside train_pool, so that the LLMs do not see the human labels indirectly. Not planned yet.

### Amendment 10 (2026-11-20, after the RQ3 and RQ3b results were seen, before any run of this experiment): RQ3c, the reviewed records alone
Exploratory. The question came from the RQ3b results (last paragraph of the RQ3b results): compared by eye with the human-label curves, the unreviewed LLM labels seemed to help when few records are reviewed and to hurt when many are, mainly for PhysicalObject. Because the hypothesis was formed from the data, the p-values below are not confirmatory. RQ1–RQ3b are not changed.

- Question: at the same review budget, is it better to train on the k reviewed records together with the LLM labels on the rest of train_pool (RQ3b), or on the k reviewed records alone?
- Runs: for the same k reviewed records as RQ3b (same models, k = 50, 100, 200, same signal and random choice, same seeds 1–10), train one NER model on the k records with their human labels only. Same model and training settings, NAISS GPU, evaluation on the human test set. The random choice does not depend on the LLM (`choose_review` uses only the seed and the pool outside the seed's initial set, which is the same for both models), so it is run once. Size: 2 LLMs × 3 budgets × 10 seeds (signal) + 3 budgets × 10 seeds (random) = 90 runs, written to `results/gpu/llm/review_only.jsonl`.
- Comparisons, paired over seeds 1–10 with the Wilcoxon signed-rank test: for each LLM, k and choice, RQ3b (k reviewed + LLM labels) minus reviewed records alone; 12 tests, no correction. Reported for each: test micro-F1 and per-type F1 (PhysicalObject and State in particular, descriptive).
- All results are reported whatever their direction.
