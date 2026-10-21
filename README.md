
## log 10.11
- 近似重复分组：词集合 Jaccard 相似度 >= 0.6（忽略占位符与 "-"），并查集连成组，整组进同一集合。
  依据：0.5 会连成 164 条的无关大组，0.6 最大组 22 条且内容一致。残留局限：0.5–0.6 之间的近似重复仍可能跨集合。

## Pilot 规则（10.12 实验前冻结）
- 训练集 = split_v1 的 train_pool（758 条）；比较只用 val（103 条）；test 不加载。
- 候选：bert-base-uncased / roberta-base / answerdotai/ModernBERT-base
- pilot 种子：101, 102, 103（与正式实验种子不同）
- encoder_lr 只在 {2e-5, 5e-5} 里选一次：先用 BERT 跑两个学习率，选验证集 micro-F1 均值高者，然后三个编码器都用它。
- 固定不调：head_lr 1e-3、none_weight 0.2、head_hidden 256、width_dim 16、dropout 0.1、batch_size 16、max_len 64、min_steps 300、min_epochs 5
- 决策：验证集 micro-F1 均值最高者胜；若第一名与第二名的差小于 1 个标准差，选更稳定、更便宜的。
- 运行失败的编码器记为"失败"，在报告里如实写原因。

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