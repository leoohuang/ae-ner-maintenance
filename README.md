
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