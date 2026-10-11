
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
