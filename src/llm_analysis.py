"""RQ1 和 RQ2 的分析函数（LLM annotation protocol；定义见 README amendment 9）。
标签都是 [start, end, type] 列表，与 gold.jsonl 同格式。"""
from collections import Counter

import numpy as np

from src.evaluate import _prf


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def error_types(gold, pred):
    """一个模型在一条记录上的错误按类型计数（amendment 9，定义 1）。"""
    G, Q = {tuple(s) for s in gold}, {tuple(s) for s in pred}
    c = Counter()
    for p in Q - G:
        if any(g[:2] == p[:2] for g in G):
            c["type"] += 1
        elif any(_overlap(g, p) and g[2] == p[2] for g in G):
            c["boundary"] += 1
        elif any(_overlap(g, p) for g in G):
            c["boundary_type"] += 1
        else:
            c["extra"] += 1
    for g in G - Q:
        if not any(_overlap(g, p) for p in Q):
            c["missed"] += 1
    return dict(c)


def _counts(gold, pred):
    G, Q = {tuple(s) for s in gold}, {tuple(s) for s in pred}
    return len(G & Q), len(Q - G), len(G - Q)


def bootstrap_micro_f1(gold, pred, n=1000, seed=0):
    """对记录重抽样的 micro-F1 95% 区间。gold / pred：与记录一一对应的标签列表。"""
    c = np.array([_counts(g, p) for g, p in zip(gold, pred)], float)
    rng = np.random.default_rng(seed)
    f = []
    for _ in range(n):
        tp, fp, fn = c[rng.integers(0, len(c), len(c))].sum(0)
        f.append(_prf(tp, fp, fn)[2])
    return float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))


def length_tertiles(lengths):
    """{id: 词数} -> {id: short / medium / long}，在 1/3 和 2/3 分位数处切，等于切点的归低一组。"""
    q1, q2 = np.quantile(list(lengths.values()), [1 / 3, 2 / 3])
    return {i: "short" if n <= q1 else "medium" if n <= q2 else "long" for i, n in lengths.items()}


def shared_errors(gold, by_model, min_agree=6):
    """gold：{id: spans}；by_model：{model: {id: spans}}（amendment 9，定义 2）。"""
    agreed, wrong, missed_all, n_gold = 0, 0, 0, 0
    for i, g in gold.items():
        G = {tuple(s) for s in g}
        votes = Counter(s for labels in by_model.values() for s in {tuple(x) for x in labels[i]})
        top = {s for s, v in votes.items() if v >= min_agree}
        agreed += len(top)
        wrong += len(top - G)
        n_gold += len(G)
        missed_all += sum(1 for s in G if votes[s] == 0)
    return {"agreed_spans": agreed, "wrong_agreed_spans": wrong,
            "wrong_share_of_agreed": wrong / agreed if agreed else 0.0,
            "gold_spans": n_gold, "missed_by_all": missed_all,
            "missed_by_all_share": missed_all / n_gold if n_gold else 0.0}


def span_errors(gold, pred):
    """一条记录的 span 错误数：多标的加漏标的（amendment 9，定义 3）。"""
    _, fp, fn = _counts(gold, pred)
    return fp + fn


def record_f1(a, b):
    """两组标签在一条记录上的 span F1；两边都为空记为 1。"""
    tp, fp, fn = _counts(a, b)
    if tp + fp + fn == 0:
        return 1.0
    return _prf(tp, fp, fn)[2]


def auroc(scores, labels):
    """分数越高越可能是 1 时 AUROC 越高；并列算一半（Mann-Whitney）。"""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def topk_error_share(scores, errors, k):
    """按分数从高到低检查前 k 条记录，能看到这个模型全部 span 错误的多少比例。
    scores、errors：{id: 值}；分数并列时按 id 排，保证可复现。"""
    order = sorted(scores, key=lambda i: (-scores[i], i))[:k]
    total = sum(errors.values())
    return sum(errors[i] for i in order) / total if total else 0.0


# ---------- RQ2 信号 ----------
def disagreement(labels, model, rid):
    """LLM 分歧：1 减去这个模型与其他每个模型在这条记录上的 span F1 的平均。labels：{model: {id: spans}}。"""
    others = [m for m in labels if m != model]
    return 1.0 - sum(record_f1(labels[model][rid], labels[m][rid]) for m in others) / len(others)


def mean_ranks(scores):
    """{id: 分数} -> {id: 名次}，分数最高的是第 1 名；并列取平均名次。"""
    order = sorted(scores, key=lambda i: -scores[i])
    ranks, k = {}, 0
    while k < len(order):
        j = k
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[k]]:
            j += 1
        for i in order[k:j + 1]:
            ranks[i] = (k + j) / 2 + 1
        k = j + 1
    return ranks


def combined_rank(uncertainty, disagree):
    """综合信号：两个信号名次的平均，越小越可疑。"""
    ru, rd = mean_ranks(uncertainty), mean_ranks(disagree)
    return {i: (ru[i] + rd[i]) / 2 for i in ru}


# ---------- RQ3 / RQ3b 的训练标签 ----------
def apply_labels(records, llm, split, reviewed=()):
    """返回 records 的副本：train_pool 的记录换成 LLM 的标签，reviewed 里的记录保留人工标签，
    val 和 test 不动。LLM 标签必须正好覆盖 train_pool。"""
    pool = set(split["train_pool"])
    if set(llm) != pool:
        raise ValueError("LLM labels must cover exactly the train_pool records")
    keep = set(reviewed)
    out = [dict(r) for r in records]
    for i in pool - keep:
        out[i]["spans"] = [list(s) for s in llm[i]]
    return out


def choose_review(pool_ids, combined, k, choice, seed):
    """RQ3b：要人工检查的 k 条记录。signal：综合名次最小的 k 条（并列按 id）；
    random：pool 按 1000 + seed 打乱后的前 k 条，所以不同 k 之间是嵌套的。"""
    if choice == "signal":
        return sorted(pool_ids, key=lambda i: (combined[i], i))[:k]
    if choice == "random":
        import random
        order = list(pool_ids)
        random.Random(1000 + seed).shuffle(order)
        return order[:k]
    raise ValueError(choice)


def review_training_set(records, llm, split, reviewed, with_llm=True):
    """RQ3b / RQ3c 的训练记录。with_llm：整个 train_pool，reviewed 用人工标签、其余用 LLM 标签（RQ3b）；
    否则只有 reviewed 这些记录、人工标签（RQ3c，amendment 10），按 reviewed 的顺序。"""
    if with_llm:
        train = apply_labels(records, llm, split, reviewed=reviewed)
        return [train[i] for i in split["train_pool"]]
    if not reviewed:
        raise ValueError("no reviewed records to train on")
    return [dict(records[i]) for i in reviewed]


def prompt_diff(gold, runs_a, runs_b, n=1000, seed=0, etype=None):
    """Amendment 11：两个提示词各跑若干遍，比较 exact-match F1（每个提示词取各遍的平均）。
    runs_a / runs_b：每遍是与 gold 一一对应的标签列表。etype 给定时只看这一类。
    返回 a、b 的平均 F1、差 b − a，以及对记录配对重抽样的 95% 区间。"""
    keep = (lambda s: [x for x in s if x[2] == etype]) if etype else (lambda s: s)
    g = [keep(x) for x in gold]
    ca, cb = ([np.array([_counts(gi, keep(p)) for gi, p in zip(g, run)], float) for run in runs]
              for runs in (runs_a, runs_b))
    ca, cb = list(ca), list(cb)

    def f1(cs, idx):
        return float(np.mean([_prf(*c[idx].sum(0))[2] for c in cs]))

    full = np.arange(len(g))
    a, b = f1(ca, full), f1(cb, full)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(g), len(g))
        d.append(f1(cb, idx) - f1(ca, idx))
    return {"a": a, "b": b, "diff": b - a,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
