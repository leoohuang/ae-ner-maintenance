import numpy as np


def span_entropy(probs):
    """probs: (词段数, 类别数)，每行是概率。返回每个词段的熵，越大越没把握。"""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def record_score(probs, n_words):
    """一条记录的不确定性：把所有词段的熵从大到小排，取前 n_words 个求平均。
    只取前 n_words 个，是为了不让长句子仅因为词段多而占便宜。"""
    h = np.sort(span_entropy(probs))[::-1]
    return float(h[:n_words].mean())


def rank_by_uncertainty(pred, ids, records):
    """pred 与 ids 一一对应，每个元素是 predict() 给出的 (词段, 概率)。
    返回按不确定性从高到低排好的记录下标；分数相同时下标小的在前。"""
    scored = [(-record_score(pr, len(records[i]["tokens"])), i)
              for i, (sp, pr) in zip(ids, pred)]
    scored.sort()
    return [i for _, i in scored]