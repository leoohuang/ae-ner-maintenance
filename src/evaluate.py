from collections import Counter
 
TYPES = ["PhysicalObject", "State", "Process", "Activity", "Property"]
 
def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
 
def evaluate(gold, pred):
    """gold / pred：与记录一一对应的列表，每项是 (start, end, type) 的集合。
    exact match：范围和类型都对才算对。用集合比较，重复预测不会抬高分数。"""
    assert len(gold) == len(pred), "记录数不一致"
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        g, p = set(map(tuple, g)), set(map(tuple, p))
        for sp in p & g: tp[sp[2]] += 1
        for sp in p - g: fp[sp[2]] += 1
        for sp in g - p: fn[sp[2]] += 1
    per = {}
    for t in TYPES:
        p, r, f = _prf(tp[t], fp[t], fn[t])
        per[t] = {"precision": p, "recall": r, "f1": f, "support": tp[t] + fn[t]}
    P, R, F = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return {"micro_f1": F, "precision": P, "recall": R,
            "macro_f1": sum(per[t]["f1"] for t in TYPES) / len(TYPES),
            "per_entity": per}
