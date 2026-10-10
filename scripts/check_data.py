import json, collections
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "gold_release.json"
data = json.load(open(RAW, encoding="utf-8"))

n_tok = [len(d["tokens"]) for d in data]
print("texts:", len(data), "| tokens total/avg/max:",
      sum(n_tok), round(sum(n_tok) / len(n_tok), 2), max(n_tok))

bad = nested = crossing = same_range = conflict = 0
top, leaf = collections.Counter(), collections.Counter()
for d in data:
    n = len(d["tokens"])
    ents = []
    for e in d["entities"]:
        s, t, typ = e["start"], e["end"], e["type"].split("/")[0]
        if not (0 <= s < t <= n):
            bad += 1
            continue
        ents.append((s, t, typ))
        top[typ] += 1
        leaf[e["type"]] += 1
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            (s1, t1, y1), (s2, t2, y2) = ents[i], ents[j]
            if max(s1, s2) >= min(t1, t2):
                continue
            if (s1, t1) == (s2, t2):
                same_range += 1
                conflict += (y1 != y2)
            elif (s1 <= s2 and t2 <= t1) or (s2 <= s1 and t1 <= t2):
                nested += 1
            else:
                crossing += 1

print("越界 span:", bad)
print("重叠实体对 -> 嵌套:", nested, "| 交叉:", crossing,
      "| 同范围:", same_range, "| 同范围不同类型:", conflict)
print("顶层实体:", top.most_common(), "| 合计:", sum(top.values()))
print("叶子类型数:", len(leaf))

uniq = collections.Counter(" ".join(d["tokens"]).lower() for d in data)
print("唯一文本:", len(uniq), "| 完全重复率: %.1f%%" % (100 * (1 - len(uniq) / len(data))))
