import json, collections, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
records = [json.loads(l) for l in open(ROOT / "data" / "processed" / "gold.jsonl", encoding="utf-8")]
PLACEHOLDERS = {"-", "<id>", "<num>", "<date>", "<sensitive>"}
sets = [set(w.lower() for w in r["tokens"]) - PLACEHOLDERS for r in records]

def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0

# 所有相似度 >= 0.4 的记录对
pairs = []
for i in range(len(sets)):
    for j in range(i + 1, len(sets)):
        s = jac(sets[i], sets[j])
        if s >= 0.4:
            pairs.append((s, i, j))
print("相似度 >= 0.4 的记录对:", len(pairs))

# 1) 每个相似度区间抽样看看
rng = random.Random(0)
for lo, hi in [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
    sel = [p for p in pairs if lo <= p[0] < hi]
    print(f"\n== 相似度 [{lo}, {min(hi, 1.0)}]：{len(sel)} 对")
    for s, i, j in rng.sample(sel, min(6, len(sel))):
        print(f"  {s:.2f} | {' '.join(records[i]['tokens'])}  ||  {' '.join(records[j]['tokens'])}")

# 2) 不同阈值下，连成的组是什么样
def groups_at(thr):
    parent = list(range(len(records)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for s, i, j in pairs:
        if s >= thr:
            parent[find(i)] = find(j)
    g = collections.defaultdict(list)
    for i in range(len(records)):
        g[find(i)].append(i)
    return list(g.values())

print("\n阈值 | 分组数 | 多于1条的组数 | 最大组大小 | 最大组占比")
for thr in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    g = groups_at(thr)
    big = max(len(x) for x in g)
    print(f"{thr:.1f}  | {len(g):5d} | {sum(len(x) > 1 for x in g):6d} | {big:4d} | {big / len(records):.1%}")

for thr in (0.5, 0.6):
    g = max(groups_at(thr), key=len)
    print(f"\n阈值 {thr} 下最大的一组（共 {len(g)} 条），前 8 条：")
    for i in g[:8]:
        print("  ", " ".join(records[i]["tokens"]))
