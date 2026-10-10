import json, random, hashlib, collections
from pathlib import Path
from src.data import near_dup_groups, word_set, jaccard

ROOT = Path(__file__).resolve().parent.parent
records = [json.loads(l) for l in open(ROOT / "data" / "processed" / "gold.jsonl", encoding="utf-8")]

THR = 0.6                          # 已冻结
RATIOS = (0.70, 0.10, 0.20)        # 跑任何实验之前定好，之后不改
START_SEED = 13
MIN_PROPERTY = {"train_pool": 15, "val": 2, "test": 5}   # 只看数据分布，不看任何模型结果
BUDGET_GRID = [0.05, 0.10, 0.20, 0.40, 0.60]

groups = near_dup_groups(records, THR)
print("分组数:", len(groups), "| 最大组:", max(len(g) for g in groups))


def split_groups(groups, n, seed):
    gs = [list(g) for g in groups]
    random.Random(seed).shuffle(gs)
    cut1, cut2 = RATIOS[0] * n, (RATIOS[0] + RATIOS[1]) * n
    split = {"train_pool": [], "val": [], "test": []}
    seen = 0
    for g in gs:
        name = "train_pool" if seen < cut1 else "val" if seen < cut2 else "test"
        split[name] += g
        seen += len(g)
    return {k: sorted(v) for k, v in split.items()}


def stats_of(split):
    out = {}
    for name, ids in split.items():
        cnt = collections.Counter(t for i in ids for _, _, t in records[i]["spans"])
        out[name] = {"n_records": len(ids),
                     "n_tokens": sum(len(records[i]["tokens"]) for i in ids),
                     "spans": dict(cnt)}
    return out


# 换种子直到稀有类型（Property）在三个集合里都够数
seed, tries = START_SEED, 1
while True:
    split = split_groups(groups, len(records), seed)
    st = stats_of(split)
    if all(st[n]["spans"].get("Property", 0) >= MIN_PROPERTY[n] for n in split):
        break
    seed += 1
    tries += 1
    assert tries <= 1000, "找不到满足条件的种子，请放宽 MIN_PROPERTY"

# 自检：三个集合互不相交，覆盖全部记录
assert not (set(split["train_pool"]) & set(split["val"]))
assert not (set(split["train_pool"]) & set(split["test"]))
assert not (set(split["val"]) & set(split["test"]))
assert sum(len(v) for v in split.values()) == len(records)

# 自检：不应该有相似度 >= THR 的记录对跨集合
where = {i: name for name, ids in split.items() for i in ids}
sets = [word_set(r["tokens"]) for r in records]
leak = sum(1 for i in range(len(records)) for j in range(i + 1, len(records))
           if where[i] != where[j] and jaccard(sets[i], sets[j]) >= THR)
print("跨集合的近似重复对（应为 0）:", leak)
assert leak == 0

h = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()[:16]
pool_tokens = st["train_pool"]["n_tokens"]
out = {"seed": seed, "tries": tries, "threshold": THR, "ratios": RATIOS, "hash": h,
       **split, "stats": st,
       "budget_tokens": {str(f): round(f * pool_tokens) for f in BUDGET_GRID}}
(ROOT / "data" / "splits").mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT / "data" / "splits" / "split_v1.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("种子:", seed, "| 尝试次数:", tries, "| hash:", h)
for name in ("train_pool", "val", "test"):
    s = st[name]
    print(f"{name:11s} 记录 {s['n_records']:4d} | token {s['n_tokens']:5d} | {s['spans']}")
print("预算点对应的 token 数:", out["budget_tokens"])
