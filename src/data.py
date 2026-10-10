def to_spans(entities):
    """顶层类型 + 去重；保留嵌套。end 不含（开区间）。"""
    out = {}
    for e in entities:
        key = (e["start"], e["end"])
        typ = e["type"].split("/")[0]
        assert out.get(key, typ) == typ, "同一范围出现两个不同的顶层类型"
        out[key] = typ
    return sorted((s, e, t) for (s, e), t in out.items())


PLACEHOLDERS = {"-", "<id>", "<num>", "<date>", "<sensitive>"}


def word_set(tokens):
    return set(w.lower() for w in tokens) - PLACEHOLDERS


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def near_dup_groups(records, thr):
    """把相似度 >= thr 的记录用并查集连成组，返回组的列表（每组是记录下标列表）"""
    sets = [word_set(r["tokens"]) for r in records]
    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if jaccard(sets[i], sets[j]) >= thr:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
