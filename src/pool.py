import random

# 预注册（实验前冻结）
GRID = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]   # 占训练池总 token 的比例
SEEDS = list(range(1, 11))                            # pilot 用的 101-103 不复用


def token_lengths(records):
    return [len(r["tokens"]) for r in records]


def pool_tokens(records, pool_ids):
    return sum(len(records[i]["tokens"]) for i in pool_ids)


def budgets(total_tokens, grid=GRID):
    return {f: round(f * total_tokens) for f in grid}


def random_order(ids, seed):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    return ids


def fill_budget(candidates, tok_len, current_tokens, target_tokens):
    """按候选顺序加入记录，放不下的跳过；总 token 数不会超过 target。
    返回 (新增的记录下标, 加入后的总 token 数)。日志里要记录实际 total。"""
    chosen, total = [], current_tokens
    for i in candidates:
        if total + tok_len[i] <= target_tokens:
            chosen.append(i)
            total += tok_len[i]
        if total == target_tokens:
            break
    return chosen, total


def random_nested_sets(order, tok_len, targets):
    """随机策略：预算从小到大依次往已选集合里补记录，所以小预算的集合
    一定被大预算的集合包含。返回 {target: (记录下标列表, 实际 token 数)}。"""
    chosen, total, out = [], 0, {}
    for t in sorted(targets):
        taken = set(chosen)
        rest = [i for i in order if i not in taken]
        new, total = fill_budget(rest, tok_len, total, t)
        chosen = chosen + new
        out[t] = (list(chosen), total)
    return out
