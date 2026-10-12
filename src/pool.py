import random

# 预注册（实验前冻结）
GRID = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]   # 预算点：占训练池总 token 的比例
SEEDS = list(range(1, 11))                            # 10 个种子，pilot 用的是 101-103


def pool_tokens(records, pool_ids):
    return sum(len(records[i]["tokens"]) for i in pool_ids)


def budgets(total_tokens, grid=GRID):
    """每个预算点对应的 token 数。100% 即整个训练池。"""
    return {f: round(f * total_tokens) for f in grid}


def random_order(pool_ids, seed):
    """同一个种子永远得到同一个顺序；不同预算点取这个顺序的前缀。"""
    order = list(pool_ids)
    random.Random(seed).shuffle(order)
    return order


def take_budget(order, records, budget):
    """按顺序加记录，累计 token 一旦 >= budget 就停。
    返回 (选中的记录下标, 实际用掉的 token 数)。"""
    chosen, used = [], 0
    for i in order:
        if used >= budget:
            break
        chosen.append(i)
        used += len(records[i]["tokens"])
    return chosen, used
