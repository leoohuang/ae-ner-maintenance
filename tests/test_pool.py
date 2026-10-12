from src.pool import budgets, random_order, take_budget, pool_tokens

# 假记录：下标 i 的记录有 (i % 5) + 1 个词，长度 1~5
RECORDS = [{"tokens": ["w"] * (i % 5 + 1)} for i in range(40)]
POOL = list(range(40))


def test_same_seed_same_order():
    assert random_order(POOL, 1) == random_order(POOL, 1)


def test_different_seed_different_order():
    assert random_order(POOL, 1) != random_order(POOL, 2)


def test_order_is_a_permutation():
    assert sorted(random_order(POOL, 3)) == POOL


def test_budget_reached_and_not_overshot_by_more_than_last_record():
    order = random_order(POOL, 1)
    chosen, used = take_budget(order, RECORDS, 30)
    assert used >= 30
    last = len(RECORDS[chosen[-1]]["tokens"])
    assert used - last < 30          # 去掉最后一条就不够，说明恰好在达到预算时停


def test_no_duplicates():
    chosen, _ = take_budget(random_order(POOL, 1), RECORDS, 50)
    assert len(chosen) == len(set(chosen))


def test_nested_prefixes():
    order = random_order(POOL, 1)
    small, _ = take_budget(order, RECORDS, 20)
    big, _ = take_budget(order, RECORDS, 60)
    assert big[:len(small)] == small


def test_zero_budget_is_empty():
    assert take_budget(random_order(POOL, 1), RECORDS, 0) == ([], 0)


def test_budget_above_total_returns_everything():
    chosen, used = take_budget(random_order(POOL, 1), RECORDS, 10**6)
    assert sorted(chosen) == POOL and used == pool_tokens(RECORDS, POOL)


def test_budgets_match_frozen_split_file():
    # split_v1.json 里已冻结的预算点（训练池 4281 个词）
    b = budgets(4281)
    assert b[0.05] == 214 and b[0.10] == 428 and b[0.20] == 856
    assert b[0.40] == 1712 and b[0.60] == 2569 and b[1.00] == 4281
