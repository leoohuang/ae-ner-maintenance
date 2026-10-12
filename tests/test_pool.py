from src.pool import (budgets, random_order, fill_budget, random_nested_sets,
                      token_lengths, pool_tokens)

# 下标 i 的记录有 (i % 5) + 1 个词；40 条共 120 个词
RECORDS = [{"tokens": ["w"] * (i % 5 + 1)} for i in range(40)]
LEN = token_lengths(RECORDS)
POOL = list(range(40))


def test_same_seed_same_order():
    assert random_order(POOL, 1) == random_order(POOL, 1)


def test_different_seed_different_order():
    assert random_order(POOL, 1) != random_order(POOL, 2)


def test_order_is_a_permutation():
    assert sorted(random_order(POOL, 3)) == POOL


def test_never_exceeds_target():
    for target in (7, 30, 55, 119):
        _, total = fill_budget(random_order(POOL, 1), LEN, 0, target)
        assert total <= target


def test_skips_record_that_does_not_fit():
    # 长度 5,5,2，目标 7：第 2 条放不下，跳过，第 3 条放得下
    chosen, total = fill_budget([0, 1, 2], [5, 5, 2], 0, 7)
    assert chosen == [0, 2] and total == 7


def test_respects_current_tokens():
    chosen, total = fill_budget([1, 2], [5, 5, 2], 5, 7)
    assert chosen == [2] and total == 7


def test_zero_budget_is_empty():
    assert fill_budget(random_order(POOL, 1), LEN, 0, 0) == ([], 0)


def test_budget_equal_to_total_takes_everything():
    chosen, total = fill_budget(random_order(POOL, 1), LEN, 0, 120)
    assert sorted(chosen) == POOL and total == 120


def test_nested_and_no_duplicates():
    sets = random_nested_sets(random_order(POOL, 1), LEN, [20, 40, 80, 120])
    prev = []
    for t in (20, 40, 80, 120):
        ids, total = sets[t]
        assert len(ids) == len(set(ids)) and total <= t
        assert ids[:len(prev)] == prev          # 小预算是大预算的前缀
        prev = ids
    assert sorted(prev) == POOL                 # 100% 包含全部记录


def test_budgets_match_frozen_split_file():
    b = budgets(4281)
    assert b[0.05] == 214 and b[0.10] == 428 and b[0.20] == 856
    assert b[0.40] == 1712 and b[0.60] == 2569 and b[1.00] == 4281
