import numpy as np
from src.hybrid import hybrid_select


def unit(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


def make_emb(vectors):
    emb = np.zeros((10, 2))
    for k, v in vectors.items():
        emb[k] = unit(v)
    return emb


def test_prefers_records_far_from_labelled_and_each_other():
    emb = make_emb({3: [1, 0], 0: [1, 0], 1: [0, 1], 2: [0, 1]})
    tok_len = {i: 1 for i in range(10)}
    chosen, total = hybrid_select([0, 1, 2], emb, [3], tok_len, 1, 3)
    assert chosen == [1, 0]
    assert total == 3


def test_only_looks_inside_the_uncertain_candidates():
    emb = make_emb({9: [1, 0], 0: [1, 1], 1: [1, 1], 2: [1, 1], 5: [0, 1]})
    tok_len = {i: 1 for i in range(10)}
    chosen, total = hybrid_select([0, 1, 2, 5], emb, [9], tok_len, 1, 2)
    assert chosen == [0]


def test_never_exceeds_budget_and_fills_from_the_rest():
    emb = make_emb({9: [1, 0], 0: [0, 1], 1: [1, 1], 2: [1, 2], 3: [2, 1]})
    tok_len = {0: 5, 1: 1, 2: 1, 3: 1, 9: 1}
    chosen, total = hybrid_select([0, 1, 2, 3], emb, [9], tok_len, 1, 3)
    assert 0 not in chosen
    assert total == 3
    assert sorted(chosen) == [1, 2]