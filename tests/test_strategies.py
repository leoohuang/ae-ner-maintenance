import numpy as np
from src.strategies import span_entropy, record_score, rank_by_uncertainty


def test_uniform_entropy_is_log_k():
    h = span_entropy(np.full((1, 6), 1 / 6))
    assert abs(h[0] - np.log(6)) < 1e-9


def test_confident_prediction_has_near_zero_entropy():
    p = np.array([[1.0, 0, 0, 0, 0, 0]])
    assert span_entropy(p)[0] < 1e-6


def test_record_score_uses_top_n_words():
    probs = np.array([[.5, .5], [.5, .5], [1.0, 0.0]])
    assert abs(record_score(probs, 2) - np.log(2)) < 1e-9
    assert abs(record_score(probs, 3) - 2 * np.log(2) / 3) < 1e-6


def test_rank_puts_uncertain_first_and_breaks_ties_by_id():
    records = {5: {"tokens": ["a"]}, 7: {"tokens": ["a"]}, 9: {"tokens": ["a"]}}
    sure = (None, np.array([[1.0, 0.0]]))
    unsure = (None, np.array([[.5, .5]]))
    pred = [unsure, sure, unsure]          # 依次对应记录 9, 7, 5
    assert rank_by_uncertainty(pred, [9, 7, 5], records) == [5, 9, 7]