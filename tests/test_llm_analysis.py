import pytest

from src.llm_analysis import (prompt_diff, apply_labels, review_training_set, auroc, bootstrap_micro_f1, choose_review, combined_rank,
                              disagreement, error_types, length_tertiles, mean_ranks, record_f1,
                              shared_errors, span_errors, topk_error_share)

P, A, S = "PhysicalObject", "Activity", "State"


# --- RQ1 error types (amendment 9, definition 1) ---
def test_error_types_cover_the_five_kinds():
    gold = [[0, 2, P], [3, 4, S], [5, 6, A], [8, 9, P]]
    pred = [[1, 2, P],            # overlaps a gold P span: boundary
            [3, 4, A],            # same start and end, other type: type
            [5, 7, S],            # overlaps only a gold A span: boundary and type
            [10, 11, P]]          # no overlap: extra
    # gold [8, 9] is overlapped by nothing: missed
    assert error_types(gold, pred) == {"boundary": 1, "type": 1, "boundary_type": 1, "extra": 1, "missed": 1}


def test_exact_matches_are_not_errors_and_overlapped_gold_is_not_missed():
    gold = [[0, 2, P], [1, 2, P]]
    pred = [[0, 2, P], [0, 1, P]]          # [0, 1] overlaps both gold spans of the same type
    assert error_types(gold, pred) == {"boundary": 1}


# --- RQ1 bootstrap ---
def test_bootstrap_interval_contains_the_point_estimate_and_is_reproducible():
    gold = [[[0, 1, P]], [[0, 1, A]], [[0, 1, S]], [[0, 2, P]]] * 5
    pred = [[[0, 1, P]], [[0, 1, S]], [[0, 1, S]], [[0, 1, P]]] * 5
    lo, hi = bootstrap_micro_f1(gold, pred, n=200, seed=0)
    assert lo <= 0.5 <= hi
    assert (lo, hi) == bootstrap_micro_f1(gold, pred, n=200, seed=0)


# --- RQ1 length tertiles (amendment 9, definition 4) ---
def test_length_tertiles_put_ties_in_the_lower_group():
    lengths = {1: 2, 2: 3, 3: 3, 4: 5, 5: 8, 6: 9}
    groups = length_tertiles(lengths)
    assert groups[1] == "short" and groups[6] == "long"
    assert set(groups.values()) == {"short", "medium", "long"}
    # records with the same length are always in the same group
    assert groups[2] == groups[3]


# --- RQ1 shared errors (amendment 9, definition 2) ---
def test_shared_errors():
    gold = {1: [[0, 1, P]], 2: [[0, 1, A]]}
    by_model = {f"m{k}": {1: [[0, 1, P], [2, 3, P]], 2: []} for k in range(6)}
    by_model["m6"] = {1: [[0, 1, P]], 2: []}
    by_model["m7"] = {1: [[0, 1, P]], 2: [[0, 1, A]]}
    res = shared_errors(gold, by_model, min_agree=6)
    # [2,3,P] in record 1 is produced by 6 models and wrong; [0,1,P] by 8 and right
    assert res["wrong_share_of_agreed"] == pytest.approx(0.5)
    # gold [0,1,A] in record 2 is produced by m7, so no gold span is missed by all eight
    assert res["missed_by_all_share"] == 0.0


# --- RQ2 (amendment 9, definition 3) ---
def test_span_errors_count_extra_plus_missed():
    assert span_errors([[0, 1, P], [2, 3, A]], [[0, 1, P], [2, 4, A], [5, 6, S]]) == 3
    assert span_errors([], []) == 0


def test_record_f1_between_two_label_sets():
    assert record_f1([], []) == 1.0
    assert record_f1([[0, 1, P]], []) == 0.0
    assert record_f1([[0, 1, P], [1, 2, A]], [[0, 1, P]]) == pytest.approx(2 / 3)


def test_auroc_perfect_random_and_ties():
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5


def test_topk_error_share():
    scores = {1: 0.9, 2: 0.8, 3: 0.1, 4: 0.0}
    errors = {1: 3, 2: 0, 3: 1, 4: 0}
    assert topk_error_share(scores, errors, 1) == pytest.approx(0.75)
    assert topk_error_share(scores, errors, 2) == pytest.approx(0.75)
    assert topk_error_share(scores, errors, 4) == pytest.approx(1.0)


# --- RQ2 signals ---
def test_disagreement_is_one_minus_mean_f1_with_the_other_models():
    labels = {"a": {1: [[0, 1, P]]}, "b": {1: [[0, 1, P]]}, "c": {1: []}}
    assert disagreement(labels, "a", 1) == pytest.approx(1 - (1.0 + 0.0) / 2)
    assert disagreement(labels, "c", 1) == pytest.approx(1.0)


def test_mean_ranks_gives_rank_one_to_the_highest_and_averages_ties():
    assert mean_ranks({1: 0.9, 2: 0.5, 3: 0.5, 4: 0.1}) == {1: 1.0, 2: 2.5, 3: 2.5, 4: 4.0}


def test_combined_rank_is_the_mean_of_the_two_ranks():
    unc = {1: 0.9, 2: 0.1, 3: 0.5}          # ranks 1, 3, 2
    dis = {1: 0.0, 2: 0.8, 3: 0.4}          # ranks 3, 1, 2
    assert combined_rank(unc, dis) == {1: 2.0, 2: 2.0, 3: 2.0}


# --- RQ3 / RQ3b training labels ---
def _records():
    return [{"id": i, "tokens": ["a", "b"], "spans": [[0, 1, P]]} for i in range(6)]


def test_apply_labels_replaces_only_train_pool_records_and_keeps_human_ones_for_review():
    split = {"train_pool": [0, 1, 2], "val": [3], "test": [4, 5]}
    llm = {0: [], 1: [[1, 2, A]], 2: [[0, 2, S]]}
    out = apply_labels(_records(), llm, split, reviewed=[2])
    assert out[0]["spans"] == [] and out[1]["spans"] == [[1, 2, A]]
    assert out[2]["spans"] == [[0, 1, P]]                 # reviewed: human label
    assert out[4]["spans"] == [[0, 1, P]]                 # test untouched
    assert _records()[0]["spans"] == [[0, 1, P]]          # input not changed


def test_apply_labels_refuses_labels_outside_train_pool_or_missing_records():
    split = {"train_pool": [0, 1], "val": [3], "test": [4, 5]}
    with pytest.raises(ValueError):
        apply_labels(_records(), {0: [], 1: [], 4: []}, split)
    with pytest.raises(ValueError):
        apply_labels(_records(), {0: []}, split)


def test_choose_review_signal_takes_the_lowest_combined_rank():
    combined = {10: 3.0, 11: 1.0, 12: 2.0, 13: 4.0}
    assert choose_review([10, 11, 12, 13], combined, 2, "signal", seed=1) == [11, 12]


def test_choose_review_random_is_nested_over_k_and_reproducible():
    pool = list(range(100))
    r50 = choose_review(pool, None, 50, "random", seed=3)
    r100 = choose_review(pool, None, 100, "random", seed=3)
    assert r50 == r100[:50] and r50 == choose_review(pool, None, 50, "random", seed=3)
    assert choose_review(pool, None, 0, "random", seed=3) == []


# --- RQ3c (amendment 10): the reviewed records alone ---
def test_review_training_set_with_llm_labels_is_all_of_train_pool():
    split = {"train_pool": [0, 1, 2], "val": [3], "test": [4, 5]}
    llm = {0: [], 1: [[1, 2, A]], 2: [[0, 2, S]]}
    train = review_training_set(_records(), llm, split, reviewed=[2], with_llm=True)
    assert [r["id"] for r in train] == [0, 1, 2]
    assert train[0]["spans"] == [] and train[2]["spans"] == [[0, 1, P]]


def test_review_training_set_without_llm_labels_is_only_the_reviewed_human_records():
    split = {"train_pool": [0, 1, 2], "val": [3], "test": [4, 5]}
    llm = {0: [], 1: [[1, 2, A]], 2: [[0, 2, S]]}
    train = review_training_set(_records(), llm, split, reviewed=[2, 0], with_llm=False)
    assert [r["id"] for r in train] == [2, 0]
    assert all(r["spans"] == [[0, 1, P]] for r in train)
    with pytest.raises(ValueError):
        review_training_set(_records(), llm, split, reviewed=[], with_llm=False)


def test_prompt_diff_is_zero_for_identical_runs_and_positive_for_a_better_prompt():
    gold = [[[0, 1, "PhysicalObject"]], [[1, 2, "State"], [0, 1, "PhysicalObject"]], []]
    bad = [[[0, 2, "PhysicalObject"]], [[1, 2, "State"]], []]
    d = prompt_diff(gold, [bad, bad], [bad, bad], n=200, seed=0)
    assert d["diff"] == 0 and d["ci95"] == [0.0, 0.0]
    d = prompt_diff(gold, [bad, bad], [gold, bad], n=200, seed=0)
    assert d["diff"] > 0 and d["ci95"][1] > 0
    assert d["a"] < d["b"]


def test_prompt_diff_by_type_ignores_other_types():
    gold = [[[0, 1, "PhysicalObject"], [1, 2, "State"]]]
    pred = [[[0, 1, "PhysicalObject"]]]
    d = prompt_diff(gold, [pred], [gold], n=50, seed=0, etype="PhysicalObject")
    assert d["a"] == 1.0 and d["b"] == 1.0
