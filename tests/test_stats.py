from src.stats import aulc, budget_to_reach


def test_aulc_constant_curve():
    assert abs(aulc([100, 200, 400], [0.5, 0.5, 0.5]) - 0.5) < 1e-9


def test_aulc_linear_curve():
    assert abs(aulc([0, 100, 200], [0.0, 0.5, 1.0]) - 0.5) < 1e-9


def test_reach_interpolates():
    assert abs(budget_to_reach([100, 200, 300], [0.5, 0.7, 0.9], 0.8) - 250) < 1e-6


def test_reach_already_at_first_point():
    assert budget_to_reach([100, 200], [0.9, 0.95], 0.8) == 100


def test_reach_never():
    assert budget_to_reach([100, 200], [0.5, 0.6], 0.8) is None


# --- paired comparison over seeds (RQ3, RQ3b) ---
from src.stats import paired


def test_paired_reports_mean_difference_wins_and_p():
    a = [0.80, 0.82, 0.81, 0.83, 0.85, 0.84, 0.80, 0.86, 0.82, 0.84]
    b = [x - 0.01 * (i + 1) for i, x in enumerate(a)]
    r = paired(a, b)
    assert abs(r["diff"] - 0.055) < 1e-9
    assert r["better"] == 10 and r["n"] == 10
    assert abs(r["p"] - 0.001953125) < 1e-9          # exact two-sided p for 10 of 10


def test_paired_all_equal_gives_nan_p():
    r = paired([0.5, 0.6], [0.5, 0.6])
    assert r["diff"] == 0 and r["better"] == 0 and r["p"] != r["p"]
