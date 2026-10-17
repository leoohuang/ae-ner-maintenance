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
