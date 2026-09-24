"""CL-2 asks whether a simulator ranking survives the transfer to hardware.

With four policies the statistic is fragile, so it is pinned here: a sign error or
a rank convention flip would otherwise reverse a headline conclusion silently.
"""

import numpy as np
import pytest

from report_gap import spearman


def test_identical_ranking_is_perfect():
    srcc, conc, pairs = spearman(np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 30, 40]))
    assert srcc == pytest.approx(1.0)
    assert (conc, pairs) == (6, 6)


def test_reversed_ranking_is_minus_one():
    srcc, conc, pairs = spearman(np.array([1.0, 2, 3, 4]), np.array([40.0, 30, 20, 10]))
    assert srcc == pytest.approx(-1.0)
    assert (conc, pairs) == (0, 6)


def test_one_swapped_pair():
    """Swapping the two closest ranks costs 0.2 with four points."""
    srcc, conc, pairs = spearman(np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 40, 30]))
    assert srcc == pytest.approx(0.8)
    assert (conc, pairs) == (5, 6)


def test_a_monotone_transform_does_not_change_the_rank():
    a = np.array([0.5, 0.7, 0.44, 0.49])
    srcc_lin, _, _ = spearman(a, a * 3.0 + 1.0)
    srcc_log, _, _ = spearman(a, np.log(a))
    assert srcc_lin == pytest.approx(1.0)
    assert srcc_log == pytest.approx(1.0)


def test_too_few_policies_returns_nan_rather_than_a_number():
    srcc, conc, pairs = spearman(np.array([1.0, 2]), np.array([2.0, 1]))
    assert np.isnan(srcc)
    assert pairs == 0
