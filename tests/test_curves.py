"""curves.py computes the numbers configs are ranked and plotted on.

The same functions write the run summary during training and recompute it in
analyze.py, so a wrong window or normalisation would silently change which
config wins. Pure numpy: runs anywhere.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
import curves  # noqa: E402


def test_smoothing_window_is_a_fraction_of_training():
    steps = np.arange(11) * 10.0  # 0..100
    values = np.arange(11, dtype=float)
    # 20 % of 100 steps: points within (s - 20, s], i.e. the current and previous one
    smoothed = curves.smooth(steps, values, 0.2)
    assert smoothed[0] == 0.0
    np.testing.assert_allclose(smoothed[1:], values[1:] - 0.5)


def test_smoothing_ignores_the_training_length_and_log_density():
    steps = np.linspace(0, 1.0, 1001)
    values = steps**2
    # the same curve over 1000x more steps is smoothed the same
    np.testing.assert_allclose(curves.smooth(steps * 1e3, values, 0.3), curves.smooth(steps, values, 0.3))
    # denser logging converges to the mean over the last 30 %: (1 - 0.7^3) / (3 * 0.3)
    dense = np.linspace(0, 1.0, 10001)
    assert curves.smooth(steps, values, 0.3)[-1] == pytest.approx(0.73, rel=0.01)
    assert curves.smooth(dense, dense**2, 0.3)[-1] == pytest.approx(0.73, rel=0.001)


def test_no_smoothing_returns_the_values():
    values = np.array([3.0, 1.0, 2.0])
    np.testing.assert_array_equal(curves.smooth([0, 1, 2], values, 0.0), values)


def test_auc_is_the_mean_value_over_training():
    assert curves.auc([0, 50, 100], [2.0, 2.0, 2.0]) == pytest.approx(2.0)
    assert curves.auc([0, 100], [0.0, 1.0]) == pytest.approx(0.5)
    # irregular steps are weighted by the step span, not the point count
    assert curves.auc([0, 90, 100], [0.0, 0.0, 1.0]) == pytest.approx(0.05)


def test_summarize_gives_the_summary_keys():
    summary = curves.summarize([0, 1, 2], [0.0, 1.0, 2.0], 0.0)
    assert summary == {"auc_return": 1.0, "max_smoothed_return": 2.0, "smooth_frac": 0.0}


def test_iqm_drops_the_outer_quartiles():
    assert curves.iqm([1, 2, 3, 100]) == pytest.approx(2.5)
    assert curves.iqm([1, 2, 30]) == pytest.approx(11.0)  # < 4 seeds: the mean
    np.testing.assert_allclose(curves.iqm([[1, 10], [2, 20], [3, 30], [100, 0]]), [2.5, 15.0])


def test_ci_contains_the_iqm_and_shrinks_with_more_seeds():
    rng = np.random.default_rng(1)
    few = rng.normal(0, 1, (3, 5))
    many = rng.normal(0, 1, (30, 5))
    for curves_ in (few, many):
        low, high = curves.bootstrap_ci(curves_, reps=500)
        center = curves.iqm(curves_)
        assert np.all(low <= center) and np.all(center <= high)
    width = lambda c: np.mean(np.subtract(*curves.bootstrap_ci(c, reps=500)[::-1]))  # noqa: E731
    assert width(many) < width(few)


def test_align_covers_only_the_steps_every_run_has():
    a, b = np.array([0.0, 10.0]), np.array([2.0, 8.0, 12.0])
    grid, values = curves.align([a, b], [a, b], 5)
    np.testing.assert_allclose(grid, [2, 4, 6, 8, 10])  # latest start, earliest end
    np.testing.assert_allclose(values, [grid, grid])
