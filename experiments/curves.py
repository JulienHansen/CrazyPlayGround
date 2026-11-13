"""Learning-curve statistics shared by training (run summary) and analyze.py.

Both sides import these functions, so the AUC and the smoothed maximum stored
in a run's W&B summary are the same numbers analyze.py would compute from the
run's history.

Steps are environment steps summed over all envs (skrl's timestep times
``num_envs``). The smoothing window is a fraction of the run's training length
in those steps, so a 5 % window means the same thing whatever the number of
steps or the log interval.

IQM and confidence intervals follow Agarwal et al. (2021), *Deep RL at the Edge
of the Statistical Precipice*: the interquartile mean across seeds, and a
percentile bootstrap that resamples seeds.
"""

from __future__ import annotations

import numpy as np

RETURN_METRIC = "Reward / Total reward (mean)"  # skrl's undiscounted mean episode return
SMOOTH_FRAC = 0.05


def smooth(steps, values, window_frac: float = SMOOTH_FRAC) -> np.ndarray:
    """Trailing moving average over the last ``window_frac`` of the training length.

    The window is measured in steps, not log points, so irregular logging is
    averaged correctly. Early points average over the shorter history they have.

    Args:
        steps: (n,) increasing env steps.
        values: (n,) metric values at ``steps``.
        window_frac: window as a fraction of ``steps[-1] - steps[0]``; 0 = no smoothing.
    """
    steps, values = np.asarray(steps, dtype=float), np.asarray(values, dtype=float)
    if len(steps) < 2 or window_frac <= 0:
        return values.copy()
    frac = (steps - steps[0]) / (steps[-1] - steps[0])
    # first point inside (frac[i] - window_frac, frac[i]] for every i; the tolerance keeps a
    # point that sits exactly on the window edge out whatever rounding the step scale brings
    first = np.searchsorted(frac, frac - window_frac + 1e-9, side="left")
    cumsum = np.concatenate(([0.0], np.cumsum(values)))
    last = np.arange(1, len(values) + 1)
    return (cumsum[last] - cumsum[first]) / (last - first)


def auc(steps, values) -> float:
    """Area under the curve divided by the step span: the mean value over training.

    Normalised so it stays in the metric's units; only compare runs trained
    for the same number of steps.
    """
    steps, values = np.asarray(steps, dtype=float), np.asarray(values, dtype=float)
    if len(steps) < 2:
        return float(values[0]) if len(values) else float("nan")
    area = np.sum((values[1:] + values[:-1]) * np.diff(steps)) / 2
    return float(area / (steps[-1] - steps[0]))


def max_smoothed(steps, values, window_frac: float = SMOOTH_FRAC) -> float:
    """Best smoothed value over training."""
    if len(values) == 0:
        return float("nan")
    return float(np.max(smooth(steps, values, window_frac)))


def summarize(steps, values, window_frac: float = SMOOTH_FRAC) -> dict:
    """The run-summary keys analyze.py ranks on."""
    return {
        "auc_return": auc(steps, values),
        "max_smoothed_return": max_smoothed(steps, values, window_frac),
        "smooth_frac": window_frac,
    }


def iqm(x, axis: int = 0) -> np.ndarray:
    """Interquartile mean: the mean after dropping the lowest and highest 25 %.

    Same cut as ``scipy.stats.trim_mean(x, 0.25)``, so with fewer than 4 seeds
    nothing is dropped and the IQM is the mean.
    """
    x = np.sort(np.asarray(x, dtype=float), axis=axis)
    n = x.shape[axis]
    cut = int(0.25 * n)
    return np.take(x, np.arange(cut, n - cut), axis=axis).mean(axis=axis)


def bootstrap_ci(curves, reps: int = 2000, ci: float = 0.95, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap CI of the IQM across seeds, at every grid point.

    Args:
        curves: (num_seeds, num_points) one config's aligned curves.
        reps: bootstrap resamples of the seeds (with replacement).
        ci: confidence level.
        rng: numpy Generator; a fixed seed by default, so figures are reproducible.

    Returns:
        (low, high), each (num_points,).
    """
    curves = np.asarray(curves, dtype=float)
    rng = np.random.default_rng(0) if rng is None else rng
    idx = rng.integers(0, curves.shape[0], size=(reps, curves.shape[0]))
    stats = iqm(curves[idx], axis=1)  # (reps, num_points)
    alpha = (1 - ci) / 2
    return np.quantile(stats, alpha, axis=0), np.quantile(stats, 1 - alpha, axis=0)


def align(steps_list, values_list, num_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate several runs onto one step grid.

    The grid spans the steps every run covers (latest first step to earliest
    last step), so no run is extrapolated.

    Returns:
        grid (num_points,) and values (num_runs, num_points).
    """
    start = max(float(s[0]) for s in steps_list)
    end = min(float(s[-1]) for s in steps_list)
    grid = np.linspace(start, end, num_points)
    return grid, np.stack([np.interp(grid, s, v) for s, v in zip(steps_list, values_list)])
