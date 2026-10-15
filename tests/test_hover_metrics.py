"""Metric correctness for analyze_hover.

Each test pins a property the sim2real conclusions depend on. `cmd_smoothness`
in particular is the headline number of the whole study, so it is checked against
an analytically known value rather than a regression snapshot.
"""

import math
import os

import numpy as np
import pytest

from analyze_hover import compute_metrics, load_flight
from conftest import write_flight


def test_perfect_hover_scores_zero_error_and_zero_chatter(tmp_path, columns, hover_rows):
    d = write_flight(tmp_path / "run", hover_rows(columns), columns, duration_s=10.0)
    m = compute_metrics(load_flight(str(d)))
    assert m["mean_pos_err_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["cmd_smoothness_mps_per_step"] == pytest.approx(0.0, abs=1e-9)
    assert m["crashed"] is False
    assert m["success"] is True


def test_constant_offset_gives_exactly_that_position_error(tmp_path, columns, hover_rows):
    rows = hover_rows(columns, pos=(0.3, 0.0, 1.0), tgt=(0.0, 0.0, 1.0))
    d = write_flight(tmp_path / "run", rows, columns, duration_s=10.0)
    m = compute_metrics(load_flight(str(d)))
    assert m["mean_pos_err_m"] == pytest.approx(0.3, abs=1e-6)
    assert m["max_drift_m"] == pytest.approx(0.3, abs=1e-6)
    # a 0.3 m steady offset must not count as a success
    assert m["success"] is False


def test_chatter_equals_known_square_wave_amplitude(tmp_path, columns, hover_rows):
    """A +/-A square wave alternating every step has mean |delta| = 2A."""
    A, n = 0.25, 1000
    rows = hover_rows(columns, n=n)
    for i, r in enumerate(rows):
        r["cmd_vx"] = A if i % 2 == 0 else -A
    d = write_flight(tmp_path / "run", rows, columns, duration_s=10.0)
    m = compute_metrics(load_flight(str(d)))
    assert m["cmd_smoothness_mps_per_step"] == pytest.approx(2 * A, rel=1e-6)


def test_crash_is_detected_only_after_the_grace_period(tmp_path, columns, hover_rows):
    # dips below the crash height at t = 1 s, inside the 3 s grace window
    early = hover_rows(columns)
    for r in early[:50]:
        r["pos_z"] = 0.02
    d = write_flight(tmp_path / "early", early, columns, duration_s=10.0)
    assert compute_metrics(load_flight(str(d)))["crashed"] is False

    late = hover_rows(columns)
    for r in late[500:]:
        r["pos_z"] = 0.02
    d = write_flight(tmp_path / "late", late, columns, duration_s=10.0)
    assert compute_metrics(load_flight(str(d)))["crashed"] is True


def test_short_run_is_flagged_as_ended_early(tmp_path, columns, hover_rows):
    rows = hover_rows(columns, n=200)          # 2 s of a declared 10 s flight
    d = write_flight(tmp_path / "run", rows, columns, duration_s=10.0)
    m = compute_metrics(load_flight(str(d)))
    assert m["ended_early"] is True
    assert m["success"] is False


def test_effective_rate_is_measured_not_assumed(tmp_path, columns, hover_rows):
    """Real flights ran at ~85 Hz, not the 100 Hz the simulator assumes; the
    analyser must report what it measures."""
    rows = hover_rows(columns, n=500, dt=1 / 85.0)
    d = write_flight(tmp_path / "run", rows, columns, duration_s=500 / 85.0)
    m = compute_metrics(load_flight(str(d)))
    assert m["effective_rate_hz"] == pytest.approx(85.0, rel=0.02)


def test_non_finite_samples_do_not_poison_the_aggregate(tmp_path, columns, hover_rows):
    """Regression: masking accumulators by multiplication let NaN through, because
    nan * 0 == nan, so one diverging sample destroyed the whole average."""
    rows = hover_rows(columns, pos=(0.1, 0.0, 1.0))
    rows[123]["pos_x"] = float("nan")
    d = write_flight(tmp_path / "run", rows, columns, duration_s=10.0)
    m = compute_metrics(load_flight(str(d)))
    assert math.isfinite(m["mean_pos_err_m"]), "a single NaN sample poisoned the mean"
    assert math.isfinite(m["cmd_smoothness_mps_per_step"])


def test_chatter_per_second_is_rate_independent(tmp_path, columns, hover_rows):
    """The real loop runs at ~85 Hz and the simulator at 100.

    Per-step chatter therefore favours the faster loop for free. The per-second
    figure must describe the same underlying command signal at either rate.
    """
    from analyze_hover import load_flight, compute_metrics

    def ramp(dt, n):
        # the same command slope, 0.5 m/s per second, sampled at two rates
        rows = hover_rows(columns, n=n, dt=dt)
        for i, r in enumerate(rows):
            r["cmd_vx"] = 0.5 * i * dt
        return rows

    fast = compute_metrics(load_flight(str(
        write_flight(tmp_path / "fast", ramp(0.010, 1000), columns, duration_s=10.0))))
    slow = compute_metrics(load_flight(str(
        write_flight(tmp_path / "slow", ramp(0.0118, 848), columns, duration_s=10.0))))

    # per step the slower loop looks 18% worse purely from its sampling rate
    assert slow["cmd_smoothness_mps_per_step"] > 1.1 * fast["cmd_smoothness_mps_per_step"]
    # per second the two agree
    assert slow["cmd_smoothness_mps2"] == pytest.approx(fast["cmd_smoothness_mps2"], rel=0.02)


def test_skip_s_drops_the_approach_transient(tmp_path, columns, hover_rows):
    """The approach is not the hover, and it is not matched between sim and drone.

    The simulator starts 0.50 m below the goal; the real drone took off to 0.36 m
    under a 1.00 m goal. Charging the drone for the longer climb inflates the gap.
    """
    rows = hover_rows(columns, n=1000, dt=0.01)
    for i, r in enumerate(rows):          # 2 s climb from 0.5 m, then exact hover
        r["pos_z"] = 0.5 + 0.5 * min(1.0, i / 200.0)

    d = write_flight(tmp_path / "run", rows, columns, duration_s=10.0)
    whole = compute_metrics(load_flight(str(d)))
    steady = compute_metrics(load_flight(str(d)), skip_s=3.0)

    assert whole["mean_pos_err_m"] > 0.04, "the climb should show in the whole flight"
    assert steady["mean_pos_err_m"] == pytest.approx(0.0, abs=1e-6)
    assert steady["n_samples"] < whole["n_samples"]
    assert steady["skip_s"] == 3.0


def test_skip_s_beyond_the_flight_keeps_the_flight(tmp_path, columns, hover_rows):
    """Trimming past the end must not silently produce an empty flight."""
    d = write_flight(tmp_path / "run", hover_rows(columns, n=100), columns, duration_s=1.0)
    m = compute_metrics(load_flight(str(d)), skip_s=999.0)
    assert m["n_samples"] == 100
