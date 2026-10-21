"""Metric correctness for analyze_vel.

Each test pins a property the sim2real conclusions depend on. Command smoothness
in particular is the headline number of the study, so it is checked against an
analytically known value rather than a regression snapshot. Flights are built as
plain numpy dicts in the columns exec_vel records; test_hover_scripts checks that
those are the columns it really writes.
"""

import json
import math
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "sim2real", "single_drone_exec", "hover"))
from analyze_vel import ACT_COLS, CMD_COLS, MOTOR_COLS, aggregate, compute_metrics, load_flight, main  # noqa: E402


def hover(n=1000, dt=0.01, pos=(0.0, 0.0, 1.0), tgt=(0.0, 0.0, 1.0)):
    """A 100 Hz flight that sits exactly on target with no command motion."""
    d = {"step": np.arange(n, dtype=np.float64), "control_time": np.arange(n) * dt}
    for i, axis in enumerate("xyz"):
        d[f"pos_{axis}"] = np.full(n, pos[i])
        d[f"target_{axis}"] = np.full(n, tgt[i])
        d[f"vel_{axis}"] = np.zeros(n)
    for c in CMD_COLS + ACT_COLS:
        d[c] = np.zeros(n)
    for c in MOTOR_COLS:
        d[c] = np.full(n, 30000.0)
    return d


def metrics(data, duration_s=None, skip_s=0.0):
    meta = {} if duration_s is None else {"duration_s": duration_s}
    return compute_metrics({"data": data, "meta": meta, "name": "run"}, skip_s=skip_s)


def write_parquet(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table(data), path)
    return path


def test_perfect_hover_scores_zero_error_and_zero_smoothness():
    m = metrics(hover(), duration_s=10.0)
    assert m["mean_pos_err_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["cmd_smoothness_mps_per_step"] == pytest.approx(0.0, abs=1e-9)
    assert m["crashed"] is False
    assert m["success"] is True


def test_constant_offset_gives_exactly_that_position_error():
    m = metrics(hover(pos=(0.3, 0.0, 1.0)), duration_s=10.0)
    assert m["mean_pos_err_m"] == pytest.approx(0.3, abs=1e-6)
    assert m["max_drift_m"] == pytest.approx(0.3, abs=1e-6)
    # a 0.3 m steady offset must not count as a success
    assert m["success"] is False


def test_smoothness_equals_known_square_wave_amplitude():
    """A +/-A square wave alternating every step has mean |delta| = 2A."""
    A = 0.25
    d = hover()
    d["vx"] = np.where(np.arange(1000) % 2 == 0, A, -A)
    assert metrics(d)["cmd_smoothness_mps_per_step"] == pytest.approx(2 * A, rel=1e-6)


def test_crash_is_detected_only_after_the_grace_period():
    # dips below the crash height at t < 0.5 s, inside the 3 s grace window
    early = hover()
    early["pos_z"][:50] = 0.02
    assert metrics(early, duration_s=10.0)["crashed"] is False

    late = hover()
    late["pos_z"][500:] = 0.02
    assert metrics(late, duration_s=10.0)["crashed"] is True


def test_short_run_is_flagged_as_ended_early():
    m = metrics(hover(n=200), duration_s=10.0)          # 2 s of a declared 10 s flight
    assert m["ended_early"] is True
    assert m["success"] is False


def test_effective_rate_is_measured_not_assumed():
    """Real flights ran at ~85 Hz, not the 100 Hz the simulator assumes; the
    analyser must report what it measures."""
    assert metrics(hover(n=500, dt=1 / 85.0))["effective_rate_hz"] == pytest.approx(85.0, rel=0.02)


def test_non_finite_samples_do_not_poison_the_metrics():
    """One dropped radio packet must cost one sample, not the whole flight."""
    d = hover(pos=(0.1, 0.0, 1.0))
    d["pos_x"][123] = np.nan
    m = metrics(d)
    assert math.isfinite(m["mean_pos_err_m"]), "a single NaN sample poisoned the mean"
    assert math.isfinite(m["cmd_smoothness_mps_per_step"])
    assert all(j is not None for j in m["jitter_std_xyz_m"])
    assert m["settle_time_s"] == 0.0
    assert m["n_nonfinite_samples"] == 1


def test_smoothness_per_second_is_rate_independent():
    """The real loop runs at ~85 Hz and the simulator at 100.

    Per-step smoothness therefore favours the faster loop for free. The per-second
    figure must describe the same underlying command signal at either rate.
    """
    def ramp(dt, n):
        # the same command slope, 0.5 m/s per second, sampled at two rates
        d = hover(n=n, dt=dt)
        d["vx"] = 0.5 * np.arange(n) * dt
        return d

    fast, slow = metrics(ramp(0.010, 1000)), metrics(ramp(0.0118, 848))
    # per step the slower loop looks 18% worse purely from its sampling rate
    assert slow["cmd_smoothness_mps_per_step"] > 1.1 * fast["cmd_smoothness_mps_per_step"]
    # per second the two agree
    assert slow["cmd_smoothness_mps2"] == pytest.approx(fast["cmd_smoothness_mps2"], rel=0.02)


def test_smoothness_skips_step_gaps():
    """A missing row must not count as one large command jump."""
    d = hover()
    d["vx"] = 0.01 * np.arange(1000)                    # 0.01 m/s per step
    keep = (np.arange(1000) < 500) | (np.arange(1000) >= 510)
    m = metrics({k: v[keep] for k, v in d.items()})
    assert m["cmd_smoothness_mps_per_step"] == pytest.approx(0.01, rel=1e-6)
    assert m["effective_rate_hz"] == pytest.approx(100.0, rel=1e-6)


def test_skip_s_drops_the_approach_transient():
    """The approach is not the hover, and it is not matched between sim and drone."""
    d = hover()
    d["pos_z"] = 0.5 + 0.5 * np.minimum(1.0, np.arange(1000) / 200.0)   # 2 s climb from 0.5 m, then exact hover
    whole, steady = metrics(d), metrics(d, skip_s=3.0)
    assert whole["mean_pos_err_m"] > 0.04, "the climb should show in the whole flight"
    assert steady["mean_pos_err_m"] == pytest.approx(0.0, abs=1e-6)
    assert steady["n_samples"] < whole["n_samples"]
    assert steady["skip_s"] == 3.0


def test_skip_s_beyond_the_flight_keeps_the_flight():
    """Trimming past the end must not silently produce an empty flight."""
    assert metrics(hover(n=100), skip_s=999.0)["n_samples"] == 100


def test_skip_s_does_not_mark_full_flight_ended_early():
    """The planned duration is compared with the whole flight, not the trimmed window."""
    m = metrics(hover(), duration_s=10.0, skip_s=3.0)
    assert m["ended_early"] is False
    assert m["success"] is True


def test_crash_during_skip_window_is_detected():
    d = hover()
    d["pos_z"][400:450] = 0.02                          # crash at t = 4 s, after the grace period
    assert metrics(d, skip_s=6.0)["crashed"] is True


def test_empty_recording_is_rejected():
    with pytest.raises(ValueError, match="no recorded steps"):
        metrics(hover(n=0))


def test_aggregate_ignores_missing_values():
    m = metrics(hover(pos=(0.1, 0.0, 1.0)))
    agg = aggregate([m, dict(m, cmd_smoothness_mps2=None), dict(m, mean_pos_err_m=0.3)])
    assert agg["cmd_smoothness_mps2"] == {"mean": m["cmd_smoothness_mps2"], "std": 0.0}
    assert agg["mean_pos_err_m"]["mean"] == pytest.approx((0.1 + 0.1 + 0.3) / 3, abs=1e-4)
    assert aggregate([dict(m, mean_motor_pwm=None)])["mean_motor_pwm"] is None


def test_load_flight_reads_parquet_file_and_dir(tmp_path):
    path = write_parquet(str(tmp_path / "run" / "flight.parquet"), hover())
    with open(tmp_path / "run" / "metadata.json", "w") as f:
        json.dump({"duration_s": 10.0}, f)
    by_file, by_dir = load_flight(path), load_flight(str(tmp_path / "run"))
    assert by_file["path"] == by_dir["path"] == path
    assert by_file["meta"] == {"duration_s": 10.0}
    np.testing.assert_array_equal(by_file["data"]["vx"], hover()["vx"])
    assert compute_metrics(by_file)["success"] is True


def test_trial_names_are_unique_across_run_dirs(tmp_path):
    a = load_flight(write_parquet(str(tmp_path / "a" / "flight.parquet"), hover()))
    b = load_flight(write_parquet(str(tmp_path / "b" / "flight.parquet"), hover()))
    assert a["name"] != b["name"]


def test_non_velocity_recording_is_rejected(tmp_path):
    d = hover()
    del d["vx"]
    with pytest.raises(ValueError, match="velocity-mode"):
        load_flight(write_parquet(str(tmp_path / "flight.parquet"), d))


def test_json_report_is_valid_json(tmp_path):
    """A one-step flight has no smoothness or rate; the report must say null, not NaN."""
    one = write_parquet(str(tmp_path / "one" / "flight.parquet"), hover(n=1))
    full = write_parquet(str(tmp_path / "full" / "flight.parquet"), hover())
    out = str(tmp_path / "report.json")
    main([one, full, "--json-out", out, "--skip-s", "1"])
    with open(out) as f:
        report = json.load(f, parse_constant=lambda c: pytest.fail(f"invalid JSON constant {c}"))
    assert report["per_trial"][0]["cmd_smoothness_mps2"] is None
    assert report["aggregate"]["n_trials"] == 2
