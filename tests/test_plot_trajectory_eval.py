"""plot_trajectory_eval.py on synthetic circle rollouts in the play.py --eval_csv layout: the
statistics are right, labels pool, the figures and tables are written, and rollouts with different
references are flagged."""

import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "sim_eval", "single_drone_exec", "traj_tracking"))
import plot_trajectory_eval as pte  # noqa: E402

DT, T, RADIUS, SPEED = 0.01, 300, 1.5, 0.4


def circle(steps, radius=RADIUS, phase=0.0, start=1):
    """(steps, 3) points of the circle reference at policy steps start, start + 1, ... [m]."""
    ang = phase + SPEED * (np.arange(steps) + start) * DT
    return np.stack([radius * np.cos(ang), radius * np.sin(ang), np.full(steps, 1.5)], axis=-1)


def write_run(path, offsets, crash_at=None, radius=RADIUS, phases=None):
    """A play.py --eval_csv file where episode n flies the circle shifted by offsets[n] (m, world).

    crash_at: {episode: steps} for episodes that die early. phases: per-episode reference phase
    (each episode its own reference, as with random yaw/phase ranges)."""
    n = len(offsets)
    phases = phases if phases is not None else [0.0] * n
    ref = np.stack([circle(T, radius, p) for p in phases])
    pos = ref + np.asarray(offsets, dtype=float)[:, None, :]
    length = np.full(n, T)
    died = np.zeros(n, dtype=bool)
    for ep, steps in (crash_at or {}).items():
        length[ep], died[ep] = steps, True
    vel_w = np.gradient(pos, DT, axis=1)
    valid = pte.valid_mask(length, T)
    rows = valid.reshape(-1)  # one row per (episode, step) of the episode, as play.py writes
    columns = {"episode": np.repeat(np.arange(n), T)[rows], "step": np.tile(np.arange(1, T + 1), n)[rows]}
    for prefix, values in (("pos", pos), ("target", ref), ("vel", vel_w)):
        columns.update({f"{prefix}_{a}": values[..., k].reshape(-1)[rows] for k, a in enumerate("xyz")})
    for k, a in enumerate(("vx", "vy", "vz")):
        columns[f"action_{a}"] = (vel_w[..., k] / 3.0).reshape(-1)[rows]
        columns[a] = vel_w[..., k].reshape(-1)[rows]
    footer = {"source": "sim", "task": "Vel-Tracking", "checkpoint": "x.pt", "dt": repr(DT),
              "env_cfg": json.dumps({"max_velocity": 3.0, "reset_thres": 0.8, "control_mode": "velocity"}),
              "died": json.dumps(died.tolist())}
    pq.write_table(pa.table(columns).replace_schema_metadata(footer), path)
    return str(path)


def test_mean_over_alive_episodes_drops_crashed_ones_after_their_crash():
    x = np.array([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]])
    valid = pte.valid_mask([4, 2], 4)
    assert pte.mean_alive(x, valid) == pytest.approx([2.0, 2.0, 1.0, 1.0])
    lo, mid, hi = pte.band(x, pte.valid_mask([0, 0], 4))  # no episode flying: NaN, no warning
    assert np.isnan(mid).all()


def test_along_track_is_negative_when_lagging_and_cross_track_is_the_radial_offset():
    ref = circle(T)[None]
    lag = circle(T, phase=-0.05)[None]                 # 0.05 rad behind = 7.5 cm of arc
    along, cross = pte.track_errors(lag, ref)
    assert np.all(along[0, 1:-1] < 0)
    assert along[0, 1:-1] == pytest.approx(-RADIUS * np.sin(0.05), rel=1e-3)
    wide = circle(T, radius=RADIUS + 0.1)[None]        # 10 cm outside the circle, not behind
    along, cross = pte.track_errors(wide, ref)
    assert along[0, 1:-1] == pytest.approx(0.0, abs=1e-6)
    assert cross[0, 1:-1] == pytest.approx(0.1, abs=1e-6)


def test_iqm_is_the_mean_of_the_central_half():
    assert pte.iqm([0, 1, 2, 3, 4, 5, 6, 100, np.nan]) == pytest.approx(3.5)


def test_runs_with_one_label_are_pooled(tmp_path):
    a = write_run(tmp_path / "s1.parquet", [[0.0, 0.0, 0.1]] * 3)
    b = write_run(tmp_path / "s2.parquet", [[0.0, 0.0, -0.1]] * 2, crash_at={0: 100})
    groups = pte.group_by_label([pte.load_run(a, "policy"), pte.load_run(b, "policy")])
    assert list(groups) == ["policy"]
    g = groups["policy"]
    assert len(g["length"]) == 5 and g["files"] == [a, b]
    # +10 cm and -10 cm episodes average to the reference while all fly; after the crash, 3 vs 1
    mean = pte.mean_alive(g["pos"], pte.valid_mask(g["length"], T))
    assert mean[:100, 2] == pytest.approx(1.5 + (3 * 0.1 - 2 * 0.1) / 5)
    assert mean[100:, 2] == pytest.approx(1.5 + (3 * 0.1 - 0.1) / 4)
    # IQM per step: 5 flying -> drop 1 at each end of [-.1, -.1, .1, .1, .1]; 4 flying -> [-.1, .1, .1, .1]
    path = pte.iqm_alive(g["pos"], pte.valid_mask(g["length"], T))
    assert path[:100, 2] == pytest.approx(1.5 + (-0.1 + 0.1 + 0.1) / 3)
    assert path[100:, 2] == pytest.approx(1.5 + 0.1)
    lo, mid, hi = pte.band(g["pos"][..., 2], pte.valid_mask(g["length"], T))
    assert lo[0] == pytest.approx(1.4) and hi[0] == pytest.approx(1.6) and mid[0] == pytest.approx(path[0, 2])
    row = pte.summarize(groups)[0]
    assert row["episodes"] == 5 and row["crash_rate"] == pytest.approx(0.2)
    assert row["rmse_iqm"] == pytest.approx(0.1) and row["rmse_p10"] == pytest.approx(0.1)
    assert row["cross_iqm"] == pytest.approx(0.1, abs=1e-6)


def test_main_writes_figures_and_tables(tmp_path, capsys):
    nominal = write_run(tmp_path / "nominal.parquet", [[0.02, 0.0, 0.0], [0.0, 0.03, 0.0]])
    dr = write_run(tmp_path / "train DR.parquet", [[0.1, 0.0, 0.0], [0.0, -0.2, 0.05]], crash_at={1: 150})  # label = name
    out = tmp_path / "figs"
    assert pte.main([nominal, dr, "--out", str(out), "--name", "t"]) == 0
    printed = capsys.readouterr().out
    assert "WARN" not in printed
    # a top view and a time-series figure per setup, one summary comparing them
    for name in ("top_t_nominal.png", "time_t_nominal.png", "top_t_train-DR.png", "time_t_train-DR.png",
                 "summary_t.png", "summary_t.csv", "summary_t.md"):
        assert (out / name).stat().st_size > 0 and f"saved {out / name}" in printed
    assert not (out / "top_t.png").exists()
    md = (out / "summary_t.md").read_text()
    assert "| nominal |" in md and "| train DR |" in md
    assert pte.main([nominal, dr, "--out", str(out), "--name", "one", "--env", "1"]) == 0
    assert (out / "top_one_nominal.png").stat().st_size > 0
    assert pte.main([nominal, dr, "--out", str(out), "--name", "both", "--overlay"]) == 0
    assert (out / "top_both.png").stat().st_size > 0 and (out / "time_both.png").stat().st_size > 0
    assert not (out / "top_both_nominal.png").exists()


def test_labels_override_and_pool_files(tmp_path, capsys):
    a = write_run(tmp_path / "a.parquet", [[0.0, 0.0, 0.0]])
    b = write_run(tmp_path / "b.parquet", [[0.0, 0.0, 0.0]])
    assert pte.main([a, b, "--labels", "same", "same", "--out", str(tmp_path)]) == 0
    assert "same: 2 episodes" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pte.main([a, b, "--labels", "only-one"])


def test_different_references_are_flagged(tmp_path, capsys):
    small = write_run(tmp_path / "small.parquet", [[0.0, 0.0, 0.0]])
    big = write_run(tmp_path / "big.parquet", [[0.0, 0.0, 0.0]], radius=2.0)
    assert pte.main([small, big, "--out", str(tmp_path)]) == 0
    assert "the labels fly different references" in capsys.readouterr().out

    random = write_run(tmp_path / "random.parquet", [[0.0, 0.0, 0.0]] * 3, phases=[0.0, 1.0, 2.0])
    groups = pte.group_by_label([pte.load_run(random)])
    assert pte.shared_reference(groups) == (False, False)
    assert pte.main([random, "--out", str(tmp_path), "--name", "random"]) == 0
    assert "episodes fly different references" in capsys.readouterr().out


def test_unrecorded_last_step_is_cut_from_the_plots(tmp_path):
    """A timed-out episode lasts T - 1 steps: the zero-filled last slot must not reach the figure."""
    path = write_run(tmp_path / "timeout.parquet", [[0.0, 0.0, 0.0]] * 2, crash_at={0: T - 1, 1: 50})
    run = pte.load_run(path, "t")
    run["died"][:] = [False, True]  # episode 0 timed out, episode 1 crashed
    groups = pte.trim_to_episodes(pte.group_by_label([run]))
    assert groups["t"]["pos"].shape[1] == T - 1
    line = pte._ref_line(groups["t"])
    assert np.isfinite(line).all() and line[:, 2] == pytest.approx(1.5)  # from the longest episode
    assert pte.plot_topview(groups, str(tmp_path / "top.png")) and pte.plot_timeseries(groups, str(tmp_path / "time.png"))


def test_metadata_comes_from_the_env_config(tmp_path):
    run = pte.load_run(write_run(tmp_path / "m.parquet", [[0.0, 0.0, 0.0]], crash_at={0: 100}))
    assert run["label"] == "m" and run["reset_thres"] == 0.8 and run["dt"] == DT
    assert run["rmse"].shape == (1,) and "tracking_rmse" not in run and run["died"].tolist() == [True]
    assert run["pos"].shape == (1, 100, 3) and run["setpoint"].shape == (1, 100, 3)  # rows back to (N, T, 3)
    assert run["length"].tolist() == [100] and run["rmse"][0] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(run["ref"][0], circle(100), atol=1e-6)
