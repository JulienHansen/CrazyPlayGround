"""Tests for execution/sim2real/hovering/plot_hover.py."""

import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "execution", "sim2real", "hovering"))
from plot_hover import (  # noqa: E402
    build_plot_parser,
    load_recording,
    load_telemetry,
    main,
    plot_comparison,
    plot_step_response,
    plot_target_error,
)

N = 50
EPOCH = 1_793_890_284.589  # control_time of a real flight: time.time() at each step


def _write(path, columns, metadata=None):
    """Parquet as the recorders write it: `step` int64, the rest float64, str footer."""
    table = pa.table({k: pa.array(v, type=pa.int64() if k == "step" else pa.float64()) for k, v in columns.items()})
    pq.write_table(table.replace_schema_metadata(metadata or {}), path)
    return str(path)


def _flight(n=N, t0=0.0, start=(0.0, 0.0, 0.5), target=(0.5, 0.0, 1.0), cmd_scale=1.0):
    """exec_vel.py columns: a first-order approach from `start` to `target`."""
    t = np.arange(n) * 0.01
    s = 1.0 - np.exp(-t / 0.1)
    cols = {"step": list(range(n)), "control_time": t0 + t}
    for i, ax in enumerate("xyz"):
        cols[f"pos_{ax}"] = start[i] + (target[i] - start[i]) * s
        cols[f"vel_{ax}"] = (target[i] - start[i]) * np.exp(-t / 0.1) / 0.1
        cols[f"target_{ax}"] = np.full(n, target[i])
        cols[f"v{ax}"] = cmd_scale * np.clip(cols[f"vel_{ax}"], -1.0, 1.0)
    return cols


def real_flight(path, **kwargs):
    return _write(path, _flight(t0=EPOCH, **kwargs))   # FlightRecorder writes no `source`


def sim_flight(path, **kwargs):
    return _write(path, _flight(**kwargs), {"source": "sim"})


def test_build_plot_parser():
    parser = build_plot_parser()
    args = parser.parse_args(["flight.parquet", "--ylim", "-0.1", "0.6"])
    assert args.files == ["flight.parquet"]
    assert args.ylim == [-0.1, 0.6]
    args = parser.parse_args(["real.parquet", "sim.parquet", "--compare", "--cmd-ylim", "-1", "1"])
    assert args.files == ["real.parquet", "sim.parquet"] and args.compare and args.cmd_ylim == [-1.0, 1.0]
    assert args.out is None
    with pytest.raises(SystemExit):   # recordings carry their target: no --target option
        parser.parse_args(["flight.parquet", "--target", "0.1", "0.2", "1.0"])


def test_load_recording_real_flight(tmp_path):
    rec = load_recording(real_flight(tmp_path / "flight.parquet"))
    cols = _flight()
    assert rec.source == "real" and rec.name == "flight"
    assert rec.times[0] == 0.0 and rec.times[-1] == pytest.approx((N - 1) * 0.01)   # epoch removed
    assert rec.positions[:, 2] == pytest.approx(cols["pos_z"])
    assert rec.velocities[:, 0] == pytest.approx(cols["vel_x"])
    assert rec.cmd_velocities[:, 0] == pytest.approx(cols["vx"])                    # from vx, not cmd_vx
    assert rec.targets.shape == (N, 3) and rec.targets[0] == pytest.approx([0.5, 0.0, 1.0])


def test_load_recording_sim_source_from_footer(tmp_path):
    assert load_recording(sim_flight(tmp_path / "sim.parquet")).source == "sim"


def test_load_recording_accepts_only_the_exec_vel_format(tmp_path):
    """One format: a file with other column names (e.g. old hover_sim.py time / cmd_v*) is refused by name."""
    cols = _flight()
    other = {"step": cols["step"], "time": cols["control_time"],
             **{k: cols[k] for k in cols if k.startswith(("pos_", "vel_"))},
             **{f"cmd_v{ax}": cols[f"v{ax}"] for ax in "xyz"}}
    with pytest.raises(ValueError, match=r"missing \['control_time', 'target_x'"):
        load_recording(_write(tmp_path / "other.parquet", other))


def test_load_recording_rejects_an_empty_file(tmp_path):
    path = _write(tmp_path / "empty.parquet", {k: [] for k in _flight()})
    with pytest.raises(ValueError, match="holds no rows"):
        load_recording(path)


def test_plot_comparison_overlays_position_velocity_and_command(tmp_path, monkeypatch, capsys):
    import matplotlib.pyplot as plt

    figs = []
    monkeypatch.setattr(plt, "close", lambda fig=None: figs.append(fig))   # keep the figure to inspect it
    real = load_recording(real_flight(tmp_path / "real.parquet"))
    sim = load_recording(sim_flight(tmp_path / "sim.parquet", start=(0.2, 0.0, 0.5), cmd_scale=0.5))
    out = plot_comparison([real, sim], str(tmp_path / "cmp.png"))

    assert os.path.getsize(out) > 0
    (fig,) = figs
    grid = np.array(fig.axes).reshape(3, 3)
    for r, attr in enumerate(("positions", "velocities", "cmd_velocities")):
        for i in range(3):
            lines = {line.get_gid(): line for line in grid[r, i].lines if line.get_gid()}
            assert set(lines) == {"real:real", "sim:sim"} | ({"target"} if r == 0 else set())
            assert lines["real:real"].get_ydata() == pytest.approx(getattr(real, attr)[:, i])
            assert lines["sim:sim"].get_ydata() == pytest.approx(getattr(sim, attr)[:, i])
            assert lines["sim:sim"].get_linestyle() == "--" and lines["real:real"].get_linestyle() == "-"
    assert len(fig.legends) == 1
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["real: real", "sim: sim", "target"]
    assert "sim starts 20 cm from real" in capsys.readouterr().out


def test_main_compare_writes_one_figure(tmp_path):
    files = [real_flight(tmp_path / "real.parquet"), sim_flight(tmp_path / "sim.parquet")]
    main(files + ["--compare"])
    assert (tmp_path / "sim_real_comparison.png").stat().st_size > 0
    main(files + ["--compare", "--out", str(tmp_path / "figs" / "cmp.png")])
    assert (tmp_path / "figs" / "cmp.png").stat().st_size > 0


def test_main_compare_needs_two_recordings(tmp_path):
    with pytest.raises(SystemExit):
        main([real_flight(tmp_path / "real.parquet"), "--compare"])


def test_main_empty_recording_is_a_clean_cli_error(tmp_path, capsys):
    path = _write(tmp_path / "empty.parquet", {k: [] for k in _flight()})
    with pytest.raises(SystemExit):
        main([path])
    assert "holds no rows" in capsys.readouterr().err


def test_main_plots_each_recording_under_its_own_name(tmp_path):
    files = [real_flight(tmp_path / "a.parquet"), sim_flight(tmp_path / "b.parquet")]
    main(files)
    for stem in ("a", "b"):
        for kind in ("step_response", "target_error"):
            assert (tmp_path / f"{stem}_{kind}.png").stat().st_size > 0


def test_load_telemetry_csv(tmp_path):
    csv_file = str(tmp_path / "telemetry.csv")
    with open(csv_file, "w") as f:
        f.write("time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,cmd_vx,cmd_vy,cmd_vz\n")
        f.write("0.0,0.5,0.0,1.0,0.0,0.0,0.0,0.1,0.0,0.0\n")
        f.write("0.1,0.4,0.0,1.0,-0.1,0.0,0.0,0.05,0.0,0.0\n")

    data = load_telemetry(csv_file)
    assert "time" in data
    assert np.allclose(data["time"], [0.0, 0.1])
    assert np.allclose(data["pos_x"], [0.5, 0.4])
    assert np.allclose(data["cmd_vx"], [0.1, 0.05])


def test_plot_step_response_creates_figure(tmp_path):
    N = 100
    times = np.linspace(0.0, 1.0, N)
    positions = np.zeros((N, 3))
    positions[:, 0] = np.sin(times)
    positions[:, 1] = 0.1 * times
    positions[:, 2] = 1.0 - 0.5 * np.exp(-times / 0.2)

    velocities = np.zeros((N, 3))
    velocities[:, 0] = np.cos(times)
    velocities[:, 1] = 0.1
    velocities[:, 2] = 2.5 * np.exp(-times / 0.2)

    cmd_velocities = velocities * 0.95
    targets = np.array([0.0, 0.0, 1.0])

    out_file = str(tmp_path / "test_response.png")
    result_path = plot_step_response(
        times=times,
        positions=positions,
        targets=targets,
        velocities=velocities,
        cmd_velocities=cmd_velocities,
        out_path=out_file,
        pos_ylim=(0.0, 2.0),
        vel_ylim=(-1.0, 1.0),
    )
    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 0


def test_plot_target_error_creates_figure(tmp_path):
    N = 100
    times = np.linspace(0.0, 1.0, N)
    positions = np.zeros((N, 3))
    positions[:, 0] = np.sin(times)
    positions[:, 1] = 0.1 * times
    positions[:, 2] = 1.0 - 0.5 * np.exp(-times / 0.2)
    targets = np.array([0.0, 0.0, 1.0])

    out_file = str(tmp_path / "test_target_error.png")
    result_path = plot_target_error(
        times=times,
        positions=positions,
        targets=targets,
        out_path=out_file,
        ylim=(-0.05, 0.6),
    )
    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 0

