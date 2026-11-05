"""Tests for execution/sim2real/hovering/plot_hover.py."""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "execution", "sim2real", "hovering"))
from plot_hover import build_plot_parser, load_telemetry, plot_step_response, plot_target_error


def test_build_plot_parser():
    parser = build_plot_parser()
    args = parser.parse_args(["flight.parquet", "--ylim", "-0.1", "0.6", "--target", "0.1", "0.2", "1.0"])
    assert args.file == "flight.parquet"
    assert args.ylim == [-0.1, 0.6]
    assert args.target == [0.1, 0.2, 1.0]


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

