"""Tests for execution/sim2real/hovering/hover_sim.py."""

import os
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "execution", "sim2real", "hovering"))
from hover_sim import build_parser, hover_row, write_recording  # noqa: E402
from plot_hover import RECORDING_COLUMNS, load_recording  # noqa: E402

# what exec_vel.py's FlightRecorder writes and plot_hover.load_recording reads
FLIGHT_RECORDER_COLUMNS = (
    ["step", "control_time", "obs_age_s"]
    + [f"{p}{ax}" for p in ("pos_", "vel_") for ax in "xyz"]
    + [f"quat_{c}" for c in "wxyz"]
    + [f"target_{ax}" for ax in "xyz"]
    + [f"action_v{ax}" for ax in "xyz"]
    + [f"v{ax}" for ax in "xyz"]
)


def _rows(n=20, dt=0.01):
    return [
        hover_row(k, k * dt, [0.1 * k, 0.0, 0.5], [1.0, -0.5, 0.2], [1.0, 0.0, 0.0, 0.0],
                  [0.5, 0.0, 1.0], np.array([0.4, -0.2, 0.1]), np.array([0.4, -0.2, 0.1]))  # max_velocity 1
        for k in range(n)
    ]


def test_hover_row_uses_exec_vel_column_names():
    assert list(_rows(1)[0]) == FLIGHT_RECORDER_COLUMNS
    assert set(RECORDING_COLUMNS) <= set(FLIGHT_RECORDER_COLUMNS)   # everything plot_hover reads


def test_hover_row_rejects_a_vector_of_the_wrong_length():
    with pytest.raises(ValueError):
        hover_row(0, 0.0, [0.0, 0.0], [0.0] * 3, [1.0, 0.0, 0.0, 0.0], [0.0] * 3, [0.0] * 3, [0.0] * 3)


def test_recording_round_trips_through_plot_hover(tmp_path):
    """The contract between the sim writer and the reader: a sim file loads as a sim recording."""
    path = str(tmp_path / "out" / "sim.parquet")
    write_recording(path, _rows(), {"source": "sim", "target": [0.5, 0.0, 1.0], "ended_at_step": None})

    schema = pq.read_schema(path)
    assert schema.field("step").type == "int64" and schema.field("vx").type == "double"
    assert schema.metadata[b"source"] == b"sim" and schema.metadata[b"ended_at_step"] == b"None"
    rec = load_recording(path)
    assert rec.source == "sim" and rec.name == "sim"
    assert rec.times == pytest.approx(np.arange(20) * 0.01)
    assert rec.positions[:, 0] == pytest.approx(0.1 * np.arange(20))
    assert rec.velocities[0] == pytest.approx([1.0, -0.5, 0.2])
    assert rec.cmd_velocities[0] == pytest.approx([0.4, -0.2, 0.1])
    assert rec.targets[0] == pytest.approx([0.5, 0.0, 1.0])


def test_build_parser_requires_a_record_path(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--checkpoint", "dummy_checkpoint.pt"])
    assert "--record-path" in capsys.readouterr().err


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["--checkpoint", "dummy_checkpoint.pt", "--record-path", "sim.parquet"])
    assert args.checkpoint == "dummy_checkpoint.pt"
    assert args.task == "Vel-Hovering"
    assert args.start_pos == [0.5, 0.0, 1.0]
    assert args.target == [0.0, 0.0, 1.0]
    assert args.time == 10.0
    assert args.video is True
    assert args.ylim is None
    assert args.error_ylim is None
    assert args.pos_ylim is None
    assert args.vel_ylim is None
    assert args.outdir == "outputs/sim2real"
    assert args.plot_path is None
    assert args.error_plot_path is None
    assert args.record_path == "sim.parquet"


def test_build_parser_custom_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--start-pos",
            "1.0",
            "-0.5",
            "0.8",
            "--target",
            "0.2",
            "0.1",
            "1.2",
            "--time",
            "5.5",
            "--no-video",
            "--ylim",
            "-0.05",
            "0.55",
            "--pos-ylim",
            "0.0",
            "1.5",
            "--vel-ylim",
            "-1.0",
            "1.0",
            "--outdir",
            "custom_out",
            "--plot-path",
            "custom_plot.png",
            "--record-path",
            "flight.parquet",
        ]
    )
    assert args.checkpoint == "model.pt"
    assert args.start_pos == [1.0, -0.5, 0.8]
    assert args.target == [0.2, 0.1, 1.2]
    assert args.time == 5.5
    assert args.video is False
    assert args.ylim == [-0.05, 0.55]
    assert args.pos_ylim == [0.0, 1.5]
    assert args.vel_ylim == [-1.0, 1.0]
    assert args.outdir == "custom_out"
    assert args.plot_path == "custom_plot.png"
    assert args.record_path == "flight.parquet"
