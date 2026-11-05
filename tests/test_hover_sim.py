"""Tests for execution/sim2real/hovering/hover_sim.py."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "execution", "sim2real", "hovering"))
from hover_sim import build_parser


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["--checkpoint", "dummy_checkpoint.pt"])
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
    assert args.record_path is None


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
