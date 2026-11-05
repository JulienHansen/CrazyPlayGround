"""Plotting routines and standalone CLI for sim2real hover evaluation.

Reads one recording format, exec_vel.py's (FlightRecorder columns: control_time, pos_*, vel_*,
target_*, vx/vy/vz). Real flights write it, and so does hover_sim.py, adding `source: sim` to the
Parquet footer; a file without that key is a real flight.

Usage:
    # step-response and target-error figures, one pair per recording (<stem>_step_response.png, ...)
    python execution/sim2real/hovering/plot_hover.py flights/flight.parquet [more.parquet ...]

    # sim and real overlaid: position, velocity and velocity command, one column per axis
    python execution/sim2real/hovering/plot_hover.py flights/flight.parquet flights/sim.parquet --compare
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np


def plot_step_response(
    times: np.ndarray,
    positions: np.ndarray,
    targets: np.ndarray,
    velocities: np.ndarray,
    cmd_velocities: np.ndarray,
    out_path: str,
    axes: Sequence[str] = ("x", "y", "z"),
    pos_ylim: tuple[float, float] | list[float] | None = None,
    vel_ylim: tuple[float, float] | list[float] | None = None,
) -> str:
    """Plot absolute position and velocity tracking without normalization.

    Args:
        times: (N,) timestamps in seconds.
        positions: (N, 3) vehicle positions in meters.
        targets: (3,) or (N, 3) target positions in meters.
        velocities: (N, 3) vehicle velocities in m/s.
        cmd_velocities: (N, 3) commanded velocities in m/s.
        out_path: Destination file path for saving the figure.
        axes: Axis names to plot.
        pos_ylim: Optional (ymin, ymax) limits for position subplots.
        vel_ylim: Optional (ymin, ymax) limits for velocity subplots.

    Returns:
        The path of the saved figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_axes = len(axes)
    fig, ax_arr = plt.subplots(2, n_axes, figsize=(5 * n_axes, 6.5), squeeze=False)

    for i, name in enumerate(axes):
        # ── Row 0: Absolute position tracking [m] ───────────────────────────
        ax_pos = ax_arr[0, i]
        ax_pos.plot(times, positions[:, i], label=f"pos_{name}", color="C0", lw=1.5)
        tgt_val = targets[i] if targets.ndim == 1 else targets[:, i]
        if np.isscalar(tgt_val):
            ax_pos.axhline(tgt_val, color="k", linestyle=":", lw=1.2, label=f"target_{name} ({tgt_val:.2f} m)")
        else:
            ax_pos.plot(times, tgt_val, ":", color="k", lw=1.2, label=f"target_{name}")
        ax_pos.set_title(f"{name.upper()} Position Step Response", fontsize=10)
        ax_pos.set_ylabel(f"pos_{name} [m]")
        ax_pos.grid(True, alpha=0.3)
        if pos_ylim is not None:
            ax_pos.set_ylim(pos_ylim)
        ax_pos.legend(fontsize=8, loc="best")

        # ── Row 1: Absolute velocity tracking [m/s] ─────────────────────────
        ax_vel = ax_arr[1, i]
        ax_vel.plot(times, velocities[:, i], label=f"vel_{name}", color="C0", lw=1.5)
        ax_vel.plot(times, cmd_velocities[:, i], ":", color="C1", lw=1.2, label=f"cmd_v{name}")
        ax_vel.axhline(0, color="k", lw=0.5)
        ax_vel.set_title(f"v{name} Velocity Tracking", fontsize=10)
        ax_vel.set_xlabel("Time [s]")
        ax_vel.set_ylabel(f"vel_{name} [m/s]")
        ax_vel.grid(True, alpha=0.3)
        if vel_ylim is not None:
            ax_vel.set_ylim(vel_ylim)
        ax_vel.legend(fontsize=8, loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved step-response plot to {out_path}")
    return out_path


def plot_target_error(
    times: np.ndarray,
    positions: np.ndarray,
    targets: np.ndarray,
    out_path: str,
    ylim: tuple[float, float] | list[float] | None = None,
) -> str:
    """Plot target tracking error over time on a single clean figure.

    Shows coordinate errors along x, y, z and the Euclidean distance error in meters [m] vs time [s].

    Args:
        times: (N,) timestamps in seconds.
        positions: (N, 3) vehicle positions in meters.
        targets: (3,) or (N, 3) target positions in meters.
        out_path: Destination file path for saving the figure.
        ylim: Optional (ymin, ymax) limits for the y-axis (e.g. [-0.05, 0.55]).

    Returns:
        The path of the saved figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_arr = targets if targets.ndim == 2 else targets[None, :]
    errors = target_arr - positions  # (N, 3), target - current position
    dist_error = np.linalg.norm(errors, axis=1)  # (N,)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}

    for i, name in enumerate(("x", "y", "z")):
        ax.plot(times, errors[:, i], label=f"e_{name} (target_{name} - pos_{name})", color=colors[name], lw=1.3, alpha=0.85)

    ax.plot(times, dist_error, label="||e|| (total distance to target)", color="black", lw=2.0)
    ax.axhline(0, color="gray", linestyle="-", lw=0.8, alpha=0.6)

    # Settling band threshold (5% of initial distance d0 or 2cm min)
    d0 = dist_error[0]
    band = max(0.05 * d0, 0.02)
    ax.axhline(band, color="firebrick", linestyle="--", lw=0.9, alpha=0.7, label=f"5% settling band ({band*1e3:.0f} mm)")

    ax.set_title("Target Tracking Error", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [m]")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved target error plot to {out_path}")
    return out_path


def load_telemetry(file_path: str) -> dict[str, np.ndarray]:
    """Load flight telemetry from a CSV or Parquet file into numpy arrays."""
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Telemetry file not found: {file_path}")

    if file_path.endswith(".csv"):
        import csv

        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            cols: dict[str, list[float]] = {}
            for row in reader:
                for k, v in row.items():
                    cols.setdefault(k, []).append(float(v))
        return {k: np.array(v, dtype=np.float64) for k, v in cols.items()}
    elif file_path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(file_path)
            return {k: np.array(v, dtype=np.float64) for k, v in table.to_pydict().items()}
        except ImportError as e:
            raise ImportError(
                "pyarrow is required to read .parquet files. "
                "Install pyarrow or use a .csv telemetry file."
            ) from e
    else:
        # Attempt CSV first, then Parquet
        try:
            return load_telemetry(file_path + ".csv")
        except FileNotFoundError:
            return load_telemetry(file_path + ".parquet")


@dataclass
class HoverRecording:
    """One hover recording, sim or real, in the units and frames the plots use.

    Attributes:
        name: File name without extension.
        source: "sim" or "real".
        times: (N,) seconds since the first recorded step.
        positions: (N, 3) position, world frame, m.
        velocities: (N, 3) measured linear velocity, world frame, m/s.
        cmd_velocities: (N, 3) commanded velocity sent to the velocity controller, world frame, m/s.
        targets: (N, 3) target position, world frame, m.
    """

    name: str
    source: str
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    cmd_velocities: np.ndarray
    targets: np.ndarray


def crop_recording(rec: HoverRecording, t_max: float) -> HoverRecording:
    """The part of `rec` with times <= t_max seconds."""
    keep = rec.times <= t_max
    return replace(rec, times=rec.times[keep], positions=rec.positions[keep], velocities=rec.velocities[keep],
                   cmd_velocities=rec.cmd_velocities[keep], targets=rec.targets[keep])


def _footer(path: str) -> dict[str, str]:
    """Parquet footer metadata as str -> str; empty for CSV."""
    if not path.endswith(".parquet"):
        return {}
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    return {k.decode(): v.decode() for k, v in metadata.items()}


RECORDING_COLUMNS = ("control_time",) + tuple(
    f"{prefix}{ax}" for prefix in ("pos_", "vel_", "target_", "v") for ax in "xyz"
)
"""The exec_vel.py (FlightRecorder) columns the plots read; hover_sim.py writes the same."""


def load_recording(path: str) -> HoverRecording:
    """Load a hover recording in exec_vel.py's format, from a real flight or from hover_sim.py.

    Args:
        path: Recording (.parquet; .csv carries no footer, so it always loads as real).

    Returns:
        The recording, with times starting at 0 (a real flight's control_time is epoch seconds).

    Raises:
        ValueError: The file holds no rows or lacks a column of RECORDING_COLUMNS.
    """
    data = load_telemetry(path)
    if len(next(iter(data.values()), [])) == 0:
        raise ValueError(f"{path} holds no rows")
    missing = [c for c in RECORDING_COLUMNS if c not in data]
    if missing:
        raise ValueError(f"{path} is not an exec_vel.py / hover_sim.py recording: missing {missing}")

    def stack(prefix: str) -> np.ndarray:
        return np.column_stack([data[f"{prefix}{ax}"] for ax in "xyz"])

    return HoverRecording(
        name=os.path.splitext(os.path.basename(path))[0],
        source=_footer(path).get("source", "real"),  # FlightRecorder writes no source
        times=data["control_time"] - data["control_time"][0],
        positions=stack("pos_"),
        velocities=stack("vel_"),
        cmd_velocities=stack("v"),
        targets=stack("target_"),
    )


def plot_comparison(
    recordings: Sequence[HoverRecording],
    out_path: str,
    axes: Sequence[str] = ("x", "y", "z"),
    pos_ylim: tuple[float, float] | list[float] | None = None,
    vel_ylim: tuple[float, float] | list[float] | None = None,
    cmd_ylim: tuple[float, float] | list[float] | None = None,
) -> str:
    """Overlay recordings (typically one sim, one real) on shared axes.

    One column per axis; rows are position, measured velocity and commanded velocity. Each
    recording keeps one colour across all subplots, solid lines for both, told apart by colour and legend.

    Args:
        recordings: Recordings from `load_recording`.
        out_path: Destination file path for saving the figure.
        axes: Axis names to plot, in column order of the recordings' arrays.
        pos_ylim: Optional (ymin, ymax) for the position row, m.
        vel_ylim: Optional (ymin, ymax) for the velocity row, m/s.
        cmd_ylim: Optional (ymin, ymax) for the command row, m/s.

    Returns:
        The path of the saved figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = recordings[0]
    for rec in recordings[1:]:
        gap = float(np.linalg.norm(rec.positions[0] - ref.positions[0]))
        if gap > 0.05:
            print(f"[WARN] {rec.name} starts {gap * 100:.0f} cm from {ref.name} "
                  f"({np.round(rec.positions[0], 2).tolist()} vs {np.round(ref.positions[0], 2).tolist()} m): "
                  "the responses are not directly comparable")

    # one shared target line if every recording holds the same constant target
    shared_target = all(
        np.ptp(rec.targets, axis=0).max() == 0 and np.allclose(rec.targets[0], ref.targets[0]) for rec in recordings
    )

    rows = (
        ("positions", "{} Position", "pos_{} [m]", pos_ylim),
        ("velocities", "v{} Velocity", "vel_{} [m/s]", vel_ylim),
        ("cmd_velocities", "v{} Command", "cmd_v{} [m/s]", cmd_ylim),
    )
    n_axes = len(axes)
    fig, ax_arr = plt.subplots(len(rows), n_axes, figsize=(5 * n_axes, 10), sharex=True, squeeze=False)

    for r, (attr, title, ylabel, ylim) in enumerate(rows):
        for i, name in enumerate(axes):
            ax = ax_arr[r, i]
            first = r == 0 and i == 0  # labels on one subplot only: they feed the single figure legend
            for j, rec in enumerate(recordings):
                ax.plot(rec.times, getattr(rec, attr)[:, i], "-",
                        color=f"C{j}", lw=1.3, gid=f"{rec.source}:{rec.name}",
                        label=f"{rec.source}: {rec.name}" if first else None)
                if r == 0 and not shared_target:
                    ax.plot(rec.times, rec.targets[:, i], ":", color=f"C{j}", lw=1.2, gid="target",
                            label=f"target ({rec.source}: {rec.name})" if first else None)
            if r == 0 and shared_target:
                ax.axhline(ref.targets[0, i], color="k", linestyle=":", lw=1.2, gid="target",
                           label="target" if first else None)
            if r > 0:
                ax.axhline(0, color="k", lw=0.5)
            ax.set_title(title.format(name.upper() if r == 0 else name), fontsize=10)
            ax.set_ylabel(ylabel.format(name))
            ax.grid(True, alpha=0.3)
            if ylim is not None:
                ax.set_ylim(ylim)
            if r == len(rows) - 1:
                ax.set_xlabel("Time [s]")

    handles, labels = ax_arr[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 4), fontsize=9, framealpha=0.9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved sim/real comparison plot to {out_path}")
    return out_path


def build_plot_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone plotting from recorded telemetry."""
    parser = argparse.ArgumentParser(
        description="Plot recorded hover flights, sim (hover_sim.py) or real (exec_vel.py), CSV or Parquet: "
        "step-response and target-error figures per recording, or one sim/real comparison with --compare."
    )
    parser.add_argument("files", nargs="+", type=str, help="Telemetry .parquet or .csv file(s).")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Overlay all recordings (at least two) in one figure: position, velocity and velocity command.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save figures (defaults to the directory of each telemetry file; "
        "with --compare, of the first one).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="With --compare: path of the figure (default: <outdir>/sim_real_comparison.png).",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Custom y-axis limits for target error plot (e.g. --ylim -0.05 0.55).",
    )
    parser.add_argument(
        "--pos-ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Custom y-axis limits for position subplots.",
    )
    parser.add_argument(
        "--vel-ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Custom y-axis limits for velocity subplots.",
    )
    parser.add_argument(
        "--cmd-ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="With --compare: custom y-axis limits for the velocity-command row.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Plot only the first SECONDS of each recording (e.g. --t-max 10).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Standalone CLI entry point."""
    parser = build_plot_parser()
    args = parser.parse_args(argv)
    if args.compare and len(args.files) < 2:
        parser.error("--compare needs at least two recordings")
    if args.out and not args.compare:
        parser.error("--out applies to --compare only")

    try:
        recordings = [load_recording(path) for path in args.files]
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))
    if args.t_max is not None:
        if args.t_max <= 0:
            parser.error("--t-max must be positive")
        recordings = [crop_recording(rec, args.t_max) for rec in recordings]

    if args.compare:
        outdir = args.outdir or os.path.dirname(os.path.abspath(args.files[0]))
        plot_comparison(
            recordings,
            out_path=args.out or os.path.join(outdir, "sim_real_comparison.png"),
            pos_ylim=args.pos_ylim,
            vel_ylim=args.vel_ylim,
            cmd_ylim=args.cmd_ylim,
        )
        return

    for path, rec in zip(args.files, recordings):
        outdir = args.outdir or os.path.dirname(os.path.abspath(path))
        plot_step_response(
            times=rec.times,
            positions=rec.positions,
            targets=rec.targets,
            velocities=rec.velocities,
            cmd_velocities=rec.cmd_velocities,
            out_path=os.path.join(outdir, f"{rec.name}_step_response.png"),
            pos_ylim=args.pos_ylim,
            vel_ylim=args.vel_ylim,
        )
        plot_target_error(
            times=rec.times,
            positions=rec.positions,
            targets=rec.targets,
            out_path=os.path.join(outdir, f"{rec.name}_target_error.png"),
            ylim=args.ylim,
        )


if __name__ == "__main__":
    main()

