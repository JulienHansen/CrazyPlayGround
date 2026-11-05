"""Plotting routines and standalone CLI for sim2real hover evaluation."""

from __future__ import annotations

import argparse
import os
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


def build_plot_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone plotting from recorded telemetry."""
    parser = argparse.ArgumentParser(
        description="Generate step-response and target-error plots from recorded hover telemetry (CSV or Parquet)."
    )
    parser.add_argument("file", type=str, help="Path to telemetry .csv or .parquet file.")
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Target position in meters (default: 0.0 0.0 1.0).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save figures (defaults to the directory of the telemetry file).",
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
    return parser


def main() -> None:
    """Standalone CLI entry point."""
    parser = build_plot_parser()
    args = parser.parse_args()

    data = load_telemetry(args.file)

    times = data["time"]
    positions = np.column_stack([data["pos_x"], data["pos_y"], data["pos_z"]])
    velocities = np.column_stack([data["vel_x"], data["vel_y"], data["vel_z"]])
    cmd_velocities = np.column_stack([data["cmd_vx"], data["cmd_vy"], data["cmd_vz"]])
    targets = np.array(args.target)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.file))
    step_resp_path = os.path.join(outdir, "step_response.png")
    target_err_path = os.path.join(outdir, "target_error.png")

    plot_step_response(
        times=times,
        positions=positions,
        targets=targets,
        velocities=velocities,
        cmd_velocities=cmd_velocities,
        out_path=step_resp_path,
        pos_ylim=args.pos_ylim,
        vel_ylim=args.vel_ylim,
    )
    plot_target_error(
        times=times,
        positions=positions,
        targets=targets,
        out_path=target_err_path,
        ylim=args.ylim,
    )


if __name__ == "__main__":
    main()

