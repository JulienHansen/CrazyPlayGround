"""Mean step response of excite_hover.py and excite_hover_sim.py flights, real and sim on the same axes.

    python execution/sim2real/plot_step_response.py flight.parquet [sim_flight.parquet ...] [--out fig.png]

One column per excited axis, every recording overlaid in its own color, labelled real or sim from
the Parquet footer. Top: the PUSH steps (from rest) windowed and normalized as in
measure_delay_noise.py (0 at the old command, 1 at the new one). Bottom: the flight in m/s over one
cycle from each push, the setpoint dotted. Both show the interquartile mean and the 10-90 % band
over the steps, split by sign: older recordings alternated ±amp, new ones push one way only.
"""

import argparse
import os
import sys

import numpy as np
import pyarrow.parquet as pq

import measure_delay_noise as mdn

BAND = (10, 90)  # percentiles of the shaded band


def iqm(windows):
    """Interquartile mean over the steps (axis 0): the mean of the central 50 % at each time step."""
    s = np.sort(windows, axis=0)
    q = len(s) // 4
    return s[q:len(s) - q].mean(axis=0)


def source(path):
    """'sim' for excite_hover_sim.py recordings, 'real' otherwise: excite_hover.py writes no source."""
    metadata = pq.read_schema(path).metadata or {}
    return metadata.get(b"source", b"real").decode()


def plot_iqm(ax_plot, t, windows, color, style, label):
    ax_plot.plot(t, iqm(windows), style, color=color, label=label)
    lo, hi = np.percentile(windows, BAND, axis=0)
    ax_plot.fill_between(t, lo, hi, color=color, alpha=0.2, lw=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Overlay the mean step response (IQM, 10-90 %% band) "
                                                 "of real and simulated excitation recordings.")
    parser.add_argument("paths", nargs="+", help="flight.parquet files from excite_hover.py or excite_hover_sim.py")
    parser.add_argument("--out", type=str, default=None, help="Save the figure here instead of showing it.")
    args = parser.parse_args(argv)

    recordings = [(path, source(path), mdn.load(path)) for path in args.paths]
    excited = [ax for ax in mdn.AXES if any(cols[f"exc_v{ax}"].any() for _, _, cols in recordings)]
    if not excited:
        sys.exit("no excited axis in the recordings")

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(excited), figsize=(7 * len(excited), 8), squeeze=False)
    for (response, flight), ax in zip(axes.T, excited):
        for i, (path, src, cols) in enumerate(recordings):
            exc, vel = cols[f"exc_v{ax}"], cols[f"vel_{ax}"]
            step_edges, windows = mdn.step_windows(exc, vel)
            if not step_edges:
                continue
            dt = float(np.median(np.diff(cols["control_time"])))
            cycle = int(np.min(np.diff(step_edges))) if len(step_edges) > 1 else 0
            positive = exc[step_edges] > 0
            for sign, mask, style in (("+", positive, "-"), ("-", ~positive, "--")):
                if not mask.any():
                    continue
                label = f"{src} {os.path.basename(path)}: push {sign}{np.abs(exc).max():g} m/s (n={mask.sum()})"
                plot_iqm(response, np.arange(windows.shape[1]) * dt * 1e3, windows[mask], f"C{i}", style, label)
                cycle_edges = [t for t, m in zip(step_edges, mask) if m and t + cycle <= len(vel)]
                if cycle and cycle_edges:
                    t_s = np.arange(cycle) * dt
                    plot_iqm(flight, t_s, np.array([vel[t:t + cycle] for t in cycle_edges]), f"C{i}", style, label)
                    setpoint = np.array([cols[f"v{ax}"][t:t + cycle] for t in cycle_edges])
                    flight.plot(t_s, iqm(setpoint), ":", color=f"C{i}", lw=1)
        for ax_plot in (response, flight):
            ax_plot.axhline(0, color="k", lw=0.5)
            ax_plot.grid(alpha=0.3)
        response.axhline(1, color="k", lw=0.5)
        response.set_title(f"v{ax} step response", fontsize=10)
        response.set_xlabel("ms after the step command")
        response.set_ylabel(f"normalized velocity (IQM, {BAND[0]}-{BAND[1]} % band)")
        response.legend(fontsize=8)
        flight.set_title(f"v{ax} over one cycle from the push (dotted: setpoint)", fontsize=10)
        flight.set_xlabel("s after the step command")
        flight.set_ylabel(f"vel_{ax} [m/s]")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
