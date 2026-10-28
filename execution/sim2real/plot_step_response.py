"""Mean step response of excite_hover.py flights: interquartile mean and 10-90 % band over the steps.

    python execution/sim2real/plot_step_response.py flight.parquet [more.parquet ...] [--out fig.png]

One panel per recording. On each excited axis, the PUSH steps (from rest) are windowed and
normalized as in measure_delay_noise.py (0 at the old command, 1 at the new one), split by sign:
older recordings alternated ±amp, new ones push one way only.
"""

import argparse
import os

import numpy as np

import measure_delay_noise as mdn

BAND = (10, 90)  # percentiles of the shaded band


def iqm(windows):
    """Interquartile mean over the steps (axis 0): the mean of the central 50 % at each time step."""
    s = np.sort(windows, axis=0)
    q = len(s) // 4
    return s[q:len(s) - q].mean(axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot the mean step response (IQM, 10-90 %% band) "
                                                 "of excite_hover.py recordings.")
    parser.add_argument("paths", nargs="+", help="flight.parquet files from excite_hover.py")
    parser.add_argument("--out", type=str, default=None, help="Save the figure here instead of showing it.")
    args = parser.parse_args(argv)

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(args.paths), figsize=(5.5 * len(args.paths), 4.2), sharey=True, squeeze=False)
    for ax_plot, path in zip(axes[0], args.paths):
        cols = mdn.load(path)
        dt = float(np.median(np.diff(cols["control_time"])))
        titles = []
        for ax in mdn.AXES:
            exc = cols[f"exc_v{ax}"]
            step_edges, windows = mdn.step_windows(exc, cols[f"vel_{ax}"])
            if not step_edges:
                continue
            t_ms = np.arange(windows.shape[1]) * dt * 1e3
            positive = np.array([exc[t] > 0 for t in step_edges])
            for name, mask, style in (("push +", positive, "-"), ("push -", ~positive, "--")):
                if not mask.any():
                    continue
                line, = ax_plot.plot(t_ms, iqm(windows[mask]), style, label=f"v{ax} {name} (n={mask.sum()})")
                lo, hi = np.percentile(windows[mask], BAND, axis=0)
                ax_plot.fill_between(t_ms, lo, hi, color=line.get_color(), alpha=0.2, lw=0)
            titles.append(f"v{ax} amp {np.abs(exc).max():g} m/s")
        ax_plot.axhline(0, color="k", lw=0.5)
        ax_plot.axhline(1, color="k", lw=0.5)
        ax_plot.set_title(f"{os.path.basename(path)}\n{', '.join(titles) or 'no excited axis'}", fontsize=9)
        ax_plot.set_xlabel("ms after the step command")
        ax_plot.grid(alpha=0.3)
        ax_plot.legend(fontsize=8)
    axes[0][0].set_ylabel(f"normalized velocity (IQM, {BAND[0]}-{BAND[1]} % band)")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
