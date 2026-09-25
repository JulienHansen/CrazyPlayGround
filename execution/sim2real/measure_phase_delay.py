"""Estimate the unmodelled loop delay from the command-to-response phase (OL-2).

The open-loop chirp flights aborted before takeoff, so the delay has never been
measured directly. This recovers it from ordinary hover flights instead, using the
simulator as the reference plant.

The simulator runs the same cascade PID on the same nominal airframe with no delay
configured, so the phase from commanded to executed velocity there is the plant
alone. On hardware the same phase also contains the radio, the estimator and the
loop scheduling. The difference between the two is the unmodelled transport delay:

    phi_hw(f) - phi_sim(f) = -2 pi f tau

so tau is the slope of the phase difference against frequency. Fitting a slope
rather than reading one frequency is what separates a transport delay, whose phase
grows linearly with frequency, from a first-order lag, whose phase saturates.

Caveats, stated because they bound the result:
  * These are closed-loop flights. The command is a function of the state it caused,
    so the estimate is not a textbook open-loop identification. Differencing against
    the simulator removes the part of that feedback both sides share, not all of it.
  * Only frequencies where both sides are coherent carry information; the fit is
    restricted to them and the band is reported.
  * The real loop samples at ~85 Hz against the simulator's 100 Hz, so both sides
    are resampled onto a common uniform grid first.

Usage:
    python execution/sim2real/measure_phase_delay.py --plot phase.pdf
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_hover import load_flight  # noqa: E402

FS = 100.0           # common resampling rate [Hz]
NPERSEG = 256        # ~2.5 s windows
COH_MIN = 0.5        # only fit where both sides are coherent
AXES = ("x", "y", "z")


def uniform(d, axis, use_excitation=False):
    """Resample command and executed velocity onto a uniform grid.

    The hardware clock jitters by 1-2 ms per step, which smears a phase estimate if
    the samples are treated as evenly spaced.
    """
    t = d["t_mono"] - d["t_mono"][0]
    grid = np.arange(0.0, t[-1], 1.0 / FS)
    key = f"exc_v{axis}" if use_excitation else f"cmd_v{axis}"
    if key not in d:
        raise KeyError(f"{key} absent; this flight predates the excitation logging")
    cmd = np.interp(grid, t, d[key])
    vel = np.interp(grid, t, d[f"vel_{axis}"])
    return grid, signal.detrend(cmd), signal.detrend(vel)


def phase_of(runs, axis, use_excitation=False):
    """Coherence-weighted mean phase and coherence across runs, per frequency."""
    P, C, f_ref = [], [], None
    for d in runs:
        _, cmd, vel = uniform(d, axis, use_excitation)
        if len(cmd) < NPERSEG * 2 or np.std(cmd) < 1e-6:
            continue
        f, Pxy = signal.csd(cmd, vel, fs=FS, nperseg=NPERSEG)
        _, coh = signal.coherence(cmd, vel, fs=FS, nperseg=NPERSEG)
        P.append(np.angle(Pxy))
        C.append(coh)
        f_ref = f
    if not P:
        return None, None, None
    P, C = np.array(P), np.array(C)
    w = C / np.maximum(C.sum(axis=0, keepdims=True), 1e-9)
    # average on the unit circle, not on the wrapped angle
    mean_phase = np.angle((w * np.exp(1j * P)).sum(axis=0))
    return f_ref, mean_phase, C.mean(axis=0)


def fit_delay(f, dphi, coh, fmin=0.5, fmax=12.0):
    """Slope of the unwrapped phase difference -> transport delay in seconds."""
    band = (f >= fmin) & (f <= fmax) & (coh >= COH_MIN)
    if band.sum() < 4:
        return None, None, band
    x = f[band]
    y = np.unwrap(dphi[band])
    # phase = -2 pi f tau + c  ->  tau = -slope / (2 pi)
    slope, intercept = np.polyfit(x, y, 1, w=coh[band])
    tau = -slope / (2.0 * np.pi)
    resid = y - (slope * x + intercept)
    r2 = 1.0 - resid.var() / max(y.var(), 1e-12)
    return tau, r2, band


def main():
    p = argparse.ArgumentParser(description="Estimate the unmodelled loop delay from phase.")
    p.add_argument("--real", default=os.path.join(HERE, "data", "2026-09-24*"))
    p.add_argument("--sim", default=os.path.join(HERE, "data_sim_gap", "*velsim*"))
    p.add_argument("--use-excitation", action="store_true",
                   help="Correlate the logged exogenous excitation against the response "
                        "instead of the full command. Valid identification, but it needs "
                        "flights recorded with --excite.")
    p.add_argument("--plot", default=None)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    manifest = json.load(open(os.path.join(HERE, "flight_checkpoints", "manifest.json")))
    by_sha = {e["sha256_16"]: e["id"] for e in manifest}

    def collect(pattern):
        out = defaultdict(list)
        for run in sorted(glob.glob(pattern)):
            if not os.path.isdir(run) or "ABORTED" in run or "chirp" in run:
                continue
            meta = os.path.join(run, "metadata.json")
            if not os.path.exists(meta):
                continue
            pid = by_sha.get(json.load(open(meta)).get("checkpoint_sha256", "")[:16])
            if pid:
                out[pid].append(load_flight(run)["data"])
        return out

    real, sim = collect(args.real), collect(args.sim)
    shared = sorted(set(real) & set(sim))

    print(f"Unmodelled loop delay from command-to-response phase\n"
          f"  fit band {0.5}-{12.0} Hz where coherence >= {COH_MIN}\n")
    print(f"{'policy':7s} {'axis':5s} {'delay [ms]':>11s} {'steps @85Hz':>12s} {'R^2':>6s} {'band pts':>9s}")
    results, curves = {}, {}
    for pid in shared:
        for axis in AXES:
            f, ph_r, coh_r = phase_of(real[pid], axis, args.use_excitation)
            _, ph_s, coh_s = phase_of(sim[pid], axis, args.use_excitation)
            if ph_r is None or ph_s is None:
                continue
            dphi = np.angle(np.exp(1j * (ph_r - ph_s)))
            coh = np.minimum(coh_r, coh_s)
            tau, r2, band = fit_delay(f, dphi, coh)
            curves[(pid, axis)] = (f, dphi, coh, band)
            if tau is None:
                print(f"{pid:7s} {axis:5s} {'-':>11s} {'-':>12s} {'-':>6s} {band.sum():9d}")
                continue
            results.setdefault(pid, {})[axis] = {"delay_s": tau, "r2": r2, "n": int(band.sum())}
            print(f"{pid:7s} {axis:5s} {tau * 1e3:11.1f} {tau * 84.4:12.2f} {r2:6.2f} {band.sum():9d}")

    taus = [v["delay_s"] for d in results.values() for v in d.values() if v["r2"] > 0.5]
    if taus:
        taus = np.array(taus)
        pooled = taus.mean()
        # A transport delay is positive and the same for every axis and policy: the
        # radio, the estimator and the scheduler do not know which axis they carry.
        # Negative or sign-inconsistent fits therefore mean the estimate is not
        # measuring a delay, and reporting a mean of them would invent a number.
        if (taus < 0).any() or taus.std() > abs(pooled):
            print(f"\nINCONCLUSIVE over {len(taus)} fits: "
                  f"{pooled * 1e3:.1f} +/- {taus.std() * 1e3:.1f} ms, "
                  f"{int((taus < 0).sum())} of them negative.")
            print("A transport delay cannot be negative, so the feedback path is\n"
                  "dominating the phase: in closed loop the command is generated from\n"
                  "the velocity it caused, and near the limit cycle that reverse path\n"
                  "swamps the plant. Differencing against the simulator removes only\n"
                  "the part of the feedback both sides share.")
            print("\nTo measure the delay, fly with an exogenous excitation:\n"
                  "  collect_hover_vel.py --excite chirp --excite-axis x ...\n"
                  "and rerun this with --use-excitation. The excitation is independent\n"
                  "of the state, which is what makes the identification valid.")
        else:
            print(f"\nPooled over {len(taus)} policy-axis fits with R^2 > 0.5: "
                  f"{pooled * 1e3:.1f} +/- {taus.std() * 1e3:.1f} ms "
                  f"({pooled * 84.4:.1f} control steps at 84.4 Hz)")

    if args.json_out:
        json.dump(results, open(args.json_out, "w"), indent=2, default=float)
        print(f"[INFO] json -> {args.json_out}")
    if args.plot:
        plot(curves, results, args.plot)


def plot(curves, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.4, "lines.linewidth": 0.9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.labelsize": 7, "axes.titlesize": 7.5, "figure.dpi": 200})
    pols = sorted({k[0] for k in curves})
    fig, axes = plt.subplots(2, len(pols), figsize=(6.9, 3.2), sharex=True,
                             sharey="row", squeeze=False)
    for j, pid in enumerate(pols):
        for axis, colour in zip(AXES, ("#c1442a", "#2e7d32", "#1f4e79")):
            if (pid, axis) not in curves:
                continue
            f, dphi, coh, band = curves[(pid, axis)]
            axes[0, j].plot(f, np.degrees(np.unwrap(dphi)), color=colour, label=f"${axis}$")
            axes[1, j].plot(f, coh, color=colour)
        r = results.get(pid, {})
        good = [v["delay_s"] for v in r.values() if v["r2"] > 0.5]
        title = pid + (f"\n{np.mean(good) * 1e3:.0f} ms" if good else "")
        axes[0, j].set_title(title)
        axes[0, j].set_xlim(0, 15)
        axes[1, j].axhline(COH_MIN, color="0.5", lw=0.6, ls=":")
        axes[1, j].set_xlabel("frequency [Hz]")
    axes[0, 0].set_ylabel("phase(hardware) $-$ phase(sim) [deg]")
    axes[1, 0].set_ylabel("coherence")
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.suptitle("Extra phase on hardware over the simulator. A transport delay is a straight line.",
                 fontsize=7.5, y=1.02)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    print(f"[INFO] plot -> {path}")


if __name__ == "__main__":
    main()
