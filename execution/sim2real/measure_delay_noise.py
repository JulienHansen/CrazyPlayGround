"""Measure loop delay and estimator noise from logged flights (protocol OL-2 / OL-4).

Two quantities drive the sim-to-real gap and are currently *guessed* in simulation:

  * **loop delay** -- radio + estimator + actuation dead time. Delay plus high gain is
    the classic cause of limit-cycle chatter.
  * **observation noise** -- its magnitude per channel, its bias, and crucially whether
    it is white or temporally correlated.

This script estimates both from data that ``collect_hover_vel.py`` already records,
and prints the exact ``Vel-Hovering-Robust`` parameters to plug into training.

    python execution/sim2real/measure_delay_noise.py execution/sim2real/data/*_vel*
    python execution/sim2real/measure_delay_noise.py <run> --mocap mocap.csv

WHAT IT CAN AND CANNOT DO
  Without ground truth, the command->response lag conflates true dead time with the
  drone's own first-order response, so the script fits BOTH (a first-order-plus-dead-
  time model) and reports them separately. The pure dead time is the number to put in
  ``env.action_latency_steps``.

  Estimator noise cannot be fully separated from real motion using onboard data alone
  -- an EKF error and a genuine wobble look identical. Two honest routes:
    STATIC  : log a stationary drone (``--static``). True velocity is 0 and position is
              constant, so everything logged IS estimator error. Clean but omits
              in-flight vibration.
    MOCAP   : pass ``--mocap`` with a time-aligned ground-truth CSV (columns
              t,x,y,z). Then error = estimate - truth, which is the real OL-4 answer.
  With neither, the script falls back to a high-frequency-residual estimate and labels
  it as an upper bound, since it cannot tell noise from genuine fast motion.
"""

import os
import csv
import glob
import json
import argparse

import numpy as np

DT_NOMINAL = 0.01           # 100 Hz policy step
AXES = ("x", "y", "z")


def load_csv(path):
    if os.path.isdir(path):
        path = os.path.join(path, "flight.csv")
    cols = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for n in r.fieldnames:
            cols[n] = []
        for row in r:
            for n in r.fieldnames:
                try:
                    cols[n].append(float(row[n]))
                except ValueError:
                    cols[n].append(np.nan)
    return {k: np.asarray(v, float) for k, v in cols.items()}, path


# ────────────────────────────── delay ──────────────────────────────

def xcorr_lag(u, y, max_lag):
    """Lag (in samples) maximising correlation between input u and response y."""
    u = u - u.mean()
    y = y - y.mean()
    if u.std() < 1e-9 or y.std() < 1e-9:
        return np.nan, np.nan
    best, best_lag = -np.inf, 0
    for L in range(0, max_lag + 1):
        a, b = u[:len(u) - L] if L else u, y[L:]
        n = min(len(a), len(b))
        if n < 20:
            break
        c = float(np.corrcoef(a[:n], b[:n])[0, 1])
        if c > best:
            best, best_lag = c, L
    return best_lag, best


def fit_fopdt(u, y, dt, max_lag):
    """Fit y_{k+1} = a*y_k + (1-a)*K*u_{k-d}: first order + dead time.

    Returns (dead_time_samples, time_constant_s, gain, r2). Separating the two matters:
    only the dead time belongs in env.action_latency_steps; the time constant is the
    drone's genuine response and is already in the simulator's dynamics.
    """
    best = None
    for d in range(0, max_lag + 1):
        ud = u[:len(u) - d] if d else u
        yd = y[d:]
        n = min(len(ud), len(yd))
        if n < 50:
            break
        ud, yd = ud[:n], yd[:n]
        # regress y_{k+1} on [y_k, u_k]
        Y1, Y0, U0 = yd[1:], yd[:-1], ud[:-1]
        A = np.stack([Y0, U0], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(A, Y1, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a, b = float(coef[0]), float(coef[1])
        if not (0.0 < a < 0.9999):
            continue
        pred = A @ coef
        ss = float(np.sum((Y1 - pred) ** 2))
        tot = float(np.sum((Y1 - Y1.mean()) ** 2)) or 1.0
        r2 = 1.0 - ss / tot
        if best is None or r2 > best[3]:
            tau = -dt / np.log(a)          # first-order time constant
            K = b / (1.0 - a)
            best = (d, tau, K, r2)
    return best if best else (np.nan, np.nan, np.nan, np.nan)


def dominant_freq(x, dt):
    x = x - x.mean()
    if len(x) < 16 or x.std() < 1e-9:
        return np.nan
    X = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), dt)
    return float(f[int(np.argmax(X[1:])) + 1])


def delay_report(d, dt):
    out = {}
    max_lag = int(round(0.15 / dt))        # look up to 150 ms
    for ax, cmd, resp in (("x", "cmd_vx", "vel_x"), ("y", "cmd_vy", "vel_y"), ("z", "cmd_vz", "vel_z")):
        u, y = d[cmd], d[resp]
        lag, corr = xcorr_lag(u, y, max_lag)
        dead, tau, K, r2 = fit_fopdt(u, y, dt, max_lag)
        f0 = dominant_freq(u, dt)
        out[ax] = {"cmd_dominant_hz": f0,
                   "quarter_period_ms": None if np.isnan(f0) or f0 <= 0 else 250.0 / f0,
                   "xcorr_lag_steps": lag, "xcorr_lag_ms": None if np.isnan(lag) else lag * dt * 1e3,
                   "peak_corr": corr, "deadtime_steps": dead,
                   "deadtime_ms": None if np.isnan(dead) else dead * dt * 1e3,
                   "time_constant_ms": None if np.isnan(tau) else tau * 1e3,
                   "gain": K, "fit_r2": r2}
    return out


# ────────────────────────────── noise ──────────────────────────────

def ar1(x):
    x = x - x.mean()
    if len(x) < 3 or x.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def highpass_residual(x, dt, f_cut=8.0):
    """Content above the drone's closed-loop bandwidth -- mostly estimator noise."""
    n = len(x)
    if n < 16:
        return x * 0.0
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(n, dt)
    X[f < f_cut] = 0.0
    return np.fft.irfft(X, n=n)


def noise_report(d, dt, static=False, mocap=None):
    """Per-channel noise. `static` treats the whole log as pure error."""
    chans = {}
    src = "static (true state known constant)" if static else \
          "mocap residual" if mocap is not None else "high-frequency residual (UPPER BOUND)"

    for ax in AXES:
        v, p = d[f"vel_{ax}"], d[f"pos_{ax}"]
        if static:
            ev, ep = v - 0.0, p - p.mean()
        elif mocap is not None:
            ev, ep = v - mocap[f"vel_{ax}"], p - mocap[f"pos_{ax}"]
        else:
            ev, ep = highpass_residual(v, dt), highpass_residual(p, dt)
        chans[f"vel_{ax}"] = {"std": float(np.std(ev)), "bias": float(np.mean(ev)), "ar1": ar1(ev)}
        chans[f"pos_{ax}"] = {"std": float(np.std(ep)), "bias": float(np.mean(ep)), "ar1": ar1(ep)}
    return {"source": src, "channels": chans}


# ────────────────────────────── main ──────────────────────────────

def analyse(path, static=False, mocap=None):
    d, csv_path = load_csv(path)
    t = d["t_mono"]
    dt = float(np.median(np.diff(t))) if len(t) > 2 else DT_NOMINAL
    rate = 1.0 / dt
    jitter_ms = float(np.std(np.diff(t)) * 1e3) if len(t) > 2 else 0.0
    return {
        "run": os.path.basename(os.path.dirname(csv_path) or csv_path),
        "n": int(len(t)),
        "rate_hz": rate,
        "step_jitter_ms": jitter_ms,
        "delay": delay_report(d, dt),
        "noise": noise_report(d, dt, static=static, mocap=mocap),
    }, dt


def main():
    p = argparse.ArgumentParser(description="Measure loop delay and estimator noise from flights.")
    p.add_argument("paths", nargs="+")
    p.add_argument("--open-loop", action="store_true",
                   help="Data came from --excite (a known exogenous command). REQUIRED for the "
                        "delay estimate to mean anything: in closed loop the policy reacts to the "
                        "state it caused, and the apparent lag collapses onto the limit-cycle "
                        "geometry (typically a quarter period) rather than the transport delay.")
    p.add_argument("--static", action="store_true",
                   help="Log is of a stationary drone: treat all logged motion as estimator error.")
    p.add_argument("--mocap", default=None,
                   help="Ground-truth CSV (t,pos_x..,vel_x..), already time-aligned.")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    mocap = None
    if args.mocap:
        mocap, _ = load_csv(args.mocap)

    paths = []
    for x in args.paths:
        paths.extend(glob.glob(x) or [x])

    reports, dts = [], []
    for path in sorted(paths):
        try:
            r, dt = analyse(path, static=args.static, mocap=mocap)
        except (FileNotFoundError, KeyError) as e:
            print(f"[skip] {path}: {e}")
            continue
        reports.append(r); dts.append(dt)

        print(f"\n=== {r['run']} ===")
        print(f"  samples {r['n']}   rate {r['rate_hz']:.1f} Hz   step jitter {r['step_jitter_ms']:.2f} ms")
        print(f"  {'axis':>4} {'xcorr lag':>11} {'dead time':>11} {'tau':>9} {'gain':>7} {'fit R2':>7} {'cmd Hz':>7} {'T/4':>8}")
        for ax, v in r["delay"].items():
            xl = "n/a" if v["xcorr_lag_ms"] is None else f"{v['xcorr_lag_ms']:.0f} ms"
            dm = "n/a" if v["deadtime_ms"] is None else f"{v['deadtime_ms']:.0f} ms"
            tc = "n/a" if v["time_constant_ms"] is None else f"{v['time_constant_ms']:.0f} ms"
            g = "n/a" if v["gain"] is None or np.isnan(v["gain"]) else f"{v['gain']:.2f}"
            r2 = "n/a" if np.isnan(v["fit_r2"]) else f"{v['fit_r2']:.3f}"
            fz = "n/a" if v["cmd_dominant_hz"] is None or np.isnan(v["cmd_dominant_hz"]) else f"{v['cmd_dominant_hz']:.2f}"
            q = "n/a" if v["quarter_period_ms"] is None else f"{v['quarter_period_ms']:.0f} ms"
            print(f"  {ax:>4} {xl:>11} {dm:>11} {tc:>9} {g:>7} {r2:>7} {fz:>7} {q:>8}")
        print(f"  noise source: {r['noise']['source']}")
        print(f"  {'channel':>8} {'std':>9} {'bias':>10} {'AR(1)':>7}")
        for c, v in r["noise"]["channels"].items():
            a1 = "n/a" if np.isnan(v["ar1"]) else f"{v['ar1']:.2f}"
            print(f"  {c:>8} {v['std']:>9.4f} {v['bias']:>10.4f} {a1:>7}")

    if not reports:
        return

    # ---- aggregate into simulator parameters ----
    dt = float(np.median(dts))
    deads = [v["deadtime_steps"] for r in reports for v in r["delay"].values()
             if v["deadtime_steps"] is not None and not np.isnan(v["deadtime_steps"])]
    vel_std = [r["noise"]["channels"][f"vel_{a}"]["std"] for r in reports for a in AXES]
    vel_ar1 = [r["noise"]["channels"][f"vel_{a}"]["ar1"] for r in reports for a in AXES]
    vel_ar1 = [v for v in vel_ar1 if not np.isnan(v)]
    rates = [r["rate_hz"] for r in reports]

    print("\n" + "=" * 66)
    print("SUGGESTED SIMULATOR PARAMETERS")
    print("=" * 66)
    if deads and args.open_loop:
        lo, med, hi = int(np.min(deads)), int(np.median(deads)), int(np.max(deads))
        print(f"  measured dead time: {med} steps ({med*dt*1e3:.0f} ms), range {lo}-{hi}")
        print(f"    env.action_latency_steps={lo}  env.action_latency_max={hi}")
    elif deads:
        med = int(np.median(deads))
        qs = [v["quarter_period_ms"] for r in reports for v in r["delay"].values()
              if v["quarter_period_ms"] is not None]
        print(f"  delay: NOT REPORTED -- this is closed-loop data.")
        print(f"    (the fit returns ~{med*dt*1e3:.0f} ms, but the command's quarter period is"
              f" ~{np.median(qs):.0f} ms:")
        print(f"     the estimate is tracking the limit cycle, not the transport delay.)")
        print(f"    To measure it properly, fly the open-loop sweep:")
        print(f"      collect_hover_vel.py --excite chirp --excite-axis x --duration 25 --tag ol2")
        print(f"      measure_delay_noise.py <run> --open-loop")
    if vel_std:
        print(f"  env.noise_std={np.median(vel_std):.4f}   # median velocity-channel error std")
    if vel_ar1:
        print(f"  env.noise_corr={max(0.0, float(np.median(vel_ar1))):.2f}   # AR(1); ~0 = white, ->1 = correlated")
    print(f"  control rate measured {np.mean(rates):.1f} Hz (sim assumes 100 Hz)")
    if not (args.static or args.mocap):
        print("\n  WARNING: noise came from the high-frequency residual, which cannot separate")
        print("  estimator error from genuine fast motion -- treat it as an UPPER BOUND.")
        print("  Re-run with --static (stationary drone) or --mocap for a real OL-4 answer.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(reports, f, indent=2, default=float)
        print(f"\n[INFO] -> {args.json_out}")


if __name__ == "__main__":
    main()
