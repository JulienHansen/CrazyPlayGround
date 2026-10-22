"""Loop delay and sensor noise of hover flights, as Vel-Hovering config values.

    python execution/sim2real/excite_hover.py --record-path flight.parquet --duration 30
    python execution/sim2real/measure_delay_noise.py flight.parquet [more.parquet ...]

Input: any Parquet recording with one row per 10 ms control step holding `step`,
`control_time`, the sent world velocity command `vx, vy, vz`, the measured velocity
`vel_x, vel_y, vel_z`, and optionally the excitation added to the command `exc_vx, exc_vy,
exc_vz` (zeros if absent). excite_hover.py writes exactly this; exec_vel.py too, minus exc_v*.

Per excited axis, the sent command -> measured velocity is fitted as a first-order lag plus
dead time by output error: fitting the model's simulated, noise-free response, unlike
regressing on the noisy measured velocity, is not biased by the sensor noise or by the hold
loop's feedback. The fit's residual is what the command does not explain, the sensor noise,
and its AR(1) gives noise_std and noise_corr as vel_hovering defines them (stationary std,
lag-1 correlation per policy step).
"""

import argparse
import sys

import numpy as np
import pyarrow.parquet as pq

MAX_DELAY = 20                       # policy steps searched by the fit and the cross-correlation
TAUS = np.geomspace(0.02, 1.0, 30)   # s, first-order time constants tried by the fit
AXES = "xyz"


def load(path):
    """Columns as float arrays. Exits on a file the fits cannot use."""
    table = pq.read_table(path)
    cols = {name: np.asarray(table.column(name).to_numpy(), dtype=float) for name in table.column_names}
    if len(cols["step"]) < 2 or np.any(np.diff(cols["step"]) != 1):
        sys.exit(f"{path}: steps are not consecutive; record every step")  # rho is per step
    for ax in AXES:
        if not (np.isfinite(cols[f"v{ax}"]).all() and np.isfinite(cols[f"vel_{ax}"]).all()):
            sys.exit(f"{path}: non-finite v{ax} or vel_{ax}")
        cols.setdefault(f"exc_v{ax}", np.zeros(len(cols["step"])))
    return cols


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


def ar1(x):
    """(stationary std, lag-1 correlation) of x."""
    x = x - x.mean()
    return float(x.std()), float(x[1:] @ x[:-1] / (x[:-1] @ x[:-1]))


def dominant_freq(x, dt):
    """Strongest oscillation of x [Hz], ignoring drift below 0.2 Hz."""
    x = x - x.mean()
    if len(x) < 16 or x.std() < 1e-9:
        return np.nan
    power = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), dt)
    power[f < 0.2] = 0
    return float(f[np.argmax(power)])


def fit_fopdt(u, y, dt, max_d=MAX_DELAY, rho=None):
    """(dead time d [steps], tau [s], gain K, residual) of y[t] = K * lag_tau(u)[t - 1 - d].

    d = 0 is the sim's action_latency_steps = 0: row t holds the state read before command t
    is sent, so command t first shows in row t + 1. Correlated noise makes one more step of
    delay with a faster lag fit nearly as well, so a second pass minimises the residual
    whitened with its own AR(1) correlation `rho`.
    """
    if rho is None:
        rho = float(np.clip(ar1(fit_fopdt(u, y, dt, max_d, rho=0.0)[3])[1], 0.0, 0.99))
    u, y = u - u.mean(), y - y.mean()
    yw = y[1:] - rho * y[:-1]
    best = (np.inf,)
    for tau in TAUS:
        a = np.exp(-dt / tau)
        f = np.zeros_like(u)
        for i in range(1, len(u)):
            f[i] = a * f[i - 1] + (1 - a) * u[i - 1]
        for d in range(max_d + 1):
            g = np.concatenate([np.zeros(d), f[:len(f) - d]])
            gw = g[1:] - rho * g[:-1]
            k = gw @ yw / (gw @ gw)
            rw = yw - k * gw
            if rw @ rw < best[0]:
                best = (rw @ rw, d, float(tau), float(k), y - k * g)
    return best[1:]


def analyse(path):
    """Per axis: the delay and noise fit if it was excited, else its dominant oscillation."""
    cols = load(path)
    dt = float(np.median(np.diff(cols["control_time"])))
    axes = {}
    for ax in AXES:
        u, y, exc = cols[f"v{ax}"], cols[f"vel_{ax}"], cols[f"exc_v{ax}"]
        if np.any(exc != 0):
            start = int(np.argmax(exc != 0))
            d, tau, k, resid = fit_fopdt(u[start:], y[start:], dt)
            std, rho = ar1(resid)
            axes[ax] = dict(xcorr_ms=xcorr_lag(exc[start:], y[start:], MAX_DELAY)[0] * dt * 1e3,
                            d=d, tau_ms=tau * 1e3, gain=k, noise_std=std, noise_corr=rho)
        else:  # the command only reacts to noise here: delay and noise are not identifiable
            axes[ax] = dict(osc_hz=dominant_freq(y, dt))
    return axes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estimate loop delay and sensor noise from flight "
                                                 "recordings and print the matching Vel-Hovering values.")
    parser.add_argument("paths", nargs="+", help="flight.parquet files, e.g. from excite_hover.py")
    args = parser.parse_args(argv)

    fits = []
    print(f"{'run':<32} ax {'xcorr ms':>8} {'d':>3} {'tau ms':>7} {'gain':>5} {'noise':>7} {'rho':>5} {'osc Hz':>6}")
    for path in args.paths:
        for ax, r in analyse(path).items():
            if "d" in r:
                fits.append(r)
            g = lambda key: r.get(key, np.nan)  # noqa: E731
            print(f"{path[-32:]:<32} {ax:>2} {g('xcorr_ms'):8.0f} {g('d'):3.0f} {g('tau_ms'):7.0f} "
                  f"{g('gain'):5.2f} {g('noise_std'):7.4f} {g('noise_corr'):5.2f} {g('osc_hz'):6.2f}")

    if not fits:
        print("delay and noise: NOT REPORTED, no excited axis. Fly for example\n"
              "  execution/sim2real/excite_hover.py --record-path flight.parquet --exc-axis x --duration 30")
        return
    ds = [r["d"] for r in fits]
    # shortcut: d also absorbs the attitude loop's lag (the drone tilts before it accelerates), so it
    # overstates the delay the sim is missing; subtract the sim's own d once WP7 records sim flights.
    print(f"env.action_latency_steps={min(ds)} env.action_latency_max_steps={max(ds) if max(ds) > min(ds) else 0} "
          f"env.add_noise=True env.noise_std={np.median([r['noise_std'] for r in fits]):.4f} "
          f"env.noise_corr={np.clip(np.median([r['noise_corr'] for r in fits]), 0, 0.99):.2f}")
    print("Upper bounds: d includes the attitude lag, the noise any motion the model does not explain.")


if __name__ == "__main__":
    main()
