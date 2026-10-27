"""Step-response delay, dead time, time constant and sensor noise of hover flights.

    python execution/sim2real/excite_hover.py --record-path flight.parquet --duration 30
    python execution/sim2real/measure_delay_noise.py flight.parquet [more.parquet ...]

Input: any Parquet recording from excite_hover.py with one row per 10 ms control step holding
`step`, `control_time`, the step excitation `exc_vx, exc_vy, exc_vz`, the firmware-echoed
velocity setpoint `echo_vx, echo_vy, echo_vz` (`ctrltarget.v*`), and the measured velocity
`vel_x, vel_y, vel_z`.

Because the step transitions in `exc_v*` are known, each step window is aligned at its command
transition and normalized by the sent step amplitude Δu:
  - Radio delay (`d` steps, `radio_ms`): steps from sending the step command until `echo_v*`
    reflects the new setpoint on the drone.
  - Dead time (`dead_ms`): time from when the command reaches the drone (`d`) until measured
    velocity reaches 5% of the sent command.
  - Time constant (`tau_ms`): time from 5% to 63.2% (1 - 1/e) of the sent command.
  - Sensor noise (`noise_std`, `noise_corr`): stationary std and lag-1 correlation (AR(1)) of
    the step-to-step velocity residuals around the mean step response.
"""

import argparse
import sys

import numpy as np
import pyarrow.parquet as pq

AXES = "xyz"
STEP_ONSET = 0.05                    # 5% of sent command: end of mechanical/tilt dead time
STEP_TAU = 1.0 - float(np.exp(-1))   # 63.2% (1 - 1/e) of sent command: one time constant


def load(path):
    """Columns as float arrays. Exits on a file the step analysis cannot use."""
    table = pq.read_table(path)
    cols = {name: np.asarray(table.column(name).to_numpy(), dtype=float) for name in table.column_names}
    if len(cols["step"]) < 2 or np.any(np.diff(cols["step"]) != 1):
        sys.exit(f"{path}: steps are not consecutive; record every step")  # rho is per step
    for ax in AXES:
        if not (np.isfinite(cols[f"v{ax}"]).all() and np.isfinite(cols[f"vel_{ax}"]).all()):
            sys.exit(f"{path}: non-finite v{ax} or vel_{ax}")
        cols.setdefault(f"exc_v{ax}", np.zeros(len(cols["step"])))
        cols.setdefault(f"echo_v{ax}", np.full(len(cols["step"]), np.nan))
    return cols


def ar1(x):
    """(stationary std, lag-1 correlation) of x."""
    x = x - x.mean()
    return float(x.std()), float(x[1:] @ x[:-1] / (x[:-1] @ x[:-1]))


def _crossing_step(response_curve, threshold, start_step=0):
    """First step index >= `start_step` where `response_curve` reaches `threshold`."""
    steps_above = np.flatnonzero(response_curve[start_step:] >= threshold)
    if len(steps_above) == 0:
        return float("nan")
    return int(steps_above[0]) + start_step


def step_response(exc, echo, vel, dt):
    """Extract (d, radio_ms, dead_ms, tau_ms, noise_std, noise_corr) from known step edges in `exc`."""
    step_edges = np.flatnonzero(np.diff(exc) != 0) + 1
    if len(step_edges) < 2:
        return None
    window_steps = int(np.min(np.diff(step_edges)))
    step_edges = [t for t in step_edges if t + window_steps <= len(vel)]
    if not step_edges:
        return None

    norm_vel_windows, echo_delays = [], []
    for t in step_edges:
        v_start, v_target = exc[t - 1], exc[t]
        norm_vel_windows.append((vel[t:t + window_steps] - v_start) / (v_target - v_start))
        # First step k in the window where the drone's echoed target matches the sent target v_target
        matched = np.flatnonzero(np.isclose(echo[t:t + window_steps], v_target))
        if len(matched):
            echo_delays.append(int(matched[0]))

    mean_vel_response = np.mean(norm_vel_windows, axis=0)
    radio_delay_steps = int(np.median(echo_delays)) if echo_delays else 0

    step_at_5pct = _crossing_step(mean_vel_response, STEP_ONSET, start_step=radio_delay_steps)
    start_63pct = step_at_5pct if isinstance(step_at_5pct, int) else radio_delay_steps
    step_at_63pct = _crossing_step(mean_vel_response, STEP_TAU, start_step=start_63pct)

    radio_ms = radio_delay_steps * dt * 1e3
    dead_ms = max(0.0, (step_at_5pct - radio_delay_steps) * dt * 1e3)
    tau_ms = (step_at_63pct - step_at_5pct) * dt * 1e3

    # Step-to-step velocity residuals around the mean step response (Bessel-corrected for num_windows).
    num_windows = len(step_edges)
    bessel_correction = np.sqrt(num_windows / max(num_windows - 1, 1))
    residuals = [
        (vel[t:t + window_steps] - (exc[t - 1] + (exc[t] - exc[t - 1]) * mean_vel_response)) * bessel_correction
        for t in step_edges
    ]
    noise_std, noise_corr = ar1(np.concatenate(residuals))
    return dict(
        d=radio_delay_steps,
        radio_ms=radio_ms,
        dead_ms=dead_ms,
        tau_ms=tau_ms,
        noise_std=noise_std,
        noise_corr=noise_corr,
    )


def analyse(path):
    """Per excited axis: radio delay, plant dead time, time constant and sensor noise."""
    cols = load(path)
    dt = float(np.median(np.diff(cols["control_time"])))
    axes = {}
    for ax in AXES:
        exc, echo, y = cols[f"exc_v{ax}"], cols[f"echo_v{ax}"], cols[f"vel_{ax}"]
        if np.any(exc != 0):
            res = step_response(exc, echo, y, dt)
            if res is not None:
                axes[ax] = res
    return axes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Characterise step response (radio delay, dead time, tau, "
                                                 "sensor noise) from excite_hover.py recordings.")
    parser.add_argument("paths", nargs="+", help="flight.parquet files from excite_hover.py")
    args = parser.parse_args(argv)

    fits = []
    print(f"{'run':<32} ax {'d':>3} {'radio ms':>8} {'dead ms':>7} {'tau ms':>7} {'noise':>7} {'rho':>5}")
    for path in args.paths:
        for ax, r in analyse(path).items():
            fits.append(r)
            print(f"{path[-32:]:<32} {ax:>2} {r['d']:3d} {r['radio_ms']:8.0f} {r['dead_ms']:7.0f} "
                  f"{r['tau_ms']:7.0f} {r['noise_std']:7.4f} {r['noise_corr']:5.2f}")

    if not fits:
        print("delay and noise: NOT REPORTED, no excited axis. Fly for example\n"
              "  execution/sim2real/excite_hover.py --record-path flight.parquet --exc-axis x --duration 30")
        return
    ds = [r["d"] for r in fits]
    print(f"env.action_latency_steps={min(ds)} env.action_latency_max_steps={max(ds) if max(ds) > min(ds) else 0} "
          f"env.add_noise=True env.noise_std={np.median([r['noise_std'] for r in fits]):.4f} "
          f"env.noise_corr={np.clip(np.median([r['noise_corr'] for r in fits]), 0, 0.99):.2f}")


if __name__ == "__main__":
    main()
