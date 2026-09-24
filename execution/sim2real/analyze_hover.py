"""Compute sim-to-real hover metrics from flights collected by collect_hover_vel.py.

Implements the protocol "Hover metrics" for CL-1:
    position error, jitter, max drift, crash/success, control smoothness,
    hover thrust vs battery.

Usage:
    # one flight
    python analyze_hover.py execution/sim2real/data/2026-07-22_14-00-00_vel

    # many trials -> per-trial table + aggregate mean/spread (protocol: report mean and spread)
    python analyze_hover.py execution/sim2real/data/*_vel --plot

Accepts run directories (containing flight.parquet or flight.csv) or direct paths
to either file.
numpy is required; matplotlib is optional (only for --plot).
"""

import os
import sys
import json
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flight_io import read_flight, resolve as _resolve_flight  # noqa: E402

# A flight counts as successful if it never crashed and stayed close to target.
SUCCESS_POS_ERR_M = 0.15   # mean position error below this = "good hover"
CRASH_Z_M = 0.10           # altitude below this = crash/ground contact
SETTLE_RADIUS_M = 0.15     # radius used for settling-time estimate


def load_flight(path: str) -> dict:
    flight_path = _resolve_flight(path)
    data = read_flight(flight_path)
    meta_path = os.path.join(os.path.dirname(flight_path), "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return {"data": data, "meta": meta,
            "name": os.path.basename(os.path.dirname(flight_path) or flight_path)}


def _finite(a: np.ndarray) -> np.ndarray:
    """Drop non-finite samples.

    Real logs can contain NaN/inf (a dropped radio packet, a diverging estimate).
    Plain np.mean then returns NaN for the whole flight, which silently destroys a
    metric rather than flagging one bad sample, so every reduction below runs on the
    finite subset and the count is reported separately.
    """
    return a[np.isfinite(a)]


def compute_metrics(flight: dict, skip_s: float = 0.0) -> dict:
    """Metrics for one flight, optionally ignoring the first `skip_s` seconds.

    The approach transient is not part of the hover being measured, and it is not
    matched between simulator and drone: the simulator starts 0.50 m below the goal
    and the real drone took off to 0.36 m under a 1.00 m goal. Comparing the two
    without trimming charges the drone for a longer climb.
    """
    d = flight["data"]
    if skip_s > 0.0 and "t_mono" in d and len(d["t_mono"]) > 1:
        keep = d["t_mono"] >= d["t_mono"][0] + skip_s
        if keep.any():
            d = {k: v[keep] for k, v in d.items()}
    n_nonfinite = int(sum(int((~np.isfinite(v)).sum()) for v in d.values()))
    t = d["t_mono"]
    n = len(t)
    duration = float(t[-1] - t[0]) if n > 1 else 0.0
    dt = np.diff(t)
    mean_dt = float(np.mean(dt)) if n > 1 else 0.0

    pos = np.stack([d["pos_x"], d["pos_y"], d["pos_z"]], axis=1)
    tgt = np.stack([d["tgt_x"], d["tgt_y"], d["tgt_z"]], axis=1)
    err = pos - tgt
    err_norm = np.linalg.norm(err, axis=1)
    err_f = _finite(err_norm)
    err_h = np.linalg.norm(err[:, :2], axis=1)   # horizontal
    err_v = np.abs(err[:, 2])                     # vertical

    # jitter: how much the position wanders around its own mean (steady wobble)
    jitter_xyz = np.std(pos, axis=0)
    speed = np.linalg.norm(np.stack([d["vel_x"], d["vel_y"], d["vel_z"]], axis=1), axis=1)

    # control smoothness: per-step change of the commanded velocity (lower = smoother)
    cmd = np.stack([d["cmd_vx"], d["cmd_vy"], d["cmd_vz"]], axis=1)
    if n > 1:
        dcmd = _finite(np.linalg.norm(np.diff(cmd, axis=0), axis=1))
        cmd_smoothness = float(np.mean(dcmd)) if dcmd.size else float("nan")
        act = np.stack([d["act_x"], d["act_y"], d["act_z"]], axis=1)
        dact = _finite(np.linalg.norm(np.diff(act, axis=0), axis=1))
        act_smoothness = float(np.mean(dact)) if dact.size else float("nan")
    else:
        cmd_smoothness = act_smoothness = 0.0

    # hover thrust vs battery
    motors = np.stack([d["m1"], d["m2"], d["m3"], d["m4"]], axis=1)
    mean_motor = float(np.mean(motors))
    vbat = d["vbat"]
    valid_bat = vbat > 1.0  # ignore zeros before first pm.vbat sample
    if np.count_nonzero(valid_bat) > 2 and np.std(vbat[valid_bat]) > 1e-6:
        thrust_bat_corr = float(np.corrcoef(np.mean(motors, axis=1)[valid_bat], vbat[valid_bat])[0, 1])
    else:
        thrust_bat_corr = float("nan")

    # crash / success
    grace_mask = t - t[0] > 3.0
    _z = _finite(pos[grace_mask, 2] if np.any(grace_mask) else pos[:, 2])
    min_z_after_grace = float(np.min(_z)) if _z.size else float("nan")
    crashed = bool(min_z_after_grace < CRASH_Z_M)
    meta = flight.get("meta", {})
    planned = meta.get("duration_s")
    ended_early = bool(planned and duration < 0.9 * float(planned))
    mean_pos_err = float(np.mean(err_f)) if err_f.size else float("nan")
    success = bool((not crashed) and (not ended_early) and (mean_pos_err < SUCCESS_POS_ERR_M))

    # settling time: first time err stays within SETTLE_RADIUS for the rest of the flight
    settle_time = float("nan")
    within = err_norm < SETTLE_RADIUS_M
    for i in range(n):
        if np.all(within[i:]):
            settle_time = float(t[i] - t[0])
            break

    # loop timing: where the control period actually goes. Written by collectors
    # that instrument the loop; absent from older flights, hence the .get().
    def _ms(col):
        v = d.get(col)
        if v is None or v.size == 0:
            return None
        v = _finite(v)
        return None if v.size == 0 else (round(float(np.mean(v)), 3),
                                         round(float(np.percentile(v, 95)), 3))
    timing = {k: _ms(c) for k, c in (("infer", "dt_infer_ms"), ("send", "dt_send_ms"),
                                     ("body", "dt_body_ms"), ("sleep", "dt_sleep_ms"))}
    period_ms = mean_dt * 1e3
    rate_loss_ms = None
    if timing["body"] is not None and period_ms > 0:
        # commanded period is 10 ms at 100 Hz; anything above it is the loss
        rate_loss_ms = round(period_ms - 1000.0 / 100.0, 3)

    # estimator quality
    varp = np.stack([d["varPX"], d["varPY"], d["varPZ"]], axis=1)
    mean_max_varp = float(np.mean(np.max(varp, axis=1)))

    return {
        "name": flight["name"],
        "skip_s": skip_s,
        "n_samples": n,
        "duration_s": round(duration, 3),
        "effective_rate_hz": round(1.0 / mean_dt, 1) if mean_dt > 0 else 0.0,
        "mean_pos_err_m": round(mean_pos_err, 4),
        "rms_pos_err_m": round(float(np.sqrt(np.mean(err_f ** 2))) if err_f.size else float("nan"), 4),
        "max_drift_m": round(float(np.max(err_f)) if err_f.size else float("nan"), 4),
        "mean_horiz_err_m": round(float(np.mean(_finite(err_h))), 4),
        "mean_vert_err_m": round(float(np.mean(_finite(err_v))), 4),
        "n_nonfinite_samples": n_nonfinite,
        "jitter_std_xyz_m": [round(float(x), 4) for x in jitter_xyz],
        "mean_speed_mps": round(float(np.mean(_finite(speed))), 4),
        "cmd_smoothness_mps_per_step": round(cmd_smoothness, 5),
        "act_smoothness_per_step": round(act_smoothness, 5),
        # Per-step chatter is not comparable across loop rates: a slower loop takes
        # bigger steps along the same trajectory. The real drone runs at ~85 Hz and
        # the simulator at 100, so the gap is reported per second as well.
        "cmd_smoothness_mps2": round(cmd_smoothness / mean_dt, 4) if mean_dt > 0 else None,
        "act_smoothness_per_s": round(act_smoothness / mean_dt, 4) if mean_dt > 0 else None,
        "mean_motor_pwm": round(mean_motor, 1),
        "thrust_battery_corr": None if np.isnan(thrust_bat_corr) else round(thrust_bat_corr, 3),
        "mean_vbat_v": round(float(np.mean(vbat[valid_bat])) if np.any(valid_bat) else 0.0, 3),
        "settle_time_s": None if np.isnan(settle_time) else round(settle_time, 3),
        "min_z_after_grace_m": round(min_z_after_grace, 3),
        "mean_max_kalman_var": round(mean_max_varp, 5),
        "loop_ms_mean_p95": {k: v for k, v in timing.items() if v is not None} or None,
        "period_excess_ms": rate_loss_ms,
        "crashed": crashed,
        "ended_early": ended_early,
        "success": success,
    }


def aggregate(all_metrics: list) -> dict:
    """Mean ± spread across trials for the numeric metrics (protocol: report mean and spread)."""
    numeric_keys = [
        "mean_pos_err_m", "rms_pos_err_m", "max_drift_m", "mean_horiz_err_m",
        "mean_vert_err_m", "mean_speed_mps", "cmd_smoothness_mps_per_step",
        "act_smoothness_per_step", "cmd_smoothness_mps2", "act_smoothness_per_s",
        "mean_motor_pwm", "mean_vbat_v", "mean_max_kalman_var",
    ]
    agg = {"n_trials": len(all_metrics),
           "success_rate": round(float(np.mean([m["success"] for m in all_metrics])), 3),
           "crash_rate": round(float(np.mean([m["crashed"] for m in all_metrics])), 3)}
    for k in numeric_keys:
        vals = np.array([m[k] for m in all_metrics], dtype=np.float64)
        agg[k] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
    return agg


def plot_flight(flight: dict, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available — skipping plot")
        return
    d = flight["data"]
    t = d["t_mono"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis in ("x", "y", "z"):
        ax[0].plot(t, d[f"pos_{axis}"], label=f"pos_{axis}")
        ax[0].plot(t, d[f"tgt_{axis}"], "--", alpha=0.5, label=f"tgt_{axis}")
    ax[0].set_ylabel("position [m]"); ax[0].legend(ncol=3, fontsize=8); ax[0].set_title(flight["name"])
    for axis in ("vx", "vy", "vz"):
        ax[1].plot(t, d[f"cmd_{axis}"], label=f"cmd_{axis}")
    ax[1].set_ylabel("cmd vel [m/s]"); ax[1].legend(ncol=3, fontsize=8)
    for i in range(1, 5):
        ax[2].plot(t, d[f"m{i}"], label=f"m{i}")
    ax2b = ax[2].twinx(); ax2b.plot(t, d["vbat"], "k--", alpha=0.4, label="vbat")
    ax[2].set_ylabel("motor PWM"); ax2b.set_ylabel("vbat [V]"); ax[2].set_xlabel("t [s]")
    ax[2].legend(ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[INFO] plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute sim2real hover metrics from collected flights.")
    parser.add_argument("paths", nargs="+",
                        help="Run directories, or flight.parquet / flight.csv paths (globs allowed).")
    parser.add_argument("--plot", action="store_true", help="Save a per-flight plot next to each recording.")
    parser.add_argument("--json-out", type=str, default=None, help="Write the full metrics report to this JSON file.")
    parser.add_argument("--skip-s", type=float, default=0.0,
                        help="Ignore the first N seconds, so the approach transient does not enter "
                             "the hover metrics.")
    args = parser.parse_args()

    expanded = []
    for p in args.paths:
        hits = glob.glob(p)
        expanded.extend(hits if hits else [p])

    all_metrics = []
    for p in expanded:
        try:
            flight = load_flight(p)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue
        m = compute_metrics(flight, skip_s=args.skip_s)
        all_metrics.append(m)
        print(f"\n=== {m['name']} ===")
        for k, v in m.items():
            if k != "name":
                print(f"  {k:28s}: {v}")
        if args.plot:
            plot_flight(flight, os.path.join(os.path.dirname(_resolve_flight(p)), "flight.png"))

    if not all_metrics:
        print("No flights loaded.")
        sys.exit(1)

    report = {"per_trial": all_metrics}
    if len(all_metrics) > 1:
        agg = aggregate(all_metrics)
        report["aggregate"] = agg
        print("\n=== AGGREGATE (mean ± std over trials) ===")
        print(f"  n_trials     : {agg['n_trials']}")
        print(f"  success_rate : {agg['success_rate']}")
        print(f"  crash_rate   : {agg['crash_rate']}")
        for k, v in agg.items():
            if isinstance(v, dict):
                print(f"  {k:28s}: {v['mean']} ± {v['std']}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[INFO] report -> {args.json_out}")


if __name__ == "__main__":
    main()
