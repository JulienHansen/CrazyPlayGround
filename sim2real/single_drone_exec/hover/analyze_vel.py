"""Hover metrics from flights recorded by execution/single_drone_exec/hover/exec_vel.py.

Reads the Parquet file FlightRecorder writes (one row per control step) and reports:
    position error on a steady-state window, jitter, max drift, settling time,
    command smoothness, motor PWM, crash/success.

Usage:
    # one flight
    python sim2real/single_drone_exec/hover/analyze_vel.py flight.parquet

    # many trials -> per-trial table + aggregate mean/spread, approach transient removed
    python sim2real/single_drone_exec/hover/analyze_vel.py logs/*.parquet --skip-s 3 --json-out report.json

Accepts Parquet files or directories holding exactly one. An optional metadata.json next
to the file supplies the planned `duration_s`. numpy and pyarrow are required;
matplotlib is optional (only for --plot).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

# A flight counts as successful if it never crashed and stayed close to target.
SUCCESS_POS_ERR_M = 0.15   # mean position error below this = "good hover"
CRASH_Z_M = 0.10           # altitude below this = crash/ground contact
CRASH_GRACE_S = 3.0        # takeoff time before the crash check applies, as in exec_vel's watchdog
SETTLE_RADIUS_M = 0.15     # radius used for settling-time estimate

# Velocity-mode columns, as exec_vel.py declares them through make_recorder.
CMD_COLS = ("vx", "vy", "vz")
ACT_COLS = ("action_vx", "action_vy", "action_vz")
MOTOR_COLS = ("motor_m1_pwm", "motor_m2_pwm", "motor_m3_pwm", "motor_m4_pwm")

NUMERIC_KEYS = [
    "mean_pos_err_m", "rms_pos_err_m", "max_drift_m", "mean_horiz_err_m",
    "mean_vert_err_m", "mean_speed_mps", "cmd_smoothness_mps_per_step",
    "act_smoothness_per_step", "cmd_smoothness_mps2", "act_smoothness_per_s",
    "mean_motor_pwm",
]


def load_flight(path: str) -> dict:
    """Load a recording as {"data": {column: float64 array}, "meta", "name", "path"}."""
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if len(hits) != 1:
            raise FileNotFoundError(f"Expected one .parquet in {path}, found {len(hits)}")
        path = hits[0]
    if not os.path.exists(path):
        raise FileNotFoundError(f"No flight recording at {path}")
    table = pq.read_table(path)
    missing = [c for c in ("step", "control_time", *CMD_COLS, *ACT_COLS) if c not in table.column_names]
    if missing:
        raise ValueError(f"{path} is not a velocity-mode (exec_vel) recording: missing {missing}")
    data = {name: np.asarray(table[name].to_numpy(), dtype=np.float64) for name in table.column_names}
    meta_path = os.path.join(os.path.dirname(path), "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    # the path, not the file name: every run directory may hold a file called flight.parquet
    return {"data": data, "meta": meta, "name": os.path.splitext(os.path.relpath(path))[0], "path": path}


def _finite(a: np.ndarray) -> np.ndarray:
    """Drop non-finite samples.

    Real logs can contain NaN/inf (a dropped radio packet, a diverging estimate).
    Plain np.mean then returns NaN for the whole flight, which silently destroys a
    metric rather than flagging one bad sample, so every reduction below runs on the
    finite subset and the count is reported separately.
    """
    return a[np.isfinite(a)]


def _fmean(a: np.ndarray) -> float:
    a = _finite(a)
    return float(np.mean(a)) if a.size else float("nan")


def _r(x, nd: int):
    """Round for the report; non-finite becomes None so the JSON stays valid."""
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


def compute_metrics(flight: dict, skip_s: float = 0.0) -> dict:
    """Metrics for one flight, the hover ones ignoring the first `skip_s` seconds.

    The approach transient is not part of the hover being measured, and it is not
    matched between simulator and drone, so position error, smoothness and settling
    are measured after `skip_s`. Crash and early end describe the whole flight and
    are always measured on all of it.
    """
    full = flight["data"]
    # shortcut: control_time is wall-clock time.time(), so an NTP jump mid-flight skews rates; upgrade if FlightRecorder records a monotonic clock
    t_full = full["control_time"]
    if t_full.size == 0:
        raise ValueError(f"{flight['name']} has no recorded steps")
    d = full
    if skip_s > 0.0 and len(t_full) > 1:
        keep = t_full >= t_full[0] + skip_s
        if keep.any():
            d = {k: v[keep] for k, v in full.items()}
    n_nonfinite = int(np.count_nonzero(~np.isfinite(np.stack(list(d.values()), axis=1)).all(axis=1)))
    t, step = d["control_time"], d["step"]
    n = len(t)
    # per control step, not per row: rows are missing when record_every > 1 or a row was skipped
    steps_spanned = step[-1] - step[0] if n > 1 else 0.0
    step_dt = float(t[-1] - t[0]) / steps_spanned if steps_spanned > 0 else 0.0

    pos = np.stack([d["pos_x"], d["pos_y"], d["pos_z"]], axis=1)
    tgt = np.stack([d["target_x"], d["target_y"], d["target_z"]], axis=1)
    err = pos - tgt
    err_norm = np.linalg.norm(err, axis=1)
    err_f = _finite(err_norm)
    mean_pos_err = float(np.mean(err_f)) if err_f.size else float("nan")

    # jitter: how much the position wanders around its own mean (steady wobble)
    jitter_xyz = [np.std(_finite(pos[:, i])) if _finite(pos[:, i]).size else np.nan for i in range(3)]
    speed = np.linalg.norm(np.stack([d["vel_x"], d["vel_y"], d["vel_z"]], axis=1), axis=1)

    # smoothness: mean change of the command between consecutive control steps (lower = smoother)
    consecutive = np.diff(step) == 1

    def _step_change(cols):
        x = np.stack([d[c] for c in cols], axis=1)
        return _fmean(np.linalg.norm(np.diff(x, axis=0), axis=1)[consecutive])

    cmd_smoothness, act_smoothness = _step_change(CMD_COLS), _step_change(ACT_COLS)

    motors = np.stack([d[c] for c in MOTOR_COLS], axis=1)

    # settling time: first time err stays within SETTLE_RADIUS for the rest of the window (NaN ignored)
    outside = np.flatnonzero(err_norm >= SETTLE_RADIUS_M)
    if outside.size == 0:
        settle_time = 0.0
    elif outside[-1] + 1 < n:
        settle_time = float(t[outside[-1] + 1] - t[0])
    else:
        settle_time = float("nan")

    # crash / early end, on the whole flight so the skipped window is still checked
    # shortcut: exec_vel's watchdog stops before recording a row below CRASH_Z_M and records no planned duration, so crashed/ended_early rarely fire; upgrade when WP6 records the exit reason and duration_s
    duration = float(t_full[-1] - t_full[0])
    grace = t_full - t_full[0] > CRASH_GRACE_S
    z = _finite(full["pos_z"][grace] if grace.any() else full["pos_z"])
    min_z_after_grace = float(np.min(z)) if z.size else float("nan")
    crashed = bool(min_z_after_grace < CRASH_Z_M)
    planned = flight.get("meta", {}).get("duration_s")
    ended_early = bool(planned and duration < 0.9 * float(planned))
    success = bool((not crashed) and (not ended_early) and (mean_pos_err < SUCCESS_POS_ERR_M))

    return {
        "name": flight["name"],
        "skip_s": skip_s,
        "n_samples": n,
        "duration_s": _r(duration, 3),
        "effective_rate_hz": _r(1.0 / step_dt, 1) if step_dt > 0 else None,
        "mean_pos_err_m": _r(mean_pos_err, 4),
        "rms_pos_err_m": _r(np.sqrt(np.mean(err_f ** 2)) if err_f.size else np.nan, 4),
        "max_drift_m": _r(np.max(err_f) if err_f.size else np.nan, 4),
        "mean_horiz_err_m": _r(_fmean(np.linalg.norm(err[:, :2], axis=1)), 4),
        "mean_vert_err_m": _r(_fmean(np.abs(err[:, 2])), 4),
        "n_nonfinite_samples": n_nonfinite,
        "jitter_std_xyz_m": [_r(x, 4) for x in jitter_xyz],
        "mean_speed_mps": _r(_fmean(speed), 4),
        "cmd_smoothness_mps_per_step": _r(cmd_smoothness, 5),
        "act_smoothness_per_step": _r(act_smoothness, 5),
        # Per-step smoothness is not comparable across loop rates: a slower loop takes
        # bigger steps along the same trajectory. The real drone runs at ~85 Hz and
        # the simulator at 100, so it is reported per second as well.
        "cmd_smoothness_mps2": _r(cmd_smoothness / step_dt, 4) if step_dt > 0 else None,
        "act_smoothness_per_s": _r(act_smoothness / step_dt, 4) if step_dt > 0 else None,
        "mean_motor_pwm": _r(_fmean(motors), 1),
        "settle_time_s": _r(settle_time, 3),
        "min_z_after_grace_m": _r(min_z_after_grace, 3),
        "crashed": crashed,
        "ended_early": ended_early,
        "success": success,
    }


def aggregate(all_metrics: list) -> dict:
    """Mean ± spread across trials for the numeric metrics; a trial without a value is left out."""
    agg = {"n_trials": len(all_metrics),
           "success_rate": round(float(np.mean([m["success"] for m in all_metrics])), 3),
           "crash_rate": round(float(np.mean([m["crashed"] for m in all_metrics])), 3)}
    for k in NUMERIC_KEYS:
        vals = np.array([m[k] for m in all_metrics if m[k] is not None], dtype=np.float64)
        agg[k] = {"mean": _r(np.mean(vals), 4), "std": _r(np.std(vals), 4)} if vals.size else None
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
    t = d["control_time"] - d["control_time"][0]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis in ("x", "y", "z"):
        ax[0].plot(t, d[f"pos_{axis}"], label=f"pos_{axis}")
        ax[0].plot(t, d[f"target_{axis}"], "--", alpha=0.5, label=f"target_{axis}")
    ax[0].set_ylabel("position [m]"); ax[0].legend(ncol=3, fontsize=8); ax[0].set_title(flight["name"], fontsize=9)
    for c in CMD_COLS:
        ax[1].plot(t, d[c], label=f"cmd_{c}")
    ax[1].set_ylabel("cmd vel [m/s]"); ax[1].legend(ncol=3, fontsize=8)
    for c in MOTOR_COLS:
        ax[2].plot(t, d[c], label=c)
    ax[2].set_ylabel("motor PWM"); ax[2].set_xlabel("t [s]")
    ax[2].legend(ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[INFO] plot -> {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute sim2real hover metrics from recorded exec_vel flights.")
    parser.add_argument("paths", nargs="+",
                        help="Parquet recordings, or directories holding one (globs allowed).")
    parser.add_argument("--plot", action="store_true", help="Save a plot next to each recording.")
    parser.add_argument("--json-out", type=str, default=None, help="Write the full metrics report to this JSON file.")
    parser.add_argument("--skip-s", type=float, default=0.0,
                        help="Ignore the first N seconds, so the approach transient does not enter "
                             "the hover metrics.")
    args = parser.parse_args(argv)

    expanded = []
    for p in args.paths:
        hits = glob.glob(p)
        expanded.extend(hits if hits else [p])

    all_metrics = []
    for p in expanded:
        try:
            flight = load_flight(p)
            m = compute_metrics(flight, skip_s=args.skip_s)
        except (FileNotFoundError, ValueError) as e:
            print(f"[SKIP] {e}")
            continue
        all_metrics.append(m)
        print(f"\n=== {m['name']} ===")
        for k, v in m.items():
            if k != "name":
                print(f"  {k:28s}: {v}")
        if args.plot:
            plot_flight(flight, os.path.splitext(flight["path"])[0] + ".png")

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
        for k in NUMERIC_KEYS:
            v = agg[k]
            print(f"  {k:28s}: {v['mean']} ± {v['std']}" if v else f"  {k:28s}: None")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        print(f"\n[INFO] report -> {args.json_out}")


if __name__ == "__main__":
    main()
