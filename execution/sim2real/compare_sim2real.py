"""Tabulate the sim-to-real gap from two analyze_hover.py JSON reports.

The protocol's CL-1 "performance gap" is sim minus real for one policy, evaluated
with the same goals. This script consumes two reports produced by
``analyze_hover.py --json-out`` and prints a per-metric gap table.

Usage:
    python execution/sim2real/analyze_hover.py execution/sim2real/data_sim/*   --json-out sim.json
    python execution/sim2real/analyze_hover.py execution/sim2real/data/2026-*  --json-out real.json
    python execution/sim2real/compare_sim2real.py sim.json real.json

Metrics that do not exist in sim (motor PWM, battery, Kalman variance) are skipped
automatically -- they are logged as 0.0 by eval_hover_vel_sim.py.
"""

import json
import argparse

# Metrics worth comparing, with a human label and whether lower is better.
METRICS = [
    ("mean_pos_err_m",              "mean position error [m]"),
    ("rms_pos_err_m",               "RMS position error [m]"),
    ("max_drift_m",                 "max drift [m]"),
    ("mean_horiz_err_m",            "horizontal error [m]"),
    ("mean_vert_err_m",             "vertical error [m]"),
    ("mean_speed_mps",              "mean speed [m/s]"),
    ("cmd_smoothness_mps_per_step", "command chatter [m/s/step]"),
    ("act_smoothness_per_step",     "action chatter [1/step]"),
]

# Present in the CSV but meaningless in sim -> never compare these.
SIM_UNAVAILABLE = {"mean_motor_pwm", "mean_vbat_v", "mean_max_kalman_var", "thrust_battery_corr"}


def load(path):
    with open(path) as f:
        rep = json.load(f)
    if "aggregate" in rep:
        agg = rep["aggregate"]
        get = lambda k: (agg[k]["mean"], agg[k]["std"]) if isinstance(agg.get(k), dict) else (None, None)  # noqa: E731
        n = agg.get("n_trials")
        extra = {"success_rate": agg.get("success_rate"), "crash_rate": agg.get("crash_rate")}
    else:
        # single trial: no spread
        t = rep["per_trial"][0]
        get = lambda k: (t.get(k), 0.0)  # noqa: E731
        n = 1
        extra = {"success_rate": float(bool(t.get("success"))), "crash_rate": float(bool(t.get("crashed")))}
    return get, n, extra, rep


def main():
    p = argparse.ArgumentParser(description="Print the sim-to-real gap from two analyze_hover reports.")
    p.add_argument("sim_json")
    p.add_argument("real_json")
    p.add_argument("--json-out", default=None, help="Optional path to save the gap table as JSON.")
    args = p.parse_args()

    sim_get, sim_n, sim_extra, _ = load(args.sim_json)
    real_get, real_n, real_extra, _ = load(args.real_json)

    print(f"\nSim-to-real gap  (sim n={sim_n} trials, real n={real_n} trials)")
    print(f"{'metric':32s} {'sim':>16s} {'real':>16s} {'gap (sim-real)':>16s}  {'ratio':>8s}")
    print("-" * 94)

    out = {"sim_trials": sim_n, "real_trials": real_n, "metrics": {}}
    for key, label in METRICS:
        s_m, s_s = sim_get(key)
        r_m, r_s = real_get(key)
        if s_m is None or r_m is None:
            continue
        gap = s_m - r_m
        ratio = (s_m / r_m) if r_m not in (0, None) else float("nan")
        print(f"{label:32s} {s_m:8.4f}±{s_s:<7.4f} {r_m:8.4f}±{r_s:<7.4f} {gap:>16.4f}  {ratio:>8.2f}")
        out["metrics"][key] = {"sim": s_m, "sim_std": s_s, "real": r_m, "real_std": r_s,
                               "gap": gap, "ratio": None if ratio != ratio else ratio}

    print("-" * 94)
    for k in ("success_rate", "crash_rate"):
        s, r = sim_extra.get(k), real_extra.get(k)
        if s is not None and r is not None:
            print(f"{k:32s} {s:16.3f} {r:16.3f} {s - r:>16.3f}")
            out["metrics"][k] = {"sim": s, "real": r, "gap": s - r}

    print("\nNote: motor PWM, battery and Kalman-variance metrics are not modelled in sim "
          "and are excluded.\nRatio > 1 means sim is larger; for error/chatter metrics "
          "ratio < 1 means the real drone is WORSE than sim.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[INFO] gap table -> {args.json_out}")


if __name__ == "__main__":
    main()
