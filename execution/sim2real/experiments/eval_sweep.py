"""Evaluate every policy from the sweep under a held-out test condition and rank them.

For each checkpoint in `sweep_index.json` this runs `eval_hover_vel_sim.py` on fixed
targets under a *common, harder* disturbance+noise condition that policies were not
necessarily trained on, then scores the resulting flights with `analyze_hover.py`
metrics.

Score (lower is better):
    robustness = mean_pos_err + max_drift + 10 * crash_rate + chatter_weight * cmd_smoothness

Chatter is included explicitly because it is the dominant sim-to-real gap measured
on the real Crazyflie (~10x worse on hardware than in sim).

Usage:
    python execution/sim2real/experiments/eval_sweep.py
    python execution/sim2real/experiments/eval_sweep.py --duration 15 --chatter-weight 5
"""

import os
import sys
import json
import glob
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PYTHON = os.environ.get("ISAAC_PYTHON", sys.executable)
EVAL = os.path.join(REPO, "execution", "sim2real", "eval_hover_vel_sim.py")

# Held-out test condition: moderate noise + disturbance with gusts, identical for
# every policy so the comparison is fair.
TEST_CONDITION = {
    "env.add_noise": "True",
    "env.noise_std": 0.03,
    "env.disturb": "True",
    "env.disturb_gust_prob": 0.02,
    "env.disturb_force_bias_range": "[-0.03,0.03]",
    "env.disturb_torque_bias_range": "[-0.0003,0.0003]",
    "env.disturb_gust_force_range": "[-0.08,0.08]",
    "env.disturb_gust_torque_range": "[-0.0008,0.0008]",
}
TARGETS = [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0)]


def run_eval(ckpt, history_len, target, duration, outdir, tag, extra_env_overrides):
    cmd = [PYTHON, EVAL, "--checkpoint", ckpt, "--task", "Vel-Hovering-Robust",
           "--target", *[str(t) for t in target], "--duration", str(duration),
           "--outdir", outdir, "--tag", tag, "--headless",
           f"env.history_len={history_len}"]
    cmd += [f"{k}={v}" for k, v in extra_env_overrides.items()]
    env = dict(os.environ, WANDB_MODE="disabled")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    return r.returncode


def metrics_for(run_glob):
    """Aggregate analyze_hover metrics over the run dirs matching run_glob."""
    from analyze_hover import load_flight, compute_metrics  # noqa
    dirs = sorted(glob.glob(run_glob))
    ms = []
    for d in dirs:
        try:
            ms.append(compute_metrics(load_flight(d)))
        except Exception as e:
            print(f"  [warn] {d}: {e}")
    return ms


def main():
    p = argparse.ArgumentParser(description="Evaluate and rank the sweep's policies.")
    p.add_argument("--index", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_index.json"))
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--chatter-weight", type=float, default=5.0)
    p.add_argument("--outroot", default=os.path.join(REPO, "execution", "sim2real", "data_sweep"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.json"))
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--nominal", action="store_true",
                   help="Also evaluate under the nominal (clean) condition for reference.")
    args = p.parse_args()

    index = json.load(open(args.index))
    runs = {k: v for k, v in index["runs"].items() if v.get("checkpoint")}
    if args.only:
        runs = {k: v for k, v in runs.items() if k in args.only}
    print(f"[INFO] evaluating {len(runs)} policies under the held-out test condition")

    conditions = {"test": TEST_CONDITION}
    if args.nominal:
        conditions["nominal"] = {"env.add_noise": "False", "env.disturb": "False"}

    rows = []
    for i, (tag, info) in enumerate(sorted(runs.items())):
        k = int(info["factors"]["history_len"])
        row = {"tag": tag, **info["factors"]}
        for cond_name, cond in conditions.items():
            outdir = os.path.join(args.outroot, cond_name, tag)
            for ti, tgt in enumerate(TARGETS):
                run_eval(info["checkpoint"], k, tgt, args.duration, outdir, f"t{ti}", cond)
            ms = metrics_for(os.path.join(outdir, "*"))
            if not ms:
                row[f"{cond_name}_ok"] = False
                continue
            import numpy as np
            row[f"{cond_name}_ok"] = True
            row[f"{cond_name}_pos_err"] = float(np.mean([m["mean_pos_err_m"] for m in ms]))
            row[f"{cond_name}_max_drift"] = float(np.mean([m["max_drift_m"] for m in ms]))
            row[f"{cond_name}_chatter"] = float(np.mean([m["cmd_smoothness_mps_per_step"] for m in ms]))
            row[f"{cond_name}_crash_rate"] = float(np.mean([m["crashed"] for m in ms]))
            row[f"{cond_name}_speed"] = float(np.mean([m["mean_speed_mps"] for m in ms]))
        if row.get("test_ok"):
            row["score"] = (row["test_pos_err"] + row["test_max_drift"]
                            + 10.0 * row["test_crash_rate"]
                            + args.chatter_weight * row["test_chatter"])
        else:
            row["score"] = float("inf")
        rows.append(row)
        print(f"[{i+1}/{len(runs)}] {tag:26s} score={row['score']:.4f}")

    rows.sort(key=lambda r: r["score"])
    print("\n=== LEADERBOARD (held-out test condition; lower score = better) ===")
    hdr = f"{'#':>2} {'tag':26s} {'K':>2} {'noise':>6} {'dist':>5} {'pos_err':>8} {'drift':>7} {'chatter':>8} {'crash':>6} {'score':>7}"
    print(hdr); print("-" * len(hdr))
    for i, r in enumerate(rows):
        if r["score"] == float("inf"):
            print(f"{i+1:>2} {r['tag']:26s}  FAILED"); continue
        print(f"{i+1:>2} {r['tag']:26s} {r['history_len']:>2} {r['noise_std']:>6} "
              f"{str(r['disturb']):>5} {r['test_pos_err']:>8.4f} {r['test_max_drift']:>7.3f} "
              f"{r['test_chatter']:>8.5f} {r['test_crash_rate']:>6.2f} {r['score']:>7.4f}")

    with open(args.out, "w") as f:
        json.dump({"chatter_weight": args.chatter_weight, "condition": TEST_CONDITION,
                   "targets": TARGETS, "rows": rows}, f, indent=2)
    print(f"\n[INFO] leaderboard -> {args.out}")


if __name__ == "__main__":
    main()
