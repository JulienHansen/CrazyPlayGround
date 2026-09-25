"""Fly the shipped checkpoints in simulation under nominal conditions, for CL-1.

The sim-to-real hover gap compares one policy against itself: the same checkpoint,
the same frozen goal, sim against hardware. This script produces the sim side.

Conditions are nominal on purpose. Every randomisation the policy was *trained*
with is switched off here, because the question is not how the policy copes with
a randomised simulator but how far the nominal simulator is from the real drone.
Each policy still needs its own observation window and action history, since those
are part of the network's input shape rather than a training condition.

Isaac Lab cannot build a second environment in one process, so every run is a
subprocess.

    python execution/sim2real/run_sim_gap.py --headless
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CKPT_DIR = os.path.join(HERE, "flight_checkpoints")
EVAL = os.path.join(HERE, "eval_hover_vel_sim.py")

# Matches the real flights exactly. The 2026-09-24 session flew two conditions:
# a short hold on a goal straight above the takeoff point, and a longer run to a
# goal displaced diagonally. Both the goal AND the duration have to match, or the
# comparison charges the drone for a traverse the simulator never made.
TRIALS = {
    "short": {"target": (0.0, 0.0, 1.0), "duration": 10.0},
    "long":  {"target": (1.0, 1.0, 1.0), "duration": 15.0},
}

NOMINAL = {
    "env.add_noise": "False",
    "env.noise_std": 0.0,
    "env.noise_corr": 0.0,
    "env.noise_bias_std": 0.0,
    "env.disturb": "False",
    "env.disturb_gust_prob": 0.0,
    "env.action_latency_steps": 0,
    "env.action_latency_max": 0,
    "env.physical_dr": "False",
    "env.privileged_state": "False",
}


def main():
    p = argparse.ArgumentParser(description="Run the sim side of the CL-1 hover gap.")
    p.add_argument("--policies", nargs="*", default=None, help="Subset of manifest ids (default: all flown).")
    p.add_argument("--outdir", default=os.path.join(HERE, "data_sim_gap"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    manifest = json.load(open(os.path.join(CKPT_DIR, "manifest.json")))
    wanted = set(args.policies) if args.policies else None
    runs, t0 = [], time.time()

    for entry in manifest:
        if wanted is not None and entry["id"] not in wanted:
            continue
        ckpt = os.path.join(CKPT_DIR, entry["file"])
        for label, trial in TRIALS.items():
            target, duration = trial["target"], trial["duration"]
            tag = f"{entry['id']}_{label}"
            cmd = [sys.executable, EVAL,
                   "--checkpoint", ckpt,
                   "--task", "Vel-Hovering-Robust",
                   "--target", *[str(v) for v in target],
                   "--duration", str(duration),
                   "--tag", tag,
                   "--seed", str(args.seed),
                   "--outdir", args.outdir,
                   "--history-len", str(entry["k"])]
            if args.headless:
                cmd.append("--headless")
            if args.device:
                cmd += ["--device", args.device]
            cmd += [f"env.history_len={entry['k']}", f"env.action_hist_len={entry['m']}"]
            cmd += [f"{k}={v}" for k, v in NOMINAL.items()]

            print(f"\n[{len(runs) + 1}] {tag}  K={entry['k']} M={entry['m']}  {duration:.0f}s", flush=True)
            log = os.path.join(args.outdir, f"{tag}.log")
            os.makedirs(args.outdir, exist_ok=True)
            started = time.time()
            with open(log, "w") as fh:
                rc = subprocess.call(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
            runs.append({"tag": tag, "id": entry["id"], "duration_s": duration,
                         "target": list(target),
                         "k": entry["k"], "m": entry["m"], "returncode": rc,
                         "minutes": round((time.time() - started) / 60, 2), "log": log})
            print(f"    {'ok' if rc == 0 else f'FAILED rc={rc} -> {log}'}"
                  f"  ({runs[-1]['minutes']:.1f} min)", flush=True)

    index = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "trials": TRIALS,
             "conditions": NOMINAL, "seed": args.seed, "runs": runs}
    with open(os.path.join(args.outdir, "sim_gap_index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    ok = sum(r["returncode"] == 0 for r in runs)
    print(f"\n{ok}/{len(runs)} runs ok in {(time.time() - t0) / 60:.1f} min -> {args.outdir}")
    sys.exit(0 if ok == len(runs) else 1)


if __name__ == "__main__":
    main()
