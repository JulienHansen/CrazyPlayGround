"""Driver: evaluate every sweep policy for DR generalization, one subprocess each.

IsaacLab cannot create a second environment inside one process -- the stage cannot be
rebuilt and the process wedges (observed: a Python loop over 12 policies hung on the
second gym.make and spun at 100% CPU for days). So each policy is evaluated in its own
subprocess, exactly like sweep_train.py runs each training in its own process.

    python execution/sim2real/experiments/eval_dr_sweep.py \
        --index execution/sim2real/experiments/sweep_index_v3.json --num-envs 256
"""

import os
import sys
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PYTHON = os.environ.get("ISAAC_PYTHON", sys.executable)
EVAL = os.path.join(HERE, "eval_dr_generalization.py")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=os.path.join(HERE, "sweep_index_v3.json"))
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--dr-width", choices=["train", "held_out", "extreme"], default="held_out")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--timeout", type=int, default=900, help="Per-policy wall-clock cap (s).")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    index = json.load(open(args.index))
    runs = {k: v for k, v in index["runs"].items() if v.get("checkpoint")}
    if args.only:
        runs = {k: v for k, v in runs.items() if k in args.only}
    print(f"[INFO] {len(runs)} policies x {args.num_envs} bodies, dr_width={args.dr_width}")

    rows = []
    for i, (tag, info) in enumerate(sorted(runs.items())):
        f = info["factors"]
        cmd = [PYTHON, "-u", EVAL, "--headless",
               "--checkpoint", info["checkpoint"], "--tag", tag,
               "--history-len", str(int(f["history_len"])),
               "--action-hist-len", str(int(f.get("action_hist_len", 0))),
               "--num-envs", str(args.num_envs), "--duration", str(args.duration),
               "--dr-width", args.dr_width]
        env = dict(os.environ, WANDB_MODE="disabled")
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                               env=env, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"[{i+1}/{len(runs)}] {tag:18s} TIMEOUT after {args.timeout}s")
            continue
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(f"[{i+1}/{len(runs)}] {tag:18s} FAILED (rc={r.returncode})")
            tail = [l for l in (r.stdout + r.stderr).splitlines() if "Error" in l or "Traceback" in l]
            for l in tail[-3:]:
                print("      ", l)
            continue
        m = json.loads(line[len("RESULT "):])
        m.update({k: f[k] for k in ("physical_dr", "history_len", "action_hist_len")})
        rows.append(m)
        print(f"[{i+1}/{len(runs)}] {tag:18s} pos_err={m['pos_err']:.4f} "
              f"chatter={m['chatter']:.4f} crash={m['crash_rate']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        print("no results"); return

    rows.sort(key=lambda r: r["pos_err"] + 5.0 * r["chatter"] + 10.0 * r["crash_rate"])
    print(f"\n=== DR GENERALIZATION ({args.dr_width}) — lower is better ===")
    h = (f"{'#':>2} {'policy':18s} {'DR':>7} {'K':>2} {'M':>2} {'pos_err':>8} "
         f"{'p90':>7} {'chatter':>8} {'crash':>6}")
    print(h); print("-" * len(h))
    for i, r in enumerate(rows):
        print(f"{i+1:>2} {r['tag']:18s} {str(r['physical_dr']):>7} {r['history_len']:>2} "
              f"{r['action_hist_len']:>2} {r['pos_err']:>8.4f} {r['pos_err_p90']:>7.4f} "
              f"{r['chatter']:>8.4f} {r['crash_rate']:>6.3f}")

    out = args.out or os.path.join(HERE, f"dr_generalization_{args.dr_width}.json")
    json.dump({"dr_width": args.dr_width, "num_envs": args.num_envs, "rows": rows},
              open(out, "w"), indent=2)
    print(f"\n[INFO] -> {out}")


if __name__ == "__main__":
    main()
