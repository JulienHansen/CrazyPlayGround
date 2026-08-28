"""Run the hover robustness training sweep, one subprocess per config.

Hydra's own `-m` multirun is not usable here: Isaac Sim is launched once per
process (before Hydra runs) and `--seed` is an argparse flag outside the Hydra
tree. So we spawn one `scripts/skrl/train.py` per config instead, which also gives
each run an independent W&B entry.

Usage (from the repo root):
    python execution/sim2real/experiments/sweep_train.py            # full sweep
    python execution/sim2real/experiments/sweep_train.py --dry-run  # print commands
    python execution/sim2real/experiments/sweep_train.py --only n0.05_k8_d1_ar_s42
    python execution/sim2real/experiments/sweep_train.py --num-envs 2048 --max-iterations 300

Writes `sweep_index.json` mapping each tag -> its run dir and best checkpoint, which
`eval_sweep.py` consumes.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Which sweep design to run: `configs` (sweep 1) or `configs_v2` (action history +
# latency). Selected with --configs; imported dynamically so both stay reproducible.
_CFG_MODULE = os.environ.get("SWEEP_CONFIGS", "configs")
for _i, _a in enumerate(sys.argv):
    if _a == "--configs" and _i + 1 < len(sys.argv):
        _CFG_MODULE = sys.argv[_i + 1]
    elif _a.startswith("--configs="):
        _CFG_MODULE = _a.split("=", 1)[1]
_cfg = __import__(_CFG_MODULE)
SWEEP, TASK = _cfg.SWEEP, _cfg.TASK
NUM_ENVS, MAX_ITERATIONS = _cfg.NUM_ENVS, _cfg.MAX_ITERATIONS
WANDB_PROJECT, WANDB_GROUP = _cfg.WANDB_PROJECT, _cfg.WANDB_GROUP

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PYTHON = os.environ.get("ISAAC_PYTHON", sys.executable)
LOG_ROOT = os.path.join(REPO, "logs", "skrl")


def newest_run_dir(before: set):
    """The run directory that appeared since `before` was captured."""
    now = set()
    for exp in os.listdir(LOG_ROOT) if os.path.isdir(LOG_ROOT) else []:
        d = os.path.join(LOG_ROOT, exp)
        if os.path.isdir(d):
            now |= {os.path.join(d, r) for r in os.listdir(d)}
    new = sorted(now - before)
    return new[-1] if new else None


def snapshot_runs():
    out = set()
    for exp in os.listdir(LOG_ROOT) if os.path.isdir(LOG_ROOT) else []:
        d = os.path.join(LOG_ROOT, exp)
        if os.path.isdir(d):
            out |= {os.path.join(d, r) for r in os.listdir(d)}
    return out


def main():
    p = argparse.ArgumentParser(description="Run the hover robustness training sweep.")
    p.add_argument("--configs", default="configs",
                   help="Sweep design module: configs (sweep 1) or configs_v2.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="Run only these tags.")
    p.add_argument("--num-envs", type=int, default=NUM_ENVS)
    p.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B for this sweep.")
    p.add_argument("--out", default=None,
                   help="Index file (default: sweep_index[_<configs>].json)")
    args = p.parse_args()
    if args.out is None:
        suffix = "" if args.configs == "configs" else "_" + args.configs.replace("configs_", "")
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sweep_index{suffix}.json")

    runs = SWEEP if not args.only else [r for r in SWEEP if r["tag"] in args.only]
    if not runs:
        print("No matching runs.")
        return

    print(f"[INFO] {len(runs)} runs, task={TASK}, num_envs={args.num_envs}, "
          f"max_iterations={args.max_iterations}, wandb={'off' if args.no_wandb else WANDB_PROJECT}")

    index = {"created": datetime.now().isoformat(), "task": TASK,
             "num_envs": args.num_envs, "max_iterations": args.max_iterations, "runs": {}}
    if os.path.exists(args.out):          # resume-friendly: keep earlier results
        try:
            index["runs"] = json.load(open(args.out)).get("runs", {})
        except Exception:
            pass

    for i, r in enumerate(runs):
        tag = r["tag"]
        if index["runs"].get(tag, {}).get("checkpoint"):
            print(f"[SKIP] {tag} already done")
            continue

        cmd = [PYTHON, "scripts/skrl/train.py", "--task", TASK, "--headless",
               "--num_envs", str(args.num_envs),
               "--max_iterations", str(args.max_iterations),
               "--seed", str(r["seed"])]
        cmd += [f"{k}={v}" for k, v in r["overrides"].items()]
        if not args.no_wandb:
            # `++` = set-or-append: `name`/`group` do not exist in skrl_ppo_cfg.yaml,
            # and plain `key=value` fails on absent keys.
            cmd += [f"++agent.agent.experiment.wandb_kwargs.name={tag}",
                    f"++agent.agent.experiment.wandb_kwargs.group={WANDB_GROUP}",
                    f"++agent.agent.experiment.wandb_kwargs.project={WANDB_PROJECT}"]

        print(f"\n[{i+1}/{len(runs)}] {tag}\n  {' '.join(cmd)}")
        if args.dry_run:
            continue

        env = dict(os.environ)
        if args.no_wandb:
            env["WANDB_MODE"] = "disabled"
        env.setdefault("WANDB_SILENT", "true")

        before = snapshot_runs()
        t0 = time.time()
        logf = os.path.join(os.path.dirname(args.out), f"train_{tag}.log")
        with open(logf, "w") as fh:
            rc = subprocess.call(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env)
        dt = time.time() - t0

        run_dir = newest_run_dir(before)
        ckpt = None
        if run_dir:
            cand = os.path.join(run_dir, "checkpoints", "best_agent.pt")
            ckpt = cand if os.path.exists(cand) else None
        status = "ok" if (rc == 0 and ckpt) else "FAILED"
        print(f"  -> {status} in {dt/60:.1f} min | {ckpt or run_dir or 'no run dir'}")
        if status != "ok":
            print(f"     see {logf}")

        index["runs"][tag] = {"status": status, "returncode": rc, "minutes": round(dt / 60, 2),
                              "run_dir": run_dir, "checkpoint": ckpt,
                              "factors": r["factors"], "overrides": r["overrides"], "log": logf}
        with open(args.out, "w") as f:
            json.dump(index, f, indent=2)

    done = sum(1 for v in index["runs"].values() if v.get("status") == "ok")
    print(f"\n[INFO] {done}/{len(index['runs'])} runs ok -> {args.out}")


if __name__ == "__main__":
    main()
