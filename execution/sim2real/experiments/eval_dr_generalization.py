"""Evaluate policies across MANY randomised bodies at once (DR generalization).

The per-flight evaluator (`eval_hover_vel_sim.py`) runs one drone per process, so with
physical DR it samples a single random body per flight -- far too few to say anything
about generalization. This script instead runs `--num-envs` bodies in parallel in one
env and aggregates per-env metrics, which is both statistically meaningful and ~50x
faster.

Protocol follows the adaptation literature: evaluate on a HELD-OUT range strictly
wider than anything seen in training, so the ranking reflects extrapolation rather
than memorisation.

    python execution/sim2real/experiments/eval_dr_generalization.py \
        --index execution/sim2real/experiments/sweep_index_v3.json --headless
"""

import argparse, json, os, sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--index", required=True)
p.add_argument("--num-envs", type=int, default=256, help="Randomised bodies per policy.")
p.add_argument("--duration", type=float, default=10.0)
p.add_argument("--out", default=None)
p.add_argument("--only", nargs="*", default=None)
p.add_argument("--dr-width", choices=["train", "held_out", "extreme"], default="held_out")
AppLauncher.add_app_launcher_args(p)
args, _ = p.parse_known_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

import torch
import gymnasium as gym
from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
import CrazyPlayGround.tasks  # noqa

# Widths. "train" = the widest TRAINING range in configs_v3; the others extrapolate.
WIDTHS = {
    "train":    dict(mass=(0.7, 1.4),  thrust=(0.80, 1.20)),
    "held_out": dict(mass=(0.6, 1.6),  thrust=(0.75, 1.25)),
    "extreme":  dict(mass=(0.5, 2.0),  thrust=(0.70, 1.30)),
}


def build_env(hist, act_hist, width):
    cfg = load_cfg_from_registry("Vel-Hovering-Robust", "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.duration + 5.0
    cfg.history_len = hist
    cfg.action_hist_len = act_hist
    cfg.privileged_state = False          # actor-only eval; keeps obs space clean
    # held-out condition: physical DR + measured latency + correlated noise + disturbance
    cfg.physical_dr = True
    w = WIDTHS[width]
    cfg.rand_mass_scale_range = w["mass"]
    cfg.rand_inertia_scale_range = w["mass"]
    cfg.rand_thrust_scale_range = w["thrust"]
    cfg.rand_torque_scale_range = w["thrust"]
    cfg.action_latency_steps = 4
    cfg.action_latency_max = 4
    cfg.add_noise = True
    cfg.noise_std = 0.02
    cfg.noise_corr = 0.9
    cfg.disturb = True
    cfg.disturb_gust_prob = 0.01
    env = gym.make("Vel-Hovering-Robust", cfg=cfg)
    return env, cfg


def evaluate(ckpt, hist, act_hist, width):
    env, cfg = build_env(hist, act_hist, width)
    u = env.unwrapped
    w = SkrlVecEnvWrapper(env)

    agent_cfg = load_cfg_from_registry("Vel-Hovering-Robust", "skrl_cfg_entry_point")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    agent_cfg["agent"]["experiment"]["wandb"] = False
    runner = Runner(w, agent_cfg)
    runner.agent.load(ckpt)
    runner.agent.enable_training_mode(False)

    obs, _ = w.reset()
    u.episode_length_buf[:] = 0
    # one fixed goal for every body, 1 m above its own origin
    target = u._terrain.env_origins.clone()
    target[:, 2] = 1.0
    u._desired_pos_w[:] = target

    n = cfg.scene.num_envs
    dev = u.device
    steps = int(round(args.duration / u.step_dt))
    err_sum = torch.zeros(n, device=dev)
    chat_sum = torch.zeros(n, device=dev)
    crashed = torch.zeros(n, dtype=torch.bool, device=dev)
    prev_a = None
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    counted = torch.zeros(n, device=dev)

    for k in range(steps):
        u._desired_pos_w[:] = target          # keep goals frozen through internal resets
        with torch.inference_mode():
            _, out = runner.agent.act(obs, None, timestep=0, timesteps=0)
            a = out["mean_actions"].clamp(-1.0, 1.0)
        pos = u._robot.data.root_pos_w - u._terrain.env_origins
        err = torch.linalg.norm(pos - (target - u._terrain.env_origins), dim=1)
        # A diverging body can produce non-finite state. Masking by multiplication is
        # NOT safe here (nan * 0 == nan), so sanitise then select with `where`.
        err = torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0).clamp(max=50.0)
        err_sum = err_sum + torch.where(alive, err, torch.zeros_like(err))
        if prev_a is not None:
            d = torch.nan_to_num(torch.linalg.norm(a - prev_a, dim=1), nan=0.0,
                                 posinf=0.0, neginf=0.0)
            chat_sum = chat_sum + torch.where(alive, d, torch.zeros_like(d))
        prev_a = torch.nan_to_num(a, nan=0.0).clone()
        counted += alive.float()
        obs, _, term, trunc, _ = w.step(a)
        newly = term.view(-1).bool() & alive
        crashed |= newly
        alive &= ~term.view(-1).bool()

    c = counted.clamp(min=1.0)
    per_env_err = torch.nan_to_num(err_sum / c, nan=0.0)
    per_env_chat = torch.nan_to_num(chat_sum / c, nan=0.0)
    w.close()
    return {
        "pos_err": float(per_env_err.mean()),
        "pos_err_p90": float(torch.quantile(per_env_err, 0.9)),
        "chatter": float(per_env_chat.mean()),
        "crash_rate": float(crashed.float().mean()),
        "bodies": n,
    }


def main():
    index = json.load(open(args.index))
    runs = {k: v for k, v in index["runs"].items() if v.get("checkpoint")}
    if args.only:
        runs = {k: v for k, v in runs.items() if k in args.only}
    print(f"[INFO] {len(runs)} policies x {args.num_envs} bodies, width={args.dr_width}")

    rows = []
    for i, (tag, info) in enumerate(sorted(runs.items())):
        f = info["factors"]
        try:
            m = evaluate(info["checkpoint"], int(f["history_len"]),
                         int(f.get("action_hist_len", 0)), args.dr_width)
        except Exception as e:
            print(f"[{i+1}/{len(runs)}] {tag:18s} FAILED: {e}")
            continue
        m.update(tag=tag, **{k: f[k] for k in ("physical_dr", "history_len", "action_hist_len")})
        rows.append(m)
        print(f"[{i+1}/{len(runs)}] {tag:18s} pos_err={m['pos_err']:.4f} "
              f"chatter={m['chatter']:.4f} crash={m['crash_rate']:.3f}")

    rows.sort(key=lambda r: r["pos_err"] + 5.0 * r["chatter"] + 10.0 * r["crash_rate"])
    print(f"\n=== DR GENERALIZATION ({args.dr_width}) — lower is better ===")
    h = f"{'#':>2} {'policy':18s} {'DR':>7} {'K':>2} {'M':>2} {'pos_err':>8} {'p90':>7} {'chatter':>8} {'crash':>6}"
    print(h); print("-" * len(h))
    for i, r in enumerate(rows):
        print(f"{i+1:>2} {r['tag']:18s} {str(r['physical_dr']):>7} {r['history_len']:>2} "
              f"{r['action_hist_len']:>2} {r['pos_err']:>8.4f} {r['pos_err_p90']:>7.4f} "
              f"{r['chatter']:>8.4f} {r['crash_rate']:>6.3f}")

    out = args.out or os.path.join(os.path.dirname(args.index), f"dr_generalization_{args.dr_width}.json")
    json.dump({"width": args.dr_width, "condition": WIDTHS[args.dr_width],
               "num_envs": args.num_envs, "rows": rows}, open(out, "w"), indent=2)
    print(f"\n[INFO] -> {out}")


if __name__ == "__main__":
    main()
    app.close()
