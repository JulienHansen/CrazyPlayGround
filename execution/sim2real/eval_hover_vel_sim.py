"""Run a velocity hover policy IN SIM and record the same CSV schema as the real drone.

This is the sim side of the CL-1 hover gap: it flies `best_agent.pt` in IsaacLab on
a *frozen* target, records one row per 100 Hz control step using the exact
``CSV_COLUMNS`` of ``collect_hover_vel.py``, and writes a run directory that
``analyze_hover.py`` can consume unmodified. Comparing the resulting metrics with
the real-flight metrics gives the sim-to-real gap.

Usage (from the repo root, with the Isaac Lab python env):
    python execution/sim2real/eval_hover_vel_sim.py \
        --checkpoint logs/skrl/.../checkpoints/best_agent.pt \
        --target 0 0 1.0 --duration 10 --tag sim_0_0_1 --headless

Differences vs the real drone (documented in metadata.json, keep in mind when
comparing):
  * No motor PWM, battery, accelerometer or Kalman variance exist in sim -> those
    columns are written as 0.0 (analyze_hover reports them as 0 / None).
  * The sim starts from a commanded hover pose (default z=0.5 m, like the real
    takeoff) instead of the real takeoff transient.
  * gyro_* is converted from sim rad/s to deg/s to match the firmware log units.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# ── CLI (parsed before Isaac starts, like scripts/skrl/play.py) ──────────────
parser = argparse.ArgumentParser(description="Evaluate a velocity hover policy in sim, logging the real-drone CSV schema.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the skrl checkpoint (.pt)")
parser.add_argument("--task", type=str, default="Vel-Hovering", help="Task id to evaluate.")
parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                    help="Frozen goal [x y z], relative to the env origin.")
parser.add_argument("--start-z", type=float, default=0.5,
                    help="Initial hover height, mimicking the real takeoff (m).")
parser.add_argument("--duration", type=float, default=10.0, help="Flight duration in seconds.")
parser.add_argument("--outdir", type=str, default=None, help="Root output dir (default: execution/sim2real/data_sim).")
parser.add_argument("--tag", type=str, default="", help="Label appended to the run folder name.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--history-len", type=int, default=1,
                    help="Observation window length; must match the trained policy (1 = no stacking).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import csv
import json
import time
import math
from datetime import datetime

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import CrazyPlayGround.tasks  # noqa: F401

# reuse the exact schema + metadata helpers the real collector writes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_hover_vel import CSV_COLUMNS, _sha256, _git_commit, _versions  # noqa: E402

RAD2DEG = 180.0 / math.pi


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    # ── single env, episode long enough that no timeout reset happens mid-run ──
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = args_cli.duration + 5.0
    env_cfg.seed = args_cli.seed
    agent_cfg["seed"] = args_cli.seed

    max_velocity = float(getattr(env_cfg, "max_velocity", 1.0))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env)

    # skrl runner, but never log/checkpoint from an eval run
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    agent_cfg["agent"]["experiment"]["wandb"] = False
    runner = Runner(env, agent_cfg)

    ckpt = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] loading checkpoint: {ckpt}")
    runner.agent.load(ckpt)
    runner.agent.enable_training_mode(False)   # skrl 2.1 API (play.py uses the removed set_running_mode)

    uenv = env.unwrapped
    device = uenv.device
    dt = uenv.step_dt                     # 0.01 s -> 100 Hz
    n_steps = int(round(args_cli.duration / dt))

    env.reset()

    # ── freeze the goal and place the drone at a hover start pose ─────────────
    origin = uenv._terrain.env_origins[0]
    target_w = origin + torch.tensor(args_cli.target, dtype=torch.float32, device=device)
    uenv._desired_pos_w[:] = target_w

    root_state = uenv._robot.data.default_root_state.clone()
    root_state[:, :3] = origin.unsqueeze(0)
    root_state[:, 2] = args_cli.start_z
    uenv._robot.write_root_pose_to_sim(root_state[:, :7])
    uenv._robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    uenv._ctrl.reset(None)
    # _reset_idx randomizes episode_length_buf to de-correlate training envs; zero it
    # so the recording is not truncated at a random point.
    uenv.episode_length_buf[:] = 0

    obs = uenv._get_observations()["policy"]
    hist = [obs.clone() for _ in range(args_cli.history_len)]

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_velsim"
    if args_cli.tag:
        run_name += f"_{args_cli.tag}"
    outroot = args_cli.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_sim")
    run_dir = os.path.join(outroot, run_name)
    os.makedirs(run_dir, exist_ok=True)

    records = []
    crashed = False
    t0 = time.time()
    for step in range(n_steps):
        # keep the goal frozen even if the env resampled it on an internal reset
        uenv._desired_pos_w[:] = target_w

        policy_in = torch.cat(hist, dim=-1) if args_cli.history_len > 1 else obs
        with torch.inference_mode():
            _, outputs = runner.agent.act(policy_in, None, timestep=0, timesteps=0)
            action = outputs["mean_actions"].clamp(-1.0, 1.0)

        # snapshot true sim state BEFORE stepping (matches "state that produced the action")
        pos = uenv._robot.data.root_pos_w[0] - origin       # report relative to env origin
        vel = uenv._robot.data.root_lin_vel_w[0]
        quat = uenv._robot.data.root_quat_w[0]              # (w, x, y, z)
        gyro = uenv._robot.data.root_ang_vel_b[0] * RAD2DEG  # deg/s, matching firmware log units
        o = policy_in[0] if args_cli.history_len == 1 else obs[0]
        a = action[0]
        cmd = a * max_velocity
        tgt = torch.tensor(args_cli.target, device=device)

        records.append({
            "t_mono": step * dt, "t_wall": 0.0, "step": step,
            "obs_vb_x": o[0].item(), "obs_vb_y": o[1].item(), "obs_vb_z": o[2].item(),
            "obs_errb_x": o[3].item(), "obs_errb_y": o[4].item(), "obs_errb_z": o[5].item(),
            "act_x": a[0].item(), "act_y": a[1].item(), "act_z": a[2].item(),
            "cmd_vx": cmd[0].item(), "cmd_vy": cmd[1].item(), "cmd_vz": cmd[2].item(), "cmd_yawrate": 0.0,
            "tgt_x": tgt[0].item(), "tgt_y": tgt[1].item(), "tgt_z": tgt[2].item(),
            "pos_x": pos[0].item(), "pos_y": pos[1].item(), "pos_z": pos[2].item(),
            "vel_x": vel[0].item(), "vel_y": vel[1].item(), "vel_z": vel[2].item(),
            "qw": quat[0].item(), "qx": quat[1].item(), "qy": quat[2].item(), "qz": quat[3].item(),
            "gyro_x": gyro[0].item(), "gyro_y": gyro[1].item(), "gyro_z": gyro[2].item(),
            "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0,      # not modelled in sim
            "m1": 0.0, "m2": 0.0, "m3": 0.0, "m4": 0.0,     # no PWM in sim (controller emits N / N.m)
            "vbat": 0.0,                                     # no battery model
            "varPX": 0.0, "varPY": 0.0, "varPZ": 0.0,        # true state, no estimator
            "dist_to_target": torch.linalg.norm(pos - tgt).item(),
        })

        obs, _, terminated, truncated, _ = env.step(action)
        if bool(terminated[0].item()):
            print(f"[WARN] env terminated at step {step} (z out of bounds) — stopping early")
            crashed = True
            break
        if bool(truncated[0].item()):
            print(f"[WARN] env truncated at step {step}")
            break

        hist.append(obs.clone())
        hist.pop(0)

    # ── write CSV / npz / metadata in the analyze_hover.py layout ─────────────
    csv_path = os.path.join(run_dir, "flight.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(records)
    print(f"[INFO] wrote {len(records)} rows -> {csv_path}")

    try:
        import numpy as np
        np.savez(os.path.join(run_dir, "flight.npz"),
                 **{c: np.array([r[c] for r in records], dtype=np.float64) for c in CSV_COLUMNS})
    except Exception as e:
        print(f"[WARN] npz not written: {e}")

    meta = {
        "source": "sim",
        "controller": "velocity",
        "task": args_cli.task,
        "checkpoint_path": ckpt,
        "checkpoint_sha256": _sha256(ckpt),
        "git_commit": _git_commit(),
        "seed": args_cli.seed,
        "tag": args_cli.tag,
        "target": list(args_cli.target),
        "start_z": args_cli.start_z,
        "duration_s": args_cli.duration,
        "control_rate_hz": round(1.0 / dt, 3),
        "sim_dt": env_cfg.sim.dt,
        "decimation": env_cfg.decimation,
        "max_velocity_mps": max_velocity,
        "history_len": args_cli.history_len,
        "add_noise": bool(getattr(env_cfg, "add_noise", False)),
        "noise_std": float(getattr(env_cfg, "noise_std", 0.0)),
        "num_samples": len(records),
        "crashed": crashed,
        "unavailable_columns": ["acc_*", "m1..m4", "vbat", "varP*", "t_wall"],
        "gyro_units": "deg/s (converted from sim rad/s)",
        "quaternion_order": "wxyz (scalar first)",
        "versions": _versions(),
        "wall_time_s": round(time.time() - t0, 2),
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[INFO] run dir: {run_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
