# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--eval_csv",
    type=str,
    default=None,
    help=(
        "Evaluate instead of playing: fly one full episode per env and write its step metrics to this CSV."
        " Envs with record_states (the tracking envs) also get every step in a .parquet next to it."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import json
import gymnasium as gym
import os
import random
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    get_published_pretrained_checkpoint = None

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import CrazyPlayGround.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            print("[ERROR] --use_pretrained_checkpoint requires isaaclab.utils.pretrained_checkpoint, "
                  "which is unavailable in this Isaac Lab installation.")
            return
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir
    # evaluation records every step where the env can (record_states), next to the CSV
    parquet_path = None
    if args_cli.eval_csv and hasattr(env_cfg, "record_states"):
        env_cfg.record_states = True
        parquet_path = os.path.splitext(args_cli.eval_csv)[0] + ".parquet"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    experiment_cfg["agent"]["experiment"]["wandb"] = False  # Runner would otherwise open an empty W&B run
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.enable_training_mode(False, apply_to_models=True)  # skrl 2.x name of set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    if args_cli.eval_csv:
        evaluate(env, runner, obs, args_cli.eval_csv, parquet_path, resume_path, args_cli.task)
        env.close()
        return
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)  # skrl 2.x: states is positional
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


def evaluate(env, runner, obs, csv_path: str, parquet_path: str | None = None, checkpoint: str = "",
             task: str = ""):
    """Fly one full-length episode per env with mean actions and write its per-episode metrics to a CSV.

    Needs an env that fills ``extras["step_metrics"]`` at reset: Vel- or Rate-Hovering (step response) or
    the tracking envs (tracking error, effort). The spawn, goal and trajectory draws follow
    ``--seed``, so two checkpoints evaluated with the same seed and ``--num_envs`` fly the same
    episodes.

    With ``parquet_path``, the env must also fill ``extras["data_record"]`` every step
    (``record_states``): each env's steps are kept until its first episode ends (the auto-reset then
    starts episodes that are not recorded) and written by ``write_parquet``.
    """
    base_env = env.unwrapped
    base_env.episode_length_buf[:] = 0  # undo the randomised first reset: every episode lasts the full time
    n_envs, n_steps, dev = base_env.num_envs, int(base_env.max_episode_length), base_env.device
    rows = {}
    states = {}  # key -> (N, T, ...): env n's step t in [n, t - 1], until its first episode ends
    length = torch.zeros(n_envs, dtype=torch.long, device=dev)  # steps of each env's first episode
    died = torch.zeros(n_envs, dtype=torch.bool, device=dev)  # it ended by a crash, not a time-out
    finished = torch.zeros(n_envs, dtype=torch.bool, device=dev)
    for _ in range(base_env.max_episode_length):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
            obs, _, terminated, truncated, info = env.step(outputs[-1].get("mean_actions", outputs[0]))
            if parquet_path:
                record = info["data_record"]  # this step, taken before the auto-reset
                active = torch.nonzero(~finished, as_tuple=True)[0]
                slot = record["step"][active] - 1
                for key, value in record.items():
                    if key != "step":
                        if key not in states:
                            states[key] = torch.zeros((n_envs, n_steps, *value.shape[1:]), dtype=value.dtype,
                                                      device=dev)
                        states[key][active, slot] = value[active]
                ended = (terminated | truncated).view(-1) & ~finished
                length[ended] = record["step"][ended]
                died[ended] = terminated.view(-1)[ended]
                finished |= ended
        if not (terminated | truncated).any():
            continue
        metrics = {k: v.cpu() for k, v in info["step_metrics"].items()}
        # finished envs auto-reset into new random episodes: keep only each env's first one
        for i, env_id in enumerate(metrics.pop("env_ids").tolist()):
            if env_id in rows:
                continue
            rows[env_id] = {"env_id": env_id}
            for key, value in metrics.items():
                if value.ndim == 2:  # (E, 3) positions
                    rows[env_id].update({f"{key}_{axis}": x for axis, x in zip("xyz", value[i].tolist())})
                else:
                    rows[env_id][key] = value[i].item()
        if len(rows) == base_env.num_envs:
            break

    if parquet_path:
        write_parquet(parquet_path, base_env, states, length, died, checkpoint, task)
    rows = [rows[i] for i in sorted(rows)]
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote {len(rows)} episodes to {csv_path}")
    for key in rows[0]:
        if key != "env_id" and not key.endswith(("_x", "_y", "_z")):
            values = torch.tensor([float(r[key]) for r in rows])
            print(f"  {key:>14}: mean {values.nanmean().item():.4f}  ({int(values.isnan().sum())} NaN)")


def write_parquet(path: str, base_env, states: dict, length: torch.Tensor, died: torch.Tensor, checkpoint: str,
                  task: str):
    """One row per (episode, step), as the flight recordings; episode n = env n, ``length[n]`` steps.

    Columns: ``pos_*``, ``target_*`` (reference), ``vel_*``, ``action_*`` and, in velocity mode, the
    setpoint ``vx/vy/vz`` [m/s]; other recorded values (DR draws) under their own names. Footer:
    ``source``, ``task``, ``checkpoint``, ``dt``, ``env_cfg`` and ``died`` (JSON).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_envs, n_steps = length.shape[0], next(iter(states.values())).shape[1]
    valid = torch.arange(n_steps, device=length.device)[None, :] < length[:, None]  # each episode's steps
    columns = {
        "episode": torch.arange(n_envs, device=length.device)[:, None].expand(-1, n_steps)[valid],
        "step": torch.arange(1, n_steps + 1, device=length.device)[None, :].expand(n_envs, -1)[valid],
    }
    names = {"pos": "pos", "ref": "target", "vel_w": "vel"}
    velocity = getattr(base_env.cfg, "control_mode", "") == "velocity"
    for key, value in states.items():
        value = value[valid]  # (rows, ...)
        if key == "action_cmd":
            for k in range(value.shape[1]):
                columns[f"action_{('vx', 'vy', 'vz')[k] if velocity else k}"] = value[:, k]
        elif key == "action_sent":
            for k in range(value.shape[1]):
                if velocity:
                    columns[("vx", "vy", "vz")[k]] = value[:, k] * base_env.cfg.max_velocity
                else:
                    columns[f"action_sent_{k}"] = value[:, k]
        elif value.ndim == 2 and value.shape[1] == 3:
            columns.update({f"{names.get(key, key)}_{axis}": value[:, k] for k, axis in enumerate("xyz")})
        else:
            columns[names.get(key, key)] = value
    table = pa.table({name: value.cpu().numpy() for name, value in columns.items()})
    footer = {
        "source": "sim", "task": task, "checkpoint": checkpoint, "dt": repr(base_env.step_dt),
        "env_cfg": json.dumps(base_env.cfg.to_dict(), default=repr), "died": json.dumps(died.tolist()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pq.write_table(table.replace_schema_metadata(footer), path, compression="zstd")
    print(f"[INFO] Wrote {n_envs} episodes, {table.num_rows} steps to {path}")

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
