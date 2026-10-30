"""Train (or replay) a PPO agent with Orcax on a CrazyPlayGround task.

Orcax is a JAX library; the Isaac Lab env stays in torch and Orcax's
``IsaacLabWrapper`` moves tensors across with DLPack. If the env's
``_get_observations`` returns a ``"critic"`` group, it becomes Orcax's
privileged state and the value network reads it (asymmetric actor-critic).

Hyperparameters come from the task's ``orcax_cfg_entry_point`` YAML.

Example (Isaac Lab venv plus a JAX overlay)::

    XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    PYTHONPATH=~/.venvs/isaac_orcax_overlay:~/Orcax/Orcax \\
    ~/.venvs/isaac/bin/python scripts/orcax/train.py --task Att-PursuitEvasion --num_envs 4096 --headless

    # replay a checkpoint
    ... scripts/orcax/train.py --task Att-PursuitEvasion --num_envs 16 --play logs/orcax/.../checkpoints/best.pkl
"""

import argparse
import ast
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with Orcax PPO.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed (defaults to the YAML seed).")
parser.add_argument(
    "--timesteps", type=int, default=None, help="Training: total transitions, summed over envs. Play: env steps (default 1000)."
)
parser.add_argument("--wandb", action="store_true", default=False, help="Log to W&B (team entity).")
parser.add_argument(
    "--env",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Set an env config field, e.g. --env curriculum_start_level=0.2 (repeatable; VALUE is a Python literal).",
)
parser.add_argument("--play", type=str, default=None, metavar="CHECKPOINT", help="Replay a checkpoint instead of training.")
parser.add_argument(
    "--init_from", type=str, default=None, metavar="CHECKPOINT", help="Start training from these weights. Needs the same obs/action layout and the same --num_envs as the run that saved them (with a different env count the first step hangs)."
)
parser.add_argument("--video", action="store_true", default=False, help="With --play: record a video of env 0.")
parser.add_argument("--video_length", type=int, default=1500, help="Video length in policy steps.")
parser.add_argument(
    "--stats", action="store_true", default=False, help="With --play: print episode outcome totals over all envs."
)
parser.add_argument(
    "--fpv", action="store_true", default=False, help="With --play: record env 0's onboard camera, with the detection box."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.video or args_cli.fpv:
    if args_cli.play is None:
        parser.error("--video and --fpv need --play CHECKPOINT")
    args_cli.enable_cameras = True

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # leave GPU memory to PhysX

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
from datetime import datetime

import cv2
import distrax
import torch
import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp
from orcax.algorithms.single_agent.model_free.ppo import PPO, PPO_CFG
from orcax.trainers import SequentialTrainer, TrainerCfg
from orcax.utils import set_seed
from orcax.utils.experiments import ExperimentCfg
from orcax.wrappers.isaaclab import IsaacLabWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ViewerCfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import CrazyPlayGround.tasks  # noqa: F401

_ACTIVATIONS = {"elu": nn.elu, "tanh": nn.tanh, "relu": nn.relu}


def _orth(gain: float):
    return nn.initializers.orthogonal(scale=gain)


class GaussianPolicy(nn.Module):
    """MLP mean with a state-independent log-std; actions are clamped by the env."""

    action_dim: int
    hidden: tuple[int, ...]
    activation: str = "elu"
    log_std_init: float = 0.0

    @nn.compact
    def __call__(self, obs):
        x = obs
        for width in self.hidden:
            x = _ACTIVATIONS[self.activation](nn.Dense(width, kernel_init=_orth(math.sqrt(2)))(x))
        mean = nn.Dense(self.action_dim, kernel_init=_orth(0.01))(x)
        log_std = self.param("log_std", nn.initializers.constant(self.log_std_init), (self.action_dim,))
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))


class ValueNet(nn.Module):
    hidden: tuple[int, ...]
    activation: str = "elu"

    @nn.compact
    def __call__(self, x):
        for width in self.hidden:
            x = _ACTIVATIONS[self.activation](nn.Dense(width, kernel_init=_orth(math.sqrt(2)))(x))
        return nn.Dense(1, kernel_init=_orth(1.0))(x).squeeze(-1)


class EpisodeStats(gym.Wrapper):
    """Play-only: sum the env's per-reset termination counts (``extras["log"]``) and print them at close."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._totals: dict[str, float] = {}

    def step(self, action):
        out = self.env.step(action)
        for key, value in self.env.unwrapped.extras.get("log", {}).items():
            if key.startswith("Episode_Termination/"):
                self._totals[key] = self._totals.get(key, 0.0) + float(value)
        return out

    def close(self):
        episodes = sum(v for k, v in self._totals.items() if k.split("/")[1] in ("capture", "crash", "time_out"))
        print(f"[orcax] episodes finished: {episodes:.0f}", flush=True)
        for key, value in sorted(self._totals.items()):
            print(f"[orcax]   {key.split('/')[1]:14s} {value:6.0f}  ({value / max(episodes, 1):.1%})", flush=True)
        return self.env.close()


class FpvRecorder(gym.Wrapper):
    """Play-only: write env 0's onboard camera to an mp4, with the ideal detection drawn on it.

    The green box is what the policy receives (``visible``, ``u``, ``v``, ``size``); a frame
    without a box means the detector reports the evader as not visible.
    """

    def __init__(self, env: gym.Env, path: str, num_frames: int, fps: float):
        super().__init__(env)
        self._path, self._num_frames, self._fps = path, num_frames, fps
        self._writer = None
        self._frames = 0

    def _write(self):
        if self._frames >= self._num_frames:
            return
        env = self.env.unwrapped
        rgb = env.scene["fpv_camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]
        if self._writer is None:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            self._writer = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h))
        det = {k: v[0].item() for k, v in env._detect().items()}
        if det["visible"] > 0:
            cx, cy = w / 2 * (1 + det["u"]), h / 2 * (1 - det["v"])
            half = max(3.0, det["size"] * w / 2)
            cv2.rectangle(frame, (int(cx - half), int(cy - half)), (int(cx + half), int(cy + half)), (0, 255, 0), 2)
            growth = env._log_growth[0].item()  # box log-growth over the last step
            label = f"TTC {env.step_dt / growth:.2f} s" if growth > 1e-4 else "not closing"
        else:
            label = "target not visible"
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        captures = int(env.capture_count[0])
        cv2.putText(frame, f"captures: {captures}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self._writer.write(frame)
        self._frames += 1

    def step(self, action):
        out = self.env.step(action)
        self._write()
        return out

    def _to_h264(self):
        """Re-encode OpenCV's MPEG-4 Part 2 output to H.264, which common players open."""
        try:
            import subprocess

            import imageio_ffmpeg
        except ImportError:
            return
        tmp = self._path + ".h264.mp4"
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", self._path, "-c:v", "libx264"]
        cmd += ["-profile:v", "main", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", "-an", tmp]
        if subprocess.run(cmd).returncode == 0:
            os.replace(tmp, self._path)

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._to_h264()
            print(f"[orcax] onboard video: {self._path} ({self._frames} frames)", flush=True)
            self._writer = None
        return self.env.close()


class PairCamera(gym.Wrapper):
    """Play-only: aim the viewport at the midpoint of env 0's two drones every step.

    Pulls back as the drones separate, and prints the video time of each capture in env 0.
    """

    def __init__(self, env: gym.Env, video_fps: float):
        super().__init__(env)
        self._video_fps = video_fps
        self._steps = 0
        self._captures = 0

    def _aim(self):
        env = self.env.unwrapped
        p = env.scene["robot"].data.root_pos_w[0]
        e = env.scene["evader"].data.root_pos_w[0]
        mid = 0.5 * (p + e)
        back = torch.tensor([-1.0, -1.0, 0.55], device=mid.device)
        back = back / back.norm() * (1.2 + 1.2 * torch.linalg.norm(p - e).clamp(max=4.0))
        env.sim.set_camera_view((mid + back).tolist(), mid.tolist())

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._aim()
        return out

    def step(self, action):
        out = self.env.step(action)
        self._steps += 1
        captures = int(getattr(self.env.unwrapped, "capture_count", torch.zeros(1))[0])
        if captures > self._captures:
            self._captures = captures
            print(f"[orcax] env 0 capture #{captures} at t = {self._steps / self._video_fps:.2f} s", flush=True)
        self._aim()
        return out


def main():
    cfg = load_cfg_from_registry(args_cli.task, "orcax_cfg_entry_point")
    seed = args_cli.seed if args_cli.seed is not None else cfg["seed"]
    set_seed(seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = seed
    for item in args_cli.env:
        key, _, value = item.partition("=")
        if not hasattr(env_cfg, key):
            raise ValueError(f"--env {item}: the env config has no field {key!r}")
        setattr(env_cfg, key, ast.literal_eval(value))
    if args_cli.video:
        # Static viewer origin: PairCamera re-aims it every step. Markers on both drones.
        env_cfg.viewer = ViewerCfg(origin_type="world")
        if hasattr(env_cfg, "debug_vis"):
            env_cfg.debug_vis = True
    if args_cli.play is not None and hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = False  # evaluate and record at the final task difficulty
    if args_cli.fpv:
        from CrazyPlayGround.tasks.direct.pursuit_evasion.pursuit_evasion_env import make_fpv_camera_cfg

        env_cfg.fpv_camera = make_fpv_camera_cfg(env_cfg.detector)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    play_dir = os.path.dirname(os.path.dirname(os.path.abspath(args_cli.play))) if args_cli.play else None
    if args_cli.stats:
        env = EpisodeStats(env)
    if args_cli.fpv:
        env = FpvRecorder(
            env, os.path.join(play_dir, "videos", "fpv.mp4"), args_cli.video_length, 1.0 / env.unwrapped.step_dt
        )
    if args_cli.video:
        env = PairCamera(env, video_fps=1.0 / env.unwrapped.step_dt)
        video_dir = os.path.join(play_dir, "videos")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[orcax] recording {args_cli.video_length} steps to {video_dir}")
    env = IsaacLabWrapper(env, seed=seed)
    num_envs = env.num_envs

    action_dim = int(env.action_space.shape[-1])
    pol, val = cfg["models"]["policy"], cfg["models"]["value"]
    models = {
        "policy": GaussianPolicy(
            action_dim=action_dim,
            hidden=tuple(pol["hidden"]),
            activation=pol["activation"],
            log_std_init=pol["log_std_init"],
        ),
        "value": ValueNet(hidden=tuple(val["hidden"]), activation=val["activation"]),
    }

    exp = cfg["experiment"]
    training = args_cli.play is None
    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.task}_s{seed}"
    experiment_cfg = ExperimentCfg(
        directory=os.path.abspath(exp["directory"]),
        experiment_name=run_name,
        # ExperimentCfg intervals count transitions summed over envs.
        write_interval=cfg["trainer"]["write_interval"] * num_envs,
        checkpoint_interval=cfg["trainer"]["checkpoint_interval"] * num_envs,
        tensorboard=training,
        wandb=training and (args_cli.wandb or exp["wandb"]),
        wandb_kwargs={**exp["wandb_kwargs"], "name": run_name},
    )

    agent_kwargs = dict(cfg["agent"])
    if "critic_inputs" in agent_kwargs and not isinstance(agent_kwargs["critic_inputs"], str):
        agent_kwargs["critic_inputs"] = tuple(agent_kwargs["critic_inputs"])
    if not training:
        # Orcax always creates an experiment dir; keep play runs under the gitignored logs/.
        experiment_cfg = ExperimentCfg(
            directory=os.path.abspath(exp["directory"]), experiment_name=f"play_{run_name}", tensorboard=False
        )
    agent_cfg = PPO_CFG(**agent_kwargs, seed=seed, experiment=experiment_cfg)
    agent = PPO(
        models=models,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        state_space=env.state_space,
        device=env.device,
    )

    trainer_cfg = TrainerCfg(
        timesteps=args_cli.timesteps
        or (cfg["trainer"]["timesteps"] if training else (args_cli.video_length if (args_cli.video or args_cli.fpv) else 1000)),  # play: eval counts env steps
        headless=args_cli.headless,
        close_environment_at_exit=False,  # the simulation app is closed below
        info_log_key="log",  # env extras["log"]: episode rewards, capture rate, ...
    )
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agent=agent)

    if training and args_cli.init_from:
        # A SequentialTrainer keeps a loaded agent's state (it does not re-initialise it).
        agent.load(args_cli.init_from)
        print(f"[orcax] initialised from {args_cli.init_from}")
    if training:
        print(f"[orcax] {args_cli.task}: {num_envs} envs, {trainer_cfg.timesteps} transitions, logs in {exp['directory']}")
        trainer.train()
    else:
        agent.load(args_cli.play)
        trainer.eval()

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Always close Isaac Sim: an exception left open makes the process hang instead of exiting.
        simulation_app.close()
    sys.exit(0)
