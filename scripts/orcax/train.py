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
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with Orcax PPO.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed (defaults to the YAML seed).")
parser.add_argument(
    "--timesteps", type=int, default=None, help="Total transitions, summed over envs (play: default 1000 steps per env)."
)
parser.add_argument("--wandb", action="store_true", default=False, help="Log to W&B (team entity).")
parser.add_argument("--play", type=str, default=None, metavar="CHECKPOINT", help="Replay a checkpoint instead of training.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # leave GPU memory to PhysX

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
from datetime import datetime

import distrax
import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp
from orcax.algorithms.single_agent.model_free.ppo import PPO, PPO_CFG
from orcax.trainers import SequentialTrainer, TrainerCfg
from orcax.utils import set_seed
from orcax.utils.experiments import ExperimentCfg
from orcax.wrappers.isaaclab import IsaacLabWrapper

import isaaclab_tasks  # noqa: F401
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


def main():
    cfg = load_cfg_from_registry(args_cli.task, "orcax_cfg_entry_point")
    seed = args_cli.seed if args_cli.seed is not None else cfg["seed"]
    set_seed(seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = seed
    env = gym.make(args_cli.task, cfg=env_cfg)
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
        timesteps=args_cli.timesteps or (cfg["trainer"]["timesteps"] if training else 1000 * num_envs),
        headless=args_cli.headless,
        close_environment_at_exit=False,  # the simulation app is closed below
        info_log_key="log",  # env extras["log"]: episode rewards, capture rate, ...
    )
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agent=agent)

    if training:
        print(f"[orcax] {args_cli.task}: {num_envs} envs, {trainer_cfg.timesteps} transitions, logs in {exp['directory']}")
        trainer.train()
    else:
        agent.load(args_cli.play)
        trainer.eval()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0)
