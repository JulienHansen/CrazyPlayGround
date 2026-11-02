"""skrl_args.py must hand train.py exactly the settings the sweep file gives.

A wrong mapping trains something other than what the run_key says, and the
W&B tags decide whether wandb_check.py sees the run as done.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from parser import SweepError, expand  # noqa: E402
from skrl_args import GROUP_MAX, run_args, wandb_group  # noqa: E402


def sweep(**defaults):
    base = {
        "trainer": "skrl", "env_name": "Vel-Hovering", "seed": [42], "num_envs": 4096,
        "total_timesteps": 78643200, "lr": 5e-4, "entropy_coef": 0.0,
        "env_kwargs": {"add_noise": True, "wind_force_range": [0.0, 0.05]},
    }
    return expand({
        "version": 1,
        "wandb": {"project": "hovering"},
        "defaults": {**base, **defaults},
        "experiments": [{"name": "scan", "sweep": {"lambda_coef": [0.9]}}],
    })


def test_settings_map_to_train_py_args():
    s = sweep()
    run = s.runs[0]
    args = run_args(run, s)
    for expected in (
        "--task=Vel-Hovering", "--seed=42", "--num_envs=4096",
        "agent.agent.learning_rate=0.0005", "agent.agent.lambda=0.9",
        "agent.agent.entropy_loss_scale=0.0",  # stays a float for Isaac Lab / skrl
        "agent.trainer.timesteps=19200",  # per env: 78643200 / 4096
        "env.add_noise=true", "env.wind_force_range=[0.0,0.05]",
        "agent.agent.experiment.wandb=True",
        f"agent.agent.experiment.experiment_name={run.run_key}",
        '++agent.agent.experiment.wandb_kwargs.entity="crazyplayground"',
        '++agent.agent.experiment.wandb_kwargs.name="scan/lambda_coef=0.9/s42"',
        f'++agent.agent.experiment.wandb_kwargs.group="{run.group}"',
        f'++agent.agent.experiment.wandb_kwargs.config.run_key="{run.run_key}"',
        "++agent.agent.experiment.wandb_kwargs.config.exp.env_kwargs.add_noise=true",
    ):
        assert expected in args


def test_unknown_key_is_refused():
    s = sweep(gamma=0.99)
    with pytest.raises(SweepError, match="gamma"):
        run_args(s.runs[0], s)


def test_timesteps_round_up_to_whole_steps_per_env():
    s = sweep(total_timesteps=10e6)  # 2441.4 steps per env
    assert "agent.trainer.timesteps=2442" in run_args(s.runs[0], s)


def test_timesteps_need_num_envs():
    s = sweep()
    del s.runs[0].config["num_envs"]
    with pytest.raises(SweepError, match="needs num_envs"):
        run_args(s.runs[0], s)


def test_long_wandb_groups_are_shortened_to_the_limit():
    s = expand({
        "version": 1,
        "wandb": {"project": "hovering"},
        "defaults": {"trainer": "skrl", "env_name": "Vel-Hovering", "seed": [42], "num_envs": 8192},
        "experiments": [{"name": "shape-x-ppo", "sweep": {
            "env_kwargs.distance_reward_shape": ["tanh", "multiexp"],
            "lr": [0.00025], "lambda_coef": [0.95], "entropy_coef": [0.0001], "rollout_length": [32, 64],
        }}],
    })
    assert any(len(run.group) > GROUP_MAX for run in s.runs)  # the case this test is for
    for run in s.runs:
        group = wandb_group(run)
        assert len(group) <= GROUP_MAX
        assert group.endswith(f" [{run.config_key}]")
        assert f'.group="{group}"' in " ".join(run_args(run, s))
        if len(run.group) > GROUP_MAX:
            assert "env_kwargs." not in group and "distance_reward_shape=" in group
            assert "lr=0.00025" in group  # values keep their dots


def test_short_wandb_groups_are_unchanged():
    s = sweep()
    assert wandb_group(s.runs[0]) == s.runs[0].group
