"""Checks for Rate-Hovering: config validation, the plain-task defaults, and tensor logging.

Run (conda env DragonFly, from CrazyPlayGround/):
    python -m pytest source/CrazyPlayGround/test/test_rate_hovering.py -v
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402

import omni.usd  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms  # noqa: E402

from CrazyPlayGround.tasks.direct.hovering.rate_hovering import QuadcopterEnv, QuadcopterEnvCfg  # noqa: E402

N = 4


def make_env(num_envs: int = N, **knobs) -> QuadcopterEnv:
    omni.usd.get_context().new_stage()
    cfg = QuadcopterEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.debug_vis = False
    for key, value in knobs.items():  # set after the cfg is built, as Hydra does
        setattr(cfg, key, value)
    env = QuadcopterEnv(cfg)
    # Drones spawn up to 2.0 m, the env's kill height, so a command can reset an env
    # mid-test. Keep only the timeout.
    env._get_dones = lambda: (
        torch.zeros_like(env.reset_terminated), env.episode_length_buf >= env.max_episode_length - 1
    )
    return env


def frame(env) -> torch.Tensor:
    """Clean 18-D observation [v_b, goal_b, R(9), w_b] recomputed from the sim state."""
    data = env._robot.data
    goal_b, _ = subtract_frame_transforms(data.root_pos_w, data.root_quat_w, env._desired_pos_w)
    rot = matrix_from_quat(data.root_quat_w).reshape(env.num_envs, 9)
    return torch.cat([data.root_lin_vel_b, goal_b, rot, data.root_ang_vel_b], dim=-1)


@pytest.mark.parametrize(
    "knobs",
    [
        {"max_body_rate": 0.0},
        {"min_thrust_scale": 0.0},
        {"min_thrust_scale": 1.1},
        {"max_thrust_scale": 0.9},
        {"motor_tau_range": (0.09, 0.05)},
        {"motor_tau_range": (-0.01, 0.05)},
        {"history_fill_value": 0.0},
        {"history_fill_value": -1.0},
        {"history_fill_value": 1.0},
    ],
)
def test_invalid_knobs_raise(knobs):
    with pytest.raises(ValueError):
        make_env(**knobs)


def test_defaults_are_plain_hovering():
    cfg = QuadcopterEnvCfg()
    assert not cfg.add_noise
    assert not cfg.domain_rand
    assert cfg.history_fill_value == -10.0
    env = make_env()
    try:
        assert env.single_observation_space["policy"].shape == (18,)
        obs, _ = env.reset()
        torch.testing.assert_close(obs["policy"], frame(env))
        obs, *_ = env.step(torch.full((N, 4), -0.2, device=env.device))
        torch.testing.assert_close(obs["policy"], frame(env))
        assert torch.all(env._dr_thrust_scale == 1.0)
        assert torch.all(env._dr_thrust_bias_n == 0.0)
        assert torch.all(env._dr_rate_scale == 1.0)
    finally:
        env.close()


def test_log_values_are_tensors():
    """skrl logs only one-element tensors and silently drops floats, so every log entry is a tensor."""
    env = make_env()
    try:
        env.reset()
        log = env.extras["log"]
        assert {"Episode_Termination/died", "Episode_Termination/time_out", "Metrics/final_distance_to_goal"} <= log.keys()
        for key, value in log.items():
            assert isinstance(value, torch.Tensor) and value.numel() == 1, key
    finally:
        env.close()
