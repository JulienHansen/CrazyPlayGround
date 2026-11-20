"""Checks for Rate-Hovering: config validation, the plain-task defaults, tensor logging, the
distance reward shapes and the distance-length curriculum.

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


def rand_cmd(env):
    """A different command for every env and axis, inside the clamp range."""
    return torch.empty(N, 4, device=env.device).uniform_(-1.0, 1.0)


def step_terms(env, cmd):
    """Step once and return (reward, {key: how much that episode sum grew this step})."""
    before = {k: v.clone() for k, v in env._episode_sums.items()}
    _, reward, *_ = env.step(cmd)
    return reward, {k: env._episode_sums[k] - before[k] for k in before}


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
        {"distance_reward_shape": "gauss"},
        {"distance_reward_shape": "exp"},  # a single exponential is multiexp with scales (1.0,)
        {"distance_reward_length": 0.0},
        {"distance_reward_shape": "multiexp", "distance_reward_scales": ()},
        {"distance_reward_shape": "multiexp", "distance_reward_scales": (1.0, 0.0)},
        {"distance_reward_curriculum_ratio": -0.1},
        {"distance_reward_curriculum_ratio": 1.1},
        {"distance_reward_curriculum_ratio": 0.5, "total_steps": 0},
        {"distance_reward_curriculum_ratio": 0.5, "total_steps": 100, "distance_reward_length_min": 0.0},
        {"distance_reward_curriculum_ratio": 0.5, "total_steps": 100, "distance_reward_length": 0.2, "distance_reward_length_min": 0.3},
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
    assert cfg.distance_reward_shape == "tanh"
    assert cfg.distance_reward_length == 0.8
    assert cfg.distance_reward_curriculum_ratio == 0.0
    env = make_env()
    try:
        # the default reward is the old hard-coded 1 - tanh(d / 0.8)
        d = torch.linspace(0.0, 2.0, 5, device=env.device)
        torch.testing.assert_close(env._distance_reward(d), 1 - torch.tanh(d / 0.8))
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
        assert {
            "Episode_Termination/died",
            "Episode_Termination/time_out",
            "Metrics/final_distance_to_goal",
            "Curriculum/distance_reward_length",
        } <= log.keys()
        assert "Curriculum/progress" not in log  # the curriculum is off by default
        for key, value in log.items():
            assert isinstance(value, torch.Tensor) and value.numel() == 1, key
    finally:
        env.close()


@pytest.mark.parametrize(
    "shape, knobs",
    [("tanh", {}), ("multiexp", {}), ("multiexp", {"distance_reward_scales": (1.0,)})],
    ids=["tanh", "multiexp", "multiexp-single"],
)
def test_reward_terms(shape, knobs):
    length = 0.2  # not the default, so a hard-coded 0.8 would fail
    env = make_env(distance_reward_shape=shape, distance_reward_length=length, **knobs)
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        cfg, data, dt = env.cfg, env._robot.data, env.step_dt
        for _ in range(3):
            reward, terms = step_terms(env, rand_cmd(env))
            # the reward is exactly the sum of the logged terms (nothing added off the books)
            torch.testing.assert_close(reward, sum(terms.values()))
            # each term recomputed from the sim state (no env reset in between, so same state)
            d = torch.linalg.norm(env._desired_pos_w - data.root_pos_w, dim=1)
            if shape == "tanh":
                mapped = 1 - torch.tanh(d / length)
            else:
                mapped = torch.stack([torch.exp(-d / (length * s)) for s in cfg.distance_reward_scales]).mean(0)
            if knobs:  # one scale of 1.0 is a single exponential
                torch.testing.assert_close(mapped, torch.exp(-d / length))
            torch.testing.assert_close(terms["distance_to_goal"], mapped * cfg.distance_to_goal_reward_scale * dt)
            torch.testing.assert_close(terms["lin_vel"], data.root_lin_vel_b.square().sum(1) * cfg.lin_vel_reward_scale * dt)
            torch.testing.assert_close(terms["ang_vel"], data.root_ang_vel_b.square().sum(1) * cfg.ang_vel_reward_scale * dt)
        # all shapes are 1 at the goal: same maximum, only the slope around it differs
        zero = torch.zeros(1, device=env.device)
        torch.testing.assert_close(env._distance_reward(zero), torch.ones(1, device=env.device))
    finally:
        env.close()


def test_distance_curriculum_ratio_decays_exponentially():
    ratio, total_steps, length, length_min = 0.5, 200, 0.8, 0.05
    env = make_env(
        distance_reward_curriculum_ratio=ratio,
        total_steps=total_steps,
        distance_reward_length=length,
        distance_reward_length_min=length_min,
    )
    try:
        # ratio 0.5 of 200 steps is 100 steps
        steps = 100
        for counter, expected in [(0, length), (steps // 2, (length * length_min) ** 0.5), (steps, length_min), (3 * steps, length_min)]:
            env.common_step_counter = counter
            assert env._distance_reward_length() == pytest.approx(expected)
        # the reward uses the scheduled L: at d = L the tanh reward is 1 - tanh(1)
        d = torch.full((1,), length_min, device=env.device)
        torch.testing.assert_close(env._distance_reward(d), 1 - torch.tanh(torch.ones(1, device=env.device)))
    finally:
        env.close()


def test_curriculum_progress_logged():
    env = make_env(distance_reward_curriculum_ratio=0.5, total_steps=200)
    try:
        env.common_step_counter = 50  # half of the 100-step curriculum
        env.reset()
        log = env.extras["log"]
        for key in ("Curriculum/progress", "Curriculum/distance_reward_length"):
            assert isinstance(log[key], torch.Tensor) and log[key].numel() == 1, key
        assert log["Curriculum/progress"].item() == pytest.approx(0.5)
        assert log["Curriculum/distance_reward_length"].item() == pytest.approx(env._distance_reward_length())
    finally:
        env.close()
