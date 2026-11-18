"""Checks for Rate-Hovering: config validation, the plain-task defaults, tensor logging, the
distance reward shapes, the distance-length curriculum and the step-response metrics.

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
from CrazyPlayGround.tasks.direct.hovering.step_response import step_response  # noqa: E402

N = 4
# the step-response means logged under Metrics/* (d0 and rise_10_90 stay in step_metrics only)
METRIC_KEYS = ("tau", "overshoot", "settling_time", "steady_state_error", "effort")


def make_env(num_envs: int = N, crashes: bool = False, **knobs) -> QuadcopterEnv:
    omni.usd.get_context().new_stage()
    cfg = QuadcopterEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.debug_vis = False
    for key, value in knobs.items():  # set after the cfg is built, as Hydra does
        setattr(cfg, key, value)
    env = QuadcopterEnv(cfg)
    if crashes:
        return env
    # Drones spawn up to 2.0 m, the env's kill height, so a command can reset an env
    # mid-test. Keep only the timeout.
    env._get_dones = lambda: (
        torch.zeros_like(env.reset_terminated), env.episode_length_buf >= env.max_episode_length - 1
    )
    return env


def spy_step_response(env) -> list[dict]:
    """Record each _log_step_response call: the trajectory it reads, then the metrics and log it leaves."""
    calls = []
    log_step_response = env._log_step_response

    def spy(env_ids):
        traj = env._traj_pos_w[env_ids].clone()
        episode_length_buf = env.episode_length_buf[env_ids].clone()
        log_step_response(env_ids)
        calls.append({
            "traj": traj,
            "episode_length_buf": episode_length_buf,
            "step_metrics": {k: v.clone() for k, v in env.extras["step_metrics"].items()},
            "log": dict(env.extras["log"]),
        })

    env._log_step_response = spy  # _reset_idx calls self._log_step_response, so the instance attribute wins
    return calls


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
            "Metrics/crash_rate",
        } <= log.keys()
        assert "Curriculum/progress" not in log  # the curriculum is off by default
        # the first reset ends no episode, so it has no step-response mean
        assert not {"Metrics/" + k for k in METRIC_KEYS} & log.keys()
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


def test_step_metrics_recorded():
    env = make_env(episode_length_s=1.0)  # 100 steps
    try:
        env.reset()
        env.episode_length_buf[:] = 0  # full-length episodes, as play.py --eval_csv does
        spawn, goal = env._robot.data.root_pos_w.clone(), env._desired_pos_w.clone()
        torch.testing.assert_close(env._spawn_pos_w, spawn)
        calls = spy_step_response(env)
        steps = env.max_episode_length - 1  # the timeout fires on this step
        pos, effort_sum = [], torch.zeros(N, device=env.device)
        for _ in range(steps):
            cmd = 1.5 * rand_cmd(env)  # partly outside [-1, 1]: the effort uses the clamped command
            env.step(cmd)
            pos.append(env._robot.data.root_pos_w.clone())
            effort_sum += cmd.clamp(-1.0, 1.0).square().sum(1)
        assert len(calls) == 1  # one reset, of every env, at the timeout
        traj, m, log = calls[0]["traj"], calls[0]["step_metrics"], calls[0]["log"]

        # one row per step: the position after it (the last one is overwritten by the reset)
        torch.testing.assert_close(traj[:, : steps - 1], torch.stack(pos[:-1], dim=1))
        length = torch.full((N,), steps, device=env.device)
        assert torch.equal(m["env_ids"].sort().values, torch.arange(N, device=env.device))
        assert torch.equal(m["length"], length) and not m["died"].any()
        torch.testing.assert_close(m["start_pos_w"], spawn)
        torch.testing.assert_close(m["goal_pos_w"], goal)
        torch.testing.assert_close(m["d0"], torch.linalg.norm(goal - spawn, dim=1))
        torch.testing.assert_close(m["effort"], effort_sum / steps)
        for key, value in step_response(traj, spawn, goal, length, env.step_dt).items():
            torch.testing.assert_close(m[key], value, equal_nan=True, msg=key)

        # every episode is full length, so each logged mean is the NaN-skipping mean over all of them
        assert log["Metrics/crash_rate"].item() == 0.0
        for key in METRIC_KEYS:
            mean = m[key].nanmean()
            if torch.isfinite(mean):
                torch.testing.assert_close(log["Metrics/" + key], mean)
            else:  # a threshold no episode crossed: no NaN in the log
                assert "Metrics/" + key not in log
        assert {"Metrics/steady_state_error", "Metrics/effort"} <= log.keys()  # never NaN
        for key, value in log.items():
            assert isinstance(value, torch.Tensor) and value.numel() == 1, key

        # the buffers restart with the next episode, from its new random spawn
        assert torch.all(env._traj_rollout_idx == 0) and torch.all(env._effort_sum == 0.0)
        assert not torch.allclose(env._spawn_pos_w, spawn)
    finally:
        env.close()


def test_crashes_counted_not_averaged():
    env = make_env(crashes=True, episode_length_s=3.0)
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        calls = spy_step_response(env)
        # zero rates and the minimum thrust, 0.5 m*g: every drone sinks below the 0.1 m kill height
        fall = torch.tensor([0.0, 0.0, 0.0, -1.0], device=env.device).repeat(N, 1)
        crashed = set()
        for _ in range(env.max_episode_length - 2):  # stop before any timeout
            env.step(fall)
            crashed = {i for call in calls for i in call["step_metrics"]["env_ids"].tolist()}
            if len(crashed) == N:
                break
        assert crashed == set(range(N))
        for call in calls:
            m, log = call["step_metrics"], call["log"]
            assert m["died"].all() and torch.all(m["length"] < env.max_episode_length - 1)
            assert log["Metrics/crash_rate"].item() == 1.0
            assert not {"Metrics/" + k for k in METRIC_KEYS} & log.keys()  # crashes are not averaged
    finally:
        env.close()


def test_partial_first_episodes_not_averaged():
    env = make_env(episode_length_s=1.0)
    try:
        env.reset()
        start = torch.arange(1, N + 1, device=env.device) * 10  # a randomised first reset, without the chance of 0
        env.episode_length_buf[:] = start
        calls = spy_step_response(env)
        for _ in range(env.max_episode_length - 1 - int(start.min())):  # until the last env times out
            env.step(rand_cmd(env))
        assert sorted(i for call in calls for i in call["step_metrics"]["env_ids"].tolist()) == list(range(N))
        for call in calls:
            m, log = call["step_metrics"], call["log"]
            # the trajectory started at the reset, so it is shorter than the episode counter says
            assert torch.equal(m["length"], env.max_episode_length - 1 - start[m["env_ids"]])
            assert torch.all(call["episode_length_buf"] == env.max_episode_length - 1)
            assert not m["died"].any() and log["Metrics/crash_rate"].item() == 0.0
            assert not {"Metrics/" + k for k in METRIC_KEYS} & log.keys()  # in step_metrics, not in the means
    finally:
        env.close()
