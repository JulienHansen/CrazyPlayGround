"""Checks for the tracking env's observation/action history, command latency and action-rate
penalty (#53), in velocity (3-D action) and attitude (4-D action) mode, and the knobs-off defaults.

Run (from CrazyPlayGround/, needs Isaac Sim and a GPU):
    python -m pytest source/CrazyPlayGround/test/test_tracking_env.py -v
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402

import omni.usd  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

from CrazyPlayGround.tasks.direct.track.tracking_env import (  # noqa: E402
    AttTrackingEnvCfg,
    TrackingEnv,
    VelTrackingEnvCfg,
)

N, K, M = 4, 3, 2
D = 22  # observation-step size with the default future_traj_steps=4
MODES = [pytest.param(VelTrackingEnvCfg, id="vel"), pytest.param(AttTrackingEnvCfg, id="att")]


def make_env(cfg_cls, num_envs: int = N, **knobs) -> TrackingEnv:
    omni.usd.get_context().new_stage()
    cfg = cfg_cls()
    cfg.scene.num_envs = num_envs
    cfg.debug_vis = False
    for key, value in knobs.items():  # set after the cfg is built, as Hydra does
        setattr(cfg, key, value)
    env = TrackingEnv(cfg)
    # Random commands can push a drone past the 0.8 m tracking error or below 0.1 m. Keep only
    # the timeout, which the tests use to force resets on purpose.
    env._get_dones = lambda: (
        torch.zeros_like(env.reset_terminated), env.episode_length_buf >= env.max_episode_length - 1
    )
    return env


def obs_step(env) -> torch.Tensor:
    """Clean 22-D observation step recomputed from the sim state, independent of the env's buffers."""
    ids = env._robot._ALL_INDICES
    waypoints_w = env._compute_traj(env.cfg.future_traj_steps, ids, env.cfg.traj_step_size)
    pos_w, quat = env._robot.data.root_pos_w, env._robot.data.root_quat_w
    rpos = [subtract_frame_transforms(pos_w, quat, waypoints_w[:, i])[0] for i in range(env.cfg.future_traj_steps)]
    return torch.cat(rpos + [env._robot.data.root_lin_vel_b, env._robot.data.root_ang_vel_b, quat], dim=-1)


def window(xs, t, n):
    """xs[t-n+1..t] stacked oldest first on dim 1; indices below 0 clamp to xs[0]."""
    return torch.stack([xs[max(i, 0)] for i in range(t - n + 1, t + 1)], dim=1)


def adim(env) -> int:
    return env.cfg.action_space


def hover(env) -> torch.Tensor:
    """(N, A) hover command, the value of the command history right after a reset."""
    return env._hover_cmd.expand(env.num_envs, adim(env)).clone()


def rand_cmd(env):
    """A different command for every env and channel, inside the clamp range."""
    return torch.empty(env.num_envs, adim(env), device=env.device).uniform_(-1.0, 1.0)


def start(env):
    """Reset, and return the first observation step. Then zero the (randomised) episode clocks so no env
    times out during the test."""
    obs, _ = env.reset()
    s0 = obs_step(env)
    env.episode_length_buf[:] = 0
    return obs, s0


@pytest.mark.parametrize(
    "cfg_cls, knobs",
    [
        (VelTrackingEnvCfg, {"obs_input_hist": 0}),
        (VelTrackingEnvCfg, {"action_input_hist": -1}),
        (VelTrackingEnvCfg, {"action_latency_steps": -1}),
        (VelTrackingEnvCfg, {"action_latency_steps": 3, "action_latency_max_steps": 2}),
        (AttTrackingEnvCfg, {"min_thrust_scale": 1.2}),
        (AttTrackingEnvCfg, {"max_thrust_scale": 0.9}),
        (AttTrackingEnvCfg, {"min_thrust_scale": 0.0}),
    ],
)
def test_invalid_knobs_raise(cfg_cls, knobs):
    with pytest.raises(ValueError):
        make_env(cfg_cls, **knobs)


@pytest.mark.parametrize("cfg_cls", MODES)
def test_defaults_are_a_single_obs_step(cfg_cls):
    env = make_env(cfg_cls)
    try:
        assert env.single_observation_space["policy"].shape == (D,)
        obs, s0 = start(env)
        torch.testing.assert_close(obs["policy"], s0)
        for _ in range(3):
            a = rand_cmd(env)
            obs, *_ = env.step(a)
            torch.testing.assert_close(obs["policy"], obs_step(env))
            torch.testing.assert_close(env._actions, a)  # no latency: the command just sent
    finally:
        env.close()


@pytest.mark.parametrize("cfg_cls", MODES)
def test_stacking_timesteps_and_reset(cfg_cls):
    env = make_env(cfg_cls, obs_input_hist=K, action_input_hist=M)
    A_dim = adim(env)

    def split(obs):
        p = obs["policy"]
        return p[:, : D * K].view(N, K, D), p[:, D * K :].view(N, M, A_dim)

    try:
        # space follows the late (Hydra-style) override
        assert env.single_observation_space["policy"].shape == (D * K + A_dim * M,)

        obs, s0 = start(env)
        S = [s0]           # S[t]: observation step after step t
        A = [hover(env)]   # A[t]: clamped command of step t; A[0] = history fill at reset
        obs_steps, acts = split(obs)
        torch.testing.assert_close(obs_steps, window(S, 0, K))  # K copies of the first observation step
        torch.testing.assert_close(acts, window(A, 0, M))    # hover command, not zeros in Att

        for t in range(1, 6):
            a = rand_cmd(env)
            a[:, 0] = 5.0 * (-1) ** t                       # out of range -> must be clamped to +-1
            obs, *_ = env.step(a)
            assert not env.reset_buf.any()
            S.append(obs_step(env))
            A.append(a.clamp(-1.0, 1.0))
            obs_steps, acts = split(obs)
            torch.testing.assert_close(obs_steps, window(S, t, K))
            torch.testing.assert_close(acts, window(A, t, M))
            torch.testing.assert_close(obs_steps[:, -1], S[t])  # newest last
            torch.testing.assert_close(acts[:, -1], A[t])    # the command just sent
        assert not torch.equal(S[5], S[4])                   # observation steps really change over time

        # force env 0 to time out on the next step; envs 1.. must keep their history
        env.episode_length_buf[0] = env.max_episode_length - 1
        a = rand_cmd(env)
        obs, *_ = env.step(a)
        assert env.reset_buf.tolist() == [True] + [False] * (N - 1)
        S.append(obs_step(env))
        A.append(a.clamp(-1.0, 1.0))
        t = len(S) - 1
        obs_steps, acts = split(obs)
        torch.testing.assert_close(obs_steps[0], S[t][0].expand(K, D))       # refilled with the new episode's first step
        torch.testing.assert_close(acts[0], env._hover_cmd.expand(M, A_dim))  # old commands replaced by hover
        torch.testing.assert_close(obs_steps[1:], window(S, t, K)[1:])
        torch.testing.assert_close(acts[1:], window(A, t, M)[1:])
    finally:
        env.close()


@pytest.mark.parametrize("cfg_cls", MODES)
def test_latency_fixed_and_reset(cfg_cls):
    d = 2
    env = make_env(cfg_cls, action_input_hist=M, action_latency_steps=d)
    try:
        start(env)
        A = [hover(env)]  # A[t]: command of step t; A[0] = before the episode
        for t in range(1, 7):
            A.append(rand_cmd(env))
            obs, *_ = env.step(A[t])
            torch.testing.assert_close(env._actions, A[max(t - d, 0)])                       # controller: t-d
            torch.testing.assert_close(obs["policy"][:, D:].view(N, M, -1), window(A, t, M))  # obs: commanded

        # env 0 times out at step 7; its old commands must never reach the controller
        env.episode_length_buf[0] = env.max_episode_length - 1
        A.append(rand_cmd(env))
        env.step(A[7])
        assert env.reset_buf.tolist() == [True] + [False] * (N - 1)
        B = [hover(env)]  # env 0's new episode, same indexing
        for s in range(1, 5):
            A.append(rand_cmd(env))
            B.append(A[-1])
            env.step(A[-1])
            torch.testing.assert_close(env._actions[0], B[max(s - d, 0)][0])
            torch.testing.assert_close(env._actions[1:], A[7 + s - d][1:])
    finally:
        env.close()


def test_latency_randomised_per_episode():
    lo, hi = 1, 4
    env = make_env(VelTrackingEnvCfg, action_latency_steps=lo, action_latency_max_steps=hi)
    try:
        seen = set()
        for _ in range(20):  # 80 draws: P(some value never drawn) < 1e-9
            env.reset()
            assert ((env.delay_steps >= lo) & (env.delay_steps <= hi)).all()
            seen |= set(env.delay_steps.tolist())
        assert seen == set(range(lo, hi + 1))

        env.episode_length_buf[:] = 0
        delay = env.delay_steps.tolist()
        A = [hover(env)]
        for t in range(1, 7):
            A.append(rand_cmd(env))
            env.step(A[t])
            expected = torch.stack([A[max(t - delay[i], 0)][i] for i in range(N)])
            torch.testing.assert_close(env._actions, expected)
    finally:
        env.close()


@pytest.mark.parametrize("cfg_cls", MODES)
def test_action_rate_penalises_commanded_action(cfg_cls):
    scale = -0.5
    env = make_env(cfg_cls, action_latency_steps=2, action_rate_reward_scale=scale)  # latency on: must not matter
    try:
        start(env)
        A = [hover(env)]
        for t in range(1, 7):
            A.append(rand_cmd(env))
            before = env._episode_sums["action_rate"].clone()
            env.step(A[t])
            penalty = scale * (A[t] - A[t - 1]).square().sum(1) * env.step_dt  # commanded, not delayed
            torch.testing.assert_close(env._episode_sums["action_rate"] - before, penalty)

        # env 0 resets at step 7; on step 8 its previous command must be the hover command, not A[7]
        env.episode_length_buf[0] = env.max_episode_length - 1
        env.step(rand_cmd(env))
        c = rand_cmd(env)
        env.step(c)
        torch.testing.assert_close(
            env._episode_sums["action_rate"][0], scale * (c[0] - env._hover_cmd).square().sum() * env.step_dt
        )
    finally:
        env.close()


def test_att_history_resets_to_hover():
    d = 3
    env = make_env(AttTrackingEnvCfg, action_latency_steps=d)
    try:
        cfg = env.cfg
        expected = (1.0 - cfg.min_thrust_scale) / (cfg.max_thrust_scale - cfg.min_thrust_scale)
        start(env)
        torch.testing.assert_close(env.cmd_history[..., 3], torch.full_like(env.cmd_history[..., 3], expected))
        assert not env.cmd_history[..., :3].any()

        hover_pwm = env._hover_thrust / env._ctrl.thrust_cmd_scale
        for t in range(1, d + 2):
            env.step(rand_cmd(env))
            if t <= d:  # the controller still gets the reset fill: hover thrust, level, no yaw rate
                torch.testing.assert_close(env._thrust_pwm, torch.full_like(env._thrust_pwm, hover_pwm))
                assert not env._att_ref.any() and not env._yaw_rate_ref.any()
            else:  # the first real command arrives
                assert not torch.allclose(env._thrust_pwm, torch.full_like(env._thrust_pwm, hover_pwm))
    finally:
        env.close()
