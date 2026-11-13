"""Checks for the tracking env's observation/action history, command latency and action-rate
penalty (#53), its sim-to-real randomisation: observation noise, wind, thrust/torque scaling (#54),
and its tracking metrics and trajectory clock (#56), in velocity (3-D action) and attitude (4-D action)
mode, and the knobs-off defaults.

Run (from CrazyPlayGround/, needs Isaac Sim and a GPU):
    python -m pytest source/CrazyPlayGround/test/test_tracking_env.py -v
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402

import omni.usd  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms  # noqa: E402

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
        (VelTrackingEnvCfg, {"noise_corr": 1.0}),
        (VelTrackingEnvCfg, {"noise_bias_std": -0.1}),
        (VelTrackingEnvCfg, {"wind_force_range": (-0.01, 0.02)}),
        (VelTrackingEnvCfg, {"wind_force_range": (0.03, 0.02)}),
        (VelTrackingEnvCfg, {"thrust_scale_range": (0.0, 1.2)}),
        (VelTrackingEnvCfg, {"torque_scale_range": (1.2, 1.1)}),
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
            assert not env._robot.instantaneous_wrench_composer.active  # no wind, no actuation scaling
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


@pytest.mark.parametrize("rho", [0.0, 0.9])
def test_noise_std_and_correlation(rho):
    n_envs, std = 256, 0.1
    env = make_env(VelTrackingEnvCfg, num_envs=n_envs, add_noise=True, noise_std=std, noise_corr=rho)
    try:
        # true step taken before start() zeroes the clocks: the waypoints depend on episode time
        obs, s0 = start(env)
        noise = [obs["policy"] - s0]                       # measured noise = obs - true step
        for _ in range(20):
            obs, *_ = env.step(hover(env))
            noise.append(obs["policy"] - obs_step(env))
        noise = torch.stack(noise) / std                   # [21 steps, envs, 22]; unit std if correct
        assert abs(noise[0].std().item() - 1.0) < 0.1      # full strength right after reset
        # regression of n_t+1 on n_t: slope = rho, residual = the fresh part, std sqrt(1 - rho^2).
        # Residuals are independent, so ~110k samples give a standard error of ~0.003 on both.
        rho_hat = ((noise[1:] * noise[:-1]).sum() / noise[:-1].square().sum()).item()
        assert abs(rho_hat - rho) < 0.03
        innovation = noise[1:] - rho * noise[:-1]
        assert abs(innovation.std().item() - (1.0 - rho * rho) ** 0.5) < 0.03
    finally:
        env.close()


def test_noise_bias_per_episode_and_true_rewards():
    env = make_env(VelTrackingEnvCfg, add_noise=True, noise_std=0.01, noise_bias_std=0.5)
    try:
        start(env)
        bias = env.obs_noise_bias.clone()
        assert bias.abs().max() > 0
        cfg = env.cfg
        for _ in range(5):
            before = env._episode_sums["distance_to_goal"].clone()
            obs, *_ = env.step(hover(env))
            # obs - true step = bias + white noise of std 0.01; 0.05 is 5 sigma
            torch.testing.assert_close(obs["policy"] - obs_step(env), bias, atol=0.05, rtol=0)
            # the reward sees the true position, not the (heavily biased) observation
            d = torch.linalg.norm(env._current_ref_w(env._robot._ALL_INDICES) - env._robot.data.root_pos_w, dim=1)
            expected = (1.0 - torch.tanh(d / 0.8)) * cfg.pos_reward_scale * env.step_dt
            torch.testing.assert_close(env._episode_sums["distance_to_goal"] - before, expected)
        env.episode_length_buf[0] = env.max_episode_length - 1
        env.step(hover(env))
        assert not torch.equal(env.obs_noise_bias[0], bias[0])
        torch.testing.assert_close(env.obs_noise_bias[1:], bias[1:])
    finally:
        env.close()


def test_noise_knobs_need_add_noise(caplog):
    env = make_env(VelTrackingEnvCfg, noise_corr=0.5, noise_bias_std=0.1)
    try:
        assert "no effect without add_noise" in caplog.text
        obs, s0 = start(env)
        torch.testing.assert_close(obs["policy"], s0)
    finally:
        env.close()


def test_wind_strength_direction_frame_and_reset():
    n_envs, lo, hi = 256, 0.01, 0.03
    env = make_env(VelTrackingEnvCfg, num_envs=n_envs, wind_force_range=(lo, hi))
    try:
        start(env)
        wind = env.wind_force_world.clone()
        strength = wind.norm(dim=1)
        assert ((strength >= lo - 1e-6) & (strength <= hi + 1e-6)).all()
        direction = wind / strength[:, None]
        # uniform over the sphere: mean direction ~ 0, and z (vertical) spread over [-1, 1]
        assert direction.mean(0).abs().max() < 0.2
        assert direction[:, 2].min() < -0.5 and direction[:, 2].max() > 0.5   # wind can push down and up
        assert abs(direction[:, 2].abs().mean().item() - 0.5) < 0.1          # E|z| = 1/2 if uniform

        for _ in range(3):
            env.step(rand_cmd(env))                        # random commands tilt the drones
        torch.testing.assert_close(env.wind_force_world, wind)   # constant within the episode

        # what Isaac Lab will apply on the next physics step, in the body frame
        env._apply_action()
        wrench = env._robot.instantaneous_wrench_composer
        b = env._body_id[0]
        expected_body = quat_apply_inverse(env._robot.data.body_link_quat_w[:, b], wind)
        assert not torch.allclose(expected_body, wind)     # tilt makes the frame matter
        torch.testing.assert_close(wrench.composed_force_as_torch[:, b], expected_body)
        assert not wrench.composed_torque_as_torch.any()   # wind adds no torque
        wrench.reset()                                     # undo the manual call

        # env 0 resets: new wind; the others keep theirs
        env.episode_length_buf[0] = env.max_episode_length - 1
        env.step(rand_cmd(env))
        assert not torch.equal(env.wind_force_world[0], wind[0])
        torch.testing.assert_close(env.wind_force_world[1:], wind[1:])
    finally:
        env.close()


@pytest.mark.parametrize("cfg_cls", MODES)
def test_thrust_torque_scaling(cfg_cls):
    lo, hi = 0.8, 1.2
    env = make_env(cfg_cls, thrust_scale_range=(lo, hi), torque_scale_range=(lo, hi))
    try:
        start(env)
        for s in (env.thrust_scale, env.torque_scale):          # drawn in range, and really drawn
            assert ((s >= lo) & (s <= hi)).all() and not torch.allclose(s, torch.ones_like(s))
        env.step(rand_cmd(env))

        # what the next physics step adds on top of the controller output T (permanent composer)
        env._apply_action()
        composer = env._robot.instantaneous_wrench_composer
        b = env._body_id[0]
        thrust, moment = env._thrust[:, 0], env._moment[:, 0]
        extra_force, extra_torque = composer.composed_force_as_torch[:, b], composer.composed_torque_as_torch[:, b]
        torch.testing.assert_close(extra_force, (env.thrust_scale[:, None] - 1) * thrust)
        torch.testing.assert_close(extra_torque, (env.torque_scale[:, None] - 1) * moment)
        # so the body feels scale x T in total
        torch.testing.assert_close(thrust + extra_force, env.thrust_scale[:, None] * thrust)
        composer.reset()                                          # undo the manual call

        # env 0 resets: new scales; the others keep theirs for the whole episode
        thrust_scale, torque_scale = env.thrust_scale.clone(), env.torque_scale.clone()
        env.episode_length_buf[0] = env.max_episode_length - 1
        env.step(rand_cmd(env))
        assert env.thrust_scale[0] != thrust_scale[0] and env.torque_scale[0] != torque_scale[0]
        torch.testing.assert_close(env.thrust_scale[1:], thrust_scale[1:])
        torch.testing.assert_close(env.torque_scale[1:], torque_scale[1:])
    finally:
        env.close()


@pytest.mark.parametrize("cfg_cls", MODES)
def test_knobs_off_no_extra_wrench(cfg_cls):
    env = make_env(cfg_cls)
    try:
        start(env)
        b = env._body_id[0]
        for _ in range(3):
            env.step(rand_cmd(env))
            env._apply_action()
            # the body gets exactly the controller output, nothing on top
            assert not env._robot.instantaneous_wrench_composer.active
            torch.testing.assert_close(env._robot.permanent_wrench_composer.composed_force_as_torch[:, b], env._thrust[:, 0])
            torch.testing.assert_close(env._robot.permanent_wrench_composer.composed_torque_as_torch[:, b], env._moment[:, 0])
    finally:
        env.close()


def ref_error(env, env_ids) -> torch.Tensor:
    """(len(env_ids),) distance to the reference of the reward and the termination, m."""
    return torch.linalg.norm(env._current_ref_w(env_ids) - env._robot.data.root_pos_w[env_ids], dim=1)


@pytest.mark.parametrize("cfg_cls", MODES)
def test_tracking_metrics_per_episode(cfg_cls):
    env = make_env(cfg_cls, episode_length_s=0.1)
    # the last step's error is only visible inside the reset, before the drone is moved back
    final_errors = []
    reset_idx = env._reset_idx

    def spy(env_ids):
        final_errors.append(ref_error(env, env_ids))
        reset_idx(env_ids)

    try:
        start(env)
        env._reset_idx = spy
        T = int(env.max_episode_length) - 1  # steps until the time-out
        ids = env._robot._ALL_INDICES
        errors, efforts = [], []
        for t in range(1, T + 1):
            a = rand_cmd(env)
            a[:, 0] = 3.0  # out of range: the effort uses the clamped command
            env.step(a)
            efforts.append(torch.sum(a.clamp(-1.0, 1.0).square(), dim=1))
            if t < T:
                assert not env.reset_buf.any()
                errors.append(ref_error(env, ids))
        assert env.reset_buf.all() and len(final_errors) == 1
        e = torch.stack(errors + final_errors)  # (T, N)

        m = env.extras["step_metrics"]
        assert m["env_ids"].tolist() == list(range(N))
        assert m["length"].tolist() == [T] * N
        assert not m["died"].any()
        torch.testing.assert_close(m["tracking_rmse"], e.square().mean(0).sqrt())
        torch.testing.assert_close(m["tracking_mean_error"], e.mean(0))
        torch.testing.assert_close(m["tracking_max_error"], e.max(0).values)
        torch.testing.assert_close(m["effort"], torch.stack(efforts).mean(0))

        log = env.extras["log"]
        assert log["Metrics/crash_rate"].item() == 0.0
        for key in ("tracking_rmse", "tracking_mean_error", "tracking_max_error", "effort"):
            torch.testing.assert_close(log["Metrics/" + key], m[key].mean())  # all episodes full length

        assert (env.episode_step == 0).all()
        assert (env._effort_sum == 0).all()

        # a shorter second episode: the error slots left over from the first must be ignored.
        # All envs just reset at once, which randomises the time-out clocks: zero them, as start() does
        env.episode_length_buf[:] = 0
        errors, efforts, final_errors[:] = [], [], []
        T2 = 3
        for t in range(1, T2 + 1):
            if t == T2:
                env.episode_length_buf[:] = env.max_episode_length - 2  # time out at this step
            a = rand_cmd(env)
            env.step(a)
            efforts.append(torch.sum(a.square(), dim=1))
            if t < T2:
                assert not env.reset_buf.any()
                errors.append(ref_error(env, ids))
        assert env.reset_buf.all() and len(final_errors) == 1
        e = torch.stack(errors + final_errors)  # (T2, N)
        m = env.extras["step_metrics"]
        assert m["length"].tolist() == [T2] * N
        torch.testing.assert_close(m["tracking_rmse"], e.square().mean(0).sqrt())
        torch.testing.assert_close(m["tracking_mean_error"], e.mean(0))
        torch.testing.assert_close(m["tracking_max_error"], e.max(0).values)
        torch.testing.assert_close(m["effort"], torch.stack(efforts).mean(0))
    finally:
        env.close()


def test_crash_counted_not_in_means():
    env = make_env(VelTrackingEnvCfg)
    crash_at = 3
    try:
        start(env)
        env._get_dones = lambda: (
            (env.episode_step == crash_at) & (env.env_idx == 0),
            env.episode_length_buf >= env.max_episode_length - 1,
        )
        for t in range(1, crash_at + 1):
            env.step(hover(env))
            assert env.reset_buf.tolist() == [t == crash_at] + [False] * (N - 1)
        m = env.extras["step_metrics"]
        assert m["env_ids"].tolist() == [0]
        assert m["died"].tolist() == [True]
        assert m["length"].tolist() == [crash_at]
        log = env.extras["log"]
        assert log["Metrics/crash_rate"].item() == 1.0
        # no full-length episode: the NaN means are left out of the log
        assert [k for k in log if k.startswith("Metrics/")] == ["Metrics/crash_rate"]
    finally:
        env.close()


def test_reference_ignores_episode_length_buf():
    """play.py --eval_csv zeroes episode_length_buf after the reset; the reference must not jump away."""
    env = make_env(VelTrackingEnvCfg)
    try:
        env.reset()
        ids = env._robot._ALL_INDICES
        assert env.episode_length_buf.any()  # the full reset randomised the episode clocks
        assert (env.episode_step == 0).all()
        # spawned on the trajectory's start, within the spawn noise (+-0.3 m in x, y, +-0.2 m in z)
        start_w = env._compute_traj(steps=1, env_ids=ids, step_size=0.0)[:, 0]
        assert (torch.linalg.norm(env._robot.data.root_pos_w - start_w, dim=1) < 0.5).all()
        ref = env._current_ref_w(ids)
        env.episode_length_buf[:] = 0
        torch.testing.assert_close(env._current_ref_w(ids), ref)
    finally:
        env.close()
