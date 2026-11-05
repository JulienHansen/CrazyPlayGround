"""Checks for Vel-Hovering and its sim-to-real knobs: stacking order, timestep indexing,
latency, noise, wind, actuation scaling, the reward terms, and the knobs-off defaults.

Run (conda env DragonFly, from CrazyPlayGround/):
    python -m pytest source/CrazyPlayGround/test/test_vel_hovering.py -v
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402

import omni.usd  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms  # noqa: E402

from CrazyPlayGround.tasks.direct.hovering.vel_hovering import QuadcopterEnv, QuadcopterEnvCfg  # noqa: E402

N, K, M = 4, 3, 2


def make_env(num_envs: int = N, **knobs) -> QuadcopterEnv:
    omni.usd.get_context().new_stage()
    cfg = QuadcopterEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.debug_vis = False
    for key, value in knobs.items():  # set after the cfg is built, as Hydra does
        setattr(cfg, key, value)
    env = QuadcopterEnv(cfg)
    # Drones spawn up to 2.0 m, the env's kill height, so a random command can reset an env
    # mid-test. Keep only the timeout, which the tests use to force resets on purpose.
    env._get_dones = lambda: (
        torch.zeros_like(env.reset_terminated), env.episode_length_buf >= env.max_episode_length - 1
    )
    return env


def frame(env) -> torch.Tensor:
    """Clean 6-D observation recomputed from the sim state, independent of the env's buffers."""
    goal_b, _ = subtract_frame_transforms(
        env._robot.data.root_pos_w, env._robot.data.root_quat_w, env._desired_pos_w
    )
    return torch.cat([env._robot.data.root_lin_vel_b, goal_b], dim=-1)


def window(seq, t, n):
    """seq[t-n+1..t] stacked oldest first on dim 1; indices below 0 clamp to seq[0]."""
    return torch.stack([seq[max(i, 0)] for i in range(t - n + 1, t + 1)], dim=1)


def split(obs):
    p = obs["policy"]
    return p[:, : 6 * K].view(N, K, 6), p[:, 6 * K :].view(N, M, 3)


def rand_cmd(env):
    """A different command for every env and axis, inside the clamp range."""
    return torch.empty(N, 3, device=env.device).uniform_(-1.0, 1.0)


def step_terms(env, cmd):
    """Step once and return (reward, {key: how much that episode sum grew this step})."""
    before = {k: v.clone() for k, v in env._episode_sums.items()}
    _, reward, *_ = env.step(cmd)
    return reward, {k: env._episode_sums[k] - before[k] for k in before}


@pytest.mark.parametrize(
    "knobs",
    [
        {"obs_input_hist": 0},
        {"action_input_hist": -1},
        {"action_latency_steps": -1},
        {"action_latency_steps": 3, "action_latency_max_steps": 2},
        {"noise_corr": 1.0},
        {"noise_bias_std": -0.1},
        {"wind_force_range": (-0.01, 0.02)},
        {"wind_force_range": (0.03, 0.02)},
        {"thrust_scale_range": (0.0, 1.2)},
        {"torque_scale_range": (1.2, 1.1)},
        {"distance_reward_shape": "gauss"},
        {"distance_reward_length": 0.0},
        {"distance_reward_shape": "multiexp", "distance_reward_scales": ()},
        {"distance_reward_shape": "multiexp", "distance_reward_scales": (1.0, 0.0)},
    ],
)
def test_invalid_knobs_raise(knobs):
    with pytest.raises(ValueError):
        make_env(**knobs)


def test_defaults_are_plain_hovering():
    env = make_env()
    try:
        assert env.single_observation_space["policy"].shape == (6,)
        obs, _ = env.reset()
        torch.testing.assert_close(obs["policy"], frame(env))
        torch.testing.assert_close(env.true_state, frame(env))
        obs, *_ = env.step(torch.full((N, 3), 0.3, device=env.device))
        torch.testing.assert_close(obs["policy"], frame(env))
        torch.testing.assert_close(env.true_state, frame(env))
        env._apply_action()
        assert not env._robot.instantaneous_wrench_composer.active  # no wind, no actuation scaling by default
    finally:
        env.close()


def test_stacking_timesteps_and_reset():
    env = make_env(obs_input_hist=K, action_input_hist=M)
    try:
        # space follows the late (Hydra-style) override
        assert env.single_observation_space["policy"].shape == (6 * K + 3 * M,)

        obs, _ = env.reset()
        env.episode_length_buf[:] = 0  # full reset randomises it; no timeout inside the test
        F = [frame(env)]                                   # F[t]: state after step t
        A = [torch.zeros(N, 3, device=env.device)]         # A[t]: clamped command of step t
        frames, acts = split(obs)
        torch.testing.assert_close(frames, window(F, 0, K))  # K copies of the first frame
        assert not acts.any()

        for t in range(1, 6):
            a = torch.full((N, 3), 0.05 * t, device=env.device)
            a[:, 2] = 5.0 * (-1) ** t                      # out of range -> must be clamped to +-1
            obs, *_ = env.step(a)
            assert not env.reset_buf.any()
            F.append(frame(env))
            A.append(a.clamp(-1.0, 1.0))
            frames, acts = split(obs)
            torch.testing.assert_close(frames, window(F, t, K))
            torch.testing.assert_close(acts, window(A, t, M))
            torch.testing.assert_close(frames[:, -1], F[t])        # newest last
            torch.testing.assert_close(acts[:, -1], A[t])          # the command just sent
        assert not torch.equal(F[5], F[4])                         # frames really change over time

        # force env 0 to time out on the next step; envs 1.. must keep their history
        env.episode_length_buf[0] = env.max_episode_length - 1
        a = torch.full((N, 3), -0.2, device=env.device)
        obs, *_ = env.step(a)
        assert env.reset_buf.tolist() == [True] + [False] * (N - 1)
        F.append(frame(env))
        A.append(a.clamp(-1.0, 1.0))
        t = len(F) - 1
        frames, acts = split(obs)
        torch.testing.assert_close(frames[0], F[t][0].expand(K, 6))   # refilled with the new episode's frame
        assert not acts[0].any()                                      # old episode's commands cleared
        torch.testing.assert_close(frames[1:], window(F, t, K)[1:])
        torch.testing.assert_close(acts[1:], window(A, t, M)[1:])
    finally:
        env.close()


def test_latency_fixed_and_reset():
    d = 2
    env = make_env(action_input_hist=M, action_latency_steps=d)
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        A = [torch.zeros(N, 3, device=env.device)]  # A[t]: command of step t; A[0] = before the episode
        for t in range(1, 7):
            A.append(rand_cmd(env))
            obs, *_ = env.step(A[t])
            torch.testing.assert_close(env._actions, A[max(t - d, 0)])                      # controller: t-d
            torch.testing.assert_close(obs["policy"][:, 6:].view(N, M, 3), window(A, t, M))  # obs: commanded

        # env 0 times out at step 7; its old commands must never reach the controller
        env.episode_length_buf[0] = env.max_episode_length - 1
        A.append(rand_cmd(env))
        env.step(A[7])
        assert env.reset_buf.tolist() == [True] + [False] * (N - 1)
        B = [torch.zeros(N, 3, device=env.device)]  # env 0's new episode, same indexing
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
    env = make_env(action_latency_steps=lo, action_latency_max_steps=hi)
    try:
        seen = set()
        for _ in range(20):  # 80 draws: P(some value never drawn) < 1e-9
            env.reset()
            assert ((env.delay_steps >= lo) & (env.delay_steps <= hi)).all()
            seen |= set(env.delay_steps.tolist())
        assert seen == set(range(lo, hi + 1))

        env.episode_length_buf[:] = 0
        delay = env.delay_steps.tolist()
        A = [torch.zeros(N, 3, device=env.device)]
        for t in range(1, 7):
            A.append(rand_cmd(env))
            env.step(A[t])
            expected = torch.stack([A[max(t - delay[i], 0)][i] for i in range(N)])
            torch.testing.assert_close(env._actions, expected)
    finally:
        env.close()


def test_action_rate_penalises_commanded_action():
    scale = -0.5
    env = make_env(action_latency_steps=2, action_rate_reward_scale=scale)  # latency on: must not matter
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        A = [torch.zeros(N, 3, device=env.device)]
        for t in range(1, 7):
            A.append(rand_cmd(env))
            before = env._episode_sums["action_rate"].clone()
            env.step(A[t])
            penalty = scale * (A[t] - A[t - 1]).square().sum(1) * env.step_dt  # commanded, not delayed
            torch.testing.assert_close(env._episode_sums["action_rate"] - before, penalty)

        # env 0 resets at step 7; on step 8 its previous command must be 0, not A[7]
        env.episode_length_buf[0] = env.max_episode_length - 1
        env.step(rand_cmd(env))
        c = rand_cmd(env)
        env.step(c)
        torch.testing.assert_close(
            env._episode_sums["action_rate"][0], scale * c[0].square().sum() * env.step_dt
        )
    finally:
        env.close()


@pytest.mark.parametrize("rho", [0.0, 0.9])
def test_noise_std_and_correlation(rho):
    n_envs, std = 256, 0.1
    env = make_env(num_envs=n_envs, add_noise=True, noise_std=std, noise_corr=rho)
    try:
        obs, _ = env.reset()
        env.episode_length_buf[:] = 0
        noise = [obs["policy"] - frame(env)]               # measured noise = obs - true state
        hover = torch.zeros(n_envs, 3, device=env.device)
        for _ in range(20):
            obs, *_ = env.step(hover)
            noise.append(obs["policy"] - frame(env))
        noise = torch.stack(noise) / std                   # [21 steps, envs, 6]; unit std if correct
        assert abs(noise[0].std().item() - 1.0) < 0.1      # full strength right after reset
        # regression of n_t+1 on n_t: slope = rho, residual = the fresh part, std sqrt(1 - rho^2).
        # Residuals are independent, so ~30k samples give a standard error of ~0.005 on both.
        rho_hat = ((noise[1:] * noise[:-1]).sum() / noise[:-1].square().sum()).item()
        assert abs(rho_hat - rho) < 0.03
        innovation = noise[1:] - rho * noise[:-1]
        assert abs(innovation.std().item() - (1.0 - rho * rho) ** 0.5) < 0.03
    finally:
        env.close()


def test_noise_bias_per_episode():
    env = make_env(add_noise=True, noise_std=0.01, noise_bias_std=0.05)
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        bias = env.obs_noise_bias.clone()
        assert bias.abs().max() > 0
        for _ in range(5):
            obs, *_ = env.step(torch.zeros(N, 3, device=env.device))
            # obs - true state = bias + white noise of std 0.01; 0.05 is 5 sigma
            torch.testing.assert_close(obs["policy"] - frame(env), bias, atol=0.05, rtol=0)
        env.episode_length_buf[0] = env.max_episode_length - 1
        env.step(torch.zeros(N, 3, device=env.device))
        assert not torch.equal(env.obs_noise_bias[0], bias[0])
        torch.testing.assert_close(env.obs_noise_bias[1:], bias[1:])
    finally:
        env.close()


def test_noise_knobs_need_add_noise(caplog):
    env = make_env(noise_corr=0.5, noise_bias_std=0.1)
    try:
        assert "no effect without add_noise" in caplog.text
        obs, _ = env.reset()
        torch.testing.assert_close(obs["policy"], frame(env))
    finally:
        env.close()


def test_wind_strength_direction_frame_and_reset():
    n_envs, lo, hi = 256, 0.01, 0.03
    env = make_env(num_envs=n_envs, wind_force_range=(lo, hi))
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        wind = env.wind_force_world.clone()
        strength = wind.norm(dim=1)
        assert ((strength >= lo - 1e-6) & (strength <= hi + 1e-6)).all()
        direction = wind / strength[:, None]
        # uniform over the sphere: mean direction ~ 0, and z (vertical) spread over [-1, 1]
        assert direction.mean(0).abs().max() < 0.2
        assert direction[:, 2].min() < -0.5 and direction[:, 2].max() > 0.5   # wind can push down and up
        assert abs(direction[:, 2].abs().mean().item() - 0.5) < 0.1          # E|z| = 1/2 if uniform

        cmd = torch.empty(n_envs, 3, device=env.device)
        for _ in range(3):
            env.step(cmd.uniform_(-1.0, 1.0))              # random commands tilt the drones
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
        env.step(cmd.uniform_(-1.0, 1.0))
        assert not torch.equal(env.wind_force_world[0], wind[0])
        torch.testing.assert_close(env.wind_force_world[1:], wind[1:])
    finally:
        env.close()


def test_thrust_torque_scaling():
    lo, hi = 0.8, 1.2
    env = make_env(thrust_scale_range=(lo, hi), torque_scale_range=(lo, hi))
    try:
        env.reset()
        env.episode_length_buf[:] = 0
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


@pytest.mark.parametrize("shape", ["tanh", "exp", "multiexp"])
def test_reward_terms(shape):
    length = 0.2                                    # not the default, so a hard-coded 0.3 or 0.8 would fail
    env = make_env(distance_reward_shape=shape, distance_reward_length=length, action_rate_reward_scale=-0.5)
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
            mapped = {
                "tanh": 1 - torch.tanh(d / length),
                "exp": torch.exp(-d / length),
                "multiexp": torch.stack([torch.exp(-d / (length * s)) for s in cfg.distance_reward_scales]).mean(0),
            }[shape]
            torch.testing.assert_close(terms["distance_to_goal"], mapped * cfg.distance_to_goal_reward_scale * dt)
            torch.testing.assert_close(terms["lin_vel"], data.root_lin_vel_b.square().sum(1) * cfg.lin_vel_reward_scale * dt)
            torch.testing.assert_close(terms["ang_vel"], data.root_ang_vel_b.square().sum(1) * cfg.ang_vel_reward_scale * dt)
        # all shapes are 1 at the goal: same maximum, only the slope around it differs
        zero = torch.zeros(1, device=env.device)
        torch.testing.assert_close(env._distance_reward(zero), torch.ones(1, device=env.device))
    finally:
        env.close()


def test_knobs_off_no_extra_wrench_no_delay():
    env = make_env()
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        b = env._body_id[0]
        for _ in range(3):
            a = rand_cmd(env) * 1.5                        # some out of range
            env.step(a)
            # the controller tracks this step's clamped command: no delay, no scaling
            torch.testing.assert_close(env._ref_vel, a.clamp(-1.0, 1.0) * env.cfg.max_velocity)
            env._apply_action()
            # the body gets exactly the controller output, nothing on top
            assert not env._robot.instantaneous_wrench_composer.active
            torch.testing.assert_close(env._robot.permanent_wrench_composer.composed_force_as_torch[:, b], env._thrust[:, 0])
            torch.testing.assert_close(env._robot.permanent_wrench_composer.composed_torque_as_torch[:, b], env._moment[:, 0])
    finally:
        env.close()
