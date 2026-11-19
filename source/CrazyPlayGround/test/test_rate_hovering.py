"""Checks for Rate-Hovering: config validation, the plain-task defaults, tensor logging, the
distance reward shapes, the distance-length curriculum, the step-response metrics, and the
observation/action history, action latency with its warm-up hold, and action-rate penalty, and
the sim-to-real randomisation: observation noise, wind, actuation scaling, setpoint bias and scale.

Run (conda env DragonFly, from CrazyPlayGround/):
    python -m pytest source/CrazyPlayGround/test/test_rate_hovering.py -v
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402

import omni.usd  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms  # noqa: E402

from CrazyPlayGround.tasks.direct.hovering.rate_hovering import QuadcopterEnv, QuadcopterEnvCfg  # noqa: E402
from CrazyPlayGround.tasks.direct.hovering.step_response import step_response  # noqa: E402

N, K, M = 4, 3, 2
FILL = -10.0  # history_fill_value: "no command" in empty action slots
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


def window(seq, t, n):
    """seq[t-n+1..t] stacked oldest first on dim 1; indices below 0 clamp to seq[0]."""
    return torch.stack([seq[max(i, 0)] for i in range(t - n + 1, t + 1)], dim=1)


def split(obs):
    p = obs["policy"]
    return p[:, : 18 * K].view(N, K, 18), p[:, 18 * K :].view(N, M, 4)


def fill(env):
    return torch.full((N, 4), FILL, device=env.device)


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
        {"noise_corr": 1.0},
        {"noise_corr": -0.1},
        {"noise_bias_std": -0.1},
        {"wind_force_range": (-0.01, 0.02)},
        {"wind_force_range": (0.03, 0.02)},
        {"thrust_scale_range": (0.0, 1.2)},
        {"torque_scale_range": (1.2, 1.1)},
        {"obs_input_hist": 0},
        {"action_input_hist": -1},
        {"action_latency_steps": -1},
        {"action_latency_steps": 3, "action_latency_max_steps": 2},
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
    assert (cfg.noise_corr, cfg.noise_bias_std, tuple(cfg.wind_force_range)) == (0.0, 0.0, (0.0, 0.0))
    assert (tuple(cfg.thrust_scale_range), tuple(cfg.torque_scale_range)) == ((1.0, 1.0), (1.0, 1.0))
    assert not any(hasattr(cfg, k) for k in ("domain_rand", "rate_scale_range", "thrust_bias_range"))
    assert cfg.history_fill_value == -10.0
    assert cfg.distance_reward_shape == "tanh"
    assert cfg.distance_reward_length == 0.8
    assert cfg.distance_reward_curriculum_ratio == 0.0
    assert (cfg.obs_input_hist, cfg.action_input_hist) == (1, 0)
    assert (cfg.action_latency_steps, cfg.action_latency_max_steps, cfg.action_rate_reward_scale) == (0, 0, 0.0)
    env = make_env()
    try:
        # the default reward is the old hard-coded 1 - tanh(d / 0.8)
        d = torch.linspace(0.0, 2.0, 5, device=env.device)
        torch.testing.assert_close(env._distance_reward(d), 1 - torch.tanh(d / 0.8))
        assert env.single_observation_space["policy"].shape == (18,)
        obs, _ = env.reset()
        torch.testing.assert_close(obs["policy"], frame(env))
        torch.testing.assert_close(env.true_state, frame(env))
        cmd = torch.full((N, 4), -0.2, device=env.device)
        obs, *_ = env.step(cmd)
        torch.testing.assert_close(obs["policy"], frame(env))
        torch.testing.assert_close(env.true_state, frame(env))
        # no latency, no hold: the controller gets the command of this very step
        assert not env.delay_steps.any() and not env._hold_cmd.any() and env._hold_ids.numel() == 0
        torch.testing.assert_close(env._actions, cmd)
        assert not env.randomize_actuation and not env.wind_force_world.any()
        env._apply_action()
        assert not env._robot.instantaneous_wrench_composer.active  # no wind, no actuation scaling by default
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
    env = make_env(
        distance_reward_shape=shape, distance_reward_length=length, action_rate_reward_scale=-0.5, **knobs
    )
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        cfg, data, dt = env.cfg, env._robot.data, env.step_dt
        prev = None
        for _ in range(3):
            cmd = rand_cmd(env)
            reward, terms = step_terms(env, cmd)
            rate = torch.zeros(N, device=env.device) if prev is None else (cmd - prev).square().sum(1)
            torch.testing.assert_close(terms["action_rate"], rate * cfg.action_rate_reward_scale * dt)
            prev = cmd
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


def test_stacking_timesteps_and_reset():
    env = make_env(obs_input_hist=K, action_input_hist=M)
    try:
        # space follows the late (Hydra-style) override
        assert env.single_observation_space["policy"].shape == (18 * K + 4 * M,)

        obs, _ = env.reset()
        env.episode_length_buf[:] = 0  # full reset randomises it; no timeout inside the test
        F = [frame(env)]  # F[t]: state after step t
        A = [fill(env)]   # A[t]: clamped command of step t; A[0] = "no command" before the episode
        frames, acts = split(obs)
        torch.testing.assert_close(frames, window(F, 0, K))  # K copies of the first frame
        assert torch.all(acts == FILL)                       # no command yet: the fill, not 0 (a climb)

        for t in range(1, 6):
            a = torch.full((N, 4), 0.05 * t, device=env.device)
            a[:, 0] = 5.0 * (-1) ** t  # out of range -> must be clamped to +-1
            obs, *_ = env.step(a)
            assert not env.reset_buf.any()
            F.append(frame(env))
            A.append(a.clamp(-1.0, 1.0))
            frames, acts = split(obs)
            torch.testing.assert_close(frames, window(F, t, K))
            torch.testing.assert_close(acts, window(A, t, M))
            torch.testing.assert_close(frames[:, -1], F[t])  # newest last
            torch.testing.assert_close(acts[:, -1], A[t])    # the command just sent
        assert not torch.equal(F[5], F[4])                   # frames really change over time

        # force env 0 to time out on the next step; envs 1.. must keep their history
        env.episode_length_buf[0] = env.max_episode_length - 1
        a = torch.full((N, 4), -0.2, device=env.device)
        obs, *_ = env.step(a)
        assert env.reset_buf.tolist() == [True] + [False] * (N - 1)
        F.append(frame(env))
        A.append(a.clamp(-1.0, 1.0))
        t = len(F) - 1
        frames, acts = split(obs)
        torch.testing.assert_close(frames[0], F[t][0].expand(K, 18))  # refilled with the new episode's frame
        assert torch.all(acts[0] == FILL)                              # old episode's commands cleared to the fill
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
        zero = torch.zeros(N, 4, device=env.device)
        A = [fill(env)]  # A[t]: command of step t; A[0] = before the episode
        for t in range(1, 7):
            A.append(rand_cmd(env))
            obs, *_ = env.step(A[t])
            # steps 1..d: hold, the controller gets 0 (never the fill); then the command of step t - d
            torch.testing.assert_close(env._actions, zero if t <= d else A[t - d])
            torch.testing.assert_close(obs["policy"][:, 18:].view(N, M, 4), window(A, t, M))  # obs: commanded

        # env 0 times out at step 7; its old commands must never reach the controller
        env.episode_length_buf[0] = env.max_episode_length - 1
        A.append(rand_cmd(env))
        env.step(A[7])
        assert env.reset_buf.tolist() == [True] + [False] * (N - 1)
        B = [fill(env)]  # env 0's new episode, same indexing
        for s in range(1, 5):
            A.append(rand_cmd(env))
            B.append(A[-1])
            env.step(A[-1])
            torch.testing.assert_close(env._actions[0], zero[0] if s <= d else B[s - d][0])
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
        A = [fill(env)]
        for t in range(1, 8):
            A.append(rand_cmd(env))
            env.step(A[t])
            zero = torch.zeros(4, device=env.device)
            expected = torch.stack([zero if t <= delay[i] else A[t - delay[i]][i] for i in range(N)])
            torch.testing.assert_close(env._actions, expected)
            assert env._hold_cmd.tolist() == [t <= delay[i] for i in range(N)]
    finally:
        env.close()


def test_warmup_hold():
    d = 3
    env = make_env(action_latency_steps=d)
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        spawn = env._robot.data.root_pos_w.clone()
        torch.testing.assert_close(spawn, env._spawn_pos_w)
        A = [fill(env)]
        for t in range(1, d + 1):
            A.append(rand_cmd(env))
            before = {k: v.clone() for k, v in env._episode_sums.items()}
            _, reward, *_ = env.step(A[t])
            assert env._hold_cmd.all()
            # frozen at spawn by an exact m*g, whatever the policy commands
            torch.testing.assert_close(env._robot.data.root_pos_w, spawn, atol=1e-4, rtol=0.0)
            torch.testing.assert_close(env._robot.data.root_lin_vel_w, torch.zeros(N, 3, device=env.device), atol=1e-3, rtol=0.0)
            torch.testing.assert_close(env._thrust[:, 0, 2], env._hold_force)
            assert not env._moment.any()
            assert not reward.any()
            for key in before:
                torch.testing.assert_close(env._episode_sums[key], before[key], msg=key)
        # hold force = weight of the simulated drone
        g = torch.tensor(env.sim.cfg.gravity).norm().item()
        torch.testing.assert_close(env._hold_force, env._robot.root_physx_view.get_masses().sum(1).to(env.device) * g)

        # step d + 1: the controller gets the first command, the drone flies and the reward is back
        A.append(rand_cmd(env))
        _, reward, *_ = env.step(A[d + 1])
        assert not env._hold_cmd.any()
        torch.testing.assert_close(env._actions, A[1])
        assert reward.all()

        # a mid-run reset of env 0 restarts the hold for env 0 only
        env.episode_length_buf[0] = env.max_episode_length - 1
        env.step(rand_cmd(env))
        assert env.reset_buf.tolist() == [True] + [False] * (N - 1)
        new_spawn = env._spawn_pos_w[0].clone()
        for _ in range(d):
            _, reward, *_ = env.step(rand_cmd(env))
            assert env._hold_cmd.tolist() == [True] + [False] * (N - 1)
            assert reward[0] == 0.0 and reward[1:].all()
            torch.testing.assert_close(env._robot.data.root_pos_w[0], new_spawn, atol=1e-4, rtol=0.0)
        env.step(rand_cmd(env))
        assert not env._hold_cmd.any()
    finally:
        env.close()


@pytest.mark.parametrize("d", [0, 2])
def test_action_rate_penalises_commanded_action(d):
    scale = -0.5
    env = make_env(action_latency_steps=d, action_rate_reward_scale=scale)
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        A = [fill(env)]
        for t in range(1, 7):
            A.append(rand_cmd(env))
            before = env._episode_sums["action_rate"].clone()
            env.step(A[t])
            # 0 at step 1 (the previous slot is the fill); then on the commanded actions, hold included
            if t == 1:
                penalty = torch.zeros(N, device=env.device)
            else:
                penalty = scale * (A[t] - A[t - 1]).square().sum(1) * env.step_dt
            torch.testing.assert_close(env._episode_sums["action_rate"] - before, penalty)
            if 1 < t <= d:  # during the hold: penalised, while the state terms stay 0
                assert env._hold_cmd.all() and env._episode_sums["action_rate"].ne(before).all()

        # env 0 resets at step 7; its first step of the new episode sees no jump from the fill
        env.episode_length_buf[0] = env.max_episode_length - 1
        env.step(rand_cmd(env))
        assert env.reset_buf[0]
        assert env._episode_sums["action_rate"][0] == 0.0
        prev = None
        for s in range(1, d + 3):
            c = rand_cmd(env)
            before = env._episode_sums["action_rate"][0].item()
            env.step(c)
            expected = 0.0 if s == 1 else (scale * (c[0] - prev[0]).square().sum() * env.step_dt).item()
            assert env._episode_sums["action_rate"][0].item() - before == pytest.approx(expected, abs=1e-7)
            prev = c
    finally:
        env.close()


def near_hover(env) -> torch.Tensor:
    """Zero rates and a thrust close to m*g (a = 0 is a 1.15 m*g climb)."""
    return torch.tensor([0.0, 0.0, 0.0, -0.23], device=env.device).repeat(env.num_envs, 1)


@pytest.mark.parametrize("rho", [0.0, 0.9])
def test_noise_std_and_correlation(rho):
    n_envs, std = 256, 0.1
    env = make_env(num_envs=n_envs, add_noise=True, noise_std=std, noise_corr=rho)
    try:
        obs, _ = env.reset()
        env.episode_length_buf[:] = 0
        noise = [obs["policy"] - frame(env)]               # measured noise = obs - true state
        torch.testing.assert_close(env.true_state, frame(env))  # the true state stays clean
        hover = near_hover(env)
        for _ in range(20):
            obs, *_ = env.step(hover)
            noise.append(obs["policy"] - frame(env))
            torch.testing.assert_close(env.true_state, frame(env))
        noise = torch.stack(noise) / std                   # [21 steps, envs, 18]; unit std if correct
        assert abs(noise[0].std().item() - 1.0) < 0.1      # full strength right after reset
        # regression of n_t+1 on n_t: slope = rho, residual = the fresh part, std sqrt(1 - rho^2).
        # Residuals are independent, so ~100k samples give a standard error of ~0.003 on both.
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
            obs, *_ = env.step(near_hover(env))
            # obs - true state = bias + white noise of std 0.01; 0.05 is 5 sigma
            torch.testing.assert_close(obs["policy"] - frame(env), bias, atol=0.05, rtol=0)
        env.episode_length_buf[0] = env.max_episode_length - 1
        env.step(near_hover(env))
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

        cmd = torch.empty(n_envs, 4, device=env.device)
        for _ in range(3):
            env.step(cmd.uniform_(-1.0, 1.0))              # random rates tilt the drones
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


def test_setpoints_stay_nominal_under_actuation_scaling():
    """The thrust scale acts on the body only: the rate and thrust setpoints are the nominal mapping."""
    env = make_env(thrust_scale_range=(0.8, 1.2))
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        cmd = rand_cmd(env)
        env.step(cmd)
        torch.testing.assert_close(env._rate_ref, cmd[:, :3] * env.cfg.max_body_rate)
        nominal = env._min_thrust + (cmd[:, 3] + 1) / 2 * (env._max_thrust - env._min_thrust)
        torch.testing.assert_close(env._thrust_pwm[:, 0] * env._ctrl.thrust_cmd_scale, nominal)
    finally:
        env.close()


def test_warmup_hold_ignores_wind_and_scaling():
    """The hold stays an exact m*g: no wind, no actuation scaling; both come back at step d + 1."""
    d = 3
    # 0.1 N of wind on a ~0.027 kg drone is ~4 m/s^2: a leak would move it far beyond 1e-4 m in d steps
    env = make_env(
        action_latency_steps=d, wind_force_range=(0.1, 0.2), thrust_scale_range=(0.7, 0.8), torque_scale_range=(0.7, 0.8)
    )
    try:
        env.reset()
        env.episode_length_buf[:] = 0
        spawn = env._robot.data.root_pos_w.clone()
        composer = env._robot.instantaneous_wrench_composer
        b = env._body_id[0]
        for _ in range(d):
            env.step(rand_cmd(env))
            assert env._hold_cmd.all()
            torch.testing.assert_close(env._robot.data.root_pos_w, spawn, atol=1e-4, rtol=0.0)
            torch.testing.assert_close(env._thrust[:, 0, 2], env._hold_force)
        env._apply_action()  # still held (step d's state): the composer adds nothing
        assert not composer.composed_force_as_torch[:, b].any() and not composer.composed_torque_as_torch[:, b].any()
        composer.reset()

        env.step(rand_cmd(env))  # step d + 1: flying, so wind and scaling are back
        assert not env._hold_cmd.any()
        env._apply_action()
        expected = quat_apply_inverse(env._robot.data.body_link_quat_w[:, b], env.wind_force_world)
        expected = expected + (env.thrust_scale[:, None] - 1) * env._thrust[:, 0]
        torch.testing.assert_close(composer.composed_force_as_torch[:, b], expected)
        torch.testing.assert_close(composer.composed_torque_as_torch[:, b], (env.torque_scale[:, None] - 1) * env._moment[:, 0])
        composer.reset()
    finally:
        env.close()
