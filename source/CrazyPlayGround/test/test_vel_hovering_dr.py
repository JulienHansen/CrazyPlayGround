"""Checks for Vel-Hovering-DR: observation/action stacking order and timestep indexing.

Run (conda env DragonFly, from CrazyPlayGround/):
    python -m pytest source/CrazyPlayGround/test/test_vel_hovering_dr.py -v
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402

import omni.usd  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

from CrazyPlayGround.tasks.direct.hovering.vel_hovering_dr import DRQuadcopterEnv, DRQuadcopterEnvCfg  # noqa: E402

N, K, M = 4, 3, 2


def make_env(**knobs) -> DRQuadcopterEnv:
    omni.usd.get_context().new_stage()
    cfg = DRQuadcopterEnvCfg()
    cfg.scene.num_envs = N
    cfg.debug_vis = False
    for key, value in knobs.items():  # set after the cfg is built, as Hydra does
        setattr(cfg, key, value)
    return DRQuadcopterEnv(cfg)


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


@pytest.mark.parametrize("knobs", [{"obs_input_hist": 0}, {"action_input_hist": -1}])
def test_invalid_knobs_raise(knobs):
    with pytest.raises(ValueError):
        make_env(**knobs)


def test_default_is_vel_hovering():
    env = make_env()
    try:
        assert env.single_observation_space["policy"].shape == (6,)
        obs, _ = env.reset()
        torch.testing.assert_close(obs["policy"], frame(env))
        obs, *_ = env.step(torch.full((N, 3), 0.3, device=env.device))
        torch.testing.assert_close(obs["policy"], frame(env))
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
