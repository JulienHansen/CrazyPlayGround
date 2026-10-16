"""Velocity hovering with domain randomisation (``Vel-Hovering-DR``, WP1).

Adds the sim-to-real knobs on top of ``Vel-Hovering``. Every knob is off by default,
so the task is identical to ``Vel-Hovering``.

Observation layout (oldest first):
    [obs_input_hist x (lin_vel_b(3), desired_pos_b(3)), action_input_hist x action(3)]

Loop latency: the command sent at policy step t reaches the controller at step t + d,
with d (in policy steps) drawn per episode from [action_latency_steps,
action_latency_max_steps]. The action history and the action-rate penalty see the
commanded action; only the controller sees the delayed one.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .vel_hovering import QuadcopterEnv, QuadcopterEnvCfg

BASE_OBS_DIM = 6  # [lin_vel_b(3), desired_pos_b(3)]
ACTION_DIM = 3    # [vx, vy, vz]


@configclass
class DRQuadcopterEnvCfg(QuadcopterEnvCfg):
    obs_input_hist: int = 1     # observation frames in the policy input; 1 = no stacking
    action_input_hist: int = 0  # past commands in the policy input; 0 = none
    # delay in policy steps between sending a command and the controller using it
    action_latency_steps: int = 0
    action_latency_max_steps: int = 0      # > action_latency_steps: drawn per episode; 0 = fixed
    action_rate_reward_scale: float = 0.0  # weight on ||c_t - c_{t-1}||^2; negative = penalty


class DRQuadcopterEnv(QuadcopterEnv):
    cfg: DRQuadcopterEnvCfg

    def __init__(self, cfg: DRQuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        # Hydra applies CLI overrides after the cfg is built, so validate and size the
        # spaces here; a wrong size makes skrl silently reshape [N, D] into [N*x, D/x].
        if cfg.obs_input_hist < 1:
            raise ValueError(f"obs_input_hist must be >= 1, got {cfg.obs_input_hist}")
        if cfg.action_input_hist < 0:
            raise ValueError(f"action_input_hist must be >= 0, got {cfg.action_input_hist}")
        if cfg.action_latency_steps < 0:
            raise ValueError(f"action_latency_steps must be >= 0, got {cfg.action_latency_steps}")
        if cfg.action_latency_max_steps and cfg.action_latency_max_steps < cfg.action_latency_steps:
            raise ValueError(
                f"action_latency_max_steps must be 0 or >= action_latency_steps "
                f"({cfg.action_latency_steps}), got {cfg.action_latency_max_steps}"
            )
        cfg.observation_space = BASE_OBS_DIM * cfg.obs_input_hist + ACTION_DIM * cfg.action_input_hist
        super().__init__(cfg, render_mode, **kwargs)

        self._delay_min_steps = cfg.action_latency_steps
        self._delay_max_steps = max(cfg.action_latency_steps, cfg.action_latency_max_steps)
        self._obs_input_hist = torch.zeros(self.num_envs, cfg.obs_input_hist, BASE_OBS_DIM, device=self.device)
        # last commanded actions, newest last; feeds the action history, the latency and the
        # action-rate penalty. Length >= 2 so the penalty always has a previous command.
        buf_len = max(cfg.action_input_hist, self._delay_max_steps + 1, 2)
        self._cmd_hist = torch.zeros(self.num_envs, buf_len, ACTION_DIM, device=self.device)
        self._delay_steps = torch.full(
            (self.num_envs,), self._delay_min_steps, dtype=torch.long, device=self.device
        )
        self._env_idx = torch.arange(self.num_envs, device=self.device)
        # True for envs just reset: _get_observations refills their window (see its docstring).
        self._hist_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_sums["action_rate"] = torch.zeros(self.num_envs, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Record the commanded action, then hand the controller the one sent ``_delay_steps``
        steps ago (zero until the episode is that old)."""
        commanded = actions.clamp(-1.0, 1.0)
        self._cmd_hist = torch.cat([self._cmd_hist[:, 1:], commanded.unsqueeze(1)], dim=1)
        # each env reads its own slot: newest is the last one, d steps back is d slots earlier
        slot = self._cmd_hist.shape[1] - 1 - self._delay_steps  # [N]
        applied = self._cmd_hist[self._env_idx, slot]          # [N, 3]: env i, slot[i]
        super()._pre_physics_step(applied)

    def _get_observations(self) -> dict:
        """Shift the newest frame into the observation window and append the action history.

        After a reset the window still holds the last frames of the previous episode, which
        had a different position, velocity and goal. Left as is, the first K-1 observations
        would mix two episodes and show the policy a fake jump. Envs flagged in
        ``_hist_fresh`` therefore get K copies of their current frame instead. The refill
        happens here, not in ``_reset_idx``, because ``step()`` calls ``_reset_idx`` before
        ``_get_observations``: the frame computed here is the new episode's first one, so it
        is computed once and the window matches it exactly, noise included.
        """
        obs = super()._get_observations()["policy"]
        new = obs.unsqueeze(1)
        hist = torch.cat([self._obs_input_hist[:, 1:], new], dim=1)
        self._obs_input_hist = torch.where(self._hist_fresh[:, None, None], new.expand_as(hist), hist)
        self._hist_fresh[:] = False
        # the M newest commands: M slots starting at slot L - M (M = 0 -> empty, adds nothing)
        m = self.cfg.action_input_hist
        act_hist = self._cmd_hist.narrow(1, self._cmd_hist.shape[1] - m, m)
        return {"policy": torch.cat([self._obs_input_hist.flatten(1), act_hist.flatten(1)], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        """Base reward plus the action-rate term on the *commanded* actions. Using the
        delayed action the controller gets would credit each change d steps late."""
        rate = torch.sum(torch.square(self._cmd_hist[:, -1] - self._cmd_hist[:, -2]), dim=1)
        action_rate = rate * self.cfg.action_rate_reward_scale * self.step_dt
        self._episode_sums["action_rate"] += action_rate
        return super()._get_rewards() + action_rate

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the base env, clear the command history, draw the episode's latency, and
        flag the observation window for a refill on the next ``_get_observations``."""
        super()._reset_idx(env_ids)
        self._cmd_hist[env_ids] = 0.0
        self._delay_steps[env_ids] = torch.randint(
            self._delay_min_steps, self._delay_max_steps + 1, (len(env_ids),), device=self.device
        )
        self._hist_fresh[env_ids] = True
