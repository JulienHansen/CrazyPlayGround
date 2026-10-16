"""Velocity hovering with domain randomisation (``Vel-Hovering-DR``, WP1).

Adds the sim-to-real knobs on top of ``Vel-Hovering``. Every knob is off by default,
so the task is identical to ``Vel-Hovering``.

Observation layout (oldest first):
    [obs_input_hist x (lin_vel_b(3), desired_pos_b(3)), action_input_hist x action(3)]
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


class DRQuadcopterEnv(QuadcopterEnv):
    cfg: DRQuadcopterEnvCfg

    def __init__(self, cfg: DRQuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        # Hydra applies CLI overrides after the cfg is built, so validate and size the
        # spaces here; a wrong size makes skrl silently reshape [N, D] into [N*x, D/x].
        if cfg.obs_input_hist < 1:
            raise ValueError(f"obs_input_hist must be >= 1, got {cfg.obs_input_hist}")
        if cfg.action_input_hist < 0:
            raise ValueError(f"action_input_hist must be >= 0, got {cfg.action_input_hist}")
        cfg.observation_space = BASE_OBS_DIM * cfg.obs_input_hist + ACTION_DIM * cfg.action_input_hist
        super().__init__(cfg, render_mode, **kwargs)

        self._obs_input_hist = torch.zeros(self.num_envs, cfg.obs_input_hist, BASE_OBS_DIM, device=self.device)
        self._act_input_hist = torch.zeros(self.num_envs, cfg.action_input_hist, ACTION_DIM, device=self.device)
        # True for envs just reset: _get_observations refills their window (see its docstring).
        self._hist_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Record the commanded action in the action history, then hand it to the base env."""
        commanded = actions.clamp(-1.0, 1.0)
        if self.cfg.action_input_hist:
            self._act_input_hist = torch.cat([self._act_input_hist[:, 1:], commanded.unsqueeze(1)], dim=1)
        super()._pre_physics_step(commanded)

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
        return {"policy": torch.cat([self._obs_input_hist.flatten(1), self._act_input_hist.flatten(1)], dim=-1)}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the base env, clear the action history, and flag the observation window
        for a refill on the next ``_get_observations`` (see its docstring)."""
        super()._reset_idx(env_ids)
        self._act_input_hist[env_ids] = 0.0
        self._hist_fresh[env_ids] = True
