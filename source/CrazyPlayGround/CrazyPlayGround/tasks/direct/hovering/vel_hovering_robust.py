"""Robustness-oriented variant of the velocity hovering environment.

Extends :mod:`vel_hovering` with three sim-to-real knobs, all disabled by default
so the environment behaves exactly like ``Vel-Hovering`` unless overridden:

1. **Observation noise** -- reuses the base ``add_noise`` / ``noise_std``.
2. **External disturbances** -- a per-episode constant force/torque bias plus
   random intra-episode gusts (decaying impulses), applied as an external wrench
   in the body frame on top of the controller output.
3. **Observation history window** -- the policy sees the last ``history_len``
   observations concatenated, giving it the temporal context needed to infer the
   disturbance/noise it is subject to (implicit online adaptation).

Motivation: the measured sim-to-real gap on the real Crazyflie was concentrated in
control smoothness -- the deployed policy chattered its velocity command ~10x more
on hardware than in sim -- because it was trained on noiseless observations and
undisturbed dynamics. An optional ``action_rate_reward_scale`` penalty directly
discourages that chatter.

A K-step window multiplies the observation size, so ``observation_space`` becomes
``6 * history_len``; the skrl/rsl_rl configs size their input layers from this
field automatically, so no agent-config change is needed.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .vel_hovering import QuadcopterEnv as _BaseQuadcopterEnv
from .vel_hovering import QuadcopterEnvCfg as _BaseQuadcopterEnvCfg

BASE_OBS_DIM = 6  # [lin_vel_b(3), desired_pos_b(3)]


@configclass
class RobustQuadcopterEnvCfg(_BaseQuadcopterEnvCfg):
    """Velocity hovering cfg with noise / disturbance / history knobs."""

    # ── Observation history (frame stacking) ─────────────────────────────────
    # 1 = no stacking (identical to Vel-Hovering).
    history_len: int = 1

    # ── External disturbances ────────────────────────────────────────────────
    disturb: bool = False
    # Per-episode constant wrench bias (body frame). Crazyflie weight ~0.265 N,
    # so 0.02 N is ~7.5% of weight -- a meaningful but survivable bias.
    disturb_force_bias_range: tuple[float, float] = (-0.02, 0.02)      # N, per axis
    disturb_torque_bias_range: tuple[float, float] = (-2.0e-4, 2.0e-4)  # N.m, per axis
    # Intra-episode gusts: with this probability per policy step a new impulse is
    # drawn, then decays geometrically.
    disturb_gust_prob: float = 0.0
    disturb_gust_force_range: tuple[float, float] = (-0.06, 0.06)       # N, per axis
    disturb_gust_torque_range: tuple[float, float] = (-6.0e-4, 6.0e-4)  # N.m, per axis
    disturb_gust_decay: float = 0.92                                     # per policy step

    # ── Anti-chatter ─────────────────────────────────────────────────────────
    # Penalty on ||a_t - a_{t-1}||^2. Negative = penalty (same sign convention as
    # the other reward scales).
    action_rate_reward_scale: float = 0.0

    def __post_init__(self):
        # DirectRLEnvCfg/configclass may or may not define __post_init__.
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        if self.history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {self.history_len}")
        self.observation_space = BASE_OBS_DIM * self.history_len


class RobustQuadcopterEnv(_BaseQuadcopterEnv):
    """Velocity hovering with observation history and external disturbances."""

    cfg: RobustQuadcopterEnvCfg

    def __init__(self, cfg: RobustQuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        # `__post_init__` runs when the cfg object is built, i.e. BEFORE Hydra applies
        # `env.history_len=K` from the CLI, so recompute here (this runs after all
        # overrides and before DirectRLEnv reads cfg.observation_space to build the
        # gym spaces). Getting this wrong makes skrl reinterpret [N, 6K] as [N*K, 6].
        if int(cfg.history_len) < 1:
            raise ValueError(f"history_len must be >= 1, got {cfg.history_len}")
        cfg.observation_space = BASE_OBS_DIM * int(cfg.history_len)

        super().__init__(cfg, render_mode, **kwargs)

        self._k = int(self.cfg.history_len)
        self._ensure_buffers()

        self._prev_actions = torch.zeros_like(self._actions)
        if self.cfg.action_rate_reward_scale != 0.0:
            self._episode_sums["action_rate"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # ---------- buffers ----------

    def _ensure_buffers(self):
        """Allocate history / disturbance buffers (lazily: base __init__ may call
        into _reset_idx or _get_observations before our __init__ body runs)."""
        if getattr(self, "_buffers_ready", False):
            return
        k = int(getattr(self.cfg, "history_len", 1))
        self._k = k
        self._obs_hist = torch.zeros(self.num_envs, k, BASE_OBS_DIM, device=self.device)
        self._dist_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._dist_torque = torch.zeros(self.num_envs, 3, device=self.device)
        self._gust_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._gust_torque = torch.zeros(self.num_envs, 3, device=self.device)
        self._buffers_ready = True

    @staticmethod
    def _uniform(shape, rng, device):
        lo, hi = rng
        return torch.empty(*shape, device=device).uniform_(lo, hi)

    # ---------- observation window ----------

    def _get_observations(self) -> dict:
        base = super()._get_observations()["policy"]  # [N, 6], noise already applied
        self._ensure_buffers()
        if self._k == 1:
            return {"policy": base}
        # oldest first, newest last
        self._obs_hist = torch.roll(self._obs_hist, shifts=-1, dims=1)
        self._obs_hist[:, -1, :] = base
        return {"policy": self._obs_hist.reshape(self.num_envs, self._k * BASE_OBS_DIM)}

    # ---------- disturbances ----------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions = self._actions.clone()
        super()._pre_physics_step(actions)

        if not self.cfg.disturb:
            return
        self._ensure_buffers()
        # decay any active gust, then possibly trigger a new one
        self._gust_force *= self.cfg.disturb_gust_decay
        self._gust_torque *= self.cfg.disturb_gust_decay
        if self.cfg.disturb_gust_prob > 0.0:
            fire = torch.rand(self.num_envs, device=self.device) < self.cfg.disturb_gust_prob
            if bool(fire.any()):
                n = int(fire.sum())
                self._gust_force[fire] = self._uniform((n, 3), self.cfg.disturb_gust_force_range, self.device)
                self._gust_torque[fire] = self._uniform((n, 3), self.cfg.disturb_gust_torque_range, self.device)

    def _apply_action(self):
        # Base fills self._thrust[:, 0, 2] / self._moment from the cascade PID and
        # applies the wrench; we add the disturbance and re-apply (last call wins).
        super()._apply_action()
        if not self.cfg.disturb:
            return
        self._ensure_buffers()
        fx = self._dist_force + self._gust_force
        tq = self._dist_torque + self._gust_torque
        # x/y are never reset by the base -> assign (not +=) to avoid accumulation.
        self._thrust[:, 0, 0] = fx[:, 0]
        self._thrust[:, 0, 1] = fx[:, 1]
        self._thrust[:, 0, 2] += fx[:, 2]
        self._moment[:, 0, :] += tq
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # ---------- reward ----------

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        scale = self.cfg.action_rate_reward_scale
        if scale != 0.0:
            action_rate = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)
            penalty = action_rate * scale * self.step_dt
            reward = reward + penalty
            if "action_rate" in self._episode_sums:
                self._episode_sums["action_rate"] += penalty
        return reward

    # ---------- reset ----------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self._ensure_buffers()
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if hasattr(self, "_prev_actions"):
            self._prev_actions[env_ids] = 0.0

        # fresh per-episode disturbance bias; clear any in-flight gust
        n = len(env_ids)
        if self.cfg.disturb:
            self._dist_force[env_ids] = self._uniform((n, 3), self.cfg.disturb_force_bias_range, self.device)
            self._dist_torque[env_ids] = self._uniform((n, 3), self.cfg.disturb_torque_bias_range, self.device)
        else:
            self._dist_force[env_ids] = 0.0
            self._dist_torque[env_ids] = 0.0
        self._gust_force[env_ids] = 0.0
        self._gust_torque[env_ids] = 0.0

        # prime the history with the post-reset observation so no stale frames from
        # the previous episode leak across the reset boundary
        if self._k > 1:
            base = _BaseQuadcopterEnv._get_observations(self)["policy"]
            self._obs_hist[env_ids] = base[env_ids].unsqueeze(1).repeat(1, self._k, 1)
