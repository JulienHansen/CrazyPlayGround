"""Robustness-oriented variant of the velocity hovering environment.

Extends :mod:`vel_hovering` with the sim-to-real knobs, all disabled by default so
the environment behaves exactly like ``Vel-Hovering`` unless overridden:

1. **Observation noise** -- white or temporally correlated (AR(1)), with an optional
   per-episode bias. Real EKF error is correlated and biased, not white; white noise
   is unrealistically easy to filter by temporal averaging.
2. **External disturbances** -- per-episode constant force/torque bias plus random
   intra-episode gusts (decaying impulses), applied as an external wrench in the body
   frame on top of the controller output.
3. **Observation history window** -- the last ``history_len`` observations.
4. **Action history** -- the last ``action_hist_len`` commanded actions. Without this
   the policy cannot perceive its own previous command, so it can neither regulate its
   step-to-step change directly nor infer a disturbance (which needs the residual
   between what was commanded and what happened).
5. **Loop latency** -- ``action_latency_steps`` (optionally randomised per episode up
   to ``action_latency_max``) delays the applied command, modelling radio + estimator
   dead time. Measured effect on the deployed policy, everything else clean:
   0 steps -> 0.0060 chatter, 2 -> 0.0300, 4 -> 0.0517, against 0.0690 on real
   hardware. Delay reproduces most of the observed sim-to-real chatter gap while
   barely moving position error -- the same asymmetry seen on the real drone.

Observation layout (all oldest-first):
    [ history_len x (lin_vel_b(3), desired_pos_b(3)) ,  action_hist_len x action(3) ]
    observation_space = 6 * history_len + 3 * action_hist_len

The skrl/rsl_rl configs size their input layers from ``observation_space``, so no
agent-config change is needed.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from .vel_hovering import QuadcopterEnv as _BaseQuadcopterEnv
from .vel_hovering import QuadcopterEnvCfg as _BaseQuadcopterEnvCfg

BASE_OBS_DIM = 6   # [lin_vel_b(3), desired_pos_b(3)]
ACTION_DIM = 3     # [vx, vy, vz]


@configclass
class RobustQuadcopterEnvCfg(_BaseQuadcopterEnvCfg):
    """Velocity hovering cfg with noise / disturbance / history / latency knobs."""

    # ── Observation history (frame stacking) ─────────────────────────────────
    history_len: int = 1          # 1 = no stacking (identical to Vel-Hovering)

    # ── Action history ───────────────────────────────────────────────────────
    action_hist_len: int = 0      # 0 = policy does not see its previous commands

    # ── Loop latency ─────────────────────────────────────────────────────────
    # Policy steps of dead time between commanding an action and it being applied.
    # At 100 Hz, 2 steps = 20 ms (radio + estimator lag is of this order).
    action_latency_steps: int = 0
    # If > action_latency_steps, the per-episode delay is sampled uniformly from
    # [action_latency_steps, action_latency_max]. The true hardware latency is not
    # measured, so randomising is safer than baking in one guessed value.
    action_latency_max: int = 0

    # ── Observation noise shape ──────────────────────────────────────────────
    # noise_corr: AR(1) coefficient. 0.0 = white (default), 0.9 = strongly correlated.
    noise_corr: float = 0.0
    # Per-episode constant observation bias, as a std (0 = unbiased).
    noise_bias_std: float = 0.0

    # ── External disturbances ────────────────────────────────────────────────
    disturb: bool = False
    # Crazyflie weight ~0.265 N, so 0.02 N is ~7.5% of weight.
    disturb_force_bias_range: tuple[float, float] = (-0.02, 0.02)       # N per axis
    disturb_torque_bias_range: tuple[float, float] = (-2.0e-4, 2.0e-4)  # N.m per axis
    disturb_gust_prob: float = 0.0                                       # per policy step
    disturb_gust_force_range: tuple[float, float] = (-0.06, 0.06)
    disturb_gust_torque_range: tuple[float, float] = (-6.0e-4, 6.0e-4)
    disturb_gust_decay: float = 0.92

    # ── Anti-chatter ─────────────────────────────────────────────────────────
    # Penalty on ||a_t - a_{t-1}||^2. Negative = penalty.
    action_rate_reward_scale: float = 0.0

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        if self.history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {self.history_len}")
        if self.action_hist_len < 0:
            raise ValueError(f"action_hist_len must be >= 0, got {self.action_hist_len}")
        if self.action_latency_steps < 0:
            raise ValueError(f"action_latency_steps must be >= 0, got {self.action_latency_steps}")
        if self.action_latency_max and self.action_latency_max < self.action_latency_steps:
            raise ValueError("action_latency_max must be >= action_latency_steps")
        self.observation_space = BASE_OBS_DIM * self.history_len + ACTION_DIM * self.action_hist_len


class RobustQuadcopterEnv(_BaseQuadcopterEnv):
    """Velocity hovering with observation/action history, latency and disturbances."""

    cfg: RobustQuadcopterEnvCfg

    def __init__(self, cfg: RobustQuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        # `__post_init__` runs when the cfg is built, i.e. BEFORE Hydra applies CLI
        # overrides, so recompute here -- this runs after all overrides and before
        # DirectRLEnv reads cfg.observation_space. Getting it wrong makes skrl
        # reinterpret [N, D] as [N*x, D/x].
        if int(cfg.history_len) < 1:
            raise ValueError(f"history_len must be >= 1, got {cfg.history_len}")
        cfg.observation_space = (BASE_OBS_DIM * int(cfg.history_len)
                                 + ACTION_DIM * int(cfg.action_hist_len))

        super().__init__(cfg, render_mode, **kwargs)

        self._ensure_buffers()
        self._prev_actions = torch.zeros_like(self._actions)
        if self.cfg.action_rate_reward_scale != 0.0:
            self._episode_sums["action_rate"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # ---------- buffers ----------

    def _ensure_buffers(self):
        """Allocate buffers lazily: the base __init__ may reach _reset_idx or
        _get_observations before our __init__ body runs."""
        if getattr(self, "_buffers_ready", False):
            return
        n, dev = self.num_envs, self.device
        self._k = int(getattr(self.cfg, "history_len", 1))
        self._m = int(getattr(self.cfg, "action_hist_len", 0))
        self._d = int(getattr(self.cfg, "action_latency_steps", 0))

        self._d_max = max(self._d, int(getattr(self.cfg, "action_latency_max", 0)))
        self._obs_hist = torch.zeros(n, self._k, BASE_OBS_DIM, device=dev)
        self._act_hist = torch.zeros(n, max(self._m, 1), ACTION_DIM, device=dev)
        # buffer holds the newest at index -1, so a delay of d reads index -1-d
        self._act_delay = torch.zeros(n, self._d_max + 1, ACTION_DIM, device=dev)
        self._delay_steps = torch.full((n,), self._d, dtype=torch.long, device=dev)
        self._env_arange = torch.arange(n, device=dev)

        self._noise_state = torch.zeros(n, BASE_OBS_DIM, device=dev)
        self._noise_bias = torch.zeros(n, BASE_OBS_DIM, device=dev)

        self._dist_force = torch.zeros(n, 3, device=dev)
        self._dist_torque = torch.zeros(n, 3, device=dev)
        self._gust_force = torch.zeros(n, 3, device=dev)
        self._gust_torque = torch.zeros(n, 3, device=dev)
        self._buffers_ready = True

    @staticmethod
    def _uniform(shape, rng, device):
        lo, hi = rng
        return torch.empty(*shape, device=device).uniform_(lo, hi)

    # ---------- observation ----------

    def _base_obs(self) -> torch.Tensor:
        """Clean 6-D observation (same content as the base env, before noise)."""
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        return torch.cat([self._robot.data.root_lin_vel_b, desired_pos_b], dim=-1)

    def _noisy_obs(self) -> torch.Tensor:
        obs = self._base_obs()
        if not self.cfg.add_noise:
            return obs
        rho = float(self.cfg.noise_corr)
        eps = torch.randn_like(obs)
        if rho > 0.0:
            # AR(1): correlated in time, unit stationary variance
            self._noise_state = rho * self._noise_state + (1.0 - rho ** 2) ** 0.5 * eps
            n = self._noise_state
        else:
            n = eps
        return obs + n * self.cfg.noise_std + self._noise_bias

    def _get_observations(self) -> dict:
        self._ensure_buffers()
        base = self._noisy_obs()

        if self._k == 1:
            obs_part = base
        else:
            self._obs_hist = torch.roll(self._obs_hist, shifts=-1, dims=1)
            self._obs_hist[:, -1, :] = base
            obs_part = self._obs_hist.reshape(self.num_envs, self._k * BASE_OBS_DIM)

        if self._m > 0:
            act_part = self._act_hist[:, -self._m:, :].reshape(self.num_envs, self._m * ACTION_DIM)
            return {"policy": torch.cat([obs_part, act_part], dim=-1)}
        return {"policy": obs_part}

    # ---------- action: history, latency, disturbances ----------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._ensure_buffers()
        self._prev_actions = self._actions.clone()

        commanded = actions.clone().clamp(-1.0, 1.0)

        # record what the policy commanded (this is what the real deploy script knows)
        if self._m > 0:
            self._act_hist = torch.roll(self._act_hist, shifts=-1, dims=1)
            self._act_hist[:, -1, :] = commanded

        # loop dead time: apply the command issued `delay_steps` steps ago
        if self._d_max > 0:
            self._act_delay = torch.roll(self._act_delay, shifts=-1, dims=1)
            self._act_delay[:, -1, :] = commanded
            idx = (self._act_delay.shape[1] - 1) - self._delay_steps
            applied = self._act_delay[self._env_arange, idx]
        else:
            applied = commanded

        super()._pre_physics_step(applied)

        if not self.cfg.disturb:
            return
        self._gust_force *= self.cfg.disturb_gust_decay
        self._gust_torque *= self.cfg.disturb_gust_decay
        if self.cfg.disturb_gust_prob > 0.0:
            fire = torch.rand(self.num_envs, device=self.device) < self.cfg.disturb_gust_prob
            if bool(fire.any()):
                k = int(fire.sum())
                self._gust_force[fire] = self._uniform((k, 3), self.cfg.disturb_gust_force_range, self.device)
                self._gust_torque[fire] = self._uniform((k, 3), self.cfg.disturb_gust_torque_range, self.device)

    def _apply_action(self):
        # Base fills self._thrust[:, 0, 2] / self._moment from the cascade PID and
        # applies the wrench; add the disturbance and re-apply (last call wins).
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
        n = len(env_ids)

        if hasattr(self, "_prev_actions"):
            self._prev_actions[env_ids] = 0.0
        self._act_hist[env_ids] = 0.0
        self._act_delay[env_ids] = 0.0
        if self._d_max > self._d:
            self._delay_steps[env_ids] = torch.randint(
                self._d, self._d_max + 1, (n,), device=self.device)
        else:
            self._delay_steps[env_ids] = self._d
        self._noise_state[env_ids] = 0.0
        self._noise_bias[env_ids] = (
            torch.randn(n, BASE_OBS_DIM, device=self.device) * self.cfg.noise_bias_std
            if (self.cfg.add_noise and self.cfg.noise_bias_std > 0.0) else 0.0
        )

        if self.cfg.disturb:
            self._dist_force[env_ids] = self._uniform((n, 3), self.cfg.disturb_force_bias_range, self.device)
            self._dist_torque[env_ids] = self._uniform((n, 3), self.cfg.disturb_torque_bias_range, self.device)
        else:
            self._dist_force[env_ids] = 0.0
            self._dist_torque[env_ids] = 0.0
        self._gust_force[env_ids] = 0.0
        self._gust_torque[env_ids] = 0.0

        # prime the observation window with the post-reset observation so no stale
        # frames leak across the reset boundary
        if self._k > 1:
            base = self._base_obs()
            self._obs_hist[env_ids] = base[env_ids].unsqueeze(1).repeat(1, self._k, 1)
