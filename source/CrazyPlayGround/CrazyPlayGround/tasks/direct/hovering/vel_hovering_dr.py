"""Velocity hovering with domain randomisation (``Vel-Hovering-DR``, WP1).

Adds the sim-to-real knobs on top of ``Vel-Hovering``. Every knob is off by default,
so the task is identical to ``Vel-Hovering``.

Observation layout (oldest first):
    [obs_input_hist x (lin_vel_b(3), desired_pos_b(3)), action_input_hist x action(3)]

Loop latency: the command sent at policy step t reaches the controller at step t + d,
with d (in policy steps) drawn per episode from [action_latency_steps,
action_latency_max_steps]. The action history and the action-rate penalty see the
commanded action; only the controller sees the delayed one.

Observation noise (``add_noise=True``): AR(1) noise of std ``noise_std`` and correlation
``noise_corr`` from one policy step to the next, plus a constant offset drawn per episode.

Wind: a constant world-frame force per episode, with random strength and 3-D direction,
added on top of the controller's thrust.

Actuation randomisation: per episode, the thrust and moment that reach the body are the
controller's output times random factors; the cascade PID still assumes nominal motors, so
the policy has to absorb the mismatch.

Distance reward shape: ``"tanh"`` (the base env's, almost flat near the goal) or ``"exp"``,
a single exponential that is steeper near the goal.
"""

from __future__ import annotations

import logging

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from .vel_hovering import QuadcopterEnv, QuadcopterEnvCfg

logger = logging.getLogger(__name__)

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
    # observation noise shape; only used with add_noise=True (noise_std is the base env's)
    noise_corr: float = 0.0      # AR(1) correlation from one policy step to the next, in [0, 1); 0 = white
    noise_bias_std: float = 0.0  # std of a constant offset per channel, drawn per episode
    # wind: strength [N] drawn uniformly from this range at each reset, direction uniform in 3-D,
    # constant in the world frame for the whole episode. (0, 0) = no wind; weight is ~0.265 N.
    wind_force_range: tuple[float, float] = (0.0, 0.0)
    # actuation effectiveness, drawn uniformly per episode; (1, 1) = off. The body gets
    # scale x the controller output (battery sag, worn props, build differences).
    thrust_scale_range: tuple[float, float] = (1.0, 1.0)
    torque_scale_range: tuple[float, float] = (1.0, 1.0)
    # distance-to-goal reward shape; both are 1 at the goal, so the term's weight keeps its meaning:
    # "tanh" = base env's 1 - tanh(d / 0.8), almost flat near the goal (94 % of max at 5 cm);
    # "exp"  = exp(-d / distance_reward_length), steeper near the goal (slope 1 / length at d = 0)
    distance_reward_shape: str = "tanh"
    distance_reward_length: float = 0.3  # metres, "exp" only: the reward is 1/e (37 %) at this distance


def check_cfg(cfg: DRQuadcopterEnvCfg):
    """Reject knob values that can't work. Called by the env constructor, after Hydra overrides."""
    wind_lo, wind_hi = cfg.wind_force_range
    thrust_lo, thrust_hi = cfg.thrust_scale_range
    torque_lo, torque_hi = cfg.torque_scale_range
    checks = [
        (cfg.obs_input_hist >= 1, f"obs_input_hist must be >= 1, got {cfg.obs_input_hist}"),
        (cfg.action_input_hist >= 0, f"action_input_hist must be >= 0, got {cfg.action_input_hist}"),
        (cfg.action_latency_steps >= 0, f"action_latency_steps must be >= 0, got {cfg.action_latency_steps}"),
        (
            not cfg.action_latency_max_steps or cfg.action_latency_max_steps >= cfg.action_latency_steps,
            f"action_latency_max_steps must be 0 or >= action_latency_steps ({cfg.action_latency_steps}), "
            f"got {cfg.action_latency_max_steps}",
        ),
        (0.0 <= cfg.noise_corr < 1.0, f"noise_corr must be in [0, 1), got {cfg.noise_corr}"),
        (cfg.noise_bias_std >= 0.0, f"noise_bias_std must be >= 0, got {cfg.noise_bias_std}"),
        (0.0 <= wind_lo <= wind_hi, f"wind_force_range must satisfy 0 <= min <= max, got {cfg.wind_force_range}"),
        # 0 would mean no thrust / no moment at all: the drone can't fly
        (0.0 < thrust_lo <= thrust_hi, f"thrust_scale_range must satisfy 0 < min <= max, got {cfg.thrust_scale_range}"),
        (0.0 < torque_lo <= torque_hi, f"torque_scale_range must satisfy 0 < min <= max, got {cfg.torque_scale_range}"),
        (
            cfg.distance_reward_shape in ("tanh", "exp"),
            f"distance_reward_shape must be 'tanh' or 'exp', got {cfg.distance_reward_shape!r}",
        ),
        (cfg.distance_reward_length > 0.0, f"distance_reward_length must be > 0, got {cfg.distance_reward_length}"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)
    if not cfg.add_noise and (cfg.noise_corr or cfg.noise_bias_std):
        logger.warning("noise_corr and noise_bias_std have no effect without add_noise=True")


class DRQuadcopterEnv(QuadcopterEnv):
    cfg: DRQuadcopterEnvCfg

    def __init__(self, cfg: DRQuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        # Hydra applies CLI overrides after the cfg is built, so validate and size the
        # spaces here; a wrong size makes skrl silently reshape [N, D] into [N*x, D/x].
        check_cfg(cfg)
        cfg.observation_space = BASE_OBS_DIM * cfg.obs_input_hist + ACTION_DIM * cfg.action_input_hist
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device

        self.delay_min_steps = cfg.action_latency_steps
        self.delay_max_steps = max(cfg.action_latency_steps, cfg.action_latency_max_steps)
        self.obs_window = torch.zeros(n, cfg.obs_input_hist, BASE_OBS_DIM, device=dev)
        # True for envs just reset: _get_observations refills their window (see its docstring)
        self.obs_window_fresh = torch.ones(n, dtype=torch.bool, device=dev)
        # last commanded actions, newest last; feeds the action history, the latency and the
        # action-rate penalty. Length >= 2 so the penalty always has a previous command.
        buf_len = max(cfg.action_input_hist, self.delay_max_steps + 1, 2)
        self.cmd_history = torch.zeros(n, buf_len, ACTION_DIM, device=dev)
        self.delay_steps = torch.full((n,), self.delay_min_steps, dtype=torch.long, device=dev)
        self.env_idx = torch.arange(n, device=dev)
        self._episode_sums["action_rate"] = torch.zeros(n, device=dev)
        # noise-free 6-D state of the last observation, before noise (also for the privileged critic)
        self.true_state = torch.zeros(n, BASE_OBS_DIM, device=dev)
        # AR(1) noise state with unit variance, carried between steps; redrawn at every reset
        self.obs_noise_state = torch.randn(n, BASE_OBS_DIM, device=dev)
        self.obs_noise_bias = torch.zeros(n, BASE_OBS_DIM, device=dev)
        self.wind_force_world = torch.zeros(n, 3, device=dev)  # drawn per episode in _reset_idx
        # actuation scales, drawn per episode in _reset_idx (stay 1 when the ranges are (1, 1))
        self.thrust_scale = torch.ones(n, device=dev)
        self.torque_scale = torch.ones(n, device=dev)
        # skip the extra composer call entirely when nothing is randomised (tuple: Hydra may give a list)
        self.randomize_actuation = (tuple(cfg.thrust_scale_range), tuple(cfg.torque_scale_range)) != ((1.0, 1.0),) * 2

    def _pre_physics_step(self, actions: torch.Tensor):
        """Record the commanded action, then hand the controller the one sent ``delay_steps``
        steps ago (zero until the episode is that old)."""
        commanded = actions.clamp(-1.0, 1.0)
        self.cmd_history = torch.cat([self.cmd_history[:, 1:], commanded.unsqueeze(1)], dim=1)
        # each env reads its own slot: newest is the last one, d steps back is d slots earlier
        slot = self.cmd_history.shape[1] - 1 - self.delay_steps  # [N]
        applied = self.cmd_history[self.env_idx, slot]          # [N, 3]: env i, slot[i]
        super()._pre_physics_step(applied)

    def _apply_action(self):
        """The controller's thrust and moment (base env), scaled by the episode's actuation
        effectiveness, plus the wind for this physics step.

        Both extras go into Isaac Lab's instantaneous wrench composer, which Isaac Lab adds to
        the base env's thrust (permanent composer) and clears after each physics step.
        """
        super()._apply_action()
        composer = self._robot.instantaneous_wrench_composer
        if self.randomize_actuation:
            # the base thrust T is already applied and the composers add up, so adding
            # (scale - 1) x T makes the body feel scale x T in total (body frame, like T)
            composer.add_forces_and_torques(
                forces=(self.thrust_scale - 1.0).view(-1, 1, 1) * self._thrust,
                torques=(self.torque_scale - 1.0).view(-1, 1, 1) * self._moment,
                body_ids=self._body_id,
            )
        if self.cfg.wind_force_range[1] > 0.0:
            # with is_global=True, Isaac Lab rotates the world-frame force into the body frame
            composer.add_forces_and_torques(
                forces=self.wind_force_world.unsqueeze(1), body_ids=self._body_id, is_global=True
            )

    def _get_observations(self) -> dict:
        """Build the true state, add the noise, shift it into the observation window and
        append the action history.

        After a reset the window still holds the last frames of the previous episode, which
        had a different position, velocity and goal. Left as is, the first K-1 observations
        would mix two episodes and show the policy a fake jump. Envs flagged in
        ``obs_window_fresh`` therefore get K copies of their current frame instead. The refill
        happens here, not in ``_reset_idx``, because ``step()`` calls ``_reset_idx`` before
        ``_get_observations``: the frame computed here is the new episode's first one, so it
        is computed once and the window matches it exactly, noise included.
        """
        goal_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        self.true_state = torch.cat([self._robot.data.root_lin_vel_b, goal_b], dim=-1)
        obs = self.true_state
        if self.cfg.add_noise:
            rho = self.cfg.noise_corr  # AR(1) with unit stationary variance; 0 = base env's white noise
            self.obs_noise_state = rho * self.obs_noise_state + (1.0 - rho * rho) ** 0.5 * torch.randn_like(obs)
            obs = self.true_state + self.cfg.noise_std * self.obs_noise_state + self.obs_noise_bias
        new = obs.unsqueeze(1)
        window = torch.cat([self.obs_window[:, 1:], new], dim=1)
        self.obs_window = torch.where(self.obs_window_fresh[:, None, None], new.expand_as(window), window)
        self.obs_window_fresh[:] = False
        # the M newest commands: M slots starting at slot L - M (M = 0 -> empty, adds nothing)
        m = self.cfg.action_input_hist
        act_hist = self.cmd_history.narrow(1, self.cmd_history.shape[1] - m, m)
        return {"policy": torch.cat([self.obs_window.flatten(1), act_hist.flatten(1)], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        """Base reward plus the action-rate term on the *commanded* actions. Using the
        delayed action the controller gets would credit each change d steps late."""
        rate = torch.sum(torch.square(self.cmd_history[:, -1] - self.cmd_history[:, -2]), dim=1)
        action_rate = rate * self.cfg.action_rate_reward_scale * self.step_dt
        self._episode_sums["action_rate"] += action_rate
        return super()._get_rewards() + action_rate

    def _distance_reward(self, distance: torch.Tensor) -> torch.Tensor:
        """Distance term used by the base _get_rewards, which weights and logs it."""
        if self.cfg.distance_reward_shape == "exp":
            return torch.exp(-distance / self.cfg.distance_reward_length)
        return super()._distance_reward(distance)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the base env, clear the command history, draw the episode's latency, noise
        offset, wind and actuation scales, and flag the observation window for a refill."""
        super()._reset_idx(env_ids)
        n, dev = len(env_ids), self.device
        self.cmd_history[env_ids] = 0.0
        self.delay_steps[env_ids] = torch.randint(self.delay_min_steps, self.delay_max_steps + 1, (n,), device=dev)
        # drawn from N(0, 1), not set to 0, so the noise is at full strength from the first step
        self.obs_noise_state[env_ids] = torch.randn(n, BASE_OBS_DIM, device=dev)
        self.obs_noise_bias[env_ids] = self.cfg.noise_bias_std * torch.randn(n, BASE_OBS_DIM, device=dev)
        # wind: uniform strength in the range, uniform 3-D direction (normalised Gaussian vector)
        direction = torch.randn(n, 3, device=dev)
        direction = direction / direction.norm(dim=1, keepdim=True)
        strength = torch.empty(n, 1, device=dev).uniform_(*self.cfg.wind_force_range)
        self.wind_force_world[env_ids] = strength * direction
        # actuation scales (uniform_(1, 1) gives exactly 1, so the defaults need no branch)
        self.thrust_scale[env_ids] = torch.empty(n, device=dev).uniform_(*self.cfg.thrust_scale_range)
        self.torque_scale[env_ids] = torch.empty(n, device=dev).uniform_(*self.cfg.torque_scale_range)
        self.obs_window_fresh[env_ids] = True
