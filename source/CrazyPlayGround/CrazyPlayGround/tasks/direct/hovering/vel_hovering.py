# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-controlled hovering environment.

Identical to pos_hovering except the action is a velocity reference [Vx, Vy, Vz]
fed directly into the velocity control loop, bypassing the position PID entirely.
This avoids the derivative kick and error accumulation issues of the position loop.

Sim-to-real knobs (WP1), all off by default, so the default task is the plain one:

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

Distance reward shape: ``"tanh"`` (default, almost flat near the goal) or ``"exp"``, a single
exponential that is steeper near the goal.
"""

from __future__ import annotations

import logging
import pathlib as _pathlib

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from CrazyPlayGround.controllers import CascadePIDController, load_config

logger = logging.getLogger(__name__)

_DEFAULT_DRONE_CONFIG = str(
    _pathlib.Path(__file__).resolve().parents[3] / "controllers" / "crazyflie.yaml"
)
BASE_OBS_DIM = 6  # [lin_vel_b(3), desired_pos_b(3)]
ACTION_DIM = 3    # [vx, vy, vz]

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 5

    action_space = 3
    observation_space = 6
    state_space = 0
    debug_vis = True

    max_velocity = 1.0

    ui_window_class_type = QuadcopterEnvWindow

    drone_config_path: str = _DEFAULT_DRONE_CONFIG

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 500,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    lin_vel_reward_scale = -0.1
    ang_vel_reward_scale = -0.05
    distance_to_goal_reward_scale = 20.0

    add_noise = False
    noise_std = 0.01

    # ── sim-to-real knobs (WP1); every default reproduces the plain task ──
    obs_input_hist: int = 1     # observation frames in the policy input; 1 = no stacking
    action_input_hist: int = 0  # past commands in the policy input; 0 = none
    # delay in policy steps between sending a command and the controller using it
    action_latency_steps: int = 0
    action_latency_max_steps: int = 0      # > action_latency_steps: drawn per episode; 0 = fixed
    action_rate_reward_scale: float = 0.0  # weight on ||c_t - c_{t-1}||^2; negative = penalty
    # observation noise shape; only used with add_noise=True (std is noise_std above)
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
    # "tanh" = 1 - tanh(d / 0.8), almost flat near the goal (94 % of max at 5 cm);
    # "exp"  = exp(-d / distance_reward_length), steeper near the goal (slope 1 / length at d = 0)
    distance_reward_shape: str = "tanh"
    distance_reward_length: float = 0.3  # metres, "exp" only: the reward is 1/e (37 %) at this distance


def check_cfg(cfg: QuadcopterEnvCfg):
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

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        # Hydra applies CLI overrides after the cfg is built, so validate and size the
        # spaces here; a wrong size makes skrl silently reshape [N, D] into [N*x, D/x].
        check_cfg(cfg)
        cfg.observation_space = BASE_OBS_DIM * cfg.obs_input_hist + ACTION_DIM * cfg.action_input_hist
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "action_rate",
            ]
        }
        self._body_id = self._robot.find_bodies("body")[0]

        drone_cfg = load_config(self.cfg.drone_config_path)
        self._ctrl = CascadePIDController.from_drone_config(
            drone_cfg,
            num_envs=self.num_envs,
            dt=self.cfg.sim.dt,
            device=self.device,
        )

        self._ref_vel = torch.zeros(self.num_envs, 3, device=self.device)

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
        self.true_state = torch.zeros(n, BASE_OBS_DIM, device=dev)
        # AR(1) noise state with unit variance, carried between steps; redrawn at every reset
        self.obs_noise_state = torch.randn(n, BASE_OBS_DIM, device=dev)
        self.obs_noise_bias = torch.zeros(n, BASE_OBS_DIM, device=dev)
        self.wind_force_world = torch.zeros(n, 3, device=dev)
        self.thrust_scale = torch.ones(n, device=dev)
        self.torque_scale = torch.ones(n, device=dev)
        # skip the extra composer call entirely when nothing is randomised (tuple: Hydra may give a list)
        self.randomize_actuation = (tuple(cfg.thrust_scale_range), tuple(cfg.torque_scale_range)) != ((1.0, 1.0),) * 2

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Record the commanded action, then hand the controller the one sent ``delay_steps``
        steps ago (zero until the episode is that old)."""
        commanded = actions.clamp(-1.0, 1.0)
        self.cmd_history = torch.cat([self.cmd_history[:, 1:], commanded.unsqueeze(1)], dim=1)
        slot = self.cmd_history.shape[1] - 1 - self.delay_steps  # [N]
        self._actions = self.cmd_history[self.env_idx, slot]     # [N, 3]: env i, slot[i]

        self._ref_vel = self._actions[:, :3] * self.cfg.max_velocity

    def _apply_action(self):
        """The controller's thrust and moment, scaled by the episode's actuation effectiveness,
        plus the wind, for this physics step.

        The controller output goes into Isaac Lab's permanent wrench composer
        (set_external_force_and_torque); the extras go into the instantaneous composer, which
        Isaac Lab adds to it and clears after each physics step.
        """
        root_state = torch.cat(
            [
                self._robot.data.root_pos_w,      # [N, 3]
                self._robot.data.root_quat_w,      # [N, 4]  (w, x, y, z)
                self._robot.data.root_lin_vel_w,   # [N, 3]  world frame
                self._robot.data.root_ang_vel_b,   # [N, 3]  body frame
            ],
            dim=-1,
        )

        thrust, moment = self._ctrl(
            root_state,
            target_vel=self._ref_vel,
            command_level="velocity",
            body_rates_in_body_frame=True,
        )

        self._thrust[:, 0, 2] = thrust.squeeze(-1)
        self._moment[:, 0, :] = moment
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        composer = self._robot.instantaneous_wrench_composer
        if self.randomize_actuation:
            # the thrust T is already applied and the composers add up, so adding
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
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )

        self.true_state = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        obs = self.true_state

        if self.cfg.add_noise:
            rho = self.cfg.noise_corr  # AR(1) with unit stationary variance; 0 = plain white noise
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
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = self._distance_reward(distance_to_goal)
        # on the *commanded* actions: the delayed ones the controller gets would credit each change d steps late
        action_rate = torch.sum(torch.square(self.cmd_history[:, -1] - self.cmd_history[:, -2]), dim=1)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _distance_reward(self, distance: torch.Tensor) -> torch.Tensor:
        """Distance-to-goal reward in [0, 1], 1 at the goal; shape set by distance_reward_shape."""
        if self.cfg.distance_reward_shape == "exp":
            return torch.exp(-distance / self.cfg.distance_reward_length)
        return 1 - torch.tanh(distance / 0.8)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-1, 1)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        spawn = default_root_state.clone()
        spawn[:, :2] += torch.rand_like(spawn[:, :2]).uniform_(-1, 1)
        spawn[:, 2] += torch.rand_like(spawn[:, 2]).uniform_(0.2, 1.5)
        self._robot.write_root_pose_to_sim(spawn[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._ctrl.reset(env_ids)

        # sim-to-real knobs: clear the command history, draw this episode's latency, noise
        # offset, wind and actuation scales, and flag the observation window for a refill
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

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
