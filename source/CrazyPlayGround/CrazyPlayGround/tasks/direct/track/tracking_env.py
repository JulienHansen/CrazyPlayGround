# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-tracking environments for the Crazyflie drone.

Three control modes are provided via sub-configs:
  - PosTrackingEnvCfg  action_space=3  (position delta)
  - VelTrackingEnvCfg  action_space=3  (velocity setpoint)
  - AttTrackingEnvCfg  action_space=4  (roll, pitch, yaw_rate, thrust)

Observation, identical across all modes, with F = future_traj_steps and A = action_space:
  [obs_input_hist x (rpos_1..F, lin_vel_b, ang_vel_b, quat), action_input_hist x action(A)]
  rpos_1..F  (3 each)  — relative position to future waypoints in body frame
  lin_vel_b  (3)       — body-frame linear velocity
  ang_vel_b  (3)       — body-frame angular velocity
  quat       (4)       — orientation [w, x, y, z]
The defaults (F=4, obs_input_hist=1, action_input_hist=0) give 22-D. Observation steps and actions are
stacked oldest first; after a reset the observation window holds copies of the first step.

Commands: the policy output is clamped to [-1, 1] (the commanded action) and pushed into a
history buffer. The controller gets the command sent ``delay_steps`` policy steps ago, drawn per
episode in [action_latency_steps, max(action_latency_steps, action_latency_max_steps)]. The
action history in the observation and the action-rate penalty use the commanded actions. At reset
the history is filled with the hover command: zeros for Pos/Vel, and for Att a thrust channel of
(hover - min) / (max - min) thrust, so delayed steps mean hover, not free fall.
"""

from __future__ import annotations

import math
import pathlib as _pathlib

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from CrazyPlayGround.controllers import CascadePIDController, load_config

from .trajectories import TRAJECTORIES, apply_traj_transform

_DEFAULT_DRONE_CONFIG = str(
    _pathlib.Path(__file__).resolve().parents[3] / "controllers" / "crazyflie.yaml"
)

class TrackingEnvWindow(BaseEnvWindow):
    def __init__(self, env: TrackingEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class TrackingEnvCfg(DirectRLEnvCfg):
    """Shared base config for all trajectory-tracking control modes."""

    episode_length_s: float = 10.0
    decimation: int = 5

    observation_space: int = 22  # recomputed in TrackingEnv.__init__
    state_space: int = 0
    debug_vis: bool = True

    ui_window_class_type = TrackingEnvWindow

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
    terrain: TerrainImporterCfg = TerrainImporterCfg(
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
        num_envs=4096, env_spacing=10.0, replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    trajectory_type: str = "lemniscate"
    future_traj_steps: int = 4          # lookahead waypoints in observation
    obs_input_hist: int = 1             # K observation steps in the policy input, oldest first; 1 = no stacking
    action_input_hist: int = 0          # last M commanded actions after the observation steps; 0 = none
    action_latency_steps: int = 0       # policy steps between command and controller; min of the per-episode draw
    action_latency_max_steps: int = 0   # 0 = fixed latency; otherwise max of the uniform per-episode draw
    traj_step_size: float = 5.0         # sim steps between consecutive waypoints
    traj_origin: list = [0.0, 0.0, 1.5]

    randomize_trajectory: bool = True

    traj_speed_range: tuple = (0.8, 1.1)    # angular speed multiplier
    traj_scale_range: tuple = (1.5, 3.0)    # x/y scale [m]
    traj_z_scale_range: tuple = (0.8, 1.5)  # z scale [m]
    traj_c_range: tuple = (-0.6, 0.6)       # lemniscate c param

    reset_thres: float = 0.8    # tracking error [m] that kills episode

    pos_reward_scale: float = 20.0
    lin_vel_reward_scale: float = -0.1
    ang_vel_reward_scale: float = -0.05
    effort_reward_scale: float = 0.0
    action_rate_reward_scale: float = 0.0  # weight on ||a_t - a_{t-1}||^2 (commanded, before latency); negative = penalty

    drone_config_path: str = _DEFAULT_DRONE_CONFIG

@configclass
class PosTrackingEnvCfg(TrackingEnvCfg):
    action_space: int = 3
    control_mode: str = "position"
    max_position_delta: float = 0.5     # m

@configclass
class VelTrackingEnvCfg(TrackingEnvCfg):
    action_space: int = 3
    control_mode: str = "velocity"
    max_velocity: float = 3.0           # m/s

@configclass
class AttTrackingEnvCfg(TrackingEnvCfg):
    action_space: int = 4
    control_mode: str = "attitude"
    max_roll_pitch: float = 30.0 * math.pi / 180.0   # rad
    max_yaw_rate: float = 90.0 * math.pi / 180.0     # rad/s
    min_thrust_scale: float = 0.5
    max_thrust_scale: float = 1.8

def check_cfg(cfg: TrackingEnvCfg):
    """Reject knob values that can't work. Called by the env constructor, after Hydra overrides.
    """
    checks = [
        (cfg.future_traj_steps >= 1, f"future_traj_steps must be >= 1, got {cfg.future_traj_steps}"),
        (cfg.obs_input_hist >= 1, f"obs_input_hist must be >= 1, got {cfg.obs_input_hist}"),
        (cfg.action_input_hist >= 0, f"action_input_hist must be >= 0, got {cfg.action_input_hist}"),
        (cfg.action_latency_steps >= 0, f"action_latency_steps must be >= 0, got {cfg.action_latency_steps}"),
        (
            not cfg.action_latency_max_steps or cfg.action_latency_max_steps >= cfg.action_latency_steps,
            f"action_latency_max_steps must be 0 or >= action_latency_steps ({cfg.action_latency_steps}), "
            f"got {cfg.action_latency_max_steps}",
        ),
    ]
    if getattr(cfg, "control_mode", None) == "attitude":
        # the hover command (history reset value) must be a reachable thrust in [0, 1]
        lo, hi = cfg.min_thrust_scale, cfg.max_thrust_scale
        checks.append(
            (0.0 < lo <= 1.0 <= hi and lo < hi,
             f"min/max_thrust_scale must satisfy 0 < min <= 1 <= max and min < max, got {lo} / {hi}")
        )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


class TrackingEnv(DirectRLEnv):
    cfg: TrackingEnvCfg

    def __init__(self, cfg: TrackingEnvCfg, render_mode: str | None = None, **kwargs):
        # Hydra applies CLI overrides after the cfg is built, so validate and size the
        # spaces here; a wrong size makes skrl silently reshape [N, D] into [N*x, D/x].
        check_cfg(cfg)
        obs_step_dim = 3 * cfg.future_traj_steps + 10  # one observation step (22-D with F=4)
        cfg.observation_space = obs_step_dim * cfg.obs_input_hist + cfg.action_space * cfg.action_input_hist
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self._obs_step_dim = obs_step_dim

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions     = torch.zeros(self.num_envs, action_dim, device=self.device)  # delayed, sent to the controller
        self._thrust      = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment      = torch.zeros(self.num_envs, 1, 3, device=self.device)

        drone_cfg = load_config(self.cfg.drone_config_path)
        self._ctrl = CascadePIDController.from_drone_config(
            drone_cfg,
            num_envs=self.num_envs,
            dt=self.cfg.sim.dt,
            device=self.device,
        )

        self._hover_thrust = drone_cfg.physics.mass * 9.81

        if self.cfg.control_mode == "attitude":
            self._att_ref      = torch.zeros(self.num_envs, 3, device=self.device)
            self._yaw_rate_ref = torch.zeros(self.num_envs, 1, device=self.device)
            self._thrust_pwm   = torch.zeros(self.num_envs, 1, device=self.device)
            min_s = self.cfg.min_thrust_scale
            max_s = self.cfg.max_thrust_scale
            self._min_thrust = min_s * self._hover_thrust
            self._max_thrust = max_s * self._hover_thrust

        # command that holds the drone in place, used to fill the command history at reset:
        # zero position delta / velocity, and for Att zero attitude and yaw rate with hover thrust
        self._hover_cmd = torch.zeros(action_dim, device=dev)
        if self.cfg.control_mode == "attitude":
            self._hover_cmd[3] = (self._hover_thrust - self._min_thrust) / (self._max_thrust - self._min_thrust)

        if self.cfg.control_mode == "position":
            self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        if self.cfg.control_mode == "velocity":
            self._ref_vel = torch.zeros(self.num_envs, 3, device=self.device)

        self._traj_w     = torch.ones(self.num_envs, device=self.device)     # angular speed
        self._traj_scale = torch.ones(self.num_envs, 3, device=self.device)  # [N, 3]
        self._traj_rot   = torch.zeros(self.num_envs, 4, device=self.device) # quaternion wxyz
        self._traj_rot[:, 0] = 1.0  # identity
        self._traj_c     = torch.zeros(self.num_envs, device=self.device)    # lemniscate c
        self._traj_t0    = torch.zeros(self.num_envs, device=self.device)    # phase offset

        origin = self.cfg.traj_origin
        self._traj_origin = torch.tensor(origin, dtype=torch.float, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal", "action_rate"]
        }

        self.delay_min_steps = cfg.action_latency_steps
        self.delay_max_steps = max(cfg.action_latency_steps, cfg.action_latency_max_steps)
        self.obs_window = torch.zeros(n, cfg.obs_input_hist, obs_step_dim, device=dev)
        # True for envs just reset: _get_observations refills their window (see its docstring)
        self.obs_window_fresh = torch.ones(n, dtype=torch.bool, device=dev)
        # last commanded actions, newest last; feeds the action history, the latency and the
        # action-rate penalty. Length >= 2 so the penalty always has a previous command.
        buf_len = max(cfg.action_input_hist, self.delay_max_steps + 1, 2)
        self.cmd_history = self._hover_cmd.expand(n, buf_len, action_dim).clone()
        self.delay_steps = torch.full((n,), self.delay_min_steps, dtype=torch.long, device=dev)
        self.env_idx = torch.arange(n, device=dev)

        self._body_id = self._robot.find_bodies("body")[0]

        import omni.kit.app
        _ext_mgr = omni.kit.app.get_app().get_extension_manager()
        if not _ext_mgr.is_extension_enabled("isaacsim.util.debug_draw"):
            _ext_mgr.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()

        self._camera_initialised = False

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

    def _compute_traj(self, steps: int, env_ids: torch.Tensor, step_size: float) -> torch.Tensor:
        """Compute `steps` consecutive waypoints for the given environments.

        Returns:
            [len(env_ids), steps, 3] world-frame positions.
        """
        N = len(env_ids)
        dt = self.cfg.sim.dt * self.cfg.decimation  # policy dt

        offsets = step_size * torch.arange(1, steps + 1, device=self.device).float()  # [steps]
        t_base = self.episode_length_buf[env_ids].float().unsqueeze(1)                 # [N, 1]
        t_idx = t_base + offsets.unsqueeze(0)                                           # [N, steps]

        t = self._traj_t0[env_ids].unsqueeze(1) + self._traj_w[env_ids].unsqueeze(1) * t_idx * dt

        traj_fn = TRAJECTORIES[self.cfg.trajectory_type]
        if self.cfg.trajectory_type == "lemniscate":
            c = self._traj_c[env_ids]  # [N]
            raw = traj_fn(t, c=0.0)    # [N, steps, 3]
            z_extra = c.unsqueeze(1) * torch.sin(2.0 * t)  # [N, steps]
            raw = raw + torch.stack([
                torch.zeros_like(z_extra),
                torch.zeros_like(z_extra),
                z_extra,
            ], dim=-1)
        else:
            raw = traj_fn(t)  # [N, steps, 3]

        origin = self._traj_origin.unsqueeze(0) + self._terrain.env_origins[env_ids]  # [N, 3]
        return apply_traj_transform(raw, self._traj_scale[env_ids], self._traj_rot[env_ids], origin)

    def _current_ref_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Reference the drone is scored against (reward, termination, metrics).

        Args:
            env_ids: environments to evaluate.

        Returns:
            (len(env_ids), 3) world-frame position of the first look-ahead waypoint, m.
        """
        return self._compute_traj(steps=1, env_ids=env_ids, step_size=self.cfg.traj_step_size)[:, 0, :]

    def _sample_traj_params(self, env_ids: torch.Tensor):
        """Sample per-episode trajectory parameters for the given envs."""
        N = len(env_ids)
        dev = self.device

        if self.cfg.randomize_trajectory:
            lo, hi = self.cfg.traj_speed_range
            self._traj_w[env_ids] = torch.empty(N, device=dev).uniform_(lo, hi)

            lo, hi = self.cfg.traj_scale_range
            xy_scale = torch.empty(N, device=dev).uniform_(lo, hi)
            lo_z, hi_z = self.cfg.traj_z_scale_range
            z_scale = torch.empty(N, device=dev).uniform_(lo_z, hi_z)
            self._traj_scale[env_ids] = torch.stack([xy_scale, xy_scale, z_scale], dim=-1)

            yaw = torch.empty(N, device=dev).uniform_(-math.pi, math.pi)
            zeros = torch.zeros(N, device=dev)
            self._traj_rot[env_ids] = quat_from_euler_xyz(zeros, zeros, yaw)

            lo, hi = self.cfg.traj_c_range
            self._traj_c[env_ids] = torch.empty(N, device=dev).uniform_(lo, hi)

            self._traj_t0[env_ids] = torch.empty(N, device=dev).uniform_(0.0, 2.0 * math.pi)
        else:
            self._traj_w[env_ids] = 1.0
            self._traj_scale[env_ids, :] = torch.tensor([2.0, 2.0, 1.0], device=dev)
            self._traj_rot[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
            self._traj_c[env_ids] = 0.0
            self._traj_t0[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        """Record the commanded (clamped) action, then hand the controller the one sent
        ``delay_steps`` steps ago (the hover command until the episode is that old)."""
        commanded = actions.clamp(-1.0, 1.0)
        self.cmd_history = torch.cat([self.cmd_history[:, 1:], commanded.unsqueeze(1)], dim=1)
        slot = self.cmd_history.shape[1] - 1 - self.delay_steps  # [N]
        self._actions = self.cmd_history[self.env_idx, slot]     # [N, A]: env i, slot[i]

        mode = self.cfg.control_mode

        if mode == "position":
            self._target_pos = (
                self._robot.data.root_pos_w
                + self._actions[:, :3] * self.cfg.max_position_delta
            )

        elif mode == "velocity":
            self._ref_vel = self._actions[:, :3] * self.cfg.max_velocity

        elif mode == "attitude":
            roll_ref  = self._actions[:, 0] * self.cfg.max_roll_pitch
            pitch_ref = self._actions[:, 1] * self.cfg.max_roll_pitch
            self._att_ref = torch.stack(
                [roll_ref, pitch_ref, torch.zeros_like(roll_ref)], dim=-1
            )
            self._yaw_rate_ref = (self._actions[:, 2] * self.cfg.max_yaw_rate).unsqueeze(-1)
            thrust_normalized  = self._actions[:, 3].clamp(0.0, 1.0)
            thrust_ref_n = self._min_thrust + thrust_normalized * (self._max_thrust - self._min_thrust)
            self._thrust_pwm = (thrust_ref_n / self._ctrl.thrust_cmd_scale).unsqueeze(-1)

    def _apply_action(self):
        root_state = torch.cat(
            [
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_b,
            ],
            dim=-1,
        )

        mode = self.cfg.control_mode

        if mode == "position":
            thrust, moment = self._ctrl(
                root_state,
                target_pos=self._target_pos,
                command_level="position",
                body_rates_in_body_frame=True,
            )
        elif mode == "velocity":
            thrust, moment = self._ctrl(
                root_state,
                target_vel=self._ref_vel,
                command_level="velocity",
                body_rates_in_body_frame=True,
            )
        elif mode == "attitude":
            thrust, moment = self._ctrl(
                root_state,
                target_attitude=self._att_ref,
                target_yaw_rate=self._yaw_rate_ref,
                thrust_cmd=self._thrust_pwm,
                command_level="attitude",
                body_rates_in_body_frame=True,
            )

        self._thrust[:, 0, 2] = thrust.squeeze(-1)
        self._moment[:, 0, :] = moment
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        """Push the current observation step into the window and append the action history.

        Envs just reset (``obs_window_fresh``) get K copies of their first step, so the window
        never mixes two episodes. This happens here, not in ``_reset_idx``, because ``step()``
        resets before it observes.

        Returns:
            {"policy": (num_envs, K * obs_step_dim + M * A)}: observation steps then commanded actions, oldest first.
        """
        all_env_ids = torch.arange(self.num_envs, device=self.device)

        waypoints_w = self._compute_traj(
            steps=self.cfg.future_traj_steps,
            env_ids=all_env_ids,
            step_size=self.cfg.traj_step_size,
        )

        rpos_list = []
        for i in range(self.cfg.future_traj_steps):
            rpos_b, _ = subtract_frame_transforms(
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                waypoints_w[:, i, :],
            )
            rpos_list.append(rpos_b)

        obs_step = torch.cat(
            rpos_list
            + [
                self._robot.data.root_lin_vel_b,  # (3)
                self._robot.data.root_ang_vel_b,  # (3)
                self._robot.data.root_quat_w,     # (4)
            ],
            dim=-1,
        )
        new = obs_step.unsqueeze(1)
        window = torch.cat([self.obs_window[:, 1:], new], dim=1)
        self.obs_window = torch.where(self.obs_window_fresh[:, None, None], new.expand_as(window), window)
        self.obs_window_fresh[:] = False
        # the M newest commands: M slots starting at slot L - M (M = 0 -> empty, adds nothing)
        m = self.cfg.action_input_hist
        act_hist = self.cmd_history.narrow(1, self.cmd_history.shape[1] - m, m)
        return {"policy": torch.cat([self.obs_window.flatten(1), act_hist.flatten(1)], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        current_wp = self._current_ref_w(self._robot._ALL_INDICES)  # [N, 3]

        distance = torch.linalg.norm(current_wp - self._robot.data.root_pos_w, dim=1)
        r_pos = 1.0 - torch.tanh(distance / 0.8)

        lin_vel_sq = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel_sq = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        # on the *commanded* actions: the delayed ones the controller gets would credit each change d steps late
        action_rate = torch.sum(torch.square(self.cmd_history[:, -1] - self.cmd_history[:, -2]), dim=1)

        rewards = {
            "distance_to_goal": r_pos * self.cfg.pos_reward_scale * self.step_dt,
            "lin_vel": lin_vel_sq * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel_sq * self.cfg.ang_vel_reward_scale * self.step_dt,
            "action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        current_wp = self._current_ref_w(self._robot._ALL_INDICES)
        tracking_error = torch.linalg.norm(current_wp - self._robot.data.root_pos_w, dim=1)

        too_low       = self._robot.data.root_pos_w[:, 2] < 0.1
        too_far       = tracking_error > self.cfg.reset_thres
        died = torch.logical_or(too_low, too_far)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        extras: dict = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = {}
        self.extras["log"].update(extras)
        extras = {}
        # tensors, not .item(): skrl logs only one-element tensors and silently drops floats
        extras["Episode_Termination/died"]     = torch.count_nonzero(self.reset_terminated[env_ids])
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids])
        self.extras["log"].update(extras)

        self._sample_traj_params(env_ids)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = self._hover_cmd

        traj_start = self._compute_traj(
            steps=1, env_ids=env_ids, step_size=0.0
        )[:, 0, :]  # [N, 3]

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = traj_start

        spawn = default_root_state.clone()
        spawn[:, :2] += torch.zeros_like(spawn[:, :2]).uniform_(-0.3, 0.3)
        spawn[:, 2]  += torch.zeros_like(spawn[:, 2]).uniform_(-0.2, 0.2)
        spawn[:, 2]   = spawn[:, 2].clamp(min=0.2)

        self._robot.write_root_pose_to_sim(spawn[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._ctrl.reset(env_ids)

        # clear the command history (hover command, not zeros, in Att mode), draw this
        # episode's latency, and flag the observation window for a refill
        self.cmd_history[env_ids] = self._hover_cmd
        self.delay_steps[env_ids] = torch.randint(
            self.delay_min_steps, self.delay_max_steps + 1, (len(env_ids),), device=self.device
        )
        self.obs_window_fresh[env_ids] = True

        if self.cfg.debug_vis and (0 in env_ids.tolist()):
            self._draw_env0_traj()
            if not self._camera_initialised:
                self._init_camera_env0()
                self._camera_initialised = True

    def _draw_env0_traj(self):
        """Draw the full trajectory for env 0 as white lines."""
        env0 = torch.tensor([0], device=self.device)
        n_pts = 800
        traj_vis = self._compute_traj(steps=n_pts, env_ids=env0, step_size=1.0)[0]  # [n_pts, 3]
        pts = [tuple(p) for p in traj_vis.cpu().tolist()]
        p0 = pts[:-1]
        p1 = pts[1:]
        n = len(p0)
        self._draw.clear_lines()
        self._draw.draw_lines(p0, p1, [(1.0, 1.0, 1.0, 1.0)] * n, [2.0] * n)

    def _init_camera_env0(self):
        """Point the default perspective camera at env 0's trajectory centre."""
        import numpy as np
        from isaacsim.core.utils.viewports import set_camera_view

        origin = self._terrain.env_origins[0].cpu().numpy()
        traj_c = origin + np.array(self.cfg.traj_origin, dtype=np.float32)
        eye    = traj_c + np.array([6.0, -6.0, 5.0], dtype=np.float32)
        set_camera_view(eye=eye, target=traj_c)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not debug_vis:
            self._draw.clear_lines()

    def _debug_vis_callback(self, event):
        pass
