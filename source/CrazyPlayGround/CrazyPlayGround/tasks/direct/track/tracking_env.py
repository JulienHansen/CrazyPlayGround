# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-tracking environments for the Crazyflie drone.

Three control modes are provided via sub-configs:
  - PosTrackingEnvCfg  action_space=3  (position delta)
  - VelTrackingEnvCfg  action_space=3  (velocity setpoint)
  - AttTrackingEnvCfg  action_space=4  (roll, pitch, yaw_rate, thrust)

Observation (22-D, identical across all modes):
  rpos_1..4  (3 each)  — relative position to future waypoints in body frame
  lin_vel_b  (3)       — body-frame linear velocity
  ang_vel_b  (3)       — body-frame angular velocity
  quat       (4)       — orientation [w, x, y, z]
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
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import SPHERE_MARKER_CFG  # isort: skip
from drone import CascadePIDController, load_config

from .trajectories import TRAJECTORIES, apply_traj_transform

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_DRONE_CONFIG = str(
    _pathlib.Path(__file__).resolve().parents[7] / "DroneModule" / "configs" / "crazyflie.yaml"
)


# ---------------------------------------------------------------------------
# UI window
# ---------------------------------------------------------------------------

class TrackingEnvWindow(BaseEnvWindow):
    def __init__(self, env: TrackingEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@configclass
class TrackingEnvCfg(DirectRLEnvCfg):
    """Shared base config for all trajectory-tracking control modes."""

    episode_length_s: float = 10.0
    decimation: int = 5

    # Observation: 4 * 3 (rpos lookahead) + 3 (lin_vel_b) + 3 (ang_vel_b) + 4 (quat) = 22
    observation_space: int = 22
    state_space: int = 0
    debug_vis: bool = True

    ui_window_class_type = TrackingEnvWindow

    # --- Simulation ---
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

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # --- Robot ---
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- Trajectory ---
    trajectory_type: str = "lemniscate"
    future_traj_steps: int = 4          # lookahead waypoints in observation
    traj_step_size: float = 5.0         # sim steps between consecutive waypoints
    traj_origin: list = [0.0, 0.0, 1.5]

    randomize_trajectory: bool = True

    # Per-episode randomisation ranges (uniform)
    traj_speed_range: tuple = (0.8, 1.1)    # angular speed multiplier
    traj_scale_range: tuple = (1.5, 3.0)    # x/y scale [m]
    traj_z_scale_range: tuple = (0.8, 1.5)  # z scale [m]
    traj_c_range: tuple = (-0.6, 0.6)       # lemniscate c param

    # --- Safety ---
    reset_thres: float = 0.8    # tracking error [m] that kills episode

    # --- Reward scales ---
    pos_reward_scale: float = 20.0
    lin_vel_reward_scale: float = -0.1
    ang_vel_reward_scale: float = -0.05
    effort_reward_scale: float = 0.0
    action_smooth_reward_scale: float = 0.0

    # --- DroneModule ---
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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TrackingEnv(DirectRLEnv):
    cfg: TrackingEnvCfg

    def __init__(self, cfg: TrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions     = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._thrust      = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment      = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # --- PID controller ---
        drone_cfg = load_config(self.cfg.drone_config_path)
        self._ctrl = CascadePIDController.from_drone_config(
            drone_cfg,
            num_envs=self.num_envs,
            dt=self.cfg.sim.dt,
            device=self.device,
        )

        self._hover_thrust = drone_cfg.physics.mass * 9.81

        # Attitude-mode specific buffers (only used when control_mode=="attitude")
        if self.cfg.control_mode == "attitude":
            self._att_ref      = torch.zeros(self.num_envs, 3, device=self.device)
            self._yaw_rate_ref = torch.zeros(self.num_envs, 1, device=self.device)
            self._thrust_pwm   = torch.zeros(self.num_envs, 1, device=self.device)
            min_s = self.cfg.min_thrust_scale
            max_s = self.cfg.max_thrust_scale
            self._min_thrust = min_s * self._hover_thrust
            self._max_thrust = max_s * self._hover_thrust

        # Position-mode specific buffers
        if self.cfg.control_mode == "position":
            self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Velocity-mode specific buffers
        if self.cfg.control_mode == "velocity":
            self._ref_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # --- Trajectory parameters (per-env) ---
        self._traj_w     = torch.ones(self.num_envs, device=self.device)     # angular speed
        self._traj_scale = torch.ones(self.num_envs, 3, device=self.device)  # [N, 3]
        self._traj_rot   = torch.zeros(self.num_envs, 4, device=self.device) # quaternion wxyz
        self._traj_rot[:, 0] = 1.0  # identity
        self._traj_c     = torch.zeros(self.num_envs, device=self.device)    # lemniscate c
        self._traj_t0    = torch.zeros(self.num_envs, device=self.device)    # phase offset

        # Trajectory origin in world frame (broadcast over envs)
        origin = self.cfg.traj_origin
        self._traj_origin = torch.tensor(origin, dtype=torch.float, device=self.device)

        # --- Episode logging ---
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal"]
        }

        self._body_id = self._robot.find_bodies("body")[0]

        # Path visualisation buffers: sample N_PATH_PTS evenly-spaced points per env
        # (updated at each episode reset, static during the episode).
        # Cap at _N_VIS_ENVS to avoid creating millions of markers.
        self._N_PATH_PTS = 300
        self._N_VIS_ENVS = min(self.num_envs, 32)
        self._path_pts_buf = torch.zeros(
            self._N_VIS_ENVS, self._N_PATH_PTS, 3, device=self.device
        )

        self.set_debug_vis(self.cfg.debug_vis)

    # -----------------------------------------------------------------------
    # Scene setup
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Trajectory helpers
    # -----------------------------------------------------------------------

    def _compute_traj(self, steps: int, env_ids: torch.Tensor, step_size: float) -> torch.Tensor:
        """Compute `steps` consecutive waypoints for the given environments.

        Returns:
            [len(env_ids), steps, 3] world-frame positions.
        """
        N = len(env_ids)
        dt = self.cfg.sim.dt * self.cfg.decimation  # policy dt

        # Time indices for each lookahead step
        offsets = step_size * torch.arange(1, steps + 1, device=self.device).float()  # [steps]
        t_base = self.episode_length_buf[env_ids].float().unsqueeze(1)                 # [N, 1]
        t_idx = t_base + offsets.unsqueeze(0)                                           # [N, steps]

        # Convert step count to radians
        t = self._traj_t0[env_ids].unsqueeze(1) + self._traj_w[env_ids].unsqueeze(1) * t_idx * dt

        # Evaluate trajectory function (supports c parameter via lemniscate wrapper)
        traj_fn = TRAJECTORIES[self.cfg.trajectory_type]
        if self.cfg.trajectory_type == "lemniscate":
            # Evaluate per-env c parameter; batch the call by iterating
            # (N can be large so we vectorise: pass c=0 then add z correction)
            c = self._traj_c[env_ids]  # [N]
            raw = traj_fn(t, c=0.0)    # [N, steps, 3]
            # Add z coupling: c * sin(2*t)  per env
            z_extra = c.unsqueeze(1) * torch.sin(2.0 * t)  # [N, steps]
            raw = raw + torch.stack([
                torch.zeros_like(z_extra),
                torch.zeros_like(z_extra),
                z_extra,
            ], dim=-1)
        else:
            raw = traj_fn(t)  # [N, steps, 3]

        # Per-env origin: local offset + env grid origin → each trajectory stays in its own env
        origin = self._traj_origin.unsqueeze(0) + self._terrain.env_origins[env_ids]  # [N, 3]
        return apply_traj_transform(raw, self._traj_scale[env_ids], self._traj_rot[env_ids], origin)

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

            # Random yaw rotation of the whole trajectory
            yaw = torch.empty(N, device=dev).uniform_(-math.pi, math.pi)
            zeros = torch.zeros(N, device=dev)
            self._traj_rot[env_ids] = quat_from_euler_xyz(zeros, zeros, yaw)

            lo, hi = self.cfg.traj_c_range
            self._traj_c[env_ids] = torch.empty(N, device=dev).uniform_(lo, hi)

            # Random phase offset so each episode starts at a different point
            self._traj_t0[env_ids] = torch.empty(N, device=dev).uniform_(0.0, 2.0 * math.pi)
        else:
            self._traj_w[env_ids] = 1.0
            self._traj_scale[env_ids, :] = torch.tensor([2.0, 2.0, 1.0], device=dev)
            self._traj_rot[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
            self._traj_c[env_ids] = 0.0
            self._traj_t0[env_ids] = 0.0

    # -----------------------------------------------------------------------
    # Physics step
    # -----------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

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

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------

    def _get_observations(self) -> dict:
        all_env_ids = torch.arange(self.num_envs, device=self.device)

        # Compute future waypoints: [N, future_traj_steps, 3]
        waypoints_w = self._compute_traj(
            steps=self.cfg.future_traj_steps,
            env_ids=all_env_ids,
            step_size=self.cfg.traj_step_size,
        )

        # Express each waypoint in body frame
        rpos_list = []
        for i in range(self.cfg.future_traj_steps):
            rpos_b, _ = subtract_frame_transforms(
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                waypoints_w[:, i, :],
            )
            rpos_list.append(rpos_b)

        obs = torch.cat(
            rpos_list
            + [
                self._robot.data.root_lin_vel_b,  # (3)
                self._robot.data.root_ang_vel_b,  # (3)
                self._robot.data.root_quat_w,     # (4)
            ],
            dim=-1,
        )
        return {"policy": obs}

    # -----------------------------------------------------------------------
    # Rewards
    # -----------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        all_env_ids = torch.arange(self.num_envs, device=self.device)

        # Current target = first lookahead waypoint
        current_wp = self._compute_traj(
            steps=1,
            env_ids=all_env_ids,
            step_size=self.cfg.traj_step_size,
        )[:, 0, :]  # [N, 3]

        distance = torch.linalg.norm(current_wp - self._robot.data.root_pos_w, dim=1)
        r_pos = 1.0 - torch.tanh(distance / 0.8)

        lin_vel_sq = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel_sq = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        rewards = {
            "distance_to_goal": r_pos * self.cfg.pos_reward_scale * self.step_dt,
            "lin_vel": lin_vel_sq * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel_sq * self.cfg.ang_vel_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # -----------------------------------------------------------------------
    # Dones
    # -----------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        current_wp = self._compute_traj(
            steps=1,
            env_ids=all_env_ids,
            step_size=self.cfg.traj_step_size,
        )[:, 0, :]
        tracking_error = torch.linalg.norm(current_wp - self._robot.data.root_pos_w, dim=1)

        too_low       = self._robot.data.root_pos_w[:, 2] < 0.1
        too_far       = tracking_error > self.cfg.reset_thres
        died = torch.logical_or(too_low, too_far)

        return died, time_out

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # --- Logging ---
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        current_wp = self._compute_traj(
            steps=1, env_ids=env_ids, step_size=self.cfg.traj_step_size
        )[:, 0, :]
        final_distance = torch.linalg.norm(
            current_wp - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras: dict = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = {}
        self.extras["log"].update(extras)
        extras = {}
        extras["Episode_Termination/died"]    = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance.item()
        self.extras["log"].update(extras)

        # --- Sample new trajectory params ---
        self._sample_traj_params(env_ids)

        # --- Reset robot ---
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # Spawn drone at (or near) the trajectory start point
        traj_start = self._compute_traj(
            steps=1, env_ids=env_ids, step_size=0.0
        )[:, 0, :]  # [N, 3]

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = traj_start

        # Add small random perturbation around spawn
        spawn = default_root_state.clone()
        spawn[:, :2] += torch.zeros_like(spawn[:, :2]).uniform_(-0.3, 0.3)
        spawn[:, 2]  += torch.zeros_like(spawn[:, 2]).uniform_(-0.2, 0.2)
        spawn[:, 2]   = spawn[:, 2].clamp(min=0.2)

        self._robot.write_root_pose_to_sim(spawn[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset PID integrators
        self._ctrl.reset(env_ids)

        # Update path buffer for any vis env that just reset.
        vis_env_ids = env_ids[env_ids < self._N_VIS_ENVS]
        if len(vis_env_ids) > 0 and self.cfg.debug_vis:
            # step_size chosen so N_PATH_PTS covers ~1.5 full periods regardless of speed.
            # With step_size=2, dt=0.01: advance = 2*N_PATH_PTS*0.01 = 6 rad ≈ 1 period.
            new_pts = self._compute_traj(
                steps=self._N_PATH_PTS, env_ids=vis_env_ids, step_size=2.0
            )  # [len(vis_env_ids), N_PATH_PTS, 3]
            self._path_pts_buf[vis_env_ids] = new_pts
            if hasattr(self, "path_visualizer"):
                flat = self._path_pts_buf.reshape(-1, 3)  # [N_VIS_ENVS * N_PATH_PTS, 3]
                self.path_visualizer.visualize(flat)

    # -----------------------------------------------------------------------
    # Debug visualisation  — same pattern as OmniDrones Track
    # -----------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # --- Trajectory path: many small yellow spheres per env ---
            if not hasattr(self, "path_visualizer"):
                path_cfg = SPHERE_MARKER_CFG.copy()
                path_cfg.markers["sphere"] = sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.9, 0.9, 0.1)  # yellow
                    ),
                )
                path_cfg.prim_path = "/Visuals/Track/path"
                self.path_visualizer = VisualizationMarkers(path_cfg)
            self.path_visualizer.set_visibility(True)

            # --- Current waypoint: red sphere (one per env) ---
            if not hasattr(self, "target_visualizer"):
                target_cfg = SPHERE_MARKER_CFG.copy()
                target_cfg.markers["sphere"] = sim_utils.SphereCfg(
                    radius=0.06,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.1, 0.1)  # red
                    ),
                )
                target_cfg.prim_path = "/Visuals/Track/target"
                self.target_visualizer = VisualizationMarkers(target_cfg)
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "path_visualizer"):
                self.path_visualizer.set_visibility(False)
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Move the red sphere to the current waypoint each step (all envs).
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        current_wp = self._compute_traj(
            steps=1, env_ids=all_env_ids, step_size=self.cfg.traj_step_size
        )[:, 0, :]
        self.target_visualizer.visualize(current_wp)
        # Path spheres are static per episode — updated only in _reset_idx.
