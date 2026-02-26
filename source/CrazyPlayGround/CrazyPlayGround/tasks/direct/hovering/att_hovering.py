# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Attitude-controlled hovering environment.

Identical to pos/vel_hovering except the action is [roll_ref, pitch_ref, yaw_rate_ref, thrust_normalized]
fed directly into the attitude angle loop, bypassing both position and velocity PIDs.
The agent directly commands the drone's orientation and thrust.

Note on yaw control vs. the original implementation
------------------------------------------------------
The original att_hovering overrode the yaw rate setpoint directly after the attitude
angle loop, effectively bypassing the yaw angle PID.  With CrazyfliePIDController the
``"attitude"`` command level uses the yaw setpoint maintained by the controller:
- ``target_yaw_rate`` integrates the yaw setpoint each step.
- The attitude angle PID then tracks that integrated setpoint.
The net effect is identical when yaw_rate = 0 (the controller holds the yaw at the
value when reset) and very similar for nonzero yaw rates once the setpoint has settled.
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
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from drone import CrazyfliePIDController, load_config

# Path to the Crazyflie config bundled with DroneModule.
# Adjust if the project layout changes.
_DEFAULT_DRONE_CONFIG = str(
    _pathlib.Path(__file__).resolve().parents[7] / "DroneModule" / "configs" / "crazyflie.yaml"
)


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

    # ACTION SPACE: [roll_ref, pitch_ref, yaw_rate_ref, thrust_normalized]
    action_space = 4
    observation_space = 6
    state_space = 0
    debug_vis = True

    # Attitude limits
    max_roll_pitch = 30.0 * math.pi / 180.0   # 30 deg max tilt
    max_yaw_rate   = 90.0 * math.pi / 180.0   # 90 deg/s max yaw rate
    # Thrust is normalized [0, 1] and mapped to [min_thrust, max_thrust]
    min_thrust_scale = 0.5   # fraction of hover thrust
    max_thrust_scale = 1.8   # fraction of hover thrust

    ui_window_class_type = QuadcopterEnvWindow

    # Path to the DroneModule YAML config used to build CrazyfliePIDController.
    drone_config_path: str = _DEFAULT_DRONE_CONFIG

    # simulation — same as pos_hovering
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

    # scene — same as pos_hovering
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot — same reward scales as pos_hovering
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    lin_vel_reward_scale = -0.1
    ang_vel_reward_scale = -0.05
    distance_to_goal_reward_scale = 20.0

    add_noise = False
    noise_std = 0.01


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]

        # Build the cascade PID controller from DroneModule.
        # At command_level="attitude" the position and velocity loops are bypassed;
        # the agent's roll/pitch references and yaw rate feed directly into the
        # attitude angle → rate cascade.
        drone_cfg = load_config(self.cfg.drone_config_path)
        self._ctrl = CrazyfliePIDController.from_drone_config(
            drone_cfg,
            num_envs=self.num_envs,
            dt=self.cfg.sim.dt,
            device=self.device,
        )

        # Thrust limits derived from the config mass (hover thrust ≈ mass × g).
        self._hover_thrust = drone_cfg.physics.mass * 9.81
        self._min_thrust = self.cfg.min_thrust_scale * self._hover_thrust
        self._max_thrust = self.cfg.max_thrust_scale * self._hover_thrust

        self._att_ref = torch.zeros(self.num_envs, 3, device=self.device)
        self._yaw_rate_ref = torch.zeros(self.num_envs, 1, device=self.device)
        self._thrust_pwm = torch.zeros(self.num_envs, 1, device=self.device)

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
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Store attitude/thrust setpoints; PID runs at physics rate in _apply_action
        roll_ref  = self._actions[:, 0] * self.cfg.max_roll_pitch
        pitch_ref = self._actions[:, 1] * self.cfg.max_roll_pitch
        self._att_ref = torch.stack([roll_ref, pitch_ref, torch.zeros_like(roll_ref)], dim=-1)
        self._yaw_rate_ref = (self._actions[:, 2] * self.cfg.max_yaw_rate).unsqueeze(-1)

        thrust_normalized = (self._actions[:, 3] + 1.0) / 2.0
        thrust_ref_n = self._min_thrust + thrust_normalized * (self._max_thrust - self._min_thrust)
        self._thrust_pwm = (thrust_ref_n / self._ctrl.thrust_cmd_scale).unsqueeze(-1)

    def _apply_action(self):
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
        # Same observations as pos/vel_hovering: lin_vel_b + desired_pos_b
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                desired_pos_b,
            ],
            dim=-1,
        )

        if self.cfg.add_noise:
            noise = torch.randn_like(obs) * self.cfg.noise_std
            obs += noise

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging — same as pos/vel_hovering
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
        # Sample new commands — same as pos/vel_hovering
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-1, 1)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state — same as pos/vel_hovering
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

        # Reset all PID integrators for the affected environments.
        # This also clears the yaw setpoint (set to NaN → re-initialised to
        # actual yaw on the first call after reset).
        self._ctrl.reset(env_ids)

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
