# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperation environment for Crazyflie drone.

Unified environment supporting three control modes:
- Position: [X_ref, Y_ref, Z_ref] in world frame
- Velocity: [Vx, Vy, Vz] in world frame
- Attitude: [roll, pitch, yaw_rate, thrust_normalized]

Control mode can be switched at runtime via set_control_mode().
"""

from __future__ import annotations

import math
import pathlib as _pathlib
from typing import Callable, Literal, Optional

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
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from drone import CascadePIDController, load_config

# Path to the Crazyflie config bundled with DroneModule.
_DEFAULT_DRONE_CONFIG = str(
    _pathlib.Path(__file__).resolve().parents[7] / "DroneModule" / "configs" / "crazyflie.yaml"
)

ControlMode = Literal["position", "velocity", "attitude"]
RateProfile = Literal["none", "betaflight", "actual", "kiss", "raceflight"]


class TeleoperationEnvWindow(BaseEnvWindow):
    """Window manager for the Teleoperation environment."""

    def __init__(self, env: TeleoperationEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class TeleoperationEnvCfg(DirectRLEnvCfg):
    # Long episode for teleoperation sessions (5 minutes)
    episode_length_s = 300.0
    decimation = 5

    # Unified action space: 7D [vx, vy, vz, roll, pitch, yaw_rate, thrust]
    # Different modes use different subsets
    action_space = 7
    # Observation: 13D [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    observation_space = 13
    state_space = 0
    debug_vis = True

    # Control limits
    max_velocity = 3.0  # m/s
    max_position_delta = 0.5  # m per step
    max_roll_pitch = 30.0 * math.pi / 180.0  # 30 deg max tilt (angle mode)
    max_yaw_rate = 90.0 * math.pi / 180.0  # 90 deg/s max yaw rate (angle mode)
    max_thrust_scale = 1.8  # fraction of hover thrust

    # Safety limits
    min_altitude = 0.1
    max_altitude = 3.0

    # Rate profile for attitude mode: "none" = angle mode (default)
    # Options: "none", "betaflight", "actual", "kiss", "raceflight"
    rate_profile: str = "none"

    ui_window_class_type = TeleoperationEnvWindow
    drone_config_path: str = _DEFAULT_DRONE_CONFIG

    # Simulation config
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

    # Single environment for teleoperation
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")


class TeleoperationEnv(DirectRLEnv):
    """Unified teleoperation environment supporting position, velocity, and attitude control."""

    cfg: TeleoperationEnvCfg

    def __init__(self, cfg: TeleoperationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Current control mode
        self._control_mode: ControlMode = "velocity"

        # Get body index
        self._body_id = self._robot.find_bodies("body")[0]

        # Build the cascade PID controller
        drone_cfg = load_config(self.cfg.drone_config_path)
        self._ctrl = CascadePIDController.from_drone_config(
            drone_cfg,
            num_envs=self.num_envs,
            dt=self.cfg.sim.dt,
            device=self.device,
        )

        # Thrust limits
        self._hover_thrust = drone_cfg.physics.mass * 9.81
        self._max_thrust = self.cfg.max_thrust_scale * self._hover_thrust

        # Rate profile function (None = angle mode)
        self._rate_profile_fn: Optional[Callable] = self._build_rate_profile_fn(cfg.rate_profile)
        if self._rate_profile_fn is not None:
            print(f"[TeleoperationEnv] Attitude mode: rate profile = {cfg.rate_profile}")
        else:
            print("[TeleoperationEnv] Attitude mode: angle mode (no rate profile)")

        # Control references
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._ref_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._att_ref = torch.zeros(self.num_envs, 3, device=self.device)
        self._body_rate_ref = torch.zeros(self.num_envs, 3, device=self.device)
        self._yaw_rate_ref = torch.zeros(self.num_envs, 1, device=self.device)
        self._thrust_cmd = torch.zeros(self.num_envs, 1, device=self.device)

        self.set_debug_vis(self.cfg.debug_vis)

    # ── Rate profile factory ──────────────────────────────────────────────────

    @staticmethod
    def _build_rate_profile_fn(profile: str) -> Optional[Callable]:
        """Return the rate profile callable, or None for angle mode."""
        if profile == "none":
            return None
        from drone.utils.rate_profiles import (
            actual_rate_profile,
            betaflight_rate_profile,
            kiss_rate_profile,
            raceflight_rate_profile,
        )
        _profiles = {
            "betaflight": betaflight_rate_profile,
            "actual":     actual_rate_profile,
            "kiss":       kiss_rate_profile,
            "raceflight": raceflight_rate_profile,
        }
        if profile not in _profiles:
            raise ValueError(f"Unknown rate profile '{profile}'. Choose from {list(_profiles)} or 'none'.")
        return _profiles[profile]

    # ── Mode management ───────────────────────────────────────────────────────

    @property
    def control_mode(self) -> ControlMode:
        """Get current control mode."""
        return self._control_mode

    def set_control_mode(self, mode: ControlMode) -> None:
        """Switch control mode at runtime."""
        if mode not in ("position", "velocity", "attitude"):
            raise ValueError(f"Invalid control mode: {mode}")

        if mode != self._control_mode:
            self._control_mode = mode
            # Reset PID integrators when switching modes
            self._ctrl.reset(torch.arange(self.num_envs, device=self.device))
            # Reset control references
            self._target_pos = self._robot.data.root_pos_w.clone()
            self._ref_vel.zero_()
            self._att_ref.zero_()
            self._body_rate_ref.zero_()
            self._yaw_rate_ref.zero_()
            # Keep thrust_cmd at current value so the first step after a switch
            # doesn't jerk — it will be overwritten by _pre_physics_step anyway.

    # ── Scene ─────────────────────────────────────────────────────────────────

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

    # ── Control ───────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions based on current control mode.

        Action format (7D): [vx, vy, vz, roll, pitch, yaw_rate, thrust]
        - Position mode : uses [vx, vy, vz] as position delta
        - Velocity mode : uses [vx, vy, vz] as velocity setpoint
        - Attitude mode :
            angle mode  — [roll, pitch] → angle setpoints; [yaw_rate] → yaw rate; [thrust] → thrust
            rate  mode  — [roll, pitch, yaw_rate] → rate profile → body rates; [thrust] → thrust
        """
        self._actions = actions.clone().clamp(-1.0, 1.0)

        if self._control_mode == "position":
            pos_delta = self._actions[:, :3] * self.cfg.max_position_delta
            self._target_pos = self._robot.data.root_pos_w + pos_delta
            self._target_pos[:, 2] = torch.clamp(
                self._target_pos[:, 2], self.cfg.min_altitude, self.cfg.max_altitude
            )

        elif self._control_mode == "velocity":
            self._ref_vel = self._actions[:, :3] * self.cfg.max_velocity

        elif self._control_mode == "attitude":
            if self._rate_profile_fn is None:
                # ── Angle mode (default) ──────────────────────────────────────
                roll_ref  = self._actions[:, 3] * self.cfg.max_roll_pitch
                pitch_ref = self._actions[:, 4] * self.cfg.max_roll_pitch
                self._att_ref = torch.stack([roll_ref, pitch_ref, torch.zeros_like(roll_ref)], dim=-1)
                self._yaw_rate_ref = (self._actions[:, 5] * self.cfg.max_yaw_rate).unsqueeze(-1)
            else:
                # ── Rate mode: stick → rate profile → body-rate setpoints ─────
                # actions[:, 3:6] = [roll, pitch, yaw] sticks in [-1, 1]
                rc_input = self._actions[:, 3:6]
                self._body_rate_ref = self._rate_profile_fn(rc_input)  # [N, 3] rad/s

            # Thrust: action[6] in [0, 1] — 0 = no thrust (falls), 1 = max thrust.
            # This is true for both angle and rate modes.
            thrust_normalized = self._actions[:, 6].clamp(0.0, 1.0)
            thrust_ref = thrust_normalized * self._max_thrust
            self._thrust_cmd = (thrust_ref / self._ctrl.thrust_cmd_scale).unsqueeze(-1)

    def _apply_action(self):
        """Apply control commands through the cascade PID controller."""
        root_state = torch.cat(
            [
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_b,
            ],
            dim=-1,
        )

        if self._control_mode == "position":
            thrust, moment = self._ctrl(
                root_state,
                target_pos=self._target_pos,
                command_level="position",
                body_rates_in_body_frame=True,
            )
        elif self._control_mode == "velocity":
            thrust, moment = self._ctrl(
                root_state,
                target_vel=self._ref_vel,
                command_level="velocity",
                body_rates_in_body_frame=True,
            )
        elif self._control_mode == "attitude":
            if self._rate_profile_fn is None:
                # Angle mode: angle setpoints → rate PID via attitude loop
                thrust, moment = self._ctrl(
                    root_state,
                    target_attitude=self._att_ref,
                    target_yaw_rate=self._yaw_rate_ref,
                    thrust_cmd=self._thrust_cmd,
                    command_level="attitude",
                    body_rates_in_body_frame=True,
                )
            else:
                # Rate mode: body-rate setpoints → rate PID directly
                thrust, moment = self._ctrl(
                    root_state,
                    target_body_rates=self._body_rate_ref,
                    thrust_cmd=self._thrust_cmd,
                    command_level="body_rate",
                    body_rates_in_body_frame=True,
                )

        self._thrust[:, 0, 2] = thrust.squeeze(-1)
        self._moment[:, 0, :] = moment
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # ── Observations / rewards / dones ────────────────────────────────────────

    def _get_observations(self) -> dict:
        """Return full state observation: pos(3), quat(4), lin_vel(3), ang_vel(3)."""
        obs = torch.cat(
            [
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_b,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """No rewards in teleoperation mode."""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check safety limits (altitude only, no timeout for teleop)."""
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < self.cfg.min_altitude,
            self._robot.data.root_pos_w[:, 2] > self.cfg.max_altitude,
        )
        time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] = 1.0
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._target_pos[env_ids] = default_root_state[:, :3]
        self._ctrl.reset(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
