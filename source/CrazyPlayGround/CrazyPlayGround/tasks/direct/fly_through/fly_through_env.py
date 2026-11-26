# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly-through obstacle task for a single Crazyflie drone.

Two control modes are available via sub-configs:
  - VelFlyThroughEnvCfg  action_space=3  (Vx, Vy, Vz setpoint per step)
  - AttFlyThroughEnvCfg  action_space=4  (roll, pitch, yaw_rate, thrust)

The drone spawns behind a gate (x < 0 in env-local frame), must fly through
the gate opening (at x=0, Y-randomised), then reach a target on the far side.

Observation (20-D with time encoding):
  quat_w          (4)  — orientation [w, x, y, z] in world frame
  lin_vel_b       (3)  — body-frame linear velocity
  ang_vel_b       (3)  — body-frame angular velocity
  target_rpos_b   (3)  — target position relative to drone (body frame)
  gate_rpos_b     (3)  — gate centre relative to drone (body frame)
  time_encoding   (4)  — episode progress (repeated scalar)

Reward:
  r_pos    = exp(-scale * dist_to_target)
  r_gate   = centering_yz * exp(-dist_to_gate_plane)  [before gate]
           = 1.0                                        [after gate]
  r_up     = 0.5 * ((up_z + 1) / 2)^2
  r_spin   = 0.5 / (1 + yaw_rate^2)
  r_effort = effort_weight * exp(-mean_squared_actions)
  total    = r_pos + 0.5*r_gate + (r_pos + 0.3)*(r_up + r_spin) + r_effort

Episode ends when the drone:
  - falls below z=0.2 or rises above z=3.0  (env-local)
  - drifts more than 3.0 m laterally         (env-local Y)
  - is more than 8.0 m from the target
  - crosses the gate plane without passing through the opening
  - reaches max episode length
"""

from __future__ import annotations

import math
import pathlib as _pathlib
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from CrazyPlayGround.controllers import CascadePIDController, load_config, resolve_max_thrust_scale

from ..crazyflie_events import CrazyflieEventCfg

_DEFAULT_DRONE_CONFIG = str(
    _pathlib.Path(__file__).resolve().parents[3] / "controllers" / "crazyflie.yaml"
)

_GATE_X: float = 0.0   # gate plane x-coordinate in env frame
_GATE_Z: float = 2.0   # gate centre height in env frame

_OBS_DIM_BASE = 16   # quat(4) + lin_vel_b(3) + ang_vel_b(3) + target_rpos_b(3) + gate_rpos_b(3)
_TIME_ENC_DIM = 4

@configclass
class FlyThroughEnvCfg(DirectRLEnvCfg):
    """Base config shared by velocity and attitude fly-through tasks."""

    episode_length_s: float = 10.0
    decimation: int = 5
    debug_vis: bool = True

    time_encoding: bool = True

    gate_moving_range: float = 1.0  # max Y offset of gate from env origin [m]
    gate_half_width: float = 0.6    # gate opening half-width  in Y [m]
    gate_half_height: float = 0.5   # gate opening half-height in Z [m]

    reward_distance_scale: float = 1.0
    reward_effort_weight: float = 0.05

    control_mode: str = "velocity"
    max_velocity: float = 3.0                         # velocity mode [m/s]
    max_roll_pitch: float = 30.0 * math.pi / 180.0   # attitude mode [rad]
    max_yaw_rate: float = 90.0 * math.pi / 180.0     # attitude mode [rad/s]
    min_thrust_scale: float = 0.5                     # attitude mode
    max_thrust_scale: float | None = None  # fraction of hover thrust; None = drone.max_thrust / (m·g), the firmware cap (1.5387)

    state_space: int = 0

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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=8.0, replicate_physics=True, clone_in_fabric=True
    )

    drone_config_path: str = _DEFAULT_DRONE_CONFIG
    # startup event: PhysX body mass = drone.mass of this yaml (see crazyflie_events.py)
    events: CrazyflieEventCfg = CrazyflieEventCfg()

@configclass
class VelFlyThroughEnvCfg(FlyThroughEnvCfg):
    """Velocity-controlled fly-through.  Action = [Vx, Vy, Vz] normalised to [-1, 1]."""

    action_space: int = 3
    observation_space: int = _OBS_DIM_BASE + _TIME_ENC_DIM  # 20
    control_mode: str = "velocity"

@configclass
class AttFlyThroughEnvCfg(FlyThroughEnvCfg):
    """Attitude-controlled fly-through.  Action = [roll, pitch, yaw_rate, thrust] in [-1, 1]."""

    action_space: int = 4
    observation_space: int = _OBS_DIM_BASE + _TIME_ENC_DIM  # 20
    control_mode: str = "attitude"

class FlyThroughEnv(DirectRLEnv):
    cfg: FlyThroughEnvCfg

    def __init__(self, cfg: FlyThroughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._body_id = self._robot.find_bodies("body")[0]

        drone_cfg = load_config(self.cfg.drone_config_path)
        self._ctrl = CascadePIDController.from_drone_config(
            drone_cfg,
            num_envs=self.num_envs,
            dt=self.cfg.sim.dt,
            device=self.device,
        )

        hover_thrust = drone_cfg.physics.mass * 9.81
        self._min_thrust = self.cfg.min_thrust_scale * hover_thrust
        self._max_thrust_scale = resolve_max_thrust_scale(
            self.cfg.max_thrust_scale, drone_cfg.physics.mass, drone_cfg.physics.max_thrust
        )
        self._max_thrust = self._max_thrust_scale * hover_thrust

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._thrust_buf = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment_buf = torch.zeros(self.num_envs, 1, 3, device=self.device)

        if self.cfg.control_mode == "velocity":
            self._ref_vel = torch.zeros(self.num_envs, 3, device=self.device)
        else:  # attitude
            self._att_ref = torch.zeros(self.num_envs, 3, device=self.device)
            self._yaw_rate_ref = torch.zeros(self.num_envs, 1, device=self.device)
            self._thrust_pwm = torch.zeros(self.num_envs, 1, device=self.device)

        self._gate_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._crossed_plane = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["pos_error", "gate_yz_dist", "success"]
        }

        import omni.kit.app
        _ext_mgr = omni.kit.app.get_app().get_extension_manager()
        if not _ext_mgr.is_extension_enabled("isaacsim.util.debug_draw"):
            _ext_mgr.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self._camera_initialised = False

    def _setup_scene(self):
        self._robot = Articulation(CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot"))

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

        if self.cfg.control_mode == "velocity":
            self._ref_vel = self._actions[:, :3] * self.cfg.max_velocity

        else:  # attitude
            roll  = self._actions[:, 0] * self.cfg.max_roll_pitch
            pitch = self._actions[:, 1] * self.cfg.max_roll_pitch
            self._att_ref = torch.stack([roll, pitch, torch.zeros_like(roll)], dim=-1)
            self._yaw_rate_ref = (self._actions[:, 2] * self.cfg.max_yaw_rate).unsqueeze(-1)
            thrust_norm = self._actions[:, 3].clamp(0.0, 1.0)
            thrust_ref  = self._min_thrust + thrust_norm * (self._max_thrust - self._min_thrust)
            self._thrust_pwm = (thrust_ref / self._ctrl.thrust_cmd_scale).unsqueeze(-1)

    def _apply_action(self) -> None:
        root_state = torch.cat(
            [
                self._robot.data.root_pos_w,      # [E, 3]  world position
                self._robot.data.root_quat_w,      # [E, 4]  [w, x, y, z]
                self._robot.data.root_lin_vel_w,   # [E, 3]  world-frame linear velocity
                self._robot.data.root_ang_vel_b,   # [E, 3]  body-frame angular velocity
            ],
            dim=-1,
        )

        if self.cfg.control_mode == "velocity":
            thrust, moment = self._ctrl(
                root_state,
                target_vel=self._ref_vel,
                command_level="velocity",
                body_rates_in_body_frame=True,
            )
        else:  # attitude
            thrust, moment = self._ctrl(
                root_state,
                target_attitude=self._att_ref,
                target_yaw_rate=self._yaw_rate_ref,
                thrust_cmd=self._thrust_pwm,
                command_level="attitude",
                body_rates_in_body_frame=True,
            )

        self._thrust_buf[:, 0, 2] = thrust.squeeze(-1)
        self._moment_buf[:, 0, :] = moment
        self._robot.set_external_force_and_torque(
            self._thrust_buf, self._moment_buf, body_ids=self._body_id
        )

    def _get_observations(self) -> dict:
        pos_w     = self._robot.data.root_pos_w       # [E, 3]
        quat_w    = self._robot.data.root_quat_w      # [E, 4]  [w, x, y, z]
        lin_vel_b = self._robot.data.root_lin_vel_b   # [E, 3]
        ang_vel_b = self._robot.data.root_ang_vel_b   # [E, 3]

        target_rpos_b, _ = subtract_frame_transforms(pos_w, quat_w, self._target_pos_w)  # [E, 3]
        gate_rpos_b,   _ = subtract_frame_transforms(pos_w, quat_w, self._gate_pos_w)    # [E, 3]

        obs_parts = [quat_w, lin_vel_b, ang_vel_b, target_rpos_b, gate_rpos_b]

        if self.cfg.time_encoding:
            t = (self.episode_length_buf / self.max_episode_length).unsqueeze(-1)  # [E, 1]
            obs_parts.append(t.expand(-1, _TIME_ENC_DIM))                          # [E, 4]

        obs = torch.cat(obs_parts, dim=-1)  # [E, obs_dim]

        dist_to_target = (self._target_pos_w - pos_w).norm(dim=-1)
        gate_yz_dist   = (pos_w[:, 1:] - self._gate_pos_w[:, 1:]).norm(dim=-1)
        self._episode_sums["pos_error"]    += dist_to_target
        self._episode_sums["gate_yz_dist"] += gate_yz_dist

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pos_w = self._robot.data.root_pos_w  # [E, 3]

        dist_to_target = (self._target_pos_w - pos_w).norm(dim=-1)   # [E]
        r_pos = torch.exp(-self.cfg.reward_distance_scale * dist_to_target)

        dist_to_gate_plane = self._gate_pos_w[:, 0] - pos_w[:, 0]    # [E]
        dist_gate_y = (pos_w[:, 1] - self._gate_pos_w[:, 1]).abs()    # [E]
        dist_gate_z = (pos_w[:, 2] - self._gate_pos_w[:, 2]).abs()    # [E]

        gate_centering = (
            (self.cfg.gate_half_width  - dist_gate_y)
            + (self.cfg.gate_half_height - dist_gate_z)
        )
        r_gate = torch.where(
            dist_to_gate_plane > 0.0,
            gate_centering * torch.exp(-dist_to_gate_plane),
            torch.ones_like(dist_to_gate_plane),  # once past gate: full reward
        )

        quat  = self._robot.data.root_quat_w
        up_z  = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))  # [E]
        r_up  = 0.5 * ((up_z + 1.0) / 2.0).pow(2)

        yaw_rate_sq = self._robot.data.root_ang_vel_b[:, 2].pow(2)   # [E]
        r_spin = 0.5 / (1.0 + yaw_rate_sq)

        effort   = self._actions.pow(2).mean(dim=-1)                  # [E]
        r_effort = self.cfg.reward_effort_weight * torch.exp(-effort)

        success = (dist_to_target < 0.2) & (pos_w[:, 0] > self._gate_pos_w[:, 0])
        self._episode_sums["success"] += success.float()

        total = (
            r_pos
            + 0.5 * r_gate
            + (r_pos + 0.3) * (r_up + r_spin)
            + r_effort
        )

        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w      = self._robot.data.root_pos_w        # [E, 3]
        pos_local  = pos_w - self.scene.env_origins     # env-local frame [E, 3]

        crossed_now = pos_w[:, 0] > self._gate_pos_w[:, 0]  # [E]
        crossing    = crossed_now & (~self._crossed_plane)   # first-time crossing
        self._crossed_plane |= crossed_now

        within_y     = (pos_w[:, 1] - self._gate_pos_w[:, 1]).abs() < self.cfg.gate_half_width
        within_z     = (pos_w[:, 2] - self._gate_pos_w[:, 2]).abs() < self.cfg.gate_half_height
        through_gate = within_y & within_z

        invalid = crossing & ~through_gate

        dist_to_target = (self._target_pos_w - pos_w).norm(dim=-1)

        misbehave = (
            (pos_local[:, 2] < 0.2)          # too low
            | (pos_local[:, 2] > 3.0)        # too high
            | (pos_local[:, 1].abs() > 3.0)  # too far lateral
            | (dist_to_target > 8.0)         # too far from target
        )

        hasnan = torch.isnan(pos_w).any(dim=-1)

        terminated = misbehave | invalid | hasnan
        time_out   = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        extras = self.extras.setdefault("log", {})
        for key, buf in self._episode_sums.items():
            extras[f"Episode/{key}"] = (
                buf[env_ids].mean() / self.max_episode_length_s
            ).item()
            buf[env_ids] = 0.0

        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        M       = len(env_ids)
        origins = self.scene.env_origins[env_ids]  # [M, 3]

        default_root = self._robot.data.default_root_state[env_ids].clone()

        spawn_off = torch.stack(
            [
                torch.empty(M, device=self.device).uniform_(-2.5, -2.0),  # x behind gate
                torch.empty(M, device=self.device).uniform_(-1.5,  1.5),  # y random
                torch.empty(M, device=self.device).uniform_( 1.5,  2.5),  # z random
            ],
            dim=-1,
        )  # [M, 3]

        roll  = torch.empty(M, device=self.device).uniform_(-0.2, 0.2) * math.pi
        pitch = torch.empty(M, device=self.device).uniform_(-0.2, 0.2) * math.pi
        yaw   = torch.zeros(M, device=self.device)
        init_quat = quat_from_euler_xyz(roll, pitch, yaw)  # [M, 4]

        default_root[:, :3]  = origins + spawn_off
        default_root[:, 3:7] = init_quat
        default_root[:, 7:]  = 0.0   # zero initial velocity

        self._robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._ctrl.reset(env_ids)

        gate_y = torch.empty(M, device=self.device).uniform_(
            -self.cfg.gate_moving_range, self.cfg.gate_moving_range
        )
        self._gate_pos_w[env_ids, 0] = origins[:, 0] + _GATE_X
        self._gate_pos_w[env_ids, 1] = origins[:, 1] + gate_y
        self._gate_pos_w[env_ids, 2] = origins[:, 2] + _GATE_Z

        target_off = torch.stack(
            [
                torch.empty(M, device=self.device).uniform_(1.5, 2.5),   # x past gate
                torch.empty(M, device=self.device).uniform_(-1.0, 1.0),  # y random
                torch.empty(M, device=self.device).uniform_(1.5, 2.5),   # z random
            ],
            dim=-1,
        )  # [M, 3]
        self._target_pos_w[env_ids] = origins + target_off

        self._crossed_plane[env_ids] = False

        if self.cfg.debug_vis and (env_ids == 0).any().item():
            self._draw_gate_env0()
            if not self._camera_initialised:
                self._init_camera_env0()
                self._camera_initialised = True

    def _draw_gate_env0(self):
        """Draw the gate rectangle and target cross for env 0."""
        self._draw.clear_lines()

        gx, gy, gz = self._gate_pos_w[0].cpu().tolist()
        hw = self.cfg.gate_half_width
        hh = self.cfg.gate_half_height

        tl = (gx, gy - hw, gz + hh)
        tr = (gx, gy + hw, gz + hh)
        br = (gx, gy + hw, gz - hh)
        bl = (gx, gy - hw, gz - hh)

        green = (0.0, 1.0, 0.0, 1.0)
        self._draw.draw_lines(
            [tl, tr, br, bl],
            [tr, br, bl, tl],
            [green] * 4,
            [3.0] * 4,
        )

        tx, ty, tz = self._target_pos_w[0].cpu().tolist()
        cross = 0.12
        red = (1.0, 0.2, 0.2, 1.0)
        self._draw.draw_lines(
            [(tx - cross, ty, tz), (tx, ty - cross, tz), (tx, ty, tz - cross)],
            [(tx + cross, ty, tz), (tx, ty + cross, tz), (tx, ty, tz + cross)],
            [red] * 3,
            [3.0] * 3,
        )

    def _init_camera_env0(self):
        """Aim the viewport camera at env 0's gate."""
        import numpy as np
        from isaacsim.core.utils.viewports import set_camera_view

        origin = self.scene.env_origins[0].cpu().numpy()
        look_at = origin + np.array([_GATE_X, 0.0, _GATE_Z], dtype=np.float32)
        eye     = look_at  + np.array([-5.0, -5.0, 3.0],     dtype=np.float32)
        set_camera_view(eye=eye, target=look_at)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not debug_vis:
            self._draw.clear_lines()

    def _debug_vis_callback(self, event):
        pass
