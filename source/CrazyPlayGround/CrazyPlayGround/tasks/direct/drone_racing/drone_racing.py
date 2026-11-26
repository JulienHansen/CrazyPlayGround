# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drone racing environment — complete laps of a fixed track as fast as possible.

Two control modes via sub-configs:
  - VelDroneRacingEnvCfg  action_space=3  (Vx, Vy, Vz setpoint, normalised to [-1,1])
  - AttDroneRacingEnvCfg  action_space=4  (roll, pitch, yaw_rate, thrust in [-1,1])

Track:
  10 gates in a kidney-shaped FPV circuit.  The track is spawned ONCE in
  the world (/World/Track/) and shared by every parallel env.  With
  env_spacing ≈ 0.001 m all drones effectively race the same physical circuit.
  Gate heights vary from 2.0 m (fast low section) to 4.5 m (high apex).
  On each reset each drone spawns 2.5 m behind a randomly chosen gate,
  facing it — giving diverse start states and avoiding gate mesh collisions.

Observation (20-D):
  quat_w        (4)  orientation [w, x, y, z] in world frame
  lin_vel_b     (3)  body-frame linear velocity
  ang_vel_b     (3)  body-frame angular velocity
  curr_gate_b   (3)  current target gate position in body frame
  next_gate_b   (3)  next gate position in body frame
  time_enc      (4)  episode progress (same scalar repeated 4 times)

Reward:
  r_progress  = (prev_dist - curr_dist) * progress_scale   [approach reward]
  r_speed     = speed_bonus_scale * vel_toward_gate         [velocity toward gate]
  r_gate_pass = gate_pass_reward * (1 + exp(-steps/200))    [time-decaying gate bonus]
  r_up        = 0.15 * ((up_z + 1) / 2)^2                  [uprightness]
  r_spin      = 0.05 / (1 + omega_z^2)                     [anti yaw-spin]
  r_effort    = -effort_weight * mean(actions^2)
  total       = r_progress + r_gate_pass + r_speed + (r_progress + 0.2) * (r_up + r_spin) + r_effort
  [crash]     = crash_penalty  (overrides all other terms)

Gate crossing criterion:
  A gate is counted as crossed when the drone transitions from the approach
  side to the exit side of the gate plane AND the in-plane (lateral) distance
  to the gate centre is within gate_radius.  The lateral distance is the
  component of (pos − gate_centre) projected onto the gate's own plane
  (normal removed), so a drone passing below or beside the gate frame does
  NOT count as a valid crossing.

Episode termination:
  - drone height < 0.3 m or > 10 m (world frame)
  - horizontal distance from world origin > 12 m
  - NaN detected in drone state
  - timeout
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
from .track_generator import spawn_track

_DEFAULT_DRONE_CONFIG = str(
    _pathlib.Path(__file__).resolve().parents[3] / "controllers" / "crazyflie.yaml"
)

_GATE_WORLD_POS: list[tuple[float, float, float]] = [
    ( 0.0, -5.0, 2.5),   # 0  start / finish  (bottom, low)
    ( 4.0, -3.0, 3.0),   # 1  bottom-right bend
    ( 6.0,  0.0, 3.5),   # 2  right side
    ( 5.0,  3.5, 4.0),   # 3  upper-right
    ( 2.0,  5.5, 4.5),   # 4  right apex  (highest gate)
    (-2.0,  5.5, 4.5),   # 5  left apex   (highest gate)
    (-5.0,  3.5, 4.0),   # 6  upper-left
    (-6.0,  0.0, 3.5),   # 7  left side
    (-4.0, -3.0, 3.0),   # 8  bottom-left bend
    (-4.0, -7.0, 2.0),   # 9  long straight, lowest gate
]

_NUM_GATES: int = len(_GATE_WORLD_POS)
_GATE_RADIUS: float = 1.0   # [m] crossing-detection radius

_GATE_CENTER_Z_OFFSET: float = 1.067

_GATE_CENTER_POS: list[tuple[float, float, float]] = [
    (x, y, z + _GATE_CENTER_Z_OFFSET) for x, y, z in _GATE_WORLD_POS
]

def _build_normals(positions: list[tuple]) -> list[tuple[float, float, float]]:
    normals = []
    n = len(positions)
    for i in range(n):
        p0 = positions[i]
        p1 = positions[(i + 1) % n]
        dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        normals.append((dx / length, dy / length, dz / length))
    return normals

_GATE_NORMALS: list[tuple[float, float, float]] = _build_normals(_GATE_CENTER_POS)

_GATE_YAWS: list[float] = [math.atan2(n[1], n[0]) for n in _GATE_NORMALS]

_TRACK_CONFIG: dict = {
    str(g): {"pos": _GATE_WORLD_POS[g], "yaw": _GATE_YAWS[g]}
    for g in range(_NUM_GATES)
}

_OBS_BASE: int = 16   # quat(4) + lin_vel_b(3) + ang_vel_b(3) + curr_gate_b(3) + next_gate_b(3)
_TIME_DIM: int = 4

@configclass
class DroneRacingEnvCfg(DirectRLEnvCfg):
    """Base config shared by velocity and attitude drone-racing tasks."""

    episode_length_s: float = 20.0
    decimation: int = 5
    debug_vis: bool = True

    gate_radius: float = _GATE_RADIUS

    progress_scale: float = 5.0
    gate_pass_reward: float = 10.0
    speed_bonus_scale: float = 0.5      # reward for velocity toward the gate
    effort_weight: float = 0.001
    crash_penalty: float = -5.0

    control_mode: str = "velocity"
    max_velocity: float = 4.0                          # velocity mode [m/s]
    max_roll_pitch: float = 30.0 * math.pi / 180.0    # attitude mode [rad]
    max_yaw_rate: float = 90.0 * math.pi / 180.0      # attitude mode [rad/s]
    min_thrust_scale: float = 0.5                      # attitude mode
    max_thrust_scale: float | None = None  # fraction of hover thrust; None = drone.max_thrust / (m·g), the firmware cap (1.5387)

    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 500.0,
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
        num_envs=4096, env_spacing=0.001, replicate_physics=True
    )

    drone_config_path: str = _DEFAULT_DRONE_CONFIG
    # startup event: PhysX body mass = drone.mass of this yaml (see crazyflie_events.py)
    events: CrazyflieEventCfg = CrazyflieEventCfg()

@configclass
class VelDroneRacingEnvCfg(DroneRacingEnvCfg):
    """Velocity-controlled racing.  Action = [Vx, Vy, Vz] in [-1, 1]."""

    action_space: int = 3
    observation_space: int = _OBS_BASE + _TIME_DIM  # 20
    control_mode: str = "velocity"

@configclass
class AttDroneRacingEnvCfg(DroneRacingEnvCfg):
    """Attitude-controlled racing.  Action = [roll, pitch, yaw_rate, thrust] in [-1, 1]."""

    action_space: int = 4
    observation_space: int = _OBS_BASE + _TIME_DIM  # 20
    control_mode: str = "attitude"

class DroneRacingEnv(DirectRLEnv):
    cfg: DroneRacingEnvCfg

    def __init__(self, cfg: DroneRacingEnvCfg, render_mode: str | None = None, **kwargs):
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
        else:
            self._att_ref = torch.zeros(self.num_envs, 3, device=self.device)
            self._yaw_rate_ref = torch.zeros(self.num_envs, 1, device=self.device)
            self._thrust_pwm = torch.zeros(self.num_envs, 1, device=self.device)

        self._gate_world_pos = torch.tensor(
            _GATE_WORLD_POS, dtype=torch.float32, device=self.device
        )  # [G, 3]
        self._gate_center_pos = torch.tensor(
            _GATE_CENTER_POS, dtype=torch.float32, device=self.device
        )  # [G, 3]
        self._gate_world_normal = torch.tensor(
            _GATE_NORMALS, dtype=torch.float32, device=self.device
        )  # [G, 3]

        self._gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_signed = torch.zeros(self.num_envs, device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._steps_since_gate = torch.zeros(self.num_envs, device=self.device)

        self._episode_sums = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in ["progress", "gates_passed"]
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

        spawn_track(_TRACK_CONFIG)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

        if self.cfg.control_mode == "velocity":
            self._ref_vel = self._actions[:, :3] * self.cfg.max_velocity
        else:
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
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_b,
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
        else:
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

        curr_gate_w = self._gate_center_pos[self._gate_idx]                      # [E, 3]
        next_gate_w = self._gate_center_pos[(self._gate_idx + 1) % _NUM_GATES]  # [E, 3]

        curr_gate_b, _ = subtract_frame_transforms(pos_w, quat_w, curr_gate_w)  # [E, 3]
        next_gate_b, _ = subtract_frame_transforms(pos_w, quat_w, next_gate_w)  # [E, 3]

        t = (self.episode_length_buf / self.max_episode_length).unsqueeze(-1)  # [E, 1]
        time_enc = t.expand(-1, _TIME_DIM)                                      # [E, 4]

        obs = torch.cat(
            [quat_w, lin_vel_b, ang_vel_b, curr_gate_b, next_gate_b, time_enc], dim=-1
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pos_w = self._robot.data.root_pos_w  # [E, 3]

        curr_gate_w = self._gate_center_pos[self._gate_idx]          # [E, 3]
        gate_normal = self._gate_world_normal[self._gate_idx]        # [E, 3]

        diff   = pos_w - curr_gate_w
        signed = torch.sum(diff * gate_normal, dim=-1)              # [E] along normal
        lateral = diff - signed.unsqueeze(-1) * gate_normal          # [E, 3]
        planar  = torch.norm(lateral, dim=-1)                        # [E] in-plane dist

        crossed = (
            (self._prev_signed <= 0.0)
            & (signed > 0.0)
            & (planar < self.cfg.gate_radius)
        )

        if crossed.any():
            self._gate_idx[crossed] = (self._gate_idx[crossed] + 1) % _NUM_GATES
            self._episode_sums["gates_passed"][crossed] += 1.0
            self._steps_since_gate[crossed] = 0.0

        new_gate_w  = self._gate_center_pos[self._gate_idx]
        new_normal  = self._gate_world_normal[self._gate_idx]
        new_diff    = pos_w - new_gate_w
        new_signed  = torch.sum(new_diff * new_normal, dim=-1)
        self._prev_signed = new_signed.detach().clone()

        curr_dist  = torch.norm(new_diff, dim=-1)
        r_progress = (self._prev_dist - curr_dist) * self.cfg.progress_scale
        self._prev_dist = curr_dist.detach().clone()
        self._episode_sums["progress"] += r_progress.clamp(min=0.0)

        lin_vel_w = self._robot.data.root_lin_vel_w                    # [E, 3]
        to_gate   = new_diff / (curr_dist.unsqueeze(-1) + 1e-6)       # unit vec drone→gate
        speed_toward = torch.sum(lin_vel_w * (-to_gate), dim=-1).clamp(min=0.0)
        r_speed = self.cfg.speed_bonus_scale * speed_toward

        self._steps_since_gate += 1.0
        time_bonus = torch.exp(-self._steps_since_gate / 200.0)
        r_gate_pass = crossed.float() * self.cfg.gate_pass_reward * (1.0 + time_bonus)

        quat = self._robot.data.root_quat_w
        up_z = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
        r_up = 0.15 * ((up_z + 1.0) / 2.0).pow(2)

        omega_z = self._robot.data.root_ang_vel_b[:, 2]
        r_spin  = 0.05 / (1.0 + omega_z.pow(2))

        r_effort = -self.cfg.effort_weight * self._actions.pow(2).mean(dim=-1)

        total = r_progress + r_gate_pass + r_speed + (r_progress + 0.2) * (r_up + r_spin) + r_effort

        crash = (
            (pos_w[:, 2] < 0.3)
            | (pos_w[:, 2] > 10.0)
            | (pos_w[:, :2].norm(dim=-1) > 12.0)
            | torch.isnan(pos_w).any(dim=-1)
        )
        total = torch.where(crash, torch.full_like(total, self.cfg.crash_penalty), total)

        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w = self._robot.data.root_pos_w

        terminated = (
            (pos_w[:, 2] < 0.3)
            | (pos_w[:, 2] > 10.0)
            | (pos_w[:, :2].norm(dim=-1) > 12.0)
            | torch.isnan(pos_w).any(dim=-1)
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1

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

        M = len(env_ids)

        gate_start = torch.randint(0, _NUM_GATES, (M,), device=self.device)  # [M]
        gate_pos   = self._gate_center_pos[gate_start]     # [M, 3]
        gate_norm  = self._gate_world_normal[gate_start]  # [M, 3]

        noise       = torch.zeros(M, 3, device=self.device).uniform_(-0.2, 0.2)
        origins     = self.scene.env_origins[env_ids]     # [M, 3] — tiny with env_spacing=0.001
        spawn_pos_w = gate_pos - 3.0 * gate_norm + origins + noise

        gate_yaws = torch.tensor(_GATE_YAWS, device=self.device)[gate_start]  # [M]
        roll  = torch.empty(M, device=self.device).uniform_(-0.05, 0.05) * math.pi
        pitch = torch.empty(M, device=self.device).uniform_(-0.05, 0.05) * math.pi
        yaw   = gate_yaws + torch.empty(M, device=self.device).uniform_(-0.1, 0.1) * math.pi
        init_quat = quat_from_euler_xyz(roll, pitch, yaw)

        default_root = self._robot.data.default_root_state[env_ids].clone()
        default_root[:, :3]  = spawn_pos_w
        default_root[:, 3:7] = init_quat
        default_root[:, 7:]  = 0.0

        self._robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._ctrl.reset(env_ids)

        self._gate_idx[env_ids] = gate_start

        diff0 = spawn_pos_w - gate_pos
        self._prev_signed[env_ids] = (diff0 * gate_norm).sum(dim=-1)
        self._prev_dist[env_ids]   = diff0.norm(dim=-1)
        self._steps_since_gate[env_ids] = 0.0

        if self.cfg.debug_vis and (env_ids == 0).any().item():
            self._draw_track()
            if not self._camera_initialised:
                self._init_camera()
                self._camera_initialised = True

    def _draw_track(self):
        """Draw the track circuit as green lines between consecutive gate centres."""
        self._draw.clear_lines()
        points = [self._gate_center_pos[g].cpu().tolist() for g in range(_NUM_GATES)]
        green  = (0.0, 1.0, 0.0, 1.0)
        starts = points
        ends   = points[1:] + [points[0]]
        self._draw.draw_lines(starts, ends, [green] * _NUM_GATES, [3.0] * _NUM_GATES)

    def _init_camera(self):
        """Aim the viewport camera at the track from a high angle."""
        import numpy as np
        from isaacsim.core.utils.viewports import set_camera_view

        center = self._gate_world_pos.mean(0).cpu().numpy()
        look_at = center.copy()
        eye     = look_at + np.array([0.0, -14.0, 10.0], dtype=np.float32)
        set_camera_view(eye=eye, target=look_at)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not debug_vis:
            self._draw.clear_lines()

    def _debug_vis_callback(self, event):
        pass
