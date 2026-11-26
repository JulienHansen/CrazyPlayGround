# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-agent competitive drone racing — 2 drones race head-to-head.

Based on "Agile Flight Emerges from Multi-Agent Competitive Racing"
(Pasumarti et al., 2025).  Competition with sparse rewards produces
emergent agile behaviours (overtaking, blocking) without explicit
reward shaping.

Two control modes via sub-configs:
  - VelDroneRacingMARLEnvCfg  action_space=3  (Vx, Vy, Vz)
  - AttDroneRacingMARLEnvCfg  action_space=4  (roll, pitch, yaw_rate, thrust)

Agents: ["blue", "red"]

Track:
  10 gates in a kidney-shaped FPV circuit (shared with single-agent racing).
  The track is spawned ONCE under /World/Track/ and shared by every parallel env
  (env_spacing ≈ 0.001 m).

Observation per agent (26-D):
  quat_w        (4)  orientation [w, x, y, z] in world frame
  lin_vel_b     (3)  body-frame linear velocity
  ang_vel_b     (3)  body-frame angular velocity
  curr_gate_b   (3)  current target gate position in body frame
  next_gate_b   (3)  next gate position in body frame
  opp_pos_b     (3)  opponent position in body frame
  opp_vel_b     (3)  opponent velocity in body frame
  time_enc      (4)  episode progress (scalar repeated 4×)

Global state for MAPPO critic (28-D):
  [pos(3) + quat(4) + lin_vel_b(3) + ang_vel_b(3)] × 2 drones + gate_idx(2)

Sparse competitive reward (from paper Eq. 2-6):
  r_pass    = 10.0 if gate passed AND leading; 5.0 if tied
  r_lap     = 50.0 if lap completed AND leading
  r_cmd     = -0.15*(ω²_roll + ω²_pitch) - 0.05*ω²_yaw
  r_crash   = -2.0 if terminally crashed
  r_contact = -0.1 if inter-drone distance < 0.15 m (proxy)
"""

from __future__ import annotations

import math
import pathlib as _pathlib
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_apply_inverse,
    subtract_frame_transforms,
)
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from CrazyPlayGround.controllers import CascadePIDController, load_config, resolve_max_thrust_scale

from ..crazyflie_events import CrazyflieEventCfg
from ..drone_racing.track_generator import spawn_track

_DEFAULT_DRONE_CONFIG = str(
    _pathlib.Path(__file__).resolve().parents[3] / "controllers" / "crazyflie.yaml"
)

_GATE_WORLD_POS: list[tuple[float, float, float]] = [
    ( 0.0, -5.0, 2.5),   # 0  start / finish
    ( 4.0, -3.0, 3.0),   # 1  bottom-right bend
    ( 6.0,  0.0, 3.5),   # 2  right side
    ( 5.0,  3.5, 4.0),   # 3  upper-right
    ( 2.0,  5.5, 4.5),   # 4  right apex
    (-2.0,  5.5, 4.5),   # 5  left apex
    (-5.0,  3.5, 4.0),   # 6  upper-left
    (-6.0,  0.0, 3.5),   # 7  left side
    (-4.0, -3.0, 3.0),   # 8  bottom-left bend
    (-4.0, -7.0, 2.0),   # 9  long straight
]

_NUM_GATES: int = len(_GATE_WORLD_POS)
_GATE_RADIUS: float = 1.0
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

_N = 2
_AGENT_NAMES = ["blue", "red"]

_OBS_DIM = 26

_STATE_DIM = 28

_CONTACT_DIST = 0.15

@configclass
class DroneRacingMARLEnvCfg(DirectMARLEnvCfg):
    """Base config for competitive 2-drone racing."""

    episode_length_s: float = 20.0
    decimation: int = 5
    debug_vis: bool = True

    possible_agents: list = _AGENT_NAMES
    observation_spaces: dict = {a: _OBS_DIM for a in _AGENT_NAMES}
    state_space: int = _STATE_DIM

    gate_radius: float = _GATE_RADIUS

    progress_scale: float = 5.0          # approach reward per metre closed
    speed_bonus_scale: float = 0.5       # velocity toward gate
    uprightness_scale: float = 0.15      # staying level
    effort_weight: float = 0.001         # action magnitude penalty

    gate_pass_reward_lead: float = 10.0
    gate_pass_reward_tie: float = 5.0
    lap_reward: float = 50.0
    crash_penalty: float = -5.0
    contact_penalty: float = -0.1
    cmd_roll_pitch_weight: float = -0.15
    cmd_yaw_weight: float = -0.05

    control_mode: str = "velocity"
    max_velocity: float = 4.0
    max_roll_pitch: float = 30.0 * math.pi / 180.0
    max_yaw_rate: float = 90.0 * math.pi / 180.0
    min_thrust_scale: float = 0.5
    max_thrust_scale: float | None = None  # fraction of hover thrust; None = drone.max_thrust / (m·g), the firmware cap (1.5387)

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
class VelDroneRacingMARLEnvCfg(DroneRacingMARLEnvCfg):
    """Velocity-controlled MARL racing.  Action = [Vx, Vy, Vz] per drone."""

    action_spaces: dict = {a: 3 for a in _AGENT_NAMES}
    control_mode: str = "velocity"

@configclass
class AttDroneRacingMARLEnvCfg(DroneRacingMARLEnvCfg):
    """Attitude-controlled MARL racing.  Action = [roll, pitch, yaw_rate, thrust] per drone."""

    action_spaces: dict = {a: 4 for a in _AGENT_NAMES}
    control_mode: str = "attitude"

class DroneRacingMARLEnv(DirectMARLEnv):
    cfg: DroneRacingMARLEnvCfg

    def __init__(self, cfg: DroneRacingMARLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._body_ids = [r.find_bodies("body")[0] for r in self._robots]

        self._thrust = [torch.zeros(self.num_envs, 1, 3, device=self.device) for _ in range(_N)]
        self._moment = [torch.zeros(self.num_envs, 1, 3, device=self.device) for _ in range(_N)]

        drone_cfg = load_config(self.cfg.drone_config_path)
        self._ctrls = [
            CascadePIDController.from_drone_config(
                drone_cfg,
                num_envs=self.num_envs,
                dt=self.cfg.sim.dt,
                device=self.device,
            )
            for _ in range(_N)
        ]

        hover_thrust = drone_cfg.physics.mass * 9.81
        self._min_thrust = self.cfg.min_thrust_scale * hover_thrust
        self._max_thrust_scale = resolve_max_thrust_scale(
            self.cfg.max_thrust_scale, drone_cfg.physics.mass, drone_cfg.physics.max_thrust
        )
        self._max_thrust = self._max_thrust_scale * hover_thrust

        if self.cfg.control_mode == "velocity":
            self._ref_vels = [
                torch.zeros(self.num_envs, 3, device=self.device) for _ in range(_N)
            ]
        else:
            self._att_refs = [
                torch.zeros(self.num_envs, 3, device=self.device) for _ in range(_N)
            ]
            self._yaw_rate_refs = [
                torch.zeros(self.num_envs, 1, device=self.device) for _ in range(_N)
            ]
            self._thrust_pwms = [
                torch.zeros(self.num_envs, 1, device=self.device) for _ in range(_N)
            ]

        self._gate_center_pos = torch.tensor(
            _GATE_CENTER_POS, dtype=torch.float32, device=self.device
        )  # [G, 3]
        self._gate_world_normal = torch.tensor(
            _GATE_NORMALS, dtype=torch.float32, device=self.device
        )  # [G, 3]

        self._gate_idx = torch.zeros(self.num_envs, _N, dtype=torch.long, device=self.device)
        self._lap_count = torch.zeros(self.num_envs, _N, dtype=torch.long, device=self.device)
        self._prev_signed = torch.zeros(self.num_envs, _N, device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, _N, device=self.device)
        self._steps_since_gate = torch.zeros(self.num_envs, _N, device=self.device)

        self._actions = {a: torch.zeros(self.num_envs, self.cfg.action_spaces[a], device=self.device)
                         for a in self.cfg.possible_agents}

        self._episode_sums = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in ["gates_blue", "gates_red", "laps_blue", "laps_red", "progress"]
        }

        import omni.kit.app
        _ext_mgr = omni.kit.app.get_app().get_extension_manager()
        if not _ext_mgr.is_extension_enabled("isaacsim.util.debug_draw"):
            _ext_mgr.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self._camera_initialised = False

        if self.cfg.debug_vis:
            self.set_debug_vis(True)

    def _setup_scene(self):
        self._robots: list[Articulation] = []
        for i in range(_N):
            robot = Articulation(
                CRAZYFLIE_CFG.replace(prim_path=f"/World/envs/env_.*/Robot_{i}")
            )
            self._robots.append(robot)

        spawn_track(_TRACK_CONFIG)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        for i, robot in enumerate(self._robots):
            self.scene.articulations[f"robot_{i}"] = robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for i, agent in enumerate(self.cfg.possible_agents):
            act = actions[agent].clone().clamp(-1.0, 1.0)
            self._actions[agent] = act

            if self.cfg.control_mode == "velocity":
                self._ref_vels[i] = act[:, :3] * self.cfg.max_velocity

            elif self.cfg.control_mode == "attitude":
                roll_ref = act[:, 0] * self.cfg.max_roll_pitch
                pitch_ref = act[:, 1] * self.cfg.max_roll_pitch
                self._att_refs[i] = torch.stack(
                    [roll_ref, pitch_ref, torch.zeros_like(roll_ref)], dim=-1
                )
                self._yaw_rate_refs[i] = (act[:, 2] * self.cfg.max_yaw_rate).unsqueeze(-1)
                thrust_norm = act[:, 3].clamp(0.0, 1.0)
                thrust_ref = self._min_thrust + thrust_norm * (self._max_thrust - self._min_thrust)
                self._thrust_pwms[i] = (thrust_ref / self._ctrls[i].thrust_cmd_scale).unsqueeze(-1)

    def _apply_action(self) -> None:
        for i, robot in enumerate(self._robots):
            root_state = torch.cat(
                [
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    robot.data.root_lin_vel_w,
                    robot.data.root_ang_vel_b,
                ],
                dim=-1,
            )

            if self.cfg.control_mode == "velocity":
                thrust, moment = self._ctrls[i](
                    root_state,
                    target_vel=self._ref_vels[i],
                    command_level="velocity",
                    body_rates_in_body_frame=True,
                )
            elif self.cfg.control_mode == "attitude":
                thrust, moment = self._ctrls[i](
                    root_state,
                    target_attitude=self._att_refs[i],
                    target_yaw_rate=self._yaw_rate_refs[i],
                    thrust_cmd=self._thrust_pwms[i],
                    command_level="attitude",
                    body_rates_in_body_frame=True,
                )

            self._thrust[i][:, 0, 2] = thrust.squeeze(-1)
            self._moment[i][:, 0, :] = moment
            robot.set_external_force_and_torque(
                self._thrust[i], self._moment[i], body_ids=self._body_ids[i]
            )

    def _detect_gate_crossings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect gate crossings for each drone independently.

        Returns:
            crossed: [E, 2] bool — whether each drone crossed its current gate
            lap_completed: [E, 2] bool — whether each drone completed a lap
        """
        crossed = torch.zeros(self.num_envs, _N, dtype=torch.bool, device=self.device)
        lap_completed = torch.zeros(self.num_envs, _N, dtype=torch.bool, device=self.device)

        for i, robot in enumerate(self._robots):
            pos_w = robot.data.root_pos_w  # [E, 3]
            gate_idx_i = self._gate_idx[:, i]  # [E]

            curr_gate_w = self._gate_center_pos[gate_idx_i]  # [E, 3]
            gate_normal = self._gate_world_normal[gate_idx_i]  # [E, 3]

            diff = pos_w - curr_gate_w
            signed = torch.sum(diff * gate_normal, dim=-1)  # [E]

            lateral = diff - signed.unsqueeze(-1) * gate_normal
            planar = torch.norm(lateral, dim=-1)

            cross_i = (
                (self._prev_signed[:, i] <= 0.0)
                & (signed > 0.0)
                & (planar < self.cfg.gate_radius)
            )
            crossed[:, i] = cross_i

            if cross_i.any():
                old_idx = gate_idx_i[cross_i]
                new_idx = (old_idx + 1) % _NUM_GATES
                lap_completed[cross_i, i] = (new_idx < old_idx)
                self._gate_idx[cross_i, i] = new_idx
                self._lap_count[cross_i, i] += lap_completed[cross_i, i].long()

            new_gate_w = self._gate_center_pos[self._gate_idx[:, i]]
            new_normal = self._gate_world_normal[self._gate_idx[:, i]]
            new_diff = pos_w - new_gate_w
            self._prev_signed[:, i] = torch.sum(new_diff * new_normal, dim=-1)

        return crossed, lap_completed

    def _get_progress_score(self) -> torch.Tensor:
        """Compute linear progress score per drone: lap_count * NUM_GATES + gate_idx.

        Returns: [E, 2] long tensor
        """
        return self._lap_count * _NUM_GATES + self._gate_idx

    def _get_observations(self) -> dict[str, torch.Tensor]:
        positions = [r.data.root_pos_w for r in self._robots]       # list of [E, 3]
        quats = [r.data.root_quat_w for r in self._robots]          # list of [E, 4]
        lin_vels_b = [r.data.root_lin_vel_b for r in self._robots]  # list of [E, 3]
        ang_vels_b = [r.data.root_ang_vel_b for r in self._robots]  # list of [E, 3]
        lin_vels_w = [r.data.root_lin_vel_w for r in self._robots]  # list of [E, 3]

        t = (self.episode_length_buf / self.max_episode_length).unsqueeze(-1)  # [E, 1]
        time_enc = t.expand(-1, 4)  # [E, 4]

        obs: dict[str, torch.Tensor] = {}

        for i, agent in enumerate(self.cfg.possible_agents):
            j = 1 - i  # opponent index

            curr_gate_w = self._gate_center_pos[self._gate_idx[:, i]]
            next_gate_w = self._gate_center_pos[(self._gate_idx[:, i] + 1) % _NUM_GATES]
            curr_gate_b, _ = subtract_frame_transforms(positions[i], quats[i], curr_gate_w)
            next_gate_b, _ = subtract_frame_transforms(positions[i], quats[i], next_gate_w)

            opp_pos_b, _ = subtract_frame_transforms(positions[i], quats[i], positions[j])

            opp_vel_b = quat_apply_inverse(quats[i], lin_vels_w[j])

            obs[agent] = torch.cat(
                [
                    quats[i],       # 4
                    lin_vels_b[i],  # 3
                    ang_vels_b[i],  # 3
                    curr_gate_b,    # 3
                    next_gate_b,    # 3
                    opp_pos_b,      # 3
                    opp_vel_b,      # 3
                    time_enc,       # 4
                ],
                dim=-1,
            )  # total = 26

        return obs

    def _get_states(self) -> torch.Tensor:
        """Global state for MAPPO centralised critic (28-D)."""
        parts: list[torch.Tensor] = []
        for r in self._robots:
            parts.extend([
                r.data.root_pos_w,       # 3
                r.data.root_quat_w,      # 4
                r.data.root_lin_vel_b,   # 3
                r.data.root_ang_vel_b,   # 3
            ])
        gate_idx_norm = self._gate_idx.float() / _NUM_GATES  # [E, 2]
        parts.append(gate_idx_norm)
        return torch.cat(parts, dim=-1)  # [E, 28]

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        crossed, lap_completed = self._detect_gate_crossings()

        progress = self._get_progress_score()  # [E, 2]

        rewards: dict[str, torch.Tensor] = {}

        for i, agent in enumerate(self.cfg.possible_agents):
            j = 1 - i  # opponent index
            pos_w = self._robots[i].data.root_pos_w  # [E, 3]

            curr_gate_w = self._gate_center_pos[self._gate_idx[:, i]]  # [E, 3]
            diff = pos_w - curr_gate_w
            curr_dist = diff.norm(dim=-1)
            r_progress = (self._prev_dist[:, i] - curr_dist) * self.cfg.progress_scale
            self._prev_dist[:, i] = curr_dist.detach().clone()
            self._episode_sums["progress"] += r_progress.clamp(min=0.0)

            lin_vel_w = self._robots[i].data.root_lin_vel_w  # [E, 3]
            to_gate = -diff / (curr_dist.unsqueeze(-1) + 1e-6)  # unit vec toward gate
            speed_toward = torch.sum(lin_vel_w * to_gate, dim=-1).clamp(min=0.0)
            r_speed = self.cfg.speed_bonus_scale * speed_toward

            quat = self._robots[i].data.root_quat_w
            up_z = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
            r_up = self.cfg.uprightness_scale * ((up_z + 1.0) / 2.0).pow(2)

            r_effort = -self.cfg.effort_weight * self._actions[agent].pow(2).mean(dim=-1)

            self._steps_since_gate[:, i] += 1.0
            time_bonus = torch.exp(-self._steps_since_gate[:, i] / 200.0)
            r_gate = torch.zeros(self.num_envs, device=self.device)

            if crossed[:, i].any():
                mask = crossed[:, i]
                leading = progress[mask, i] > progress[mask, j]
                tied = progress[mask, i] == progress[mask, j]

                gate_r = torch.zeros(mask.sum(), device=self.device)
                gate_r[leading] = self.cfg.gate_pass_reward_lead
                gate_r[tied] = self.cfg.gate_pass_reward_tie
                gate_r = gate_r * (1.0 + time_bonus[mask])
                r_gate[mask] = gate_r

                self._steps_since_gate[mask, i] = 0.0
                self._episode_sums[f"gates_{agent}"][mask] += 1.0

            r_lap = torch.zeros(self.num_envs, device=self.device)
            if lap_completed[:, i].any():
                mask = lap_completed[:, i]
                leading = progress[mask, i] >= progress[mask, j]
                lap_r = torch.zeros(mask.sum(), device=self.device)
                lap_r[leading] = self.cfg.lap_reward
                r_lap[mask] = lap_r
                self._episode_sums[f"laps_{agent}"][mask] += 1.0

            ang_vel = self._robots[i].data.root_ang_vel_b  # [E, 3]
            r_cmd = (
                self.cfg.cmd_roll_pitch_weight * (ang_vel[:, 0].pow(2) + ang_vel[:, 1].pow(2))
                + self.cfg.cmd_yaw_weight * ang_vel[:, 2].pow(2)
            )

            inter_dist = (pos_w - self._robots[j].data.root_pos_w).norm(dim=-1)
            r_contact = torch.where(
                inter_dist < _CONTACT_DIST,
                torch.full_like(inter_dist, self.cfg.contact_penalty),
                torch.zeros_like(inter_dist),
            )

            total = (
                r_progress + r_gate + r_lap + r_speed
                + (r_progress + 0.2) * r_up  # uprightness scaled by progress
                + r_effort + r_cmd + r_contact
            )

            crash = (
                (pos_w[:, 2] < 0.3)
                | (pos_w[:, 2] > 10.0)
                | (pos_w[:, :2].norm(dim=-1) > 12.0)
                | torch.isnan(pos_w).any(dim=-1)
            )
            total = torch.where(crash, torch.full_like(total, self.cfg.crash_penalty), total)

            rewards[agent] = total

        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for robot in self._robots:
            pos_w = robot.data.root_pos_w
            terminated = terminated | (
                (pos_w[:, 2] < 0.3)
                | (pos_w[:, 2] > 10.0)
                | (pos_w[:, :2].norm(dim=-1) > 12.0)
                | torch.isnan(pos_w).any(dim=-1)
            )

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return (
            {a: terminated for a in self.cfg.possible_agents},
            {a: time_out for a in self.cfg.possible_agents},
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

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

        gate_start = torch.randint(0, _NUM_GATES, (M,), device=self.device)
        gate_pos = self._gate_center_pos[gate_start]      # [M, 3]
        gate_norm = self._gate_world_normal[gate_start]    # [M, 3]

        lateral_dir = torch.stack([-gate_norm[:, 1], gate_norm[:, 0], torch.zeros(M, device=self.device)], dim=-1)
        lateral_dir = lateral_dir / (lateral_dir.norm(dim=-1, keepdim=True) + 1e-6)

        origins = self.scene.env_origins[env_ids]  # [M, 3]
        gate_yaws = torch.tensor(_GATE_YAWS, device=self.device)[gate_start]

        for i, robot in enumerate(self._robots):
            offset = 0.4 if i == 0 else -0.4
            noise = torch.zeros(M, 3, device=self.device).uniform_(-0.15, 0.15)
            spawn_pos_w = gate_pos - 3.0 * gate_norm + origins + offset * lateral_dir + noise

            roll = torch.empty(M, device=self.device).uniform_(-0.05, 0.05) * math.pi
            pitch = torch.empty(M, device=self.device).uniform_(-0.05, 0.05) * math.pi
            yaw = gate_yaws + torch.empty(M, device=self.device).uniform_(-0.1, 0.1) * math.pi
            init_quat = quat_from_euler_xyz(roll, pitch, yaw)

            default_root = robot.data.default_root_state[env_ids].clone()
            default_root[:, :3] = spawn_pos_w
            default_root[:, 3:7] = init_quat
            default_root[:, 7:] = 0.0

            robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)

            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

            self._ctrls[i].reset(env_ids)

        self._gate_idx[env_ids, :] = gate_start.unsqueeze(-1).expand(-1, _N)
        self._lap_count[env_ids, :] = 0

        for i, robot in enumerate(self._robots):
            pos_w = robot.data.root_pos_w[env_ids]
            g_pos = self._gate_center_pos[self._gate_idx[env_ids, i]]
            g_norm = self._gate_world_normal[self._gate_idx[env_ids, i]]
            diff = pos_w - g_pos
            self._prev_signed[env_ids, i] = (diff * g_norm).sum(dim=-1)
            self._prev_dist[env_ids, i] = diff.norm(dim=-1)
        self._steps_since_gate[env_ids, :] = 0.0

        if self.cfg.debug_vis and (env_ids == 0).any().item():
            self._draw_track()
            if not self._camera_initialised:
                self._init_camera()
                self._camera_initialised = True

    def _draw_track(self):
        """Draw the track circuit as green lines between consecutive gate centres."""
        self._draw.clear_lines()
        points = [self._gate_center_pos[g].cpu().tolist() for g in range(_NUM_GATES)]
        green = (0.0, 1.0, 0.0, 1.0)
        starts = points
        ends = points[1:] + [points[0]]
        self._draw.draw_lines(starts, ends, [green] * _NUM_GATES, [3.0] * _NUM_GATES)

    def _init_camera(self):
        """Aim the viewport camera at the track from a high angle."""
        import numpy as np
        from isaacsim.core.utils.viewports import set_camera_view

        gate_positions = torch.tensor(_GATE_WORLD_POS, dtype=torch.float32)
        center = gate_positions.mean(0).numpy()
        look_at = center.copy()
        eye = look_at + np.array([0.0, -14.0, 10.0], dtype=np.float32)
        set_camera_view(eye=eye, target=look_at)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_blue_marker"):
                blue_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/TeamMarker/blue",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.06,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 0.2, 1.0),
                                emissive_color=(0.0, 0.1, 0.5),
                            ),
                        ),
                    },
                )
                self._blue_marker = VisualizationMarkers(blue_cfg)

                red_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/TeamMarker/red",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.06,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.1, 0.0),
                                emissive_color=(0.5, 0.05, 0.0),
                            ),
                        ),
                    },
                )
                self._red_marker = VisualizationMarkers(red_cfg)

            self._blue_marker.set_visibility(True)
            self._red_marker.set_visibility(True)
        else:
            if hasattr(self, "_blue_marker"):
                self._blue_marker.set_visibility(False)
                self._red_marker.set_visibility(False)
            self._draw.clear_lines()

    def _debug_vis_callback(self, event):
        """Update team marker positions (floating 0.15 m above each drone)."""
        if hasattr(self, "_blue_marker"):
            blue_pos = self._robots[0].data.root_pos_w.clone()
            blue_pos[:, 2] += 0.15
            self._blue_marker.visualize(blue_pos)

            red_pos = self._robots[1].data.root_pos_w.clone()
            red_pos[:, 2] += 0.15
            self._red_marker.visualize(red_pos)
