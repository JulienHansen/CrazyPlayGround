# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Formation control environments for 4 Crazyflie drones.

Two control modes are provided via sub-configs:
  - VelFormationEnvCfg  action_space=3  (velocity setpoint per drone)
  - AttFormationEnvCfg  action_space=4  (roll, pitch, yaw_rate, thrust per drone)

Each drone is a separate MARL agent (cooperative task, shared reward).

Observation per drone (25-D):
  target_pos_b  (3)       — formation target for this drone in its body frame
  lin_vel_b     (3)       — body-frame linear velocity
  ang_vel_b     (3)       — body-frame angular velocity
  quat          (4)       — orientation [w, x, y, z]
  rel_pos_b_j   (3 each)  — positions of the other 3 drones in this drone's body frame
  dist_j        (1 each)  — distances to the other 3 drones
  => 13 + 3 * (3 + 1) = 25

Global state (52-D, for MAPPO):
  per drone: pos(3) + quat(4) + lin_vel_b(3) + ang_vel_b(3) = 13
  total: 4 * 13 = 52

Reward (cooperative — all agents receive the same scalar):
  r_formation  : 1 / (1 + (hausdorff * 1.6)^2)
  r_pos        : exp(- centroid_dist)
  r_separation : clamp((min_pairwise_dist / safe_dist)^2, 0, 1)  [gate reward]
  penalty_ang  : mean squared angular velocity (small negative coefficient)
"""

from __future__ import annotations

import math
import pathlib as _pathlib
from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from CrazyPlayGround.controllers import CascadePIDController, load_config, resolve_max_thrust_scale

from ..crazyflie_events import CrazyflieEventCfg

_DEFAULT_DRONE_CONFIG = str(
    _pathlib.Path(__file__).resolve().parents[3] / "controllers" / "crazyflie.yaml"
)

FORMATIONS: dict[str, list[list[float]]] = {
    "tetragon": [
        [ 1.0,  1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [-1.0,  1.0, 0.0],
    ],
    "line": [
        [-1.5, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [ 0.5, 0.0, 0.0],
        [ 1.5, 0.0, 0.0],
    ],
    "diamond": [
        [ 0.0,  1.5, 0.0],
        [ 1.0,  0.0, 0.0],
        [ 0.0, -1.5, 0.0],
        [-1.0,  0.0, 0.0],
    ],
}

_N = 4
_OBS_DIM = 13 + (_N - 1) * 4   # 13 + 12 = 25
_STATE_DIM = _N * 13            # 52
_AGENT_NAMES = [f"drone_{i}" for i in range(_N)]

@configclass
class FormationEnvCfg(DirectMARLEnvCfg):
    """Base config shared by velocity and attitude formation tasks."""

    episode_length_s: float = 15.0
    decimation: int = 5
    debug_vis: bool = True

    possible_agents: list = _AGENT_NAMES
    observation_spaces: dict = {a: _OBS_DIM for a in _AGENT_NAMES}
    state_space: int = _STATE_DIM

    formation_type: str = "tetragon"
    target_height: float = 1.5      # metres above env origin
    safe_distance: float = 0.3      # collision threshold [m]

    formation_reward_scale: float = 1.0
    pos_reward_scale: float = 0.4
    ang_vel_penalty_scale: float = -0.02

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
        num_envs=512, env_spacing=8.0, replicate_physics=True, clone_in_fabric=True
    )

    drone_config_path: str = _DEFAULT_DRONE_CONFIG
    # startup event: PhysX body mass = drone.mass of this yaml (see crazyflie_events.py)
    events: CrazyflieEventCfg = CrazyflieEventCfg()

@configclass
class VelFormationEnvCfg(FormationEnvCfg):
    """Velocity-controlled formation. Action = [Vx, Vy, Vz] per drone."""

    action_spaces: dict = {a: 3 for a in _AGENT_NAMES}
    control_mode: str = "velocity"
    max_velocity: float = 2.0       # m/s

@configclass
class AttFormationEnvCfg(FormationEnvCfg):
    """Attitude-controlled formation. Action = [roll, pitch, yaw_rate, thrust] per drone."""

    action_spaces: dict = {a: 4 for a in _AGENT_NAMES}
    control_mode: str = "attitude"
    max_roll_pitch: float = 30.0 * math.pi / 180.0   # rad
    max_yaw_rate: float = 90.0 * math.pi / 180.0     # rad/s
    min_thrust_scale: float = 0.5
    max_thrust_scale: float | None = None  # fraction of hover thrust; None = drone.max_thrust / (m·g), the firmware cap (1.5387)

class FormationEnv(DirectMARLEnv):
    cfg: FormationEnvCfg

    def __init__(self, cfg: FormationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = _N

        shape_raw = torch.tensor(
            FORMATIONS[self.cfg.formation_type], dtype=torch.float, device=self.device
        )
        self._formation_shape = shape_raw.clone()
        self._formation_shape[:, 2] = self.cfg.target_height  # [N, 3]

        self._body_ids = [r.find_bodies("body")[0] for r in self._robots]

        self._thrust = [torch.zeros(self.num_envs, 1, 3, device=self.device) for _ in range(N)]
        self._moment = [torch.zeros(self.num_envs, 1, 3, device=self.device) for _ in range(N)]

        drone_cfg = load_config(self.cfg.drone_config_path)
        self._ctrls = [
            CascadePIDController.from_drone_config(
                drone_cfg,
                num_envs=self.num_envs,
                dt=self.cfg.sim.dt,
                device=self.device,
            )
            for _ in range(N)
        ]

        if self.cfg.control_mode == "velocity":
            self._ref_vels = [
                torch.zeros(self.num_envs, 3, device=self.device) for _ in range(N)
            ]
        elif self.cfg.control_mode == "attitude":
            hover_thrust = drone_cfg.physics.mass * 9.81
            self._min_thrust = self.cfg.min_thrust_scale * hover_thrust
            self._max_thrust_scale = resolve_max_thrust_scale(
                self.cfg.max_thrust_scale, drone_cfg.physics.mass, drone_cfg.physics.max_thrust
            )
            self._max_thrust = self._max_thrust_scale * hover_thrust
            self._att_refs = [
                torch.zeros(self.num_envs, 3, device=self.device) for _ in range(N)
            ]
            self._yaw_rate_refs = [
                torch.zeros(self.num_envs, 1, device=self.device) for _ in range(N)
            ]
            self._thrust_pwms = [
                torch.zeros(self.num_envs, 1, device=self.device) for _ in range(N)
            ]

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["formation_cost", "centroid_dist", "min_separation"]
        }

        import omni.kit.app
        _ext_mgr = omni.kit.app.get_app().get_extension_manager()
        if not _ext_mgr.is_extension_enabled("isaacsim.util.debug_draw"):
            _ext_mgr.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self._camera_initialised = False
        self._formation_drawn = False

    def _setup_scene(self):
        self._robots: list[Articulation] = []
        for i in range(_N):
            robot = Articulation(
                CRAZYFLIE_CFG.replace(prim_path=f"/World/envs/env_.*/Robot_{i}")
            )
            self._robots.append(robot)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        for i, robot in enumerate(self._robots):
            self.scene.articulations[f"robot_{i}"] = robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for i, agent in enumerate(self.cfg.possible_agents):
            act = actions[agent].clone().clamp(-1.0, 1.0)

            if self.cfg.control_mode == "velocity":
                self._ref_vels[i] = act[:, :3] * self.cfg.max_velocity

            elif self.cfg.control_mode == "attitude":
                roll_ref  = act[:, 0] * self.cfg.max_roll_pitch
                pitch_ref = act[:, 1] * self.cfg.max_roll_pitch
                self._att_refs[i] = torch.stack(
                    [roll_ref, pitch_ref, torch.zeros_like(roll_ref)], dim=-1
                )
                self._yaw_rate_refs[i] = (act[:, 2] * self.cfg.max_yaw_rate).unsqueeze(-1)
                thrust_norm = act[:, 3].clamp(0.0, 1.0)
                thrust_ref  = self._min_thrust + thrust_norm * (self._max_thrust - self._min_thrust)
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

    def _get_observations(self) -> dict[str, torch.Tensor]:
        positions  = [r.data.root_pos_w    for r in self._robots]   # list of [E, 3]
        quats      = [r.data.root_quat_w   for r in self._robots]   # list of [E, 4]
        lin_vels_b = [r.data.root_lin_vel_b for r in self._robots]  # list of [E, 3]
        ang_vels_b = [r.data.root_ang_vel_b for r in self._robots]  # list of [E, 3]

        obs: dict[str, torch.Tensor] = {}

        for i, agent in enumerate(self.cfg.possible_agents):
            target_w = self.scene.env_origins + self._formation_shape[i]  # [E, 3]

            target_b, _ = subtract_frame_transforms(positions[i], quats[i], target_w)  # [E, 3]

            rel_parts: list[torch.Tensor] = []
            for j in range(_N):
                if j == i:
                    continue
                rel_b, _ = subtract_frame_transforms(positions[i], quats[i], positions[j])
                dist = (positions[j] - positions[i]).norm(dim=-1, keepdim=True)  # [E, 1]
                rel_parts.append(rel_b)
                rel_parts.append(dist)

            obs[agent] = torch.cat(
                [target_b, lin_vels_b[i], ang_vels_b[i], quats[i]] + rel_parts, dim=-1
            )

        return obs

    def _get_states(self) -> torch.Tensor:
        """Global state for MAPPO's centralised critic."""
        parts: list[torch.Tensor] = []
        for r in self._robots:
            parts.extend(
                [r.data.root_pos_w, r.data.root_quat_w, r.data.root_lin_vel_b, r.data.root_ang_vel_b]
            )
        return torch.cat(parts, dim=-1)  # [E, N*13]

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        positions = torch.stack([r.data.root_pos_w for r in self._robots], dim=1)  # [E, N, 3]

        pos_c = positions - positions.mean(dim=1, keepdim=True)  # [E, N, 3]
        des_c = (
            self._formation_shape - self._formation_shape.mean(dim=0, keepdim=True)
        ).unsqueeze(0).expand_as(pos_c)  # [E, N, 3]
        cost_h = _batch_hausdorff(pos_c, des_c)  # [E]
        r_formation = 1.0 / (1.0 + torch.square(cost_h * 1.6))

        centroid   = positions.mean(dim=1)  # [E, 3]
        target_c_w = self.scene.env_origins + torch.tensor(
            [0.0, 0.0, self.cfg.target_height], device=self.device
        )  # [E, 3]
        centroid_dist = (centroid - target_c_w).norm(dim=-1)  # [E]
        r_pos = torch.exp(-centroid_dist)

        diff   = positions.unsqueeze(2) - positions.unsqueeze(1)  # [E, N, N, 3]
        pdist  = diff.norm(dim=-1)                                 # [E, N, N]
        eye    = torch.eye(_N, device=self.device, dtype=torch.bool).unsqueeze(0)
        pdist  = pdist.masked_fill(eye, float("inf"))
        min_dist = pdist.flatten(1).min(dim=-1).values             # [E]
        r_sep  = (min_dist / self.cfg.safe_distance).clamp(0.0, 1.0) ** 2

        ang_vel_sq = sum(
            torch.sum(torch.square(r.data.root_ang_vel_b), dim=1) for r in self._robots
        ) / _N  # [E], mean over drones

        total = r_sep * (
            self.cfg.formation_reward_scale * r_formation
            + self.cfg.pos_reward_scale * r_pos
        ) + self.cfg.ang_vel_penalty_scale * ang_vel_sq * self.step_dt

        self._episode_sums["formation_cost"] += cost_h
        self._episode_sums["centroid_dist"]  += centroid_dist
        self._episode_sums["min_separation"] += min_dist.clamp(max=self.cfg.safe_distance * 4)

        return {a: total for a in self.cfg.possible_agents}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        positions = torch.stack([r.data.root_pos_w for r in self._robots], dim=1)  # [E, N, 3]

        too_low  = (positions[:, :, 2] < 0.15).any(dim=1)   # [E]
        too_high = (positions[:, :, 2] > 3.5).any(dim=1)

        diff   = positions.unsqueeze(2) - positions.unsqueeze(1)
        pdist  = diff.norm(dim=-1)
        eye    = torch.eye(_N, device=self.device, dtype=torch.bool).unsqueeze(0)
        pdist  = pdist.masked_fill(eye, float("inf"))
        too_close = pdist.flatten(1).min(dim=-1).values < (self.cfg.safe_distance * 0.75)

        terminated = too_low | too_high | too_close
        time_out   = self.episode_length_buf >= self.max_episode_length - 1

        return (
            {a: terminated for a in self.cfg.possible_agents},
            {a: time_out   for a in self.cfg.possible_agents},
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

        for i, robot in enumerate(self._robots):
            target_w = self.scene.env_origins[env_ids] + self._formation_shape[i]  # [M, 3]

            default_root = robot.data.default_root_state[env_ids].clone()

            spawn = default_root.clone()
            spawn[:, :3] = target_w
            spawn[:, 0] += torch.empty(len(env_ids), device=self.device).uniform_(-0.3, 0.3)
            spawn[:, 1] += torch.empty(len(env_ids), device=self.device).uniform_(-0.3, 0.3)
            spawn[:, 2] += torch.empty(len(env_ids), device=self.device).uniform_(-0.2, 0.2)
            spawn[:, 2]  = spawn[:, 2].clamp(min=0.3)

            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]

            robot.write_root_pose_to_sim(spawn[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

            self._ctrls[i].reset(env_ids)

        if self.cfg.debug_vis and (env_ids == 0).any().item():
            if not self._formation_drawn:
                self._draw_formation_env0()
                self._formation_drawn = True
            if not self._camera_initialised:
                self._init_camera_env0()
                self._camera_initialised = True

    def _draw_formation_env0(self):
        """Draw formation target markers and shape outline for env 0."""
        self._draw.clear_lines()

        origin = self.scene.env_origins[0]  # [3]
        pts_cpu = [(origin + self._formation_shape[i]).cpu().tolist() for i in range(_N)]

        cross = 0.12
        for x, y, z in pts_cpu:
            p0 = [(x - cross, y, z), (x, y - cross, z), (x, y, z - cross)]
            p1 = [(x + cross, y, z), (x, y + cross, z), (x, y, z + cross)]
            self._draw.draw_lines(p0, p1, [(1.0, 1.0, 0.0, 1.0)] * 3, [3.0] * 3)

        if self.cfg.formation_type == "line":
            p0 = pts_cpu[:-1]
            p1 = pts_cpu[1:]
        else:
            p0 = pts_cpu
            p1 = pts_cpu[1:] + [pts_cpu[0]]
        n = len(p0)
        self._draw.draw_lines(
            [tuple(p) for p in p0],
            [tuple(p) for p in p1],
            [(0.0, 1.0, 1.0, 0.8)] * n,
            [2.0] * n,
        )

    def _init_camera_env0(self):
        """Point the viewport camera at the formation centroid of env 0."""
        import numpy as np
        from isaacsim.core.utils.viewports import set_camera_view

        origin = self.scene.env_origins[0].cpu().numpy()
        centroid = origin + np.array([0.0, 0.0, self.cfg.target_height], dtype=np.float32)
        eye = centroid + np.array([6.0, -6.0, 4.0], dtype=np.float32)
        set_camera_view(eye=eye, target=centroid)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not debug_vis:
            self._draw.clear_lines()

    def _debug_vis_callback(self, event):
        pass

def _batch_hausdorff(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Symmetric Hausdorff distance between two batched point clouds.

    Args:
        p: [E, N, 3]
        q: [E, N, 3]

    Returns:
        [E] symmetric Hausdorff distances.
    """
    d   = torch.cdist(p, q)                            # [E, N, N]
    h_pq = d.min(dim=-1).values.max(dim=-1).values     # directed p → q  [E]
    h_qp = d.min(dim=-2).values.max(dim=-1).values     # directed q → p  [E]
    return torch.maximum(h_pq, h_qp)
