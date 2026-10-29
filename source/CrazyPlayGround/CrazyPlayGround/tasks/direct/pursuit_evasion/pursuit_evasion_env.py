"""Two-Crazyflie pursuit-evasion environment.

One drone (the *pursuer*, ``Robot``) is trained with PPO to collide with the
other one (the *evader*, ``Evader``). The evader is not learning: a scripted
policy sends velocity setpoints (straight line or circle) to its own cascaded
PID, so both drones fly with the same dynamics.

Critic state (privileged, 26-D), positions relative to the env origin:

    [pos_w pursuer(3), pos_w evader(3),
     lin_vel_w pursuer(3), lin_vel_w evader(3),
     quat_w pursuer(4), quat_w evader(4),
     ang_vel_b pursuer(3), ang_vel_b evader(3)]

Actor observation (16-D):

    [q_pursuer_evader(4)   orientation of the evader in the pursuer frame, w >= 0
     ang_vel_b(3)          pursuer body rates, rad/s, times ``ang_vel_obs_scale``
     visible, u, v, size   ideal camera detection of the evader (see ``detection.py``)
     time_since_seen(1)    seconds since the last detection, saturated at 1 s
     last action(4)]

The actor never sees a ground-truth position. Its shaping reward is a
*progress* term computed from the detections alone: the decrease of the range
implied by the box size between two consecutive detections, plus a term that
keeps the evader centred in the image. The episode ends with a success bonus
when a contact sensor on the pursuer reports a contact with the evader (a
collision).
"""

from __future__ import annotations

import math
import pathlib as _pathlib

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, subtract_frame_transforms, yaw_quat
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from CrazyPlayGround.controllers import CascadePIDController, load_config

from .detection import Detector, DetectorCfg

_DEFAULT_DRONE_CONFIG = str(
    _pathlib.Path(__file__).resolve().parents[3] / "controllers" / "crazyflie.yaml"
)

_EVADER_STRAIGHT = 0
_EVADER_CIRCLE = 1


@configclass
class PursuitEvasionEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 5

    control_mode: str = "attitude"
    """Setpoints the policy sends to the cascaded PID:

    * ``"rate"`` (ACRO): [roll_rate, pitch_rate, yaw_rate, thrust], body rates straight to the rate loop.
    * ``"attitude"`` (angle mode, self-levelling): [roll, pitch, yaw_rate, thrust].
    * ``"velocity"``: [vx, vy, yaw_rate, vz], velocities in the heading frame.
    """
    action_space = 4
    observation_space = 16
    state_space = 26

    drone_config_path: str = _DEFAULT_DRONE_CONFIG

    # Rate-mode (ACRO) action scaling (same as Rate-Hovering).
    max_body_rate = math.pi
    # Attitude-mode action scaling (same as Att-Hovering).
    max_roll_pitch = 30.0 * math.pi / 180.0
    max_yaw_rate = 180.0 * math.pi / 180.0
    min_thrust_scale = 0.5
    max_thrust_scale = 1.8
    # Velocity-mode action scaling.
    max_velocity = 2.0

    ang_vel_obs_scale = 0.1
    """Fixed input scaling of the body rates (Orcax PPO has no observation normaliser)."""

    # Camera and detector on the pursuer.
    detector: DetectorCfg = DetectorCfg()

    # Evader scripted policy.
    evader_speed_range: tuple[float, float] = (0.0, 0.6)
    """Evader speed, sampled per episode, m/s."""
    evader_circle_prob: float = 0.3
    """Fraction of episodes where the evader flies a circle instead of a straight line."""
    evader_circle_radius_range: tuple[float, float] = (0.8, 2.0)
    evader_climb_rate_max: float = 0.2
    """Max |vz| of a straight-line evader; it bounces between the altitude limits below."""
    evader_altitude_range: tuple[float, float] = (0.7, 2.5)

    # Spawn.
    initial_distance_range: tuple[float, float] = (1.5, 3.0)
    initial_heading_noise = 30.0 * math.pi / 180.0
    """Pursuer yaw error w.r.t. the line of sight at spawn; keep < hfov/2 so the evader starts in view."""

    # Termination.
    capture_contact_force = 1.0e-3
    """Pursuer-evader contact force (N) above which the evader counts as captured."""
    arena_radius = 10.0
    """The evader leaving this disc truncates the episode."""
    pursuer_arena_margin = 2.0
    """The pursuer crashes only beyond ``arena_radius + pursuer_arena_margin``, so chasing near the edge is not punished."""
    min_altitude = 0.15
    max_altitude = 4.0

    # Rewards.
    progress_reward_scale = 10.0
    """Per metre of detected range decrease."""
    max_progress_per_step = 0.1
    """Clip of the per-step detected range decrease, m."""
    centering_reward_scale = 1.0
    lost_reward_scale = -1.0
    ang_vel_reward_scale = -0.05
    action_rate_reward_scale = -0.05
    capture_reward = 50.0
    crash_reward = -20.0

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
    # clone_in_fabric=False: contact sensors resolve their envs on the USD stage, and Fabric clones
    # are not there (the sensor would see 1 env and fail to initialise).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=8.0, replicate_physics=True, clone_in_fabric=False
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=CRAZYFLIE_CFG.spawn.replace(activate_contact_sensors=True),
    )
    # Offset default pose so the two drones never overlap before the first reset.
    evader: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Evader",
        init_state=CRAZYFLIE_CFG.init_state.replace(pos=(1.0, 0.0, 0.5)),
    )
    # Capture sensor: contacts between a pursuer body and any evader body, over every physics
    # substep of the last policy step (history_length = decimation). PhysX filters need exactly
    # one filter match per sensor body, so the env clones this cfg once per pursuer body.
    capture_bodies: tuple[str, ...] = ("body", "m1_prop", "m2_prop", "m3_prop", "m4_prop")
    capture_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        filter_prim_paths_expr=[
            f"/World/envs/env_.*/Evader/{body}" for body in ("body", "m1_prop", "m2_prop", "m3_prop", "m4_prop")
        ],
        history_length=decimation,
        update_period=0.0,
    )


@configclass
class RatePursuitEvasionEnvCfg(PursuitEvasionEnvCfg):
    control_mode = "rate"


@configclass
class AttPursuitEvasionEnvCfg(PursuitEvasionEnvCfg):
    control_mode = "attitude"


@configclass
class VelPursuitEvasionEnvCfg(PursuitEvasionEnvCfg):
    control_mode = "velocity"


class PursuitEvasionEnv(DirectRLEnv):
    cfg: PursuitEvasionEnvCfg

    def __init__(self, cfg: PursuitEvasionEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.control_mode not in ("rate", "attitude", "velocity"):
            raise ValueError(f"control_mode must be 'rate', 'attitude' or 'velocity', got {cfg.control_mode!r}")
        super().__init__(cfg, render_mode, **kwargs)

        n = self.num_envs
        self._actions = torch.zeros(n, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)

        drone_cfg = load_config(self.cfg.drone_config_path)
        make_ctrl = lambda: CascadePIDController.from_drone_config(  # noqa: E731
            drone_cfg, num_envs=n, dt=self.cfg.sim.dt, device=self.device
        )
        self._ctrl = make_ctrl()
        self._evader_ctrl = make_ctrl()

        hover_thrust = drone_cfg.physics.mass * 9.81
        self._min_thrust = self.cfg.min_thrust_scale * hover_thrust
        self._max_thrust = self.cfg.max_thrust_scale * hover_thrust

        self._body_id = self._robot.find_bodies("body")[0]
        self._evader_body_id = self._evader.find_bodies("body")[0]
        self._thrust = torch.zeros(n, 1, 3, device=self.device)
        self._moment = torch.zeros(n, 1, 3, device=self.device)
        self._evader_thrust = torch.zeros(n, 1, 3, device=self.device)
        self._evader_moment = torch.zeros(n, 1, 3, device=self.device)

        # Pursuer setpoints.
        self._rate_ref = torch.zeros(n, 3, device=self.device)
        self._att_ref = torch.zeros(n, 3, device=self.device)
        self._yaw_rate_ref = torch.zeros(n, 1, device=self.device)
        self._thrust_pwm = torch.zeros(n, 1, device=self.device)
        self._vel_ref_w = torch.zeros(n, 3, device=self.device)

        # Evader scripted policy parameters, sampled per episode.
        self._evader_mode = torch.zeros(n, dtype=torch.long, device=self.device)
        self._evader_speed = torch.zeros(n, device=self.device)
        self._evader_heading = torch.zeros(n, device=self.device)
        self._evader_vz = torch.zeros(n, device=self.device)
        self._evader_circle_center_w = torch.zeros(n, 3, device=self.device)
        self._evader_circle_radius = torch.zeros(n, 1, device=self.device)
        self._evader_circle_dir = torch.ones(n, device=self.device)
        self._evader_vel_ref_w = torch.zeros(n, 3, device=self.device)
        self._evader_yaw_ref = torch.zeros(n, 1, device=self.device)

        # Detection memory, used by the progress reward and the observation.
        self._detector = Detector(self.cfg.detector, self.device)
        self._det: dict[str, torch.Tensor] = {}
        self._prev_visible = torch.zeros(n, device=self.device)
        self._prev_range_est = torch.zeros(n, device=self.device)
        self._time_since_seen = torch.zeros(n, device=self.device)
        self._captured = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._crashed = torch.zeros(n, dtype=torch.bool, device=self.device)

        self._episode_sums = {
            key: torch.zeros(n, device=self.device)
            for key in ["progress", "centering", "lost", "ang_vel", "action_rate", "capture", "crash"]
        }
        self._episode_visible_steps = torch.zeros(n, device=self.device)
        self._episode_steps = torch.zeros(n, device=self.device)
        self._min_distance = torch.full((n,), float("inf"), device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._evader = Articulation(self.cfg.evader)
        self.scene.articulations["evader"] = self._evader
        self._capture_sensors = []
        for body in self.cfg.capture_bodies:
            sensor = ContactSensor(self.cfg.capture_sensor.replace(prim_path=f"/World/envs/env_.*/Robot/{body}"))
            self.scene.sensors[f"capture_{body}"] = sensor
            self._capture_sensors.append(sensor)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.extras["log"] = {}  # only steps with resets carry episode logs
        self._prev_actions[:] = self._actions
        self._actions = actions.clone().clamp(-1.0, 1.0)

        self._yaw_rate_ref = (self._actions[:, 2] * self.cfg.max_yaw_rate).unsqueeze(-1)
        if self.cfg.control_mode in ("rate", "attitude"):
            thrust_norm = (self._actions[:, 3] + 1.0) * 0.5
            thrust_n = self._min_thrust + thrust_norm * (self._max_thrust - self._min_thrust)
            self._thrust_pwm = (thrust_n / self._ctrl.thrust_cmd_scale).unsqueeze(-1)
        if self.cfg.control_mode == "rate":
            self._rate_ref = self._actions[:, :3] * self.cfg.max_body_rate
        elif self.cfg.control_mode == "attitude":
            roll_ref = self._actions[:, 0] * self.cfg.max_roll_pitch
            pitch_ref = self._actions[:, 1] * self.cfg.max_roll_pitch
            self._att_ref = torch.stack([roll_ref, pitch_ref, torch.zeros_like(roll_ref)], dim=-1)
        else:
            # [vx, vy, vz] are expressed in the heading (yaw-only) frame, since the
            # policy does not observe its own world yaw.
            vel_h = torch.stack([self._actions[:, 0], self._actions[:, 1], self._actions[:, 3]], dim=-1)
            self._vel_ref_w = quat_apply(yaw_quat(self._robot.data.root_quat_w), vel_h * self.cfg.max_velocity)

        self._update_evader_policy()

    def _update_evader_policy(self):
        """Scripted evader: velocity setpoint for a straight line or a circle, world frame."""
        pos_w = self._evader.data.root_pos_w
        origin_z = self._terrain.env_origins[:, 2]
        alt = pos_w[:, 2] - origin_z
        lo, hi = self.cfg.evader_altitude_range

        # Straight line: constant horizontal velocity, vertical speed bounces between the altitude limits.
        self._evader_vz = torch.where(alt < lo, self._evader_vz.abs(), self._evader_vz)
        self._evader_vz = torch.where(alt > hi, -self._evader_vz.abs(), self._evader_vz)
        straight = torch.stack(
            [
                self._evader_speed * torch.cos(self._evader_heading),
                self._evader_speed * torch.sin(self._evader_heading),
                self._evader_vz,
            ],
            dim=-1,
        )

        # Circle: tangential velocity plus a radial correction towards the nominal radius.
        radial = pos_w - self._evader_circle_center_w
        radial[:, 2] = 0.0
        r = torch.linalg.norm(radial, dim=-1, keepdim=True).clamp_min(1e-3)
        radial_dir = radial / r
        tangent = torch.stack([-radial_dir[:, 1], radial_dir[:, 0], torch.zeros_like(r[:, 0])], dim=-1)
        tangent = tangent * self._evader_circle_dir.unsqueeze(-1)
        circle = self._evader_speed.unsqueeze(-1) * tangent - 0.5 * (r - self._evader_circle_radius) * radial_dir

        is_circle = (self._evader_mode == _EVADER_CIRCLE).unsqueeze(-1)
        self._evader_vel_ref_w = torch.where(is_circle, circle, straight)

        # Face the direction of travel (only matters visually: the pursuer sees a sphere).
        moving = torch.linalg.norm(self._evader_vel_ref_w[:, :2], dim=-1) > 0.05
        heading = torch.atan2(self._evader_vel_ref_w[:, 1], self._evader_vel_ref_w[:, 0])
        self._evader_yaw_ref[:, 0] = torch.where(moving, heading, self._evader_yaw_ref[:, 0])

    def _apply_action(self):
        root_state = self._root_state(self._robot)
        if self.cfg.control_mode == "rate":
            thrust, moment = self._ctrl(
                root_state,
                target_body_rates=self._rate_ref,
                thrust_cmd=self._thrust_pwm,
                command_level="body_rate",
                body_rates_in_body_frame=True,
            )
        elif self.cfg.control_mode == "attitude":
            thrust, moment = self._ctrl(
                root_state,
                target_attitude=self._att_ref,
                target_yaw_rate=self._yaw_rate_ref,
                thrust_cmd=self._thrust_pwm,
                command_level="attitude",
                body_rates_in_body_frame=True,
            )
        else:
            thrust, moment = self._ctrl(
                root_state,
                target_vel=self._vel_ref_w,
                target_yaw_rate=self._yaw_rate_ref,
                command_level="velocity",
                body_rates_in_body_frame=True,
            )
        self._thrust[:, 0, 2] = thrust.squeeze(-1)
        self._moment[:, 0, :] = moment
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        thrust, moment = self._evader_ctrl(
            self._root_state(self._evader),
            target_vel=self._evader_vel_ref_w,
            target_yaw=self._evader_yaw_ref,
            command_level="velocity",
            body_rates_in_body_frame=True,
        )
        self._evader_thrust[:, 0, 2] = thrust.squeeze(-1)
        self._evader_moment[:, 0, :] = moment
        self._evader.set_external_force_and_torque(
            self._evader_thrust, self._evader_moment, body_ids=self._evader_body_id
        )

    @staticmethod
    def _root_state(robot: Articulation) -> torch.Tensor:
        return torch.cat(
            [robot.data.root_pos_w, robot.data.root_quat_w, robot.data.root_lin_vel_w, robot.data.root_ang_vel_b],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Observations, rewards, dones
    # ------------------------------------------------------------------

    def _detect(self) -> dict[str, torch.Tensor]:
        return self._detector(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._evader.data.root_pos_w
        )

    def _get_observations(self) -> dict:
        det = self._detect()
        if self._det:
            # Envs that did not reset keep the detection drawn in _get_dones, so the actor sees
            # the same (possibly noisy) detection that the reward and the memory used.
            kept = ~self.reset_buf
            det = {k: torch.where(kept, self._det[k], v) for k, v in det.items()}
        p, e = self._robot.data, self._evader.data

        _, q_rel = subtract_frame_transforms(p.root_pos_w, p.root_quat_w, e.root_pos_w, e.root_quat_w)
        q_rel = torch.where(q_rel[:, :1] < 0.0, -q_rel, q_rel)  # q and -q are the same rotation

        obs = torch.cat(
            [
                q_rel,
                p.root_ang_vel_b * self.cfg.ang_vel_obs_scale,
                torch.stack([det["visible"], det["u"], det["v"], det["size"]], dim=-1),
                self._time_since_seen.clamp(max=1.0).unsqueeze(-1),
                self._actions,
            ],
            dim=-1,
        )

        origins = self._terrain.env_origins
        state = torch.cat(
            [
                p.root_pos_w - origins,
                e.root_pos_w - origins,
                p.root_lin_vel_w,
                e.root_lin_vel_w,
                p.root_quat_w,
                e.root_quat_w,
                p.root_ang_vel_b,
                e.root_ang_vel_b,
            ],
            dim=-1,
        )
        return {"policy": obs, "critic": state}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        p_pos = self._robot.data.root_pos_w
        e_pos = self._evader.data.root_pos_w
        origins = self._terrain.env_origins

        distance = torch.linalg.norm(p_pos - e_pos, dim=-1)
        self._min_distance = torch.minimum(self._min_distance, distance)
        # force_matrix_w_history: (num_envs, T, 1, evader bodies, 3), N, one sensor per pursuer body.
        contact_force = torch.stack(
            [sensor.data.force_matrix_w_history.norm(dim=-1).flatten(1).amax(dim=1) for sensor in self._capture_sensors],
            dim=-1,
        ).amax(dim=-1)
        self._captured = contact_force > self.cfg.capture_contact_force

        p_alt = p_pos[:, 2] - origins[:, 2]
        p_xy = torch.linalg.norm(p_pos[:, :2] - origins[:, :2], dim=-1)
        self._crashed = (
            (p_alt < self.cfg.min_altitude)
            | (p_alt > self.cfg.max_altitude)
            | (p_xy > self.cfg.arena_radius + self.cfg.pursuer_arena_margin)
            | torch.isnan(p_pos).any(dim=-1)
        ) & ~self._captured

        # The evader leaving the arena is not the pursuer's fault: truncate, don't punish.
        e_xy = torch.linalg.norm(e_pos[:, :2] - origins[:, :2], dim=-1)
        time_out = (self.episode_length_buf >= self.max_episode_length - 1) | (e_xy > self.cfg.arena_radius)

        # Detections at the post-physics state, shared with _get_rewards.
        self._det = self._detect()
        return self._captured | self._crashed, time_out

    def _get_rewards(self) -> torch.Tensor:
        det = self._det
        visible = det["visible"]

        # Progress from detections only: range implied by the box size, compared with the previous detection.
        range_est = self._detector.size_to_range(det["size"].clamp_min(1e-6))
        both_seen = visible * self._prev_visible
        progress = (self._prev_range_est - range_est).clamp(
            -self.cfg.max_progress_per_step, self.cfg.max_progress_per_step
        ) * both_seen
        centering = visible * (1.0 - 0.5 * (det["u"].square() + det["v"].square()))
        lost = 1.0 - visible
        ang_vel = self._robot.data.root_ang_vel_b.square().sum(dim=-1)
        action_rate = (self._actions - self._prev_actions).square().sum(dim=-1)

        dt = self.step_dt
        rewards = {
            "progress": progress * self.cfg.progress_reward_scale,
            "centering": centering * self.cfg.centering_reward_scale * dt,
            "lost": lost * self.cfg.lost_reward_scale * dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * dt,
            "action_rate": action_rate * self.cfg.action_rate_reward_scale * dt,
            "capture": self._captured.float() * self.cfg.capture_reward,
            "crash": self._crashed.float() * self.cfg.crash_reward,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # Update the detection memory.
        self._prev_range_est = torch.where(visible > 0, range_est, self._prev_range_est)
        self._prev_visible = visible
        self._time_since_seen = torch.where(visible > 0, torch.zeros_like(self._time_since_seen), self._time_since_seen + dt)
        self._episode_visible_steps += visible
        self._episode_steps += 1.0
        return reward

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        steps = self._episode_steps[env_ids].clamp_min(1.0)
        log = {f"Episode_Reward/{k}": v[env_ids].mean() / self.max_episode_length_s for k, v in self._episode_sums.items()}
        log["Episode_Termination/capture"] = torch.count_nonzero(self._captured[env_ids]).item()
        log["Episode_Termination/crash"] = torch.count_nonzero(self._crashed[env_ids]).item()
        log["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        log["Metrics/capture_rate"] = self._captured[env_ids].float().mean().item()
        min_distance = self._min_distance[env_ids]
        stepped = torch.isfinite(min_distance)  # false on the initial reset, before any step
        if stepped.any():
            log["Metrics/min_distance"] = min_distance[stepped].mean().item()
        log["Metrics/visible_fraction"] = (self._episode_visible_steps[env_ids] / steps).mean().item()
        self.extras["log"] = log
        for v in self._episode_sums.values():
            v[env_ids] = 0.0
        self._episode_visible_steps[env_ids] = 0.0
        self._episode_steps[env_ids] = 0.0
        self._min_distance[env_ids] = float("inf")

        self._robot.reset(env_ids)
        self._evader.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread resets over time so that all envs don't restart together.
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        k = len(env_ids)
        origins = self._terrain.env_origins[env_ids]

        def uniform(lo: float, hi: float, *shape: int) -> torch.Tensor:
            return torch.empty(*shape, device=self.device).uniform_(lo, hi)

        # Evader spawn, near the env origin.
        evader_pos_w = origins.clone()
        evader_pos_w[:, :2] += uniform(-1.0, 1.0, k, 2)
        evader_pos_w[:, 2] += uniform(*self.cfg.evader_altitude_range, k)

        # Pursuer spawn: at a random bearing and distance from the evader, facing it up to some heading noise.
        bearing = uniform(-math.pi, math.pi, k)
        dist = uniform(*self.cfg.initial_distance_range, k)
        pursuer_pos_w = evader_pos_w.clone()
        pursuer_pos_w[:, 0] += dist * torch.cos(bearing)
        pursuer_pos_w[:, 1] += dist * torch.sin(bearing)
        pursuer_pos_w[:, 2] = origins[:, 2] + (evader_pos_w[:, 2] - origins[:, 2] + uniform(-0.3, 0.3, k)).clamp(0.5, 3.0)
        pursuer_yaw = bearing + math.pi + uniform(-self.cfg.initial_heading_noise, self.cfg.initial_heading_noise, k)

        # Evader policy parameters.
        self._evader_mode[env_ids] = (torch.rand(k, device=self.device) < self.cfg.evader_circle_prob).long()
        self._evader_speed[env_ids] = uniform(*self.cfg.evader_speed_range, k)
        self._evader_heading[env_ids] = uniform(-math.pi, math.pi, k)
        self._evader_vz[env_ids] = uniform(-self.cfg.evader_climb_rate_max, self.cfg.evader_climb_rate_max, k)
        circle_radius = uniform(*self.cfg.evader_circle_radius_range, k)
        center_angle = uniform(-math.pi, math.pi, k)
        self._evader_circle_center_w[env_ids, 0] = evader_pos_w[:, 0] - circle_radius * torch.cos(center_angle)
        self._evader_circle_center_w[env_ids, 1] = evader_pos_w[:, 1] - circle_radius * torch.sin(center_angle)
        self._evader_circle_center_w[env_ids, 2] = evader_pos_w[:, 2]
        self._evader_circle_radius[env_ids, 0] = circle_radius
        self._evader_circle_dir[env_ids] = torch.where(torch.rand(k, device=self.device) < 0.5, -1.0, 1.0)
        evader_yaw = self._evader_heading[env_ids]
        self._evader_yaw_ref[env_ids, 0] = evader_yaw

        zeros = torch.zeros(k, device=self.device)
        for robot, pos_w, yaw in ((self._robot, pursuer_pos_w, pursuer_yaw), (self._evader, evader_pos_w, evader_yaw)):
            pose = torch.cat([pos_w, quat_from_euler_xyz(zeros, zeros, yaw)], dim=-1)
            robot.write_root_pose_to_sim(pose, env_ids)
            robot.write_root_velocity_to_sim(torch.zeros(k, 6, device=self.device), env_ids)
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos[env_ids], robot.data.default_joint_vel[env_ids], None, env_ids
            )

        self._ctrl.reset(env_ids)
        self._evader_ctrl.reset(env_ids)
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_visible[env_ids] = 0.0
        self._prev_range_est[env_ids] = 0.0
        self._time_since_seen[env_ids] = 0.0
        self._captured[env_ids] = False
        self._crashed[env_ids] = False
