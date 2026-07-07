"""Test env: att_hovering where the ghost drone IS the target.

The learning robot must follow a moving ghost drone split 50/50 across envs:

* **Physics (first 50 %)** – the ghost is a *real PhysX actor*.
  A CascadePIDController (DroneModule) computes thrust and moment each
  substep; these are applied via ``set_external_force_and_torque`` exactly
  like the learning robot.  The ghost flies between waypoints under position
  control with true dynamics.

* **Teleport (last 50 %)** – the ghost's pose is overwritten every substep
  (kinematic).  It snaps instantly to each waypoint.

Same ArticulationCfg for both groups; the difference is whether forces or
direct pose writes are used at runtime.

Ghost ArticulationCfg
---------------------
* ``disable_gravity = False``  — PhysX gravity is active; physics ghost
  uses body-frame thrust (same as learning robot) to oppose it.  Teleport
  ghost gets its pose overwritten every substep so gravity has no visible
  effect.
* ``linear_damping = angular_damping = 0``  — no PhysX damping so forces
  produce realistic motion.

Target / reward
---------------
``_desired_pos_w`` tracks the ghost position at every step.  The base-class
obs ``[lin_vel_b, desired_pos_b]`` and reward (distance-to-goal) therefore
chase the ghost automatically.  Red cube debug visualiser is disabled.

Observation: 6-D  ``[lin_vel_b(3), ghost_rel_b(3)]``
"""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils import configclass
from isaaclab_assets import CRAZYFLIE_CFG

from .att_hovering import QuadcopterEnv, QuadcopterEnvCfg
from CrazyPlayGround.controllers import load_config

# DroneModule – CascadePIDController for the physics ghost
from CrazyPlayGround.controllers.cascade_pid import CascadePIDController as GhostPIDController


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@configclass
class TestHoveringEnvCfg(QuadcopterEnvCfg):
    observation_space = 6   # [lin_vel_b(3), ghost_rel_b(3)]
    debug_vis = False        # no red cube; ghost IS the target

    # Circular trajectory parameters (waypoints are generated at runtime).
    circle_radius: float = 0.6   # metres
    circle_height: float = 1.0   # metres above terrain origin
    n_waypoints: int = 24        # points evenly distributed around the circle

    # Teleport group: env-steps between snaps — small value → visibly fast orbit.
    waypoint_hold_steps: int = 15

    # Physics group: env-steps between target advances — larger → PID has time to catch up.
    phys_waypoint_hold_steps: int = 60

    # Ghost: gravity ON (PhysX handles it like the learning robot), zero damping.
    # Teleport ghost gets pose overwritten every substep so gravity is irrelevant.
    ghost: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Ghost",
        spawn=CRAZYFLIE_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=10.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TestHoveringEnv(QuadcopterEnv):
    cfg: TestHoveringEnvCfg

    def __init__(self, cfg: TestHoveringEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Generate circular waypoints: N equally-spaced points on a horizontal circle.
        angles = torch.linspace(0.0, 2.0 * math.pi, cfg.n_waypoints + 1)[:-1]
        self._waypoints = torch.stack([
            cfg.circle_radius * torch.cos(angles),
            cfg.circle_radius * torch.sin(angles),
            torch.full_like(angles, cfg.circle_height),
        ], dim=-1).to(self.device)

        # --- 50 / 50 split --------------------------------------------------
        # Physics: indices [0, n_phys)   — forces applied, PhysX integrates
        # Teleport: indices [n_phys, N)  — pose overwritten every substep
        self._n_phys = self.num_envs // 2
        self._phys_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._phys_mask[: self._n_phys] = True
        self._phys_ids = self._phys_mask.nonzero(as_tuple=True)[0]
        self._tele_ids = (~self._phys_mask).nonzero(as_tuple=True)[0]

        # Waypoint tracking (independent per group via the same index tensor).
        self._wp_idx      = torch.zeros(self.num_envs, dtype=torch.long,  device=self.device)
        self._wp_hold_ctr = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Ghost world-frame pose (used for observation / teleport writes).
        self._ghost_pos_w    = torch.zeros(self.num_envs, 3, device=self.device)
        self._ghost_quat_w   = torch.zeros(self.num_envs, 4, device=self.device)
        self._ghost_quat_w[:, 0] = 1.0
        self._ghost_vel_zero = torch.zeros(self.num_envs, 6, device=self.device)

        # Body ID for set_external_force_and_torque (ghost has same body name as robot).
        self._ghost_body_id = self._ghost.find_bodies("body")[0]

        # Physics ghost controller and force/torque buffers.
        if self._n_phys > 0:
            drone_cfg = load_config(self.cfg.drone_config_path)
            self._ghost_ctrl = GhostPIDController.from_drone_config(
                drone_cfg,
                num_envs=self._n_phys,
                dt=self.cfg.sim.dt,
                device=str(self.device),
            )

            # World-frame waypoint target for the physics ghost controller.
            self._ghost_target_w = torch.zeros(self._n_phys, 3, device=self.device)

        # Full-size force/torque buffers [num_envs, 1, 3] — same shape as every other
        # articulation in the codebase.  Tele-ghost rows stay zero; only phys rows are set.
        self._ghost_forces  = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._ghost_torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._ghost = Articulation(self.cfg.ghost)
        self.scene.articulations["ghost"] = self._ghost

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        self._advance_ghost()

    def _apply_action(self):
        super()._apply_action()

        # --- Teleport ghost: kinematic pose override ----------------------
        if len(self._tele_ids) > 0:
            ghost_pose_tele = torch.cat(
                [self._ghost_pos_w[self._tele_ids], self._ghost_quat_w[self._tele_ids]], dim=-1
            )
            self._ghost.write_root_pose_to_sim(ghost_pose_tele, env_ids=self._tele_ids)
            self._ghost.write_root_velocity_to_sim(
                self._ghost_vel_zero[self._tele_ids], env_ids=self._tele_ids
            )

        # --- Physics ghost: apply controller forces each substep ----------
        if self._n_phys > 0:
            # Read latest ghost state from PhysX (updated after previous substep).
            root_state = torch.cat(
                [
                    self._ghost.data.root_pos_w[self._phys_ids],
                    self._ghost.data.root_quat_w[self._phys_ids],
                    self._ghost.data.root_lin_vel_w[self._phys_ids],
                    self._ghost.data.root_ang_vel_b[self._phys_ids],
                ],
                dim=-1,
            )
            thrust, moment = self._ghost_ctrl(
                root_state,
                target_pos=self._ghost_target_w,
                command_level="position",
                body_rates_in_body_frame=True,
            )
            # Apply thrust in body-frame z (same as att_hovering robot).
            # PhysX handles gravity; no manual subtraction needed.
            self._ghost_forces[self._phys_ids, 0, 2]  = thrust.squeeze(-1)
            self._ghost_torques[self._phys_ids, 0, :]  = moment
            # No env_ids: full-size buffers, same as every other articulation.
            self._ghost.set_external_force_and_torque(
                self._ghost_forces,
                self._ghost_torques,
                body_ids=self._ghost_body_id,
            )

    # ------------------------------------------------------------------
    # Observations — read physics ghost pos from PhysX after substeps
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # Sync physics ghost position from PhysX (latest after all substeps).
        if self._n_phys > 0:
            self._ghost_pos_w[self._phys_mask] = (
                self._ghost.data.root_pos_w[self._phys_ids]
            )
        # Ghost IS the target: update base-class desired_pos.
        self._desired_pos_w[:] = self._ghost_pos_w
        return super()._get_observations()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # Guard: may be called during super().__init__() before our tensors exist.
        if not hasattr(self, "_ghost_pos_w"):
            return

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._ghost._ALL_INDICES

        self._wp_idx[env_ids]      = 0
        self._wp_hold_ctr[env_ids] = 0.0

        wp0 = self._waypoints[0]
        self._ghost_pos_w[env_ids]      = wp0.unsqueeze(0) + self._terrain.env_origins[env_ids]
        self._ghost_quat_w[env_ids, 0]  = 1.0
        self._ghost_quat_w[env_ids, 1:] = 0.0

        # Physics ghost starts 0.5 m below the first waypoint so it must actively
        # fly up — makes force application immediately visible on first reset.
        if hasattr(self, "_phys_ids"):
            phys_in_reset = env_ids[env_ids < self._n_phys]
            if len(phys_in_reset) > 0:
                self._ghost_pos_w[phys_in_reset, 2] -= 0.5

        # Sync desired_pos for first observation/reward after reset.
        self._desired_pos_w[env_ids] = self._ghost_pos_w[env_ids]

        # Reset physics controller for physics envs in env_ids.
        if hasattr(self, "_ghost_ctrl") and self._n_phys > 0:
            phys_reset = env_ids[env_ids < self._n_phys]  # local = global for [0, n_phys)
            if len(phys_reset) > 0:
                self._ghost_ctrl.reset(phys_reset)
                self._ghost_target_w[phys_reset] = wp0.unsqueeze(0) + (
                    self._terrain.env_origins[phys_reset]
                )

        # Write initial pose for all reset envs (pose + zero velocity).
        ghost_pose = torch.cat([self._ghost_pos_w[env_ids], self._ghost_quat_w[env_ids]], dim=-1)
        self._ghost.write_root_pose_to_sim(ghost_pose, env_ids=env_ids)
        self._ghost.write_root_velocity_to_sim(self._ghost_vel_zero[env_ids], env_ids=env_ids)
        self._ghost.reset(env_ids)

    # ------------------------------------------------------------------
    # Ghost advancement (waypoints + teleport pos; physics target)
    # ------------------------------------------------------------------

    def _advance_ghost(self):
        """Update waypoints and ghost positions for both groups."""
        num_wps = self._waypoints.shape[0]

        # ── Teleport group ───────────────────────────────────────────────
        if len(self._tele_ids) > 0:
            wp_local = self._waypoints[self._wp_idx]
            self._ghost_pos_w[self._tele_ids] = (
                wp_local[self._tele_ids] + self._terrain.env_origins[self._tele_ids]
            )

            self._wp_hold_ctr[self._tele_ids] += 1
            advance = (self._wp_hold_ctr >= self.cfg.waypoint_hold_steps) & (~self._phys_mask)
            self._wp_idx[advance]      = (self._wp_idx[advance] + 1) % num_wps
            self._wp_hold_ctr[advance] = 0

        # ── Physics group ────────────────────────────────────────────────
        if self._n_phys > 0:
            # World-frame target for the ghost controller.
            self._ghost_target_w[:] = (
                self._waypoints[self._wp_idx[self._phys_mask]]
                + self._terrain.env_origins[self._phys_mask]
            )

            self._wp_hold_ctr[self._phys_mask] += 1
            phys_advance = (
                (self._wp_hold_ctr >= self.cfg.phys_waypoint_hold_steps) & self._phys_mask
            )
            self._wp_idx[phys_advance]      = (self._wp_idx[phys_advance] + 1) % num_wps
            self._wp_hold_ctr[phys_advance] = 0
