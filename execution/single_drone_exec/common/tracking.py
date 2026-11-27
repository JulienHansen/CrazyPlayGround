"""Trajectory-tracking pieces shared by the traj_tracking scripts, as TrackingEnv defines them.

`Reference` moves along one lap at a constant linear speed, as TrackingEnv._compute_traj does
since #96: the look-ahead point k of step s lies `speed * (s + k * TRAJ_STEP_SIZE) * CONTROL_DT`
metres along the lap, resampled into equal-length segments from the shape's phase 0.
`build_observation` builds the 22-D frame of TrackingEnv._get_observations.

The deploy flies the reference without TrackingEnv's per-episode rotation and phase: the policy
only sees positions relative to the drone, in the body frame.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import torch

from .trajectories import TRAJECTORIES, build_lap, interp_waypoints
from .utils import ARENA_FENCE, Fence, hover_for, quat_apply, quat_inv

logger = logging.getLogger("CrazyflieRL")

# ── Reference (TrackingEnvCfg names in brackets) ───────────────────────────────
TRAJ_TYPE      = "lemniscate"     # [trajectory_type] default shape; --trajectory overrides
# Applied to every shape; --traj-scale overrides XY. A unit circle is as wide in y as in x,
# so it fits ARENA_FENCE only up to --traj-scale 0.8.
TRAJ_SCALE     = [1.5, 1.5, 1.0]  # [traj_scale_range, traj_z_scale_range] X, Y and Z scale, m
TRAJ_ORIGIN    = [0.0, 0.0, 1.0]  # world-frame centre, m; the policy sees only relative positions
TRAJ_C         = 0.0              # [traj_c_range] lemniscate z coupling: 0 keeps the lap flat
TRAJ_SPEED     = 1.0              # m/s, when env.yaml has no traj_speed_range; --traj-speed overrides
FUTURE_STEPS   = 4                # [future_traj_steps] look-ahead points in the observation
TRAJ_STEP_SIZE = 5.0              # [traj_step_size] policy steps between look-ahead points
CONTROL_DT     = 0.01             # policy step, s: decimation 5 x sim.dt 0.002
REF_FENCE_MARGIN = 0.2            # m: the whole lap must stay this far inside ARENA_FENCE

# ── Flight to the start ────────────────────────────────────────────────────────
TAKEOFF_DURATION_S  = 3.0
GOTO_SPEED          = 0.5         # m/s, mean speed of the HLC go_to the trajectory start
GOTO_MIN_DURATION_S = 2.0         # shortest go_to, even when already close
STABILIZE_PAUSE_S   = 1.0         # hover after the takeoff and after the go_to

# ── Policy input (must match TrackingEnv._get_observations) ────────────────────
OBS_STEP_DIM = 3 * FUTURE_STEPS + 10  # one frame: rpos_1..4 (12) | lin_vel_b | ang_vel_b | quat = 22
OBS_FIELDS = ([f"rpos{i}_b_{c}" for i in range(1, FUTURE_STEPS + 1) for c in "xyz"]
              + [f"lin_vel_b_{c}" for c in "xyz"] + [f"ang_vel_b_{c}" for c in "xyz"]
              + [f"quat_{c}" for c in "wxyz"])


class Reference:
    """The reference lap and where it is at each control step.

    Args:
        traj_type: a key of TRAJECTORIES.
        scale: (x, y, z) scale, m.
        speed: constant speed along the lap, m/s.
        c: lemniscate z coupling.
        origin: (x, y, z) centre, world frame, m.
        device: where the waypoints live (the controller's device).
    """

    def __init__(self, traj_type: str = TRAJ_TYPE, scale: Sequence[float] = TRAJ_SCALE,
                 speed: float = TRAJ_SPEED, c: float = TRAJ_C, origin: Sequence[float] = TRAJ_ORIGIN,
                 device: torch.device = torch.device("cpu")):
        if traj_type not in TRAJECTORIES:
            raise ValueError(f"unknown trajectory {traj_type!r}; known: {sorted(TRAJECTORIES)}")
        if speed <= 0 or any(s <= 0 for s in scale):
            raise ValueError(f"speed and scale must be positive, got speed {speed}, scale {list(scale)}")
        self.traj_type, self.scale, self.speed = traj_type, [float(s) for s in scale], float(speed)
        self.c, self.origin, self.device = float(c), [float(o) for o in origin], device
        waypoints_w, length = build_lap(traj_type, self.scale, self.origin, c=self.c)
        self.waypoints_w = waypoints_w.to(device)          # (k+1, 3) world frame, m
        self.length = length.to(device)                    # () lap length, m
        self._speed = torch.tensor(self.speed, device=device)
        self._offsets = TRAJ_STEP_SIZE * torch.arange(1, FUTURE_STEPS + 1, device=device).float()

    def at_distance(self, distance: torch.Tensor) -> torch.Tensor:
        """(S,) distances along the lap, m, lap after lap -> (S, 3) world frame, m."""
        return interp_waypoints(self.waypoints_w[None], self.length[None], distance[None])[0]

    def waypoints(self, step: int) -> torch.Tensor:
        """The FUTURE_STEPS look-ahead points of control step `step`, as TrackingEnv._compute_traj.

        Returns:
            (FUTURE_STEPS, 3) world frame, m.
        """
        time = (float(step) + self._offsets) * CONTROL_DT  # s, in the env's float32 order
        return self.at_distance(self._speed * time)

    def scored(self, step: int) -> torch.Tensor:
        """The point TrackingEnv scores step `step` against (_current_ref_w): look-ahead point 1. (3,), m."""
        return self.waypoints(step)[0]

    def distance(self, step: int) -> float:
        """Distance the scored point has covered at `step`, m (laps included)."""
        return self.speed * (step + TRAJ_STEP_SIZE) * CONTROL_DT

    def start(self) -> torch.Tensor:
        """Reference at distance 0, where TrackingEnv spawns the drone. (3,) world frame, m."""
        return self.at_distance(torch.zeros(1, device=self.device))[0]

    def limit_check(self, fence: Fence = ARENA_FENCE, margin: float = REF_FENCE_MARGIN) -> Optional[str]:
        """Pre-flight check: the whole lap stays `margin` inside `fence`.

        The flown reference interpolates linearly between the waypoints, so it stays inside any
        box that holds them all.

        Returns:
            None when the lap fits, otherwise a message giving its extent.
        """
        ref = self.waypoints_w.double().cpu()
        lo = torch.tensor(fence.lo, dtype=torch.float64) + margin - 1e-9
        hi = torch.tensor(fence.hi, dtype=torch.float64) - margin + 1e-9
        if ((ref >= lo) & (ref <= hi)).all():
            return None
        ref_lo = [round(v, 2) for v in ref.min(dim=0).values.tolist()]
        ref_hi = [round(v, 2) for v in ref.max(dim=0).values.tolist()]
        return (f"{self.traj_type} at scale {self.scale} spans {ref_lo}..{ref_hi} m, closer than "
                f"{margin} m to ARENA_FENCE {fence.lo}..{fence.hi}")

    def metadata(self) -> dict:
        """The reference that flew, for the recording's Parquet footer (str -> str)."""
        return {"trajectory": self.traj_type, "traj_scale_m": str(self.scale), "traj_speed_m_s": str(self.speed),
                "traj_origin_m": str(self.origin), "traj_c": str(self.c),
                "traj_lap_length_m": f"{self.length.item():.4f}", "traj_clock": "linear speed (#96)"}


def build_observation(pos_w: torch.Tensor, vel_w: torch.Tensor, ang_vel_b: torch.Tensor,
                      quat_wxyz: torch.Tensor, waypoints_w: torch.Tensor) -> torch.Tensor:
    """One 22-D frame (OBS_STEP_DIM) as TrackingEnv._get_observations builds it.

    Args:
        pos_w: (3,) world frame, m.
        vel_w: (3,) world frame, m/s.
        ang_vel_b: (3,) body frame, rad/s.
        quat_wxyz: (4,) body to world.
        waypoints_w: (FUTURE_STEPS, 3) world frame, m (Reference.waypoints).

    Returns:
        (22,): rpos_1..4 body frame (12) | lin_vel_b (3) | ang_vel_b (3) | quat (4).
    """
    q_inv = quat_inv(quat_wxyz)
    rpos_b = [quat_apply(q_inv, waypoints_w[i] - pos_w) for i in range(FUTURE_STEPS)]
    lin_vel_b = quat_apply(q_inv, vel_w)
    return torch.cat(rpos_b + [lin_vel_b, ang_vel_b, quat_wxyz], dim=-1)


def take_off(controller, height: float, stale_timeout_s: float) -> bool:
    """Enable the high-level commander, arm, take off to `height` m and hover. False if the flight must end.

    Args:
        controller: has cf, lock, landing, current_pos and _last_pos_time (as utils.hover_for).
        height: takeoff height, m.
        stale_timeout_s: land in place if the position goes stale this long.
    """
    # The firmware ignores takeoff/land unless commander.enHighLevel == 1 (default 0).
    controller.cf.param.set_value("commander.enHighLevel", "1")
    # Firmware >= 2022.12 keeps the motors locked until an arming request (cfclient sends one).
    controller.cf.platform.send_arming_request(True)
    logger.info(f"Takeoff to {height} m ...")
    controller.cf.high_level_commander.takeoff(height, TAKEOFF_DURATION_S)
    if not hover_for(controller, TAKEOFF_DURATION_S + STABILIZE_PAUSE_S, stale_timeout_s):
        return False
    logger.info(f"Takeoff complete. Current pos: {controller.current_pos}")
    return True


def go_to_trajectory_start(controller, start_w: torch.Tensor, stale_timeout_s: float) -> bool:
    """HLC go_to the reference's start and hover there. False if the flight must end.

    TrackingEnv spawns each episode within 0.3 m of the reference at distance 0 and ends it past
    reset_thres (0.8 m), so the policy starts there, with the trajectory clock at step 0.

    Args:
        controller: as take_off.
        start_w: (3,) world frame, m (Reference.start).
        stale_timeout_s: as take_off.
    """
    with controller.lock:
        dist = (start_w - controller.current_pos).norm().item()
    duration = max(GOTO_MIN_DURATION_S, dist / GOTO_SPEED)
    logger.info(f"Go to trajectory start {[round(v, 2) for v in start_w.tolist()]} ({dist:.2f} m, {duration:.1f} s) ...")
    controller.cf.high_level_commander.go_to(*start_w.tolist(), 0.0, duration)
    if not hover_for(controller, duration + STABILIZE_PAUSE_S, stale_timeout_s):
        return False
    logger.info(f"At trajectory start. Current pos: {controller.current_pos}")
    return True


# ── Command line ───────────────────────────────────────────────────────────────

def add_trajectory_args(parser) -> None:
    """Register the reference and takeoff flags shared by the traj_tracking scripts."""
    parser.add_argument("--trajectory", choices=sorted(TRAJECTORIES), default=None,
                        help=f"Reference shape (default: the run's trajectory_type, else {TRAJ_TYPE}).")
    parser.add_argument("--traj-scale", type=float, default=TRAJ_SCALE[0], metavar="XY",
                        help=f"XY scale of the shape in m, the circle's radius (default: {TRAJ_SCALE[0]}). "
                             f"The script refuses a lap that comes closer than {REF_FENCE_MARGIN} m to the "
                             f"arena fence: the circle needs 0.8.")
    parser.add_argument("--traj-speed", type=float, default=None, metavar="M_S",
                        help="Constant speed along the lap in m/s (default: the middle of the run's "
                             f"traj_speed_range, else {TRAJ_SPEED}). Runs trained before #96 read that "
                             "range in rad/s: they fly here at a linear speed they never saw.")
    parser.add_argument("--takeoff-height", type=float, default=None,
                        help=f"High-level-commander takeoff height in m, 0.2-2.0 (default: the trajectory's "
                             f"z, {TRAJ_ORIGIN[2]}). The go_to the start then reaches the trajectory.")


def _outside(value: float, bounds) -> bool:
    return isinstance(bounds, list) and len(bounds) == 2 and not bounds[0] <= value <= bounds[1]


def reference_from_args(args, env_cfg: dict, parser, device: torch.device) -> Reference:
    """The Reference the flags and the training run ask for; parser.error before the radio on a bad one.

    Warns, without refusing, when the shape, the speed or the scale is outside what the run
    trained on (env.yaml trajectory_type, traj_speed_range, traj_scale_range).
    """
    if args.takeoff_height is not None and not 0.2 <= args.takeoff_height <= 2.0:
        parser.error(f"--takeoff-height must be between 0.2 m and 2.0 m, got {args.takeoff_height}")
    if args.traj_scale <= 0:
        parser.error(f"--traj-scale must be positive, got {args.traj_scale}")
    if args.traj_speed is not None and args.traj_speed <= 0:
        parser.error(f"--traj-speed must be positive, got {args.traj_speed}")

    trained_type = env_cfg.get("trajectory_type")
    traj_type = args.trajectory or (trained_type if trained_type in TRAJECTORIES else TRAJ_TYPE)
    if trained_type and traj_type != trained_type:
        logger.warning(f"Flying a {traj_type}; the policy was trained on a {trained_type}")

    speed_range = env_cfg.get("traj_speed_range")
    if args.traj_speed is not None:
        speed = args.traj_speed
    elif isinstance(speed_range, list) and len(speed_range) == 2:
        speed = 0.5 * (speed_range[0] + speed_range[1])  # as TrackingEnv without randomize_trajectory
    else:
        speed = TRAJ_SPEED
    if _outside(speed, speed_range):
        logger.warning(f"--traj-speed {speed} m/s is outside the run's traj_speed_range {speed_range}")
    if _outside(args.traj_scale, env_cfg.get("traj_scale_range")):
        logger.warning(f"--traj-scale {args.traj_scale} m is outside the run's traj_scale_range "
                       f"{env_cfg['traj_scale_range']}")

    reference = Reference(traj_type, [args.traj_scale, args.traj_scale, TRAJ_SCALE[2]], speed, device=device)
    error = reference.limit_check()
    if error:
        parser.error(error)
    logger.info(f"Reference: {traj_type}, scale {reference.scale} m, {speed:g} m/s, "
                f"lap {reference.length.item():.2f} m ({reference.length.item() / speed:.1f} s)")
    return reference
