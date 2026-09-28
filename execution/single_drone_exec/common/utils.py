import argparse
import logging
import math
import time
from typing import Optional, Sequence, Tuple

import torch


def emergency_land(controller) -> None:
    logger = logging.getLogger("CrazyflieRL")
    logger.warning("EMERGENCY LANDING triggered")
    controller.running = False
    try:
        controller.cf.high_level_commander.land(0.0, 2.0)
    except Exception as e:
        logger.error(f"Emergency land command failed: {e}")
        try:
            controller.cf.commander.send_stop_setpoint()
        except Exception:
            pass


def wait_until(deadline: float, spin_margin: float) -> tuple[float, float, int]:
    """Hold until `deadline` (a perf_counter value). Returns (held_until, slept_ms, overran).

    time.sleep returns late -- by ~0.1 ms on Linux but 1-2 ms on macOS -- and the
    error lands on every iteration, so a 10 ms period becomes 12 ms and a 100 Hz
    loop runs at 85 Hz. Sleeping to `spin_margin` before the deadline and
    busy-waiting the rest removes the overshoot, at the cost of a few percent of
    one core.

    An overrun resynchronises the schedule to now instead of running flat out to
    catch up: a burst of back-to-back setpoints is worse for the drone than a late
    one.

    Usage: `deadline = time.perf_counter()` before the loop, then at the end of
    each iteration `deadline, _, _ = wait_until(deadline + INTERVAL, SPIN_MARGIN)`.
    """
    now = time.perf_counter()
    if now >= deadline:
        return now, 0.0, 1
    t0 = now
    if deadline - now > spin_margin:
        time.sleep(deadline - now - spin_margin)
    while time.perf_counter() < deadline:
        pass
    return deadline, (time.perf_counter() - t0) * 1e3, 0


# 0.10 m stalled half the baseline flights (mean pos err ~0.13 m); 0.15 m let every
# tested policy advance. The right radius depends on the policy, hence the CLI flags.
WAYPOINT_REACH_RADIUS_M = 0.15
WAYPOINT_HOLD_TIME_S = 5.0


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def nonnegative_float(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def add_run_args(parser) -> None:
    """Register the recording, status-logging and waypoint flags shared by the hover scripts."""
    parser.add_argument("--record-path", type=str, default=None,
                        help="Parquet file path to record flight data to. If not set, recording is disabled.")
    parser.add_argument("--record-every", type=positive_int, default=1,
                        help="Record one row every N control steps (1 = every step, 100 Hz).")
    parser.add_argument("--log-every", type=positive_int, default=100,
                        help="Print a status line every N control steps (100 = once per second at 100 Hz).")
    parser.add_argument("--waypoint-radius", type=nonnegative_float, default=WAYPOINT_REACH_RADIUS_M,
                        help="Distance in metres the drone must stay within to count as on the waypoint. "
                             "Depends on the policy's steady-state error. 0 keeps the target fixed.")
    parser.add_argument("--waypoint-hold", type=nonnegative_float, default=WAYPOINT_HOLD_TIME_S,
                        help="Seconds the drone must stay continuously within --waypoint-radius "
                             "before a new random waypoint is drawn.")


def apply_run_args(args) -> "WaypointGate":
    """Build the WaypointGate from the parsed flags and log its setting."""
    logging.getLogger("CrazyflieRL").info(
        f"Waypoint gate: {args.waypoint_radius:.2f} m for {args.waypoint_hold:.1f} s")
    return WaypointGate(args.waypoint_radius, args.waypoint_hold)


class WaypointGate:
    """Draw a new target (training range: XY in [-1, 1], Z in [0.5, 1.5]) once the
    drone has stayed within `radius_m` of the current one for `hold_s`."""

    def __init__(self, radius_m: float = WAYPOINT_REACH_RADIUS_M, hold_s: float = WAYPOINT_HOLD_TIME_S):
        self.radius_m, self.hold_s = radius_m, hold_s
        self._since: Optional[float] = None  # when the drone entered the radius

    def step(self, current_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        now = time.monotonic()
        if torch.dist(current_pos, target_pos) >= self.radius_m:
            self._since = None
        elif self._since is None:
            self._since = now
        elif now - self._since >= self.hold_s:
            self._since = None
            target_pos = torch.empty_like(target_pos)
            target_pos[:2].uniform_(-1.0, 1.0)
            target_pos[2].uniform_(0.5, 1.5)
            logging.getLogger("CrazyflieRL").info(f"/!\\ New target={target_pos}")
        return target_pos


def quat_apply(quat, vec):
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_conjugate(q):
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


def quat_inv(q, eps=1e-9):
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


def quat_to_euler_deg(quat: Sequence[float]) -> Tuple[float, float, float]:
    """Convert (qw, qx, qy, qz) to (roll, pitch, yaw) in degrees, for display only."""
    qw, qx, qy, qz = quat
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
