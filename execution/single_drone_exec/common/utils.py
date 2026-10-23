import argparse
import logging
import math
import time
from typing import Sequence, Tuple

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


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def add_run_args(parser) -> None:
    """Register the recording and status-logging flags shared by the hover scripts."""
    parser.add_argument("--record-path", type=str, default=None,
                        help="Parquet file path to record flight data to. If not set, recording is disabled.")
    parser.add_argument("--record-every", type=positive_int, default=1,
                        help="Record one row every N control steps (1 = every step, 100 Hz).")
    parser.add_argument("--log-every", type=positive_int, default=100,
                        help="Print a status line every N control steps (100 = once per second at 100 Hz).")


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
