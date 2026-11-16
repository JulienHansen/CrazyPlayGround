import argparse
import logging
import math
import os
import re
import time
from typing import Optional, Sequence, Tuple

import torch


def hand_over_and_land(cf, duration: float = 2.0) -> None:
    """Hand control from the low-level setpoints to the high-level commander, and land.

    Low-level setpoints outrank the high-level commander, so land() alone is ignored. The
    notify expires the last low-level setpoint at once: until a high-level trajectory runs,
    the firmware has no setpoint and cuts the motors. So land() must follow it immediately.
    """
    cf.commander.send_notify_setpoint_stop()
    cf.high_level_commander.land(0.0, duration)


def land_once(controller, duration: float = 2.0) -> None:
    """hand_over_and_land, unless this flight is already landing.

    The control loop calls it the moment it exits, so the drone is never left without a
    setpoint; stop() calls it again for a loop that never ran. Never end a flight with
    send_stop_setpoint instead: it cuts the motors in mid-air, and outranks a land() in progress.
    """
    with controller.lock:
        if controller.landing:
            return
        controller.landing = True
    hand_over_and_land(controller.cf, duration)


def emergency_land(controller) -> None:
    logger = logging.getLogger("CrazyflieRL")
    logger.warning("EMERGENCY LANDING triggered")
    controller.running = False
    try:
        land_once(controller)
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


logger = logging.getLogger("CrazyflieRL")


def read_obs_hist(checkpoint_path: Optional[str]) -> Optional[Tuple[int, int]]:
    """(obs_input_hist, action_input_hist) from the training run's params/env.yaml, or None.

    train.py writes <run>/params/env.yaml next to <run>/checkpoints/. Its `observation_space`
    is stale (dumped before the env recomputes it), so read the two knobs instead, with a
    regex because yaml.safe_load rejects the file's !!python/tuple tags.
    """
    if not checkpoint_path:
        return None
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    path = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    k = re.search(r"^obs_input_hist:\s*(\d+)\s*$", text, re.MULTILINE)
    m = re.search(r"^action_input_hist:\s*(\d+)\s*$", text, re.MULTILINE)
    return (int(k.group(1)), int(m.group(1))) if k and m else None


def resolve_obs_hist(checkpoint_path: Optional[str], flag_k: Optional[int],
                     flag_m: Optional[int]) -> Tuple[int, int]:
    """Frames K and past actions M the policy was trained with.

    The checkpoint only fixes the input width (frame size x K + action size x M), and equal
    widths load silently (hover: K=1, M=2 and K=2, M=0 both give 12), so env.yaml wins and a
    flag that disagrees with it is an error. Without env.yaml (copied .pt, run before WP1)
    the flags decide, then the plain K=1, M=0.
    """
    trained = read_obs_hist(checkpoint_path)
    if trained is None:
        k, m = (1 if flag_k is None else flag_k), (0 if flag_m is None else flag_m)
        logger.info(f"No obs_input_hist/action_input_hist in the run's env.yaml: K={k}, M={m} from the flags")
        return k, m
    for flag, name, value in ((flag_k, "--obs-input-hist", trained[0]),
                              (flag_m, "--action-input-hist", trained[1])):
        if flag is not None and flag != value:
            raise ValueError(f"{name} {flag} disagrees with the training run's env.yaml ({value})")
    logger.info(f"K={trained[0]}, M={trained[1]} from the training run's env.yaml")
    return trained


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
