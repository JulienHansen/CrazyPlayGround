import argparse
import logging
import math
import os
import re
import time
from dataclasses import dataclass
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


def hand_over_and_go_to(cf, pos_w: Sequence[float], duration: float) -> None:
    """As hand_over_and_land, but the high-level commander flies to `pos_w` (world, m, yaw 0) and hovers."""
    cf.commander.send_notify_setpoint_stop()
    cf.high_level_commander.go_to(*[float(v) for v in pos_w], 0.0, duration)


def _before_hand_over(controller) -> None:
    """Run the controller's `before_hand_over` hook, if it has one.

    The attitude and rate modes use it to send one level, hover-thrust setpoint: if a
    high-level packet is lost, the firmware then holds that, not the policy's last command.
    """
    hook = getattr(controller, "before_hand_over", None)
    if hook is not None:
        hook()


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
    _before_hand_over(controller)
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


@dataclass(frozen=True)
class Fence:
    """A world-frame box the policy must keep the drone in, and how to fly back once it leaves.

    Attributes:
        lo, hi: corners (x, y, z), world frame, m.
        lookahead_s: also trip when pos + vel * lookahead_s leaves the box.
        safe_hover_pos: (x, y, z) to fly back to, world frame, m.
        brake_s: go_to the breach point (clamped into the box) over this time, to stop the drone.
        fly_back_speed: mean speed of the go_to safe_hover_pos, m/s.
        min_fly_back_s: shortest go_to safe_hover_pos, even when already close.
        hover_s: hover at safe_hover_pos before landing.
    """
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]
    lookahead_s: float = 0.1
    safe_hover_pos: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    brake_s: float = 1.0
    fly_back_speed: float = 0.5
    min_fly_back_s: float = 2.0
    hover_s: float = 2.0

    def breach(self, pos: torch.Tensor, vel_w: torch.Tensor) -> Optional[str]:
        """Why the drone must leave the policy, or None while it stays inside.

        Args:
            pos: (3,) world frame, m.
            vel_w: (3,) world frame, m/s.
        """
        lo, hi = (torch.tensor(c, dtype=pos.dtype, device=pos.device) for c in (self.lo, self.hi))
        for what, p in (("Position", pos), (f"Position in {self.lookahead_s:g} s", pos + vel_w * self.lookahead_s)):
            if (p < lo).any() or (p > hi).any():
                return f"{what} {[round(v, 2) for v in p.tolist()]} outside the fence {self.lo}..{self.hi}"
        return None

    def clamp(self, pos: torch.Tensor) -> torch.Tensor:
        """`pos` moved onto the box if outside it. (3,) world frame, m."""
        lo, hi = (torch.tensor(c, dtype=pos.dtype, device=pos.device) for c in (self.lo, self.hi))
        return torch.max(torch.min(pos, hi), lo)


# Set inside the arena with a margin: the drone keeps moving while it brakes.
# The tracking trajectory spans x ±1.5, y ±0.53, z 1.0 m.
ARENA_FENCE = Fence(lo=(-2.0, -1.0, 0.2), hi=(2.0, 1.0, 1.8))


def hover_for(controller, duration: float, stale_timeout_s: float) -> bool:
    """Wait while the high-level commander flies. False when the flight must end now.

    That is: stop() has landed, the link is gone, or the position went stale for longer than
    `stale_timeout_s` (then it lands in place). Polls every 0.1 s.
    """
    for _ in range(max(1, round(duration / 0.1))):
        if controller.landing or not controller.cf.is_connected():
            return False
        if time.time() - controller._last_pos_time > stale_timeout_s:
            logger.error("Position stale while hovering — landing in place")
            land_once(controller)
            return False
        time.sleep(0.1)
    return True


def fly_back_and_land(controller, fence: Fence, reason: str, pos: torch.Tensor, stale_timeout_s: float) -> None:
    """Leave the policy: the high-level commander brakes, flies to fence.safe_hover_pos, hovers, lands.

    Args:
        controller: has cf, lock, landing, running, current_pos, _last_pos_time (as land_once).
        fence: the fence the drone left.
        reason: logged, from fence.breach.
        pos: (3,) world frame, m, at the breach.
        stale_timeout_s: land in place if the position goes stale this long during the fly-back.
    """
    logger.error(f"{reason} — braking, then hover at {fence.safe_hover_pos} and land")
    controller.running = False
    try:
        brake = fence.clamp(pos)
        _before_hand_over(controller)
        hand_over_and_go_to(controller.cf, brake.tolist(), fence.brake_s)
        if not hover_for(controller, fence.brake_s, stale_timeout_s):
            return
        dist = math.dist(fence.safe_hover_pos, brake.tolist())
        duration = max(fence.min_fly_back_s, dist / fence.fly_back_speed)
        if controller.landing:
            return
        controller.cf.high_level_commander.go_to(*fence.safe_hover_pos, 0.0, duration)
        if not hover_for(controller, duration + fence.hover_s, stale_timeout_s):
            return
        logger.info(f"Hovering at {fence.safe_hover_pos}. Current pos: {controller.current_pos} — landing")
        land_once(controller)
    except Exception as e:
        logger.error(f"Fly-back failed: {e}")
        emergency_land(controller)


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


def env_yaml_path(checkpoint_path: Optional[str]) -> Optional[str]:
    """The training run's env.yaml: <name>_params/env.yaml beside an exported <name>.pt, else
    <run>/params/env.yaml. None when neither exists."""
    if not checkpoint_path:
        return None
    ckpt = os.path.abspath(checkpoint_path)
    candidates = (os.path.join(os.path.splitext(ckpt)[0] + "_params", "env.yaml"),
                  os.path.join(os.path.dirname(os.path.dirname(ckpt)), "params", "env.yaml"))
    return next((p for p in candidates if os.path.exists(p)), None)


def _yaml_scalar(text: str):
    """A plain YAML scalar as Python: bool, None, int, float, else the unquoted string."""
    text = text.strip()
    if text in ("true", "false"):
        return text == "true"
    if text in ("null", "~", ""):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text.strip("'\"")


def read_env_cfg(checkpoint_path: Optional[str]) -> dict:
    """Top-level settings of the training run's env.yaml (see env_yaml_path), or {} without one.

    yaml.safe_load rejects the file's !!python/tuple tags, so this reads it line by line:
    `key: scalar` at column 0, a list of `- scalar` lines under `key:` (tuples included), and
    the scalars one level down as "parent.child" (e.g. "sim.dt"). Deeper levels are skipped.

    Returns:
        name -> bool, None, int, float, str or list of those; "_path" holds the file read.
    """
    path = env_yaml_path(checkpoint_path)
    if path is None:
        return {}
    cfg, parent = {"_path": path}, None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("- "):              # list item of the last top-level key
                if parent is not None:
                    if not isinstance(cfg.get(parent), list):
                        cfg[parent] = []
                    cfg[parent].append(_yaml_scalar(line[2:]))
                continue
            top = re.match(r"^([A-Za-z_]\w*):\s*(.*)$", line)
            if top:
                parent, value = top.group(1), top.group(2)
                cfg[parent] = None if value in ("", "!!python/tuple") else _yaml_scalar(value)
                continue
            child = re.match(r"^  ([A-Za-z_]\w*):\s*(.*)$", line)
            if child and parent is not None and child.group(2) not in ("", "!!python/tuple"):
                cfg[f"{parent}.{child.group(1)}"] = _yaml_scalar(child.group(2))
    return cfg


def check_env_contract(env_cfg: dict, control_mode: str, policy_dt: float = 0.01) -> None:
    """Refuse a checkpoint whose training env the tracking scripts cannot reproduce.

    Only settings that env.yaml holds are checked, so an older run without one of them passes.

    Args:
        env_cfg: from read_env_cfg.
        control_mode: the TrackingEnvCfg.control_mode this script flies ("velocity", "body_rate").
        policy_dt: the script's control period, s.

    Raises:
        ValueError: naming the first setting that differs.
    """
    expected = {"control_mode": control_mode, "future_traj_steps": 4, "traj_step_size": 5.0}
    for name, value in expected.items():
        if env_cfg.get(name) is not None and env_cfg[name] != value:
            raise ValueError(f"the training run's {name} is {env_cfg[name]!r}; this script flies {value!r}")
    dt, decimation = env_cfg.get("sim.dt"), env_cfg.get("decimation")
    if dt is not None and decimation is not None and abs(dt * decimation - policy_dt) > 1e-9:
        raise ValueError(f"the training run's policy step is sim.dt {dt} x decimation {decimation} "
                         f"= {dt * decimation:g} s; this script steps every {policy_dt:g} s")


def read_obs_hist(checkpoint_path: Optional[str]) -> Optional[Tuple[int, int]]:
    """(obs_input_hist, action_input_hist) from the training run's env.yaml, or None.

    Reads <name>_params/env.yaml beside an exported <name>.pt, else <run>/params/env.yaml. Takes
    the two knobs, not the stale `observation_space`, by regex: yaml.safe_load rejects the
    file's !!python/tuple tags.
    """
    path = env_yaml_path(checkpoint_path)
    if path is None:
        return None
    with open(path) as f:
        text = f.read()
    k = re.search(r"^obs_input_hist:\s*(\d+)\s*$", text, re.MULTILINE)
    m = re.search(r"^action_input_hist:\s*(\d+)\s*$", text, re.MULTILINE)
    return (int(k.group(1)), int(m.group(1))) if k and m else None


def resolve_obs_hist(checkpoint_path: Optional[str], flag_k: Optional[int],
                     flag_m: Optional[int]) -> Tuple[int, int]:
    """Frames K and past actions M the policy was trained with.

    The checkpoint fixes only the input width, and equal widths load silently (hover: K=1, M=2
    and K=2, M=0 both give 12), so env.yaml wins and a flag that disagrees raises. Without
    env.yaml the flags decide, defaulting to K=1, M=0.
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
