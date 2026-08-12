import os
import sys
import time
import threading
import argparse
import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from flight_recorder import FlightRecorder
from flight_logger import FlightLogger, quat_to_euler_deg
from utils import setup_state_logging, emergency_land, quat_apply, quat_inv

# skrl imports
from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(format="{asctime} [{levelname}] {message}",
                        style="{",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
logger = logging.getLogger("CrazyflieRL")
target_pos = None  # initialized in control_loop after takeoff

# ── Safety thresholds ────────────────────────────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without a position callback before emergency land
POS_VARIANCE_THRESHOLD = 0.5   # kalman position variance [m²] above which tracking is unreliable


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0,
                 initial_log_std=0.0):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 32), nn.ELU(),
            nn.Linear(32, 32), nn.ELU()
        )
        self.policy_layer = nn.Linear(32, self.num_actions)
        self.value_layer = nn.Linear(32, 1)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * initial_log_std)

    def compute(self, inputs, role):
        x = self.net_container(inputs["observations"])
        if role == "policy":
            mean = self.policy_layer(x)
        else:
            mean = self.value_layer(x)
        return mean, {"log_std": self.log_std_parameter}


class CrazyflieController:
    """Crazyflie controller for body-rate RL agents.

    Uses rate mode to send body rates in deg/s + thrust percentage.

    Thrust mapping is anchored on a *measured* real-drone hover thrust
    percentage (``hover_thrust_pct``) instead of the theoretical
    ``100 * weight_N / max_thrust_N``. This closes the most common
    sim2real altitude-drift gap caused by battery sag, propeller wear,
    and the non-linear PWM→thrust curve baked into ``send_setpoint_manual``
    (``PWM = 10001 + 0.01 * pct * 49999``).

    Pipeline:
        norm   = (a[3] + 1) / 2                 # [-1, 1] -> [0, 1]
        norm_hover = (1 - min_scale) / (max_scale - min_scale)
        if norm <= norm_hover:
            pct = lerp(min_pct, hover_pct, norm / norm_hover)
        else:
            pct = lerp(hover_pct, max_pct, (norm - norm_hover) / (1 - norm_hover))
    """
    def __init__(
        self,
        uri: str,
        agent: PPO,
        initial_target=None,
        mass_kg: float = 0.027,
        max_thrust_N: float = 0.638,
        min_thrust_scale: float = 0.5,
        max_thrust_scale: float = 1.8,
        hover_thrust_pct: Optional[float] = None,
        min_thrust_pct: float = 25.0,
        max_thrust_pct: float = 90.0,
        record_path: Optional[str] = None,
        record_interval_s: float = 0.1,
        log_interval_s: float = 1.0,
    ):
        self.uri = uri
        self.cf = Crazyflie(rw_cache='./cache')
        self.agent = agent
        self.initial_target = initial_target
        self.mass_kg = float(mass_kg)
        self.weight_N = self.mass_kg * 9.81
        self.max_thrust_N = float(max_thrust_N)
        self.min_thrust_scale = float(min_thrust_scale)
        self.max_thrust_scale = float(max_thrust_scale)
        if hover_thrust_pct is None:
            # Backward-compatible physics-based mapping (assumes linear PWM↔N
            # and ideal max thrust — typically too low for the real drone).
            self.hover_thrust_pct = 100.0 * self.weight_N / self.max_thrust_N
            self._calibrated = False
        else:
            self.hover_thrust_pct = float(hover_thrust_pct)
            self._calibrated = True
        self.min_thrust_pct = float(min_thrust_pct)
        self.max_thrust_pct = float(max_thrust_pct)
        # Hover sits at this point in the [0, 1] thrust_norm axis.
        self._hover_norm = (1.0 - self.min_thrust_scale) / (
            self.max_thrust_scale - self.min_thrust_scale
        )
        logger.info(
            f"Thrust mapping: m={self.mass_kg:.4f} kg, weight={self.weight_N:.4f} N, "
            f"max_thrust={self.max_thrust_N:.4f} N, "
            f"hover_pct={'(calibrated) ' if self._calibrated else '(physics) '}"
            f"{self.hover_thrust_pct:.1f}% "
            f"[min_pct={self.min_thrust_pct:.1f}%, max_pct={self.max_thrust_pct:.1f}%]"
        )
        if not self._calibrated:
            logger.warning(
                "Using uncalibrated thrust mapping. If the drone cannot hold altitude, "
                "measure the real hover thrust percentage (manually trim a stable hover "
                "in rate mode with zero rates) and pass it via --hover-thrust-pct."
            )
        self.current_pos = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_vel = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_quat = torch.zeros(4, dtype=torch.float32, device=device)
        self.current_ang_vel = torch.zeros(3, dtype=torch.float32, device=device)
        self.position_received = False
        self.running = True
        self._last_pos_time: float = 0.0
        self._pos_variance = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_motor_pwm = [0, 0, 0, 0]
        self.lock = threading.Lock()

        # Latest policy inference outputs, for the recorder thread
        self.last_obs: Optional[torch.Tensor] = None
        self.last_action: Optional[torch.Tensor] = None
        self.last_cmd: Optional[torch.Tensor] = None
        self.last_control_time: float = 0.0

        # File recording (separate cadence from CF telemetry logging)
        self.record_path = record_path
        self.recorder: Optional[FlightRecorder] = None
        if self.record_path:
            self.recorder = FlightRecorder(
                self.record_path,
                self._sample_for_recording,
                obs_fields=["lin_vel_b_x", "lin_vel_b_y", "lin_vel_b_z",
                            "des_pos_b_x", "des_pos_b_y", "des_pos_b_z",
                            "rotmat_0", "rotmat_1", "rotmat_2", "rotmat_3", "rotmat_4",
                            "rotmat_5", "rotmat_6", "rotmat_7", "rotmat_8",
                            "ang_vel_x", "ang_vel_y", "ang_vel_z"],
                action_fields=["action_roll", "action_pitch", "action_yaw", "action_thrust"],
                cmd_fields=["roll_rate_deg_s", "pitch_rate_deg_s", "yaw_rate_deg_s", "thrust_pct"],
                record_interval_s=record_interval_s,
            )

        self.flight_logger = FlightLogger(logger, self._sample_for_status, log_interval_s=log_interval_s)

        self._setup_callbacks()

    def _thrust_pct_from_action(self, action_thrust: float) -> float:
        """Piecewise-linear map from action[3] in [-1, 1] to thrust pct in [0, 100]."""
        norm = max(0.0, min(1.0, (action_thrust + 1.0) * 0.5))
        if norm <= self._hover_norm:
            t = norm / self._hover_norm if self._hover_norm > 0 else 0.0
            pct = self.min_thrust_pct + t * (self.hover_thrust_pct - self.min_thrust_pct)
        else:
            t = (norm - self._hover_norm) / (1.0 - self._hover_norm)
            pct = self.hover_thrust_pct + t * (self.max_thrust_pct - self.hover_thrust_pct)
        return max(0.0, min(100.0, pct))

    def _setup_callbacks(self):
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

    def _connected(self, uri: str):
        logger.info(f"Connected to {uri}")
        setup_state_logging(self)
        self._start_gyro_logging()
        threading.Thread(target=self.control_loop, daemon=True).start()
        if self.recorder:
            self.recorder.start()
        self.flight_logger.start()

    def _disconnected(self, uri: str):
        pass

    def _connection_failed(self, uri: str, msg: str):
        logger.error(f"Connection to {uri} failed: {msg}")
        self.running = False

    def _connection_lost(self, uri: str, msg: str):
        logger.warning(f"Connection to {uri} lost: {msg} — triggering safe landing")
        self.running = False

    def _start_gyro_logging(self):
        """Body angular velocity (gyro, deg/s -> rad/s in callback). Specific to
        this script's 18-dim observation — not part of the shared state-logging
        blocks in common/utils.py."""
        LOG_FREQUENCY_IN_MS = 10  # 100 Hz
        log_gyro = LogConfig(name="gyro", period_in_ms=LOG_FREQUENCY_IN_MS)
        log_gyro.add_variable("gyro.x", "float")
        log_gyro.add_variable("gyro.y", "float")
        log_gyro.add_variable("gyro.z", "float")
        self.cf.log.add_config(log_gyro)
        log_gyro.data_received_cb.add_callback(self._log_gyro_callback)
        log_gyro.start()

    def _log_gyro_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        # Crazyflie gyro is in deg/s, body frame. Sim's root_ang_vel_b is rad/s.
        deg2rad = math.pi / 180.0
        with self.lock:
            self.current_ang_vel = torch.tensor([
                data["gyro.x"] * deg2rad,
                data["gyro.y"] * deg2rad,
                data["gyro.z"] * deg2rad,
            ], dtype=torch.float32, device=device)

    # ---------- Flight recording ----------

    def _sample_for_recording(self):
        with self.lock:
            if self.last_obs is None:
                return None
            return {
                "control_time": self.last_control_time,
                "pos": self.current_pos.tolist(),
                "vel": self.current_vel.tolist(),
                "quat": self.current_quat.tolist(),
                "target": target_pos.tolist() if target_pos is not None else None,
                "obs": self.last_obs.tolist(),
                "action": self.last_action.tolist(),
                "cmd": self.last_cmd,
                "motor": self.current_motor_pwm,
            }

    def _sample_for_status(self):
        with self.lock:
            if self.last_cmd is None:
                return None
            px, py, pz = self.current_pos.tolist()
            vx_m, vy_m, vz_m = self.current_vel.tolist()
            quat = self.current_quat.tolist()
            roll_rate, pitch_rate, yaw_rate, thrust_pct = self.last_cmd
            motor = self.current_motor_pwm

        roll, pitch, yaw = quat_to_euler_deg(quat)
        state_line = (
            f"State: pos=({px:+.2f}, {py:+.2f}, {pz:+.2f}) m "
            f"vel=({vx_m:+.2f}, {vy_m:+.2f}, {vz_m:+.2f}) m/s "
            f"rpy=({roll:+.1f}, {pitch:+.1f}, {yaw:+.1f})°"
        )
        cmd_line = (
            f"Cmd: roll={roll_rate:+.1f} pitch={pitch_rate:+.1f} yaw={yaw_rate:+.1f} deg/s T={thrust_pct:5.1f}% "
            f"| PWM: m1={motor[0]} m2={motor[1]} m3={motor[2]} m4={motor[3]}"
        )
        return state_line, cmd_line

    def control_loop(self):
        """Main control loop: attitude-controlled takeoff, then NN body-rate control."""
        INTERVAL = 0.01  # 100 Hz control loop
        # Must match the sim: action[:3] * pi rad/s = action[:3] * 180 deg/s.
        MAX_BODY_RATE_DEG = 180.0

        TAKEOFF_HEIGHT   = 0.5
        TAKEOFF_DURATION = 2.5

        logger.info("Waiting for position data...")
        while not self.position_received and self.running:
            time.sleep(0.1)
        logger.info(f"Position received: {self.current_pos}")

        # ── Enable rate mode ─────────────────────────────────────────────────
        # stabMode=0 means send_setpoint interprets roll/pitch as body rates
        # in deg/s (not angles). This is required for body-rate control.
        self.cf.param.set_value("flightmode.stabModeRoll", "0")
        self.cf.param.set_value("flightmode.stabModePitch", "0")
        self.cf.param.set_value("flightmode.stabModeYaw", "0")
        logger.info("Rate mode enabled (stabMode=0 for all axes)")

        # ── Phase 1: thrust-ramp takeoff ─────────────────────────────────────
        logger.info(f"Thrust-ramp takeoff to ~{TAKEOFF_HEIGHT} m ...")
        RAMP_STEPS = int(TAKEOFF_DURATION / INTERVAL)
        for step in range(RAMP_STEPS):
            # Zero rates + ramp thrust from 30% of hover to hover, in rate mode
            frac = min(1.0, step / (RAMP_STEPS * 0.4))  # ramp over first 40%
            thrust_pct = self.hover_thrust_pct * (0.3 + 0.7 * frac)
            self.cf.commander.send_setpoint_manual(0, 0, 0, thrust_pct, True)
            time.sleep(INTERVAL)
        logger.info(f"Takeoff complete. Current pos: {self.current_pos}")

        # Initialize target
        global target_pos
        if self.initial_target is not None:
            target_pos = torch.tensor(self.initial_target, dtype=torch.float32, device=device)
        else:
            target_pos = self.current_pos.clone()
            target_pos[2] = max(0.5, min(1.5, target_pos[2].item()))
        logger.info(f"Init target pos={target_pos}")

        # ── Phase 2: NN body-rate control loop ───────────────────────────────
        logger.info("NN body-rate control active.")
        nn_start_time = time.time()
        GRACE_PERIOD = 3.0
        while self.cf.is_connected() and self.running:
            start_time = time.time()

            # ── Safety watchdog ──────────────────────────────────────────────
            elapsed_since_nn = time.time() - nn_start_time
            z = self.current_pos[2].item()
            if (elapsed_since_nn > GRACE_PERIOD and z < 0.1) or z > 2.5:
                logger.error(f"Position out of bounds z={z:.2f} — emergency landing")
                emergency_land(self)
                break
            if self._last_pos_time > 0 and time.time() - self._last_pos_time > POS_STALE_TIMEOUT_S:
                logger.error(
                    f"Position data stale ({time.time() - self._last_pos_time:.2f} s) — emergency landing"
                )
                emergency_land(self)
                break
            with self.lock:
                var = self._pos_variance.clone()
            if var.max().item() > POS_VARIANCE_THRESHOLD:
                logger.error(f"Position variance too high {var.tolist()} — emergency landing")
                emergency_land(self)
                break

            obs = retrieve_and_create_observation(
                self.current_vel, self.current_pos, self.current_quat, self.current_ang_vel
            )
            if obs is None:
                logger.warning("No observation received, holding hover thrust...")
                self.cf.commander.send_setpoint_manual(0, 0, 0, self.hover_thrust_pct, True)
                time.sleep(INTERVAL)
                continue

            with torch.no_grad():
                actions, info = self.agent.act(obs, None, timestep=0, timesteps=1)
                action = info.get("mean_actions", actions).squeeze(0)
                action = action.clamp(-1.0, 1.0)

            # Body rates: [-1, 1] -> [-MAX_BODY_RATE_DEG, MAX_BODY_RATE_DEG] deg/s.
            roll_rate  = action[0].item() * MAX_BODY_RATE_DEG
            pitch_rate = action[1].item() * MAX_BODY_RATE_DEG
            yaw_rate   = action[2].item() * MAX_BODY_RATE_DEG

            # Thrust: piecewise-linear in action space, anchored on hover_thrust_pct
            # so the policy's hover output (a[3] ≈ -0.23) maps to a real-drone
            # thrust percentage that actually holds altitude.
            thrust_pct = self._thrust_pct_from_action(action[3].item())

            with self.lock:
                self.last_obs = obs
                self.last_action = action
                self.last_cmd = [roll_rate, pitch_rate, yaw_rate, thrust_pct]
                self.last_control_time = time.time()

            # Rate mode (rate=True): roll/pitch/yaw in deg/s, thrust in [0, 100] %.
            self.cf.commander.send_setpoint_manual(roll_rate, pitch_rate, yaw_rate, thrust_pct, True)

            elapsed = time.time() - start_time
            time.sleep(max(0, INTERVAL - elapsed))

        self.cf.commander.send_stop_setpoint()
        logger.info("Control loop stopped")

    def start(self):
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf.open_link(self.uri)

    def stop(self):
        logger.info("Stopping controller...")
        self.running = False
        time.sleep(0.2)
        logger.info("Landing...")
        self.cf.high_level_commander.land(0.0, 2.0)
        time.sleep(2.5)
        self.cf.close_link()
        if self.recorder:
            self.recorder.close()
        self.flight_logger.close()
        logger.info("Link closed")


def retrieve_and_create_observation(
    current_vel, current_pos, current_quat, current_ang_vel
) -> Optional[torch.Tensor]:
    # Target is fixed per run, matching the sim episode behaviour.
    global target_pos
    if target_pos is None:
        return None
    linear_vel_b = quat_apply(quat_inv(current_quat), current_vel)
    desired_pos_b = quat_apply(quat_inv(current_quat), target_pos - current_pos)
    rot_mat_flat = quat_to_rotmat_flat(current_quat)
    obs = torch.cat([linear_vel_b, desired_pos_b, rot_mat_flat, current_ang_vel], dim=-1)
    return obs


def quat_to_rotmat_flat(quat: torch.Tensor) -> torch.Tensor:
    """Row-major flattened body→world rotation matrix.

    Matches isaaclab.utils.math.matrix_from_quat(quat).reshape(-1, 9) so the
    deployed observation lines up with the one produced in sim.
    quat is (qw, qx, qy, qz).
    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    two_s = 2.0 / (qw * qw + qx * qx + qy * qy + qz * qz)
    return torch.stack([
        1 - two_s * (qy * qy + qz * qz),
        two_s * (qx * qy - qz * qw),
        two_s * (qx * qz + qy * qw),
        two_s * (qx * qy + qz * qw),
        1 - two_s * (qx * qx + qz * qz),
        two_s * (qy * qz - qx * qw),
        two_s * (qx * qz - qy * qw),
        two_s * (qy * qz + qx * qw),
        1 - two_s * (qx * qx + qy * qy),
    ])


def load_agent(checkpoint_path: Optional[str], device: torch.device) -> PPO:
    obs_space = 18
    act_space = 4
    policy = Policy(observation_space=obs_space, action_space=act_space, device=device)
    models = {"policy": policy}
    cfg = PPO_CFG(
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": obs_space, "device": device},
    )
    agent = PPO(models=models, memory=None, cfg=cfg,
                observation_space=obs_space, action_space=act_space, device=device)
    assert checkpoint_path and os.path.exists(checkpoint_path), "No valid checkpoint provided."
    agent.load(checkpoint_path)
    agent.enable_training_mode(False)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Run a trained body-rate RL agent on a Crazyflie drone.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E8")
    parser.add_argument("--mass-kg", type=float, default=0.027,
                        help="Drone mass in kg (must match drone.mass in crazyflie.yaml).")
    parser.add_argument("--max-thrust-N", type=float, default=0.638,
                        help="Total static thrust of all 4 motors in N (drone.max_thrust).")
    parser.add_argument("--min-thrust-scale", type=float, default=0.5,
                        help="Lower bound of thrust action, as fraction of hover weight.")
    parser.add_argument("--max-thrust-scale", type=float, default=1.8,
                        help="Upper bound of thrust action, as fraction of hover weight.")
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        help="Initial target [x, y, z] in world frame. If not set, hovers above takeoff pos.")
    parser.add_argument("--hover-thrust-pct", type=float, default=None,
                        help="Measured real-drone hover thrust percentage (0-100). "
                             "Find by trimming a stable rate-mode hover with zero rates. "
                             "If unset, falls back to the (often inaccurate) physics formula.")
    parser.add_argument("--min-thrust-pct", type=float, default=25.0,
                        help="Thrust pct corresponding to action[3] = -1.")
    parser.add_argument("--max-thrust-pct", type=float, default=90.0,
                        help="Thrust pct corresponding to action[3] = +1.")
    parser.add_argument("--record-path", type=str, default=None,
                        help="Parquet file path to record flight data to. If not set, recording is disabled.")
    parser.add_argument("--record-interval", type=float, default=0.1,
                        help="Interval in seconds between recorded rows (independent of the 100Hz control loop).")
    parser.add_argument("--log-interval", type=float, default=1.0,
                        help="Interval in seconds between terminal status lines (independent of the 100Hz control loop).")
    args = parser.parse_args()
    agent = load_agent(args.checkpoint, device)
    controller = CrazyflieController(
        uri=args.uri,
        agent=agent,
        initial_target=args.target,
        mass_kg=args.mass_kg,
        max_thrust_N=args.max_thrust_N,
        min_thrust_scale=args.min_thrust_scale,
        max_thrust_scale=args.max_thrust_scale,
        hover_thrust_pct=args.hover_thrust_pct,
        min_thrust_pct=args.min_thrust_pct,
        max_thrust_pct=args.max_thrust_pct,
        record_path=args.record_path,
        record_interval_s=args.record_interval,
        log_interval_s=args.log_interval,
    )
    try:
        controller.start()
        timeout = 10
        elapsed = 0
        while not controller.cf.is_connected() and controller.running:
            time.sleep(1)
            elapsed += 1
            if elapsed >= timeout:
                logger.error(f"Connection timeout after {timeout}s — make sure cfclient is closed and drone is on")
                return
        if not controller.running:
            logger.error("Connection failed — check radio URI and that cfclient is closed")
            return
        logger.info("Cf is connected !")
        while controller.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        controller.stop()
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
