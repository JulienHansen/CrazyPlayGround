import os
import sys
import time
import threading
import argparse
import logging
from typing import Optional

import torch
import torch.nn as nn

import cflib.crtp
from cflib.crazyflie import Crazyflie

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
# Target is initialized after takeoff. In training, targets are absolute
# positions in XY=[-1,1] Z=[0.5,1.5] relative to env origin (= world origin).
# Your room is 3x3x1.7m with origin centered, so walls at ±1.5m — this
# leaves 0.5m margin on each side.
target_pos = None  # initialized in control_loop after takeoff

# ── Safety thresholds ────────────────────────────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without a position callback before emergency land
POS_VARIANCE_THRESHOLD = 0.5   # kalman position variance [m²] above which tracking is unreliable

# [Policy class identical to exec_vel.py — same architecture with act_space=4]

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
    """Crazyflie controller for attitude (angle) RL agents.

    Uses send_setpoint_manual with rate=False so roll/pitch are angles in
    degrees; yaw is always a yaw rate in deg/s regardless of the flag.
    Thrust mapping mirrors ``att_hovering.py`` exactly:
        norm     = (a[3] + 1) / 2
        thrust_N = m * g * (min_scale + (max_scale - min_scale) * norm)
        thrust_% = 100 * thrust_N / max_thrust_N
    Hover lands at ``a[3] ≈ -0.23`` (with 0.5/1.8 scaling), not a=0.
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
        self.hover_thrust_pct = 100.0 * self.weight_N / self.max_thrust_N
        logger.info(
            f"Thrust mapping: m={self.mass_kg:.4f} kg, weight={self.weight_N:.4f} N, "
            f"max_thrust={self.max_thrust_N:.4f} N, hover_pct≈{self.hover_thrust_pct:.1f}%"
        )
        self.current_pos = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_vel = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_quat = torch.zeros(4, dtype=torch.float32, device=device)
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
                            "des_pos_b_x", "des_pos_b_y", "des_pos_b_z"],
                action_fields=["action_roll", "action_pitch", "action_yaw", "action_thrust"],
                cmd_fields=["roll_deg", "pitch_deg", "yaw_rate_deg_s", "thrust_pct"],
                record_interval_s=record_interval_s,
            )

        self.flight_logger = FlightLogger(logger, self._sample_for_status, log_interval_s=log_interval_s)

        self._setup_callbacks()

    def _setup_callbacks(self):
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

    def _connected(self, uri: str):
        logger.info(f"Connected to {uri}")
        setup_state_logging(self)
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
            roll, pitch, yaw, thrust_pct = self.last_cmd
            motor = self.current_motor_pwm

        roll_meas, pitch_meas, yaw_meas = quat_to_euler_deg(quat)
        state_line = (
            f"State: pos=({px:+.2f}, {py:+.2f}, {pz:+.2f}) m "
            f"vel=({vx_m:+.2f}, {vy_m:+.2f}, {vz_m:+.2f}) m/s "
            f"rpy=({roll_meas:+.1f}, {pitch_meas:+.1f}, {yaw_meas:+.1f})°"
        )
        cmd_line = (
            f"Cmd: roll={roll:+.1f}° pitch={pitch:+.1f}° yaw={yaw:+.1f}°/s T={thrust_pct:5.1f}% "
            f"| PWM: m1={motor[0]} m2={motor[1]} m3={motor[2]} m4={motor[3]}"
        )
        return state_line, cmd_line

    def control_loop(self):
        """Main control loop: thrust-ramp takeoff, then NN attitude control."""
        INTERVAL = 0.01  # 100 Hz control loop
        MAX_ANGLE = 30.0          # degrees — must match att_hovering.py max_roll_pitch
        MAX_YAW_RATE = 90.0       # deg/s   — must match att_hovering.py max_yaw_rate

        TAKEOFF_HEIGHT   = 0.5
        TAKEOFF_DURATION = 2.5

        logger.info("Waiting for position data...")
        while not self.position_received and self.running:
            time.sleep(0.1)
        logger.info(f"Position received: {self.current_pos}")

        # ── Phase 1: attitude-controlled takeoff (angle mode) ────────────────
        # Ramp the thrust percentage from 30% of hover to hover, zero tilt.
        logger.info(f"Attitude takeoff to ~{TAKEOFF_HEIGHT} m ...")
        RAMP_STEPS = int(TAKEOFF_DURATION / INTERVAL)
        for step in range(RAMP_STEPS):
            frac = min(1.0, step / (RAMP_STEPS * 0.4))  # ramp over first 40%
            thrust_pct = self.hover_thrust_pct * (0.3 + 0.7 * frac)
            self.cf.commander.send_setpoint_manual(0, 0, 0, thrust_pct, False)
            time.sleep(INTERVAL)
        logger.info(f"Takeoff complete. Current pos: {self.current_pos}")
        # Initialize target: use CLI --target if given, else hover above takeoff pos
        global target_pos
        if self.initial_target is not None:
            target_pos = torch.tensor(self.initial_target, dtype=torch.float32, device=device)
        else:
            target_pos = self.current_pos.clone()
            target_pos[2] = max(0.5, min(1.5, target_pos[2].item()))
        logger.info(f"Init target pos={target_pos}")

        # ── Phase 2: NN attitude control loop ────────────────────────────────
        logger.info("NN attitude control active.")
        nn_start_time = time.time()
        GRACE_PERIOD = 3.0  # seconds before enforcing z lower bound (let drone gain altitude)
        while self.cf.is_connected() and self.running:
            start_time = time.time()

            # ── Safety watchdog ──────────────────────────────────────────────
            # Position bounds matching training termination (z<0.1 or z>2.0)
            elapsed_since_nn = time.time() - nn_start_time
            z = self.current_pos[2].item()
            if (elapsed_since_nn > GRACE_PERIOD and z < 0.1) or z > 2.5:
                logger.error(f"Position out of bounds z={self.current_pos[2].item():.2f} — emergency landing")
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

            obs = retrieve_and_create_observation(self.current_vel, self.current_pos, self.current_quat)
            if obs is None:
                logger.warning("No observation received, holding hover thrust...")
                self.cf.commander.send_setpoint_manual(0, 0, 0, self.hover_thrust_pct, False)
                time.sleep(INTERVAL)
                continue

            with torch.no_grad():
                _, outputs = self.agent.act(obs.unsqueeze(0), None, timestep=0, timesteps=1)
                action = outputs["mean_actions"].squeeze(0)  # deterministic mean, not sampled
                action = action.clamp(-1.0, 1.0)

            roll  = action[0].item() * MAX_ANGLE
            pitch = action[1].item() * MAX_ANGLE
            yaw   = action[2].item() * MAX_YAW_RATE
            # Thrust: mirror of att_hovering.py (Newtons → % of max static thrust).
            thrust_norm = (action[3].item() + 1.0) * 0.5
            thrust_norm = max(0.0, min(1.0, thrust_norm))
            thrust_N = self.weight_N * (
                self.min_thrust_scale + thrust_norm * (self.max_thrust_scale - self.min_thrust_scale)
            )
            thrust_pct = max(0.0, min(100.0, 100.0 * thrust_N / self.max_thrust_N))

            with self.lock:
                self.last_obs = obs
                self.last_action = action
                self.last_cmd = [roll, pitch, yaw, thrust_pct]
                self.last_control_time = time.time()

            # Angle mode (rate=False): roll/pitch in deg, yaw in deg/s, thrust in [0, 100] %.
            self.cf.commander.send_setpoint_manual(roll, pitch, yaw, thrust_pct, False)

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


def retrieve_and_create_observation(current_vel, current_pos, current_quat) -> Optional[torch.Tensor]:
    # Target is fixed per run, matching the sim episode behaviour.
    global target_pos
    if target_pos is None:
        return None
    # Rotate world-frame Kalman velocity into body frame
    linear_vel_b = quat_apply(quat_inv(current_quat), current_vel)
    desired_pos_b = quat_apply(quat_inv(current_quat), target_pos - current_pos)
    obs = torch.cat([linear_vel_b, desired_pos_b], dim=-1)
    return obs

def load_agent(checkpoint_path: Optional[str], device: torch.device) -> PPO:
    obs_space = 6
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
    parser = argparse.ArgumentParser(description="Run a trained SKRL PPO agent on a Crazyflie drone.")
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
        record_path=args.record_path,
        record_interval_s=args.record_interval,
        log_interval_s=args.log_interval,
    )
    try:
        controller.start()
        timeout = 10  # seconds to wait for connection
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
