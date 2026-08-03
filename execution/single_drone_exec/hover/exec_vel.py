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
target_pos = None  # initialized in control_loop after takeoff
target_reached_since: Optional[float] = None  # timestamp when drone entered reach radius, or None

# ── Safety thresholds ────────────────────────────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without a position callback before emergency land
POS_VARIANCE_THRESHOLD = 0.5   # kalman position variance [m²] above which tracking is unreliable

# ── Waypoint switching ───────────────────────────────────────────────────────
WAYPOINT_REACH_RADIUS_M = 0.10  # must be within this distance of target...
WAYPOINT_HOLD_TIME_S    = 5.0   # ...continuously for this long before advancing to a new waypoint

# ============================================================
#                      MODEL DEFINITION
# ============================================================


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0,
                 initial_log_std=0.0):

        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)

        # === Must match checkpoint architecture ===
        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU()
        )

        # Policy head
        self.policy_layer = nn.Linear(32, self.num_actions)

        # Value head (SKRL requires it even for policy model)
        self.value_layer = nn.Linear(32, 1)

        # Shared log std
        self.log_std_parameter = nn.Parameter(
            torch.ones(self.num_actions) * initial_log_std
        )

    def compute(self, inputs, role):
        x = self.net_container(inputs["observations"])

        if role == "policy":
            mean = self.policy_layer(x)
        else:  # value function
            mean = self.value_layer(x)

        return mean, {"log_std": self.log_std_parameter}

# ============================================================
#                    CRAZYFLIE CONTROLLER
# ============================================================

class CrazyflieController:
    """Crazyflie controller for velocity-based RL agents.

    Uses send_velocity_world_setpoint to command velocity in world frame.
    Sim equivalent: vel_hovering.py (command_level="velocity", max_velocity=1.0 m/s).
    """
    def __init__(self, uri: str, agent: PPO, initial_target=None,
                 record_path: Optional[str] = None, record_interval_s: float = 0.1,
                 log_interval_s: float = 1.0):
        self.uri = uri
        self.cf = Crazyflie(rw_cache='./cache')
        self.agent = agent
        self.initial_target = initial_target
        self.current_pos = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_vel = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_quat = torch.zeros(4, dtype=torch.float32, device=device)
        self.running = True
        self.position_received = False
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
                action_fields=["action_vx", "action_vy", "action_vz"],
                cmd_fields=["vx", "vy", "vz"],
                record_interval_s=record_interval_s,
            )

        self.flight_logger = FlightLogger(logger, self._sample_for_status, log_interval_s=log_interval_s)

        self._setup_callbacks()

    # ---------- Crazyflie callbacks ----------

    def _setup_callbacks(self):
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

    def _connected(self, uri: str):
        logger.info(f"Connected to {uri}, taking off...")
        self.cf.high_level_commander.takeoff(0.5, 1)
        time.sleep(1.5)
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
                "cmd": self.last_cmd.tolist(),
                "motor": self.current_motor_pwm,
            }

    def _sample_for_status(self):
        with self.lock:
            if self.last_cmd is None:
                return None
            px, py, pz = self.current_pos.tolist()
            vx_m, vy_m, vz_m = self.current_vel.tolist()
            quat = self.current_quat.tolist()
            vx, vy, vz = self.last_cmd.tolist()
            motor = self.current_motor_pwm

        roll, pitch, yaw = quat_to_euler_deg(quat)
        state_line = (
            f"State: pos=({px:+.2f}, {py:+.2f}, {pz:+.2f}) m "
            f"vel=({vx_m:+.2f}, {vy_m:+.2f}, {vz_m:+.2f}) m/s "
            f"rpy=({roll:+.1f}, {pitch:+.1f}, {yaw:+.1f})°"
        )
        cmd_line = (
            f"Cmd: vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} m/s "
            f"| PWM: m1={motor[0]} m2={motor[1]} m3={motor[2]} m4={motor[3]}"
        )
        return state_line, cmd_line

    # ---------- Control loop ----------

    def control_loop(self):
        """Main control loop: send world-frame velocity commands.

        Must match vel_hovering.py: action in [-1, 1] scaled by max_velocity=1.0 m/s.
        """
        INTERVAL = 0.01  # 100 Hz — matches sim (dt=1/500, decimation=5)
        MAX_VEL = 1.0    # m/s — must match vel_hovering.py QuadcopterEnvCfg.max_velocity

        logger.info("Waiting for first position estimate...")
        while not self.position_received and self.running:
            time.sleep(0.05)
        if not self.running:
            return
        logger.info(f"Position received: {self.current_pos}")

        # Initialize target
        global target_pos
        if self.initial_target is not None:
            target_pos = torch.tensor(self.initial_target, dtype=torch.float32, device=device)
        else:
            target_pos = self.current_pos.clone()
            target_pos[2] = max(0.5, min(1.5, target_pos[2].item()))
        logger.info(f"Init target pos={target_pos}")

        nn_start_time = time.time()
        GRACE_PERIOD = 3.0  # seconds before enforcing z lower bound
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

            obs = retrieve_and_create_observation(self.current_vel, self.current_pos, self.current_quat)
            if obs is None:
                logger.warning("No observation received, hovering...")
                time.sleep(INTERVAL)
                continue

            with torch.no_grad():
                _, outputs = self.agent.act(obs.unsqueeze(0), None, timestep=0, timesteps=1)
                action = outputs["mean_actions"].squeeze(0)  # deterministic mean
                action = action.clamp(-1.0, 1.0)

            # Scale action to velocity in m/s (matching sim: actions * max_velocity)
            velocity_cmd = action * MAX_VEL

            with self.lock:
                self.last_obs = obs
                self.last_action = action
                self.last_cmd = velocity_cmd
                self.last_control_time = time.time()

            self.cf.commander.send_velocity_world_setpoint(
                velocity_cmd[0].item(),
                velocity_cmd[1].item(),
                velocity_cmd[2].item(),
                0.0  # yaw_rate = 0
            )

            elapsed = time.time() - start_time
            time.sleep(max(0, INTERVAL - elapsed))

        self.cf.commander.send_stop_setpoint()
        logger.info("Control loop stopped")

    # ---------- Connection management ----------

    def start(self):
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf.open_link(self.uri)

    def stop(self):
        logger.info("Stopping controller...")
        self.running = False
        time.sleep(0.6)
        logger.info("Landing...")
        self.cf.high_level_commander.land(0.0, 2.0)
        time.sleep(2.5)
        self.cf.close_link()
        if self.recorder:
            self.recorder.close()
        self.flight_logger.close()
        logger.info("Link closed")


# ============================================================
#                    OBSERVATION CREATION
# ============================================================

def retrieve_and_create_observation(current_vel, current_pos, current_quat) -> Optional[torch.Tensor]:
    """Build obs tensor matching vel_hovering.py: [lin_vel_b(3), desired_pos_b(3)]."""
    global target_pos, target_reached_since
    if target_pos is None:
        return None
    dist_to_target = torch.dist(current_pos, target_pos)
    if dist_to_target < WAYPOINT_REACH_RADIUS_M:
        if target_reached_since is None:
            target_reached_since = time.time()
        elif time.time() - target_reached_since >= WAYPOINT_HOLD_TIME_S:
            # New target matching training distribution: XY=[-1,1], Z=[0.5,1.5]
            target_pos = torch.empty(3, dtype=torch.float32, device=device)
            target_pos[:2].uniform_(-1.0, 1.0)
            target_pos[2].uniform_(0.5, 1.5)
            target_reached_since = None
            logger.info(f"/!\\ New target={target_pos}")
    else:
        target_reached_since = None

    # Rotate world-frame Kalman velocity into body frame
    linear_vel_b = quat_apply(quat_inv(current_quat), current_vel)
    desired_pos_b = quat_apply(quat_inv(current_quat), target_pos - current_pos)

    obs = torch.cat([linear_vel_b, desired_pos_b], dim=-1)
    return obs


# ============================================================
#                    MODEL LOADING / MAIN
# ============================================================

def load_agent(checkpoint_path: Optional[str], device: torch.device) -> PPO:
    obs_space = 6
    act_space = 3

    policy = Policy(observation_space=obs_space, action_space=act_space, device=device)
    models = {"policy": policy}

    cfg = PPO_CFG(
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": obs_space, "device": device},
    )
    agent = PPO(models=models, memory=None, cfg=cfg,
                observation_space=obs_space, action_space=act_space, device=device)

    assert checkpoint_path and os.path.exists(checkpoint_path), "No valid checkpoint provided. Please give a path for weights."

    agent.load(checkpoint_path)
    agent.enable_training_mode(False)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return agent


def main():
    parser = argparse.ArgumentParser(description="Run a trained velocity RL agent on a Crazyflie drone.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E8", help="URI of the Crazyflie")
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

    controller = CrazyflieController(uri=args.uri, agent=agent, initial_target=args.target,
                                      record_path=args.record_path, record_interval_s=args.record_interval,
                                      log_interval_s=args.log_interval)

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
