import os
import time
import threading
import argparse
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

# skrl imports
from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
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
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 32), nn.ELU(),
            nn.Linear(32, 32), nn.ELU()
        )
        self.policy_layer = nn.Linear(32, self.num_actions)
        self.value_layer = nn.Linear(32, 1)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * initial_log_std)

    def compute(self, inputs, role):
        x = self.net_container(inputs["states"])
        if role == "policy":
            mean = self.policy_layer(x)
        else:
            mean = self.value_layer(x)
        return mean, self.log_std_parameter, {}

class CrazyflieController:
    """Main Crazyflie controller for running trained RL agents"""
    def __init__(self, uri: str, agent: PPO, hover_thrust: int = 30000, initial_target=None):
        self.uri = uri
        self.cf = Crazyflie()
        self.agent = agent
        self.hover_thrust = hover_thrust
        self.initial_target = initial_target
        self.current_pos = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_vel = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_quat = torch.zeros(4, dtype=torch.float32, device=device)
        self.position_received = False
        self.running = True
        self._last_pos_time: float = 0.0
        self._pos_variance = torch.zeros(3, dtype=torch.float32, device=device)
        self.lock = threading.Lock()
        self._setup_callbacks()

    def _setup_callbacks(self):
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

    def _connected(self, uri: str):
        logger.info(f"Connected to {uri}")
        self._start_logging()
        threading.Thread(target=self.control_loop, daemon=True).start()

    def _disconnected(self, uri: str):
        pass

    def _connection_failed(self, uri: str, msg: str):
        pass

    def _connection_lost(self, uri: str, msg: str):
        logger.warning(f"Connection to {uri} lost: {msg} — triggering safe landing")
        self.running = False

    def _start_logging(self):
        LOG_FREQUENCY_IN_MS = 10  # 100 Hz — matches control loop
        # Block 1: position + velocity (6 floats = 24 bytes, within 26-byte limit)
        log_posvel = LogConfig(name="posvel", period_in_ms=LOG_FREQUENCY_IN_MS)
        log_posvel.add_variable("stateEstimate.x", "float")
        log_posvel.add_variable("stateEstimate.y", "float")
        log_posvel.add_variable("stateEstimate.z", "float")
        log_posvel.add_variable("stateEstimate.vx", "float")
        log_posvel.add_variable("stateEstimate.vy", "float")
        log_posvel.add_variable("stateEstimate.vz", "float")
        self.cf.log.add_config(log_posvel)
        log_posvel.data_received_cb.add_callback(self._log_posvel_callback)
        log_posvel.start()

        # Block 2: quaternion (4 floats = 16 bytes)
        log_quat = LogConfig(name="quat", period_in_ms=LOG_FREQUENCY_IN_MS)
        log_quat.add_variable("stateEstimate.qx", "float")
        log_quat.add_variable("stateEstimate.qy", "float")
        log_quat.add_variable("stateEstimate.qz", "float")
        log_quat.add_variable("stateEstimate.qw", "float")
        self.cf.log.add_config(log_quat)
        log_quat.data_received_cb.add_callback(self._log_data_quat_callback)
        log_quat.start()

        log_var = LogConfig(name="quality", period_in_ms=200)
        log_var.add_variable("kalman.varPX", "float")
        log_var.add_variable("kalman.varPY", "float")
        log_var.add_variable("kalman.varPZ", "float")
        self.cf.log.add_config(log_var)
        log_var.data_received_cb.add_callback(self._log_variance_callback)
        log_var.start()

    def _log_posvel_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        with self.lock:
            self.current_pos = torch.tensor([
                data["stateEstimate.x"],
                data["stateEstimate.y"],
                data["stateEstimate.z"]
            ], dtype=torch.float32, device=device)
            self.current_vel = torch.tensor([
                data["stateEstimate.vx"],
                data["stateEstimate.vy"],
                data["stateEstimate.vz"]
            ], dtype=torch.float32, device=device)
            self._last_pos_time = time.time()
            self.position_received = True

    def _log_data_quat_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        with self.lock:
            self.current_quat = torch.tensor([
                data['stateEstimate.qw'],
                data['stateEstimate.qx'],
                data['stateEstimate.qy'],
                data['stateEstimate.qz'],
            ], dtype=torch.float32, device=device)

    def _log_variance_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        """Track Kalman filter position variance to detect lighthouse tracking loss."""
        with self.lock:
            self._pos_variance = torch.tensor([
                data["kalman.varPX"],
                data["kalman.varPY"],
                data["kalman.varPZ"],
            ], dtype=torch.float32, device=device)

    def _emergency_land(self):
        """Trigger a safe landing regardless of the current commander mode."""
        logger.warning("EMERGENCY LANDING triggered")
        self.running = False
        try:
            self.cf.high_level_commander.land(0.0, 2.0)
        except Exception as e:
            logger.error(f"Emergency land command failed: {e}")
            try:
                self.cf.commander.send_stop_setpoint()
            except Exception:
                pass

    def control_loop(self):
        """Main control loop: position-controlled takeoff, then NN attitude control"""
        INTERVAL = 0.01  # control frequency (s) - 100 Hz
        MAX_ANGLE = 30.0          # degrees  — must match att_hovering.py max_roll_pitch
        MAX_YAW_RATE = 90.0       # deg/s    — must match att_hovering.py max_yaw_rate
        MIN_THRUST_SCALE = 0.5    # fraction of hover — must match att_hovering.py
        MAX_THRUST_SCALE = 1.8    # fraction of hover — must match att_hovering.py

        TAKEOFF_HEIGHT   = 0.5   # metres — hover height before NN takes over
        TAKEOFF_DURATION = 2.5   # seconds for the HLC to reach the height
        STABILIZE_PAUSE  = 1.5   # extra seconds to let oscillations settle

        logger.info("Waiting for position data...")
        while not self.position_received and self.running:
            time.sleep(0.1)
        logger.info(f"Position received: {self.current_pos}")

        # ── Phase 1: attitude-controlled takeoff (no HLC) ─────────────────────
        # Use send_setpoint directly from the start to avoid the HLC→low-level
        # priority transition problem. Ramp thrust from zero to hover smoothly.
        logger.info(f"Attitude takeoff to ~{TAKEOFF_HEIGHT} m ...")
        RAMP_STEPS = int(TAKEOFF_DURATION / INTERVAL)
        for step in range(RAMP_STEPS):
            frac = min(1.0, step / (RAMP_STEPS * 0.4))  # ramp over first 40%
            thrust = int(self.hover_thrust * (0.3 + 0.7 * frac))  # 30% → 100%
            self.cf.commander.send_setpoint(0, 0, 0, thrust)
            time.sleep(INTERVAL)
        # Stabilise at hover thrust
        logger.info("Stabilising at hover...")
        for _ in range(int(STABILIZE_PAUSE / INTERVAL)):
            self.cf.commander.send_setpoint(0, 0, 0, self.hover_thrust)
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
        while self.cf.is_connected() and self.running:
            start_time = time.time()

            # ── Safety watchdog ──────────────────────────────────────────────
            # Position bounds matching training termination (z<0.1 or z>2.0)
            if self.current_pos[2].item() < 0.1 or self.current_pos[2].item() > 2.5:
                logger.error(f"Position out of bounds z={self.current_pos[2].item():.2f} — emergency landing")
                self._emergency_land()
                break
            if self._last_pos_time > 0 and time.time() - self._last_pos_time > POS_STALE_TIMEOUT_S:
                logger.error(
                    f"Position data stale ({time.time() - self._last_pos_time:.2f} s) — emergency landing"
                )
                self._emergency_land()
                break
            with self.lock:
                var = self._pos_variance.clone()
            if var.max().item() > POS_VARIANCE_THRESHOLD:
                logger.error(f"Position variance too high {var.tolist()} — emergency landing")
                self._emergency_land()
                break

            obs = retrieve_and_create_observation(self.current_vel, self.current_pos, self.current_quat)
            if obs is None:
                logger.warning("No observation received, hovering...")
                self.cf.commander.send_setpoint(0, 0, 0, self.hover_thrust)
                time.sleep(INTERVAL)
                continue

            with torch.no_grad():
                action_dict = self.agent.act(obs, 1, 0)
                action = action_dict[0]
                logger.debug(f"Action={action}")

            roll  = action[0].item() * MAX_ANGLE
            pitch = action[1].item() * MAX_ANGLE
            yaw   = action[2].item() * MAX_YAW_RATE
            # Thrust: action[3] in [0,1] — matches att_hovering.py and teleop_env.py convention.
            # 0 = min thrust, 1 = max thrust, ~0.556 = hover.
            thrust_norm = float(max(0.0, min(1.0, action[3].item())))
            thrust = int(self.hover_thrust * (MIN_THRUST_SCALE + thrust_norm * (MAX_THRUST_SCALE - MIN_THRUST_SCALE)))
            thrust = max(10000, min(60000, thrust))

            logger.info(f"Cmd: roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} thrust={thrust} | pos={self.current_pos}")
            self.cf.commander.send_setpoint(roll, pitch, yaw, thrust)

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
        logger.info("Link closed")


def retrieve_and_create_observation(current_vel, current_pos, current_quat) -> Optional[torch.Tensor]:
    global target_pos
    if target_pos is None:
        return None
    dist_to_target = torch.dist(current_pos, target_pos)
    if dist_to_target < 0.2:
        # New target at absolute positions matching training distribution:
        # XY in [-1, 1], Z in [0.5, 1.5] (within 3x3x1.7m room)
        target_pos = torch.empty(3, dtype=torch.float32, device=device)
        target_pos[:2].uniform_(-1.0, 1.0)
        target_pos[2].uniform_(0.5, 1.5)
        logger.info(f"/!\\ New target={target_pos}")
    # Rotate world-frame Kalman velocity into body frame
    linear_vel_b = quat_apply(quat_inv(current_quat), current_vel)
    desired_pos_b = quat_apply(quat_inv(current_quat), target_pos - current_pos)
    obs = torch.cat([linear_vel_b, desired_pos_b], dim=-1)
    return obs

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

def load_agent(checkpoint_path: Optional[str], device: torch.device) -> PPO:
    obs_space = 6
    act_space = 4
    policy = Policy(observation_space=obs_space, action_space=act_space, device=device)
    models = {"policy": policy}
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": obs_space, "device": device}
    agent = PPO(models=models, memory=None, cfg=cfg,
                observation_space=obs_space, action_space=act_space, device=device)
    assert checkpoint_path and os.path.exists(checkpoint_path), "No valid checkpoint provided."
    agent.load(checkpoint_path)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return agent

def main():
    parser = argparse.ArgumentParser(description="Run a trained SKRL PPO agent on a Crazyflie drone.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--uri", type=str, default="radio://0/40/2M/E7E7E7E7E1")
    parser.add_argument("--hover-thrust", type=int, default=30000,
                        help="Hover thrust PWM (default: 30000, calibrate empirically)")
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        help="Initial target [x, y, z] in world frame. If not set, hovers above takeoff pos.")
    args = parser.parse_args()
    agent = load_agent(args.checkpoint, device)
    initial_target = args.target  # None or [x, y, z]
    controller = CrazyflieController(uri=args.uri, agent=agent, hover_thrust=args.hover_thrust,
                                     initial_target=initial_target)
    try:
        controller.start()
        while not controller.cf.is_connected():
            time.sleep(1)
        logger.info("Cf is connected !")
        while True:
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
