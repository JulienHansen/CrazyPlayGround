import os
import time
import threading
import argparse
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

# skrl imports
from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(format="{asctime} [{levelname}] {message}",
                        style="{",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
logger = logging.getLogger("CrazyflieRL")
target_pos = torch.empty(3, dtype=torch.float32, device=device)
target_pos[:2].uniform_(-1.5, 1.5)
target_pos[2].uniform_(0.7, 1.5)

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
    def __init__(self, uri: str, agent: PPO):
        self.uri = uri
        self.cf = Crazyflie()
        self.agent = agent
        self.previous_pos = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_pos = torch.zeros(3, dtype=torch.float32, device=device)
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
        LOG_FREQUENCY_IN_MS = 50
        log_pos = LogConfig(name="pos", period_in_ms=LOG_FREQUENCY_IN_MS)
        log_pos.add_variable("stateEstimate.x", "float")
        log_pos.add_variable("stateEstimate.y", "float")
        log_pos.add_variable("stateEstimate.z", "float")
        self.cf.log.add_config(log_pos)
        log_pos.data_received_cb.add_callback(self._log_pos_callback)
        log_pos.start()

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

    def _log_pos_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        with self.lock:
            self.previous_pos = self.current_pos.clone()
            self.current_pos = torch.tensor([
                data["stateEstimate.x"],
                data["stateEstimate.y"],
                data["stateEstimate.z"]
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
        """Main control loop: send attitude commands"""
        INTERVAL = 0.01  # control frequency (s) - 100 Hz
        MAX_ANGLE = 30.0          # degrees  — must match att_hovering.py max_roll_pitch
        MAX_YAW_RATE = 90.0       # deg/s    — must match att_hovering.py max_yaw_rate
        # Hover PWM: mass(0.027kg)×g / (max_thrust(0.638N)/PWM_max(65535)) ≈ 27 200
        # Using firmware estimate from crazyflie.yaml thrust_base = 30 000.
        # Calibrate empirically if the drone doesn't hold altitude at action[3]=0.
        HOVER_THRUST = 30000
        MIN_THRUST_SCALE = 0.5    # fraction of hover — must match att_hovering.py
        MAX_THRUST_SCALE = 1.8    # fraction of hover — must match att_hovering.py

        logger.info("Waiting for position data...")
        while not self.position_received and self.running:
            time.sleep(0.1)
        logger.info(f"Position received: {self.current_pos}")

        logger.info("Unlocking motors with thrust=0...")
        for _ in range(10):
            self.cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.01)
        logger.info("Motors unlocked")

        TAKEOFF_THRUST = int(HOVER_THRUST * 1.2)
        TAKEOFF_DURATION = 2.0
        logger.info(f"Manual takeoff with thrust={TAKEOFF_THRUST} for {TAKEOFF_DURATION}s...")
        takeoff_start = time.time()
        while time.time() - takeoff_start < TAKEOFF_DURATION and self.running:
            self.cf.commander.send_setpoint(0, 0, 0, TAKEOFF_THRUST)
            time.sleep(INTERVAL)
            logger.info(f"Takeoff: thrust={TAKEOFF_THRUST}, pos={self.current_pos}")
        logger.info(f"Takeoff complete, current pos: {self.current_pos}")
        logger.info(f"Init target pos={target_pos}")

        while self.cf.is_connected() and self.running:
            start_time = time.time()

            # ── Safety watchdog ──────────────────────────────────────────────
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

            obs = retrieve_and_create_observation(self.previous_pos, self.current_pos, self.current_quat, INTERVAL)
            if obs is None:
                logger.warning("No observation received, hovering...")
                time.sleep(INTERVAL)
                continue

            with torch.no_grad():
                action = torch.zeros(4, device=device)
                logger.debug(f"Action={action}")

            roll  = action[0].item() * MAX_ANGLE
            pitch = action[1].item() * MAX_ANGLE
            yaw   = action[2].item() * MAX_YAW_RATE
            # Thrust: action[3] in [0,1] — matches att_hovering.py and teleop_env.py convention.
            # 0 = min thrust, 1 = max thrust, ~0.556 = hover.
            thrust_norm = float(max(0.0, min(1.0, action[3].item())))
            thrust = int(HOVER_THRUST * (MIN_THRUST_SCALE + thrust_norm * (MAX_THRUST_SCALE - MIN_THRUST_SCALE)))
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


def retrieve_and_create_observation(previous_pos, current_pos, current_quat, time_elapsed) -> Optional[torch.Tensor]:
    global target_pos
    dist_to_target = torch.dist(current_pos, target_pos)
    if dist_to_target < 0.2:
        target_pos = torch.empty(3, dtype=torch.float32, device=device)
        target_pos[:2].uniform_(-1.5, 1.5)
        target_pos[2].uniform_(0.7, 1.5)
        logger.info(f"/!\ New target={target_pos}")
    linear_vel_world = (current_pos - previous_pos) / time_elapsed
    linear_vel = quat_apply(quat_inv(current_quat), linear_vel_world)
    desired_pos_b = quat_apply(quat_inv(current_quat), target_pos - current_pos)
    obs = torch.cat([linear_vel, desired_pos_b], dim=-1)
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
    agent = PPO(models=models, memory=None, cfg=cfg,
                observation_space=obs_space, action_space=act_space, device=device)
    assert checkpoint_path and os.path.exists(checkpoint_path), "No valid checkpoint provided."
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return agent

def main():
    parser = argparse.ArgumentParser(description="Run a trained SKRL PPO agent on a Crazyflie drone.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--uri", type=str, default="radio://0/40/2M/E7E7E7E7E1")
    args = parser.parse_args()
    agent = load_agent(args.checkpoint, device)
    controller = CrazyflieController(uri=args.uri, agent=agent)
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
