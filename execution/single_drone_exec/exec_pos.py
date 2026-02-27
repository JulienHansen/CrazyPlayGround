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
# Arena limits
ARENA_X = (-0.5, 0.5)
ARENA_Y = (-0.5, 0.5)
ARENA_Z = (0.3, 1.2)

target_pos = torch.empty(3, dtype=torch.float32, device=device)
target_pos[0].uniform_(*ARENA_X)
target_pos[1].uniform_(*ARENA_Y)
target_pos[2].uniform_(*ARENA_Z)

# ── Safety thresholds ────────────────────────────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without a position callback before emergency land
POS_VARIANCE_THRESHOLD = 0.5   # kalman position variance [m²] above which tracking is unreliable

# ============================================================
#                      MODEL DEFINITION
# ============================================================


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

# ============================================================
#                    CRAZYFLIE CONTROLLER
# ============================================================

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
        self.running = True  # Flag pour arrêter proprement le control loop
        self._last_pos_time: float = 0.0
        self._pos_variance = torch.zeros(3, dtype=torch.float32, device=device)
        self.lock = threading.Lock()
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
        self._start_logging()
        threading.Thread(target=self.control_loop, daemon=True).start()

    def _disconnected(self, uri: str):
        #logger.info(f"Disconnected from {uri}")
        pass

    def _connection_failed(self, uri: str, msg: str):
        #logger.error(f"Connection to {uri} failed: {msg}")
        pass

    def _connection_lost(self, uri: str, msg: str):
        logger.warning(f"Connection to {uri} lost: {msg} — triggering safe landing")
        self.running = False

    # ---------- Logging setup ----------

    def _start_logging(self):
        """Configure and start position logging"""
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
        """Update current position from the Crazyflie logs"""
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
        """Update current quaternion from the Crazyflie logs"""
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

    # ---------- Control loop ----------

    def control_loop(self):
        """Main control loop: send position displacements as commands"""
        INTERVAL = 0.5  # control frequency (s) - 2 Hz
        MAX_DISPLACEMENT = 0.15  # max displacement per step (m)

        logger.info("Waiting for position data...")
        while not self.position_received and self.running:
            time.sleep(0.1)
        logger.info(f"Position received: {self.current_pos}")
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
                action_dict = self.agent.act(obs, 1, 0)
                action = action_dict[0]
                logger.info(f"Action={action}")

            with self.lock:
                displacement = torch.clamp(action, -MAX_DISPLACEMENT, MAX_DISPLACEMENT)
                desired_new_pos = self.current_pos + displacement
                desired_new_pos[0] = torch.clamp(desired_new_pos[0], *ARENA_X)
                desired_new_pos[1] = torch.clamp(desired_new_pos[1], *ARENA_Y)
                desired_new_pos[2] = torch.clamp(desired_new_pos[2], *ARENA_Z)

            logger.info(f"Moving {self.current_pos} -> {desired_new_pos}")
            self.cf.high_level_commander.go_to(
                desired_new_pos[0].item(),
                desired_new_pos[1].item(),
                desired_new_pos[2].item(),
                0.0,
                INTERVAL
            )

            elapsed = time.time() - start_time
            time.sleep(max(0, INTERVAL - elapsed))

        logger.info("Control loop stopped")

    # ---------- Connection management ----------

    def start(self):
        """Connect to Crazyflie"""
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf.open_link(self.uri)

    def stop(self):
        """Disconnect safely"""
        logger.info("Stopping controller...")
        self.running = False
        time.sleep(0.6)
        logger.info("Landing...")
        self.cf.high_level_commander.land(0.0, 2.0)
        time.sleep(2.5)
        self.cf.close_link()
        logger.info("Link closed")


# ============================================================
#                    OBSERVATION CREATION
# ============================================================

def retrieve_and_create_observation(previous_pos, current_pos, current_quat, time_elapsed) -> Optional[torch.Tensor]:
    """
    Retrieve sensor data (IMU + Lighthouse + target info)
    and return observation tensor.
    """
    global target_pos
    dist_to_target = torch.dist(current_pos, target_pos)
    logger.debug(f"Distance to target={dist_to_target}")
    if dist_to_target < 0.2:
        target_pos = torch.empty(3, dtype=torch.float32, device=device)
        target_pos[0].uniform_(*ARENA_X)
        target_pos[1].uniform_(*ARENA_Y)
        target_pos[2].uniform_(*ARENA_Z)

        logger.info(f"/!\ New target={target_pos}")

    linear_vel_world = (current_pos - previous_pos) / time_elapsed
    linear_vel = quat_apply(quat_inv(current_quat), linear_vel_world)

    desired_pos_b = quat_apply(quat_inv(current_quat), target_pos - current_pos)

    obs = torch.cat([linear_vel, desired_pos_b], dim=-1)
    logger.debug(f"previous_pos={previous_pos}, current_pos={current_pos}, current_quat={current_quat}, \
                  time_elapsed={time_elapsed}, target_pos={target_pos}, linear_vel={linear_vel}, desired_pos_b={desired_pos_b}")
    logger.debug(f"Obs={obs}")

    return obs


def quat_apply(quat, vec):
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_conjugate(q):
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


def quat_inv(q, eps=1e-9):
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)

# ============================================================
#                    MODEL LOADING / MAIN
# ============================================================

def load_agent(checkpoint_path: Optional[str], device: torch.device) -> PPO:
    """Load PPO agent and policy"""
    obs_space = 6
    act_space = 3

    policy = Policy(observation_space=obs_space, action_space=act_space, device=device)
    models = {"policy": policy}

    cfg = PPO_DEFAULT_CONFIG.copy()
    agent = PPO(models=models, memory=None, cfg=cfg,
                observation_space=obs_space, action_space=act_space, device=device)

    assert checkpoint_path and os.path.exists(checkpoint_path), "No valid checkpoint provided. Please give a path for weights."

    agent.load(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")

    return agent


def main():
    parser = argparse.ArgumentParser(description="Run a trained SKRL PPO agent on a Crazyflie drone.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--uri", type=str, default="radio://0/40/2M/E7E7E7E7E1", help="URI of the Crazyflie")
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
