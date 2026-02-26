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

# ============================================================
#                      MODEL DEFINITION
# ============================================================


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0,
                 initial_log_std=0.0):

        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                               min_log_std, max_log_std)

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
        x = self.net_container(inputs["states"])

        if role == "policy":
            mean = self.policy_layer(x)
        else:  # value function
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
        #logger.warning(f"Connection to {uri} lost: {msg}")
        pass

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


    def _log_pos_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        """Update current position from the Crazyflie logs"""
        with self.lock:
            self.previous_pos = self.current_pos
            self.current_pos = torch.tensor([
                data["stateEstimate.x"],
                data["stateEstimate.y"],
                data["stateEstimate.z"]
            ], dtype=torch.float32, device=device)

    def _log_data_quat_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        """Update current quaternion from the Crazyflie logs"""
        with self.lock:
            self.current_quat = torch.tensor([
                data['stateEstimate.qw'],
                data['stateEstimate.qx'],
                data['stateEstimate.qy'],
                data['stateEstimate.qz'],
            ], dtype=torch.float32, device=device)


    # ---------- Control loop ----------

    def control_loop(self):
        """Main control loop: send position displacements as commands"""
        MAX_VEL = 0.5  # displacement scale (m per action unit)
        INTERVAL = 0.05  # control frequency (s)

        #logger.info("Starting control loop")
        
        logger.info(f"Init target pos={target_pos}")
        while self.cf.is_connected():
            start_time = time.time()

            obs = retrieve_and_create_observation(self.previous_pos, self.current_pos, self.current_quat, INTERVAL)
            if obs is None:
                logger.warning("No observation received, hovering...")
                time.sleep(INTERVAL)
                continue

            with torch.no_grad():
                action_dict = self.agent.act(obs, 1, 0) # TODO 1 and 0 are tmp should have no impact at all
                action = action_dict[0]
                logger.debug(f"Action={action}")

            velocity_cmd = torch.clamp(action, -1.0, 1.0) * MAX_VEL

            self.cf.commander.send_velocity_world_setpoint( # send_velocity_world ou send_velocity ? TODO check
                velocity_cmd[0].item(),
                velocity_cmd[1].item(),
                velocity_cmd[2].item(),
                0.0
            )

            elapsed = time.time() - start_time
            time.sleep(max(0, INTERVAL - elapsed))

        self.cf.commander.send_stop_setpoint()
        #logger.info("Control loop stopped")

    # ---------- Connection management ----------

    def start(self):
        """Connect to Crazyflie"""
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf.open_link(self.uri)

    def stop(self):
        """Disconnect safely"""
        self.cf.high_level_commander.land(0, 3)
        time.sleep(3.5)
        self.cf.close_link()
        #logger.info("Link closed")


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
        target_pos[:2].uniform_(-1.5, 1.5)
        target_pos[2].uniform_(0.7, 1.5)

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
        from cflib.utils.power_switch import PowerSwitch
        PowerSwitch(args.uri).stm_power_cycle()


if __name__ == "__main__":
    main()
