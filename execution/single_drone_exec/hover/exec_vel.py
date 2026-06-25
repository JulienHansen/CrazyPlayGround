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
target_pos = None  # initialized in control_loop after takeoff

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
    """Crazyflie controller for velocity-based RL agents.

    Uses send_velocity_world_setpoint to command velocity in world frame.
    Sim equivalent: vel_hovering.py (command_level="velocity", max_velocity=1.0 m/s).
    """
    def __init__(self, uri: str, agent: PPO, initial_target=None):
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
        pass

    def _connection_failed(self, uri: str, msg: str):
        logger.error(f"Connection to {uri} failed: {msg}")
        self.running = False

    def _connection_lost(self, uri: str, msg: str):
        logger.warning(f"Connection to {uri} lost: {msg} — triggering safe landing")
        self.running = False

    # ---------- Logging setup ----------

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

        # Block 3: Kalman variance (for safety watchdog)
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
        with self.lock:
            self._pos_variance = torch.tensor([
                data["kalman.varPX"],
                data["kalman.varPY"],
                data["kalman.varPZ"],
            ], dtype=torch.float32, device=device)

    def _emergency_land(self):
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
                time.sleep(INTERVAL)
                continue

            with torch.no_grad():
                action_dict = self.agent.act(obs, 1, 0)
                action = action_dict[2]["mean_actions"].squeeze(0)  # deterministic mean
                action = action.clamp(-1.0, 1.0)

            # Scale action to velocity in m/s (matching sim: actions * max_velocity)
            velocity_cmd = action * MAX_VEL

            logger.info(
                f"Cmd: vx={velocity_cmd[0].item():+.2f} vy={velocity_cmd[1].item():+.2f} "
                f"vz={velocity_cmd[2].item():+.2f} m/s  | pos={self.current_pos}"
            )
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
        logger.info("Link closed")


# ============================================================
#                    OBSERVATION CREATION
# ============================================================

def retrieve_and_create_observation(current_vel, current_pos, current_quat) -> Optional[torch.Tensor]:
    """Build obs tensor matching vel_hovering.py: [lin_vel_b(3), desired_pos_b(3)]."""
    global target_pos
    if target_pos is None:
        return None
    dist_to_target = torch.dist(current_pos, target_pos)
    if dist_to_target < 0.2:
        # New target matching training distribution: XY=[-1,1], Z=[0.5,1.5]
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

# ============================================================
#                    MODEL LOADING / MAIN
# ============================================================

def load_agent(checkpoint_path: Optional[str], device: torch.device) -> PPO:
    obs_space = 6
    act_space = 3

    policy = Policy(observation_space=obs_space, action_space=act_space, device=device)
    models = {"policy": policy}

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": obs_space, "device": device}
    agent = PPO(models=models, memory=None, cfg=cfg,
                observation_space=obs_space, action_space=act_space, device=device)

    assert checkpoint_path and os.path.exists(checkpoint_path), "No valid checkpoint provided. Please give a path for weights."

    agent.load(checkpoint_path)
    agent.set_running_mode("eval")
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return agent


def main():
    parser = argparse.ArgumentParser(description="Run a trained velocity RL agent on a Crazyflie drone.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E8", help="URI of the Crazyflie")
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        help="Initial target [x, y, z] in world frame. If not set, hovers above takeoff pos.")
    args = parser.parse_args()

    agent = load_agent(args.checkpoint, device)

    controller = CrazyflieController(uri=args.uri, agent=agent, initial_target=args.target)

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
