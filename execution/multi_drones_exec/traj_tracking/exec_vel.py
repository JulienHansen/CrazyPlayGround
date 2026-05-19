import os
import math
import time
import threading
import argparse
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(format="{asctime} [{levelname}] [{name}] {message}",
                    style="{", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("CrazyflieRL")

# ── Safety thresholds ──────────────────────────────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without position callback
POS_VARIANCE_THRESHOLD = 0.5   # kalman variance [m²]

# ── Trajectory parameters (must match VelTrackingEnvCfg) ──────────────────────
TRAJ_W         = 1.0             # angular speed multiplier
TRAJ_SCALE     = [2.0, 2.0, 1.0] # [m] XY and Z scale
TRAJ_C         = 0.0             # lemniscate z-coupling parameter
FUTURE_STEPS   = 4               # lookahead waypoints in observation
TRAJ_STEP_SIZE = 5.0             # sim-steps between consecutive waypoints
CONTROL_DT     = 0.01            # policy step (s) — must match decimation*sim_dt (5*0.002)

# ── Velocity action params ─────────────────────────────────────────────────────
MAX_VELOCITY = 3.0  # m/s  (must match VelTrackingEnvCfg.max_velocity)


# ── Trajectory utilities ───────────────────────────────────────────────────────

def _lemniscate(t: float, c: float = 0.0):
    """Lemniscate of Bernoulli. Returns unnormalised (unit-amplitude) x, y, z."""
    denom = 1.0 + math.sin(t) ** 2
    x = math.cos(t) / denom
    y = math.sin(t) * math.cos(t) / denom
    z = c * math.sin(2.0 * t)
    return x, y, z


def compute_waypoints_world(step: int, origin: torch.Tensor) -> torch.Tensor:
    """Compute FUTURE_STEPS lookahead waypoints in world frame for a given origin."""
    scale = torch.tensor(TRAJ_SCALE, dtype=torch.float32, device=device)
    pts = []
    for k in range(1, FUTURE_STEPS + 1):
        t_k = TRAJ_W * (step + k * TRAJ_STEP_SIZE) * CONTROL_DT
        x, y, z = _lemniscate(t_k, c=TRAJ_C)
        raw = torch.tensor([x, y, z], dtype=torch.float32, device=device)
        pts.append(raw * scale + origin)
    return torch.stack(pts, dim=0)


# ── Quaternion math ────────────────────────────────────────────────────────────

def quat_apply(quat, vec):
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec  = vec.reshape(-1, 3)
    xyz  = quat[:, 1:]
    t    = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_conjugate(q):
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


def quat_inv(q, eps=1e-9):
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


# ── Observation builder ────────────────────────────────────────────────────────

def build_observation(step: int, pos: torch.Tensor, vel_w: torch.Tensor,
                      quat: torch.Tensor, ang_vel_b: torch.Tensor,
                      origin: torch.Tensor) -> torch.Tensor:
    """22-D obs: rpos_1..4 (12) | lin_vel_b (3) | ang_vel_b (3) | quat (4)."""
    waypoints_w = compute_waypoints_world(step, origin)
    rpos_list = []
    for i in range(FUTURE_STEPS):
        rpos_b = quat_apply(quat_inv(quat), waypoints_w[i] - pos)
        rpos_list.append(rpos_b)
    lin_vel_b = quat_apply(quat_inv(quat), vel_w)
    return torch.cat(rpos_list + [lin_vel_b, ang_vel_b, quat], dim=-1)


# ── Model definition ───────────────────────────────────────────────────────────

class Policy(GaussianMixin, Model):
    """Must match the skrl_ppo_cfg.yaml network: [256, 128, 64] with ELU."""

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0, initial_log_std=0.0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        self.policy_layer = nn.Linear(64, self.num_actions)
        self.value_layer  = nn.Linear(64, 1)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * initial_log_std)

    def compute(self, inputs, role):
        x = self.net_container(inputs["states"])
        mean = self.policy_layer(x) if role == "policy" else self.value_layer(x)
        return mean, self.log_std_parameter, {}


# ── Crazyflie controller ───────────────────────────────────────────────────────

class CrazyflieController:
    """Per-drone trajectory-tracking velocity-control RL agent runner."""

    def __init__(self, name: str, uri: str, agent: PPO, origin: List[float]):
        self.name   = name
        self.uri    = uri
        self.cf     = Crazyflie()
        self.agent  = agent
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.log    = logging.getLogger(f"CrazyflieRL.{name}")

        self.current_pos     = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_vel_w   = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_quat    = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device)
        self.current_ang_vel = torch.zeros(3, dtype=torch.float32, device=device)

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
        self.log.info(f"Connected to {uri}")
        self._start_logging()
        threading.Thread(target=self.control_loop, daemon=True).start()

    def _disconnected(self, uri: str): pass
    def _connection_failed(self, uri: str, msg: str): pass

    def _connection_lost(self, uri: str, msg: str):
        self.log.warning(f"Connection lost: {msg} — triggering safe landing")
        self.running = False

    def _start_logging(self):
        LOG_FREQ_MS = 20

        log_pos = LogConfig(name="pos", period_in_ms=LOG_FREQ_MS)
        for v in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z"):
            log_pos.add_variable(v, "float")
        self.cf.log.add_config(log_pos)
        log_pos.data_received_cb.add_callback(self._log_pos_callback)
        log_pos.start()

        log_vel = LogConfig(name="vel", period_in_ms=LOG_FREQ_MS)
        for v in ("stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"):
            log_vel.add_variable(v, "float")
        self.cf.log.add_config(log_vel)
        log_vel.data_received_cb.add_callback(self._log_vel_callback)
        log_vel.start()

        log_quat = LogConfig(name="quat", period_in_ms=LOG_FREQ_MS)
        for v in ("stateEstimate.qw", "stateEstimate.qx", "stateEstimate.qy", "stateEstimate.qz"):
            log_quat.add_variable(v, "float")
        self.cf.log.add_config(log_quat)
        log_quat.data_received_cb.add_callback(self._log_quat_callback)
        log_quat.start()

        log_gyro = LogConfig(name="gyro", period_in_ms=LOG_FREQ_MS)
        for v in ("gyro.x", "gyro.y", "gyro.z"):
            log_gyro.add_variable(v, "float")
        self.cf.log.add_config(log_gyro)
        log_gyro.data_received_cb.add_callback(self._log_gyro_callback)
        log_gyro.start()

        log_var = LogConfig(name="quality", period_in_ms=200)
        for v in ("kalman.varPX", "kalman.varPY", "kalman.varPZ"):
            log_var.add_variable(v, "float")
        self.cf.log.add_config(log_var)
        log_var.data_received_cb.add_callback(self._log_variance_callback)
        log_var.start()

    def _log_pos_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        with self.lock:
            self.current_pos = torch.tensor([
                data["stateEstimate.x"],
                data["stateEstimate.y"],
                data["stateEstimate.z"],
            ], dtype=torch.float32, device=device)
            self._last_pos_time = time.time()
            self.position_received = True

    def _log_vel_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        with self.lock:
            self.current_vel_w = torch.tensor([
                data["stateEstimate.vx"],
                data["stateEstimate.vy"],
                data["stateEstimate.vz"],
            ], dtype=torch.float32, device=device)

    def _log_quat_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        with self.lock:
            self.current_quat = torch.tensor([
                data["stateEstimate.qw"],
                data["stateEstimate.qx"],
                data["stateEstimate.qy"],
                data["stateEstimate.qz"],
            ], dtype=torch.float32, device=device)

    def _log_gyro_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        DEG2RAD = math.pi / 180.0
        with self.lock:
            self.current_ang_vel = torch.tensor([
                data["gyro.x"] * DEG2RAD,
                data["gyro.y"] * DEG2RAD,
                data["gyro.z"] * DEG2RAD,
            ], dtype=torch.float32, device=device)

    def _log_variance_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        with self.lock:
            self._pos_variance = torch.tensor([
                data["kalman.varPX"],
                data["kalman.varPY"],
                data["kalman.varPZ"],
            ], dtype=torch.float32, device=device)

    def _emergency_land(self):
        self.log.warning("EMERGENCY LANDING triggered")
        self.running = False
        try:
            self.cf.high_level_commander.land(0.0, 2.0)
        except Exception as e:
            self.log.error(f"Emergency land failed: {e}")
            try:
                self.cf.commander.send_stop_setpoint()
            except Exception:
                pass

    def control_loop(self):
        """Phase 1: HLC takeoff. Phase 2: NN velocity-control trajectory tracking."""
        TAKEOFF_HEIGHT   = float(self.origin[2].item())
        TAKEOFF_DURATION = 3.0
        STABILIZE_PAUSE  = 1.0

        self.log.info("Waiting for position data...")
        while not self.position_received and self.running:
            time.sleep(0.1)
        self.log.info(f"Position received: {self.current_pos}")

        self.log.info(f"Takeoff to {TAKEOFF_HEIGHT} m ...")
        self.cf.high_level_commander.takeoff(TAKEOFF_HEIGHT, TAKEOFF_DURATION)
        time.sleep(TAKEOFF_DURATION + STABILIZE_PAUSE)
        self.log.info(f"Takeoff complete. Current pos: {self.current_pos}")

        self.log.info("NN velocity-control trajectory tracking active.")
        step = 0
        while self.cf.is_connected() and self.running:
            start_time = time.time()

            if self._last_pos_time > 0 and time.time() - self._last_pos_time > POS_STALE_TIMEOUT_S:
                self.log.error(
                    f"Position stale ({time.time() - self._last_pos_time:.2f} s) — emergency landing"
                )
                self._emergency_land()
                break
            with self.lock:
                var = self._pos_variance.clone()
            if var.max().item() > POS_VARIANCE_THRESHOLD:
                self.log.error(f"Position variance too high {var.tolist()} — emergency landing")
                self._emergency_land()
                break

            with self.lock:
                pos  = self.current_pos.clone()
                vel  = self.current_vel_w.clone()
                quat = self.current_quat.clone()
                ang  = self.current_ang_vel.clone()

            obs = build_observation(step, pos, vel, quat, ang, self.origin)

            with torch.no_grad():
                action_dict = self.agent.act(obs, 1, 0)
                action = action_dict[0].clamp(-1.0, 1.0)

            vel_cmd = action * MAX_VELOCITY
            self.log.info(
                f"Step={step} | "
                f"vel_cmd={[f'{v:.2f}' for v in vel_cmd.tolist()]} | "
                f"pos={[f'{v:.2f}' for v in pos.tolist()]}"
            )

            self.cf.commander.send_velocity_world_setpoint(
                vel_cmd[0].item(), vel_cmd[1].item(), vel_cmd[2].item(), 0.0
            )

            step += 1
            elapsed = time.time() - start_time
            time.sleep(max(0, CONTROL_DT - elapsed))

        try:
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass
        self.log.info("Control loop stopped")

    def start(self):
        self.cf.open_link(self.uri)

    def stop(self):
        self.log.info("Stopping controller...")
        self.running = False
        time.sleep(0.2)
        self.log.info("Landing...")
        try:
            self.cf.high_level_commander.land(0.0, 2.0)
        except Exception as e:
            self.log.error(f"Land failed: {e}")
        time.sleep(2.5)
        try:
            self.cf.close_link()
        except Exception:
            pass
        self.log.info("Link closed")


# ── Agent loading / main ───────────────────────────────────────────────────────

def load_agent(checkpoint_path: str, device: torch.device) -> PPO:
    obs_space = 22
    act_space = 3
    policy = Policy(observation_space=obs_space, action_space=act_space, device=device)
    models  = {"policy": policy}
    cfg     = PPO_DEFAULT_CONFIG.copy()
    agent   = PPO(models=models, memory=None, cfg=cfg,
                  observation_space=obs_space, action_space=act_space, device=device)
    assert checkpoint_path and os.path.exists(checkpoint_path), \
        f"No valid checkpoint provided: {checkpoint_path}"
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Multi-drone trajectory tracking — velocity control (VelTrackingEnvCfg)"
    )
    parser.add_argument("--checkpoint1", type=str, required=True,
                        help="Checkpoint for drone 1")
    parser.add_argument("--checkpoint2", type=str, required=True,
                        help="Checkpoint for drone 2")
    parser.add_argument("--uri1", type=str, default="radio://0/40/2M/E7E7E7E7E1",
                        help="Radio URI for drone 1")
    parser.add_argument("--uri2", type=str, default="radio://0/40/2M/E7E7E7E7E2",
                        help="Radio URI for drone 2")
    parser.add_argument("--origin1", type=float, nargs=3, default=[-1.0, 0.0, 1.5],
                        metavar=("X", "Y", "Z"),
                        help="Trajectory origin for drone 1 [m]")
    parser.add_argument("--origin2", type=float, nargs=3, default=[1.0, 0.0, 1.5],
                        metavar=("X", "Y", "Z"),
                        help="Trajectory origin for drone 2 [m]")
    args = parser.parse_args()

    agent1 = load_agent(args.checkpoint1, device)
    agent2 = load_agent(args.checkpoint2, device)

    cflib.crtp.init_drivers(enable_debug_driver=False)

    c1 = CrazyflieController(name="cf1", uri=args.uri1, agent=agent1, origin=args.origin1)
    c2 = CrazyflieController(name="cf2", uri=args.uri2, agent=agent2, origin=args.origin2)

    try:
        c1.start()
        c2.start()
        while not (c1.cf.is_connected() and c2.cf.is_connected()):
            time.sleep(1)
        logger.info("Both Crazyflies connected!")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        c1.stop()
        c2.stop()
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
