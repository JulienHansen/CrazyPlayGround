import os
import math
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

from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(format="{asctime} [{levelname}] {message}",
                    style="{", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("CrazyflieRL")

# ── Safety thresholds ──────────────────────────────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without position callback
POS_VARIANCE_THRESHOLD = 0.5   # kalman variance [m²]

# ── Trajectory parameters (must match AttTrackingEnvCfg) ──────────────────────
TRAJ_W         = 1.0             # angular speed multiplier
TRAJ_SCALE     = [2.0, 2.0, 1.0] # [m] XY and Z scale
TRAJ_ORIGIN    = [0.0, 0.0, 1.5] # world-frame origin [m]
TRAJ_C         = 0.0             # lemniscate z-coupling parameter
FUTURE_STEPS   = 4               # lookahead waypoints in observation
TRAJ_STEP_SIZE = 5.0             # sim-steps between consecutive waypoints
CONTROL_DT     = 0.01            # policy step (s) — must match decimation*sim_dt (5*0.002)

# ── Attitude action params (must match AttTrackingEnvCfg) ─────────────────────
MAX_ANGLE     = 30.0   # degrees  (max_roll_pitch for send_setpoint_manual angle mode)
MAX_YAW_RATE  = 90.0   # deg/s
GRAVITY       = 9.81   # m/s²


# ── Trajectory utilities ───────────────────────────────────────────────────────

def _lemniscate(t: float, c: float = 0.0):
    """Lemniscate of Bernoulli. Returns unnormalised (unit-amplitude) x, y, z."""
    denom = 1.0 + math.sin(t) ** 2
    x = math.cos(t) / denom
    y = math.sin(t) * math.cos(t) / denom
    z = c * math.sin(2.0 * t)
    return x, y, z


def compute_waypoints_world(step: int) -> torch.Tensor:
    """Compute FUTURE_STEPS lookahead waypoints in world frame.

    Uses the same formula as TrackingEnv._compute_traj (fixed params, no env offset).

    Returns:
        Tensor [FUTURE_STEPS, 3]
    """
    scale  = torch.tensor(TRAJ_SCALE,  dtype=torch.float32, device=device)
    origin = torch.tensor(TRAJ_ORIGIN, dtype=torch.float32, device=device)
    pts = []
    for k in range(1, FUTURE_STEPS + 1):
        t_k = TRAJ_W * (step + k * TRAJ_STEP_SIZE) * CONTROL_DT
        x, y, z = _lemniscate(t_k, c=TRAJ_C)
        raw = torch.tensor([x, y, z], dtype=torch.float32, device=device)
        pts.append(raw * scale + origin)
    return torch.stack(pts, dim=0)  # [4, 3]


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
                      quat: torch.Tensor, ang_vel_b: torch.Tensor) -> torch.Tensor:
    """Build the 22-D observation matching TrackingEnv._get_observations().

    Layout: rpos_1..4 (3×4=12) | lin_vel_b (3) | ang_vel_b (3) | quat (4) = 22
    """
    waypoints_w = compute_waypoints_world(step)  # [4, 3]
    rpos_list = []
    for i in range(FUTURE_STEPS):
        rpos_b = quat_apply(quat_inv(quat), waypoints_w[i] - pos)
        rpos_list.append(rpos_b)
    lin_vel_b = quat_apply(quat_inv(quat), vel_w)
    return torch.cat(rpos_list + [lin_vel_b, ang_vel_b, quat], dim=-1)  # [22]


# ── Model definition ───────────────────────────────────────────────────────────

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0, initial_log_std=0.0):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 32), nn.ELU(),
            nn.Linear(32, 32), nn.ELU()
        )
        self.policy_layer = nn.Linear(32, self.num_actions)
        self.value_layer  = nn.Linear(32, 1)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * initial_log_std)

    def compute(self, inputs, role):
        x = self.net_container(inputs["observations"])
        mean = self.policy_layer(x) if role == "policy" else self.value_layer(x)
        return mean, {"log_std": self.log_std_parameter}


# ── Crazyflie controller ───────────────────────────────────────────────────────

class CrazyflieController:
    """Runs a trained trajectory-tracking attitude-control RL agent on the Crazyflie."""

    def __init__(self, uri: str, agent: PPO,
                 mass_kg: float = 0.027,
                 max_thrust_N: float = 0.638,
                 min_thrust_scale: float = 0.5,
                 max_thrust_scale: float = 1.8):
        self.uri   = uri
        self.cf    = Crazyflie()
        self.agent = agent

        self.mass_kg          = mass_kg
        self.max_thrust_N     = max_thrust_N
        self.min_thrust_scale = min_thrust_scale
        self.max_thrust_scale = max_thrust_scale
        self.weight_N         = mass_kg * GRAVITY
        self.hover_thrust_pct = max(0.0, min(100.0, 100.0 * self.weight_N / self.max_thrust_N))
        logger.info(
            f"Physics: m={mass_kg} kg, max_thrust={max_thrust_N} N, "
            f"hover_pct={self.hover_thrust_pct:.1f}%"
        )

        self.current_pos     = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_vel_w   = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_quat    = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device)
        self.current_ang_vel = torch.zeros(3, dtype=torch.float32, device=device)  # body frame, rad/s

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

    def _disconnected(self, uri: str): pass
    def _connection_failed(self, uri: str, msg: str): pass

    def _connection_lost(self, uri: str, msg: str):
        logger.warning(f"Connection to {uri} lost: {msg} — triggering safe landing")
        self.running = False

    def _start_logging(self):
        # 50 Hz for all primary state logs — raise if CF firmware runs out of bandwidth
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
        # Crazyflie gyro logs in deg/s; simulation uses rad/s
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
        logger.warning("EMERGENCY LANDING triggered")
        self.running = False
        try:
            self.cf.high_level_commander.land(0.0, 2.0)
        except Exception as e:
            logger.error(f"Emergency land failed: {e}")
            try:
                self.cf.commander.send_stop_setpoint()
            except Exception:
                pass

    def control_loop(self):
        """Phase 1: HLC takeoff. Phase 2: hover handoff. Phase 3: NN attitude tracking."""
        TAKEOFF_HEIGHT   = float(TRAJ_ORIGIN[2])
        TAKEOFF_DURATION = 3.0
        STABILIZE_PAUSE  = 1.5

        logger.info("Waiting for position data...")
        while not self.position_received and self.running:
            time.sleep(0.1)
        logger.info(f"Position received: {self.current_pos}")

        # ── Phase 1: position-controlled takeoff ─────────────────────────────
        logger.info(f"Takeoff to {TAKEOFF_HEIGHT} m ...")
        self.cf.high_level_commander.takeoff(TAKEOFF_HEIGHT, TAKEOFF_DURATION)
        time.sleep(TAKEOFF_DURATION + STABILIZE_PAUSE)
        logger.info(f"Takeoff complete. Current pos: {self.current_pos}")

        # ── Phase 2: hover handoff to low-level attitude commander ───────────
        # Sending send_setpoint_manual() switches the firmware away from HLC mode.
        # Hold zero roll/pitch/yaw and hover thrust (angle mode) for a smooth handoff.
        logger.info("Transitioning to attitude control (hover handoff)...")
        for _ in range(20):
            self.cf.commander.send_setpoint_manual(0.0, 0.0, 0.0, self.hover_thrust_pct, False)
            time.sleep(CONTROL_DT)

        # ── Phase 3: NN attitude control loop ────────────────────────────────
        logger.info("NN attitude-control trajectory tracking active.")
        step = 0
        while self.cf.is_connected() and self.running:
            start_time = time.time()

            # Safety watchdog
            if self._last_pos_time > 0 and time.time() - self._last_pos_time > POS_STALE_TIMEOUT_S:
                logger.error(
                    f"Position stale ({time.time() - self._last_pos_time:.2f} s) — emergency landing"
                )
                self._emergency_land()
                break
            with self.lock:
                var = self._pos_variance.clone()
            if var.max().item() > POS_VARIANCE_THRESHOLD:
                logger.error(f"Position variance too high {var.tolist()} — emergency landing")
                self._emergency_land()
                break

            with self.lock:
                pos  = self.current_pos.clone()
                vel  = self.current_vel_w.clone()
                quat = self.current_quat.clone()
                ang  = self.current_ang_vel.clone()

            obs = build_observation(step, pos, vel, quat, ang)

            with torch.no_grad():
                _, outputs = self.agent.act(obs.unsqueeze(0), None, timestep=0, timesteps=1)
                action = outputs["mean_actions"].squeeze(0)
                action = action.clamp(-1.0, 1.0)

            # action[0:2] = roll/pitch in [-1, 1] → degrees
            # action[2]   = yaw_rate   in [-1, 1] → deg/s
            # action[3]   = thrust     → clamped to [0, 1] (AttTrackingEnvCfg convention)
            roll  = action[0].item() * MAX_ANGLE
            pitch = action[1].item() * MAX_ANGLE
            yaw   = action[2].item() * MAX_YAW_RATE
            thrust_norm = max(0.0, min(1.0, action[3].item()))
            thrust_N = self.weight_N * (
                self.min_thrust_scale + thrust_norm * (self.max_thrust_scale - self.min_thrust_scale)
            )
            thrust_pct = max(0.0, min(100.0, 100.0 * thrust_N / self.max_thrust_N))

            logger.info(
                f"Step={step} | "
                f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} thrust={thrust_pct:.1f}% | "
                f"pos={[f'{v:.2f}' for v in pos.tolist()]}"
            )
            self.cf.commander.send_setpoint_manual(roll, pitch, yaw, thrust_pct, False)

            step += 1
            elapsed = time.time() - start_time
            time.sleep(max(0, CONTROL_DT - elapsed))

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


# ── Agent loading / main ───────────────────────────────────────────────────────

def load_agent(checkpoint_path: Optional[str], device: torch.device) -> PPO:
    obs_space = 22
    act_space = 4
    policy = Policy(observation_space=obs_space, action_space=act_space, device=device)
    models  = {"policy": policy}
    cfg     = PPO_CFG(
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": obs_space, "device": device},
    )
    agent   = PPO(models=models, memory=None, cfg=cfg,
                  observation_space=obs_space, action_space=act_space, device=device)
    assert checkpoint_path and os.path.exists(checkpoint_path), \
        "No valid checkpoint provided."
    agent.load(checkpoint_path)
    agent.enable_training_mode(False)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory tracking — attitude control (AttTrackingEnvCfg)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--uri", type=str, default="radio://0/40/2M/E7E7E7E7E1",
                        help="Crazyflie radio URI")
    parser.add_argument("--mass-kg", type=float, default=0.027,
                        help="Drone mass (kg). Must match the sim cfg used for training.")
    parser.add_argument("--max-thrust-N", type=float, default=0.638,
                        help="Max total thrust (N, sum of 4 motors). Must match sim cfg.")
    parser.add_argument("--min-thrust-scale", type=float, default=0.5,
                        help="Min thrust scale factor of m·g (AttTrackingEnvCfg).")
    parser.add_argument("--max-thrust-scale", type=float, default=1.8,
                        help="Max thrust scale factor of m·g (AttTrackingEnvCfg).")
    args = parser.parse_args()

    agent = load_agent(args.checkpoint, device)
    controller = CrazyflieController(
        uri=args.uri,
        agent=agent,
        mass_kg=args.mass_kg,
        max_thrust_N=args.max_thrust_N,
        min_thrust_scale=args.min_thrust_scale,
        max_thrust_scale=args.max_thrust_scale,
    )

    try:
        controller.start()
        while not controller.cf.is_connected():
            time.sleep(1)
        logger.info("Crazyflie connected!")
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
