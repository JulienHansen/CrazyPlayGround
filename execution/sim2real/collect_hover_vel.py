"""Sim-to-real data collection for the VELOCITY hover controller.

Runs the trained velocity hover policy on a real Crazyflie 2.1 (exactly like
``execution/single_drone_exec/hover/exec_vel.py``) and, on top of flying it,
records everything the sim-to-real protocol needs to a file:

    position, velocity, quaternion, gyro, accelerometer, motor.m1..m4,
    battery (pm.vbat), kalman variance, plus the policy observation, the raw
    policy action and the velocity command actually sent.

Output (per run, under ``--outdir``):
    <run>/flight.csv       one row per control step (100 Hz), time-aligned
    <run>/flight.npz       same data as numpy arrays (if numpy available)
    <run>/metadata.json    frozen config: checkpoint hash, git commit, rates,
                           frame/quaternion conventions, thresholds, versions

This directly feeds the protocol tests:
    OL-1 (offline replay)  -> replay ``cmd_*`` in sim from the logged start state
    OL-3 (actuator check)  -> ``m1..m4`` / ``vbat`` vs thrust
    OL-4 (state estimation)-> compare logged EKF state vs motion capture
    CL-1 (hover gap)       -> analyze_hover.py computes the hover metrics

Conventions (must match vel_hovering.py):
    quaternion order  (qw, qx, qy, qz)  -- scalar first
    velocity frame    world (stateEstimate.v*), rotated to body via quat_inv
    observation       [lin_vel_b(3), desired_pos_b(3)]  (dim 6)
    action            [-1, 1]^3, scaled by max_velocity (1.0 m/s)
    command           send_velocity_world_setpoint(vx, vy, vz, yaw_rate=0)
    control rate      100 Hz  (sim dt=1/500, decimation=5)
"""

import os
import csv
import json
import time
import hashlib
import threading
import argparse
import logging
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

# skrl imports
from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler

try:
    import numpy as np
except Exception:  # numpy should be present, but never let it block a flight
    np = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(format="{asctime} [{levelname}] {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger("CrazyflieCollect")

target_pos = None  # initialized in control_loop after takeoff

# ── Constants that must match vel_hovering.py ────────────────────────────────
MAX_VELOCITY = 1.0     # m/s  -- QuadcopterEnvCfg.max_velocity
CONTROL_RATE_HZ = 100  # policy step rate (sim dt=1/500 * decimation=5)
SIM_DT = 1.0 / 500.0
DECIMATION = 5

# ── Safety thresholds (same as exec_vel.py) ──────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without a position callback before emergency land
POS_VARIANCE_THRESHOLD = 0.5   # kalman position variance [m²] above which tracking is unreliable
TARGET_REACHED_DIST    = 0.2   # m, radius that counts as "target reached" (used in --roam mode)

# ============================================================
#                      MODEL DEFINITION
#         (identical to exec_vel.py -- must match checkpoint)
# ============================================================


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0,
                 initial_log_std=0.0):

        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)

        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU()
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


# ============================================================
#                 COLLECTING CRAZYFLIE CONTROLLER
# ============================================================

# CSV columns, in order. Kept flat so the file is trivial to load anywhere.
CSV_COLUMNS = [
    "t_mono", "t_wall", "step",
    # policy observation (matches vel_hovering.py _get_observations)
    "obs_vb_x", "obs_vb_y", "obs_vb_z", "obs_errb_x", "obs_errb_y", "obs_errb_z",
    # raw policy action, clamped to [-1, 1]
    "act_x", "act_y", "act_z",
    # command actually sent to the drone
    "cmd_vx", "cmd_vy", "cmd_vz", "cmd_yawrate",
    # target position (world frame)
    "tgt_x", "tgt_y", "tgt_z",
    # EKF state
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",          # world frame
    "qw", "qx", "qy", "qz",
    # raw IMU
    "gyro_x", "gyro_y", "gyro_z",        # deg/s (as logged by firmware)
    "acc_x", "acc_y", "acc_z",           # g
    # actuation + power
    "m1", "m2", "m3", "m4",              # motor PWM
    "vbat",                              # V
    # estimator quality
    "varPX", "varPY", "varPZ",
    "dist_to_target",
]


class CollectingController:
    """Velocity hover controller that also records full telemetry to disk."""

    def __init__(self, uri: str, agent: PPO, run_dir: str, meta: dict,
                 initial_target=None, duration: Optional[float] = None,
                 roam: bool = False, fast_rate_hz: int = 100):
        self.uri = uri
        self.cf = Crazyflie(rw_cache='./cache')
        self.agent = agent
        self.run_dir = run_dir
        self.meta = meta
        self.initial_target = initial_target
        self.duration = duration
        self.roam = roam
        self.fast_period_ms = max(10, int(round(1000.0 / fast_rate_hz)))

        # latest telemetry snapshot (updated by log callbacks, read by control loop)
        self.current_pos = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_vel = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        self._gyro = [0.0, 0.0, 0.0]
        self._acc = [0.0, 0.0, 0.0]
        self._motor = [0.0, 0.0, 0.0, 0.0]
        self._vbat = 0.0
        self._pos_variance = torch.zeros(3, dtype=torch.float32, device=device)

        self.running = True
        self.position_received = False
        self._last_pos_time: float = 0.0
        self.lock = threading.Lock()

        self.records = []  # list of dict rows
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

    def _add_block(self, name: str, period_ms: int, variables, callback):
        """Add one LogConfig block, tolerating a missing variable on some firmware."""
        block = LogConfig(name=name, period_in_ms=period_ms)
        for var, typ in variables:
            block.add_variable(var, typ)
        try:
            self.cf.log.add_config(block)
        except (KeyError, AttributeError) as e:
            logger.warning(f"Log block '{name}' rejected ({e}); its columns will stay at defaults.")
            return None
        block.data_received_cb.add_callback(callback)
        block.start()
        return block

    def _start_logging(self):
        fast = self.fast_period_ms

        # Block: position + velocity (6 floats = 24 B)
        self._add_block("posvel", fast, [
            ("stateEstimate.x", "float"), ("stateEstimate.y", "float"), ("stateEstimate.z", "float"),
            ("stateEstimate.vx", "float"), ("stateEstimate.vy", "float"), ("stateEstimate.vz", "float"),
        ], self._cb_posvel)

        # Block: quaternion (4 floats = 16 B)
        self._add_block("quat", fast, [
            ("stateEstimate.qw", "float"), ("stateEstimate.qx", "float"),
            ("stateEstimate.qy", "float"), ("stateEstimate.qz", "float"),
        ], self._cb_quat)

        # Block: IMU -- gyro + accelerometer (6 floats = 24 B)
        self._add_block("imu", fast, [
            ("gyro.x", "float"), ("gyro.y", "float"), ("gyro.z", "float"),
            ("acc.x", "float"), ("acc.y", "float"), ("acc.z", "float"),
        ], self._cb_imu)

        # Block: motor PWM (4 uint32 = 16 B). Slower rate -- changes are slow vs 100 Hz.
        self._add_block("motor", max(fast, 20), [
            ("motor.m1", "uint32_t"), ("motor.m2", "uint32_t"),
            ("motor.m3", "uint32_t"), ("motor.m4", "uint32_t"),
        ], self._cb_motor)

        # Block: estimator quality + battery (4 floats = 16 B) at 10 Hz.
        self._add_block("quality", 100, [
            ("kalman.varPX", "float"), ("kalman.varPY", "float"), ("kalman.varPZ", "float"),
            ("pm.vbat", "float"),
        ], self._cb_quality)

    def _cb_posvel(self, timestamp, data, logconf):
        with self.lock:
            self.current_pos = torch.tensor([data["stateEstimate.x"], data["stateEstimate.y"],
                                             data["stateEstimate.z"]], dtype=torch.float32, device=device)
            self.current_vel = torch.tensor([data["stateEstimate.vx"], data["stateEstimate.vy"],
                                             data["stateEstimate.vz"]], dtype=torch.float32, device=device)
            self._last_pos_time = time.time()
            self.position_received = True

    def _cb_quat(self, timestamp, data, logconf):
        with self.lock:
            self.current_quat = torch.tensor([data["stateEstimate.qw"], data["stateEstimate.qx"],
                                              data["stateEstimate.qy"], data["stateEstimate.qz"]],
                                             dtype=torch.float32, device=device)

    def _cb_imu(self, timestamp, data, logconf):
        with self.lock:
            self._gyro = [data["gyro.x"], data["gyro.y"], data["gyro.z"]]
            self._acc = [data["acc.x"], data["acc.y"], data["acc.z"]]

    def _cb_motor(self, timestamp, data, logconf):
        with self.lock:
            self._motor = [data["motor.m1"], data["motor.m2"], data["motor.m3"], data["motor.m4"]]

    def _cb_quality(self, timestamp, data, logconf):
        with self.lock:
            self._pos_variance = torch.tensor([data["kalman.varPX"], data["kalman.varPY"],
                                               data["kalman.varPZ"]], dtype=torch.float32, device=device)
            self._vbat = data["pm.vbat"]

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
        INTERVAL = 1.0 / CONTROL_RATE_HZ

        logger.info("Waiting for first position estimate...")
        while not self.position_received and self.running:
            time.sleep(0.05)
        if not self.running:
            return

        global target_pos
        if self.initial_target is not None:
            target_pos = torch.tensor(self.initial_target, dtype=torch.float32, device=device)
        else:
            target_pos = self.current_pos.clone()
            target_pos[2] = max(0.5, min(1.5, target_pos[2].item()))
        logger.info(f"Init target pos={target_pos.tolist()} | roam={self.roam}")
        self.meta["target_initial"] = target_pos.tolist()

        loop_start = time.time()
        nn_start_time = loop_start
        GRACE_PERIOD = 3.0
        step = 0
        while self.cf.is_connected() and self.running:
            start_time = time.time()

            # ── Safety watchdog (identical policy to exec_vel.py) ────────────
            elapsed_since_nn = time.time() - nn_start_time
            z = self.current_pos[2].item()
            if (elapsed_since_nn > GRACE_PERIOD and z < 0.1) or z > 2.5:
                logger.error(f"Position out of bounds z={z:.2f} — emergency landing")
                self._emergency_land()
                break
            if self._last_pos_time > 0 and time.time() - self._last_pos_time > POS_STALE_TIMEOUT_S:
                logger.error(f"Position data stale ({time.time() - self._last_pos_time:.2f} s) — emergency landing")
                self._emergency_land()
                break
            with self.lock:
                var = self._pos_variance.clone()
            if var.max().item() > POS_VARIANCE_THRESHOLD:
                logger.error(f"Position variance too high {var.tolist()} — emergency landing")
                self._emergency_land()
                break

            # duration limit -> clean land + stop
            if self.duration is not None and (time.time() - loop_start) >= self.duration:
                logger.info(f"Duration {self.duration:.1f}s reached — landing")
                self.running = False
                break

            # snapshot state under the lock so the recorded row is self-consistent
            with self.lock:
                pos = self.current_pos.clone()
                vel = self.current_vel.clone()
                quat = self.current_quat.clone()
                gyro = list(self._gyro)
                acc = list(self._acc)
                motor = list(self._motor)
                vbat = float(self._vbat)
                varp = self._pos_variance.clone()

            obs, tgt, dist = build_observation(vel, pos, quat, roam=self.roam)
            if obs is None:
                time.sleep(INTERVAL)
                continue

            with torch.no_grad():
                _, outputs = self.agent.act(obs.unsqueeze(0), None, timestep=0, timesteps=1)
                action = outputs["mean_actions"].squeeze(0).clamp(-1.0, 1.0)

            velocity_cmd = action * MAX_VELOCITY
            self.cf.commander.send_velocity_world_setpoint(
                velocity_cmd[0].item(), velocity_cmd[1].item(), velocity_cmd[2].item(), 0.0)

            now = time.time()
            self.records.append({
                "t_mono": now - loop_start, "t_wall": now, "step": step,
                "obs_vb_x": obs[0].item(), "obs_vb_y": obs[1].item(), "obs_vb_z": obs[2].item(),
                "obs_errb_x": obs[3].item(), "obs_errb_y": obs[4].item(), "obs_errb_z": obs[5].item(),
                "act_x": action[0].item(), "act_y": action[1].item(), "act_z": action[2].item(),
                "cmd_vx": velocity_cmd[0].item(), "cmd_vy": velocity_cmd[1].item(),
                "cmd_vz": velocity_cmd[2].item(), "cmd_yawrate": 0.0,
                "tgt_x": tgt[0].item(), "tgt_y": tgt[1].item(), "tgt_z": tgt[2].item(),
                "pos_x": pos[0].item(), "pos_y": pos[1].item(), "pos_z": pos[2].item(),
                "vel_x": vel[0].item(), "vel_y": vel[1].item(), "vel_z": vel[2].item(),
                "qw": quat[0].item(), "qx": quat[1].item(), "qy": quat[2].item(), "qz": quat[3].item(),
                "gyro_x": gyro[0], "gyro_y": gyro[1], "gyro_z": gyro[2],
                "acc_x": acc[0], "acc_y": acc[1], "acc_z": acc[2],
                "m1": motor[0], "m2": motor[1], "m3": motor[2], "m4": motor[3],
                "vbat": vbat,
                "varPX": varp[0].item(), "varPY": varp[1].item(), "varPZ": varp[2].item(),
                "dist_to_target": dist,
            })
            step += 1

            elapsed = time.time() - start_time
            time.sleep(max(0, INTERVAL - elapsed))

        try:
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass
        logger.info(f"Control loop stopped after {step} steps")

    # ---------- Connection / persistence ----------

    def start(self):
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf.open_link(self.uri)

    def stop(self):
        logger.info("Stopping controller...")
        self.running = False
        time.sleep(0.6)
        try:
            logger.info("Landing...")
            self.cf.high_level_commander.land(0.0, 2.0)
            time.sleep(2.5)
        except Exception:
            pass
        try:
            self.cf.close_link()
        except Exception:
            pass
        logger.info("Link closed")
        self.save()

    def save(self):
        if not self.records:
            logger.warning("No records collected — nothing written.")
            return
        os.makedirs(self.run_dir, exist_ok=True)
        csv_path = os.path.join(self.run_dir, "flight.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(self.records)
        logger.info(f"Wrote {len(self.records)} rows -> {csv_path}")

        if np is not None:
            arrays = {c: np.array([r[c] for r in self.records], dtype=np.float64) for c in CSV_COLUMNS}
            np.savez(os.path.join(self.run_dir, "flight.npz"), **arrays)
            logger.info(f"Wrote {os.path.join(self.run_dir, 'flight.npz')}")

        self.meta["num_samples"] = len(self.records)
        self.meta["end_wall"] = time.time()
        self.meta["end_iso"] = datetime.now().isoformat()
        with open(os.path.join(self.run_dir, "metadata.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
        logger.info(f"Wrote {os.path.join(self.run_dir, 'metadata.json')}")


# ============================================================
#                    OBSERVATION CREATION
# ============================================================

def build_observation(current_vel, current_pos, current_quat, roam: bool = False):
    """Build obs matching vel_hovering.py: [lin_vel_b(3), desired_pos_b(3)].

    Returns (obs, target, dist). With roam=False the target is held fixed
    (protocol: freeze goals). With roam=True it resamples like exec_vel.py.
    """
    global target_pos
    if target_pos is None:
        return None, None, None
    dist = torch.dist(current_pos, target_pos).item()
    if roam and dist < TARGET_REACHED_DIST:
        target_pos = torch.empty(3, dtype=torch.float32, device=device)
        target_pos[:2].uniform_(-1.0, 1.0)
        target_pos[2].uniform_(0.5, 1.5)
        logger.info(f"/!\\ New target={target_pos.tolist()}")
        dist = torch.dist(current_pos, target_pos).item()

    linear_vel_b = quat_apply(quat_inv(current_quat), current_vel)
    desired_pos_b = quat_apply(quat_inv(current_quat), target_pos - current_pos)
    obs = torch.cat([linear_vel_b, desired_pos_b], dim=-1)
    return obs, target_pos, dist


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
    obs_space, act_space = 6, 3
    policy = Policy(observation_space=obs_space, action_space=act_space, device=device)
    cfg = PPO_CFG(
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": obs_space, "device": device},
    )
    agent = PPO(models={"policy": policy}, memory=None, cfg=cfg,
                observation_space=obs_space, action_space=act_space, device=device)
    assert checkpoint_path and os.path.exists(checkpoint_path), \
        "No valid checkpoint provided. Please give a path for weights."
    agent.load(checkpoint_path)
    agent.enable_training_mode(False)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return agent


def _sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _versions() -> dict:
    v = {"torch": torch.__version__}
    try:
        import skrl
        v["skrl"] = skrl.__version__
    except Exception:
        pass
    try:
        import cflib
        v["cflib"] = getattr(cflib, "__version__", "unknown")
    except Exception:
        pass
    return v


def build_metadata(args, checkpoint_path) -> dict:
    """Freeze everything about this run (protocol: freeze before collecting)."""
    return {
        "controller": "velocity",
        "task": "Vel-Hovering",
        "command_primitive": "send_velocity_world_setpoint(vx, vy, vz, yaw_rate=0)",
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "git_commit": _git_commit(),
        "uri": args.uri,
        "seed": args.seed,
        "trial_id": args.trial_id,
        "tag": args.tag,
        "roam": args.roam,
        "duration_s": args.duration,
        "control_rate_hz": CONTROL_RATE_HZ,
        "sim_dt": SIM_DT,
        "decimation": DECIMATION,
        "max_velocity_mps": MAX_VELOCITY,
        "obs_layout": ["lin_vel_b(3)", "desired_pos_b(3)"],
        "action_layout": ["vx", "vy", "vz"],
        "quaternion_order": "wxyz (scalar first)",
        "velocity_frame": "world (stateEstimate.v*), rotated to body via quat_inv",
        "gyro_units": "deg/s (raw firmware)",
        "acc_units": "g",
        "motor_units": "PWM (uint32)",
        "log_blocks": {
            "posvel": {"rate_hz": args.log_rate_hz, "vars": ["stateEstimate.x/y/z", "stateEstimate.vx/vy/vz"]},
            "quat":   {"rate_hz": args.log_rate_hz, "vars": ["stateEstimate.qw/qx/qy/qz"]},
            "imu":    {"rate_hz": args.log_rate_hz, "vars": ["gyro.x/y/z", "acc.x/y/z"]},
            "motor":  {"rate_hz": min(args.log_rate_hz, 50), "vars": ["motor.m1..m4"]},
            "quality":{"rate_hz": 10, "vars": ["kalman.varPX/PY/PZ", "pm.vbat"]},
        },
        "safety": {
            "pos_stale_timeout_s": POS_STALE_TIMEOUT_S,
            "pos_variance_threshold": POS_VARIANCE_THRESHOLD,
            "z_bounds_m": [0.1, 2.5],
        },
        "versions": _versions(),
        "device": str(device),
        "start_wall": time.time(),
        "start_iso": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect sim2real data while flying the velocity hover policy.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the skrl model checkpoint (.pt)")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E8", help="URI of the Crazyflie")
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        help="Fixed target [x y z] world frame. Default: hover above takeoff position.")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Flight duration in seconds before auto-land (protocol trial length). 0 = until Ctrl-C.")
    parser.add_argument("--roam", action="store_true",
                        help="Resample the target when reached (exec_vel.py behaviour). Default: hold fixed target.")
    parser.add_argument("--log-rate-hz", type=int, default=100,
                        help="Rate for the fast log blocks (posvel/quat/imu). Lower to 50 if radio drops packets.")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                        help="Root output directory. A per-run subfolder is created.")
    parser.add_argument("--tag", type=str, default="", help="Optional label added to the run folder name.")
    parser.add_argument("--trial-id", type=str, default=None, help="Optional trial identifier stored in metadata.")
    parser.add_argument("--seed", type=int, default=None, help="Seed recorded in metadata (for reproducibility).")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    duration = None if (args.duration is not None and args.duration <= 0) else args.duration

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_vel"
    if args.tag:
        run_name += f"_{args.tag}"
    run_dir = os.path.join(args.outdir, run_name)

    agent = load_agent(args.checkpoint, device)
    meta = build_metadata(args, args.checkpoint)

    controller = CollectingController(
        uri=args.uri, agent=agent, run_dir=run_dir, meta=meta,
        initial_target=args.target, duration=duration, roam=args.roam,
        fast_rate_hz=args.log_rate_hz)

    logger.info(f"Run directory: {run_dir}")
    try:
        controller.start()
        timeout, elapsed = 10, 0
        while not controller.cf.is_connected() and controller.running:
            time.sleep(1)
            elapsed += 1
            if elapsed >= timeout:
                logger.error(f"Connection timeout after {timeout}s — make sure cfclient is closed and drone is on")
                return
        if not controller.running:
            logger.error("Connection failed — check radio URI and that cfclient is closed")
            return
        logger.info("Crazyflie connected!")
        while controller.running:
            time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        controller.stop()
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
