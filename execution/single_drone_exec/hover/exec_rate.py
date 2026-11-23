import os
import sys
import time
import hashlib
import subprocess
import threading
import argparse
import logging
import math
from typing import Optional

import torch
import torch.nn as nn

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

# Make the `common` package importable; appended so it cannot shadow installed modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.flight_recorder import CrazyflieStateBase
from common.utils import (emergency_land, land_once, quat_apply, quat_inv, wait_until,
                          add_run_args, positive_int, nonnegative_int, resolve_obs_hist)

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

# ── Safety thresholds ────────────────────────────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without a position callback before emergency land
POS_VARIANCE_THRESHOLD = 0.5   # kalman position variance [m²] above which tracking is unreliable

# ── Policy input ─────────────────────────────────────────────────────────────
BASE_OBS_DIM = 18  # lin_vel_b(3), desired_pos_b(3), rotmat(9), ang_vel_b(3), as rate_hovering.py
ACTION_DIM = 4     # roll/pitch/yaw rate, thrust; each in [-1, 1]

# ── Thrust mapping (cflib percent: PWM = 10001 + pct * 499.99) ───────────────
HOVER_THRUST_PCT = 77.0  # measured: steady hover (vz ~ 0, ~50000 PWM) of the 2026-11-23 rate_dr_01 flights

# ── Controllers ──────────────────────────────────────────────────────────────
# rl:  the policy outputs body rates + thrust, tracked by the firmware rate PID.
# pid: the firmware position PID (posCtlPid) stands in for the policy: the stock baseline
#      for the same position -> rate + thrust block. Its outer loops run onboard, so unlike
#      the policy it pays no radio round trip. Both take off with the same high-level-commander takeoff.
# Raw IMU, recorded for analysis: the policy's ang_vel is the gyro of this same block.
IMU_VARS = ["gyro.x", "gyro.y", "gyro.z", "acc.x", "acc.y", "acc.z"]  # deg/s, g
FW_CMD_VARS = ["controller.rollRate", "controller.pitchRate", "controller.yawRate", "controller.cmd_thrust"]
FW_PARAMS = ["stabilizer.controller"] + \
            [f"pid_rate.{axis}_k{term}" for axis in ("roll", "pitch", "yaw") for term in "pid"]  # tracks our rates in rl
PID_PARAMS = [f"posCtlPid.{axis}K{term}" for axis in "xyz" for term in "pi"] + \
             ["posCtlPid.xyVelMax", "posCtlPid.zVelMax"]
EXTRA_FIELDS = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "vbat",
                "var_px", "var_py", "var_pz", "dt_infer_ms", "dt_send_ms", "slept_ms", "overran"]
                
# cflib.Commander.send_setpoint_manual sends thrust = 10001 + pct * (60000 - 10001) / 100.
CFLIB_THRUST_MIN, CFLIB_THRUST_MAX = 10001.0, 60000.0


def thrust_pct_from_cmd_thrust(cmd_thrust: float) -> float:
    """Inverse of cflib's send_setpoint_manual thrust mapping, for the firmware's thrust command."""
    return (cmd_thrust - CFLIB_THRUST_MIN) * 100.0 / (CFLIB_THRUST_MAX - CFLIB_THRUST_MIN)


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


class CrazyflieController(CrazyflieStateBase):
    """Crazyflie controller for body-rate RL agents.

    Uses rate mode to send body rates in deg/s + thrust percentage, or, with
    controller="pid", send_position_setpoint to fly the firmware position PID instead.

    Thrust pipeline:
        norm   = (a[3] + 1) / 2                 # [-1, 1] -> [0, 1]
        norm_hover = (1 - min_scale) / (max_scale - min_scale)
        if norm <= norm_hover:
            pct = lerp(min_pct, hover_pct, norm / norm_hover)
        else:
            pct = lerp(hover_pct, 100, (norm - norm_hover) / (1 - norm_hover))
    """
    def __init__(
        self,
        uri: str,
        agent: PPO,
        run_args,
        initial_target=None,
        min_thrust_scale: float = 0.5,
        max_thrust_scale: float = 1.8,
        hover_thrust_pct: float = HOVER_THRUST_PCT,
        min_thrust_pct: float = 25.0,
        obs_input_hist: int = 1,
        action_input_hist: int = 0,
        controller: str = "rl",
        duration: float = 0.0,
        metadata=None,
    ):
        super().__init__(device)
        self.uri = uri
        self.cf = Crazyflie(rw_cache='./cache')
        self.agent = agent
        self.initial_target = initial_target
        self.k, self.m = obs_input_hist, action_input_hist  # see resolve_obs_hist
        self.controller = controller
        self.duration = duration  # seconds of flight before landing, 0 = until Ctrl-C
        self.min_thrust_scale = float(min_thrust_scale)
        self.max_thrust_scale = float(max_thrust_scale)
        self.hover_thrust_pct = float(hover_thrust_pct)
        self.min_thrust_pct = float(min_thrust_pct)
        # Hover sits at this point in the [0, 1] thrust_norm axis.
        self._hover_norm = (1.0 - self.min_thrust_scale) / (
            self.max_thrust_scale - self.min_thrust_scale
        )
        logger.info(
            f"Thrust mapping: hover_pct={self.hover_thrust_pct:.1f}% "
            f"[min_pct={self.min_thrust_pct:.1f}%, max_pct=100%]"
        )
        self.current_ang_vel = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_imu = [float("nan")] * len(IMU_VARS)
        self.fw_cmd = [float("nan")] * len(FW_CMD_VARS)

        self.make_recorder(run_args,
                           obs_fields=["lin_vel_b_x", "lin_vel_b_y", "lin_vel_b_z",
                                       "des_pos_b_x", "des_pos_b_y", "des_pos_b_z",
                                       "rotmat_0", "rotmat_1", "rotmat_2", "rotmat_3", "rotmat_4",
                                       "rotmat_5", "rotmat_6", "rotmat_7", "rotmat_8",
                                       "ang_vel_x", "ang_vel_y", "ang_vel_z"],
                           action_fields=["action_roll", "action_pitch", "action_yaw", "action_thrust"],
                           cmd_fields=["roll_rate_deg_s", "pitch_rate_deg_s", "yaw_rate_deg_s", "thrust_pct"],
                           extra_fields=EXTRA_FIELDS)
        self.recorder.metadata.update(
            metadata or {}, controller=controller,
            hover_thrust_pct=str(self.hover_thrust_pct),
            min_thrust_pct=str(self.min_thrust_pct),
            min_thrust_scale=str(self.min_thrust_scale), max_thrust_scale=str(self.max_thrust_scale),
        )

        self._setup_callbacks()

    def _thrust_pct_from_action(self, action_thrust: float) -> float:
        """Piecewise-linear map from action[3] in [-1, 1] to thrust pct in [0, 100]."""
        norm = max(0.0, min(1.0, (action_thrust + 1.0) * 0.5))
        if norm <= self._hover_norm:
            t = norm / self._hover_norm if self._hover_norm > 0 else 0.0
            pct = self.min_thrust_pct + t * (self.hover_thrust_pct - self.min_thrust_pct)
        else:
            t = (norm - self._hover_norm) / (1.0 - self._hover_norm)
            pct = self.hover_thrust_pct + t * (100.0 - self.hover_thrust_pct)
        return max(0.0, min(100.0, pct))

    def _setup_callbacks(self):
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

    def _connected(self, uri: str):
        logger.info(f"Connected to {uri}")
        self.setup_state_logging()
        self._start_imu_logging()
        if self.controller == "pid":  # its rate command is computed onboard, so log it back; rl computes its own
            self._add_log_block("controller", FW_CMD_VARS, "fw_cmd")
        threading.Thread(target=self.control_loop, daemon=True).start()

    def _start_imu_logging(self):
        """Gyro and accelerometer every 10 ms. The gyro is the policy's ang_vel observation,
        so unlike an _add_log_block block, a firmware without it fails here."""
        log_imu = LogConfig(name="imu", period_in_ms=10)  # 6 floats = 24 bytes, within the 26-byte limit
        for var in IMU_VARS:
            log_imu.add_variable(var, "float")
        self.cf.log.add_config(log_imu)
        log_imu.data_received_cb.add_callback(self._log_imu_callback)
        log_imu.start()

    def _log_imu_callback(self, timestamp, data, logconf):
        # Crazyflie gyro is in deg/s, body frame. Sim's root_ang_vel_b is rad/s.
        deg2rad = math.pi / 180.0
        with self.lock:
            self.current_imu = [data[var] for var in IMU_VARS]
            self.current_ang_vel = torch.tensor([
                data["gyro.x"] * deg2rad,
                data["gyro.y"] * deg2rad,
                data["gyro.z"] * deg2rad,
            ], dtype=torch.float32, device=device)

    def _add_log_block(self, name, variables, attr):
        """Log `variables` every 10 ms into self.<attr>. The names come from the firmware, not
        from a TOC on disk, so one this firmware lacks leaves its columns NaN instead of aborting."""
        logconf = LogConfig(name=name, period_in_ms=10)
        for var in variables:
            logconf.add_variable(var, "float")
        try:
            self.cf.log.add_config(logconf)
        except (KeyError, AttributeError) as e:
            logger.warning(f"Log block {name} unavailable ({e}); its columns stay NaN")
            return

        def _callback(timestamp, data, logconf):
            with self.lock:
                setattr(self, attr, [data[var] for var in variables])
        logconf.data_received_cb.add_callback(_callback)
        logconf.start()

    def _read_firmware_params(self):
        """Firmware settings that define what flew, for the recording's metadata. Called from the
        control thread: cflib refuses param reads from its callbacks before params are fetched."""
        for name in FW_PARAMS + (PID_PARAMS if self.controller == "pid" else []):
            try:
                self.recorder.metadata[name] = str(self.cf.param.get_value(name))
            except KeyError:
                self.recorder.metadata[name] = "not in firmware"
            except Exception as e:  # params never arrived: each further read would wait 60 s too
                logger.warning(f"Could not read firmware params: {e}")
                break

    def _fw_rate_cmd(self) -> list:
        """The firmware's rate setpoints and thrust, in the units and signs of the rl command
        columns: cflib negates pitch on the way in, so the pitch is negated back."""
        with self.lock:
            roll_rate, pitch_rate, yaw_rate, cmd_thrust = self.fw_cmd
        return [roll_rate, -pitch_rate, yaw_rate, thrust_pct_from_cmd_thrust(cmd_thrust)]

    def _disconnected(self, uri: str):
        pass

    def _connection_failed(self, uri: str, msg: str):
        logger.error(f"Connection to {uri} failed: {msg}")
        self.running = False

    def _connection_lost(self, uri: str, msg: str):
        logger.warning(f"Connection to {uri} lost: {msg} — triggering safe landing")
        self.running = False

    def control_loop(self):
        """Main control loop: high-level-commander takeoff, then NN body-rate control."""
        INTERVAL = 0.01  # 100 Hz control loop
        SPIN_MARGIN = 0.002  # sleep to 2 ms before each deadline, busy-wait the rest (see wait_until)
        # Must match the sim: action[:3] * pi rad/s = action[:3] * 180 deg/s.
        MAX_BODY_RATE_DEG = 180.0

        TAKEOFF_HEIGHT   = 0.5
        TAKEOFF_DURATION = 3.0
        STABILIZE_PAUSE  = 1.0

        self._read_firmware_params()
        self.recorder.start()

        logger.info("Waiting for position data...")
        while not self.position_received and self.running:
            time.sleep(0.1)
        if not self.running:
            return
        logger.info(f"Position received: {self.current_pos}")

        # Enable the high-level commander; the firmware ignores takeoff/land
        # setpoints unless commander.enHighLevel == 1 (default is 0).
        self.cf.param.set_value("commander.enHighLevel", "1")
        time.sleep(0.1)

        # ── Enable rate mode ─────────────────────────────────────────────────
        # stabMode=0 means send_setpoint interprets roll/pitch as body rates
        # in deg/s (not angles). This is required for body-rate control.
        self.cf.param.set_value("flightmode.stabModeRoll", "0")
        self.cf.param.set_value("flightmode.stabModePitch", "0")
        self.cf.param.set_value("flightmode.stabModeYaw", "0")
        logger.info("Rate mode enabled (stabMode=0 for all axes)")

        # Arm the drone; recent firmware (>=2022.12) keeps motors locked until
        # an explicit arming request is sent (cfclient does this automatically).
        self.cf.platform.send_arming_request(True)
        time.sleep(0.5)

        # ── Phase 1: position-controlled takeoff (high-level commander) ──────
        # As traj_tracking/exec_vel.py. The onboard position PID lifts off on its own thrust
        # base, so takeoff does not depend on hover_thrust_pct (a thrust ramp up to hover thrust
        # cannot climb). The stabMode params above only affect low-level RPYT setpoints, so the
        # first send_setpoint_manual in Phase 2 takes over from the HLC in rate mode.
        logger.info(f"Takeoff to {TAKEOFF_HEIGHT} m ...")
        self.cf.high_level_commander.takeoff(TAKEOFF_HEIGHT, TAKEOFF_DURATION)
        time.sleep(TAKEOFF_DURATION + STABILIZE_PAUSE)
        logger.info(f"Takeoff complete. Current pos: {self.current_pos}")

        # Initialize target
        global target_pos
        if self.initial_target is not None:
            target_pos = torch.tensor(self.initial_target, dtype=torch.float32, device=device)
        else:
            target_pos = self.current_pos.clone()
            target_pos[2] = max(0.5, min(1.5, target_pos[2].item()))
        logger.info(f"Init target pos={target_pos}")

        # Obs as in vel_hovering: last K frames (oldest first, primed by repeating the first)
        # + last M actions (zeros at first). K=1, M=0 gives rate_hovering.py's plain 18-D frame.
        window = None
        act_hist = torch.zeros(self.m, ACTION_DIM, device=device)

        # ── Phase 2: NN body-rate control loop ───────────────────────────────
        logger.info("Firmware position PID active." if self.controller == "pid" else "NN body-rate control active.")
        nn_start_time = time.time()
        GRACE_PERIOD = 3.0
        deadline = time.perf_counter()
        step = 0
        while self.cf.is_connected() and self.running:

            # ── Safety watchdog ──────────────────────────────────────────────
            elapsed_since_nn = time.time() - nn_start_time
            if self.duration and elapsed_since_nn >= self.duration:
                logger.info(f"Flight duration {self.duration:g} s reached — landing")
                self.running = False
                break
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

            state = self.snapshot()  # one consistent read, used for the obs and its recorded row
            with self.lock:
                ang_vel = self.current_ang_vel.clone()  # own IMU log block, not in the snapshot
            obs = retrieve_and_create_observation(state.vel, state.pos, state.quat, ang_vel)
            if obs is None:
                logger.warning("No observation received, holding hover thrust...")
                self.cf.commander.send_setpoint_manual(0, 0, 0, self.hover_thrust_pct, True)
                step += 1
                deadline, _, _ = wait_until(deadline + INTERVAL, SPIN_MARGIN)
                continue

            t_infer = time.perf_counter()
            if self.controller == "pid":
                action = torch.full((ACTION_DIM,), float("nan"), device=device)  # no policy action
            else:
                window = obs.expand(self.k, -1) if window is None else torch.cat([window[1:], obs[None]])
                policy_input = torch.cat([window.flatten(), act_hist.flatten()])

                with torch.no_grad():
                    _, outputs = self.agent.act(policy_input.unsqueeze(0), None, timestep=0, timesteps=1)
                    action = outputs["mean_actions"].squeeze(0)  # deterministic mean
                    action = action.clamp(-1.0, 1.0)
                if self.m:  # cat([act_hist[1:], a]) would grow an empty (M=0) history to one row
                    act_hist = torch.cat([act_hist[1:], action[None]])

            t_send = time.perf_counter()
            control_time = time.time()
            if self.controller == "pid":
                self.cf.commander.send_position_setpoint(*target_pos.tolist(), 0.0)  # yaw = 0
                # Firmware's rates + thrust (one log period old), comparable to the rl cmd columns
                cmd = self._fw_rate_cmd()
            else:
                # Body rates: [-1, 1] -> [-MAX_BODY_RATE_DEG, MAX_BODY_RATE_DEG] deg/s.
                roll_rate  = action[0].item() * MAX_BODY_RATE_DEG
                pitch_rate = action[1].item() * MAX_BODY_RATE_DEG
                yaw_rate   = action[2].item() * MAX_BODY_RATE_DEG

                # Thrust: piecewise-linear, anchored so the policy's hover action (a[3] ≈ -0.23) gives hover_thrust_pct
                thrust_pct = self._thrust_pct_from_action(action[3].item())

                # Rate mode (rate=True): roll/pitch/yaw in deg/s, thrust in [0, 100] %.
                self.cf.commander.send_setpoint_manual(roll_rate, pitch_rate, yaw_rate, thrust_pct, True)
                cmd = [roll_rate, pitch_rate, yaw_rate, thrust_pct]
            t_sent = time.perf_counter()

            deadline, slept_ms, overran = wait_until(deadline + INTERVAL, SPIN_MARGIN)

            with self.lock:
                telemetry = self.current_imu + [self.vbat]
            extra = telemetry + var.tolist() + [(t_send - t_infer) * 1e3, (t_sent - t_send) * 1e3,
                                                slept_ms, overran]
            self.record_step(step, state, control_time, target_pos, obs, action, cmd, extra)
            step += 1

        land_once(self)  # at once, never send_stop_setpoint: it cuts the motors mid-air
        logger.info("Control loop stopped")

    def start(self):
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf.open_link(self.uri)

    def stop(self):
        logger.info("Stopping controller...")
        self.running = False
        try:
            time.sleep(0.2)
            logger.info("Landing...")
            land_once(self)  # no-op once the control loop has landed
            time.sleep(2.5)
            self.cf.close_link()
            logger.info("Link closed")
        finally:
            self.recorder.close()  # always write the Parquet footer, or the file is unreadable


def retrieve_and_create_observation(
    current_vel, current_pos, current_quat, current_ang_vel
) -> Optional[torch.Tensor]:
    """One obs frame as in rate_hovering.py: [lin_vel_b(3), desired_pos_b(3), rotmat(9), ang_vel_b(3)]."""
    # Target fixed per flight, like the sim goal per episode.
    if target_pos is None:
        return None
    linear_vel_b = quat_apply(quat_inv(current_quat), current_vel)
    desired_pos_b = quat_apply(quat_inv(current_quat), target_pos - current_pos)
    rot_mat_flat = quat_to_rotmat_flat(current_quat)
    obs = torch.cat([linear_vel_b, desired_pos_b, rot_mat_flat, current_ang_vel], dim=-1)
    return obs


def quat_to_rotmat_flat(quat: torch.Tensor) -> torch.Tensor:
    """Row-major flattened body→world rotation matrix.

    Matches isaaclab.utils.math.matrix_from_quat(quat).reshape(-1, 9) so the
    deployed observation lines up with the one produced in sim.
    quat is (qw, qx, qy, qz).
    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    two_s = 2.0 / (qw * qw + qx * qx + qy * qy + qz * qz)
    return torch.stack([
        1 - two_s * (qy * qy + qz * qz),
        two_s * (qx * qy - qz * qw),
        two_s * (qx * qz + qy * qw),
        two_s * (qx * qy + qz * qw),
        1 - two_s * (qx * qx + qz * qz),
        two_s * (qy * qz - qx * qw),
        two_s * (qx * qz - qy * qw),
        two_s * (qy * qz + qx * qw),
        1 - two_s * (qx * qx + qy * qy),
    ])


def load_agent(checkpoint_path: Optional[str], device: torch.device, obs_space: int = BASE_OBS_DIM) -> PPO:
    act_space = ACTION_DIM
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


def run_metadata(args, k: Optional[int], m: Optional[int]) -> dict:
    """What flew, stored in the recording's Parquet footer so a flight file names its own controller.
    The controller adds its resolved thrust mapping."""
    meta = {"uri": args.uri, "argv": " ".join(sys.argv), "target": str(args.target),
            "duration_s": str(args.duration), "imu_units": "gyro deg/s, acc g"}
    if args.checkpoint:
        with open(args.checkpoint, "rb") as f:
            meta["checkpoint_sha256"] = hashlib.sha256(f.read()).hexdigest()
        meta.update(checkpoint=os.path.abspath(args.checkpoint), obs_input_hist=str(k), action_input_hist=str(m))
    try:
        meta["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        meta["git_commit"] = "unknown"
    return meta


def main():
    parser = argparse.ArgumentParser(description="Run a trained body-rate RL agent, or the firmware "
                                                 "position PID baseline, on a Crazyflie drone.")
    parser.add_argument("--controller", choices=["rl", "pid"], default="rl",
                        help="rl: the policy at --checkpoint. pid: after the same high-level takeoff, the "
                             "firmware position PID flies to --target, as the baseline for the policy (no checkpoint).")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Seconds of controlled flight after takeoff before landing; 0 flies until Ctrl-C.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the policy checkpoint; required with --controller rl (the default)")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E8")
    parser.add_argument("--min-thrust-scale", type=float, default=0.5,
                        help="Lower bound of thrust action, as fraction of hover weight.")
    parser.add_argument("--max-thrust-scale", type=float, default=1.8,
                        help="Upper bound of thrust action, as fraction of hover weight.")
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        help="Initial target [x, y, z] in world frame. If not set, hovers above takeoff pos.")
    parser.add_argument("--hover-thrust-pct", type=float, default=HOVER_THRUST_PCT,
                        help=f"Real-drone hover thrust percentage (0-100), mapped to the policy's hover action. "
                             f"Default {HOVER_THRUST_PCT:g}, measured on this drone.")
    parser.add_argument("--min-thrust-pct", type=float, default=25.0,
                        help="Thrust pct corresponding to action[3] = -1.")
    parser.add_argument("--obs-input-hist", type=positive_int, default=None,
                        help="Observation frames K the policy was trained with. Read from <name>_params/env.yaml "
                             "beside an exported <name>.pt, or the run's params/env.yaml; only needed "
                             "for a checkpoint without either.")
    parser.add_argument("--action-input-hist", type=nonnegative_int, default=None,
                        help="Past commanded actions M the policy was trained with. Same rule as K.")
    add_run_args(parser)
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("--duration must be >= 0")

    if args.controller == "rl":
        if not args.checkpoint:
            parser.error("--controller rl needs --checkpoint, the trained policy to fly")
        try:
            k, m = resolve_obs_hist(args.checkpoint, args.obs_input_hist, args.action_input_hist)
        except ValueError as e:
            parser.error(str(e))
        agent = load_agent(args.checkpoint, device, BASE_OBS_DIM * k + ACTION_DIM * m)
    elif args.controller == "pid":
        if args.checkpoint:
            parser.error("--checkpoint is for --controller rl; pid flies the firmware position PID")
        agent, k, m = None, None, None
    else:
        parser.error(f"unknown controller {args.controller!r}")

    controller = CrazyflieController(
        uri=args.uri,
        agent=agent,
        run_args=args,
        initial_target=args.target,
        min_thrust_scale=args.min_thrust_scale,
        max_thrust_scale=args.max_thrust_scale,
        hover_thrust_pct=args.hover_thrust_pct,
        min_thrust_pct=args.min_thrust_pct,
        obs_input_hist=k or 1,
        action_input_hist=m or 0,
        controller=args.controller,
        duration=args.duration,
        metadata=run_metadata(args, k, m),
    )
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
