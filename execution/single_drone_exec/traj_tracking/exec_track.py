"""Trajectory tracking on a real Crazyflie, for every TrackingEnv control mode.

One script, as there is one training env: the run's env.yaml names the control mode
(TrackingEnvCfg.control_mode) and its action scales, and a Command turns the policy's action
into the cflib setpoint the firmware tracks with the same PID level the sim models:

    position   Pos-Tracking   send_position_setpoint(pos + a * max_position_delta)
    velocity   Vel-Tracking   send_velocity_world_setpoint(a * max_velocity)
    attitude   Att-Tracking   send_setpoint_manual(roll, pitch, yaw rate, thrust %, rate=False)
    body_rate  Rate-Tracking  send_setpoint_manual(roll, pitch, yaw rates, thrust %, rate=True)

Everything else is shared by the four modes: the HLC takeoff and go_to the reference's start,
the reference at a constant linear speed (#96), the 22-D observation, the K-frame / M-action
history, the watchdog, the fence, and the recording. The low-level modes (attitude, body_rate)
add a thrust map, a level setpoint before every hand-over, and tilt / body-rate / tracking-error
checks, since the firmware no longer caps the tilt for them.
"""

import os
import sys
import math
import time
import hashlib
import subprocess
import threading
import argparse
import logging
from typing import List, Optional

import torch
import torch.nn as nn

import cflib.crtp
from cflib.crazyflie import Crazyflie

from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler

# Make the `common` package importable; appended so it cannot shadow installed modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.flight_recorder import CrazyflieStateBase
from common.tracking import (CONTROL_DT, OBS_FIELDS, OBS_STEP_DIM, TRAJ_ORIGIN, Reference, add_trajectory_args,
                             build_observation, go_to_trajectory_start, reference_from_args, take_off)
from common.tracking import GOTO_MIN_DURATION_S, GOTO_SPEED, REF_FENCE_MARGIN, TRAJ_SCALE  # noqa: F401 (for tests)
from common.trajectories import TRAJECTORIES  # noqa: F401 (for tests)
from common.utils import (ARENA_FENCE, add_run_args, check_env_contract, emergency_land, fly_back_and_land,
                          land_once, nonnegative_int, positive_int, read_env_cfg, resolve_obs_hist, wait_until)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(format="{asctime} [{levelname}] {message}",
                    style="{", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("CrazyflieRL")

# ── Safety thresholds ──────────────────────────────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without position callback
POS_VARIANCE_THRESHOLD = 0.5   # kalman variance [m²]
# Low-level modes only (attitude, body_rate): the firmware tracks whatever angle or rate is sent.
MAX_TILT_DEG           = 45.0  # past acos(1 / 1.5387) = 49.5°, full thrust no longer holds altitude
MAX_BODY_RATE_DEG_S    = 270.0 # measured gyro; the rate command is capped at max_body_rate (180°/s)
BODY_RATE_STEPS        = 3     # consecutive steps above MAX_BODY_RATE_DEG_S
MAX_TRACK_ERR_M        = 0.6   # TrackingEnv ends an episode at reset_thres = 0.8 m
TRACK_ERR_STEPS        = 10    # consecutive steps above MAX_TRACK_ERR_M

# ── Loop timing ────────────────────────────────────────────────────────────────
SPIN_MARGIN = 0.002  # sleep to 2 ms before each deadline, busy-wait the rest (see wait_until)

# ── Action scales: TrackingEnvCfg defaults, used when the run's env.yaml lacks them ────
DEFAULT_SCALES = {
    "max_position_delta": 0.5,             # m          PosTrackingEnvCfg
    "max_velocity": 3.0,                   # m/s        VelTrackingEnvCfg
    "max_roll_pitch": math.radians(30.0),  # rad        AttTrackingEnvCfg
    "max_yaw_rate": math.radians(90.0),    # rad/s      AttTrackingEnvCfg
    "max_body_rate": math.pi,              # rad/s      RateTrackingEnvCfg
    "min_thrust_scale": 0.5,               # x hover    Att/RateTrackingEnvCfg
}
# max_thrust_scale: None in env.yaml means drone.max_thrust / (mass g) of the run's drone yaml
# (controllers/config.py:resolve_max_thrust_scale): 0.48 N / (0.0318 kg x 9.81) for the CF2.1.
FIRMWARE_MAX_THRUST_SCALE = 0.48 / (0.0318 * 9.81)

# ── Thrust mapping (cflib percent: PWM = 10001 + pct * 499.99) ─────────────────
HOVER_THRUST_PCT = 77.0  # measured: steady hover (vz ~ 0, ~50000 PWM) of the 2026-11-23 rate_dr_01 flights
MIN_THRUST_PCT   = 25.0  # thrust % at the bottom of the action range

# ── Telemetry beyond the state blocks of CrazyflieStateBase ────────────────────
# gyro.* feeds ang_vel_b, so the block is required; acc.* is recorded for analysis only.
IMU_VARS = ["gyro.x", "gyro.y", "gyro.z", "acc.x", "acc.y", "acc.z"]  # deg/s, g
FW_PARAMS = ["stabilizer.controller"]  # onboard controller tracking our setpoints
EXTRA_FIELDS = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "vbat",
                "var_px", "var_py", "var_pz", "track_err_m", "traj_dist_m", "tilt_deg",
                "dt_infer_ms", "dt_send_ms", "slept_ms", "overran"]


# ── Commands: policy action -> cflib setpoint, one per TrackingEnvCfg.control_mode ─────

class ThrustMap:
    """Thrust normalised as the env does it (0: min_thrust_scale, 1: max_thrust_scale x hover) -> cflib %.

    Piecewise linear, anchored at the measured hover thrust instead of a thrust model: the
    policy's hover command, norm = (1 - min_scale) / (max_scale - min_scale), maps to
    hover_pct, the bottom of the range to min_pct and the top to 100 %.
    """

    def __init__(self, min_scale: float, max_scale: float, hover_pct: float = HOVER_THRUST_PCT,
                 min_pct: float = MIN_THRUST_PCT):
        if not 0.0 < min_scale < 1.0 < max_scale:
            raise ValueError(f"need min_thrust_scale < 1 < max_thrust_scale, got {min_scale}, {max_scale}")
        if not 0.0 <= min_pct < hover_pct < 100.0:
            raise ValueError(f"need 0 <= min_thrust_pct < hover_thrust_pct < 100, got {min_pct}, {hover_pct}")
        self.min_scale, self.max_scale = float(min_scale), float(max_scale)
        self.hover_pct, self.min_pct = float(hover_pct), float(min_pct)
        self.hover_norm = (1.0 - self.min_scale) / (self.max_scale - self.min_scale)

    def pct(self, norm: float) -> float:
        norm = max(0.0, min(1.0, norm))
        if norm <= self.hover_norm:
            return self.min_pct + norm / self.hover_norm * (self.hover_pct - self.min_pct)
        return self.hover_pct + (norm - self.hover_norm) / (1.0 - self.hover_norm) * (100.0 - self.hover_pct)

    def metadata(self) -> dict:
        return {"min_thrust_scale": str(self.min_scale), "max_thrust_scale": str(self.max_scale),
                "hover_thrust_pct": str(self.hover_pct), "min_thrust_pct": str(self.min_pct)}


class Command:
    """How one control mode sends the policy's action. Subclasses set the class attributes."""
    mode: str = ""
    action_fields: List[str] = []
    cmd_fields: List[str] = []
    low_level = False  # True: the firmware tracks raw angles or rates, with no position loop

    @property
    def act_dim(self) -> int:
        return len(self.action_fields)

    def hover_action(self) -> List[float]:
        """TrackingEnv._hover_cmd: the action that holds the drone, filling the action history at reset."""
        return [0.0] * self.act_dim

    def enter(self, cf) -> None:
        """Firmware settings before the takeoff."""

    def exit(self, cf) -> None:
        """Undo `enter`, once the drone has landed."""

    def level(self, cf) -> None:
        """Before every hand-over to the high-level commander."""

    def send(self, cf, action: torch.Tensor, pos_w: torch.Tensor) -> List[float]:
        """Send `action` (clamped to [-1, 1]); returns the command sent, in cmd_fields units."""
        raise NotImplementedError

    def metadata(self) -> dict:
        return {"control_mode": self.mode}


class PositionCommand(Command):
    mode = "position"
    action_fields = ["action_dx", "action_dy", "action_dz"]
    cmd_fields = ["target_pos_x", "target_pos_y", "target_pos_z"]

    def __init__(self, max_position_delta: float):
        self.max_position_delta = float(max_position_delta)

    def send(self, cf, action, pos_w):
        target_w = pos_w + action * self.max_position_delta  # TrackingEnv: root_pos_w + a * max_position_delta
        cf.commander.send_position_setpoint(*target_w.tolist(), 0.0)  # yaw 0
        return target_w.tolist()

    def metadata(self):
        return dict(super().metadata(), max_position_delta_m=str(self.max_position_delta))


class VelocityCommand(Command):
    mode = "velocity"
    action_fields = ["action_vx", "action_vy", "action_vz"]
    cmd_fields = ["vx", "vy", "vz"]

    def __init__(self, max_velocity: float):
        self.max_velocity = float(max_velocity)

    def send(self, cf, action, pos_w):
        vel_cmd = (action * self.max_velocity).tolist()
        cf.commander.send_velocity_world_setpoint(*vel_cmd, 0.0)  # yaw rate 0
        return vel_cmd

    def metadata(self):
        return dict(super().metadata(), max_velocity_m_s=str(self.max_velocity))


class AttitudeCommand(Command):
    mode = "attitude"
    action_fields = ["action_roll", "action_pitch", "action_yaw_rate", "action_thrust"]
    cmd_fields = ["roll_deg", "pitch_deg", "yaw_rate_deg_s", "thrust_pct"]
    low_level = True

    def __init__(self, max_roll_pitch: float, max_yaw_rate: float, thrust: ThrustMap):
        self.max_roll_pitch_deg = math.degrees(max_roll_pitch)
        self.max_yaw_rate_deg_s = math.degrees(max_yaw_rate)
        self.thrust = thrust

    def hover_action(self):
        return [0.0, 0.0, 0.0, self.thrust.hover_norm]  # TrackingEnv: (T_hover - T_min) / (T_max - T_min)

    def level(self, cf):
        cf.commander.send_setpoint_manual(0.0, 0.0, 0.0, self.thrust.hover_pct, False)

    def send(self, cf, action, pos_w):
        roll_deg = action[0].item() * self.max_roll_pitch_deg
        pitch_deg = action[1].item() * self.max_roll_pitch_deg
        yaw_rate_deg_s = action[2].item() * self.max_yaw_rate_deg_s
        thrust_pct = self.thrust.pct(action[3].item())  # TrackingEnv: a[3].clamp(0, 1), not (a + 1) / 2
        cf.commander.send_setpoint_manual(roll_deg, pitch_deg, yaw_rate_deg_s, thrust_pct, False)
        return [roll_deg, pitch_deg, yaw_rate_deg_s, thrust_pct]

    def metadata(self):
        return dict(super().metadata(), **self.thrust.metadata(), max_roll_pitch_deg=str(self.max_roll_pitch_deg),
                    max_yaw_rate_deg_s=str(self.max_yaw_rate_deg_s))


class RateCommand(Command):
    mode = "body_rate"
    action_fields = ["action_roll_rate", "action_pitch_rate", "action_yaw_rate", "action_thrust"]
    cmd_fields = ["roll_rate_deg_s", "pitch_rate_deg_s", "yaw_rate_deg_s", "thrust_pct"]
    low_level = True
    STAB_MODE_PARAMS = ["flightmode.stabModeRoll", "flightmode.stabModePitch", "flightmode.stabModeYaw"]

    def __init__(self, max_body_rate: float, thrust: ThrustMap):
        self.max_body_rate_deg_s = math.degrees(max_body_rate)
        self.thrust = thrust
        self._stab_modes = {}  # param -> value before enter()

    def hover_action(self):
        return [0.0, 0.0, 0.0, 2.0 * self.thrust.hover_norm - 1.0]  # inverse of norm = (a + 1) / 2

    def enter(self, cf):
        # stabMode = 0: the low-level roll/pitch/yaw setpoints are body rates (as hover/exec_rate.py)
        for name in self.STAB_MODE_PARAMS:
            try:
                self._stab_modes[name] = str(cf.param.get_value(name))
            except Exception as e:  # restored to the firmware default (angle mode) if unknown
                logger.warning(f"Could not read {name} ({e})")
            cf.param.set_value(name, "0")
        logger.info("Rate mode enabled (stabMode=0 for all axes)")

    def exit(self, cf):
        for name, value in self._stab_modes.items():
            cf.param.set_value(name, value)

    def level(self, cf):
        cf.commander.send_setpoint_manual(0.0, 0.0, 0.0, self.thrust.hover_pct, False)  # angle mode: level

    def send(self, cf, action, pos_w):
        roll_rate, pitch_rate, yaw_rate = (action[:3] * self.max_body_rate_deg_s).tolist()
        thrust_pct = self.thrust.pct((action[3].item() + 1.0) * 0.5)  # TrackingEnv: [-1, 1] -> [0, 1]
        cf.commander.send_setpoint_manual(roll_rate, pitch_rate, yaw_rate, thrust_pct, True)
        return [roll_rate, pitch_rate, yaw_rate, thrust_pct]

    def metadata(self):
        return dict(super().metadata(), **self.thrust.metadata(), max_body_rate_deg_s=str(self.max_body_rate_deg_s))


COMMANDS = {cls.mode: cls for cls in (PositionCommand, VelocityCommand, AttitudeCommand, RateCommand)}


def _scale(env_cfg: dict, name: str):
    value = env_cfg.get(name)
    if value is None:
        logger.warning(f"No {name} in the run's env.yaml: using the TrackingEnvCfg default {DEFAULT_SCALES[name]:.4g}")
        return DEFAULT_SCALES[name]
    return value


def max_thrust_scale(env_cfg: dict) -> tuple:
    """(max_thrust_scale, where it came from), as TrackingEnv resolves it.

    A null in env.yaml means max_thrust / (mass g) of the run's drone yaml. Its path is the one
    of the training machine; when it is not readable here, the CF2.1 value is used.
    """
    if env_cfg.get("max_thrust_scale") is not None:
        return float(env_cfg["max_thrust_scale"]), "env.yaml"
    path = env_cfg.get("drone_config_path")
    if path and os.path.exists(path):
        import yaml  # only here: the drone yaml is plain YAML, unlike env.yaml
        with open(path) as f:
            drone = yaml.safe_load(f)["drone"]
        return float(drone["max_thrust"]) / (float(drone["mass"]) * 9.81), path
    logger.warning(f"max_thrust_scale null and the run's drone yaml {path!r} is not readable here: "
                   f"using the CF2.1 firmware cap {FIRMWARE_MAX_THRUST_SCALE:.4f}")
    return FIRMWARE_MAX_THRUST_SCALE, "CF2.1 firmware cap"


def make_command(mode: str, env_cfg: dict, hover_thrust_pct: float = HOVER_THRUST_PCT,
                 min_thrust_pct: float = MIN_THRUST_PCT) -> Command:
    """The Command for `mode`, with the action scales of the training run's env.yaml."""
    if mode == "position":
        return PositionCommand(_scale(env_cfg, "max_position_delta"))
    if mode == "velocity":
        return VelocityCommand(_scale(env_cfg, "max_velocity"))
    scale, source = max_thrust_scale(env_cfg)
    thrust = ThrustMap(_scale(env_cfg, "min_thrust_scale"), scale, hover_thrust_pct, min_thrust_pct)
    logger.info(f"Thrust map: scales {thrust.min_scale:g}-{thrust.max_scale:.4f} x hover ({source}), "
                f"hover command {thrust.hover_norm:.3f} -> {thrust.hover_pct:g} %")
    if mode == "attitude":
        return AttitudeCommand(_scale(env_cfg, "max_roll_pitch"), _scale(env_cfg, "max_yaw_rate"), thrust)
    if mode == "body_rate":
        return RateCommand(_scale(env_cfg, "max_body_rate"), thrust)
    raise ValueError(f"unknown control mode {mode!r}; known: {sorted(COMMANDS)}")


# ── Model definition ───────────────────────────────────────────────────────────

class Policy(GaussianMixin, Model):
    """Must match the track skrl_ppo_cfg.yaml network, the same for every mode: [256, 128, 64] with ELU."""

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0, initial_log_std=0.0):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        self.policy_layer = nn.Linear(64, self.num_actions)
        self.value_layer  = nn.Linear(64, 1)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * initial_log_std)

    def compute(self, inputs, role):
        x = self.net_container(inputs["observations"])
        mean = self.policy_layer(x) if role == "policy" else self.value_layer(x)
        return mean, {"log_std": self.log_std_parameter}


# ── Crazyflie controller ───────────────────────────────────────────────────────

def tilt_deg(quat_wxyz: torch.Tensor) -> float:
    """Angle between the body z axis and the world z axis, degrees."""
    w, x, y, z = quat_wxyz.tolist()
    cos_tilt = (1.0 - 2.0 * (x * x + y * y)) / max(w * w + x * x + y * y + z * z, 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt))))


class CrazyflieController(CrazyflieStateBase):
    """Runs a trained TrackingEnv policy of any control mode on the Crazyflie.

    Args:
        uri: radio URI.
        agent: the loaded policy.
        command: how the action is sent (make_command).
        run_args: recording flags (utils.add_run_args); None records nothing.
        reference: the lap to fly; default Reference().
        obs_input_hist, action_input_hist: K and M of the run.
        takeoff_height: HLC takeoff height, m; default the trajectory's z.
        duration: seconds of policy flight before landing, 0 = until Ctrl-C.
        metadata: str -> str for the recording's footer.
    """

    def __init__(self, uri: str, agent: PPO, command: Command, run_args=None, reference: Optional[Reference] = None,
                 obs_input_hist: int = 1, action_input_hist: int = 0, takeoff_height: Optional[float] = None,
                 duration: float = 0.0, metadata=None):
        super().__init__(device)
        self.reference = reference if reference is not None else Reference(device=device)
        error = self.reference.limit_check()
        if error:
            raise ValueError(error)
        self.uri   = uri
        self.cf    = Crazyflie()
        self.agent = agent
        self.command = command
        self.k, self.m = obs_input_hist, action_input_hist  # frames K, past actions M (see resolve_obs_hist)
        self.takeoff_height = float(TRAJ_ORIGIN[2]) if takeoff_height is None else float(takeoff_height)
        self.duration = duration
        self.current_imu = [float("nan")] * len(IMU_VARS)
        self.loop_thread: Optional[threading.Thread] = None  # the control loop, once connected
        self.policy_started = False  # True from the policy's first setpoint
        self.stop_reason = ""

        if run_args is None:  # no recording, status line once per second
            run_args = argparse.Namespace(record_path=None, record_every=1, log_every=100)
        self.make_recorder(run_args, obs_fields=OBS_FIELDS, action_fields=command.action_fields,
                           cmd_fields=command.cmd_fields, extra_fields=EXTRA_FIELDS)
        self.recorder.metadata.update(metadata or {}, **self.reference.metadata(), **command.metadata())
        self._setup_callbacks()

    def _setup_callbacks(self):
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

    def _connected(self, uri: str):
        logger.info(f"Connected to {uri}")
        self.setup_state_logging()  # posvel, quat and motor every 10 ms; variance and battery every 200 ms
        if not self.add_log_block("imu", IMU_VARS, "current_imu"):
            logger.error("No gyro log block: the policy cannot see its body rates — not flying")
            self.running = False
            return
        self.loop_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.loop_thread.start()

    def _disconnected(self, uri: str):
        pass

    def _connection_failed(self, uri: str, msg: str):
        logger.error(f"Connection to {uri} failed: {msg}")
        self.running = False

    def _connection_lost(self, uri: str, msg: str):
        logger.warning(f"Connection to {uri} lost: {msg} — triggering safe landing")
        self.running = False

    def before_hand_over(self):
        """Called by utils.land_once and fly_back_and_land before the high-level commander takes over.

        Only once the policy has sent a setpoint: before that the HLC flies, and a level
        hover-thrust setpoint on the ground would spin the motors up.
        """
        if self.policy_started:
            self.command.level(self.cf)

    def _read_firmware_params(self):
        """Firmware settings that define what flew, for the recording's metadata. Called from the
        control thread: cflib refuses param reads from its callbacks before params are fetched."""
        for name in FW_PARAMS:
            try:
                self.recorder.metadata[name] = str(self.cf.param.get_value(name))
            except KeyError:
                self.recorder.metadata[name] = "not in firmware"
            except Exception as e:  # params never arrived: each further read would wait 60 s too
                logger.warning(f"Could not read firmware params: {e}")
                break

    def _recover(self, reason: str, pos: torch.Tensor):
        """Leave the policy where it is: level, brake at `pos`, fly back to the safe hover point, land."""
        self.stop_reason = reason
        fly_back_and_land(self, ARENA_FENCE, reason, pos, POS_STALE_TIMEOUT_S)

    def control_loop(self):
        """Phase 1: HLC takeoff, then go_to the trajectory start. Phase 2: the policy at 100 Hz."""
        self._read_firmware_params()
        self.recorder.start()

        logger.info("Waiting for position data...")
        while not self.position_received and self.running:
            time.sleep(0.1)
        if not self.running:
            return
        logger.info(f"Position received: {self.current_pos}")

        # ── Phase 1: HLC takeoff and go_to the trajectory start ───────────────
        self.command.enter(self.cf)
        if not (take_off(self, self.takeoff_height, POS_STALE_TIMEOUT_S)
                and go_to_trajectory_start(self, self.reference.start(), POS_STALE_TIMEOUT_S)):
            land_once(self)  # no-op if the wait already landed
            logger.info("Control loop stopped before the policy")
            return

        # ── Phase 2: the policy ───────────────────────────────────────────────
        ref, command = self.reference, self.command
        logger.info(f"NN {command.mode} tracking active ({ref.traj_type}, scale {ref.scale} m, {ref.speed:g} m/s).")
        # As TrackingEnv: last K frames (primed with copies of the first), then last M clamped
        # actions, primed with the mode's hover command (_hover_cmd)
        window = None
        act_hist = torch.tensor([command.hover_action()] * self.m, dtype=torch.float32,
                                device=device).reshape(self.m, command.act_dim)
        high_rate_steps = high_err_steps = 0
        nn_start_time = time.time()
        deadline = time.perf_counter()
        step = 0
        while self.cf.is_connected() and self.running:
            # Safety watchdog
            if self.duration and time.time() - nn_start_time >= self.duration:
                logger.info(f"Flight duration {self.duration:g} s reached — landing")
                self.stop_reason = "duration"
                self.running = False
                break
            if self._last_pos_time > 0 and time.time() - self._last_pos_time > POS_STALE_TIMEOUT_S:
                logger.error(
                    f"Position stale ({time.time() - self._last_pos_time:.2f} s) — emergency landing"
                )
                self.stop_reason = "position stale"
                emergency_land(self)  # hands over from the low-level setpoints first, or land() is ignored
                break
            with self.lock:
                var = self._pos_variance.clone()
                telemetry = list(self.current_imu) + [self.vbat]
            if var.max().item() > POS_VARIANCE_THRESHOLD:
                logger.error(f"Position variance too high {var.tolist()} — emergency landing")
                self.stop_reason = "position variance"
                emergency_land(self)
                break

            state = self.snapshot()  # one consistent read, used for the obs and its recorded row
            breach = ARENA_FENCE.breach(state.pos, state.vel)
            if breach:
                self.stop_reason = "fence"
                fly_back_and_land(self, ARENA_FENCE, breach, state.pos, POS_STALE_TIMEOUT_S)
                break

            waypoints_w = ref.waypoints(step)
            target = waypoints_w[0]  # the point TrackingEnv scores this step against
            track_err = (state.pos - target).norm().item()
            tilt = tilt_deg(state.quat)
            if command.low_level:
                high_rate_steps = high_rate_steps + 1 if max(map(abs, telemetry[:3])) > MAX_BODY_RATE_DEG_S else 0
                high_err_steps = high_err_steps + 1 if track_err > MAX_TRACK_ERR_M else 0
                reason = (f"Tilt {tilt:.0f}° above {MAX_TILT_DEG:g}°" if tilt > MAX_TILT_DEG else
                          f"Body rate above {MAX_BODY_RATE_DEG_S:g}°/s for {BODY_RATE_STEPS} steps"
                          if high_rate_steps >= BODY_RATE_STEPS else
                          f"Tracking error above {MAX_TRACK_ERR_M:g} m for {TRACK_ERR_STEPS} steps"
                          if high_err_steps >= TRACK_ERR_STEPS else None)
                if reason:
                    self._recover(reason, state.pos)
                    break

            # Crazyflie gyro logs in deg/s; simulation uses rad/s
            ang_vel_b = torch.tensor(telemetry[:3], dtype=torch.float32, device=device) * (math.pi / 180.0)
            obs = build_observation(state.pos, state.vel, ang_vel_b, state.quat, waypoints_w)
            window = obs.expand(self.k, -1) if window is None else torch.cat([window[1:], obs[None]])
            policy_input = torch.cat([window.flatten(), act_hist.flatten()])

            t_infer = time.perf_counter()
            with torch.no_grad():
                _, outputs = self.agent.act(policy_input.unsqueeze(0), None, timestep=0, timesteps=1)
                action = outputs["mean_actions"].squeeze(0)
                action = action.clamp(-1.0, 1.0)
            if self.m:  # cat([act_hist[1:], a]) would grow an empty (M=0) history to one row
                act_hist = torch.cat([act_hist[1:], action[None]])

            t_send = time.perf_counter()
            control_time = time.time()
            with self.lock:  # land_once sets `landing` under this lock: no setpoint may follow its hand-over
                if self.landing:
                    break
                cmd = command.send(self.cf, action, state.pos)
                self.policy_started = True
            t_sent = time.perf_counter()

            deadline, slept_ms, overran = wait_until(deadline + CONTROL_DT, SPIN_MARGIN)

            extra = telemetry + var.tolist() + [track_err, ref.distance(step), tilt,
                                                (t_send - t_infer) * 1e3, (t_sent - t_send) * 1e3,
                                                slept_ms, overran]
            self.record_step(step, state, control_time, target, obs, action, cmd, extra)
            step += 1

        land_once(self)  # at once, never send_stop_setpoint: it cuts the motors mid-air
        logger.info(f"Control loop stopped{f' ({self.stop_reason})' if self.stop_reason else ''}")

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
            self.command.exit(self.cf)
            self.cf.close_link()
            logger.info("Link closed")
        finally:
            self.recorder.close()  # always write the Parquet footer, or the file is unreadable


# ── Agent loading / main ───────────────────────────────────────────────────────

def load_agent(checkpoint_path: Optional[str], device: torch.device, obs_space: int, act_space: int) -> PPO:
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


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_metadata(args, k: int, m: int, env_cfg: dict) -> dict:
    """What flew, stored in the recording's Parquet footer so a flight file names its own policy."""
    meta = {"uri": args.uri, "argv": " ".join(sys.argv), "duration_s": str(args.duration),
            "controller": "rl", "imu_units": "gyro deg/s, acc g", "obs_input_hist": str(k),
            "action_input_hist": str(m), "fence_lo_m": str(ARENA_FENCE.lo), "fence_hi_m": str(ARENA_FENCE.hi)}
    if args.checkpoint and os.path.exists(args.checkpoint):
        meta.update(checkpoint=os.path.abspath(args.checkpoint), checkpoint_sha256=_sha256(args.checkpoint))
    if env_cfg.get("_path"):
        meta.update(env_yaml=env_cfg["_path"], env_yaml_sha256=_sha256(env_cfg["_path"]))
    try:
        meta["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        meta["git_commit"] = "unknown"
    return meta


def resolve_control_mode(env_cfg: dict, flag: Optional[str]) -> str:
    """The run's control_mode; --control-mode only for a checkpoint without env.yaml, and must agree."""
    trained = env_cfg.get("control_mode")
    if trained is None:
        if flag is None:
            raise ValueError("no control_mode in the run's env.yaml: pass --control-mode")
        return flag
    if flag is not None and flag != trained:
        raise ValueError(f"--control-mode {flag} disagrees with the training run's env.yaml ({trained})")
    return trained


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory tracking with a TrackingEnv policy of any control mode (Pos/Vel/Att/Rate-Tracking)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--uri", type=str, default="radio://0/40/2M/E7E7E7E7E8",
                        help="Crazyflie radio URI")
    parser.add_argument("--control-mode", choices=sorted(COMMANDS), default=None,
                        help="TrackingEnvCfg.control_mode of the policy. Read from the run's env.yaml; only "
                             "needed for a checkpoint without one.")
    parser.add_argument("--obs-input-hist", type=positive_int, default=None,
                        help="Observation frames K the policy was trained with. Read from <name>_params/env.yaml "
                             "beside an exported <name>.pt, or the run's params/env.yaml; only needed "
                             "for a checkpoint without either.")
    parser.add_argument("--action-input-hist", type=nonnegative_int, default=None,
                        help="Past commanded actions M the policy was trained with. Same rule as K.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Seconds of policy flight before landing; 0 flies until Ctrl-C.")
    parser.add_argument("--hover-thrust-pct", type=float, default=HOVER_THRUST_PCT,
                        help=f"attitude / body_rate: cflib thrust %% that holds a hover (default {HOVER_THRUST_PCT:g}, "
                             "measured); the policy's hover command maps to it.")
    parser.add_argument("--min-thrust-pct", type=float, default=MIN_THRUST_PCT,
                        help=f"attitude / body_rate: thrust %% at the bottom of the action range (default {MIN_THRUST_PCT:g}).")
    add_trajectory_args(parser)
    add_run_args(parser)
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("--duration must be >= 0")

    env_cfg = read_env_cfg(args.checkpoint)
    try:
        mode = resolve_control_mode(env_cfg, args.control_mode)
        check_env_contract(env_cfg, mode, CONTROL_DT)
        command = make_command(mode, env_cfg, args.hover_thrust_pct, args.min_thrust_pct)
        k, m = resolve_obs_hist(args.checkpoint, args.obs_input_hist, args.action_input_hist)
    except ValueError as e:
        parser.error(str(e))
    reference = reference_from_args(args, env_cfg, parser, device)

    agent = load_agent(args.checkpoint, device, OBS_STEP_DIM * k + command.act_dim * m, command.act_dim)
    controller = CrazyflieController(uri=args.uri, agent=agent, command=command, run_args=args, reference=reference,
                                     obs_input_hist=k, action_input_hist=m, takeoff_height=args.takeoff_height,
                                     duration=args.duration, metadata=run_metadata(args, k, m, env_cfg))

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
        logger.info("Crazyflie connected!")
        while controller.running:
            time.sleep(1)
        # running goes False as a fly-back starts: let it finish before stop() lands in place.
        # Ctrl-C during the wait still lands at once.
        while controller.loop_thread is not None and controller.loop_thread.is_alive():
            controller.loop_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        controller.stop()
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
