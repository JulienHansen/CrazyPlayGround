"""
YAML configuration loader for drone_control.

Config files follow the schema defined by the dataclasses below.
Limits on PID outputs are assumed symmetric (±limit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# PID gain block
# ---------------------------------------------------------------------------

@dataclass
class PIDConfig:
    """Gains and output saturation for a single PID axis."""
    kp: float
    ki: float
    kd: float
    tau: float   # Derivative low-pass time constant [s]
    limit: float  # Symmetric output saturation: output ∈ [-limit, +limit]


# ---------------------------------------------------------------------------
# Attitude controller config
# ---------------------------------------------------------------------------

@dataclass
class AttitudeRateConfig:
    """Rate (inner) loop — error in [rad/s], output in [rad/s²]."""
    roll:  PIDConfig
    pitch: PIDConfig
    yaw:   PIDConfig


@dataclass
class AttitudeAngleConfig:
    """Angle (outer) loop — error in [rad], output in [rad/s]."""
    roll:  PIDConfig
    pitch: PIDConfig
    yaw:   PIDConfig


@dataclass
class AttitudeControllerConfig:
    freq_rate_hz:  float  # Inner loop frequency [Hz]
    freq_angle_hz: float  # Outer loop frequency [Hz]
    rate:  AttitudeRateConfig
    angle: AttitudeAngleConfig


# ---------------------------------------------------------------------------
# Position controller config
# ---------------------------------------------------------------------------

@dataclass
class PositionVelConfig:
    """Velocity (inner) loop — error in [m/s], output in [m/s²] (acceleration)."""
    vx: PIDConfig
    vy: PIDConfig
    vz: PIDConfig


@dataclass
class PositionPosConfig:
    """Position (outer) loop — error in [m], output in [m/s]."""
    x: PIDConfig
    y: PIDConfig
    z: PIDConfig


@dataclass
class PositionControllerConfig:
    freq_vel_hz: float  # Inner loop frequency [Hz]
    freq_pos_hz: float  # Outer loop frequency [Hz]
    max_horizontal_angle_deg: float  # Roll/pitch saturation when chasing velocity [deg]
    max_thrust_scale: float          # Safety factor on drone max thrust (0–1)
    velocity: PositionVelConfig
    position: PositionPosConfig


# ---------------------------------------------------------------------------
# Drone physics
# ---------------------------------------------------------------------------

@dataclass
class InertiaConfig:
    """Principal moments of inertia [kg·m²], assuming diagonal inertia tensor."""
    ixx: float
    iyy: float
    izz: float


@dataclass
class MotorConfig:
    """Motor and frame geometry for the QuadMixer."""
    arm_length: float   # m    — center-to-motor distance
    k_thrust:   float   # N·s²  — thrust coeff  (F = k_thrust · ω²)
    k_drag:     float   # N·m·s² — drag coeff   (τ = k_drag   · ω²)
    layout:     str     # 'x' or '+' quad configuration
    speed_min:  float   # rad/s — minimum motor speed (clamped)
    speed_max:  float   # rad/s — maximum motor speed (clamped)


@dataclass
class DronePhysicsConfig:
    name: str
    mass: float         # [kg]
    inertia: InertiaConfig
    max_thrust: float   # Total maximum thrust (all motors combined) [N]
    motor: Optional[MotorConfig] = field(default=None)


# ---------------------------------------------------------------------------
# Lee geometric controller config
# ---------------------------------------------------------------------------

@dataclass
class LeeControllerConfig:
    """Gain vectors for the geometric (Lee 2010) controller."""
    position_gain:     list   # k_pos  [3]  pos error  → force      [N/m]
    velocity_gain:     list   # k_vel  [3]  vel error  → force      [N·s/m]
    attitude_gain:     list   # k_att  [3]  att error  → moment     [N·m/rad]
    angular_rate_gain: list   # k_rate [3]  rate error → moment     [N·m·s/rad]
    max_acceleration:  float = field(default=math.inf)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class DroneConfig:
    physics:  DronePhysicsConfig
    attitude: AttitudeControllerConfig
    position: PositionControllerConfig
    # Raw params dict for CascadePIDController.
    # Present only when the YAML contains a ``controllers.cascade_pid`` section.
    cascade_pid: Optional[dict]             = field(default=None)
    # Gains for LeePositionController.
    # Present only when the YAML contains a ``controllers.lee`` section.
    lee:           Optional[LeeControllerConfig] = field(default=None)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _pid(d: dict) -> PIDConfig:
    return PIDConfig(
        kp=float(d["kp"]),
        ki=float(d["ki"]),
        kd=float(d["kd"]),
        tau=float(d["tau"]),
        limit=float(d["limit"]),
    )


def load_config(path: str | Path) -> DroneConfig:
    """
    Parse a drone YAML config file and return a DroneConfig dataclass.

    Example
    -------
    >>> cfg = load_config("configs/crazyflie.yaml")
    >>> att_ctrl = AttController_Vectorized.from_drone_config(cfg, num_envs=4, device="cpu")
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # --- Physics ---
    d = raw["drone"]
    motor_raw = d.get("motor", None)
    motor: Optional[MotorConfig] = None
    if motor_raw is not None:
        motor = MotorConfig(
            arm_length=float(motor_raw["arm_length"]),
            k_thrust=float(motor_raw["k_thrust"]),
            k_drag=float(motor_raw["k_drag"]),
            layout=str(motor_raw.get("layout", "x")),
            speed_min=float(motor_raw.get("speed_min", 0.0)),
            speed_max=float(motor_raw.get("speed_max", float("inf"))),
        )
    physics = DronePhysicsConfig(
        name=str(d["name"]),
        mass=float(d["mass"]),
        inertia=InertiaConfig(
            ixx=float(d["inertia"]["ixx"]),
            iyy=float(d["inertia"]["iyy"]),
            izz=float(d["inertia"]["izz"]),
        ),
        max_thrust=float(d["max_thrust"]),
        motor=motor,
    )

    # --- Attitude controller ---
    att = raw["controllers"]["attitude"]
    attitude = AttitudeControllerConfig(
        freq_rate_hz=float(att["freq_rate_hz"]),
        freq_angle_hz=float(att["freq_angle_hz"]),
        rate=AttitudeRateConfig(
            roll=_pid(att["rate"]["roll"]),
            pitch=_pid(att["rate"]["pitch"]),
            yaw=_pid(att["rate"]["yaw"]),
        ),
        angle=AttitudeAngleConfig(
            roll=_pid(att["angle"]["roll"]),
            pitch=_pid(att["angle"]["pitch"]),
            yaw=_pid(att["angle"]["yaw"]),
        ),
    )

    # --- Position controller ---
    pos = raw["controllers"]["position"]
    position = PositionControllerConfig(
        freq_vel_hz=float(pos["freq_vel_hz"]),
        freq_pos_hz=float(pos["freq_pos_hz"]),
        max_horizontal_angle_deg=float(pos["max_horizontal_angle_deg"]),
        max_thrust_scale=float(pos["max_thrust_scale"]),
        velocity=PositionVelConfig(
            vx=_pid(pos["velocity"]["vx"]),
            vy=_pid(pos["velocity"]["vy"]),
            vz=_pid(pos["velocity"]["vz"]),
        ),
        position=PositionPosConfig(
            x=_pid(pos["position"]["x"]),
            y=_pid(pos["position"]["y"]),
            z=_pid(pos["position"]["z"]),
        ),
    )

    # Optional CascadePIDController params (passed as-is to the controller)
    cascade_pid = raw.get("controllers", {}).get("cascade_pid", None)

    # Optional Lee geometric controller gains
    lee_raw = raw.get("controllers", {}).get("lee", None)
    lee: Optional[LeeControllerConfig] = None
    if lee_raw is not None:
        lee = LeeControllerConfig(
            position_gain=list(lee_raw["position_gain"]),
            velocity_gain=list(lee_raw["velocity_gain"]),
            attitude_gain=list(lee_raw["attitude_gain"]),
            angular_rate_gain=list(lee_raw["angular_rate_gain"]),
            max_acceleration=float(lee_raw.get("max_acceleration", math.inf)),
        )

    return DroneConfig(
        physics=physics,
        attitude=attitude,
        position=position,
        cascade_pid=cascade_pid,
        lee=lee,
    )
