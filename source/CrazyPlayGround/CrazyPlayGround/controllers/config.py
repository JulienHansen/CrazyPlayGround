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
    """Motor and frame geometry (``drone.motor`` in the YAML), used by the motor chain."""
    arm_length: float   # m    — center-to-motor distance
    layout:     str     # 'x' or '+' quad configuration
    rpm2thrust: tuple[float, float, float]  # per-motor F = c0 + c1·rpm + c2·rpm² [N]
    speed_max:  float   # rad/s — rotor speed at THRUST_MAX on rpm2thrust
    tau_up:        float  # s — motor time constant on spin-up (first-order lag)
    tau_down:      float  # s — motor time constant on spin-down
    thrust_min:    float  # N — per-motor thrust below which the firmware cuts the motor
    thrust2torque: float  # m — yaw drag torque per newton of thrust


RPM_PER_RAD_S = 60.0 / (2.0 * math.pi)


def rpm_to_thrust(rpm: float, rpm2thrust: tuple[float, float, float]) -> float:
    """Per-motor thrust [N] at ``rpm`` on the curve ``F = c0 + c1·rpm + c2·rpm²``."""
    c0, c1, c2 = rpm2thrust
    return c0 + c1 * rpm + c2 * rpm * rpm


def resolve_max_thrust_scale(scale: float | None, mass: float, max_thrust: float, g: float = 9.81) -> float:
    """Return the max-thrust fraction of hover thrust used by the attitude/rate action maps.

    Args:
        scale: Value from the env cfg; None derives it from the firmware cap.
        mass: Drone mass [kg] (``drone.mass``).
        max_thrust: Total thrust cap of the 4 motors [N] (``drone.max_thrust``).
        g: Gravity [m/s²].

    Returns:
        ``max_thrust / (mass · g)`` when ``scale`` is None (1.5387 for the CF2.1), else ``scale``.

    Raises:
        ValueError: If ``scale · mass · g`` exceeds ``max_thrust``: the top of the action range
            would ask for thrust the motors cannot give.
    """
    hover_thrust = mass * g
    if scale is None:
        return max_thrust / hover_thrust
    if scale * hover_thrust > max_thrust * (1.0 + 1e-6):
        raise ValueError(
            f"max_thrust_scale={scale} asks for {scale * hover_thrust:.4f} N, above drone.max_thrust "
            f"{max_thrust} N (max scale {max_thrust / hover_thrust:.4f}); use None to derive it"
        )
    return float(scale)


@dataclass
class FirmwareConfig:
    """Firmware integer formats (``drone.firmware`` in the YAML), used by the motor chain."""
    motor_cmd_max:   float  # motor command full scale (uint16)
    control_cmd_max: float  # saturation of the roll/pitch/yaw control outputs (int16)


@dataclass
class DronePhysicsConfig:
    name: str
    mass: float         # [kg]
    inertia: InertiaConfig
    max_thrust: float   # Total maximum thrust (all motors combined) [N]
    motor: Optional[MotorConfig] = field(default=None)
    firmware: Optional[FirmwareConfig] = field(default=None)


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
            layout=str(motor_raw.get("layout", "x")),
            rpm2thrust=tuple(float(c) for c in motor_raw["rpm2thrust"]),
            speed_max=float(motor_raw["speed_max"]),
            tau_up=float(motor_raw["tau_up"]),
            tau_down=float(motor_raw["tau_down"]),
            thrust_min=float(motor_raw["thrust_min"]),
            thrust2torque=float(motor_raw["thrust2torque"]),
        )
        if len(motor.rpm2thrust) != 3:
            raise ValueError(f"drone.motor.rpm2thrust must be [c0, c1, c2], got {motor.rpm2thrust}")
        # the thrust curve must reach the firmware cap (THRUST_MAX per motor) at speed_max
        thrust_at_max = rpm_to_thrust(motor.speed_max * RPM_PER_RAD_S, motor.rpm2thrust)
        motor_thrust_max = float(d["max_thrust"]) / 4.0
        if abs(thrust_at_max - motor_thrust_max) > 0.01 * motor_thrust_max:
            raise ValueError(
                f"drone.motor: rpm2thrust gives {thrust_at_max:.5f} N at speed_max={motor.speed_max} rad/s, "
                f"expected max_thrust/4 = {motor_thrust_max:.5f} N within 1%"
            )
    firmware_raw = d.get("firmware", None)
    firmware: Optional[FirmwareConfig] = None
    if firmware_raw is not None:
        firmware = FirmwareConfig(
            motor_cmd_max=float(firmware_raw["motor_cmd_max"]),
            control_cmd_max=float(firmware_raw["control_cmd_max"]),
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
        firmware=firmware,
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
