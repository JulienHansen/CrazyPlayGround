from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .utils.math_utils import quat_apply_inverse, euler_xyz_from_quat, expand_to
from .pid import PID_Vectorized
from .motor_chain import CMD_MAX, CrazyflieMotorChain

DEG2RAD = math.pi / 180.0

# ---------------------------------------------------------------------------
# Default gains & limits  (match Crazyflie 2.x firmware)
# ---------------------------------------------------------------------------

DEFAULT_GAINS: dict = {
    "pos":  {"kp": [2.0,   2.0,   2.0],   "ki": [0.0,  0.0,  0.5],  "kd": [0.0, 0.0, 0.0], "kff": [0.0, 0.0, 0.0]},
    "vel":  {"kp": [25.0, 25.0,  25.0],   "ki": [1.0,  1.0, 15.0],  "kd": [0.0, 0.0, 0.0], "kff": [0.0, 0.0, 0.0]},
    "att":  {"kp": [6.0,   6.0,   6.0],   "ki": [3.0,  3.0,  1.0],  "kd": [0.0, 0.0, 0.35], "kff": [0.0, 0.0, 0.0]},
    "rate": {"kp": [250.0, 250.0, 120.0], "ki": [500.0, 500.0, 16.7], "kd": [2.5, 2.5, 0.0], "kff": [0.0, 0.0, 0.0]},
}

DEFAULT_LIMITS: dict = {
    "att_integral":  [20.0  * DEG2RAD, 20.0  * DEG2RAD, 360.0 * DEG2RAD],
    "rate_integral": [33.3  * DEG2RAD, 33.3  * DEG2RAD, 166.7 * DEG2RAD],
    "pos_vel_max":   [1.0, 1.0, 1.0],
    "roll_max":      20.0 * DEG2RAD,
    "pitch_max":     20.0 * DEG2RAD,
    "yaw_max_delta": 0.0,
    "thrust_base":   36_000.0,
    "thrust_min":    20_000.0,
    "thrust_cmd_max": 65_535.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _as_column(value, reference: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = _as_tensor(value, device, dtype)
    if tensor.dim() == 0:
        tensor = tensor.view(1, 1)
    elif tensor.dim() == 1:
        tensor = tensor.view(-1, 1)
    ref_batch = reference.shape[0] if reference.dim() > 0 else 1
    if tensor.shape[0] not in (1, ref_batch):
        raise ValueError("Batch dimension mismatch for yaw inputs.")
    if tensor.shape[0] == 1 and ref_batch != 1:
        tensor = tensor.expand(ref_batch, 1)
    return tensor


def _wrap_angle_rad(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class CascadePIDController:
    """
    Generic 4-level cascaded PID controller for multirotor drones.

    Loop hierarchy
    ──────────────
    position  → pos_pid  → [vel setpoint]
    velocity  → vel_pid  → [roll/pitch/thrust cmd]
    attitude  → att_pid  → [body-rate setpoint]
    rate      → rate_pid → [α_ref rad/s²]
    α_ref, thrust cmd → CrazyflieMotorChain (firmware mixer, cap, thrust map,
                        motor lag) → [thrust N, moment N·m]

    The controller can be entered at any level via ``command_level``.
    Thrust commands are per-motor firmware units (0–65535, the same value on
    every motor); ``thrust_cmd_scale`` converts total newtons to that unit.

    Parameters
    ----------
    dt : float
        Simulation timestep [s].
    num_envs : int
        Number of parallel environments.
    device : str
        PyTorch device string.
    params : dict, optional
        Override any gain or limit.  See ``DEFAULT_GAINS`` / ``DEFAULT_LIMITS``
        for available keys.  Values from a YAML ``cascade_pid`` section can
        be passed directly.
    """

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: str = "cpu",
        params: Optional[dict] = None,
        # physics — can be set later via set_physical_params()
        mass: float = 1.0,
        inertia: list[float] | None = None,
        max_thrust: float = 0.48,
        arm_length: float = 0.046,
    ) -> None:
        self.dt      = float(dt)
        self.device  = torch.device(device)
        self._num_envs = num_envs
        params = params or {}

        # ── Loop rates & decimation ──────────────────────────────────────────
        sim_rate_hz      = float(params.get("sim_rate_hz",            1.0 / self.dt))
        posvel_rate_hz   = float(params.get("pid_posvel_loop_rate_hz", 100.0))
        att_rate_hz      = float(params.get("pid_loop_rate_hz",        500.0))
        self.posvel_decimation = max(1, int(round(sim_rate_hz / posvel_rate_hz)))
        self.att_decimation    = max(1, int(round(sim_rate_hz / att_rate_hz)))
        self.posvel_dt   = self.dt * self.posvel_decimation
        self.att_dt      = self.dt * self.att_decimation

        # ── Physical params ──────────────────────────────────────────────────
        self.mass    = torch.tensor(mass,    device=self.device, dtype=torch.float32)
        inertia      = inertia or [1.0, 1.0, 1.0]
        self.inertia = torch.tensor(inertia, device=self.device, dtype=torch.float32)
        self._inertia_tensor: Optional[torch.Tensor] = None

        # ── Gains ────────────────────────────────────────────────────────────
        def _g(loop: str, term: str) -> torch.Tensor:
            key = f"{loop}_{term}"
            return torch.as_tensor(
                params.get(key, DEFAULT_GAINS[loop][term]),
                device=self.device, dtype=torch.float32,
            )

        # Velocity gains are in deg/s for x/y so we convert kp/ki/kd to rad
        _vel_scale = torch.tensor([DEG2RAD, DEG2RAD, 1.0], device=self.device, dtype=torch.float32)

        # ── Integral limits ──────────────────────────────────────────────────
        def _lim_vec(rad_key: str, deg_key: str, default: list) -> torch.Tensor:
            if rad_key in params:
                return torch.as_tensor(params[rad_key], device=self.device, dtype=torch.float32)
            if deg_key in params:
                return torch.as_tensor(params[deg_key], device=self.device, dtype=torch.float32) * DEG2RAD
            return torch.as_tensor(default, device=self.device, dtype=torch.float32)

        att_lim  = _lim_vec("att_integral_limit",  "att_integral_limit_deg",  DEFAULT_LIMITS["att_integral"])
        rate_lim = _lim_vec("rate_integral_limit", "rate_integral_limit_deg", DEFAULT_LIMITS["rate_integral"])

        # ── Build inner PID objects ──────────────────────────────────────────
        # Multi-axis mode: gains are [3] tensors, state is lazily [N, 3].
        # tau=0 → raw backward difference (matches firmware; no Tustin filter).
        self.pos_pid = PID_Vectorized(
            kp=_g("pos", "kp"), ki=_g("pos", "ki"), kd=_g("pos", "kd"),
            kff=_g("pos", "kff"), tau=0.0, device=self.device,
        )
        self.vel_pid = PID_Vectorized(
            kp=_g("vel", "kp") * _vel_scale,
            ki=_g("vel", "ki") * _vel_scale,
            kd=_g("vel", "kd") * _vel_scale,
            kff=_g("vel", "kff") * _vel_scale,
            tau=0.0, device=self.device,
        )
        self.att_pid = PID_Vectorized(
            kp=_g("att", "kp"), ki=_g("att", "ki"), kd=_g("att", "kd"),
            kff=_g("att", "kff"), tau=0.0, device=self.device,
            integral_limit=att_lim,
        )
        # Rate: gains kept separate for per-env updates; PID state managed manually
        self.rate_kp  = _g("rate", "kp")
        self.rate_ki  = _g("rate", "ki")
        self.rate_kd  = _g("rate", "kd")
        self.rate_kff = _g("rate", "kff")
        self._rate_integral_limit = rate_lim

        # ── Saturation limits ────────────────────────────────────────────────
        def _lim_scalar(rad_keys, deg_keys, default):
            for k in rad_keys:
                if k in params: return float(params[k])
            for k in deg_keys:
                if k in params: return float(params[k]) * DEG2RAD
            return float(default)

        self.vel_max     = torch.as_tensor(
            params.get("pos_vel_max", DEFAULT_LIMITS["pos_vel_max"]),
            device=self.device, dtype=torch.float32,
        )
        self.roll_limit  = _lim_scalar(("vel_roll_max",  "roll_max"),
                                       ("vel_roll_max_deg",  "roll_max_deg"),
                                       DEFAULT_LIMITS["roll_max"])
        self.pitch_limit = _lim_scalar(("vel_pitch_max", "pitch_max"),
                                       ("vel_pitch_max_deg", "pitch_max_deg"),
                                       DEFAULT_LIMITS["pitch_max"])
        self.yaw_max_delta = _lim_scalar(("yaw_max_delta",), ("yaw_max_delta_deg",),
                                         DEFAULT_LIMITS["yaw_max_delta"])

        # ── Thrust scaling ───────────────────────────────────────────────────
        thrust_cmd_max           = float(params.get("thrust_cmd_max", DEFAULT_LIMITS["thrust_cmd_max"]))
        self.vel_thrust_scale    = float(params.get("vel_thrust_scale", params.get("thrust_scale", 1000.0)))
        # total newtons per unit of collective command: the same command goes to all 4 motors,
        # each giving max_thrust / 4 at 65535, so F_total = 4 · cmd · max_thrust / (4 · 65535)
        self.thrust_cmd_scale    = float(max_thrust) / CMD_MAX
        self.thrust_base_cmd     = float(params.get("thrust_base",  params.get("thrustBase",  DEFAULT_LIMITS["thrust_base"])))
        self.thrust_min_cmd      = float(params.get("thrust_min",   params.get("thrustMin",   DEFAULT_LIMITS["thrust_min"])))
        self.thrust_max_cmd      = float(params.get("thrust_max",   thrust_cmd_max))
        self._thrust_base_from_params = "thrust_base" in params or "thrustBase" in params

        # ── Motor chain (firmware mixer, cap, thrust map, motor lag) ─────────
        self.motor_chain = CrazyflieMotorChain(
            num_envs=num_envs,
            dt=self.dt,
            device=self.device,
            max_thrust_total=max_thrust,
            thrust_min=float(params.get("motor_thrust_min", 0.0128)),
            arm_length=arm_length,
            thrust2torque=float(params.get("motor_thrust2torque", 0.00699)),
            tau=float(params.get("motor_tau", 0.072)),
        )

        # ── Derivative low-pass filter ────────────────────────────────────────
        # 1st-order IIR on the rate measurement, used by the D term only (P and I
        # use the raw rate). Not the firmware's filter: without it, kd amplifies
        # the discrete 500 Hz steps. Set gyro_lpf_cutoff_hz: 0 to disable.
        # TODO: firmware 80 Hz 2-pole gyro LPF on P/I/D (filter.c lpf2pApply)
        _gyro_lpf_hz = float(params.get("gyro_lpf_cutoff_hz", 80.0))
        if _gyro_lpf_hz > 0.0:
            self._gyro_lpf_alpha = math.exp(-2.0 * math.pi * _gyro_lpf_hz * self.dt)
        else:
            self._gyro_lpf_alpha = 0.0  # α=0 → no smoothing (pass-through)

        # ── Mutable buffers (lazy-initialised in _ensure_buffers) ────────────
        self._step_count:   int                       = 0
        self._vel_sp:       Optional[torch.Tensor]    = None
        self._att_sp:       Optional[torch.Tensor]    = None
        self._rate_sp:      Optional[torch.Tensor]    = None
        self._thrust_cmd:   Optional[torch.Tensor]    = None
        self._yaw_sp:       Optional[torch.Tensor]    = None
        self._rate_integral:      Optional[torch.Tensor] = None
        self._prev_rate_meas:     Optional[torch.Tensor] = None
        self._rate_meas_filtered: Optional[torch.Tensor] = None

        self._command_handlers = {
            "position":  self._cmd_position,
            "velocity":  self._cmd_velocity,
            "attitude":  self._cmd_attitude,
            "body_rate": self._cmd_body_rate,
        }

    # ── Classmethod factory ──────────────────────────────────────────────────

    @classmethod
    def from_drone_config(
        cls,
        drone_config,                # DroneConfig (avoids circular import)
        num_envs: int,
        dt: float,
        device: str = "cpu",
    ) -> "CascadePIDController":
        """
        Build from a :class:`~drone_control.config.loader.DroneConfig`.

        If the config contains a ``cascade_pid`` section (YAML key
        ``controllers.cascade_pid``), those params override the defaults.

        Example
        -------
        >>> cfg  = load_config("configs/crazyflie.yaml")
        >>> ctrl = CascadePIDController.from_drone_config(cfg, num_envs=4, dt=0.002)
        """
        params = getattr(drone_config, "cascade_pid", None) or {}
        phys   = drone_config.physics
        if phys.motor is None:
            raise ValueError("drone.motor (arm_length) is required by the motor chain")

        ctrl = cls(
            dt=dt,
            num_envs=num_envs,
            device=device,
            params=params,
            mass=phys.mass,
            inertia=[phys.inertia.ixx, phys.inertia.iyy, phys.inertia.izz],
            max_thrust=phys.max_thrust,
            arm_length=phys.motor.arm_length,
        )

        # Mass sets the hover thrust (motor reset, thrust_base if not given);
        # PhysX handles the rigid-body dynamics, so the inertia is informational.
        _dev = torch.device(device)
        diag = torch.tensor(
            [phys.inertia.ixx, phys.inertia.iyy, phys.inertia.izz],
            dtype=torch.float32, device=_dev,
        )
        J = torch.diag(diag).unsqueeze(0).expand(num_envs, -1, -1).contiguous()
        ctrl.set_physical_params(
            mass=torch.tensor(phys.mass, dtype=torch.float32),
            inertia_tensor=J,
        )
        return ctrl

    # ── Physics setters ──────────────────────────────────────────────────────

    def set_physical_params(
        self,
        mass: Optional[torch.Tensor] = None,
        inertia_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        if mass is not None:
            self.mass = _as_tensor(mass, self.device, torch.float32).view(())
            if not self._thrust_base_from_params:
                self.thrust_base_cmd = (self.mass.item() * 9.81) / max(self.thrust_cmd_scale, 1e-6)
        if inertia_tensor is not None:
            inertia_tensor = _as_tensor(inertia_tensor, self.device, torch.float32)
            if inertia_tensor.dim() == 2:
                inertia_tensor = inertia_tensor.unsqueeze(0)
            self._inertia_tensor = inertia_tensor
            self.inertia = torch.diagonal(inertia_tensor, dim1=-2, dim2=-1).mean(dim=0)

    def set_rate_gains(
        self,
        rate_kp: Optional[torch.Tensor] = None,
        rate_ki: Optional[torch.Tensor] = None,
        rate_kd: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Update rate-loop gains, optionally for a subset of environments."""
        def _ensure_batched(g: torch.Tensor) -> torch.Tensor:
            if g.dim() == 1 and self._num_envs is not None:
                return g.view(1, -1).expand(self._num_envs, -1).contiguous()
            return g

        if env_ids is None:
            if rate_kp is not None: self.rate_kp = _as_tensor(rate_kp, self.device, torch.float32)
            if rate_ki is not None: self.rate_ki = _as_tensor(rate_ki, self.device, torch.float32)
            if rate_kd is not None: self.rate_kd = _as_tensor(rate_kd, self.device, torch.float32)
        else:
            ids = env_ids.to(dtype=torch.long, device=self.device)
            if rate_kp is not None:
                self.rate_kp = _ensure_batched(self.rate_kp); self.rate_kp[ids] = _as_tensor(rate_kp, self.device, torch.float32)
            if rate_ki is not None:
                self.rate_ki = _ensure_batched(self.rate_ki); self.rate_ki[ids] = _as_tensor(rate_ki, self.device, torch.float32)
            if rate_kd is not None:
                self.rate_kd = _ensure_batched(self.rate_kd); self.rate_kd[ids] = _as_tensor(rate_kd, self.device, torch.float32)

    # ── Reset ────────────────────────────────────────────────────────────────

    def set_motor_tau(self, tau, env_ids: Optional[torch.Tensor] = None) -> None:
        """Set the motor time constant [s], a float or one value per env in ``env_ids``."""
        self.motor_chain.set_tau(tau, env_ids)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset all integrators and put the motors at hover thrust.

        Pass ``env_ids`` to reset only a subset.
        """
        self.motor_chain.reset(env_ids, self.mass.item() * 9.81 / 4.0)
        if env_ids is None:
            self.pos_pid.reset(); self.vel_pid.reset(); self.att_pid.reset()
            self._rate_integral       = None
            self._prev_rate_meas      = None
            self._rate_meas_filtered  = None
            self._vel_sp = self._att_sp = self._rate_sp = self._thrust_cmd = self._yaw_sp = None
            self._step_count = 0
            return

        ids = env_ids.to(dtype=torch.long, device=self.device)
        self.pos_pid.reset(ids); self.vel_pid.reset(ids); self.att_pid.reset(ids)
        for buf in (self._rate_integral, self._prev_rate_meas, self._rate_meas_filtered,
                    self._vel_sp, self._att_sp, self._rate_sp):
            if buf is not None:
                buf[ids] = 0.0
        if self._thrust_cmd is not None:
            self._thrust_cmd[ids] = self.thrust_base_cmd
        if self._yaw_sp is not None:
            self._yaw_sp[ids] = float("nan")

    # ── Main call ────────────────────────────────────────────────────────────

    def __call__(
        self,
        root_state: torch.Tensor,
        target_pos:        Optional[torch.Tensor] = None,
        target_vel:        Optional[torch.Tensor] = None,
        target_attitude:   Optional[torch.Tensor] = None,
        target_body_rates: Optional[torch.Tensor] = None,
        target_yaw:        Optional[torch.Tensor] = None,
        target_yaw_rate:   Optional[torch.Tensor] = None,
        thrust_cmd:        Optional[torch.Tensor] = None,
        *,
        command_level: str,
        body_rates_in_body_frame: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one control step.

        Args:
            root_state:  ``[N, 13]`` = ``[pos(3), quat(4), lin_vel(3), ang_vel(3)]``.
                         Quaternion as ``[w, x, y, z]``.
            command_level: One of ``"position"``, ``"velocity"``,
                           ``"attitude"``, ``"body_rate"``.
            body_rates_in_body_frame: If True, ``ang_vel`` is already in the
                           body frame.  If False (default), it is in the world
                           frame and will be rotated.

        Returns:
            ``(thrust [N, 1], moment [N, 3])`` both in SI units (N, N·m).
        """
        if root_state.dim() == 1:
            root_state = root_state.unsqueeze(0)

        pos, quat, lin_vel, ang_vel = torch.split(
            root_state.to(device=self.device, dtype=torch.float32), [3, 4, 3, 3], dim=-1
        )

        if not body_rates_in_body_frame:
            ang_vel = quat_apply_inverse(quat, ang_vel)

        command_level = command_level.lower()
        if command_level not in self._command_handlers:
            raise ValueError(f"Unknown command_level '{command_level}'. "
                             f"Choose from {list(self._command_handlers)}")

        update_posvel = (self._step_count % self.posvel_decimation) == 0
        update_att    = (self._step_count % self.att_decimation)    == 0
        self._step_count += 1

        self._ensure_buffers(pos.shape[0])

        target_pos = expand_to(_as_tensor(target_pos, self.device, pos.dtype),     pos)     if target_pos is not None else pos.clone()
        target_vel = expand_to(_as_tensor(target_vel, self.device, lin_vel.dtype), lin_vel) if target_vel is not None else torch.zeros_like(lin_vel)

        euler    = euler_xyz_from_quat(quat)
        att_actual = torch.stack(euler, dim=-1)          # [N, 3]  (roll, pitch, yaw)
        yaw_actual = att_actual[..., 2:3]

        if target_yaw is not None:
            target_yaw = _as_column(target_yaw, yaw_actual, self.device, yaw_actual.dtype)
        if target_yaw_rate is not None:
            target_yaw_rate = _as_column(target_yaw_rate, yaw_actual, self.device, yaw_actual.dtype)

        yaw_sp = self._update_yaw_setpoint(yaw_actual, target_yaw, target_yaw_rate)

        # Velocity command overrides vel_sp directly (skip pos loop)
        if command_level == "velocity" and update_posvel:
            self._vel_sp = target_vel

        self._command_handlers[command_level](
            pos=pos, lin_vel=lin_vel, ang_vel=ang_vel,
            att_actual=att_actual, yaw_actual=yaw_actual, yaw_sp=yaw_sp,
            target_pos=target_pos, target_vel=target_vel,
            target_attitude=target_attitude, target_body_rates=target_body_rates,
            thrust_cmd=thrust_cmd,
            update_posvel=update_posvel, update_att=update_att,
        )

        alpha_ref = self._rate_pid_alpha(self._rate_sp, ang_vel)
        return self.motor_chain(self._thrust_cmd, alpha_ref)

    # ── Internal buffer management ───────────────────────────────────────────

    def _ensure_buffers(self, batch: int) -> None:
        def _need(buf): return buf is None or buf.shape[0] != batch
        if _need(self._vel_sp):     self._vel_sp    = torch.zeros((batch, 3), device=self.device)
        if _need(self._att_sp):     self._att_sp    = torch.zeros((batch, 3), device=self.device)
        if _need(self._rate_sp):    self._rate_sp   = torch.zeros((batch, 3), device=self.device)
        if _need(self._thrust_cmd): self._thrust_cmd = torch.full((batch, 1), self.thrust_base_cmd, device=self.device)

    # ── Command-level handlers ───────────────────────────────────────────────

    def _cmd_position(self, **ctx) -> None:
        if ctx["update_posvel"]:
            vel_sp = self.pos_pid(ctx["target_pos"] - ctx["pos"], self.posvel_dt)
            self._vel_sp = torch.clamp(vel_sp, -self.vel_max, self.vel_max)
        self._cmd_velocity(**ctx)

    def _cmd_velocity(self, **ctx) -> None:
        yaw_sp        = ctx["yaw_sp"]
        yaw_sp_scalar = yaw_sp.squeeze(-1)
        self._att_sp[:, 2] = yaw_sp_scalar

        if ctx["update_posvel"]:
            yaw = ctx["yaw_actual"].squeeze(-1)
            c, s = torch.cos(yaw), torch.sin(yaw)
            lv   = ctx["lin_vel"]

            # Project velocities into body-horizontal frame
            vel_bx = c * lv[:, 0] + s * lv[:, 1]
            vel_by = -s * lv[:, 0] + c * lv[:, 1]
            sp_bx  = c * self._vel_sp[:, 0] + s * self._vel_sp[:, 1]
            sp_by  = -s * self._vel_sp[:, 0] + c * self._vel_sp[:, 1]

            vel_err = torch.stack(
                [sp_bx - vel_bx, sp_by - vel_by, self._vel_sp[:, 2] - lv[:, 2]], dim=-1
            )
            vel_out = self.vel_pid(vel_err, self.posvel_dt)

            pitch_cmd = vel_out[:, 0].clamp(-self.pitch_limit, self.pitch_limit)
            roll_cmd  = (-vel_out[:, 1]).clamp(-self.roll_limit, self.roll_limit)
            thrust_raw = self.thrust_base_cmd + vel_out[:, 2] * self.vel_thrust_scale
            thrust_raw = torch.clamp(thrust_raw, self.thrust_min_cmd, self.thrust_max_cmd)

            self._att_sp    = torch.stack((roll_cmd, pitch_cmd, yaw_sp_scalar), dim=-1)
            self._thrust_cmd = thrust_raw.unsqueeze(-1)

        if ctx["update_att"]:
            self._update_rate_from_attitude(ctx["att_actual"])

    def _cmd_attitude(self, **ctx) -> None:
        if ctx["update_att"]:
            target_att = ctx["target_attitude"]
            att_des = ctx["att_actual"].clone() if target_att is None \
                else expand_to(_as_tensor(target_att, self.device, ctx["ang_vel"].dtype), ctx["ang_vel"])
            att_des[..., 2:3] = ctx["yaw_sp"]
            self._att_sp = att_des
        if ctx["thrust_cmd"] is not None:
            self._thrust_cmd = _as_tensor(ctx["thrust_cmd"], self.device, torch.float32).view(-1, 1)
        self._update_rate_from_attitude(ctx["att_actual"])

    def _cmd_body_rate(self, **ctx) -> None:
        if ctx["target_body_rates"] is None:
            self._rate_sp = torch.zeros_like(ctx["ang_vel"])
        else:
            self._rate_sp = expand_to(
                _as_tensor(ctx["target_body_rates"], self.device, ctx["ang_vel"].dtype), ctx["ang_vel"]
            )
        if ctx["thrust_cmd"] is not None:
            self._thrust_cmd = _as_tensor(ctx["thrust_cmd"], self.device, torch.float32).view(-1, 1)

    def _update_rate_from_attitude(self, att_actual: torch.Tensor) -> None:
        att_error     = _wrap_angle_rad(self._att_sp - att_actual)
        self._rate_sp = self.att_pid(att_error, self.att_dt)

    # ── Rate PID → α_ref ─────────────────────────────────────────────────────

    def _rate_pid_alpha(
        self,
        rate_sp:   torch.Tensor,
        rate_meas: torch.Tensor,
    ) -> torch.Tensor:
        """Rate PID, with derivative on measurement to avoid derivative kick.

        Args:
            rate_sp: ``(num_envs, 3)`` body-rate setpoint, body frame [rad/s].
            rate_meas: ``(num_envs, 3)`` measured body rate, body frame [rad/s].

        Returns:
            ``(num_envs, 3)`` α_ref [rad/s²]. Its value in degrees is the firmware's
            rate-PID output in motor-command units (same gains, deg/s errors).
        """
        if self._rate_integral is None or self._rate_integral.shape != rate_sp.shape:
            self._rate_integral = torch.zeros_like(rate_sp)

        # ── Gyroscope low-pass filter ─────────────────────────────────────
        # Applied only to the derivative path; error uses the raw measurement.
        if self._rate_meas_filtered is None or self._rate_meas_filtered.shape != rate_meas.shape:
            self._rate_meas_filtered = rate_meas.clone()
        else:
            self._rate_meas_filtered = (
                self._gyro_lpf_alpha * self._rate_meas_filtered
                + (1.0 - self._gyro_lpf_alpha) * rate_meas
            )

        if self._prev_rate_meas is None or self._prev_rate_meas.shape != rate_meas.shape:
            self._prev_rate_meas = self._rate_meas_filtered.clone()

        rate_error = rate_sp - rate_meas
        self._rate_integral = torch.clamp(
            self._rate_integral + rate_error * self.dt,
            -expand_to(self._rate_integral_limit, self._rate_integral),
             expand_to(self._rate_integral_limit, self._rate_integral),
        )

        rate_meas_dot        = (self._rate_meas_filtered - self._prev_rate_meas) / self.dt
        self._prev_rate_meas = self._rate_meas_filtered.clone()

        return (
            self.rate_kp * rate_error
            + self.rate_ki * self._rate_integral
            - self.rate_kd * rate_meas_dot
        )

    # ── Yaw setpoint ─────────────────────────────────────────────────────────

    def _update_yaw_setpoint(
        self,
        yaw_actual:    torch.Tensor,
        yaw_target:    Optional[torch.Tensor],
        yaw_rate:      Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._yaw_sp is None or self._yaw_sp.shape != yaw_actual.shape:
            self._yaw_sp = yaw_actual.clone()

        self._yaw_sp = torch.where(torch.isnan(self._yaw_sp), yaw_actual, self._yaw_sp)

        if yaw_rate is not None:
            self._yaw_sp = _wrap_angle_rad(self._yaw_sp + yaw_rate * self.dt)
        elif yaw_target is not None:
            self._yaw_sp = _wrap_angle_rad(yaw_target)

        if self.yaw_max_delta > 0.0:
            delta = _wrap_angle_rad(self._yaw_sp - yaw_actual)
            delta = torch.clamp(delta, -self.yaw_max_delta, self.yaw_max_delta)
            self._yaw_sp = _wrap_angle_rad(yaw_actual + delta)

        return self._yaw_sp
