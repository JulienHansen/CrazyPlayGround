"""Crazyflie firmware power distribution and brushed-motor lag, batched over envs.

Turns the rate-PID output into the body wrench the way the drone does:

1. rate output in firmware units (deg/s gains → motor-command units), int16 saturation,
   firmware sign conventions (legacy pitch, negated yaw);
2. legacy mixer, ``collective ± roll/2 ± pitch/2 ± yaw`` per motor;
3. thrust cap: the excess over 65535 is taken from all four motors, then negatives clip to 0;
4. command → thrust, ``F = cmd · THRUST_MAX / 65535`` per motor, off below THRUST_MIN;
5. first-order lag per motor on the thrust, ``tau_up`` when it rises and ``tau_down`` when it falls (per env);
6. motor thrusts → body wrench (total thrust along +z, roll/pitch from the arms, yaw from drag).

All parameters are passed in from ``crazyflie.yaml`` (``drone.max_thrust``, ``drone.motor``,
``drone.firmware``); this file holds no parameter values. They are the CF2.1 brushed with 2.1+
propellers values of the Crazyflie firmware 2026.08: ``src/platform/interface/platform_defaults_cf2.h``
(``CONFIG_CRAZYFLIE_21_PLUS``), used in ``power_distribution_quadrotor.c`` and ``motors.c``; the motor
time constants and the thrust curve (``thrust_to_omega``) are Crazyflow ``cf2x_P250``'s.
Body frame is FLU (x forward, y left, z up), as in Isaac Lab.
"""

from __future__ import annotations

import math

import torch


def thrust_to_omega(force: torch.Tensor, rpm2thrust: tuple[float, float, float]) -> torch.Tensor:
    """Rotor speed from per-motor thrust, the inverse of ``F = c0 + c1·rpm + c2·rpm²``.

    Static map: the motor lag sits on the thrust in ``CrazyflieMotorChain``, so this gives the speed
    of a rotor that produces ``force`` now.

    Args:
        force: Per-motor thrust [N], any shape (e.g. ``(num_envs, 4)``).
        rpm2thrust: ``(c0, c1, c2)`` of the thrust curve (``drone.motor.rpm2thrust``), rpm in rev/min.

    Returns:
        Rotor speed [rad/s], same shape as ``force``; 0 where ``force <= 0`` (motor off).
    """
    c0, c1, c2 = rpm2thrust
    disc = torch.clamp(c1 * c1 + 4.0 * c2 * (force - c0), min=0.0)
    rpm = (-c1 + torch.sqrt(disc)) / (2.0 * c2)
    # with c1 < 0 the positive root at F = 0 is -c1/c2 (~2254 rpm), not 0: a motor with no thrust is off
    return torch.where(force > 0.0, rpm * (2.0 * math.pi / 60.0), torch.zeros_like(force))


class CrazyflieMotorChain:
    """Firmware mixer, thrust map and motor lag for ``num_envs`` drones.

    Motor order is the firmware's: M1 front-right, M2 rear-right, M3 rear-left, M4 front-left.

    Args:
        num_envs: Number of drones.
        dt: Time between calls [s] (the physics step, 1/500 s).
        device: Torch device (the env's device).
        max_thrust_total: Thrust of all four motors at command 65535 [N] (4 × THRUST_MAX).
        thrust_min: Per-motor thrust below which the firmware switches the motor off [N].
        arm_length: Centre-to-motor distance [m].
        thrust2torque: Yaw drag torque per newton of thrust [m] (firmware THRUST2TORQUE).
        tau_up: Motor time constant when the thrust rises [s]; 0 means no lag.
        tau_down: Motor time constant when the thrust falls [s]; 0 means no lag.
        motor_cmd_max: Motor command full scale (firmware uint16, 65535).
        control_cmd_max: Saturation of the roll/pitch/yaw control outputs (firmware int16, 32767).
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        *,
        device: str | torch.device,
        max_thrust_total: float,
        thrust_min: float,
        arm_length: float,
        thrust2torque: float,
        tau_up: float,
        tau_down: float,
        motor_cmd_max: float,
        control_cmd_max: float,
    ) -> None:
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.motor_cmd_max = float(motor_cmd_max)
        self.control_cmd_max = float(control_cmd_max)
        self.newton_per_cmd = float(max_thrust_total) / (4.0 * self.motor_cmd_max)  # per motor
        self.thrust_min = float(thrust_min)
        self.thrust2torque = float(thrust2torque)
        arm_xy = float(arm_length) * math.sqrt(0.5)  # motor x and y offset, arms at 45°
        self.motor_x = torch.tensor([arm_xy, -arm_xy, -arm_xy, arm_xy], device=self.device)
        self.motor_y = torch.tensor([-arm_xy, -arm_xy, arm_xy, arm_xy], device=self.device)
        # yaw reaction: M1/M3 spin CCW (react -z), M2/M4 spin CW (react +z)
        self.spin = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=self.device)
        # per-step blend factors 1 - exp(-dt / tau), (num_envs, 1)
        self.blend_up = torch.ones(self.num_envs, 1, device=self.device)
        self.blend_down = torch.ones(self.num_envs, 1, device=self.device)
        self.set_tau(tau_up, tau_down)
        self.motor_forces = torch.zeros(self.num_envs, 4, device=self.device)

    def set_tau(
        self,
        tau_up: float | torch.Tensor,
        tau_down: float | torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Set the spin-up and spin-down time constants.

        Args:
            tau_up: Time constant when the thrust rises [s], a float or a ``(len(env_ids),)`` tensor;
                0 means no lag.
            tau_down: Time constant when the thrust falls [s], same form as ``tau_up``.
            env_ids: Envs to update; None updates all.
        """
        ids = None if env_ids is None else env_ids.to(dtype=torch.long, device=self.device)
        for blend_buf, tau in ((self.blend_up, tau_up), (self.blend_down, tau_down)):
            tau = torch.as_tensor(tau, device=self.device, dtype=torch.float32)
            blend = torch.where(tau > 0.0, -torch.expm1(-self.dt / tau.clamp(min=1e-9)), torch.ones_like(tau))
            blend = blend.reshape(-1, 1) if blend.dim() else blend
            if ids is None:
                blend_buf[:] = blend
            else:
                blend_buf[ids] = blend

    def reset(self, env_ids: torch.Tensor | None, motor_force: float) -> None:
        """Set the motors' thrust, e.g. to hover (``mass · g / 4``), so episodes start without a drop.

        Args:
            env_ids: Envs to reset; None resets all.
            motor_force: Per-motor thrust [N].
        """
        if env_ids is None:
            self.motor_forces[:] = motor_force
        else:
            self.motor_forces[env_ids.to(dtype=torch.long, device=self.device)] = motor_force

    def motor_commands(self, thrust_cmd: torch.Tensor, alpha_ref: torch.Tensor) -> torch.Tensor:
        """Firmware blocks 1-3: saturation and signs, mixer, cap.

        Args:
            thrust_cmd: ``(num_envs, 1)`` collective command, the same for every motor [0..65535].
            alpha_ref: ``(num_envs, 3)`` rate-PID output, body frame (FLU) [rad/s²].

        Returns:
            ``(num_envs, 4)`` motor commands in [0, 65535], order M1..M4.
        """
        # *_cmd: firmware motor-command units; with firmware gains rad2deg(alpha_ref) is already in them
        # firmware control_t roll/pitch/yaw: int16-saturated rate-PID output
        control_cmd = torch.trunc(
            torch.clamp(torch.rad2deg(alpha_ref), -self.control_cmd_max, self.control_cmd_max)
        )
        roll_cmd = control_cmd[:, 0]
        pitch_cmd = -control_cmd[:, 1]  # firmware legacy pitch: positive = nose up
        yaw_cmd = -control_cmd[:, 2]    # controller_pid.c negates yaw before the mixer
        # legacy mixer (power_distribution_quadrotor.c): roll and pitch are halved, yaw is not
        roll_half_cmd = torch.trunc(roll_cmd / 2.0)
        pitch_half_cmd = torch.trunc(pitch_cmd / 2.0)
        collective_cmd = thrust_cmd.reshape(-1)
        motor_cmd = torch.stack(
            (
                collective_cmd - roll_half_cmd + pitch_half_cmd + yaw_cmd,  # M1 front-right
                collective_cmd - roll_half_cmd - pitch_half_cmd - yaw_cmd,  # M2 rear-right
                collective_cmd + roll_half_cmd - pitch_half_cmd + yaw_cmd,  # M3 rear-left
                collective_cmd + roll_half_cmd + pitch_half_cmd - yaw_cmd,  # M4 front-left
            ),
            dim=-1,
        )
        # cap: the excess of the highest motor over motor_cmd_max is taken from all four
        motor_cmd = motor_cmd - torch.clamp(motor_cmd.amax(dim=-1, keepdim=True) - self.motor_cmd_max, min=0.0)
        return torch.clamp(motor_cmd, min=0.0)

    def __call__(self, thrust_cmd: torch.Tensor, alpha_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the motors one step and return the body wrench.

        Args:
            thrust_cmd: ``(num_envs, 1)`` collective command [0..65535].
            alpha_ref: ``(num_envs, 3)`` rate-PID output, body frame (FLU) [rad/s²].

        Returns:
            ``(thrust (num_envs, 1) [N], torque_b (num_envs, 3) [N·m])``; the thrust acts along body +z.
        """
        # per-motor thrust the command asks for [N], before the motor lag
        force_target = self.motor_commands(thrust_cmd, alpha_ref) * self.newton_per_cmd
        force_target = torch.where(force_target < self.thrust_min, torch.zeros_like(force_target), force_target)
        blend = torch.where(force_target > self.motor_forces, self.blend_up, self.blend_down)
        self.motor_forces = self.motor_forces + blend * (force_target - self.motor_forces)
        force = self.motor_forces
        thrust = force.sum(dim=-1, keepdim=True)
        torque_b = torch.stack(
            (
                (self.motor_y * force).sum(dim=-1),
                -(self.motor_x * force).sum(dim=-1),
                self.thrust2torque * (self.spin * force).sum(dim=-1),
            ),
            dim=-1,
        )
        return thrust, torque_b
