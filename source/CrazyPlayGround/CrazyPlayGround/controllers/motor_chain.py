"""Crazyflie firmware power distribution and brushed-motor lag, batched over envs.

Turns the rate-PID output into the body wrench the way the drone does:

1. rate output in firmware units (deg/s gains → motor-command units), int16 saturation,
   firmware sign conventions (legacy pitch, negated yaw);
2. legacy mixer, ``T ± r ± p ± y`` with roll and pitch halved;
3. thrust cap: the excess over 65535 is taken from all four motors, then negatives clip to 0;
4. command → thrust, ``F = cmd · THRUST_MAX / 65535`` per motor, off below THRUST_MIN;
5. first-order lag per motor (time constant tau, per env);
6. motor thrusts → body wrench (total thrust along +z, roll/pitch from the arms, yaw from drag).

Constants are for the CF2.1 brushed with 2.1+ propellers, firmware 2026.08
(``platform_defaults_cf2.h``, ``power_distribution_quadrotor.c``, ``motors.c``).
Body frame is FLU (x forward, y left, z up), as in Isaac Lab.
"""

from __future__ import annotations

import math

import torch

CMD_MAX = 65535.0     # motor command full scale (uint16)
INT16_MAX = 32767.0   # saturateSignedInt16 bound on the roll/pitch/yaw outputs


class CrazyflieMotorChain:
    """Firmware mixer, thrust map and motor lag for ``num_envs`` drones.

    Motor order is the firmware's: M1 front-right, M2 rear-right, M3 rear-left, M4 front-left.

    Args:
        num_envs: Number of drones.
        dt: Time between calls [s] (the physics step, 1/500 s).
        device: Torch device.
        max_thrust_total: Thrust of all four motors at command 65535 [N] (4 × THRUST_MAX).
        thrust_min: Per-motor thrust below which the firmware switches the motor off [N].
        arm_length: Centre-to-motor distance [m].
        thrust2torque: Yaw drag torque per newton of thrust [m] (firmware THRUST2TORQUE).
        tau: Motor time constant [s]; 0 means no lag.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: str | torch.device = "cpu",
        max_thrust_total: float = 0.48,
        thrust_min: float = 0.0128,
        arm_length: float = 0.046,
        thrust2torque: float = 0.00699,
        tau: float = 0.072,
    ) -> None:
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.k = float(max_thrust_total) / (4.0 * CMD_MAX)  # N per command unit, per motor
        self.thrust_min = float(thrust_min)
        self.thrust2torque = float(thrust2torque)
        d = float(arm_length) * math.sqrt(0.5)
        self.motor_x = torch.tensor([d, -d, -d, d], device=self.device)
        self.motor_y = torch.tensor([-d, -d, d, d], device=self.device)
        # yaw reaction: M1/M3 spin CCW (react -z), M2/M4 spin CW (react +z)
        self.spin = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=self.device)
        self.blend = torch.ones(self.num_envs, 1, device=self.device)
        self.set_tau(tau)
        self.motor_forces = torch.zeros(self.num_envs, 4, device=self.device)

    def set_tau(self, tau: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Set the motor time constant.

        Args:
            tau: Time constant [s], a float or a ``(len(env_ids),)`` tensor; 0 means no lag.
            env_ids: Envs to update; None updates all.
        """
        tau = torch.as_tensor(tau, device=self.device, dtype=torch.float32)
        blend = torch.where(tau > 0.0, -torch.expm1(-self.dt / tau.clamp(min=1e-9)), torch.ones_like(tau))
        if env_ids is None:
            self.blend[:] = blend.reshape(-1, 1) if blend.dim() else blend
        else:
            ids = env_ids.to(dtype=torch.long, device=self.device)
            self.blend[ids] = blend.reshape(-1, 1) if blend.dim() else blend

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
        u = torch.trunc(torch.clamp(torch.rad2deg(alpha_ref), -INT16_MAX, INT16_MAX))
        roll = u[:, 0]
        pitch = -u[:, 1]  # firmware legacy pitch: positive = nose up
        yaw = -u[:, 2]    # controller_pid.c negates yaw before the mixer
        r = torch.trunc(roll / 2.0)
        p = torch.trunc(pitch / 2.0)
        t = thrust_cmd.reshape(-1)
        m = torch.stack((t - r + p + yaw, t - r - p - yaw, t + r - p + yaw, t + r + p - yaw), dim=-1)
        m = m - torch.clamp(m.amax(dim=-1, keepdim=True) - CMD_MAX, min=0.0)
        return torch.clamp(m, min=0.0)

    def __call__(self, thrust_cmd: torch.Tensor, alpha_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the motors one step and return the body wrench.

        Args:
            thrust_cmd: ``(num_envs, 1)`` collective command [0..65535].
            alpha_ref: ``(num_envs, 3)`` rate-PID output, body frame (FLU) [rad/s²].

        Returns:
            ``(thrust (num_envs, 1) [N], torque_b (num_envs, 3) [N·m])``; the thrust acts along body +z.
        """
        f_cmd = self.motor_commands(thrust_cmd, alpha_ref) * self.k
        f_cmd = torch.where(f_cmd < self.thrust_min, torch.zeros_like(f_cmd), f_cmd)
        self.motor_forces = self.motor_forces + self.blend * (f_cmd - self.motor_forces)
        f = self.motor_forces
        thrust = f.sum(dim=-1, keepdim=True)
        torque_b = torch.stack(
            (
                (self.motor_y * f).sum(dim=-1),
                -(self.motor_x * f).sum(dim=-1),
                self.thrust2torque * (self.spin * f).sum(dim=-1),
            ),
            dim=-1,
        )
        return thrust, torque_b
