# Cascade PID Controller

## Architecture

<!-- RL agent (100 Hz) -> _apply_action() x5 (500 Hz) -> thrust + moment diagram -->

## Control loops

| Loop | Rate | Input → Output |
|---|---|---|
| Position | 100 Hz | position error [m] → velocity setpoint [m/s] |
| Velocity | 100 Hz | velocity error [m/s] → roll/pitch command [rad] + thrust Δ |
| Attitude | 500 Hz | attitude error [rad] → body-rate setpoint [rad/s] |
| Rate | 500 Hz | rate error [rad/s] → α_ref [rad/s²] |
| Motor chain | 500 Hz | α_ref + collective command → thrust [N] + moment [N·m] |

## Simulation parameters

<!-- dt, decimation, policy rate, gyro LPF -->

## Implementation notes

<!-- cascade_pid.py, pid.py, how it maps to the Crazyflie firmware controller -->

### Motor chain (`controllers/motor_chain.py`)

Firmware power distribution + brushed-motor model (CF2.1, 2.1+ props, fw 2026.08). With firmware
gains, `rad2deg(α_ref)` is already in motor units. Clip to ±32767, signs pitch `−α_y`, yaw `−α_z`;
mixer, with `T` the collective command and `roll_half, pitch_half = roll/2, pitch/2` (truncated):
M1 front-right `T − roll_half + pitch_half + yaw`, M2 rear-right `T − roll_half − pitch_half − yaw`,
M3 rear-left `T + roll_half − pitch_half + yaw`, M4 front-left `T + roll_half + pitch_half − yaw`;
excess over 65535 removed from all four, negatives to 0; `F = m·max_thrust/(4·65535)`, 0 below
`thrust_min` (12.8 mN); lag τ = `tau` (72 ms, reset at hover; per episode from `motor_tau_range` in
Vel/Rate-Hovering); wrench: arm 0.046 m at 45°, yaw `thrust2torque·Σ±Fᵢ` (0.00699 m). `T` is in firmware units (hover ≈ 36 165);
tasks convert newtons with `ctrl.thrust_cmd_scale = max_thrust / 65535`. Not modelled: 80 Hz gyro
filter (sim filters D only, 20 Hz), Kalman estimator, 1 kHz thrust update, arming.

All parameters live in `crazyflie.yaml`: `drone.max_thrust`, `drone.motor` (`arm_length`, `tau`,
`thrust_min`, `thrust2torque`) and `drone.firmware` (`motor_cmd_max` 65535, `control_cmd_max` 32767,
the firmware's uint16/int16 formats), with no defaults in the code. Except τ, they are the firmware's
`THRUST_MAX`, `ARM_LENGTH`, `THRUST_MIN` and `THRUST2TORQUE` from
`src/platform/interface/platform_defaults_cf2.h` (`CONFIG_CRAZYFLIE_21_PLUS`), not yet identified on our drones.
