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
mixer M1–M4 (FR, RR, RL, FL) `= T−r+p+y, T−r−p−y, T+r−p+y, T+r+p−y`, `r, p = roll/2, pitch/2`;
excess over 65535 removed from all four, negatives to 0; `F = m·max_thrust/(4·65535)`, 0 below
12.8 mN; lag τ = `motor_tau` (72 ms, reset at hover; per episode from `motor_tau_range` in
Vel/Rate-Hovering); wrench: arm 0.046 m at 45°, yaw `0.00699·Σ±Fᵢ`. `T` is in firmware units (hover ≈ 36 165);
tasks convert newtons with `ctrl.thrust_cmd_scale = max_thrust / 65535`. Not modelled: 80 Hz gyro
filter (sim filters D only, 20 Hz), Kalman estimator, 1 kHz thrust update, arming.
