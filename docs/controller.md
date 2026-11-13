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

The rate PID uses the firmware gains, so its output in degrees, `rad2deg(α_ref)`, is the firmware
rate output in motor-command units. `CrazyflieMotorChain` then applies the firmware's power
distribution and a brushed-motor model (CF2.1, 2.1+ props, firmware 2026.08):

1. Saturation to ±32767 (int16) and the firmware signs: legacy pitch (`−α_y`), negated yaw (`−α_z`).
2. Legacy mixer, motor order M1 front-right, M2 rear-right, M3 rear-left, M4 front-left:
   `r = roll/2`, `p = pitch/2`, `m = T−r+p+y, T−r−p−y, T+r−p+y, T+r+p−y`.
3. Cap: the excess over 65535 is taken from all four motors, then negatives clip to 0.
4. Thrust map: `F = m · max_thrust / (4 · 65535)` per motor (0.12 N at full scale), 0 below 12.8 mN.
5. First-order lag per motor, `F += (1 − e^(−dt/τ))(F_cmd − F)`, τ = `motor_tau` (72 ms). Motors
   start at hover (`m·g/4`) on reset. Vel- and Rate-Hovering draw τ per episode from `motor_tau_range`.
6. Body wrench: thrust `ΣF` along +z, `τx = Σ yᵢFᵢ`, `τy = −Σ xᵢFᵢ` (arm 0.046 m at 45°),
   `τz = 0.00699 · Σ ±Fᵢ` (sign by spin). PhysX integrates the rigid-body dynamics.

The collective command `T` goes unchanged to every motor (firmware units; hover ≈ 36 165, `thrust_base`
36000). Tasks convert total newtons with `ctrl.thrust_cmd_scale = max_thrust / 65535`.

Not modelled yet: the firmware's 80 Hz 2-pole gyro filter (the sim keeps a 20 Hz filter on D only),
the Kalman estimator, the 1 kHz thrust update and arming.
