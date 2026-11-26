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
`thrust_min` (12.8 mN); first-order lag on each motor thrust, τ = `tau_up` while it rises and `tau_down`
while it falls (112 / 103 ms, reset at hover; τ_up per episode from `motor_tau_range` in
Vel/Rate-Hovering, τ_down = τ_up); wrench: arm 0.046 m at 45°, yaw `thrust2torque·Σ±Fᵢ`
(0.00699 m). `T` is in firmware units (hover ≈ 42 592 at 31.8 g; the firmware `thrust_base` 36 000 is
below it, and the vz integrator makes up the rest); tasks convert newtons with
`ctrl.thrust_cmd_scale = max_thrust / 65535`. Rotor speed from thrust: `thrust_to_omega` inverts the
curve `F = c0 + c1·rpm + c2·rpm²` (0 when the motor is off). Not modelled: 80 Hz gyro filter (sim
filters D only, 20 Hz), Kalman estimator, 1 kHz thrust update, arming, rotor drag (#99).

All parameters live in `crazyflie.yaml`: `drone.mass`, `drone.max_thrust`, `drone.motor` (`arm_length`,
`tau_up`, `tau_down`, `thrust_min`, `thrust2torque`, `rpm2thrust`, `speed_max`) and `drone.firmware`
(`motor_cmd_max` 65535, `control_cmd_max` 32767, the firmware's uint16/int16 formats), with no
defaults in the code. `load_config` checks that `rpm2thrust` gives `max_thrust/4` at `speed_max`
(within 1 %). None are identified on our drones:

| constant | value | source |
|---|---|---|
| `mass` | 0.0318 kg | [C] |
| `max_thrust` | 4 × 0.12 N | [F] `THRUST_MAX` |
| `thrust_min` | 0.0128 N | [F] `THRUST_MIN` |
| `thrust2torque` | 0.00699 m | [F] `THRUST2TORQUE` |
| `arm_length` | 0.046 m | [F] `ARM_LENGTH` |
| `rpm2thrust` | 0, −3.62e-7, 1.606e-10 | [C] |
| `speed_max` | 2983 rad/s | derived: rpm at 0.12 N |
| `tau_up` / `tau_down` | 0.1117 / 0.1029 s | [C] `rotor_dyn_coef`, linearised at hover |

[F] Crazyflie firmware `src/platform/interface/platform_defaults_cf2.h` (`CONFIG_CRAZYFLIE_21_PLUS`).
[C] [Crazyflow](https://github.com/utiasDSL/crazyflow) `cf2x_P250` (`dynamics/first_principles/params.toml`).

The yaml mass drives the controller; the PhysX body mass is set to it by the `set_total_mass` startup
event (`tasks/direct/crazyflie_events.py`), which scales every body's mass and inertia of the USD.
In the attitude/rate envs, `max_thrust_scale = None` (default) is `max_thrust/(m·g)` = 1.5387, the
firmware cap; a larger value set by hand is rejected.
