# Cascade PID Controller

## Architecture

<!-- RL agent (100 Hz) -> _apply_action() x5 (500 Hz) -> thrust + moment diagram -->

## Control loops

| Loop | Rate | Input → Output |
|---|---|---|
| Position | 100 Hz | position error [m] → velocity setpoint [m/s] |
| Velocity | 100 Hz | velocity error [m/s] → roll/pitch command [rad] + thrust Δ |
| Attitude | 500 Hz | attitude error [rad] → body-rate setpoint [rad/s] |
| Rate | 500 Hz | rate error [rad/s] → moment [N·m] |

## Simulation parameters

<!-- dt, decimation, policy rate, gyro LPF -->

## Implementation notes

<!-- cascade_pid.py, pid.py, how it maps to the Crazyflie firmware controller -->
