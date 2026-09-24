# Sim-to-real evaluation pipeline — Crazyflie velocity hover

Tools to measure the gap between the IsaacLab simulator and a real Crazyflie 2.1 on
the hovering task. The protocol follows Aljalbout et al., *The Reality Gap in
Robotics* (2026): each controller is reported on its own and never averaged with
others, because each routes through a different part of the firmware stack.

The pipeline measures one policy the same way in both worlds and subtracts.

## Scripts

| script | role |
|---|---|
| `collect_hover_vel.py` | Flies a velocity policy on the drone and logs every control step |
| `eval_hover_vel_sim.py` | Flies the same policy in simulation, same schema |
| `analyze_hover.py` | Computes hover metrics from either source |
| `compare_sim2real.py` | Prints the per-metric gap between two metric reports |
| `measure_delay_noise.py` | Estimates loop delay and estimator noise from logs |
| `check_policy.py` | Offline checkpoint check; no drone, no simulator |

## Recorded schema

One row per control step, at 100 Hz nominal. Columns cover the policy observation,
the raw action, the command sent, the EKF state, the raw IMU, motor PWM, battery and
Kalman variance. `collect_hover_vel.CSV_COLUMNS` is the single definition;
`eval_hover_vel_sim.py` imports it so the two cannot drift apart. Columns with no
simulator counterpart (motor PWM, battery, accelerometer, estimator variance) are
written as zero and excluded from comparisons.

Each run directory holds `flight.csv`, `flight.npz` and a `metadata.json` that pins
the checkpoint path and SHA-256, the git commit, the control rates and the frame
conventions. The SHA-256 identifies which policy actually flew, independently of the
`--tag` given on the command line.

## Conventions

These must match between simulator and hardware, or the comparison is meaningless.

- Quaternion order `(qw, qx, qy, qz)`, scalar first.
- Velocity is world frame (`stateEstimate.v*`), rotated into the body frame with
  `quat_inv`. This matches `root_lin_vel_b` in the environment.
- Policy at 100 Hz, inner loop at 500 Hz, decimation 5.
- Action in `[-1, 1]^3`, scaled by `max_velocity = 1.0` m/s, sent as
  `send_velocity_world_setpoint(vx, vy, vz, yaw_rate=0)`.
- Gyro is logged in deg/s on hardware; the simulator converts rad/s to deg/s so the
  units agree.

## Usage

Offline check, then a short flight, then repeats:

```bash
python execution/sim2real/check_policy.py --checkpoint <ckpt> --device cpu \
    --history-len <K> --action-hist-len <M>

python execution/sim2real/collect_hover_vel.py --checkpoint <ckpt> \
    --history-len <K> --action-hist-len <M> --target 0 0 1.0 --duration 15 --tag run1

python execution/sim2real/analyze_hover.py execution/sim2real/data/*run1 --json-out real.json
```

The same policy in simulation, then the gap:

```bash
python execution/sim2real/eval_hover_vel_sim.py --checkpoint <ckpt> \
    --target 0 0 1.0 --duration 15 --tag run1 --headless
python execution/sim2real/analyze_hover.py execution/sim2real/data_sim/*run1 --json-out sim.json
python execution/sim2real/compare_sim2real.py sim.json real.json
```

A policy trained with a K-step observation window expects `6K + 3M` inputs. Passing
the wrong window raises a size mismatch at load time rather than flying a
wrongly-shaped policy.

## Measuring delay and noise

Loop delay **cannot** be identified from closed-loop policy flights. The command is
then a function of the response, and the estimated lag collapses onto the limit-cycle
geometry: on real logs the fit returned about 150 ms, which is a quarter period of
the 1.5 Hz oscillation, not a transport delay. `measure_delay_noise.py` refuses to
report a delay unless `--open-loop` is passed.

Use a known exogenous command instead:

```bash
python execution/sim2real/collect_hover_vel.py --excite chirp --excite-axis x \
    --excite-amp 0.3 --duration 25 --tag chirp_x
python execution/sim2real/measure_delay_noise.py execution/sim2real/data/*chirp_x --open-loop
```

Estimator noise needs a known true state. With Lighthouse there is no independent
ground truth, because Lighthouse is itself the source feeding the EKF. Log a
stationary drone instead: true velocity is zero and position is constant, so
everything recorded is estimator error.

```bash
python execution/sim2real/collect_hover_vel.py --excite step --excite-amp 0.0 \
    --duration 60 --tag static
python execution/sim2real/measure_delay_noise.py execution/sim2real/data/*static --static
```

## Protocol coverage

| test | how this pipeline supports it |
|---|---|
| OL-1 replay error | `cmd_*` and the start state give the real actions to replay in simulation |
| OL-2 loop delay | chirp excitation plus a first-order-plus-dead-time fit |
| OL-3 actuator check | `m1..m4` against `vbat` across battery levels |
| OL-4 state estimation | static log, or an external ground truth via `--mocap` |
| CL-1 hover gap | same policy, same goals, both worlds, then `compare_sim2real.py` |

## Tests

`pytest` from the repository root. The suite is CPU-only and does not need Isaac Sim.
