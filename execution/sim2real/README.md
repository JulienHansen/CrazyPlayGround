# Sim2real: delay, step response and noise

Fly a hovering Crazyflie through velocity steps, then measure the radio delay, dead time, time constant and velocity noise used to set `env.action_latency_steps`, `env.noise_std` and `env.noise_corr` in simulation.

## Excitation cycle

`excite_hover.py` holds the hover with a position P loop on the host (100 Hz `send_velocity_world_setpoint`). After a 3 s grace period, the excited axis repeats:

| Phase   | Duration        | Velocity command | P hold on the excited axis |
|---------|-----------------|------------------|----------------------------|
| REST    | `--exc-step`    | 0                | off: settles at rest       |
| PUSH    | `--exc-step`    | amp              | off: step from rest        |
| RECOVER | `--exc-recover` | P hold           | on: brakes, flies back     |

Every step starts from rest and goes the same way (the sign of `--exc-amp`), so the response is not averaged over two directions that may differ. The analysis keeps only these PUSH edges: the edge back to 0 hands over to the P hold. The two other axes stay held throughout.

## Commands

Run from `CrazyPlayGround/`:

```bash
# 1. Record
python execution/sim2real/excite_hover.py --record-path flights/excite_x.parquet \
    --exc-axis x --exc-amp 0.2 --exc-step 1.0 --exc-recover 4 --duration 90
# 2. Delay, dead time, tau, noise (prints the env.* values)
python execution/sim2real/measure_delay_noise.py flights/excite_x.parquet [more.parquet ...]
# 3. Mean step response: IQM over the steps, 10-90 % band
python execution/sim2real/plot_step_response.py flights/excite_x.parquet [more.parquet ...] --out flights/excite_x.png
```

| `excite_hover.py` option | Default                       | Limit / note                                |
|--------------------------|-------------------------------|---------------------------------------------|
| `--uri`                  | `radio://0/80/2M/E7E7E7E7E8`  | close `cfclient` first                      |
| `--target X Y Z`         | above the takeoff position    | world frame                                 |
| `--exc-axis`             | `x`                           | prefer x/y: z nears the 0.1 m floor cut-out |
| `--exc-amp`              | 0.2 m/s                       | 0 < \|amp\| ≤ 1, sign = push direction      |
| `--exc-step`             | 1.0 s                         | [0.1, 5]                                    |
| `--exc-recover`          | 4.0 s                         | ≥ 0                                         |
| `--duration`             | 90 s                          | > 3 s                                       |
| `--record-path`          | required                      | `--record-every` must stay 1                |

## Choosing the parameters

- `--exc-step` must exceed the step's t90 (~0.45 s at 0.2 m/s, ~0.7 s at 1 m/s), hence 1 s.
- The drone moves ~amp × step + braking: ~0.3 m at 0.2 m/s, ~1.3 m at 1 m/s. The P hold returns at most 0.3 m/s, so use `--exc-recover 6` at 1 m/s.
- `--duration 90` gives ~14 steps (1 per cycle of 2 × step + recover); fly longer or several flights for more.
- Above ~0.5 m/s the response is acceleration-limited (a ramp at ~3.8 m/s²), not first order: measure `tau` at a small amplitude.

## Simulated step response (WP7)

`excite_hover_sim.py` flies the same excitation (same `excitation.py` signal, P hold and flags) on the Isaac Lab `Vel-Hovering` plant, with no policy. The step plus the P hold is the env action, tracked by the cascade PID at its velocity level like the firmware velocity PID. The recording has the real flight's columns, so the two tools above read both.

```bash
# On a machine with Isaac Lab (in Docker: /workspace/isaaclab/isaaclab.sh -p ... --headless)
python execution/sim2real/excite_hover_sim.py --record-path flights/sim_excite_x.parquet \
    --exc-axis x --exc-amp 0.2 --exc-step 1.0 --exc-recover 4 --duration 90 --headless \
    [env.action_latency_steps=4 ...]
python execution/sim2real/measure_delay_noise.py flights/excite_x.parquet flights/sim_excite_x.parquet
python execution/sim2real/plot_step_response.py flights/excite_x.parquet flights/sim_excite_x.parquet --out flights/real_vs_sim_x.png
```

Trailing `env.*=value` arguments are Hydra overrides of the simulated plant, e.g. the line `measure_delay_noise.py` prints. `--target` is relative to the env origin; the drone starts at rest on it. The Parquet footer holds `source=sim`, the env knobs and the excitation.

What the estimator measures in sim:

- `d` / `radio ms` is `env.action_latency_steps` exactly: `echo_v*` is the setpoint the controller used, after the env's command delay. With the default 0 there is no radio delay.
- `dead ms` is the simulated plant from that setpoint to 5 % of the step (cascade PID and rigid body; no motor dynamics), plus the one step of the row's state read, the same as on the drone. It does not depend on the latency.
- `noise` is ~0: `vel_*` is the true sim state, `env.add_noise` only perturbs the policy observation.

Compare against flights recorded with the current `excite_hover.py` and the same flags: older recordings used other cycles. Known plant gaps: no motor dynamics, no integral limit on the velocity PID, `thrust_base` 30000 against the real hover thrust.
