# Sim2Real Tools

This directory contains sim-to-real tools organized into two distinct functional domains:

## 1. System Identification (`system-identification/`)
Characterize plant dynamics, measurement noise, and communication delays:
- **`excitation.py`**: Velocity step excitation generator and P-hold controllers.
- **`excite_hover.py`**: Velocity step excitation flight script for real Crazyflie hardware.
- **`excite_hover_sim.py`**: Simulated plant twin running the identical excitation protocol in Isaac Lab (WP7).
- **`measure_delay_noise.py`**: Estimates radio delay $d$, dead time, time constant $\tau$, and velocity noise.
- **`plot_step_response.py`**: Overlays real and simulated step responses from recorded Parquet files.

For full usage details, see [`system-identification/README.md`](system-identification/README.md).

## 2. Hovering Evaluation (`hovering/`)
Evaluate trained reinforcement learning hover policies against target setpoints:
- **`hover_sim.py`**: Executes an RL policy checkpoint in Isaac Lab under identical training conditions, logging full flight telemetry to Parquet. The recording uses `exec_vel.py`'s column names, with `source: sim` in the footer.
- **`plot_hover.py`**: Generates step-response tracking and target error figures, for a sim run or a real `exec_vel.py` flight alike, and overlays sim and real with `--compare` (importable module or standalone CLI, no GPU needed).

To compare a real flight with the simulation, run the sim from the same start: `exec_vel.py` takes off to 0.5 m above the takeoff point before the policy starts, and `t = 0` is the first control step in both recordings.

```bash
# real Crazyflie (flies the drone: Crazyradio plugged in, clear flight area): takes off to 0.5 m,
# flies the policy to --target for --duration seconds, lands, and records every 100 Hz step
python execution/single_drone_exec/hover/exec_vel.py \
    --checkpoint execution/single_drone_exec/checkpoints/hovering_vel_best_multiexp_seed7.pt \
    --uri radio://0/80/2M/E7E7E7E7E8 --target 0.5 0.0 1.0 --duration 30 \
    --record-path flights/vel_x0.5_z1_$(date +%Y%m%d_%H%M%S).parquet

# Isaac Lab machine: the same policy, start and target as the real flight, recorded
python execution/sim2real/hovering/hover_sim.py --checkpoint logs/.../best_agent.pt \
    --start-pos 0.0 0.0 0.5 --target 0.5 0.0 1.0 --time 30 --no-video \
    --record-path flights/sim_x0.5_z1.parquet

# anywhere: <stem>_step_response.png and <stem>_target_error.png next to each recording
python execution/sim2real/hovering/plot_hover.py flights/<real_flight>.parquet flights/sim_x0.5_z1.parquet

# sim and real overlaid (position, velocity, velocity command per axis) -> flights/sim_real_comparison.png
python execution/sim2real/hovering/plot_hover.py flights/<real_flight>.parquet flights/sim_x0.5_z1.parquet --compare
```

