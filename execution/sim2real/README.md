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
- **`hover_sim.py`**: Executes an RL policy checkpoint in Isaac Lab under identical training conditions, logging full flight telemetry to Parquet.
- **`plot_hover.py`**: Generates step-response tracking and target error figures (importable module or standalone CLI for re-plotting Parquet flights without GPU).

