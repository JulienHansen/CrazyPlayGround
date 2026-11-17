# Sim2Real Tools

This directory contains sim-to-real tools, grouped by functional domain:

## 1. System Identification (`system-identification/`)
Characterize plant dynamics, measurement noise, and communication delays:
- **`excitation.py`**: Velocity step excitation generator and P-hold controllers.
- **`excite_hover.py`**: Velocity step excitation flight script for real Crazyflie hardware.
- **`excite_hover_sim.py`**: Simulated plant twin running the identical excitation protocol in Isaac Lab (WP7).
- **`measure_delay_noise.py`**: Estimates radio delay $d$, dead time, time constant $\tau$, and velocity noise.
- **`plot_step_response.py`**: Overlays real and simulated step responses from recorded Parquet files.

For full usage details, see [`system-identification/README.md`](system-identification/README.md).
