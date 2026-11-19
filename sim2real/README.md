# Sim2Real Tools

This directory contains sim-to-real tools, grouped by functional domain:

## 1. System Identification (`system-identification/`)
Characterize plant dynamics, measurement noise, and communication delays:
- **`excitation.py`**: Velocity step excitation generator and P-hold controllers.
- **`excite_hover.py`**: Velocity step excitation flight script for real Crazyflie hardware. **It flies a real drone.**
- **`excite_hover_sim.py`**: Simulated plant twin running the identical excitation protocol in Isaac Lab (WP7).
- **`measure_delay_noise.py`**: Estimates radio delay $d$, dead time, time constant $\tau$, and velocity noise.
- **`plot_step_response.py`**: Overlays real and simulated step responses from recorded Parquet files.

For full usage details, see [`system-identification/README.md`](system-identification/README.md).

## 2. Flight analysis (`single_drone_exec/`)
Metrics of real flights. The layout mirrors `execution/single_drone_exec/`, so each tool sits next
to the flight script whose recordings it reads. Nothing here flies a drone.
- **`hover/analyze_vel.py`**: Hover metrics of flights recorded by `execution/single_drone_exec/hover/exec_vel.py`
  (position error, jitter, drift, settling time, command smoothness, motor PWM, crash/success).
  Needs numpy and pyarrow; matplotlib only for `--plot`.

```bash
# Run from CrazyPlayGround/
python sim2real/single_drone_exec/hover/analyze_vel.py flight.parquet
python sim2real/single_drone_exec/hover/analyze_vel.py logs/*.parquet --skip-s 3 --json-out report.json
```
