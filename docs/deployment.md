# Real-Drone Deployment

## Hardware requirements

Make sure you have the following available:

- A Crazyflie 2.1+
- A Lighthouse positioning deck
- 2-4 Lighthouse basestations V2.0 (recommended), or 2 Lighthouse basestations V1.0
- Crazyradio 2.0 or Crazyradio PA

## Setup

1. Install the Lighthouse positioning deck on the Crazyflie, following [Getting started with expansion decks](https://www.bitcraze.io/documentation/tutorials/getting-started-with-expansion-decks/).

2. Create a Python virtual environment on your machine for the deployment, either a conda venv or a [`uv`](https://docs.astral.sh/uv/getting-started/installation/) venv.

3. Install `cfclient` via pip or from source, following [Installation of the cfclient](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/installation/install/).

4. Install CrazyPlayGround from source with `pip install -e "source/CrazyPlayGround[deploy]"`.

5. Install the Crazyradio drivers and set up radio permissions, following the [Install drivers](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyradio-2-0/#install-drivers) section of the Crazyradio 2.0 tutorial (Windows driver install / Linux udev permissions).

6. Install and configure the Lighthouse base stations' channel, if not already done.

7. Make sure the Crazyradio has the latest firmware installed, following [Getting started with the Crazyradio 2.0](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyradio-2-0/).

8. If the Crazyflie isn't configured yet, plug it in over USB and configure it with the Crazyflie client: set it to 2 Mbit radio mode and note its address — you'll need this to connect to it via the Crazyradio.

9. Flash the firmware on the Crazyflie.

10. Follow the [Getting started with Lighthouse](https://www.bitcraze.io/documentation/tutorials/getting-started-with-lighthouse/) tutorial from "Wait for calibration of the base stations" through "Take off and fly" to calibrate the Lighthouse system and make your first test flight.

## Exporting a trained policy

<!-- checkpoint -> deployable format used by execution/ scripts -->

Locate the checkpoint produced by the training scripts under `scripts/`. For skrl training, checkpoints are written to `logs/skrl/`. You can either reference a checkpoint by its absolute path, or copy it into the `execution/` folder for easier retrieval.

## Single-drone execution

For example, to run a policy trained for velocity-command hovering:

```bash
python exec_vel.py --checkpoint ../checkpoints/best_agent.pt --uri radio://0/80/2M/E7E7E7E7E8
```

- Replace `E7E7E7E7E8` with your Crazyflie's actual radio address. Make sure `cfclient` isn't already connected to the Crazyflie before running the script — the radio link only supports one client at a time.
- `--target x y z` sets an initial hover target in world-frame coordinates (optional — if omitted, the drone hovers above its takeoff position).

```bash
python exec_vel.py --checkpoint ../checkpoints/best_agent.pt --uri radio://0/80/2M/E7E7E7E7E8 
```

<!-- execution/single_drone_exec/: hover, traj_tracking -->

### Trajectory tracking (single drone)

`execution/single_drone_exec/traj_tracking/exec_track.py` flies a `TrackingEnv` policy of any control mode along the lemniscate or the circle. As there is one training env, there is one script: the run's `params/env.yaml` names the mode (`control_mode`) and its action scales.

```bash
python exec_track.py --checkpoint logs/skrl/<dir>/<run>/checkpoints/best_agent.pt --uri radio://0/40/2M/E7E7E7E7E8 \
    --record-path flights/track.parquet
```

| `control_mode` | Task | Setpoint sent at 100 Hz | Action scales read from env.yaml |
|---|---|---|---|
| `position` | `Pos-Tracking` | `send_position_setpoint(pos + a * max_position_delta)` | `max_position_delta` |
| `velocity` | `Vel-Tracking` | `send_velocity_world_setpoint(a * max_velocity)` | `max_velocity` |
| `attitude` | `Att-Tracking` | `send_setpoint_manual(roll, pitch, yaw rate, thrust %, rate=False)` | `max_roll_pitch`, `max_yaw_rate`, thrust scales |
| `body_rate` | `Rate-Tracking` | `send_setpoint_manual(roll, pitch, yaw rates, thrust %, rate=True)` | `max_body_rate`, thrust scales |

The flight goes: high-level-commander takeoff to `--takeoff-height` (default: the trajectory's z, 1 m), `go_to` the reference's start (where `TrackingEnv` spawns the drone), then the policy at 100 Hz until Ctrl-C or `--duration` seconds, then a landing.

- **Thrust (attitude, body_rate).** The env's normalised thrust (0 at `min_thrust_scale`, 1 at `max_thrust_scale` x hover; Att takes `clamp(a, 0, 1)`, Rate `(a + 1) / 2`) is mapped piecewise linearly to cflib percent: the policy's hover command lands on `--hover-thrust-pct` (default 77, measured), the bottom on `--min-thrust-pct` (25), the top on 100. A null `max_thrust_scale` is resolved from the run's drone yaml when it is readable, else to the CF2.1 cap 1.5387.
- **Low-level modes (attitude, body_rate).** Before every hand-over to the high-level commander the script sends one level setpoint at hover thrust. It also leaves the policy (brake, fly back, land) on a tilt above 45°, a gyro rate above 270°/s for 3 steps, or a tracking error above 0.6 m for 10 steps. Rate sets `flightmode.stabMode*` to 0 before the takeoff and restores them after the landing.
- `--control-mode` is only needed for a checkpoint without env.yaml; a flag that disagrees with it is an error.

- **Reference.** As `TrackingEnv` since #96, it moves at a constant linear speed along one lap resampled into 256 equal-length segments; the 4 look-ahead points of step `s` lie `speed * (s + 5k) * 0.01` m along it. `--traj-speed` (m/s) defaults to the middle of the run's `traj_speed_range`; `--trajectory` defaults to its `trajectory_type`; `--traj-scale` sets the XY scale (default 1.5 m). A value outside the training ranges is flown with a warning. Runs trained before #96 stored `traj_speed_range` in rad/s, so they fly here at a linear speed they never saw.
- **Fence.** The whole lap must stay 0.2 m inside the arena fence (the circle needs `--traj-scale 0.8` or less), or the script stops before connecting. In flight, a position, or a position 0.1 s ahead, outside the fence hands over to the high-level commander: it brakes, flies back to (0, 0, 1), hovers and lands.
- **Training contract.** The run's `params/env.yaml` must be a tracking run with 4 look-ahead points, 5 steps apart, at a 0.01 s policy step.
- **History.** A policy trained with history (`obs_input_hist` K frames, `action_input_hist` M past actions) gets the same input as in the env: the last K 22-D frames, oldest first and primed with copies of the first, then the last M commanded actions, primed with the mode's hover command (zeros for position and velocity, hover thrust for attitude and body_rate). K and M are read from the run's `params/env.yaml`, next to `checkpoints/`. `--obs-input-hist K` / `--action-input-hist M` are only needed for a checkpoint copied out of its run; a flag that disagrees with `env.yaml` is an error.
- **Recording.** `--record-path` writes one Parquet row per step, as the hover scripts do: the 22-D frame, the action and the setpoint sent (columns named per mode), the state, `target_*` (the point the env scores the step against, look-ahead point 1), `track_err_m`, `traj_dist_m`, `tilt_deg`, IMU, battery, Kalman variance and loop timing. The footer names the checkpoint, the run's env.yaml, the control mode with its scales and thrust map, and the reference.
- The two-drone script (`execution/multi_drones_exec/traj_tracking/exec_vel.py`) still takes only policies trained without history (22-D input).

## Multi-drone execution

<!-- execution/multi_drones_exec/ -->

## Sim2real notes

To measure the radio delay, step response and velocity noise of the real drone, see `execution/sim2real/README.md`.

<!-- known gaps, bodyrate control validation status -->
