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

`hover/exec_rate.py` flies a `Rate-Hovering` policy (body rates + thrust). It takes the same `--controller rl|pid`, `--duration`, `--obs-input-hist` / `--action-input-hist` and recording flags as `hover/exec_vel.py`, plus the thrust mapping: `--hover-thrust-pct` (default 77 %, measured on our drone) and `--min-thrust-pct`; the policy's full-thrust action is always cflib's 100 % (60000 PWM), with no cap. Both controllers take off with the high-level commander (0.5 m in 3 s, then 1 s to settle), so takeoff does not depend on `--hover-thrust-pct`. With `--controller pid`, the firmware position PID then flies to `--target`. Its rate and thrust commands are logged back into the same `cmd` columns as an RL flight.

<!-- execution/single_drone_exec/: hover, traj_tracking -->

### Trajectory tracking (single drone)

`execution/single_drone_exec/traj_tracking/exec_vel.py` flies a `Vel-Tracking` policy along the lemniscate:

```bash
python exec_vel.py --checkpoint logs/skrl/<dir>/<run>/checkpoints/best_agent.pt --uri radio://0/40/2M/E7E7E7E7E8
```

- A policy trained with history (`obs_input_hist` K frames, `action_input_hist` M past actions) gets the same input as in the env: the last K 22-D frames, oldest first and primed with copies of the first, then the last M commanded actions. K and M are read from the run's `params/env.yaml`, next to `checkpoints/`.
- `--obs-input-hist K` / `--action-input-hist M` are only needed for a checkpoint copied out of its run; a flag that disagrees with `env.yaml` is an error.
- The two-drone script (`execution/multi_drones_exec/traj_tracking/exec_vel.py`) still takes only policies trained without history (22-D input).

## Multi-drone execution

<!-- execution/multi_drones_exec/ -->

## Sim2real notes

To measure the radio delay, step response and velocity noise of the real drone, see `execution/sim2real/README.md`.

<!-- known gaps, bodyrate control validation status -->
