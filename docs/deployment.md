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

4. Install CrazyPlayGround from source with the `all` extra ([repo](https://github.com/JulienHansen/CrazyPlayGround)). The `deploy` extra alone isn't sufficient yet.

   <!-- pip install -e "source/CrazyPlayGround[deploy]", radio permissions/udev -->

5. Install and configure the Lighthouse base stations' channel, if not already done.

6. Make sure the Crazyradio has the latest firmware installed, following [Getting started with the Crazyradio 2.0](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyradio-2-0/).

7. If the Crazyflie isn't configured yet, plug it in over USB and configure it with the Crazyflie client: set it to 2 Mbit radio mode and note its address — you'll need this to connect to it via the Crazyradio.

8. Flash the firmware on the Crazyflie.

9. Follow the [Getting started with Lighthouse](https://www.bitcraze.io/documentation/tutorials/getting-started-with-lighthouse/) tutorial from "Wait for calibration of the base stations" through "Take off and fly" to calibrate the Lighthouse system and make your first test flight.

## Exporting a trained policy

<!-- checkpoint -> deployable format used by execution/ scripts -->

Locate the checkpoint produced by the training scripts under `scripts/`. For skrl training, checkpoints are written to `logs/skrl/`. You can either reference a checkpoint by its absolute path, or copy it into the `execution/` folder for easier retrieval.

## Single-drone execution

For example, to run a policy trained for velocity-command hovering:

```bash
python exec_vel.py --checkpoint ../checkpoints/best_agent.pt --uri radio://0/80/2M/E7E7E7E7E8
```

- Replace `E7E7E7E7E8` with your Crazyflie's actual radio address. Make sure `cfclient` isn't already connected to the Crazyflie before running the script — the radio link only supports one client at a time.
- `--target x y z` sets an initial hover target in world-frame coordinates (optional — if omitted, the drone hovers above its takeoff position)

If you want to record the flight data, add `--record-path path.csv` and `--record-interval 0.01` (0.01 means one record every 10 ms).

```bash
python exec_vel.py --checkpoint ../checkpoints/best_agent.pt --uri radio://0/80/2M/E7E7E7E7E8 --record-path ./flight_log2.csv --record-interval 0.01
```

<!-- execution/single_drone_exec/: hover, traj_tracking -->

## Multi-drone execution

<!-- execution/multi_drones_exec/ -->

## Sim2real notes

<!-- known gaps, bodyrate control validation status -->
