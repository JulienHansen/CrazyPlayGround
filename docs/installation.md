# Installation

## Prerequisites

Install [Isaac Lab 2.3.2](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (Isaac Sim 4.5+).

**NVIDIA driver:** requires driver **535 or newer** (Linux). This is a floor, not a target — install whatever the latest driver is for your GPU rather than pinning to 535. Known-working reference: driver `580.159.03`.

⚠️ The 535 minimum comes from Isaac Sim 4.5's own requirements and may be raised by future Isaac Lab/Isaac Sim releases — always check the [Isaac Lab installation page](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for the current minimum before assuming 535 is still sufficient.

## Install CrazyPlayGround

<!-- pip install -e "source/CrazyPlayGround[rl]" -->

### Optional extras

| Extra | Who needs it | Command |
|---|---|---|
| `rl` | Training in simulation | `pip install -e "source/CrazyPlayGround[rl]"` |
| `deploy` | Real Crazyflie deployment | `pip install -e "source/CrazyPlayGround[deploy]"` |
| `teleop` | Teleoperation scripts | `pip install -e "source/CrazyPlayGround[teleop]"` |
| `docs` | Building this documentation | `pip install -e "source/CrazyPlayGround[docs]"` |
| `all` | Everything | `pip install -e "source/CrazyPlayGround[all]"` |

## Verify

<!-- python scripts/list_envs.py, expected output -->

## Docker

### Requirements

- Linux + NVIDIA GPU
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- An account on the [NVIDIA GPU Cloud (NGC) catalog](https://catalog.ngc.nvidia.com/), used to pull the base image from `nvcr.io`

### Build

The Dockerfile builds from a base image hosted on the [NGC catalog](https://catalog.ngc.nvidia.com/), so you must authenticate with `nvcr.io` before building.

Generate an API key from your account at [catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com/) (Setup → Generate API Key), then log in using that key as the password with `$oauthtoken` as the username:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <Your Key>
```

Then build the image:

```bash
docker build -t crazyplayground -f docker/Dockerfile .
```

### Run

```bash
docker run --gpus all -it --rm -v $(pwd)/logs:/workspace/crazyplayground/logs crazyplayground
```

If you're actively editing code (configs, scripts, source) and don't want to rebuild the image every time, mount the whole repo instead of just `logs/`: local changes are picked up immediately since the package is installed editable (`pip install -e`) from that same path. Rebuild only when the Dockerfile itself changes or dependencies need updating.

```bash
# Alternative: mount the whole repo for live local edits (no rebuild needed)
docker run --gpus all -it --rm -v $(pwd):/workspace/crazyplayground crazyplayground
```

Once inside the container, use the following command to verify and train (inside the container `--headless` is mandatory):

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/list_envs.py
/workspace/isaaclab/isaaclab.sh -p scripts/skrl/train.py --task=Vel-Hovering --num_envs=4096 --headless
```
