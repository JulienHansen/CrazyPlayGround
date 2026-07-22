---
<div align="center">
  <img src="https://github.com/JulienHansen/CrazyPlayGround/blob/main/docs/assets/banner.png"
       alt="Pearl's banner"
       width="1200"
       height="800" />
</div>

---

# CrazyPlayGround - Collection of CrazyFlie Environments

A collection of Crazyflie reinforcement learning environments built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab). Trained policies can be deployed on real Crazyflie 2.1 drones via the scripts present in `execution/`.

⚠️ This repository is under construction. If you find a bug or a problem, do not hesitate to open an issue! ⚠️

## Quick start

### 1. Install Isaac Lab

Install [Isaac Lab 2.3.2](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (Isaac Sim 4.5+).

**NVIDIA driver:** requires driver **535 or newer** (Linux). This is a floor, not a target — install whatever the latest driver is for your GPU rather than pinning to 535. Known-working reference: driver `580.159.03`.

⚠️ The 535 minimum comes from Isaac Sim 4.5's own requirements and may be raised by future Isaac Lab/Isaac Sim releases — always check the [Isaac Lab installation page](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for the current minimum before assuming 535 is still sufficient.

### 2. Install CrazyPlayGround

Install with the extras you need:

```bash
pip install -e "source/CrazyPlayGround[rl]"      # Simulation / Training
pip install -e "source/CrazyPlayGround[deploy]"  # Deployment Hardware
pip install -e "source/CrazyPlayGround[all]"    
```

### 3. Verify and train

```bash
python scripts/list_envs.py
python scripts/skrl/train.py --task=Vel-Hovering --num_envs=4096
```

### Docker

A Dockerfile is provided for reproducible training (Linux + NVIDIA GPU required). The base image is pulled from the [NVIDIA GPU Cloud (NGC) catalog](https://catalog.ngc.nvidia.com/), so you must log in to `nvcr.io` before building. Create an API key from your account at [catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com/) (Setup → Generate API Key), then log in using that key as the password with `$oauthtoken` as the username:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <Your Key>
docker build -t crazyplayground -f docker/Dockerfile .
docker run --gpus all -it --rm -v $(pwd)/logs:/workspace/crazyplayground/logs crazyplayground
```

If you're actively editing code (configs, scripts, source) and don't want to rebuild the image every time, mount the whole repo instead of just `logs/`: local changes are picked up immediately since the package is installed editable. Rebuild only when the Dockerfile itself changes or dependencies need updating.

```bash
# Alternative: mount the whole repo for live local edits (no rebuild needed)
docker run --gpus all -it --rm -v $(pwd):/workspace/crazyplayground crazyplayground
```

Once inside the container, use the following command to verify and train (inside the container the `--headless` is mandatory):

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/list_envs.py
/workspace/isaaclab/isaaclab.sh -p scripts/skrl/train.py --task=Vel-Hovering --num_envs=4096 --headless
```

## Documentation

Full documentation (environments, controller architecture, configuration, real-drone deployment) lives in [`docs/`](docs/) and is built with MkDocs Material, an online version of the documentation will soon be available
