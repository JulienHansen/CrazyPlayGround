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

A Dockerfile is provided for reproducible training (Linux + NVIDIA GPU required):

```bash
docker login nvcr.io                    
docker build -t crazyplayground -f docker/Dockerfile .
docker run --gpus all -it --rm -v $(pwd)/logs:/workspace/crazyplayground/logs crazyplayground
```

Once inside the container, use the following command to verify and train (inside the container the `--headless` is mandatory):

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/list_envs.py
/workspace/isaaclab/isaaclab.sh -p scripts/skrl/train.py --task=Vel-Hovering --num_envs=4096 --headless
```

## Documentation

Full documentation (environments, controller architecture, configuration, real-drone deployment) lives in [`docs/`](docs/) and is built with MkDocs Material, an online version of the documentation will soon be available
