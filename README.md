---
<div align="center">
  <img src="https://github.com/JulienHansen/CrazyPlayGround/blob/main/docs/assets/banner.png"
       alt="Pearl's banner"
       width="1200"
       height="800" />
</div>

---

# CrazyPlayGround - Collection of CrazyFlie Environments

A collection of Crazyflie reinforcement learning environments built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab), using a self-contained cascaded firmware-style PID inner-loop controller (position → velocity → attitude → rate). Trained policies can be deployed on real Crazyflie 2.1 drones via the scripts in `execution/`.

⚠️ This repository is still under construction. If you find a bug, a mistake, or a problem, do not hesitate to open an issue! ⚠️

## Quick start

**1.** Install [Isaac Lab 2.1](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (Isaac Sim 4.5+).

**2.** Install CrazyPlayGround with the extras you need:

```bash
pip install -e "source/CrazyPlayGround[rl]"      # simulation / training
pip install -e "source/CrazyPlayGround[deploy]"  # real-drone deployment (no Isaac Lab needed)
pip install -e "source/CrazyPlayGround[all]"     # everything
```

**3.** Verify and train:

```bash
python scripts/list_envs.py
python scripts/skrl/train.py --task=Vel-Hovering --num_envs=4096
```

### Docker

A Dockerfile based on the official Isaac Lab image is provided for reproducible training (Linux + NVIDIA GPU required):

```bash
docker login nvcr.io                    # NGC account required
docker build -t crazyplayground -f docker/Dockerfile .
docker run --gpus all -it --rm -v $(pwd)/logs:/workspace/crazyplayground/logs crazyplayground
```

## Documentation

Full documentation (environments, controller architecture, configuration, real-drone deployment) lives in [`docs/`](docs/) and is built with MkDocs Material:

```bash
pip install -e "source/CrazyPlayGround[docs]"
mkdocs serve   # http://127.0.0.1:8000
```

## License

Apache-2.0
