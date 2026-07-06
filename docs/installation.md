# Installation

## Prerequisites

<!-- Isaac Lab 2.1.0 / Isaac Sim 4.5+, link to official install guide -->

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

<!-- Linux + NVIDIA GPU, nvidia-container-toolkit, NGC account -->

### Build

<!-- docker login nvcr.io, docker build command -->

### Run

<!-- docker run --gpus all, volume mounts for logs, headless training example -->
