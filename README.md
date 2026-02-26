---
<div align="center">
  <img src="https://github.com/JulienHansen/CrazyPlayGround/blob/main/docs/assets/banner.png"
       alt="Pearl's banner"
       width="1200"
       height="800" />
</div>

---

# CrazyPlayGround - Collection of CrazyFlie Environments 

A collection of Crazyflie 2.1 reinforcement learning environments built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab), using a cascaded PID inner-loop controller from [DroneModule](https://github.com/JulienHansen/DroneModule).

---

## Overview

CrazyPlayGround provides Isaac Lab environments for training RL agents on a simulated Crazyflie 2.1. Instead of letting the RL agent directly command motor thrusts, a **cascaded firmware-style PID controller** (position → velocity → attitude → rate) runs at 500 Hz as the inner loop. The RL agent operates at a higher level (100 Hz), commanding position deltas, velocity references, or attitude setpoints depending on the environment.

### Control architecture

```
RL agent (100 Hz)
  └─ sets target (pos delta / vel ref / attitude)
       └─ _apply_action() × 5  (500 Hz, each physics step)
            └─ Cascade PID  →  thrust [N] + moment [N·m]
                 └─ applied to Crazyflie rigid body via Isaac Lab
```

| Loop | Rate | Input → Output |
|---|---|---|
| Position | 100 Hz | position error [m] → velocity setpoint [m/s] |
| Velocity | 100 Hz | velocity error [m/s] → roll/pitch command [rad] + thrust Δ |
| Attitude | 500 Hz | attitude error [rad] → body-rate setpoint [rad/s] |
| Rate | 500 Hz | rate error [rad/s] → moment [N·m] |

### Simulation parameters

| Parameter | Value |
|---|---|
| Physics timestep (`dt`) | 1/500 s = 2 ms |
| Decimation | 5 |
| Policy rate | 100 Hz |
| Gyro LPF cutoff | 20 Hz |

---

## Environments

### Single-drone hovering

Three variants differing only in the abstraction level of the RL action space:

| Task ID | Action | Action space |
|---|---|---|
| `Pos-Hovering` | Position delta `[dx, dy, dz]` (m), clamped to ±0.1 | 3 |
| `Vel-Hovering` | Velocity reference `[Vx, Vy, Vz]` (m/s), scaled by `max_velocity=1.0` | 3 |
| `Att-Hovering` | `[roll, pitch, yaw_rate, thrust_normalized]` | 4 |

All three share the same **observation space** (dim=6):

```
[lin_vel_b (3), desired_pos_b (3)]
```
- `lin_vel_b`: linear velocity in body frame [m/s]
- `desired_pos_b`: goal position expressed in body frame [m]

And the same **reward**:

```
r = - lin_vel_scale × ||ω_lin||²
  - ang_vel_scale × ||ω_ang||²
  + distance_scale × (1 - tanh(||pos - goal|| / 0.8))
```

Episode terminates when the drone goes below 0.1 m or above 2.0 m.

### Multi-agent (MARL)

| Task ID | Description |
|---|---|
| `Template-Crazyplayground-Marl-Direct-v0` | Multi-agent collaborative task |

---

## Dependencies

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab) (Isaac Sim 4.5+)
- [DroneModule](https://github.com/JulienHansen/DroneModule) — cascade PID controller and Crazyflie YAML config

---

## Installation

**1. Install Isaac Lab** following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

**2. Install DroneModule** (editable):

```bash
pip install -e /path/to/DroneModule
```

**3. Install CrazyPlayGround** (editable):

```bash
pip install -e source/CrazyPlayGround
```

**4. Verify** — list available environments:

```bash
python scripts/list_envs.py
```

Expected output includes `Vel-Hovering`, `Pos-Hovering`, `Att-Hovering`, and `Template-Crazyplayground-Marl-Direct-v0`.

---

## Usage

### Train

```bash
# SKRL (PPO)
python scripts/skrl/train.py --task=Vel-Hovering --num_envs=4096

# RSL-RL
python scripts/rsl_rl/train.py --task=Pos-Hovering --num_envs=4096

# Stable Baselines 3
python scripts/sb3/train.py --task=Att-Hovering --num_envs=512
```

### Play / evaluate

```bash
python scripts/skrl/play.py --task=Vel-Hovering --num_envs=16
```

### Debug with dummy agents

```bash
# Zero-action agent (checks physics stability)
python scripts/zero_agent.py --task=Vel-Hovering

# Random-action agent (checks environment bounds)
python scripts/random_agent.py --task=Vel-Hovering
```

---

## Project structure

```
CrazyPlayGround/
├── source/CrazyPlayGround/CrazyPlayGround/
│   └── tasks/direct/
│       ├── hovering/               # Single-drone envs
│       │   ├── pos_hovering.py     # Position-delta control
│       │   ├── vel_hovering.py     # Velocity-reference control
│       │   ├── att_hovering.py     # Attitude control
│       │   └── agents/             # RL agent configs (skrl, sb3, rsl_rl…)
│       └── crazyplayground_marl/   # Multi-agent env
├── scripts/
│   ├── skrl/                       # SKRL train & play scripts
│   ├── rsl_rl/
│   ├── sb3/
│   ├── rl_games/
│   ├── list_envs.py
│   ├── zero_agent.py
│   └── random_agent.py
└── DroneModule/configs/crazyflie.yaml   # PID gains & physics params
```

---

## Configuration

PID gains and simulation parameters are centralized in `DroneModule/configs/crazyflie.yaml` under the `crazyflie_pid` section. Key parameters:

```yaml
crazyflie_pid:
  sim_rate_hz:            500.0   # Must match physics dt
  pid_posvel_loop_rate_hz: 100.0
  pid_loop_rate_hz:        500.0
  gyro_lpf_cutoff_hz:       20.0  # Gyro low-pass filter cutoff

  pos_kp:  [2.0, 2.0, 2.0]
  vel_kp:  [25.0, 25.0, 25.0]    # x/y in deg/(m/s), internally ×DEG2RAD
  att_kp:  [6.0, 6.0, 6.0]
  rate_kp: [250.0, 250.0, 120.0]
```
