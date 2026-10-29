# Environments

## Single-drone hovering

<!-- Task table: Pos-Hovering / Vel-Hovering / Att-Hovering, action spaces -->

### Observation space

<!-- dim=6: lin_vel_b, desired_pos_b -->

### Reward

<!-- reward formula -->

### Termination

<!-- below 0.1 m / above 2.0 m -->

## Trajectory tracking

<!-- track/ task -->

## Drone racing

<!-- drone_racing/, drone_racing_marl/ -->

## Formation & fly-through

<!-- formation/, fly_through/ -->

## Multi-agent (MARL)

<!-- Template-Crazyplayground-Marl-Direct-v0 -->

## Pursuit-evasion

`tasks/direct/pursuit_evasion/`, IDs `Rate-PursuitEvasion` (ACRO), `Att-PursuitEvasion` and `Vel-PursuitEvasion`.

Two Crazyflies per env. The pursuer (`Robot`) learns with PPO to collide with
the evader (`Evader`). The evader runs a scripted policy: a straight line
(speed sampled in `evader_speed_range`, vertical speed bouncing between
`evader_altitude_range`) or, with probability `evader_circle_prob`, a circle.
It sends these as velocity setpoints to its own cascaded PID, so both drones
have the same dynamics.

| Level | Action (in [-1, 1]) |
|---|---|
| `Rate` (ACRO) | roll rate, pitch rate, yaw rate, thrust |
| `Att` (angle mode) | roll, pitch, yaw rate, thrust |
| `Vel` | vx, vy, yaw rate, vz, in the heading frame |

### Observation space

The actor gets 16 values:

- `q_pursuer_evader` (4): the evader's orientation in the pursuer's frame, with w >= 0.
- Pursuer body rates (3), times `ang_vel_obs_scale`.
- Camera detection (4): `visible`, `u`, `v`, `size`.
- Time since the last detection (1), saturated at 1 s.
- Last action (4).

The camera looks along body +x. `detection.py` projects the evader through a
pinhole model (`hfov_deg`, `vfov_deg`, `max_range`) and returns the
normalised box centre `u` (right) and `v` (up), and the box width `size`. The
default detector is ideal. `pixel_noise_std`, `size_noise_rel_std` and
`miss_prob` make it imperfect without changing the layout.

The critic gets a privileged 26-D `"critic"` state:

- world position, relative to the env origin, of both drones;
- world linear velocity of both drones;
- quaternion of both drones;
- body angular velocity of both drones.

### Reward

The shaping uses only what the camera reports, never the true distance.

- **progress**: the decrease of the range implied by the box size
  (`Detector.size_to_range`) between two consecutive detections. It is clipped
  to `max_progress_per_step`.
- **centering**: `1 - (u² + v²) / 2` while the evader is visible.
- **lost**: a penalty while the evader is not visible.

There are also small penalties on body rates and on the action rate. Capture
gives `+capture_reward` and a crash gives `crash_reward`.

### Termination

- **Success:** distance < `capture_radius` (a collision).
- **Crash:** the pursuer goes below `min_altitude`, above `max_altitude`, or
  outside `arena_radius`.
- **Truncation:** time-out, or the evader leaving the arena.

The episode logs report `Metrics/capture_rate`, `Metrics/min_distance` and
`Metrics/visible_fraction`.

### Training with Orcax

`scripts/orcax/train.py` trains with Orcax PPO (JAX). The critic is
asymmetric and reads the `"critic"` state. Hyperparameters come from
`agents/orcax_ppo_cfg.yaml`. On the training machine, Orcax is not installed in
the Isaac Lab venv, so add the JAX overlay to the path:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=~/.venvs/isaac_orcax_overlay:~/Orcax/Orcax \
~/.venvs/isaac/bin/python scripts/orcax/train.py --task Rate-PursuitEvasion --num_envs 4096 --headless
# replay: add --play logs/orcax/pursuit_evasion/<run>/checkpoints/best.pkl
```

## Training

<!-- skrl / rsl_rl / sb3 / rl_games train commands -->

## Play / evaluate

<!-- play.py, zero_agent.py, random_agent.py -->
