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

The actor gets 19 values, only what an IMU and the camera can estimate. There is
no world yaw (unobservable without a magnetometer), no position and nothing
about the evader's state:

- Gravity in the body frame (3): roll and pitch from an IMU attitude filter.
- Pursuer body rates from the gyro (3), times `ang_vel_obs_scale`.
- Camera detection (4): `visible`, `u`, `v`, `size`.
- Line of sight (3): the camera bearing to the box centre, from `u` and `v`,
  in the heading frame (level, yawing with the drone), using roll and pitch only.
  Zero when not visible.
- Expansion (1): the box log-growth since the previous detection, scaled to [-1, 1]
  (a 1/time-to-contact cue).
- Time since the last detection (1), saturated at 1 s.
- Last action (4).

The camera looks along body +x. `detection.py` projects the evader through a
pinhole model (`hfov_deg`, `vfov_deg`, `max_range`) and returns the
normalised box centre `u` (right) and `v` (up), and the box width `size`. The
default detector is noisy: `pixel_noise_std` = 0.01 (about 3 px of centre
jitter), `size_noise_rel_std` = 0.05 and `miss_prob` = 0.05. Set them to 0
for an ideal detector.

The IMU inputs are noisy too: the roll/pitch estimate has a per-episode bias
(up to `attitude_bias_max_deg`) plus white noise (`attitude_noise_std_deg`),
and the gyro has a per-episode bias (`gyro_bias_max`) plus white noise
(`gyro_noise_std`). The noise is on the policy inputs only; the rate
controller and the critic use the true state.

The critic gets a privileged 26-D `"critic"` state:

- world position, relative to the env origin, of both drones;
- world linear velocity of both drones;
- quaternion of both drones;
- body angular velocity of both drones.

### Reward

The shaping uses only what the camera reports, never the true distance. The
target size is unknown: the detector's target radius is resampled every
episode in `target_radius_range`, so the box size alone does not give a range.

- **progress** (time-to-contact): the growth of the box log-size, filtered with
  an exponential filter (`size_filter_alpha`, held through missed frames),
  `log(size_t / size_{t-1})`. For small angles this equals
  `log(d_{t-1} / d_t)`, so it needs no target size. Its rate is the inverse
  time-to-contact. It is clipped to `max_log_growth_per_step`.
- **centering**: `1 - (u² + v²) / 2` while the evader is visible.
- **lost**: a penalty while the evader is not visible.

There are also small penalties on body rates and on the action rate. Capture
gives `+capture_reward` and a crash gives `crash_reward`.

### Termination

- **Success:** a contact sensor on the pursuer, filtered to the evader's bodies,
  reports a contact force above `capture_contact_force` in any physics substep
  (a collision).
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
