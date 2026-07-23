# Sim-to-real evaluation — Crazyflie hovering (velocity controller)

Tooling to measure the reality gap between the IsaacLab simulator and the real
Crazyflie 2.1 on the hovering task, following the protocol in
`sim2real_protocol_crazyflie.docx` (metric names after Aljalbout et al., *The
Reality Gap in Robotics*, 2026).

This directory covers the **velocity** controller end to end: train a policy,
fly it on the real drone while logging everything, and compute the hover
metrics. Report each controller on its own — never average across controllers.

## Files

| File | Purpose |
|---|---|
| `collect_hover_vel.py` | Flies the velocity hover policy on the real Crazyflie **and records full telemetry to disk** (extends `../single_drone_exec/hover/exec_vel.py`). |
| `analyze_hover.py` | Reads collected flights and computes the protocol hover metrics; aggregates over trials. |
| `data/<run>/` | Output of a collection run: `flight.csv`, `flight.npz`, `metadata.json` (+ `flight.png` with `--plot`). |

## 1. Train a deployable velocity policy

The real-drone script loads a **skrl** PPO checkpoint, so train with skrl (not
rsl_rl). The architecture (`[32,32]` ELU + `RunningStandardScaler`) and the
6-dim obs / 3-dim action are fixed by `exec_vel.py` / `vel_hovering.py`.

```bash
# from the repo root, using your Isaac Lab python env
WANDB_MODE=disabled python scripts/skrl/train.py --task=Vel-Hovering --num_envs=4096 --headless --seed 42
```

Checkpoints land in `logs/skrl/cartpole_direct/<timestamp>_PPO_torch/checkpoints/`
(`best_agent.pt` and `agent_<N>.pt`). That `.pt` is loaded directly by both
`exec_vel.py` and `collect_hover_vel.py` — no export step is needed.

> Notes
> - `WANDB_MODE=disabled` avoids a Weights & Biases login hang; the shipped
>   `skrl_ppo_cfg.yaml` has `wandb: True` with a hardcoded entity.
> - `experiment_name` is a leftover `"cartpole_direct"`; pass a cleaner path if
>   you like. `--max_iterations N` sets skrl `timesteps = N * rollouts(32)`.

## 2. Collect real-flight data (freeze everything first)

```bash
pip install -e "source/CrazyPlayGround[deploy]"   # cflib, once

python execution/sim2real/collect_hover_vel.py \
    --checkpoint logs/skrl/cartpole_direct/<run>/checkpoints/best_agent.pt \
    --target 0 0 1.0 \
    --duration 20 \
    --tag trial01 --trial-id 1 --seed 42
```

The drone takes off, holds the fixed target for `--duration` seconds (protocol:
freeze goals), then auto-lands. Every control step (100 Hz) one time-aligned row
is written with: the policy observation, the raw action, the velocity command
**sent**, EKF position/velocity/quaternion, raw gyro + accelerometer,
`motor.m1..m4`, `pm.vbat`, and Kalman variance. `metadata.json` freezes the
checkpoint hash, git commit, rates, frame/quaternion conventions, thresholds and
library versions so the run is reproducible.

Useful flags: `--roam` (resample target when reached, the old `exec_vel.py`
behaviour — off by default), `--log-rate-hz 50` (halve the fast log blocks if
the radio drops packets), `--uri`, `--outdir`.

Run 20+ trials per cell for CL-1, 10+ logs for OL-1 (protocol).

### Output schema (`flight.csv`)

`t_mono, t_wall, step` · `obs_vb_{x,y,z}, obs_errb_{x,y,z}` ·
`act_{x,y,z}` (clamped [-1,1]) · `cmd_{vx,vy,vz}, cmd_yawrate` (sent) ·
`tgt_{x,y,z}` · `pos_{x,y,z}` · `vel_{x,y,z}` (world) · `qw,qx,qy,qz` ·
`gyro_{x,y,z}` (deg/s) · `acc_{x,y,z}` (g) · `m1..m4` (PWM) · `vbat` (V) ·
`varP{X,Y,Z}` · `dist_to_target`.

## 3. Compute metrics

```bash
# one trial
python execution/sim2real/analyze_hover.py execution/sim2real/data/<run> --plot
# many trials -> per-trial + mean ± spread
python execution/sim2real/analyze_hover.py execution/sim2real/data/*_vel --json-out report.json
```

Reports position error (mean/RMS), max drift, horizontal/vertical split, jitter,
mean speed, control smoothness, mean motor PWM + battery correlation, settling
time, estimator quality, and per-trial crash/success plus an aggregate
success/crash rate.

## How the pieces map to the protocol

| Test | How this tooling supports it |
|---|---|
| **OL-1** Replay error | `cmd_*` + start state in `flight.csv` are the exact real actions to replay in sim; compare drift per state group (pos / vel / attitude / body rate). |
| **OL-3** Actuator check | `m1..m4` vs `vbat` across battery levels; check `max_thrust_N=0.638` and the thrust scaling. |
| **OL-4** State estimation | Logged EKF state (`pos/vel/quat`, `varP*`) vs motion capture → EKF noise/bias/delay to inject into sim obs. |
| **CL-1** Hover gap | `analyze_hover.py` metrics in sim vs real, same goal/seed, 20+ trials; report the performance gap + success rate. |
| **CL-2** SRCC | Collect with several checkpoints/seeds; correlate real vs sim score. |

## Conventions (must stay matched sim ↔ real)

- Quaternion order `(qw, qx, qy, qz)`, scalar first.
- Velocity is world-frame (`stateEstimate.v*`) rotated into body frame via
  `quat_inv` — matches `root_lin_vel_b` in `vel_hovering.py`.
- 100 Hz policy, 500 Hz inner loop, decimation 5. Action `[-1,1]³ × 1.0 m/s`,
  sent as `send_velocity_world_setpoint(vx, vy, vz, yaw_rate=0)`.
- Velocity/position controllers run the **firmware cascade PID**, so their gap
  mixes in a controller difference — do not use them to judge the propeller
  model (CTBR/attitude test that most). Match the 20 Hz gyro filter on both sides.
