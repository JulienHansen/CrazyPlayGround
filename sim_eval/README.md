# Policy Evaluation Tools

Offline evaluation of trained policies: metrics and figures computed from recorded flights or from
simulated rollouts. Nothing in this directory flies a drone. The layout mirrors
`execution/single_drone_exec/`, so each tool sits next to the flight script it evaluates.

## Single drone, hover (`single_drone_exec/hover/`)
- **`analyze_vel.py`**: Hover metrics of flights recorded by `execution/single_drone_exec/hover/exec_vel.py`
  (position error, jitter, drift, settling time, command smoothness, motor PWM, crash/success).
  Needs numpy and pyarrow; matplotlib only for `--plot`.

```bash
# Run from CrazyPlayGround/
python sim_eval/single_drone_exec/hover/analyze_vel.py flight.parquet
python sim_eval/single_drone_exec/hover/analyze_vel.py logs/*.parquet --skip-s 3 --json-out report.json
```

## Single drone, trajectory tracking (`single_drone_exec/traj_tracking/`)
Evaluate with `scripts/skrl/play.py --eval_csv`, which also records every step to a `.parquet` next to the CSV,
then plot it with **`plot_trajectory_eval.py`** (numpy, pyarrow, matplotlib): per label a top view and a
time-series figure, plus a summary across labels. Lines are IQMs, bands 10–90 %; same label = pooled.

```bash
python scripts/skrl/play.py --task Vel-Tracking --headless --checkpoint <ckpt> --eval_csv out/run.csv [env.<field>=<value> ...]
python sim_eval/single_drone_exec/traj_tracking/plot_trajectory_eval.py out/*.parquet [--name <name>]
```

| Argument | Meaning |
|---|---|
| `--eval_csv out/run.csv` | one full episode per env: metrics in `run.csv`, every step in `run.parquet` |
| `--num_envs N`, `--seed S` | episodes; same seed and N = same spawns and draws |
| `env.obs_input_hist=8 env.action_latency_steps=2 ...` | the run's own settings (must match training) |
| `env.traj_speed_range=[0.4,0.4] env.traj_scale_range=[1.5,1.5]` | one reference speed and size for every env |
| `env.traj_yaw_range=[0.0,0.0] env.traj_phase_range=[0.0,0.0]` | same orientation and start point for every env |
| `env.spawn_xy_offset=0.0 env.spawn_z_offset=0.0` | every drone spawns on the reference's start point (training default: ±0.3 m in x/y, ±0.2 m in z) |
| `env.episode_length_s=50` | about 3 laps at speed 0.4 (one lap = 2π / speed ≈ 16 s) |
| `env.add_noise=true env.noise_std=0.03 env.wind_force_range=[0.0,0.05] env.action_latency_max_steps=6` | DR; off by default |
| plot: `--labels A B`, `--out DIR`, `--env I`, `--overlay` | legend names (same label = pooled), output folder, one episode, all labels in one figure |

System identification (step excitation, delay and noise) lives in [`../sim2real/`](../sim2real/README.md).
