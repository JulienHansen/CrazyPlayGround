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

## Training

<!-- skrl / rsl_rl / sb3 / rl_games train commands -->

## Play / evaluate

<!-- play.py, zero_agent.py, random_agent.py -->

### Evaluating a tracking policy

`play.py --eval_csv out.csv` flies one full episode per env and, for the tracking envs, records the
tracking env's states every step (`record_states`): drone path, scored reference, velocity and
actions. With `env.traj_yaw_range=[0.0,0.0] env.traj_phase_range=[0.0,0.0]` and single-valued speed
and scale ranges, every env flies the same reference; with `env.spawn_xy_offset=0.0
env.spawn_z_offset=0.0` every drone also starts on it (by default the spawn is offset by up to
±0.3 m in x/y and ±0.2 m in z). DR stays off unless its `env.*` knobs are set.
`sim_eval/single_drone_exec/traj_tracking/plot_trajectory_eval.py` turns the recordings into a top
view of the path with the reference, the error over time, and the per-episode RMSE (tables: also
its along-track (lag) and cross-track parts). Usage: `sim_eval/README.md`.
