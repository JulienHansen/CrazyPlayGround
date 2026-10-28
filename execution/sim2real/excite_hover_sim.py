"""Isaac Lab twin of excite_hover.py: the same step excitation flown on the simulated plant (WP7).

No policy: as on the real drone, a position P hold plus the step of excitation.py is sent at
100 Hz as the world velocity setpoint. In Vel-Hovering that setpoint is the env action (times
max_velocity), tracked by the cascade PID at its velocity level, the sim twin of the firmware
velocity PID. The recording has excite_hover.py's columns, so measure_delay_noise.py and
plot_step_response.py read real and simulated flights alike.

    python execution/sim2real/excite_hover_sim.py --record-path flights/sim_excite_x.parquet \
        --exc-axis x --exc-amp 0.2 --duration 90 --headless [env.action_latency_steps=4 ...]
    python execution/sim2real/measure_delay_noise.py flights/sim_excite_x.parquet
    python execution/sim2real/plot_step_response.py flights/excite_x.parquet flights/sim_excite_x.parquet

Trailing key=value arguments are Hydra overrides of the env, e.g. the env.* line that
measure_delay_noise.py prints. Row k holds the state read before command k, as on the drone, and
echo_v* is the setpoint the controller used at step k, after env.action_latency_steps: the
estimator's radio delay d is that latency, and its dead time is the simulated plant's.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# excitation.py sits next to this script; appended so it cannot shadow installed modules.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from excitation import (EXC_START_S, step_signal, excitation_command, add_excitation_args,  # noqa: E402
                        check_excitation_args, excitation_metadata, excitation_row, write_recording)

parser = argparse.ArgumentParser(description="Fly excite_hover.py's step excitation on the simulated plant, "
                                             "recorded for measure_delay_noise.py.")
parser.add_argument("--task", type=str, default="Vel-Hovering", help="Velocity-action task to excite.")
parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                    help="Hover target [x, y, z] relative to the env origin, z in [0.5, 1.5] as on the drone.")
parser.add_argument("--record-path", type=str, required=True, help="Parquet file to write.")
parser.add_argument("--seed", type=int, default=42, help="Env seed.")
add_excitation_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
check_excitation_args(parser, args_cli)
if not 0.5 <= args_cli.target[2] <= 1.5:
    parser.error("--target z must be in [0.5, 1.5] m: the env ends the episode below 0.1 m and above 2 m")
sys.argv = [sys.argv[0]] + hydra_args  # what is left is for Hydra

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import CrazyPlayGround.tasks  # noqa: F401

ENV_KNOBS = ("max_velocity", "action_latency_steps", "action_latency_max_steps", "add_noise", "noise_std",
             "noise_corr", "noise_bias_std", "wind_force_range", "thrust_scale_range", "torque_scale_range",
             "drone_config_path")


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    # one env, no timeout reset mid-flight
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = args_cli.duration + 5.0
    env_cfg.seed = args_cli.seed
    env_cfg.debug_vis = False
    dt = env_cfg.sim.dt * env_cfg.decimation
    # the estimator finds d where echo_v* first equals the step: the latency must end within REST
    latency_steps = max(env_cfg.action_latency_steps, env_cfg.action_latency_max_steps)
    if latency_steps * dt >= args_cli.exc_step:
        raise SystemExit(f"env latency {latency_steps} steps must be shorter than --exc-step {args_cli.exc_step:g} s")

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()

    # start at rest on the target, the hover the real drone reaches before EXC_START_S
    origin = u._terrain.env_origins[0]
    target = torch.tensor(args_cli.target, dtype=torch.float32, device=u.device)
    target_w = origin + target
    root_state = u._robot.data.default_root_state.clone()
    root_state[:, :3] = target_w
    u._robot.write_root_pose_to_sim(root_state[:, :7])
    u._robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    # reset(None) also zeroes the controller's step counter, so the velocity setpoint is latched at
    # the first physics substep of every env step: a command acts for the whole 10 ms, as on the drone
    u._ctrl.reset(None)
    assert u._ctrl._step_count % env_cfg.decimation == 0
    u.episode_length_buf[:] = 0  # _reset_idx randomises it to de-correlate training envs

    axis = "xyz".index(args_cli.exc_axis)
    n_steps = int(round(args_cli.duration / dt))
    rows, ended_at = [], None
    print(f"[INFO] Holding {args_cli.target}, step excitation on axis {args_cli.exc_axis} from {EXC_START_S:g} s, "
          f"{n_steps} steps of {dt * 1e3:g} ms")
    for k in range(n_steps):
        t = k * dt
        u._desired_pos_w[:] = target_w  # the goal only drives the observation and the marker
        # the state behind this step's command, read before it is applied
        pos = (u._robot.data.root_pos_w[0] - origin).clone()
        vel = u._robot.data.root_lin_vel_w[0].tolist()
        quat = u._robot.data.root_quat_w[0].tolist()  # (w, x, y, z)
        err = target - pos
        exc, hold_on = step_signal(axis, args_cli.exc_amp, args_cli.exc_step, args_cli.exc_recover, t)
        hold, cmd = excitation_command(err, axis, exc, hold_on)
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step((cmd / env_cfg.max_velocity).unsqueeze(0))
        echo = u._ref_vel[0].tolist()  # the setpoint the cascade PID tracked during this step
        rows.append(excitation_row(k, t, pos.tolist(), vel, quat, args_cli.target, err.tolist(),
                                   hold.tolist(), cmd.tolist(), exc, echo))
        if terminated.any() or truncated.any():
            print(f"[WARN] Episode ended at step {k} (z={pos[2].item():.2f} m) — keeping the {len(rows)} rows so far")
            ended_at = k
            break

    metadata = {"source": "sim", "task": args_cli.task, "seed": args_cli.seed, "sim_dt": env_cfg.sim.dt,
                "decimation": env_cfg.decimation, "ended_at_step": ended_at, "torch_version": torch.__version__,
                **{f"env.{knob}": getattr(env_cfg, knob, None) for knob in ENV_KNOBS},
                **excitation_metadata(args_cli)}
    if os.path.dirname(args_cli.record_path):
        os.makedirs(os.path.dirname(args_cli.record_path), exist_ok=True)
    write_recording(args_cli.record_path, rows, metadata)
    print(f"[INFO] Wrote {len(rows)} rows to {args_cli.record_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
