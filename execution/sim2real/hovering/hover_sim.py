"""Evaluate a trained velocity-control hover policy in Isaac Lab under sim2real conditions.

Flies an RL checkpoint from a user-specified starting position to a target hover setpoint
for a duration of --time seconds, computing step-response metrics (tau, rise time, overshoot,
settling time, steady-state error) and delegating plot generation to plot_hover.py.

Every run is recorded to --record-path (required), in exec_vel.py's format: the same columns
(control_time, pos_*, vel_*, target_*, vx/vy/vz, ...) plus `source: sim` in the Parquet footer, so
plot_hover.py reads a sim run and a real flight the same way and can overlay them (--compare).

Usage:
    python execution/sim2real/hovering/hover_sim.py --checkpoint logs/.../best_agent.pt \
        --start-pos 0.5 0.0 1.0 --target 0.0 0.0 1.0 --time 10.0 --no-video \
        --ylim -0.05 0.55 --outdir outputs/sim2real/test --record-path outputs/sim2real/test/sim.parquet

    # matched to a real exec_vel.py flight (which starts at the 0.5 m takeoff height), for --compare
    python execution/sim2real/hovering/hover_sim.py --checkpoint logs/.../best_agent.pt \
        --start-pos 0.0 0.0 0.5 --target 0.5 0.0 1.0 --time 30 --no-video \
        --record-path flights/sim_x0.5_z1.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np

# Import plotting routines from companion module
try:
    from .plot_hover import plot_step_response, plot_target_error
except ImportError:
    from plot_hover import plot_step_response, plot_target_error


def hover_row(
    step: int,
    t: float,
    pos: Sequence[float],
    vel: Sequence[float],
    quat_wxyz: Sequence[float],
    target: Sequence[float],
    action: Sequence[float],
    cmd: Sequence[float],
) -> dict:
    """One control step under the column names FlightRecorder uses for exec_vel.py.

    Args:
        step: Control-step index.
        t: Seconds since the first step; stored as `control_time` (a real flight stores epoch
            seconds there, and plot_hover.py subtracts the first value from both).
        pos: (3,) position relative to the env origin, world frame, m.
        vel: (3,) linear velocity, world frame, m/s.
        quat_wxyz: (4,) attitude quaternion (w, x, y, z).
        target: (3,) target position, world frame, m.
        action: (3,) policy action in [-1, 1] (`action_v*`).
        cmd: (3,) commanded velocity, world frame, m/s (`vx/vy/vz`): action * max_velocity.

    Returns:
        The row as {column: value}; `obs_age_s` is 0.0 since the sim has no radio.
    """
    row = {"step": int(step), "control_time": float(t), "obs_age_s": 0.0}
    for prefix, values in (("pos_", pos), ("vel_", vel)):
        row.update({prefix + ax: float(v) for ax, v in zip("xyz", values, strict=True)})
    row.update({"quat_" + c: float(v) for c, v in zip("wxyz", quat_wxyz, strict=True)})
    for prefix, values in (("target_", target), ("action_v", action), ("v", cmd)):
        row.update({prefix + ax: float(v) for ax, v in zip("xyz", values, strict=True)})
    return row


def write_recording(path: str, rows: list[dict], metadata: dict) -> None:
    """Write hover_row rows to Parquet as FlightRecorder does: `step` int64, the rest float64.

    Twin of `write_recording` in system-identification/excitation.py (that folder is not a package).

    Args:
        path: Parquet file to write; its folder is created if needed.
        rows: Rows from `hover_row`, all with the same columns.
        metadata: Footer key/values, stored as str -> str.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    names = list(rows[0]) if rows else ["step"]
    table = pa.table(
        {name: pa.array([row[name] for row in rows], type=pa.int64() if name == "step" else pa.float64())
         for name in names}
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pq.write_table(table.replace_schema_metadata({str(k): str(v) for k, v in metadata.items()}), path)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for hover simulation."""
    parser = argparse.ArgumentParser(
        description="Run an RL hover policy checkpoint in Isaac Lab and generate step-response evaluation plots."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the skrl agent checkpoint (.pt file).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Vel-Hovering",
        help="Task name (default: Vel-Hovering).",
    )
    parser.add_argument(
        "--start-pos",
        nargs=3,
        type=float,
        default=[0.5, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Initial drone position in meters [x, y, z] (default: 0.5 0.0 1.0).",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Target hover position in meters [x, y, z] (default: 0.0 0.0 1.0).",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=10.0,
        help="Simulation flight duration in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--video",
        dest="video",
        action="store_true",
        default=True,
        help="Enable video recording (requires a node with display capabilities).",
    )
    parser.add_argument("--no-video", dest="video", action="store_false", help="Disable video recording.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/sim2real",
        help="Output directory for generated plots and video (default: outputs/sim2real).",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="Custom path to save the detailed step-response plot (defaults to <outdir>/step_response.png).",
    )
    parser.add_argument(
        "--error-plot-path",
        type=str,
        default=None,
        help="Custom path to save the simplified target error plot (defaults to <outdir>/target_error.png).",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Custom y-axis limits for the target error plot (e.g. --ylim -0.05 0.55).",
    )
    parser.add_argument(
        "--error-ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Custom y-axis limits for the target error plot.",
    )
    parser.add_argument(
        "--pos-ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Custom y-axis limits for position subplots in the step-response plot.",
    )
    parser.add_argument(
        "--vel-ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Custom y-axis limits for velocity subplots in the step-response plot.",
    )
    parser.add_argument(
        "--record-path",
        type=str,
        required=True,
        help="Parquet file to write the flight to, in exec_vel.py's format (read by plot_hover.py).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser


# ── Execution with Isaac Lab runtime ──────────────────────────────────────────

if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    if not 0.1 <= args_cli.target[2] <= 2.0:
        parser.error("--target z must be between 0.1 m and 2.0 m")
    if not 0.1 <= args_cli.start_pos[2] <= 2.0:
        parser.error("--start-pos z must be between 0.1 m and 2.0 m")

    if args_cli.video:
        args_cli.enable_cameras = True

    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    from skrl.utils.runner.torch import Runner

    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import CrazyPlayGround.tasks  # noqa: F401
    from CrazyPlayGround.tasks.direct.hovering.step_response import step_response

    @hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
    def run_simulation(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.episode_length_s = args_cli.time + 5.0
        env_cfg.seed = args_cli.seed
        agent_cfg["seed"] = args_cli.seed
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        agent_cfg["agent"]["experiment"]["wandb"] = False

        dt = env_cfg.sim.dt * env_cfg.decimation
        n_steps = int(round(args_cli.time / dt))
        max_velocity = float(getattr(env_cfg, "max_velocity", 1.0))

        outdir = os.path.abspath(args_cli.outdir)
        os.makedirs(outdir, exist_ok=True)

        render_mode = "rgb_array" if args_cli.video else None
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

        if args_cli.video:
            video_kwargs = {
                "video_folder": outdir,
                "step_trigger": lambda step: step == 0,
                "video_length": n_steps,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = SkrlVecEnvWrapper(env, ml_framework="torch")

        runner = Runner(env, agent_cfg)
        checkpoint_path = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO] Loading model checkpoint: {checkpoint_path}")
        runner.agent.load(checkpoint_path)
        runner.agent.enable_training_mode(False)

        u = env.unwrapped
        device = u.device
        env.reset()

        origin = u._terrain.env_origins[0]
        start_w = origin + torch.tensor(args_cli.start_pos, dtype=torch.float32, device=device)
        target_w = origin + torch.tensor(args_cli.target, dtype=torch.float32, device=device)

        root_state = u._robot.data.default_root_state.clone()
        root_state[:, :3] = start_w
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        root_state[:, 7:] = 0.0
        u._robot.write_root_pose_to_sim(root_state[:, :7])
        u._robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
        u._ctrl.reset(None)
        u.episode_length_buf[:] = 0
        u._desired_pos_w[:] = target_w

        obs = u._get_observations()["policy"]

        times = []
        positions = []
        velocities = []
        cmd_velocities = []
        rows = []
        ended_at = None

        print(
            f"[INFO] Simulating hover: start={args_cli.start_pos}, target={args_cli.target}, "
            f"duration={args_cli.time}s ({n_steps} steps at {dt * 1e3:.1f}ms)"
        )

        for step in range(n_steps):
            u._desired_pos_w[:] = target_w

            with torch.inference_mode():
                outputs = runner.agent.act(obs, None, timestep=step, timesteps=n_steps)
                if isinstance(outputs, tuple):
                    action = outputs[-1].get("mean_actions", outputs[0])
                elif isinstance(outputs, dict):
                    action = outputs.get("mean_actions", outputs.get("actions", outputs))
                else:
                    action = outputs
                action = action.clamp(-1.0, 1.0)

            pos = (u._robot.data.root_pos_w[0] - origin).clone()
            vel = u._robot.data.root_lin_vel_w[0].clone()
            action_row = action[0] if action.ndim > 1 else action
            action_np = action_row[:3].cpu().numpy()  # policy action in [-1, 1]
            cmd = action_np * max_velocity  # commanded velocity, world frame, m/s (already NumPy)
            t = round(step * dt, 4)

            times.append(t)
            positions.append(pos.cpu().numpy())
            velocities.append(vel.cpu().numpy())
            cmd_velocities.append(cmd)

            quat_wxyz = u._robot.data.root_quat_w[0].tolist()  # Isaac Lab 2.x order
            rows.append(hover_row(step, t, pos.tolist(), vel.tolist(), quat_wxyz, args_cli.target, action_np, cmd))

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated.any() or truncated.any():
                print(f"[WARN] Episode ended at step {step}/{n_steps} (z={pos[2].item():.2f} m)")
                ended_at = step
                break

        env.close()

        times_arr = np.array(times)
        positions_arr = np.array(positions)
        velocities_arr = np.array(velocities)
        cmd_velocities_arr = np.array(cmd_velocities)
        target_arr = np.array(args_cli.target)

        # ── Step-response metrics printed to console ─────────────────────────
        pos_tensor = torch.from_numpy(positions_arr).unsqueeze(0).to(device)
        start_tensor = torch.tensor([args_cli.start_pos], dtype=torch.float32, device=device)
        goal_tensor = torch.tensor([args_cli.target], dtype=torch.float32, device=device)
        length_tensor = torch.tensor([len(positions_arr)], dtype=torch.long, device=device)

        metrics = step_response(
            pos_w=pos_tensor,
            start_pos_w=start_tensor,
            goal_pos_w=goal_tensor,
            length=length_tensor,
            dt=dt,
        )

        d0 = metrics["d0"].item()
        tau = metrics["tau"].item()
        rise = metrics["rise_10_90"].item()
        overshoot = metrics["overshoot"].item()
        settling = metrics["settling_time"].item()
        ss_err = metrics["steady_state_error"].item()

        print("\n=== STEP RESPONSE SUMMARY ===")
        print(f"  Displacement d0    : {d0:.4f} m")
        print(f"  Tau (63.2% rise)   : {tau:.4f} s" if not np.isnan(tau) else "  Tau (63.2% rise)   : NaN")
        print(f"  Rise time (10-90%) : {rise:.4f} s" if not np.isnan(rise) else "  Rise time (10-90%) : NaN")
        print(f"  Overshoot          : {overshoot * 100:.2f} %" if not np.isnan(overshoot) else "  Overshoot          : NaN")
        print(f"  Settling time (5%) : {settling:.4f} s" if not np.isnan(settling) else "  Settling time (5%) : NaN")
        print(f"  Steady-state error : {ss_err:.4f} m" if not np.isnan(ss_err) else "  Steady-state error : NaN")
        print("==============================\n")

        # ── Plot step response ──────────────────────────────────────────────
        plot_path = args_cli.plot_path or os.path.join(outdir, "step_response.png")
        plot_step_response(
            times=times_arr,
            positions=positions_arr,
            targets=target_arr,
            velocities=velocities_arr,
            cmd_velocities=cmd_velocities_arr,
            out_path=plot_path,
            pos_ylim=args_cli.pos_ylim,
            vel_ylim=args_cli.vel_ylim,
        )

        # ── Plot simplified target error ────────────────────────────────────
        error_plot_path = args_cli.error_plot_path or os.path.join(outdir, "target_error.png")
        error_ylim = args_cli.error_ylim or args_cli.ylim
        plot_target_error(
            times=times_arr,
            positions=positions_arr,
            targets=target_arr,
            out_path=error_plot_path,
            ylim=error_ylim,
        )

        # ── Write the Parquet recording (exec_vel.py's format) ──────────────
        if rows:
            rec_path = os.path.abspath(args_cli.record_path)
            metadata = {
                "source": "sim",
                "task": args_cli.task,
                "checkpoint": checkpoint_path,
                "seed": args_cli.seed,
                "start_pos": list(args_cli.start_pos),
                "target": list(args_cli.target),
                "sim_dt": env_cfg.sim.dt,
                "decimation": env_cfg.decimation,
                "max_velocity": max_velocity,
                "ended_at_step": ended_at,
            }
            write_recording(rec_path, rows, metadata)
            print(f"[INFO] Wrote {len(rows)} telemetry rows to {rec_path}")

    try:
        run_simulation()
    finally:
        simulation_app.close()

