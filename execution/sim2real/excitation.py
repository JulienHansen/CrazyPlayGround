"""The step excitation shared by the real flight (excite_hover.py) and its Isaac Lab twin (excite_hover_sim.py).

Standard library only at import (pyarrow is imported when writing), no cflib and no Isaac Lab,
so both scripts and the tests run the exact same signal, command law, CLI and recording format,
and the simulated flight can check its flags before Isaac Sim starts.

From EXC_START_S on, the excited axis repeats a cycle of three phases (see step_signal):

    REST     step_s     exc 0      P hold off: the drone settles at rest under a zero command
    PUSH     step_s     exc amp    P hold off: the step, from rest, always in the sign of amp
    RECOVER  recover_s  exc 0      P hold on: the drone brakes and flies back to its hover target
"""

import os
import subprocess
import sys

INTERVAL = 0.01          # 100 Hz, the policy step (sim dt=1/500, decimation=5)
MAX_VEL = 1.0            # m/s, vel_hovering.py max_velocity
KP, HOLD_MAX = 1.0, 0.3  # 1/s, m/s: holds the unexcited axes on target
EXC_START_S = 3.0        # after excite_hover.py's GRACE_PERIOD: the z-floor watchdog is armed first
EXC_MAX_AMP = 1.0        # m/s, MAX_VEL: the command is clamped there anyway


def step_signal(axis: int, amp: float, step_s: float, recover_s: float, t: float) -> tuple:
    """(step_vel, hold_on_exc_axis) on `axis`, `t` s into the flight.

    Cycles REST / PUSH (step_s each, P hold off on `axis`) then RECOVER (recover_s, P hold on).
    The PUSH is always `amp`, so every step is in one direction. Before EXC_START_S: no step and
    the P hold is on.
    Returns (step_vel, hold_on_exc_axis): step_vel is a length-3 list of world velocities [m/s],
    non-zero only on `axis` during PUSH; hold_on_exc_axis is whether the P hold acts on `axis`.
    """
    step_vel = [0.0, 0.0, 0.0]
    s = t - EXC_START_S
    if s < 0:
        return step_vel, True
    phase_t = s % (2 * step_s + recover_s)
    if step_s <= phase_t < 2 * step_s:
        step_vel[axis] = amp
    return step_vel, phase_t >= 2 * step_s


def excitation_command(pos_error, exc_axis: int, step_vel, hold_on_exc_axis: bool) -> tuple:
    """The world velocity to send, built from a position P hold plus the excitation step.

    pos_error: target - pos [m], tensor shaped [3] or [N, 3].
    exc_axis, step_vel, hold_on_exc_axis: the excited axis index and step_signal's output.

    Returns (hold_vel, setpoint_vel), tensors shaped like `pos_error`, in m/s:
      hold_vel      the P hold alone, KP * pos_error clamped to ±HOLD_MAX. It keeps the two
                    unexcited axes in place, and is zeroed on the excited axis outside RECOVER.
      setpoint_vel  what is sent to the drone or env: hold_vel + step_vel, clamped to ±MAX_VEL.
                    On the excited axis during REST and PUSH it is exactly step_vel: a pure step.
    """
    hold_vel = (KP * pos_error).clamp(-HOLD_MAX, HOLD_MAX)
    if not hold_on_exc_axis:
        hold_vel[..., exc_axis] = 0.0
    setpoint_vel = (hold_vel + hold_vel.new_tensor(step_vel)).clamp(-MAX_VEL, MAX_VEL)
    return hold_vel, setpoint_vel


def add_excitation_args(parser) -> None:
    """The excitation flags, identical for the real and the simulated flight."""
    parser.add_argument("--exc-axis", choices=["x", "y", "z"], default="x",
                        help="World axis to excite. Prefer x/y: z from a 0.5 m hover nears the 0.1 m floor cut-out.")
    parser.add_argument("--exc-amp", type=float, default=0.2, help=f"Step amplitude [m/s], |amp| at most {EXC_MAX_AMP}. Its sign is the push direction.")
    parser.add_argument("--exc-step", type=float, default=1.0,
                        help="Seconds of each REST and PUSH phase, in [0.1, 5]. Longer than the step's t90.")
    parser.add_argument("--exc-recover", type=float, default=4.0,
                        help="Seconds of P hold back to the target between steps. ~6 s at --exc-amp 1.")
    parser.add_argument("--duration", type=float, default=90.0, help="Seconds of flight before landing.")


def check_excitation_args(parser, args) -> None:
    """parser.error on an excitation the estimator cannot use."""
    if args.duration <= EXC_START_S:
        parser.error(f"--duration must be above {EXC_START_S:g} s, when the excitation starts")
    if not 0 < abs(args.exc_amp) <= EXC_MAX_AMP or not 0.1 <= args.exc_step <= 5 or args.exc_recover < 0:
        parser.error(f"|--exc-amp| must be in (0, {EXC_MAX_AMP}] m/s, --exc-step in [0.1, 5] s "
                     f"and --exc-recover >= 0 s")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def excitation_metadata(args) -> dict:
    """What was excited, stored in the recording's Parquet footer."""
    return {"argv": " ".join(sys.argv), "target": str(args.target), "duration_s": str(args.duration),
            "excitation": "push_recover", "exc_axis": args.exc_axis, "exc_amp": str(args.exc_amp),
            "exc_step_s": str(args.exc_step), "exc_recover_s": str(args.exc_recover), "kp": str(KP),
            "git_commit": git_commit()}


def excitation_row(step: int, control_time: float, pos, vel, quat, target, pos_error, hold_vel, setpoint_vel,
                   step_vel, echo_vel) -> dict:
    """One control step under excite_hover.py's column names (obs_age_s is 0: no radio).
    Vectors are length-3 sequences, `quat` is (w, x, y, z). Columns: pos_error -> err_*,
    hold_vel -> hold_v*, setpoint_vel -> vx/vy/vz, step_vel -> exc_v*, echo_vel -> echo_v*."""
    row = {"step": int(step), "control_time": float(control_time), "obs_age_s": 0.0}
    for prefix, values in (("pos_", pos), ("vel_", vel), ("target_", target), ("err_", pos_error)):
        row.update({prefix + ax: float(v) for ax, v in zip("xyz", values)})
    row.update({"quat_" + c: float(v) for c, v in zip("wxyz", quat)})
    for prefix, values in (("hold_v", hold_vel), ("v", setpoint_vel), ("exc_v", step_vel), ("echo_v", echo_vel)):
        row.update({prefix + ax: float(v) for ax, v in zip("xyz", values)})
    return row


def write_recording(path: str, rows: list, metadata: dict) -> None:
    """Write excitation_row rows to Parquet as FlightRecorder does: `step` int64, the rest float64,
    `metadata` as str -> str in the footer."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    names = list(rows[0]) if rows else ["step"]
    table = pa.table({name: pa.array([row[name] for row in rows], type=pa.int64() if name == "step" else pa.float64())
                      for name in names})
    pq.write_table(table.replace_schema_metadata({str(k): str(v) for k, v in metadata.items()}), path)
