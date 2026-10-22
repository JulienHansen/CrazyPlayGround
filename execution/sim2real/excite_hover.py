"""Excitation flight for measure_delay_noise.py: a hover held from the host, pushed by a known signal.

A position P loop on the host holds the hover and sends world velocities at 100 Hz through
send_velocity_world_setpoint, the radio and firmware path the velocity policy flies through,
so the loop delay and sensor noise measured here are the ones the policy meets. From
EXC_START_S on, a step (square wave) or chirp is added on one axis, and every row records
the sent command (vx/vy/vz), the excitation alone (exc_v*) and the measured velocity (vel_*).

    python execution/sim2real/excite_hover.py --record-path flight.parquet --exc-axis x --duration 30
    python execution/sim2real/measure_delay_noise.py flight.parquet
"""

import os
import sys
import time
import math
import importlib.metadata
import subprocess
import threading
import argparse
import logging

import torch

import cflib.crtp
from cflib.crazyflie import Crazyflie

# Make the `common` package importable; appended so it cannot shadow installed modules.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "single_drone_exec"))
from common.flight_recorder import CrazyflieStateBase  # noqa: E402
from common.utils import emergency_land, wait_until, add_run_args  # noqa: E402

device = torch.device("cpu")  # no model: nothing to gain from a GPU
logging.basicConfig(format="{asctime} [{levelname}] {message}", style="{",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("CrazyflieRL")

# ── Safety thresholds (as hover/exec_vel.py) ─────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without a position callback before emergency land
POS_VARIANCE_THRESHOLD = 0.5   # kalman position variance [m²] above which tracking is unreliable
GRACE_PERIOD           = 3.0   # seconds before enforcing the z lower bound

INTERVAL = 0.01      # 100 Hz, the policy step (sim dt=1/500, decimation=5)
SPIN_MARGIN = 0.002  # sleep to 2 ms before each deadline, busy-wait the rest (see wait_until)
MAX_VEL = 1.0        # m/s, vel_hovering.py max_velocity
KP, HOLD_MAX = 1.0, 0.3  # 1/s, m/s: a gentle hold, so the excitation dominates the command

# ── Excitation ───────────────────────────────────────────────────────────────
EXC_START_S = GRACE_PERIOD  # the z-floor watchdog is armed before the drone is pushed
EXC_MAX_AMP = 0.5           # m/s
CHIRP_F0, CHIRP_F1, CHIRP_S = 0.2, 5.0, 20.0  # Hz, Hz, s: the sweep repeats; dead time needs content up to ~4 Hz


def excitation_signal(kind: str, axis: int, amp: float, freq: float, t: float) -> list:
    """World velocity [m/s] added on `axis` to the hold command, `t` s into the flight.

    step is a square wave at `freq` (steps alternate, so the drone does not drift); chirp
    sweeps CHIRP_F0 -> CHIRP_F1 every CHIRP_S s and ignores `freq`. No sine: one frequency
    cannot separate the dead time from the lag. Zero before EXC_START_S.
    """
    exc = [0.0, 0.0, 0.0]
    s = t - EXC_START_S
    if s < 0:
        return exc
    if kind == "step":
        exc[axis] = math.copysign(amp, math.sin(2 * math.pi * freq * s))
    else:  # chirp
        s %= CHIRP_S
        exc[axis] = amp * math.sin(2 * math.pi * (CHIRP_F0 * s + (CHIRP_F1 - CHIRP_F0) * s * s / (2 * CHIRP_S)))
    return exc


class ExciteHover(CrazyflieStateBase):
    def __init__(self, uri: str, run_args, kind: str, axis: int, amp: float, freq: float,
                 duration: float, target=None, metadata=None):
        super().__init__(device)
        self.uri = uri
        self.cf = Crazyflie(rw_cache='./cache')
        self.kind, self.axis, self.amp, self.freq = kind, axis, amp, freq
        self.duration = duration
        self.initial_target = target
        self.make_recorder(run_args, obs_fields=["err_x", "err_y", "err_z"],
                           action_fields=["hold_vx", "hold_vy", "hold_vz"], cmd_fields=["vx", "vy", "vz"],
                           extra_fields=["exc_vx", "exc_vy", "exc_vz", "dt_body_ms", "slept_ms", "overran"])
        self.recorder.metadata.update(metadata or {})
        self.cf.connected.add_callback(self._connected)
        self.cf.connection_failed.add_callback(self._connection_lost)
        self.cf.connection_lost.add_callback(self._connection_lost)

    def _connected(self, uri: str):
        logger.info(f"Connected to {uri}, taking off...")
        self.cf.high_level_commander.takeoff(0.5, 1)
        time.sleep(1.5)
        self.setup_state_logging()
        threading.Thread(target=self.control_loop, daemon=True).start()

    def _connection_lost(self, uri: str, msg: str):
        logger.warning(f"Connection to {uri} lost or failed: {msg} — triggering safe landing")
        self.running = False

    def control_loop(self):
        self.recorder.start()
        logger.info("Waiting for first position estimate...")
        while not self.position_received and self.running:
            time.sleep(0.05)
        if not self.running:
            return
        if self.initial_target is not None:
            target = torch.tensor(self.initial_target, dtype=torch.float32, device=device)
        else:
            target = self.current_pos.clone()
            target[2] = max(0.5, min(1.5, target[2].item()))
        logger.info(f"Holding {target.tolist()}, {self.kind} excitation on axis {'xyz'[self.axis]} from {EXC_START_S:g} s")

        start = time.time()
        deadline = time.perf_counter()
        step = 0
        while self.cf.is_connected() and self.running:
            t_body = time.perf_counter()

            # ── Safety watchdog (as hover/exec_vel.py) ───────────────────────
            elapsed = time.time() - start
            if elapsed >= self.duration:
                logger.info(f"Flight duration {self.duration:g} s reached — landing")
                self.running = False
                break
            z = self.current_pos[2].item()
            if (elapsed > GRACE_PERIOD and z < 0.1) or z > 2.5:
                logger.error(f"Position out of bounds z={z:.2f} — emergency landing")
                emergency_land(self)
                break
            if self._last_pos_time > 0 and time.time() - self._last_pos_time > POS_STALE_TIMEOUT_S:
                logger.error(f"Position data stale ({time.time() - self._last_pos_time:.2f} s) — emergency landing")
                emergency_land(self)
                break
            with self.lock:
                var = self._pos_variance.clone()
            if var.max().item() > POS_VARIANCE_THRESHOLD:
                logger.error(f"Position variance too high {var.tolist()} — emergency landing")
                emergency_land(self)
                break

            state = self.snapshot()  # one consistent read, used for the command and its recorded row
            err = target - state.pos
            hold = (KP * err).clamp(-HOLD_MAX, HOLD_MAX)
            exc = excitation_signal(self.kind, self.axis, self.amp, self.freq, elapsed)
            cmd = (hold + torch.tensor(exc)).clamp(-MAX_VEL, MAX_VEL)
            control_time = time.time()
            self.cf.commander.send_velocity_world_setpoint(*cmd.tolist(), 0.0)  # yaw_rate = 0
            t_sent = time.perf_counter()

            deadline, slept_ms, overran = wait_until(deadline + INTERVAL, SPIN_MARGIN)
            self.record_step(step, state, control_time, target, err, hold, cmd,
                             exc + [(t_sent - t_body) * 1e3, slept_ms, overran])
            step += 1

        self.cf.commander.send_stop_setpoint()
        logger.info("Control loop stopped")

    def start(self):
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf.open_link(self.uri)

    def stop(self):
        logger.info("Stopping...")
        self.running = False
        try:
            time.sleep(0.6)
            logger.info("Landing...")
            self.cf.high_level_commander.land(0.0, 2.0)
            time.sleep(2.5)
            self.cf.close_link()
        finally:
            self.recorder.close()  # always write the Parquet footer, or the file is unreadable


def run_metadata(args) -> dict:
    """What flew, stored in the recording's Parquet footer."""
    meta = {"uri": args.uri, "argv": " ".join(sys.argv), "target": str(args.target),
            "duration_s": str(args.duration), "excitation": args.excitation, "exc_axis": args.exc_axis,
            "exc_amp": str(args.exc_amp), "exc_freq": str(args.exc_freq), "kp": str(KP)}
    for package in ("torch", "cflib"):  # cflib has no __version__
        try:
            meta[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            meta[f"{package}_version"] = "unknown"
    try:
        meta["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        meta["git_commit"] = "unknown"
    return meta


def main():
    parser = argparse.ArgumentParser(description="Hover from the host and add a known velocity excitation, "
                                                 "recorded for measure_delay_noise.py.")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E8", help="URI of the Crazyflie")
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        help="Hover target [x, y, z] in world frame. If not set, hovers above takeoff pos.")
    parser.add_argument("--excitation", choices=["step", "chirp"], default="step")
    parser.add_argument("--exc-axis", choices=["x", "y", "z"], default="x",
                        help="World axis to excite. Prefer x/y: z from a 0.5 m hover nears the 0.1 m floor cut-out.")
    parser.add_argument("--exc-amp", type=float, default=0.2, help=f"Excitation amplitude [m/s], at most {EXC_MAX_AMP}.")
    parser.add_argument("--exc-freq", type=float, default=0.5, help="Square-wave frequency [Hz] for step, at most 10.")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds of flight before landing.")
    add_run_args(parser)
    args = parser.parse_args()
    if not args.record_path:
        parser.error("--record-path is required: the flight exists to be recorded")
    if args.record_every != 1:
        parser.error("--record-every must be 1: the noise correlation is per control step")
    if args.duration <= EXC_START_S:
        parser.error(f"--duration must be above {EXC_START_S:g} s, when the excitation starts")
    if not 0 < args.exc_amp <= EXC_MAX_AMP or not 0 < args.exc_freq <= 10:
        parser.error(f"--exc-amp must be in (0, {EXC_MAX_AMP}] m/s and --exc-freq in (0, 10] Hz")

    flight = ExciteHover(uri=args.uri, run_args=args, kind=args.excitation, axis="xyz".index(args.exc_axis),
                         amp=args.exc_amp, freq=args.exc_freq, duration=args.duration, target=args.target,
                         metadata=run_metadata(args))
    try:
        flight.start()
        for _ in range(10):
            if flight.cf.is_connected() or not flight.running:
                break
            time.sleep(1)
        if not flight.cf.is_connected():
            logger.error("Connection failed — check radio URI and that cfclient is closed")
            return
        while flight.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        flight.stop()


if __name__ == "__main__":
    main()
