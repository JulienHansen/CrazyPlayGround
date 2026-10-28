"""Step-excitation flight for measure_delay_noise.py: a hover held from the host, pushed by a velocity step.

A position P loop on the host holds the hover and sends world velocities at 100 Hz through
send_velocity_world_setpoint, the radio and firmware path the velocity policy flies through.
From EXC_START_S on, the excited axis repeats a cycle of three phases (see step_signal):

    REST     step_s     exc 0      P hold off: the drone settles at rest under a zero command
    PUSH     step_s     exc amp    P hold off: the step, from rest, always in the sign of amp
    RECOVER  recover_s  exc 0      P hold on: the drone brakes and flies back to its hover target

so every step starts from rest and goes the same way: the response is not averaged over two
directions that may not be symmetric. The P hold stays active on the two unexcited axes.
Every row records the sent command (vx/vy/vz), the step excitation (exc_v*), the setpoint echoed
back by the firmware (echo_v* = ctrltarget.v*), and the measured velocity (vel_*).

    python execution/sim2real/excite_hover.py --record-path flight.parquet --exc-axis x --duration 90
    python execution/sim2real/measure_delay_noise.py flight.parquet
    python execution/sim2real/plot_step_response.py flight.parquet --out flight.png
"""

import os
import sys
import time
import importlib.metadata
import threading
import argparse
import logging

import torch

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

# Make the `common` package importable; appended so it cannot shadow installed modules.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "single_drone_exec"))
from common.flight_recorder import CrazyflieStateBase  # noqa: E402
from common.utils import emergency_land, land_once, wait_until, add_run_args  # noqa: E402
from excitation import (INTERVAL, EXC_START_S, step_signal, excitation_command,  # noqa: E402
                        add_excitation_args, check_excitation_args, excitation_metadata)

device = torch.device("cpu")  # no model: nothing to gain from a GPU
logging.basicConfig(format="{asctime} [{levelname}] {message}", style="{",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("CrazyflieRL")

# ── Safety thresholds (as hover/exec_vel.py) ─────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without a position callback before emergency land
POS_VARIANCE_THRESHOLD = 0.5   # kalman position variance [m²] above which tracking is unreliable
GRACE_PERIOD           = 3.0   # seconds before enforcing the z lower bound, at most EXC_START_S

SPIN_MARGIN = 0.002  # sleep to 2 ms before each deadline, busy-wait the rest (see wait_until)

# ── Step Excitation (signal, command law and flags in excitation.py) ─────────
CTRLTARGET_VARS = ["ctrltarget.vx", "ctrltarget.vy", "ctrltarget.vz"]


class ExciteHover(CrazyflieStateBase):
    def __init__(self, uri: str, run_args, axis: int, amp: float, step_s: float, recover_s: float,
                 duration: float, target=None, metadata=None):
        super().__init__(device)
        self.uri = uri
        self.cf = Crazyflie(rw_cache='./cache')
        self.axis, self.amp = axis, amp
        self.step_s, self.recover_s = step_s, recover_s
        self.duration = duration
        self.initial_target = target
        self.echo_vel = [float("nan")] * 3
        self.make_recorder(
            run_args,
            obs_fields=["err_x", "err_y", "err_z"],
            action_fields=["hold_vx", "hold_vy", "hold_vz"],
            cmd_fields=["vx", "vy", "vz"],
            extra_fields=["exc_vx", "exc_vy", "exc_vz", "echo_vx", "echo_vy", "echo_vz",
                          "dt_body_ms", "slept_ms", "overran"],
        )
        self.recorder.metadata.update(metadata or {})
        self.cf.connected.add_callback(self._connected)
        self.cf.connection_failed.add_callback(self._connection_lost)
        self.cf.connection_lost.add_callback(self._connection_lost)

    def _connected(self, uri: str):
        logger.info(f"Connected to {uri}, taking off...")
        self.cf.high_level_commander.takeoff(0.5, 1)
        time.sleep(1.5)
        self.setup_state_logging()
        self._setup_target_logging()
        threading.Thread(target=self.control_loop, daemon=True).start()

    def _setup_target_logging(self):
        """Log ctrltarget.vx/vy/vz at 100 Hz so measure_delay_noise.py sees when each step reached the drone."""
        logconf = LogConfig(name="ctrltarget", period_in_ms=10)
        for var in CTRLTARGET_VARS:
            logconf.add_variable(var, "float")
        try:
            self.cf.log.add_config(logconf)
        except (KeyError, AttributeError) as e:
            logger.warning(f"Log block ctrltarget unavailable ({e}); echo_v* columns stay NaN")
            return

        def _callback(timestamp, data, logconf):
            with self.lock:
                self.echo_vel = [data[var] for var in CTRLTARGET_VARS]

        logconf.data_received_cb.add_callback(_callback)
        logconf.start()

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
        logger.info(f"Holding {target.tolist()}, step excitation on axis {'xyz'[self.axis]} from {EXC_START_S:g} s")

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
                echo = list(self.echo_vel)
            if var.max().item() > POS_VARIANCE_THRESHOLD:
                logger.error(f"Position variance too high {var.tolist()} — emergency landing")
                emergency_land(self)
                break

            state = self.snapshot()  # one consistent read, used for the command and its recorded row
            pos_error = target - state.pos
            step_vel, hold_on_exc_axis = step_signal(self.axis, self.amp, self.step_s, self.recover_s, elapsed)
            hold_vel, setpoint_vel = excitation_command(pos_error, self.axis, step_vel, hold_on_exc_axis)
            control_time = time.time()
            self.cf.commander.send_velocity_world_setpoint(*setpoint_vel.tolist(), 0.0)  # yaw_rate = 0
            t_sent = time.perf_counter()

            deadline, slept_ms, overran = wait_until(deadline + INTERVAL, SPIN_MARGIN)
            self.record_step(step, state, control_time, target, pos_error, hold_vel, setpoint_vel,
                             step_vel + echo + [(t_sent - t_body) * 1e3, slept_ms, overran])
            step += 1

        land_once(self)  # at once, never send_stop_setpoint: it cuts the motors mid-air
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
            land_once(self)  # no-op once the control loop has landed
            time.sleep(2.5)
            self.cf.close_link()
        finally:
            self.recorder.close()  # always write the Parquet footer, or the file is unreadable


def run_metadata(args) -> dict:
    """What flew, stored in the recording's Parquet footer."""
    meta = {"uri": args.uri, **excitation_metadata(args)}
    for package in ("torch", "cflib"):  # cflib has no __version__
        try:
            meta[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            meta[f"{package}_version"] = "unknown"
    return meta


def main():
    parser = argparse.ArgumentParser(description="Hover from the host and push it with one-direction velocity steps, "
                                                 "recorded for measure_delay_noise.py.")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E8", help="URI of the Crazyflie")
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        help="Hover target [x, y, z] in world frame. If not set, hovers above takeoff pos.")
    add_excitation_args(parser)
    add_run_args(parser)
    args = parser.parse_args()
    if not args.record_path:
        parser.error("--record-path is required: the flight exists to be recorded")
    if args.record_every != 1:
        parser.error("--record-every must be 1: the noise correlation is per control step")
    check_excitation_args(parser, args)

    flight = ExciteHover(uri=args.uri, run_args=args, axis="xyz".index(args.exc_axis),
                         amp=args.exc_amp, step_s=args.exc_step, recover_s=args.exc_recover,
                         duration=args.duration, target=args.target, metadata=run_metadata(args))
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

