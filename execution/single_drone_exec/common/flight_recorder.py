import logging
import queue
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from cflib.crazyflie.log import LogConfig

from .utils import quat_to_euler_deg

DEFAULT_MOTOR_FIELDS = ("motor_m1_pwm", "motor_m2_pwm", "motor_m3_pwm", "motor_m4_pwm")

_STATE_FIELDS = [
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "target_x", "target_y", "target_z",
]


class FlightRecorder:
    """Parquet flight-data recorder and status logger shared across the hover execution scripts.

    The control loop calls `record()` once per control step and this
    recorder's thread only drains the queue, writes, and prints a status line
    every `log_every` steps, so the file holds exactly one row per recorded
    step, built from the same state the policy saw, and disk and terminal I/O
    stay off the control thread. Columns:
        step        control-step index; a gap means steps without a policy row
        control_time  time.time() at the step
        obs_age_s   control_time minus the host receive time of the posvel
                    packet behind the observation (excludes radio and on-drone
                    estimation delay)

    Rows are flushed to the Parquet file in row-group batches (rather than one
    write per row, which Parquet is not designed for). Each script declares its
    own obs/action/cmd column names, so columns stay human-readable (e.g.
    "vx, vy, vz") regardless of control mode. Motor PWM columns default to the
    same 4-motor names across scripts since that shape never varies.
    """

    BASE_FIELDS = ["step", "control_time", "obs_age_s"] + _STATE_FIELDS

    def __init__(
        self,
        record_path: Optional[str] = None,
        obs_fields: Sequence[str] = (),
        action_fields: Sequence[str] = (),
        cmd_fields: Sequence[str] = (),
        motor_fields: Sequence[str] = DEFAULT_MOTOR_FIELDS,
        record_every: int = 1,
        log_every: int = 100,
        batch_size: int = 50,
        logger: Optional[logging.Logger] = None,
    ):
        """
        A sample, passed to `record()`, is a dict:
            "step": int
            "control_time": float
            "obs_age_s": float
            "pos": Sequence[float] (len 3)
            "vel": Sequence[float] (len 3)
            "quat": Sequence[float] (len 4)
            "target": Sequence[float] (len 3) or None
            "obs": Sequence[float]    (len == len(obs_fields))
            "action": Sequence[float] (len == len(action_fields))
            "cmd": Sequence[float]    (len == len(cmd_fields); the value
                actually sent to the drone, whatever control mode)
            "motor": Sequence[int]    (len == len(motor_fields); raw PWM
                ticks sent to each motor, as reported by the firmware)
        record_path: Parquet file to write, or None to only log.
        record_every: keep one step in N in the file (1 = every step).
        log_every: print a status line every N steps (100 = 1 s at 100 Hz).
        batch_size: number of buffered rows per Parquet row-group flush.
        """
        if record_every < 1 or log_every < 1:
            raise ValueError("record_every and log_every must be >= 1")
        self.record_path = record_path
        self.obs_fields = list(obs_fields)
        self.action_fields = list(action_fields)
        self.cmd_fields = list(cmd_fields)
        self.motor_fields = list(motor_fields)
        self.record_every = record_every
        self.log_every = log_every
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger("CrazyflieRL")

        self._fieldnames = (
            self.BASE_FIELDS + self.obs_fields + self.action_fields
            + self.cmd_fields + self.motor_fields
        )
        int_fields = set(self.motor_fields) | {"step"}
        self._schema = pa.schema([
            (name, pa.int64() if name in int_fields else pa.float64())
            for name in self._fieldnames
        ])

        self._queue: "queue.SimpleQueue[Optional[Dict[str, object]]]" = queue.SimpleQueue()
        self._writer: Optional[pq.ParquetWriter] = None
        self._buffer: List[Dict[str, object]] = []
        self._thread: Optional[threading.Thread] = None
        self.skipped = 0  # rows that raised while being written or logged

    def start(self) -> None:
        """Open the Parquet writer (if recording) and launch the recorder thread."""
        if self.record_path:
            self._writer = pq.ParquetWriter(self.record_path, self._schema)
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()

    def record(self, sample: Dict[str, object]) -> None:
        """Queue one control step's row. Called from the control thread; never blocks."""
        step = sample["step"]
        if step % self.log_every == 0 or (self.record_path and step % self.record_every == 0):
            self._queue.put(sample)

    def _drain_loop(self) -> None:
        # None is the stop sentinel put by close(): rows queued before it are all written.
        while (sample := self._queue.get()) is not None:
            try:
                if self._writer and sample["step"] % self.record_every == 0:
                    self._buffer.append(self._build_row(sample))
                    if len(self._buffer) >= self.batch_size:
                        self._flush_buffer()
                if sample["step"] % self.log_every == 0:
                    self._log(sample)
            except Exception:  # one bad row must not stop recording and status lines for the flight
                self.skipped += 1
                if self.skipped == 1:
                    self.logger.exception(f"Recorder failed at step {sample.get('step')}; skipping bad rows")
        self._flush_buffer()

    def _log(self, sample: Dict[str, object]) -> None:
        (px, py, pz), (vx, vy, vz) = sample["pos"], sample["vel"]
        roll, pitch, yaw = quat_to_euler_deg(sample["quat"])
        cmd = " ".join(f"{name}={value:+.2f}" for name, value in zip(self.cmd_fields, sample["cmd"]))
        pwm = " ".join(f"m{i + 1}={value}" for i, value in enumerate(sample["motor"]))
        self.logger.info(
            f"[step {sample['step']}] pos=({px:+.2f}, {py:+.2f}, {pz:+.2f}) m "
            f"vel=({vx:+.2f}, {vy:+.2f}, {vz:+.2f}) m/s rpy=({roll:+.1f}, {pitch:+.1f}, {yaw:+.1f})°"
        )
        self.logger.info(f"Cmd: {cmd} | PWM: {pwm}")

    def _build_row(self, sample: Dict[str, object]) -> Dict[str, object]:
        pos, vel, quat = sample["pos"], sample["vel"], sample["quat"]
        target = sample.get("target") or [None, None, None]
        row = {
            "step": sample["step"],
            "control_time": sample["control_time"],
            "obs_age_s": sample["obs_age_s"],
            "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
            "vel_x": vel[0], "vel_y": vel[1], "vel_z": vel[2],
            "quat_w": quat[0], "quat_x": quat[1], "quat_y": quat[2], "quat_z": quat[3],
            "target_x": target[0], "target_y": target[1], "target_z": target[2],
        }
        row.update(zip(self.obs_fields, sample["obs"]))
        row.update(zip(self.action_fields, sample["action"]))
        row.update(zip(self.cmd_fields, sample["cmd"]))
        row.update(zip(self.motor_fields, sample["motor"]))
        return row

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        columns = {name: [row[name] for row in self._buffer] for name in self._fieldnames}
        table = pa.Table.from_pydict(columns, schema=self._schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        if self._thread is not None:
            self._queue.put(None)         # stop sentinel
            self._thread.join()           # drain loop flushes before exiting
            self._thread = None
            if self.skipped:
                self.logger.error(f"Recorder skipped {self.skipped} row(s), see the first error above")
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class StateSnapshot(NamedTuple):
    pos: torch.Tensor
    vel: torch.Tensor
    quat: torch.Tensor
    pos_time: float  # host time the posvel packet arrived
    motor: list      # motor PWM ticks


class CrazyflieStateBase:
    """Telemetry state shared by every hover controller.

    Declares the fields the log callbacks write, next to the callbacks, so they
    exist under one name only: a controller cannot declare its own copy under a
    different name that the callbacks never update. Subclasses set `self.cf`.
    """

    def __init__(self, device: torch.device):
        self.lock = threading.Lock()
        self.running = True
        self.position_received = False
        self._last_pos_time: float = 0.0
        self.current_pos = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_vel = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # identity until the first packet
        self._pos_variance = torch.zeros(3, dtype=torch.float32, device=device)
        self.current_motor_pwm = [0, 0, 0, 0]

    def make_recorder(self, args, obs_fields: Sequence[str], action_fields: Sequence[str],
                      cmd_fields: Sequence[str]) -> None:
        """Create `self.recorder` from the flags registered by utils.add_run_args."""
        self.recorder = FlightRecorder(args.record_path, obs_fields=obs_fields,
                                       action_fields=action_fields, cmd_fields=cmd_fields,
                                       record_every=args.record_every, log_every=args.log_every)

    def record_step(self, step: int, state: StateSnapshot, control_time: float,
                    target, obs, action, cmd) -> None:
        """Push this control step's row: the state behind `obs`, and what was sent."""
        self.recorder.record({
            "step": step, "control_time": control_time,
            "obs_age_s": control_time - state.pos_time,  # age of the state behind this obs
            "pos": state.pos.tolist(), "vel": state.vel.tolist(), "quat": state.quat.tolist(),
            "target": target.tolist(), "obs": obs.tolist(), "action": action.tolist(),
            "cmd": cmd.tolist() if torch.is_tensor(cmd) else list(cmd), "motor": state.motor,
        })

    def snapshot(self) -> StateSnapshot:
        """One consistent read of the state, for building an observation and its recorded row."""
        with self.lock:
            return StateSnapshot(self.current_pos.clone(), self.current_vel.clone(),
                                 self.current_quat.clone(), self._last_pos_time,
                                 list(self.current_motor_pwm))

    def setup_state_logging(self, log_frequency_ms: int = 10) -> None:
        """Register the telemetry LogConfig blocks (position/velocity, quaternion,
        Kalman variance, motor PWM) whose callbacks update the fields above."""
        device = self.current_pos.device

        def _log_posvel_callback(timestamp: float, data: Dict[str, Any], logconf: LogConfig):
            with self.lock:
                self.current_pos = torch.tensor([
                    data["stateEstimate.x"],
                    data["stateEstimate.y"],
                    data["stateEstimate.z"]
                ], dtype=torch.float32, device=device)
                self.current_vel = torch.tensor([
                    data["stateEstimate.vx"],
                    data["stateEstimate.vy"],
                    data["stateEstimate.vz"]
                ], dtype=torch.float32, device=device)
                self._last_pos_time = time.time()
                self.position_received = True

        def _log_data_quat_callback(timestamp: float, data: Dict[str, Any], logconf: LogConfig):
            with self.lock:
                self.current_quat = torch.tensor([
                    data['stateEstimate.qw'],
                    data['stateEstimate.qx'],
                    data['stateEstimate.qy'],
                    data['stateEstimate.qz'],
                ], dtype=torch.float32, device=device)

        def _log_variance_callback(timestamp: float, data: Dict[str, Any], logconf: LogConfig):
            with self.lock:
                self._pos_variance = torch.tensor([
                    data["kalman.varPX"],
                    data["kalman.varPY"],
                    data["kalman.varPZ"],
                ], dtype=torch.float32, device=device)

        def _log_motor_callback(timestamp: float, data: Dict[str, Any], logconf: LogConfig):
            with self.lock:
                self.current_motor_pwm = [
                    data["motor.m1"], data["motor.m2"], data["motor.m3"], data["motor.m4"],
                ]

        # Block 1: position + velocity (6 floats = 24 bytes, within 26-byte limit)
        log_posvel = LogConfig(name="posvel", period_in_ms=log_frequency_ms)
        log_posvel.add_variable("stateEstimate.x", "float")
        log_posvel.add_variable("stateEstimate.y", "float")
        log_posvel.add_variable("stateEstimate.z", "float")
        log_posvel.add_variable("stateEstimate.vx", "float")
        log_posvel.add_variable("stateEstimate.vy", "float")
        log_posvel.add_variable("stateEstimate.vz", "float")
        self.cf.log.add_config(log_posvel)
        log_posvel.data_received_cb.add_callback(_log_posvel_callback)
        log_posvel.start()

        # Block 2: quaternion (4 floats = 16 bytes)
        log_quat = LogConfig(name="quat", period_in_ms=log_frequency_ms)
        log_quat.add_variable("stateEstimate.qx", "float")
        log_quat.add_variable("stateEstimate.qy", "float")
        log_quat.add_variable("stateEstimate.qz", "float")
        log_quat.add_variable("stateEstimate.qw", "float")
        self.cf.log.add_config(log_quat)
        log_quat.data_received_cb.add_callback(_log_data_quat_callback)
        log_quat.start()

        # Block 3: Kalman variance (for safety watchdog)
        log_var = LogConfig(name="quality", period_in_ms=200)
        log_var.add_variable("kalman.varPX", "float")
        log_var.add_variable("kalman.varPY", "float")
        log_var.add_variable("kalman.varPZ", "float")
        self.cf.log.add_config(log_var)
        log_var.data_received_cb.add_callback(_log_variance_callback)
        log_var.start()

        # Block 4: motor PWM (4 uint16 = 8 bytes, within 26-byte limit)
        log_motor = LogConfig(name="motor", period_in_ms=log_frequency_ms)
        log_motor.add_variable("motor.m1", "uint16_t")
        log_motor.add_variable("motor.m2", "uint16_t")
        log_motor.add_variable("motor.m3", "uint16_t")
        log_motor.add_variable("motor.m4", "uint16_t")
        self.cf.log.add_config(log_motor)
        log_motor.data_received_cb.add_callback(_log_motor_callback)
        log_motor.start()
