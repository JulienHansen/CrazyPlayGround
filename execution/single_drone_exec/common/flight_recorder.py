import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_MOTOR_FIELDS = ("motor_m1_pwm", "motor_m2_pwm", "motor_m3_pwm", "motor_m4_pwm")


class FlightRecorder:
    """Generic Parquet flight-data recorder shared across the hover execution scripts.

    Runs its own thread at a fixed cadence, snapshotting whatever fields the
    caller provides via `sample_fn` and buffering one row per tick. Rows are
    flushed to the Parquet file in row-group batches (rather than one write
    per row, which Parquet is not designed for). Each script declares its own
    obs/action/cmd column names at construction time, so columns stay
    human-readable (e.g. "vx, vy, vz") instead of a generic "cmd_0, cmd_1,
    cmd_2" regardless of control mode. Motor PWM columns default to the same
    4-motor names across scripts since that shape never varies.
    """

    BASE_FIELDS = [
        "write_time", "control_time", "obs_age_s",
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "quat_w", "quat_x", "quat_y", "quat_z",
        "target_x", "target_y", "target_z",
    ]

    def __init__(
        self,
        record_path: str,
        sample_fn: Callable[[], Optional[Dict[str, object]]],
        obs_fields: Sequence[str],
        action_fields: Sequence[str],
        cmd_fields: Sequence[str],
        motor_fields: Sequence[str] = DEFAULT_MOTOR_FIELDS,
        record_interval_s: float = 0.1,
        batch_size: int = 50,
    ):
        """
        sample_fn: zero-arg callable, invoked once per tick, responsible for
            taking the controller's lock internally and returning either
            None (nothing to record yet) or a dict with keys:
                "control_time": float
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
        obs_fields/action_fields/cmd_fields/motor_fields: column names for
            each vector, task-specific and chosen by the caller (motor_fields
            defaults to the standard 4-motor naming, shared by all scripts).
        batch_size: number of buffered rows per Parquet row-group flush.
        """
        self.record_path = record_path
        self.sample_fn = sample_fn
        self.obs_fields = list(obs_fields)
        self.action_fields = list(action_fields)
        self.cmd_fields = list(cmd_fields)
        self.motor_fields = list(motor_fields)
        self.record_interval_s = record_interval_s
        self.batch_size = batch_size

        self._fieldnames = (
            self.BASE_FIELDS + self.obs_fields + self.action_fields
            + self.cmd_fields + self.motor_fields
        )
        motor_field_set = set(self.motor_fields)
        self._schema = pa.schema([
            (name, pa.int64() if name in motor_field_set else pa.float64())
            for name in self._fieldnames
        ])

        self._writer: Optional[pq.ParquetWriter] = None
        self._buffer: List[Dict[str, object]] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Open the Parquet writer and launch the recorder thread."""
        self._running = True
        self._writer = pq.ParquetWriter(self.record_path, self._schema)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            sample = self.sample_fn()
            if sample is not None:
                self._buffer.append(self._build_row(time.time(), sample))
                if len(self._buffer) >= self.batch_size:
                    self._flush_buffer()
            time.sleep(self.record_interval_s)
        self._flush_buffer()

    def _build_row(self, write_time: float, sample: Dict[str, object]) -> Dict[str, object]:
        control_time = sample["control_time"]
        pos, vel, quat = sample["pos"], sample["vel"], sample["quat"]
        target = sample.get("target") or [None, None, None]
        obs, action, cmd, motor = sample["obs"], sample["action"], sample["cmd"], sample["motor"]

        row = {
            "write_time": write_time,
            "control_time": control_time,
            "obs_age_s": write_time - control_time,
            "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
            "vel_x": vel[0], "vel_y": vel[1], "vel_z": vel[2],
            "quat_w": quat[0], "quat_x": quat[1], "quat_y": quat[2], "quat_z": quat[3],
            "target_x": target[0], "target_y": target[1], "target_z": target[2],
        }
        row.update(dict(zip(self.obs_fields, obs)))
        row.update(dict(zip(self.action_fields, action)))
        row.update(dict(zip(self.cmd_fields, cmd)))
        row.update(dict(zip(self.motor_fields, motor)))
        return row

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        columns = {name: [row[name] for row in self._buffer] for name in self._fieldnames}
        table = pa.Table.from_pydict(columns, schema=self._schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.record_interval_s * 5 + 1.0)
        self._flush_buffer()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
