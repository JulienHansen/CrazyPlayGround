import logging
import math
import threading
import time
from typing import Callable, Optional, Sequence, Tuple


def quat_to_euler_deg(quat: Sequence[float]) -> Tuple[float, float, float]:
    """Convert (qw, qx, qy, qz) to (roll, pitch, yaw) in degrees, for display only."""
    qw, qx, qy, qz = quat
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class FlightLogger:
    """Terminal status logger, throttled and decoupled from the 100Hz control loop."""

    def __init__(
        self,
        logger: logging.Logger,
        sample_fn: Callable[[], Optional[Tuple[str, str]]],
        log_interval_s: float = 1.0,
        level: int = logging.INFO,
    ):
        """
        sample_fn: zero-arg callable, invoked once per tick on the logger's
            own cadence (not per control-loop iteration). Takes the
            controller's lock internally and returns either None (nothing
            ready yet) or a (state_line, cmd_line) tuple of already-formatted
            strings to log, in that order.
        log_interval_s: seconds between printed lines, independent of the
            100Hz control loop and of FlightRecorder's own cadence.
        """
        self.logger = logger
        self.sample_fn = sample_fn
        self.log_interval_s = log_interval_s
        self.level = level
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            lines = self.sample_fn()
            if lines is not None:
                state_line, cmd_line = lines
                self.logger.log(self.level, state_line)
                self.logger.log(self.level, cmd_line)
            time.sleep(self.log_interval_s)

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.log_interval_s * 5 + 1.0)
