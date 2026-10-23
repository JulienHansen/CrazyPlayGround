"""The control loop must hold 100 Hz even when time.sleep returns late.

Hardware flights ran at 84-85 Hz against a commanded 100 Hz: the old loop slept
`INTERVAL - elapsed`, so it paid sleep's overshoot (1-2 ms on macOS) every step.
`wait_until` schedules against an absolute deadline instead. Reduced from the
sim2real pipeline's suite (#18) to the behaviour the hover scripts rely on.
"""

import os
import sys
import time

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "execution", "single_drone_exec"))
from common.utils import wait_until  # noqa: E402

INTERVAL = 0.010
SPIN = 0.002


@pytest.fixture
def late_sleep(monkeypatch):
    """time.sleep that returns ~2 ms late, like macOS timer coalescing."""
    real = time.sleep
    monkeypatch.setattr(time, "sleep", lambda sec: real(sec + 0.002))


def _mean_period_ms(n, spin_margin, body_s=0.0):
    deadline = time.perf_counter()
    stamps = []
    for _ in range(n):
        end = time.perf_counter() + body_s
        while time.perf_counter() < end:
            pass
        deadline, _, _ = wait_until(deadline + INTERVAL, spin_margin)
        stamps.append(time.perf_counter())
    return (stamps[-1] - stamps[0]) / (n - 1) * 1e3


def test_absolute_deadline_absorbs_the_overshoot(late_sleep):
    """The overshoot becomes a constant phase offset, not a per-step cost."""
    assert 9.8 < _mean_period_ms(40, spin_margin=0.0) < 10.4


def test_body_time_is_absorbed_not_added():
    """A 3 ms body must still give a 10 ms period, not 13 ms."""
    assert 9.8 < _mean_period_ms(40, spin_margin=SPIN, body_s=0.003) < 10.6


def test_an_overrun_resynchronises_instead_of_bursting():
    """After a missed deadline the loop must not fire back-to-back setpoints to catch up."""
    stale = time.perf_counter() - 3 * INTERVAL
    held, slept, overran = wait_until(stale, SPIN)
    assert (overran, slept) == (1, 0.0)
    assert held == pytest.approx(time.perf_counter(), abs=2e-3)
