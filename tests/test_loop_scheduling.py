"""The control loop must hold 100 Hz even when time.sleep returns late.

Hardware flights ran at 84-85 Hz against a commanded 100 Hz, with the period
centred at 11.99 ms. The old loop scheduled relatively:

    elapsed = now - start_of_iteration
    sleep(INTERVAL - elapsed)

so the period is INTERVAL plus however late sleep returns, every iteration. On
macOS that overshoot is 1-2 ms, which is the whole 100 -> 85 Hz loss. Scheduling
against an absolute deadline absorbs it instead; the spin margin then removes the
residual phase error. These tests pin both.
"""

import random
import time

import pytest

from collect_hover_vel import wait_until

INTERVAL = 0.010
SPIN = 0.002


@pytest.fixture
def late_sleep(monkeypatch):
    """time.sleep that returns ~2 ms late, like macOS timer coalescing."""
    real = time.sleep

    def sleeper(sec):
        real(sec + 0.002)

    monkeypatch.setattr(time, "sleep", sleeper)
    return sleeper


@pytest.fixture
def jittery_sleep(monkeypatch):
    """time.sleep whose lateness varies from step to step, within the spin margin."""
    real = time.sleep
    rng = random.Random(0)

    def sleeper(sec):
        real(sec + rng.uniform(0.0005, 0.0018))

    monkeypatch.setattr(time, "sleep", sleeper)
    return sleeper


def _run(n, spin_margin, body_s=0.0):
    deadline = time.perf_counter()
    stamps = []
    for _ in range(n):
        if body_s:
            end = time.perf_counter() + body_s
            while time.perf_counter() < end:
                pass
        deadline, _, _ = wait_until(deadline + INTERVAL, spin_margin)
        stamps.append(time.perf_counter())
    return [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]


def _run_relative(n):
    """The scheduler the hardware flights used: sleep the remainder of the period."""
    stamps = []
    for _ in range(n):
        start = time.perf_counter()
        time.sleep(max(0.0, INTERVAL - (time.perf_counter() - start)))
        stamps.append(time.perf_counter())
    return [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]


def test_relative_scheduling_loses_the_rate_when_sleep_overshoots(late_sleep):
    """Reproduces the 84 Hz measured on hardware: the overshoot is paid every step."""
    periods = _run_relative(40)
    mean = sum(periods) / len(periods)
    assert mean > 11.5, f"expected the overshoot to show, got {mean:.2f} ms"
    assert 1000.0 / mean < 90.0


def test_absolute_deadline_absorbs_the_overshoot(late_sleep):
    """The same overshoot becomes a constant phase offset, not a per-step cost."""
    periods = _run(40, spin_margin=0.0)
    mean = sum(periods) / len(periods)
    assert 9.8 < mean < 10.4, f"expected ~10 ms, got {mean:.2f} ms"


def _iqr(xs):
    xs = sorted(xs)
    return xs[int(len(xs) * 0.9)] - xs[int(len(xs) * 0.1)]


def test_spin_margin_cuts_the_jitter_when_the_overshoot_varies(jittery_sleep):
    """An absolute deadline fixes the rate; the spin is what fixes the jitter."""
    without = _iqr(_run(60, spin_margin=0.0))
    with_spin = _iqr(_run(60, spin_margin=SPIN))
    assert with_spin < without / 2, (
        f"spin should cut the period spread: {with_spin:.2f} vs {without:.2f} ms"
    )
    assert with_spin < 0.5, f"expected a tight period, spread {with_spin:.2f} ms"
    # the spin only helps while the margin covers the overshoot; beyond it the
    # deadline is missed and the period wanders again.


def test_spin_margin_does_not_inflate_the_period_without_overshoot():
    """The spin must not cost anything when sleep is already accurate."""
    periods = _run(40, spin_margin=SPIN)
    mean = sum(periods) / len(periods)
    assert 9.8 < mean < 10.4, f"expected ~10 ms, got {mean:.2f} ms"


def test_an_overrun_resynchronises_instead_of_bursting():
    """After a long body, the loop must not fire back-to-back setpoints to catch up."""
    slow = INTERVAL * 3
    deadline = time.perf_counter()
    held, slept, overran = wait_until(deadline - slow, SPIN)   # deadline already past
    assert overran == 1
    assert slept == 0.0
    assert held == pytest.approx(time.perf_counter(), abs=2e-3)
    # the next deadline is one interval from now, not from the stale schedule
    assert held > deadline - slow


def test_body_time_is_absorbed_not_added():
    """A 3 ms body must still give a 10 ms period, not 13 ms."""
    periods = _run(40, spin_margin=SPIN, body_s=0.003)
    mean = sum(periods) / len(periods)
    assert 9.8 < mean < 10.6, f"expected ~10 ms, got {mean:.2f} ms"
