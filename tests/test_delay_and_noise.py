"""System-identification guards for measure_delay_noise.

The central hazard: a loop delay cannot be identified from closed-loop policy
flights. On the real logs the fit returned ~150 ms, which was a quarter period of
the 1.5 Hz limit cycle rather than the transport delay. The tool must refuse to
report a delay unless the data came from open-loop excitation.
"""

import numpy as np
import pytest

from measure_delay_noise import ar1, dominant_freq, fit_fopdt, xcorr_lag


def test_ar1_recovers_a_known_correlation():
    rng = np.random.default_rng(0)
    rho, n = 0.9, 20000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal() * (1 - rho ** 2) ** 0.5
    assert ar1(x) == pytest.approx(rho, abs=0.03)


def test_ar1_of_white_noise_is_about_zero():
    rng = np.random.default_rng(1)
    assert ar1(rng.normal(size=20000)) == pytest.approx(0.0, abs=0.03)


def test_dominant_frequency_is_recovered():
    dt, f0 = 0.01, 3.0
    t = np.arange(0, 20, dt)
    assert dominant_freq(np.sin(2 * np.pi * f0 * t), dt) == pytest.approx(f0, rel=0.05)


def test_deadtime_is_recovered_from_open_loop_excitation():
    """A known pure delay plus first-order lag must be identified."""
    dt, delay, n = 0.01, 5, 4000
    rng = np.random.default_rng(2)
    u = rng.normal(size=n)                       # exogenous input
    y = np.zeros(n)
    a = np.exp(-dt / 0.15)
    for k in range(delay + 1, n):
        y[k] = a * y[k - 1] + (1 - a) * u[k - 1 - delay]
    d, tau, gain, r2 = fit_fopdt(u, y, dt, max_lag=20)
    assert d == pytest.approx(delay, abs=1)
    assert r2 > 0.9


def test_closed_loop_lag_tracks_the_oscillation_not_the_delay():
    """Guards the trap: with the command a function of the response, the estimated
    lag collapses onto the limit-cycle geometry. The quarter period must be
    reported so the number can be recognised as an artefact."""
    dt, f = 0.01, 1.5
    t = np.arange(0, 20, dt)
    u = np.sin(2 * np.pi * f * t)
    y = np.sin(2 * np.pi * f * t - np.pi / 2)    # 90 deg lag, no transport delay
    lag, corr = xcorr_lag(u, y, max_lag=int(0.4 / dt))
    quarter_period_steps = (1.0 / f) / 4 / dt
    assert lag == pytest.approx(quarter_period_steps, rel=0.15), (
        "closed-loop lag should land on the quarter period, which is why it must "
        "never be reported as a transport delay")
