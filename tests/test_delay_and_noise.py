"""measure_delay_noise.py on synthetic hover flights whose delay and sensor noise are known.

No cflib: the estimator only reads recordings, so these run wherever numpy and pyarrow do.
"""

import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "execution", "sim2real"))
import measure_delay_noise as mdn  # noqa: E402

DT, TAU, NOISE = 0.01, 0.15, 0.03


def closed_loop(d, rho, n=4000, amp=0.2, freq=0.5, seed=0):
    """100 Hz hover: first-order velocity plant with d steps of delay, a PD hold on AR(1)-noisy
    measurements, and a square-wave excitation added to its command. Returns (exc, u, y)."""
    rng = np.random.default_rng(seed)
    exc = amp * np.sign(np.sin(2 * np.pi * freq * np.arange(n) * DT))
    a = np.exp(-DT / TAU)
    u, y = np.zeros(n), np.zeros(n)
    v = p = e = 0.0
    for i in range(n):
        e = rho * e + np.sqrt(1 - rho ** 2) * rng.standard_normal()
        y[i] = v + NOISE * e
        u[i] = np.clip(-2 * p - 0.6 * y[i] + exc[i], -1, 1)
        p += v * DT
        v = a * v + (1 - a) * (u[i - d] if i >= d else 0.0)   # command i reaches the velocity at i + 1 + d
    return exc, u, y


@pytest.mark.parametrize("d", [2, 6])
@pytest.mark.parametrize("rho", [0.0, 0.8])   # 0.8: one unwhitened pass took d=2 for 3
def test_fopdt_recovers_delay_and_noise_in_closed_loop(d, rho):
    _, u, y = closed_loop(d, rho)
    fit_d, tau, k, resid = mdn.fit_fopdt(u, y, DT)
    std, corr = mdn.ar1(resid)
    assert fit_d == d
    assert tau == pytest.approx(TAU, rel=0.15) and k == pytest.approx(1.0, abs=0.1)
    assert std == pytest.approx(NOISE, abs=0.003) and corr == pytest.approx(rho, abs=0.05)


def test_xcorr_lag_of_a_shifted_step_train():
    rng = np.random.default_rng(1)
    u = np.repeat(rng.choice([-1.0, 1.0], 100), 20)
    assert mdn.xcorr_lag(u, np.roll(u, 7) + 0.1 * rng.standard_normal(len(u)), 20)[0] == 7


def test_dominant_freq_ignores_drift():
    t = np.arange(3000) * DT
    assert mdn.dominant_freq(0.1 * np.sin(2 * np.pi * 3 * t) + 0.05 * t, DT) == pytest.approx(3.0, abs=0.05)


def write_flight(path, d=4, excited=True, steps=None):
    """A recording in the estimator's input contract: x excited, y and z hovering unexcited."""
    exc, u, y = closed_loop(d, 0.5)
    n = len(u)
    rng = np.random.default_rng(2)
    cols = {"step": np.arange(n) if steps is None else steps, "control_time": np.arange(n) * DT,
            "vx": u, "vel_x": y, "exc_vx": exc if excited else np.zeros(n)}
    for ax in "yz":
        cols.update({f"v{ax}": 0.1 * rng.standard_normal(n), f"vel_{ax}": 0.03 * rng.standard_normal(n)})
    pq.write_table(pa.table(cols), path)   # no exc_vy/exc_vz: absent columns read as zeros
    return str(path)


def test_main_prints_vel_hovering_values(tmp_path, capsys):
    mdn.main([write_flight(tmp_path / "step.parquet")])
    out = capsys.readouterr().out
    assert "env.action_latency_steps=4 env.action_latency_max_steps=0 env.add_noise=True" in out
    values = dict(kv.split("=") for kv in out.splitlines()[-2].split())
    assert float(values["env.noise_std"]) == pytest.approx(NOISE, abs=0.003)
    assert float(values["env.noise_corr"]) == pytest.approx(0.5, abs=0.05)


def test_main_reports_nothing_without_excitation(tmp_path, capsys):
    path = write_flight(tmp_path / "hover.parquet", excited=False)
    assert all("d" not in r and "osc_hz" in r for r in mdn.analyse(path).values())
    mdn.main([path])
    out = capsys.readouterr().out
    assert "NOT REPORTED" in out and "action_latency_steps" not in out


def test_rejects_a_recording_with_skipped_steps(tmp_path):
    path = write_flight(tmp_path / "every2.parquet", steps=np.arange(0, 8000, 2))
    with pytest.raises(SystemExit, match="not consecutive"):
        mdn.main([path])
