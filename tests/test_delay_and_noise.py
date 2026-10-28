"""measure_delay_noise.py on synthetic step-response flights whose radio delay, dead time, tau and noise are known.

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


def synthetic_step_flight(d_radio, rho, dead_steps=2, n=8000, amp=0.4, alternate=False, seed=0):
    """100 Hz excite_hover flight: from 1 s on, 20 cycles of REST / PUSH / RECOVER (1 s at 0, 1 s at
    `amp`, 2 s at 0; `amp` gives each push the 0.4 m/s edge of the old ±0.2 square wave). The
    command reaches the drone after `d_radio` steps (`echo`), then after `dead_steps` of plant tilt
    dead time the velocity responds with time constant TAU and AR(1) sensor noise (`NOISE`, `rho`).
    `alternate` flips the PUSH sign every cycle, as older recordings did. Returns (exc, echo, y)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    cycle, phase_t = np.divmod(np.round(t - 1.0, 6), 4.0)
    sign = np.where(alternate & (cycle % 2 == 1), -1.0, 1.0)
    exc = np.where((t >= 1.0) & (phase_t >= 1.0) & (phase_t < 2.0), sign * amp, 0.0)
    echo = np.zeros(n)
    if d_radio == 0:
        echo[:] = exc
    else:
        echo[d_radio:] = exc[:-d_radio]

    a = np.exp(-DT / TAU)
    y = np.zeros(n)
    v = e = 0.0
    total_d = d_radio + dead_steps
    for i in range(n):
        e = rho * e + np.sqrt(1 - rho ** 2) * rng.standard_normal()
        y[i] = v + NOISE * e
        u_plant = exc[i - total_d] if i >= total_d else 0.0
        v = a * v + (1 - a) * u_plant
    return exc, echo, y


@pytest.mark.parametrize("alternate", [False, True])
@pytest.mark.parametrize("d_radio", [1, 4])
@pytest.mark.parametrize("rho", [0.0, 0.8])
def test_step_response_recovers_radio_delay_dead_time_tau_and_noise(d_radio, rho, alternate):
    exc, echo, y = synthetic_step_flight(d_radio, rho, dead_steps=3, alternate=alternate)
    res = mdn.step_response(exc, echo, y, DT)
    assert res["d"] == d_radio
    assert res["radio_ms"] == pytest.approx(d_radio * DT * 1e3)
    # 3 steps (30 ms) of plant dead time + 1 step loop read + 1 step to cross 5% = 40..50 ms (integer steps)
    assert res["dead_ms"] == pytest.approx(40.0, abs=10.0)
    # 5% to 63.2% of a first-order step is TAU * (1 + ln(0.95)) ≈ 0.95 * TAU
    assert res["tau_ms"] == pytest.approx(TAU * 1e3, rel=0.15)
    assert res["noise_std"] == pytest.approx(NOISE, abs=0.003)
    assert res["noise_corr"] == pytest.approx(rho, abs=0.05)


def test_step_windows_keep_only_the_pushes_from_rest():
    exc, _, y = synthetic_step_flight(1, 0.0)
    step_edges, windows = mdn.step_windows(exc, y)
    assert windows.shape == (20, 100)  # one window per PUSH, as long as the PUSH
    assert all(exc[t - 1] == 0 and exc[t] == 0.4 for t in step_edges)  # never the edge back to 0
    mean = windows.mean(axis=0)
    assert abs(mean[:5].mean()) < 0.03 and mean[-10:].mean() == pytest.approx(1.0, abs=0.03)
    assert mdn.step_windows(np.zeros(100), np.zeros(100)) == ([], None)


def write_flight(path, d_radio=4, excited=True, steps=None, alternate=False):
    """A recording in the estimator's input contract: x excited, y and z hovering unexcited."""
    exc, echo, y = synthetic_step_flight(d_radio, 0.5, alternate=alternate)
    n = len(exc)
    rng = np.random.default_rng(2)
    cols = {
        "step": np.arange(n) if steps is None else steps,
        "control_time": np.arange(n) * DT,
        "vx": exc,
        "vel_x": y,
        "exc_vx": exc if excited else np.zeros(n),
        "echo_vx": echo if excited else np.zeros(n),
    }
    for ax in "yz":
        cols.update({f"v{ax}": 0.1 * rng.standard_normal(n), f"vel_{ax}": 0.03 * rng.standard_normal(n)})
    pq.write_table(pa.table(cols), path)
    return str(path)


def test_main_prints_vel_hovering_values(tmp_path, capsys):
    mdn.main([write_flight(tmp_path / "step.parquet", d_radio=4)])
    out = capsys.readouterr().out
    assert "env.action_latency_steps=4 env.action_latency_max_steps=0 env.add_noise=True" in out
    values = dict(kv.split("=") for kv in out.splitlines()[-1].split())
    assert float(values["env.noise_std"]) == pytest.approx(NOISE, abs=0.003)
    assert float(values["env.noise_corr"]) == pytest.approx(0.5, abs=0.05)


def test_main_reports_nothing_without_excitation(tmp_path, capsys):
    path = write_flight(tmp_path / "hover.parquet", excited=False)
    assert mdn.analyse(path) == {}
    mdn.main([path])
    out = capsys.readouterr().out
    assert "NOT REPORTED" in out and "action_latency_steps" not in out


def test_rejects_a_recording_with_skipped_steps(tmp_path):
    path = write_flight(tmp_path / "every2.parquet", steps=np.arange(0, 16000, 2))
    with pytest.raises(SystemExit, match="not consecutive"):
        mdn.main([path])

