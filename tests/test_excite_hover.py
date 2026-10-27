"""excite_hover.py against the fake Crazyflie of test_hover_scripts: the real control loop,
watchdog and recorder, with only the radio faked."""

import argparse
import os
import sys
import time
import types

import pyarrow.parquet as pq
import pytest
from cflib.crazyflie.log import LogConfig

from test_hover_scripts import FakeCf, POS, _StopBeforeFlight

SIM2REAL = os.path.join(os.path.dirname(__file__), "..", "execution", "sim2real")
sys.path.append(SIM2REAL)
import excite_hover  # noqa: E402
import measure_delay_noise  # noqa: E402

AMP = 0.2


@pytest.fixture
def mod(monkeypatch):
    fast_time = types.SimpleNamespace(time=time.time, perf_counter=time.perf_counter, sleep=lambda s: None)
    monkeypatch.setattr(excite_hover, "time", fast_time)
    monkeypatch.setattr(LogConfig, "start", lambda logconf: logconf.fake_cf.fire(logconf))
    return excite_hover


def fly(mod, tmp_path, steps, target, duration=30.0, **cf_kwargs):
    cf = FakeCf(steps, **cf_kwargs)
    orig_packet = cf.packet
    cf.packet = lambda name: (
        {"ctrltarget.vx": AMP, "ctrltarget.vy": 0.0, "ctrltarget.vz": 0.0}
        if name == "ctrltarget" else orig_packet(name)
    )
    mod.Crazyflie = lambda **kwargs: cf
    path = tmp_path / "flight.parquet"
    flight = mod.ExciteHover(uri="fake", run_args=argparse.Namespace(record_path=str(path), record_every=1, log_every=100),
                             axis=0, amp=AMP, freq=0.5, duration=duration, target=target)
    flight._connected("fake")
    assert cf.loop_done.wait(timeout=30), "control loop did not finish"
    flight.stop()
    return flight, cf, pq.read_table(path).to_pydict(), str(path)


def test_pure_step_on_excited_axis_and_hold_on_others(mod, tmp_path, monkeypatch):
    """During excitation, the excited axis gets a pure step (hold=0) while unexcited axes stay held."""
    monkeypatch.setattr(mod, "EXC_START_S", 0.0)   # step starts at +AMP and stays there for 1 s
    target = [POS[0] + 0.1, POS[1], POS[2] + 1.0]  # z error saturates the hold at HOLD_MAX
    _, cf, rows, path = fly(mod, tmp_path, steps=20, target=target)
    assert rows["hold_vx"] == pytest.approx([0.0] * 20)
    assert rows["vx"] == pytest.approx([AMP] * 20)
    assert rows["vy"] == pytest.approx([0.0] * 20) and rows["vz"] == pytest.approx([mod.HOLD_MAX] * 20)
    assert rows["exc_vx"] == pytest.approx([AMP] * 20) and rows["exc_vy"] == rows["exc_vz"] == [0.0] * 20
    assert rows["echo_vx"] == pytest.approx([AMP] * 20) and rows["echo_vy"] == rows["echo_vz"] == [0.0] * 20
    sent = cf.calls_to("send_velocity_world_setpoint")
    assert [list(s[:3]) for s in sent] == [pytest.approx(list(r)) for r in zip(rows["vx"], rows["vy"], rows["vz"])]
    assert all(body >= 0.0 for body in rows["dt_body_ms"])
    # the estimator reads this recording as it is: the column contract between the two
    cols = measure_delay_noise.load(path)
    assert cols["exc_vx"].any() and not cols["exc_vy"].any() and not cols["exc_vz"].any()
    assert cols["echo_vx"].any()


def test_out_of_bounds_position_lands_immediately(mod, tmp_path, caplog):
    flight, _, rows, _ = fly(mod, tmp_path, steps=50, target=None, pos=(0.0, 0.0, 3.0))
    assert "EMERGENCY LANDING triggered" in caplog.messages
    assert rows["step"] == [] and not flight.running


def test_lands_after_duration(mod, tmp_path):
    flight, _, rows, _ = fly(mod, tmp_path, steps=1000, target=None, duration=0.2)
    assert 15 <= len(rows["step"]) <= 25 and not flight.running


def test_step_signal_shape():
    start = excite_hover.EXC_START_S
    assert excite_hover.step_signal(0, AMP, 0.5, start - 0.01) == [0.0, 0.0, 0.0]
    values = [excite_hover.step_signal(1, AMP, 0.5, start + 0.01 * i) for i in range(2000)]
    assert all(v[0] == v[2] == 0.0 and abs(v[1]) == AMP for v in values)
    assert excite_hover.step_signal(0, AMP, 0.5, start + 0.5)[0] == AMP
    assert excite_hover.step_signal(0, AMP, 0.5, start + 1.5)[0] == -AMP


def test_cli_refuses_flights_the_estimator_cannot_use(mod, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(mod, "ExciteHover", lambda **kw: seen.append(kw) or (_ for _ in ()).throw(_StopBeforeFlight()))
    ok = ["--record-path", str(tmp_path / "f.parquet"), "--exc-axis", "y"]
    monkeypatch.setattr(sys, "argv", ["excite_hover"] + ok)
    with pytest.raises(_StopBeforeFlight):
        mod.main()
    assert (seen[0]["axis"], seen[0]["metadata"]["excitation"]) == (1, "step")
    assert seen[0]["metadata"]["cflib_version"] != "unknown"
    for bad in ([], ok + ["--record-every", "2"], ok + ["--duration", "2"], ok + ["--exc-amp", "0.6"],
                ok + ["--exc-freq", "0"]):
        monkeypatch.setattr(sys, "argv", ["excite_hover"] + bad)
        with pytest.raises(SystemExit):
            mod.main()
    assert len(seen) == 1

