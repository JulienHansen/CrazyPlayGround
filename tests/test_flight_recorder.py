"""FlightRecorder must write exactly one row per recorded control step.

The polling recorder sampled the controller on its own clock, so rows were
duplicated or dropped against control steps with no way to tell afterwards.
Now the control loop pushes each row; these tests pin that the file then maps
1:1 to steps and that close() drains everything.
"""

import os
import sys
import threading
import time

import pyarrow.parquet as pq
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "execution", "single_drone_exec"))
from common.flight_recorder import CrazyflieStateBase, FlightRecorder  # noqa: E402
from common.utils import wait_until  # noqa: E402

FIELDS = dict(obs_fields=["o0", "o1"], action_fields=["a0"], cmd_fields=["c0"])


def _sample(step, t=0.0):
    return {
        "step": step, "control_time": t, "obs_age_s": 0.004,
        "pos": [1.0, 2.0, 3.0], "vel": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
        "target": [0.0, 0.0, 1.0],
        "obs": [0.1, 0.2], "action": [0.3], "cmd": [0.4], "motor": [1, 2, 3, 4],
    }


def _read(path):
    return pq.read_table(path).to_pydict()


def test_one_row_per_step_from_a_100hz_loop(tmp_path):
    path = tmp_path / "flight.parquet"
    rec = FlightRecorder(str(path), **FIELDS)
    rec.start()

    def control_loop():
        deadline = time.perf_counter()
        for step in range(300):
            rec.record(_sample(step, time.time()))
            deadline, _, _ = wait_until(deadline + 0.01, 0.002)

    t = threading.Thread(target=control_loop)
    t.start()
    t.join()
    rec.close()

    cols = _read(path)
    assert cols["step"] == list(range(300))
    periods = [b - a for a, b in zip(cols["control_time"], cols["control_time"][1:])]
    assert 0.0095 < sum(periods) / len(periods) < 0.0105


def test_record_every_keeps_one_step_in_n(tmp_path):
    path = tmp_path / "flight.parquet"
    rec = FlightRecorder(str(path), record_every=10, **FIELDS)
    rec.start()
    for step in range(100):
        rec.record(_sample(step))
    rec.close()
    assert _read(path)["step"] == list(range(0, 100, 10))


def test_close_drains_every_queued_row(tmp_path):
    """The stop sentinel is queued after the rows, so none are lost on shutdown."""
    path = tmp_path / "flight.parquet"
    rec = FlightRecorder(str(path), batch_size=50, **FIELDS)
    rec.start()
    for step in range(5003):          # not a multiple of batch_size
        rec.record(_sample(step))
    rec.close()
    assert _read(path)["step"] == list(range(5003))


def test_step_mode_schema(tmp_path):
    path = tmp_path / "flight.parquet"
    rec = FlightRecorder(str(path), **FIELDS)
    rec.start()
    rec.record(_sample(0))
    rec.close()
    schema = pq.read_schema(path)
    assert schema.names[:3] == ["step", "control_time", "obs_age_s"]
    assert "write_time" not in schema.names
    assert str(schema.field("step").type) == "int64"
    assert _read(path)["obs_age_s"] == [0.004]


def test_record_every_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        FlightRecorder(str(tmp_path / "x.parquet"), record_every=0, **FIELDS)
    with pytest.raises(ValueError):
        FlightRecorder(str(tmp_path / "x.parquet"), log_every=0, **FIELDS)


# ── Status logging ───────────────────────────────────────────────────────────

def _logged_steps(caplog):
    return [int(r.getMessage().split("]")[0].removeprefix("[step "))
            for r in caplog.records if r.getMessage().startswith("[step ")]


def test_logs_every_n_steps(tmp_path, caplog):
    caplog.set_level("INFO", logger="CrazyflieRL")
    rec = FlightRecorder(str(tmp_path / "flight.parquet"), log_every=10, **FIELDS)
    rec.start()
    for step in range(50):
        rec.record(_sample(step))
    rec.close()
    assert _logged_steps(caplog) == [0, 10, 20, 30, 40]
    assert "Cmd: c0=+0.40 | PWM: m1=1 m2=2 m3=3 m4=4" in caplog.messages


def test_logs_without_recording(tmp_path, caplog):
    caplog.set_level("INFO", logger="CrazyflieRL")
    rec = FlightRecorder(None, log_every=10, **FIELDS)
    rec.start()
    for step in range(30):
        rec.record(_sample(step))
    rec.close()
    assert _logged_steps(caplog) == [0, 10, 20]
    assert list(tmp_path.iterdir()) == []


def test_record_and_log_rates_are_independent(tmp_path, caplog):
    caplog.set_level("INFO", logger="CrazyflieRL")
    path = tmp_path / "flight.parquet"
    rec = FlightRecorder(str(path), record_every=10, log_every=25, **FIELDS)
    rec.start()
    for step in range(50):
        rec.record(_sample(step))
    rec.close()
    assert _read(path)["step"] == list(range(0, 50, 10))
    assert _logged_steps(caplog) == [0, 25]


# ── CrazyflieStateBase ───────────────────────────────────────────────────────

class _FakeLog:
    def __init__(self):
        self.configs = {}

    def add_config(self, logconf):
        self.configs[logconf.name] = logconf


class _FakeCf:
    def __init__(self):
        self.log = _FakeLog()


@pytest.fixture
def controller(monkeypatch):
    from cflib.crazyflie.log import LogConfig
    monkeypatch.setattr(LogConfig, "start", lambda self: None)   # no radio

    class Controller(CrazyflieStateBase):
        def __init__(self):
            super().__init__(torch.device("cpu"))
            self.cf = _FakeCf()

    c = Controller()
    c.setup_state_logging()
    return c


def _fire(c, name, data):
    logconf = c.cf.log.configs[name]
    logconf.data_received_cb.call(0, data, logconf)


def test_callbacks_update_the_declared_fields(controller):
    _fire(controller, "posvel", {"stateEstimate.x": 1.0, "stateEstimate.y": 2.0, "stateEstimate.z": 3.0,
                                 "stateEstimate.vx": 0.1, "stateEstimate.vy": 0.2, "stateEstimate.vz": 0.3})
    _fire(controller, "quat", {"stateEstimate.qw": 1.0, "stateEstimate.qx": 0.0,
                               "stateEstimate.qy": 0.0, "stateEstimate.qz": 0.0})
    _fire(controller, "quality", {"kalman.varPX": 0.01, "kalman.varPY": 0.02, "kalman.varPZ": 0.03})
    _fire(controller, "motor", {"motor.m1": 10, "motor.m2": 20, "motor.m3": 30, "motor.m4": 40})

    assert controller.position_received
    assert controller.current_pos.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert controller.current_vel.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert controller.current_quat.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert controller._pos_variance.tolist() == pytest.approx([0.01, 0.02, 0.03])
    assert controller.current_motor_pwm == [10, 20, 30, 40]
    assert controller._last_pos_time == pytest.approx(time.time(), abs=1.0)


def test_snapshot_is_a_copy(controller):
    snap = controller.snapshot()
    _fire(controller, "posvel", {"stateEstimate.x": 5.0, "stateEstimate.y": 5.0, "stateEstimate.z": 5.0,
                                 "stateEstimate.vx": 0.0, "stateEstimate.vy": 0.0, "stateEstimate.vz": 0.0})
    assert snap.pos.tolist() == [0.0, 0.0, 0.0]
    assert snap.pos_time == 0.0
    assert controller.snapshot().pos.tolist() == [5.0, 5.0, 5.0]


def test_record_step_builds_the_row(controller, tmp_path):
    import argparse
    args = argparse.Namespace(record_path=str(tmp_path / "flight.parquet"), record_every=1, log_every=100)
    controller.make_recorder(args, obs_fields=["o0", "o1"], action_fields=["a0"], cmd_fields=["c0", "c1"])
    controller.recorder.start()
    state = controller.snapshot()._replace(pos_time=10.0, motor=[1, 2, 3, 4])
    controller.record_step(7, state, 10.005, torch.tensor([0.0, 0.0, 1.0]),
                           torch.tensor([0.1, 0.2]), torch.tensor([0.3]), [0.4, 0.5])
    controller.recorder.close()
    row = {k: v[0] for k, v in _read(tmp_path / "flight.parquet").items()}
    assert row["step"] == 7
    assert row["obs_age_s"] == pytest.approx(0.005)
    assert (row["o1"], row["a0"], row["c1"], row["motor_m4_pwm"], row["target_z"]) == pytest.approx((0.2, 0.3, 0.5, 4, 1.0))


def test_run_args_reject_invalid_values():
    import argparse
    from common.utils import add_run_args
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    assert parser.parse_args([]).log_every == 100
    for bad in (["--log-every", "0"], ["--record-every", "0"], ["--waypoint-radius", "-1"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_a_bad_row_is_skipped_not_fatal(tmp_path, caplog):
    """A row that raises must not stop recording and status lines for the rest of the flight."""
    caplog.set_level("INFO", logger="CrazyflieRL")
    path = tmp_path / "flight.parquet"
    rec = FlightRecorder(str(path), log_every=1, **FIELDS)
    rec.start()
    for step in range(10):
        sample = _sample(step)
        if step in (3, 6):
            del sample["pos"]
        rec.record(sample)
    rec.close()
    assert _read(path)["step"] == [0, 1, 2, 4, 5, 7, 8, 9]
    assert rec.skipped == 2
    assert sum("Recorder failed at step 3" in m for m in caplog.messages) == 1   # first error only
    assert "Recorder skipped 2 row(s), see the first error above" in caplog.messages
