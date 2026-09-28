"""End-to-end tests of the four hover scripts against a fake Crazyflie.

Each test runs a script's real controller: `_connected` registers the real log
callbacks, the control loop runs at its real 100 Hz schedule, and `stop` closes
the recorder. Only the radio is faked: `FakeCf` records every command it is sent
and feeds one telemetry packet per control step through the real callbacks.
The script's own `time.sleep` is made instant so takeoff ramps and landing waits
do not slow the tests down; `wait_until` keeps real time.
"""

import argparse
import importlib
import importlib.util
import math
import os
import sys
import threading
import time
import types

import pyarrow.parquet as pq
import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "execution", "single_drone_exec"))
sys.path.append(os.path.join(HERE, "..", "execution", "single_drone_exec", "hover"))
from cflib.crazyflie.log import LogConfig  # noqa: E402
from common.utils import WaypointGate  # noqa: E402

COMMON_DIR = os.path.realpath(os.path.join(HERE, "..", "execution", "single_drone_exec", "common"))

SCRIPTS = {
    # name: (action size, obs size, setpoint method the policy's command goes through)
    "exec_vel": (3, 6, "send_velocity_world_setpoint"),
    "exec_pos": (3, 6, "send_position_setpoint"),
    "exec_att": (4, 6, "send_setpoint_manual"),
    "exec_rate": (4, 18, "send_setpoint_manual"),
}
ACTION = [0.1, -0.2, 0.3, -0.23]
POS, VEL, GYRO_DEG = (0.2, -0.1, 1.0), (0.05, 0.0, -0.02), (10.0, -20.0, 5.0)
MOTOR = [30000, 30100, 30200, 30300]


class _Commands:
    """Stands in for commander / high_level_commander / param: records every call."""

    def __init__(self, cf):
        self._cf = cf

    def __getattr__(self, name):
        def call(*args):
            self._cf.sent.append((name, args))
            if name == "send_stop_setpoint":      # every control loop ends with it
                self._cf.loop_done.set()
        return call


class _Caller:
    def add_callback(self, cb):
        pass


class FakeCf:
    def __init__(self, steps, pos=POS, telemetry_steps=None, close_link_error=None):
        self.connected = self.disconnected = self.connection_failed = self.connection_lost = _Caller()
        self.commander = self.high_level_commander = self.param = _Commands(self)
        self.log = types.SimpleNamespace(add_config=self._add_config)
        self.configs, self.sent, self.loop_done = {}, [], threading.Event()
        self.steps, self.pos, self.calls = steps, pos, 0
        self.telemetry_steps = steps if telemetry_steps is None else telemetry_steps
        self.close_link_error = close_link_error

    def _add_config(self, logconf):
        logconf.fake_cf = self
        self.configs[logconf.name] = logconf

    def packet(self, name):
        x, y, z = self.pos
        return {
            "posvel": {"stateEstimate.x": x, "stateEstimate.y": y, "stateEstimate.z": z,
                       "stateEstimate.vx": VEL[0], "stateEstimate.vy": VEL[1], "stateEstimate.vz": VEL[2]},
            "quat": {"stateEstimate.qw": 1.0, "stateEstimate.qx": 0.0,
                     "stateEstimate.qy": 0.0, "stateEstimate.qz": 0.0},
            "quality": {"kalman.varPX": 1e-3, "kalman.varPY": 1e-3, "kalman.varPZ": 1e-3},
            "motor": {f"motor.m{i + 1}": v for i, v in enumerate(MOTOR)},
            "gyro": {"gyro.x": GYRO_DEG[0], "gyro.y": GYRO_DEG[1], "gyro.z": GYRO_DEG[2]},
        }[name]

    def fire(self, logconf):
        logconf.data_received_cb.call(0, self.packet(logconf.name), logconf)

    def is_connected(self):
        """Called once per control step: deliver telemetry, then stay up for `steps` steps."""
        if self.calls < self.telemetry_steps:
            for logconf in self.configs.values():
                self.fire(logconf)
        self.calls += 1
        return self.calls <= self.steps

    def close_link(self):
        if self.close_link_error:
            raise self.close_link_error

    def calls_to(self, name):
        return [args for method, args in self.sent if method == name]


class FakeAgent:
    def __init__(self, n_act):
        self.action, self.seen = torch.tensor([ACTION[:n_act]]), []

    def act(self, obs, *args, **kwargs):
        self.seen.append(obs.reshape(-1).tolist())
        return self.action, {"mean_actions": self.action}


def load_script(name, monkeypatch):
    mod = importlib.import_module(name)
    fast_time = types.SimpleNamespace(time=time.time, perf_counter=time.perf_counter, sleep=lambda s: None)
    monkeypatch.setattr(mod, "time", fast_time)
    monkeypatch.setattr(mod, "waypoint_gate", WaypointGate(0.0, 5.0))   # radius 0: fixed target
    monkeypatch.setattr(LogConfig, "start", lambda logconf: logconf.fake_cf.fire(logconf))
    return mod


@pytest.fixture(params=list(SCRIPTS))
def script(request, monkeypatch):
    return load_script(request.param, monkeypatch)


def fly(mod, tmp_path, steps, record=True, log_every=100, **cf_kwargs):
    """Connect, run the control loop for `steps` steps, stop. Returns (controller, cf, agent, rows)."""
    n_act = SCRIPTS[mod.__name__][0]
    cf = FakeCf(steps, **cf_kwargs)
    mod.Crazyflie = lambda **kwargs: cf
    path = str(tmp_path / "flight.parquet") if record else None
    run_args = argparse.Namespace(record_path=path, record_every=1, log_every=log_every)
    agent = FakeAgent(n_act)
    controller = mod.CrazyflieController(uri="fake", agent=agent, run_args=run_args,
                                         initial_target=[0.0, 0.0, 1.0])
    controller._connected("fake")
    assert cf.loop_done.wait(timeout=30), "control loop did not finish"
    try:
        controller.stop()
    finally:
        rows = pq.read_table(path).to_pydict() if record and os.path.exists(path) else None
    return controller, cf, agent, rows


def _flat(rows):
    return [value for row in rows for value in row]


def _columns(table, names):
    """Row-major values of `names`, to compare with per-step lists."""
    return _flat(zip(*(table[name] for name in names)))


# ── Normal flight ────────────────────────────────────────────────────────────

def test_records_one_consistent_row_per_step(script, tmp_path, caplog):
    caplog.set_level("INFO", logger="CrazyflieRL")
    n_act, n_obs, setpoint = SCRIPTS[script.__name__]
    controller, cf, agent, rows = fly(script, tmp_path, steps=110)
    rec = controller.recorder

    assert rows["step"] == list(range(110))
    assert len(rec.obs_fields) == n_obs and len(rec.action_fields) == n_act
    # the row holds exactly what the policy saw and what the drone was sent
    assert _columns(rows, rec.obs_fields) == pytest.approx(_flat(agent.seen))
    assert all(rows[f] == pytest.approx([a] * 110) for f, a in zip(rec.action_fields, ACTION))
    sent = cf.calls_to(setpoint)[-110:]
    assert _columns(rows, rec.cmd_fields) == pytest.approx(_flat(args[:len(rec.cmd_fields)] for args in sent))
    # state columns come from the telemetry packets
    assert rows["pos_z"] == pytest.approx([POS[2]] * 110)
    assert rows["vel_x"] == pytest.approx([VEL[0]] * 110)
    assert rows["target_z"] == pytest.approx([1.0] * 110)
    assert rows["motor_m4_pwm"] == [MOTOR[3]] * 110
    assert all(0.0 <= age < 0.05 for age in rows["obs_age_s"])
    periods = [b - a for a, b in zip(rows["control_time"], rows["control_time"][1:])]
    assert 0.009 < sum(periods) / len(periods) < 0.011

    status = [m for m in caplog.messages if m.startswith("[step ")]
    assert [m.split("]")[0] for m in status] == ["[step 0", "[step 100"]
    assert any(m.startswith(f"Cmd: {rec.cmd_fields[0]}=") for m in caplog.messages)
    assert rec.skipped == 0


def test_without_record_path_only_logs(script, tmp_path, caplog):
    caplog.set_level("INFO", logger="CrazyflieRL")
    controller, cf, agent, rows = fly(script, tmp_path, steps=10, record=False, log_every=5)
    assert rows is None and list(tmp_path.iterdir()) == []
    assert [m.split("]")[0] for m in caplog.messages if m.startswith("[step ")] == ["[step 0", "[step 5"]


def test_stop_writes_the_file_even_if_close_link_fails(script, tmp_path):
    with pytest.raises(RuntimeError, match="radio gone"):
        fly(script, tmp_path, steps=20, close_link_error=RuntimeError("radio gone"))
    assert pq.read_table(tmp_path / "flight.parquet").to_pydict()["step"] == list(range(20))


# ── Safety watchdog ──────────────────────────────────────────────────────────

def test_out_of_bounds_position_lands_immediately(script, tmp_path, caplog):
    controller, cf, agent, rows = fly(script, tmp_path, steps=50, pos=(0.0, 0.0, 3.0))
    assert "EMERGENCY LANDING triggered" in caplog.messages
    assert rows["step"] == [] and agent.seen == []
    assert not controller.running


def test_stale_position_lands_after_timeout(script, tmp_path, caplog):
    controller, cf, agent, rows = fly(script, tmp_path, steps=300, telemetry_steps=20)
    assert any(m.startswith("Position data stale") for m in caplog.messages)
    # ~0.5 s (POS_STALE_TIMEOUT_S) of steps after the last packet, then no more rows
    assert 20 + 40 <= len(rows["step"]) <= 20 + 60
    assert rows["step"] == list(range(len(rows["step"])))


# ── Script-specific behaviour ────────────────────────────────────────────────

def test_exec_pos_commands_current_position_plus_displacement(tmp_path, monkeypatch):
    _, cf, _, rows = fly(load_script("exec_pos", monkeypatch), tmp_path, steps=5)
    expected = [p + 0.1 * a for p, a in zip(POS, ACTION)]   # MAX_DISPLACEMENT = 0.1 m
    assert [rows[f"desired_pos_{c}"][0] for c in "xyz"] == pytest.approx(expected)


def test_exec_rate_uses_gyro_and_enables_rate_mode(tmp_path, monkeypatch):
    _, cf, _, rows = fly(load_script("exec_rate", monkeypatch), tmp_path, steps=5)
    assert [rows[f"ang_vel_{c}"][0] for c in "xyz"] == pytest.approx([math.radians(g) for g in GYRO_DEG])
    assert [rows[f"rotmat_{i}"][0] for i in range(9)] == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert ("flightmode.stabModeRoll", "0") in cf.calls_to("set_value")


def test_default_quaternion_is_identity(script):
    """Until the first quat packet, the obs must stay finite (exec_rate divides by |q|²)."""
    controller = script.CrazyflieController(
        uri="fake", agent=FakeAgent(SCRIPTS[script.__name__][0]),
        run_args=argparse.Namespace(record_path=None, record_every=1, log_every=100))
    assert controller.current_quat.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_common_modules_do_not_shadow_top_level_names(script):
    """Review point 4: `common` is imported as a package, never put on sys.path itself."""
    assert COMMON_DIR not in {os.path.realpath(p) for p in sys.path}
    for name in ("utils", "flight_recorder"):
        spec = importlib.util.find_spec(name)
        assert spec is None or not os.path.realpath(spec.origin).startswith(COMMON_DIR)
    assert script.CrazyflieStateBase.__module__ == "common.flight_recorder"


# ── Command line ─────────────────────────────────────────────────────────────

class _StopBeforeFlight(Exception):
    pass


def _run_main(mod, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", [mod.__name__] + argv)
    monkeypatch.setattr(mod, "load_agent", lambda *a: (_ for _ in ()).throw(_StopBeforeFlight()))
    mod.main()


def test_cli_shared_flags(script, monkeypatch):
    with pytest.raises(_StopBeforeFlight):
        _run_main(script, monkeypatch, ["--waypoint-radius", "0.3", "--waypoint-hold", "2",
                                        "--record-every", "10", "--log-every", "50"])
    assert (script.waypoint_gate.radius_m, script.waypoint_gate.hold_s) == (0.3, 2.0)
    for bad in (["--record-every", "0"], ["--log-every", "0"], ["--waypoint-radius", "-1"],
                ["--record-interval", "0.1"], ["--log-interval", "1"]):
        with pytest.raises(SystemExit):
            _run_main(script, monkeypatch, bad)
