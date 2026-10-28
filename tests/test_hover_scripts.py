"""End-to-end tests of the four hover scripts against a fake Crazyflie.

Each test runs a script's real controller: `_connected` registers the real log
callbacks, the control loop runs at its real 100 Hz schedule, and `stop` closes
the recorder. Only the radio is faked: `FakeCf` records every command it is sent
and feeds one telemetry packet per control step through the real callbacks.
The script's own `time.sleep` is made instant so takeoff ramps and landing waits
do not slow the tests down; `wait_until` keeps real time.
"""

import argparse
import hashlib
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
sys.path.append(os.path.join(HERE, "..", "sim2real", "single_drone_exec", "hover"))
from analyze_vel import compute_metrics, load_flight  # noqa: E402
from cflib.crazyflie.log import LogConfig  # noqa: E402

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
IMU = (10.0, -20.0, 5.0, 0.01, -0.02, 1.0)   # gyro deg/s, acc g
FW_VEL = (0.3, -0.4, 0.05)                    # posCtl.targetV*: the firmware PID's velocity command
VBAT = 3.9


class _Commands:
    """Stands in for commander / high_level_commander / param: records every call."""

    def __init__(self, cf):
        self._cf = cf

    def __getattr__(self, name):
        def call(*args):
            self._cf.sent.append((name, args))
            if name == "land":                    # every control loop ends by landing
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
        x, y, z = self.pos(self.calls) if callable(self.pos) else self.pos
        return {
            "posvel": {"stateEstimate.x": x, "stateEstimate.y": y, "stateEstimate.z": z,
                       "stateEstimate.vx": VEL[0], "stateEstimate.vy": VEL[1], "stateEstimate.vz": VEL[2]},
            "quat": {"stateEstimate.qw": 1.0, "stateEstimate.qx": 0.0,
                     "stateEstimate.qy": 0.0, "stateEstimate.qz": 0.0},
            "quality": {"kalman.varPX": 1e-3, "kalman.varPY": 1e-3, "kalman.varPZ": 1e-3, "pm.vbat": VBAT},
            "motor": {f"motor.m{i + 1}": v for i, v in enumerate(MOTOR)},
            "gyro": {"gyro.x": GYRO_DEG[0], "gyro.y": GYRO_DEG[1], "gyro.z": GYRO_DEG[2]},
            "imu": dict(zip(["gyro.x", "gyro.y", "gyro.z", "acc.x", "acc.y", "acc.z"], IMU)),
            "posCtl": dict(zip(["posCtl.targetVX", "posCtl.targetVY", "posCtl.targetVZ"], FW_VEL)),
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
    monkeypatch.setattr(LogConfig, "start", lambda logconf: logconf.fake_cf.fire(logconf))
    return mod


@pytest.fixture(params=list(SCRIPTS))
def script(request, monkeypatch):
    return load_script(request.param, monkeypatch)


def fly(mod, tmp_path, steps, record=True, log_every=100, controller_kwargs=None, **cf_kwargs):
    """Connect, run the control loop for `steps` steps, stop. Returns (controller, cf, agent, rows)."""
    n_act = SCRIPTS[mod.__name__][0]
    cf = FakeCf(steps, **cf_kwargs)
    mod.Crazyflie = lambda **kwargs: cf
    path = str(tmp_path / "flight.parquet") if record else None
    run_args = argparse.Namespace(record_path=path, record_every=1, log_every=log_every)
    agent = FakeAgent(n_act)
    controller = mod.CrazyflieController(uri="fake", agent=agent, run_args=run_args,
                                         initial_target=[0.0, 0.0, 1.0], **(controller_kwargs or {}))
    controller._connected("fake")
    assert cf.loop_done.wait(timeout=30), "control loop did not finish"
    try:
        controller.stop()
    finally:
        rows = pq.read_table(path).to_pydict() if record and os.path.exists(path) else None
    assert cf.calls_to("send_stop_setpoint") == [], "a motor cut in flight drops the drone"
    assert cf.calls_to("land") == [(0.0, 2.0)], "the flight lands exactly once"
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
    assert cf.sent[-2:] == [("send_notify_setpoint_stop", ()), ("land", (0.0, 2.0))]


def test_without_record_path_only_logs(script, tmp_path, caplog):
    caplog.set_level("INFO", logger="CrazyflieRL")
    controller, cf, agent, rows = fly(script, tmp_path, steps=10, record=False, log_every=5)
    assert rows is None and list(tmp_path.iterdir()) == []
    assert [m.split("]")[0] for m in caplog.messages if m.startswith("[step ")] == ["[step 0", "[step 5"]


def test_stop_writes_the_file_even_if_close_link_fails(script, tmp_path):
    with pytest.raises(RuntimeError, match="radio gone"):
        fly(script, tmp_path, steps=20, close_link_error=RuntimeError("radio gone"))
    assert pq.read_table(tmp_path / "flight.parquet").to_pydict()["step"] == list(range(20))


def test_analyser_reads_exec_vel_recording(tmp_path, monkeypatch):
    """analyze_vel reads the columns exec_vel really records, so the two cannot drift apart."""
    fly(load_script("exec_vel", monkeypatch), tmp_path, steps=50)
    m = compute_metrics(load_flight(str(tmp_path / "flight.parquet")))
    assert m["mean_pos_err_m"] == pytest.approx(math.dist(POS, (0.0, 0.0, 1.0)), abs=1e-3)
    assert m["effective_rate_hz"] == pytest.approx(100.0, rel=0.1)
    assert m["mean_motor_pwm"] == pytest.approx(sum(MOTOR) / 4)
    assert m["cmd_smoothness_mps_per_step"] is not None and m["n_nonfinite_samples"] == 0


# ── Safety watchdog ──────────────────────────────────────────────────────────

def test_out_of_bounds_position_lands_immediately(script, tmp_path, caplog):
    controller, cf, agent, rows = fly(script, tmp_path, steps=50, pos=(0.0, 0.0, 3.0))
    assert "EMERGENCY LANDING triggered" in caplog.messages
    assert rows["step"] == [] and agent.seen == []
    assert not controller.running
    assert cf.calls_to("send_notify_setpoint_stop") == [()]
    assert cf.calls_to("land") == [(0.0, 2.0)]


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


def test_exec_vel_feeds_window_and_action_history(tmp_path, monkeypatch):
    """K=3, M=1, as vel_hovering._get_observations: frames oldest first, primed with copies
    of the first one, then the previous clamped command (zeros before the first)."""
    controller, _, agent, rows = fly(load_script("exec_vel", monkeypatch), tmp_path, steps=20,
                                     pos=lambda n: (0.2, -0.1, 1.0 + 0.01 * n),   # every frame differs
                                     controller_kwargs=dict(obs_input_hist=3, action_input_hist=1))
    frames = list(zip(*(rows[f] for f in controller.recorder.obs_fields)))
    assert len(agent.seen) == len(frames) == 20 and frames[0] != frames[1]
    for i, seen in enumerate(agent.seen):
        window = [frames[max(j, 0)] for j in (i - 2, i - 1, i)]
        assert seen == pytest.approx(_flat(window) + (ACTION[:3] if i else [0.0] * 3))


def test_exec_vel_pid_flies_the_firmware_position_pid(tmp_path, monkeypatch):
    """No policy: the target goes to the onboard position PID, and the recorded command is
    the velocity that PID asked for, so it compares column for column with an RL flight."""
    controller, cf, agent, rows = fly(load_script("exec_vel", monkeypatch), tmp_path, steps=20,
                                      controller_kwargs=dict(controller="pid"))
    assert cf.calls_to("send_position_setpoint") == [(0.0, 0.0, 1.0, 0.0)] * 20
    assert cf.calls_to("send_velocity_world_setpoint") == [] and agent.seen == []
    assert _columns(rows, controller.recorder.cmd_fields) == pytest.approx(list(FW_VEL) * 20)
    assert all(math.isnan(v) for f in controller.recorder.action_fields for v in rows[f])
    meta = pq.read_schema(tmp_path / "flight.parquet").metadata
    assert meta[b"controller"] == b"pid"
    assert {b"stabilizer.controller", b"posCtlPid.xKp", b"posCtlPid.zVelMax"} <= set(meta)


def test_exec_vel_records_telemetry_and_loop_timing(tmp_path, monkeypatch):
    controller, cf, _, rows = fly(load_script("exec_vel", monkeypatch), tmp_path, steps=20)
    imu = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
    assert _columns(rows, imu) == pytest.approx(list(IMU) * 20)
    assert rows["vbat"] == pytest.approx([VBAT] * 20) and rows["var_pz"] == pytest.approx([1e-3] * 20)
    assert all(0.0 <= ms <= 10.5 for ms in rows["slept_ms"]) and set(rows["overran"]) <= {0.0, 1.0}
    assert all(ms >= 0.0 for ms in rows["dt_infer_ms"] + rows["dt_send_ms"])
    meta = pq.read_schema(tmp_path / "flight.parquet").metadata
    assert meta[b"controller"] == b"rl" and b"posCtlPid.xKp" not in meta
    assert "posCtl" not in cf.configs   # the extra 100 Hz block is only paid for in pid mode


def test_exec_vel_lands_after_duration(tmp_path, monkeypatch):
    controller, cf, _, rows = fly(load_script("exec_vel", monkeypatch), tmp_path, steps=1000,
                                 controller_kwargs=dict(duration=0.2))
    assert 15 <= len(rows["step"]) <= 25 and not controller.running
    # the hand-over follows the last policy setpoint directly: no gap without a setpoint
    assert [m for m, _ in cf.sent[-3:]] == ["send_velocity_world_setpoint", "send_notify_setpoint_stop", "land"]


def test_exec_vel_missing_log_variable_leaves_nan(tmp_path, monkeypatch):
    """A firmware without posCtl.* must still fly the baseline, with NaN commands, not abort."""
    mod = load_script("exec_vel", monkeypatch)
    real_add = FakeCf._add_config

    def add_config(self, logconf):
        if logconf.name == "posCtl":
            raise KeyError("Variable posCtl.targetVX not in TOC")
        real_add(self, logconf)
    monkeypatch.setattr(FakeCf, "_add_config", add_config)
    controller, cf, _, rows = fly(mod, tmp_path, steps=5, controller_kwargs=dict(controller="pid"))
    assert len(cf.calls_to("send_position_setpoint")) == 5
    assert all(math.isnan(v) for v in rows["vx"])


def test_run_metadata_names_the_checkpoint(tmp_path, monkeypatch):
    mod = load_script("exec_vel", monkeypatch)
    ckpt = tmp_path / "best_agent.pt"
    ckpt.write_bytes(b"weights")
    args = argparse.Namespace(uri="fake", target=[0.0, 0.0, 1.0], duration=0.0, checkpoint=str(ckpt))
    meta = mod.run_metadata(args, 2, 1)
    assert meta["checkpoint_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert (meta["obs_input_hist"], meta["action_input_hist"]) == ("2", "1") and meta["git_commit"]
    assert all(isinstance(v, str) for v in meta.values())


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
        _run_main(script, monkeypatch, ["--checkpoint", "x.pt", "--record-every", "10", "--log-every", "50"])
    # the waypoint gate was removed: the target stays fixed, as within a sim episode
    for bad in (["--record-every", "0"], ["--log-every", "0"], ["--waypoint-radius", "0.3"],
                ["--record-interval", "0.1"], ["--log-interval", "1"]):
        with pytest.raises(SystemExit):
            _run_main(script, monkeypatch, bad)


# ── Policy input width: K frames + M past actions ────────────────────────────

def _save_checkpoint(mod, path, obs_space):
    """A checkpoint saved the way training saves one, with a non-trivial observation scaler."""
    policy = mod.Policy(observation_space=obs_space, action_space=3, device="cpu")
    cfg = mod.PPO_CFG(observation_preprocessor=mod.RunningStandardScaler,
                      observation_preprocessor_kwargs={"size": obs_space, "device": "cpu"})
    agent = mod.PPO(models={"policy": policy}, memory=None, cfg=cfg,
                    observation_space=obs_space, action_space=3, device="cpu")
    agent._observation_preprocessor.running_mean.fill_(0.3)
    agent.enable_training_mode(False)
    agent.save(str(path))
    return agent


def _mean_action(agent, obs):
    with torch.no_grad():
        return agent.act(obs, None, timestep=0, timesteps=1)[1]["mean_actions"]


def test_load_agent_roundtrip_and_wrong_width_fails(tmp_path, monkeypatch):
    mod = load_script("exec_vel", monkeypatch)
    monkeypatch.setattr(mod, "device", torch.device("cpu"))
    path = tmp_path / "best_agent.pt"
    saved = _save_checkpoint(mod, path, obs_space=6 * 8 + 3 * 1)
    loaded = mod.load_agent(str(path), torch.device("cpu"), 6 * 8 + 3 * 1)
    obs = torch.randn(4, 51)
    assert torch.equal(_mean_action(loaded, obs), _mean_action(saved, obs))
    for wrong in (6, 48, 54):
        with pytest.raises(RuntimeError, match="size mismatch"):
            mod.load_agent(str(path), torch.device("cpu"), wrong)


def _run_dir(tmp_path, name, env_yaml):
    run = tmp_path / name
    (run / "params").mkdir(parents=True)
    (run / "params" / "env.yaml").write_text(env_yaml)
    return str(run / "checkpoints" / "best_agent.pt")


ENV_YAML = """viewer:
  eye: !!python/tuple
  - 7.5
observation_space: 6
obs_input_hist: {k}
action_input_hist: {m}
sim:
  obs_input_hist: 99
"""


def test_obs_hist_comes_from_the_training_run(tmp_path, monkeypatch):
    """Width 12 is K=1, M=2 or K=2, M=0: only the run's env.yaml tells them apart."""
    mod = load_script("exec_vel", monkeypatch)
    k1m2 = _run_dir(tmp_path, "k1m2", ENV_YAML.format(k=1, m=2))
    k2m0 = _run_dir(tmp_path, "k2m0", ENV_YAML.format(k=2, m=0))
    assert mod.resolve_obs_hist(k1m2, None, None) == (1, 2)
    assert mod.resolve_obs_hist(k2m0, None, None) == (2, 0)
    assert mod.resolve_obs_hist(k2m0, 2, 0) == (2, 0)
    with pytest.raises(ValueError, match="--obs-input-hist 1 disagrees"):
        mod.resolve_obs_hist(k2m0, 1, None)
    with pytest.raises(ValueError, match="--action-input-hist 2 disagrees"):
        mod.resolve_obs_hist(k2m0, None, 2)
    # no env.yaml, or one written before WP1 (no knobs): the flags, then K=1, M=0
    pre_wp1 = _run_dir(tmp_path, "pre_wp1", "observation_space: 6\nhistory_len: 8\n")
    loose = str(tmp_path / "loose.pt")
    assert mod.resolve_obs_hist(pre_wp1, None, None) == (1, 0)
    assert mod.resolve_obs_hist(loose, 8, 1) == (8, 1)
    assert mod.resolve_obs_hist(None, None, None) == (1, 0)


def test_cli_sizes_the_policy_from_env_yaml_and_rejects_a_conflict(tmp_path, monkeypatch):
    mod = load_script("exec_vel", monkeypatch)
    ckpt = _run_dir(tmp_path, "run", ENV_YAML.format(k=8, m=1))
    seen = []
    monkeypatch.setattr(sys, "argv", ["exec_vel", "--checkpoint", ckpt])
    monkeypatch.setattr(mod, "load_agent", lambda *a: seen.append(a) or (_ for _ in ()).throw(_StopBeforeFlight()))
    with pytest.raises(_StopBeforeFlight):
        mod.main()
    assert seen[0][2] == 6 * 8 + 3 * 1
    for bad in (["--obs-input-hist", "4"], ["--obs-input-hist", "0"], ["--action-input-hist", "-1"]):
        monkeypatch.setattr(sys, "argv", ["exec_vel", "--checkpoint", ckpt] + bad)
        with pytest.raises(SystemExit):     # before load_agent, so before the radio
            mod.main()
    assert len(seen) == 1


def test_cli_pid_needs_no_checkpoint_and_rejects_one(monkeypatch):
    mod = load_script("exec_vel", monkeypatch)
    seen = []
    monkeypatch.setattr(mod, "load_agent", lambda *a: pytest.fail("pid must not load a policy"))
    monkeypatch.setattr(mod, "CrazyflieController",
                        lambda **kw: seen.append(kw) or (_ for _ in ()).throw(_StopBeforeFlight()))
    monkeypatch.setattr(sys, "argv", ["exec_vel", "--controller", "pid", "--duration", "5", "--target", "0", "0", "1"])
    with pytest.raises(_StopBeforeFlight):
        mod.main()
    assert (seen[0]["controller"], seen[0]["agent"], seen[0]["duration"]) == ("pid", None, 5.0)
    assert "checkpoint" not in seen[0]["metadata"]
    for bad in ([], ["--controller", "pid", "--checkpoint", "x.pt"], ["--duration", "-1"], ["--controller", "lqr"]):
        monkeypatch.setattr(sys, "argv", ["exec_vel"] + bad)
        with pytest.raises(SystemExit):
            mod.main()
    assert len(seen) == 1
