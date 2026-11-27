"""Tests of the single-drone trajectory-tracking script against a fake Crazyflie.

They check that `traj_tracking/exec_track.py` reproduces TrackingEnv for every control mode:
- the policy input is the last K 22-D frames, oldest first, primed with K copies of the first
  one, then the last M commanded (clamped) actions, primed with the mode's hover command;
- the reference moves at a constant linear speed along a lap resampled by arc length (#96), the
  look-ahead points of step s lying speed * (s + 5k) * 0.01 m along it;
- each mode sends the setpoint of TrackingEnv._pre_physics_step, with the action scales of the
  run's env.yaml.
The fake radio is the one of `test_hover_scripts.py`. The high-level-commander waits of the
takeoff and of the go_to the start are recorded instead of polled, so the fake's step budget
counts policy steps only.
"""

import argparse
import importlib.util
import math
import os
import sys
import time
import types

import pyarrow.parquet as pq
import pytest
import torch
from cflib.crazyflie.log import LogConfig

from test_hover_scripts import (ACTION, ENV_YAML, IMU, MOTOR, VBAT, VEL, FakeAgent, FakeCf, _Commands, _export_dir,
                                _run_dir, _save_checkpoint, _StopBeforeFlight)

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "..", "execution", "single_drone_exec", "traj_tracking", "exec_track.py")
TRAINING_TRAJECTORIES = os.path.join(HERE, "..", "source", "CrazyPlayGround", "CrazyPlayGround", "tasks", "direct",
                                     "track", "trajectories.py")
FRAME = 22
MAX_VEL = 3.0  # VelTrackingEnvCfg.max_velocity


def load_script(monkeypatch):
    spec = importlib.util.spec_from_file_location("traj_exec_track", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fast_time = types.SimpleNamespace(time=time.time, perf_counter=time.perf_counter, sleep=lambda s: None)
    monkeypatch.setattr(mod, "time", fast_time)
    monkeypatch.setattr(sys.modules["common.utils"], "time", fast_time)  # the fly-back's hover waits
    monkeypatch.setattr(mod, "device", torch.device("cpu"))
    monkeypatch.setattr(LogConfig, "start", lambda logconf: logconf.fake_cf.fire(logconf))
    mod.hlc_waits = []

    def hover_for(controller, duration, stale_timeout_s):  # takeoff and go_to: record, do not poll
        mod.hlc_waits.append(duration)
        return not controller.landing
    monkeypatch.setattr(sys.modules["common.tracking"], "hover_for", hover_for)
    return mod


class TrajFakeCf(FakeCf):
    """FakeCf with a platform, for the arming request, and an optional attitude `quat_wxyz`."""

    def __init__(self, steps, quat_wxyz=(1.0, 0.0, 0.0, 0.0), **kwargs):
        super().__init__(steps, **kwargs)
        self.platform = _Commands(self)
        self.quat_wxyz = quat_wxyz

    def packet(self, name):
        if name == "quat":
            return dict(zip(("stateEstimate.qw", "stateEstimate.qx", "stateEstimate.qy", "stateEstimate.qz"),
                            self.quat_wxyz))
        return super().packet(name)


def moving(n):
    """Position at control step n: every frame differs, so the window order is visible."""
    return (0.2, -0.1, 1.0 + 0.01 * n)


def on_reference(mod):
    """Position at step n on the default reference: no tracking error, for the low-level modes."""
    ref = mod.Reference()
    return lambda n: tuple(ref.scored(n).tolist())


def thrust(mod):
    return mod.ThrustMap(0.5, mod.FIRMWARE_MAX_THRUST_SCALE)


def commands(mod):
    """One Command per mode, at the TrackingEnvCfg defaults."""
    s = mod.DEFAULT_SCALES
    return {"position": mod.PositionCommand(s["max_position_delta"]),
            "velocity": mod.VelocityCommand(s["max_velocity"]),
            "attitude": mod.AttitudeCommand(s["max_roll_pitch"], s["max_yaw_rate"], thrust(mod)),
            "body_rate": mod.RateCommand(s["max_body_rate"], thrust(mod))}


SETPOINT = {"position": "send_position_setpoint", "velocity": "send_velocity_world_setpoint",
            "attitude": "send_setpoint_manual", "body_rate": "send_setpoint_manual"}
LOW_LEVEL = ("send_setpoint_manual",)


def run(controller):
    """Connect: register the real log callbacks and run the control loop to its end."""
    controller._connected("fake")
    if controller.loop_thread is not None:
        controller.loop_thread.join(timeout=30)
        assert not controller.loop_thread.is_alive(), "control loop did not finish"
    return controller


def make(mod, cf, record_path=None, **kwargs):
    mod.Crazyflie = lambda **kw: cf
    run_args = argparse.Namespace(record_path=record_path, record_every=1, log_every=100)
    kwargs.setdefault("command", mod.VelocityCommand(MAX_VEL))
    kwargs.setdefault("agent", FakeAgent(kwargs["command"].act_dim))
    return mod.CrazyflieController(uri="fake", run_args=run_args, **kwargs)


def fly(mod, steps, k=1, m=0, action=None, pos=moving, command=None, **kwargs):
    """Run the control loop for `steps` steps. Returns (controller, cf, agent)."""
    cf = TrajFakeCf(steps, pos=pos)
    command = command or mod.VelocityCommand(MAX_VEL)
    agent = FakeAgent(command.act_dim)
    if action is not None:
        agent.action = torch.tensor([action])
    controller = run(make(mod, cf, agent=agent, command=command, obs_input_hist=k, action_input_hist=m, **kwargs))
    assert len(agent.seen) == steps
    return controller, cf, agent


def test_default_input_is_one_frame(monkeypatch):
    """K=1, M=0 (a checkpoint trained without history): the input is the plain 22-D frame."""
    mod = load_script(monkeypatch)
    _, cf, agent = fly(mod, steps=10)
    assert all(len(seen) == FRAME for seen in agent.seen)
    sent = cf.calls_to("send_velocity_world_setpoint")
    assert [list(args) for args in sent] == [pytest.approx([a * MAX_VEL for a in ACTION[:3]] + [0.0])] * 10


def test_frame_is_built_as_tracking_env(monkeypatch):
    """rpos_1..4 to the look-ahead points, lin_vel_b, gyro in rad/s, quat: identity attitude here."""
    mod = load_script(monkeypatch)
    controller, _, agent = fly(mod, steps=3)
    for step, seen in enumerate(agent.seen):
        pos = torch.tensor(moving(step))
        rpos = (controller.reference.waypoints(step) - pos).flatten().tolist()
        assert seen == pytest.approx(rpos + list(VEL) + [math.radians(g) for g in IMU[:3]] + [1.0, 0.0, 0.0, 0.0])


def test_window_and_action_history(monkeypatch):
    """K=3, M=2: frames oldest first, primed with copies of the first, then the last two commands."""
    mod = load_script(monkeypatch)
    _, _, plain = fly(mod, steps=12)
    frames = plain.seen
    assert frames[0] != frames[1]
    _, _, agent = fly(mod, steps=12, k=3, m=2)
    zeros, a = [0.0] * 3, ACTION[:3]
    for i, seen in enumerate(agent.seen):
        window = [frames[max(j, 0)] for j in (i - 2, i - 1, i)]
        hist = [zeros + zeros, zeros + a][i] if i < 2 else a + a
        assert len(seen) == FRAME * 3 + 3 * 2
        assert seen == pytest.approx([v for frame in window for v in frame] + hist)


def test_history_holds_clamped_actions(monkeypatch):
    """The history holds the command actually sent, clamped to [-1, 1], as cmd_history does."""
    mod = load_script(monkeypatch)
    _, cf, agent = fly(mod, steps=5, m=1, action=[1.5, -2.0, 0.3])
    assert agent.seen[0][-3:] == [0.0, 0.0, 0.0]
    assert all(seen[-3:] == pytest.approx([1.0, -1.0, 0.3]) for seen in agent.seen[1:])
    sent = cf.calls_to("send_velocity_world_setpoint")[-1]
    assert list(sent[:3]) == pytest.approx([MAX_VEL, -MAX_VEL, 0.3 * MAX_VEL])


def test_load_agent_width(tmp_path, monkeypatch):
    mod = load_script(monkeypatch)
    path = tmp_path / "best_agent.pt"
    width = FRAME * 2 + 3 * 1
    saved = _save_checkpoint(mod, path, obs_space=width)
    loaded = mod.load_agent(str(path), torch.device("cpu"), width, 3)
    obs = torch.randn(4, width)
    with torch.no_grad():
        act = [ag.act(obs, None, timestep=0, timesteps=1)[1]["mean_actions"] for ag in (loaded, saved)]
    assert torch.equal(*act)
    for wrong in (FRAME, FRAME * 2, FRAME * 2 + 3 * 2):
        with pytest.raises(RuntimeError, match="size mismatch"):
            mod.load_agent(str(path), torch.device("cpu"), wrong, 3)
    with pytest.raises(RuntimeError, match="size mismatch"):  # a 4-action head for a 3-action checkpoint
        mod.load_agent(str(path), torch.device("cpu"), width, 4)


# ── Control modes: TrackingEnv._pre_physics_step on the radio ──────────────────

@pytest.mark.parametrize("mode", ["position", "velocity", "attitude", "body_rate"])
def test_each_mode_sends_its_setpoint(monkeypatch, mode):
    mod = load_script(monkeypatch)
    command = commands(mod)[mode]
    _, cf, agent = fly(mod, steps=5, command=command, pos=on_reference(mod))
    a = ACTION[:command.act_dim]
    sent = cf.calls_to(SETPOINT[mode])[:5]
    assert len(sent) == 5 and all(len(seen) == FRAME for seen in agent.seen)
    th = command.thrust if command.low_level else None
    expected = {  # the first four arguments: setpoint, then yaw (rate) or thrust %
        "position": lambda step: [p + 0.5 * x for p, x in zip(on_reference(mod)(step), a)] + [0.0],
        "velocity": lambda step: [x * MAX_VEL for x in a] + [0.0],
        "attitude": lambda step: [a[0] * 30.0, a[1] * 30.0, a[2] * 90.0, th.pct(max(0.0, a[3]))],   # norm = clamp(a, 0, 1)
        "body_rate": lambda step: [a[0] * 180.0, a[1] * 180.0, a[2] * 180.0, th.pct((a[3] + 1) / 2)],  # norm = (a + 1) / 2
    }[mode]
    for step, args in enumerate(sent):
        assert list(args[:4]) == pytest.approx(expected(step)), step
        if command.low_level:  # send_setpoint_manual's `rate` flag: angles for Att, body rates for Rate
            assert args[4] is (mode == "body_rate")


@pytest.mark.parametrize("mode, hover", [("position", [0.0] * 3), ("velocity", [0.0] * 3),
                                         ("attitude", [0.0, 0.0, 0.0, 0.5 / 1.0387]),
                                         ("body_rate", [0.0, 0.0, 0.0, 2 * 0.5 / 1.0387 - 1])])
def test_action_history_starts_at_the_hover_command(monkeypatch, mode, hover):
    """TrackingEnv fills cmd_history with _hover_cmd at reset: zero setpoints for Pos/Vel, hover
    thrust for Att (0.481) and Rate (-0.037) with max_thrust_scale 1.5387."""
    mod = load_script(monkeypatch)
    command = commands(mod)[mode]
    _, _, agent = fly(mod, steps=3, m=2, command=command, pos=on_reference(mod))
    assert command.hover_action() == pytest.approx(hover, abs=1e-3)
    assert agent.seen[0][-2 * len(hover):] == pytest.approx(hover * 2, abs=1e-3)
    assert agent.seen[1][-2 * len(hover):] == pytest.approx(hover + ACTION[:len(hover)], abs=1e-3)


def test_thrust_map_puts_the_hover_command_on_the_measured_hover(monkeypatch):
    mod = load_script(monkeypatch)
    th = thrust(mod)
    assert th.pct(th.hover_norm) == pytest.approx(mod.HOVER_THRUST_PCT)
    assert (th.pct(0.0), th.pct(1.0), th.pct(-0.5), th.pct(2.0)) == pytest.approx((25.0, 100.0, 25.0, 100.0))
    rate = commands(mod)["body_rate"]
    assert th.pct((rate.hover_action()[3] + 1) / 2) == pytest.approx(77.0)  # Rate: norm = (a + 1) / 2
    att = commands(mod)["attitude"]
    assert th.pct(att.hover_action()[3]) == pytest.approx(77.0)            # Att: norm = clamp(a, 0, 1)
    # the old hover/exec_rate default 1.8 would put the Rate hover command (-0.037) above hover
    assert mod.ThrustMap(0.5, 1.8).pct((rate.hover_action()[3] + 1) / 2) > 80.0
    with pytest.raises(ValueError):
        mod.ThrustMap(0.5, 0.9)


def test_rate_mode_is_set_before_takeoff_and_restored(monkeypatch):
    mod = load_script(monkeypatch)
    cf = TrajFakeCf(3, pos=on_reference(mod))
    controller = run(make(mod, cf, command=commands(mod)["body_rate"]))
    names = [(n, a) for n, a in cf.sent]
    first_takeoff = names.index(("takeoff", (mod.TRAJ_ORIGIN[2], 3.0)))
    assert ("set_value", ("flightmode.stabModeRoll", "0")) in names[:first_takeoff]
    controller.stop()
    assert cf.calls_to("set_value")[-3:] == [(p, "None") for p in mod.RateCommand.STAB_MODE_PARAMS]  # fake reads None


@pytest.mark.parametrize("mode", ["attitude", "body_rate"])
def test_low_level_modes_level_before_the_hand_over(monkeypatch, mode):
    """The last setpoint before the notify is level at hover thrust, not the policy's command."""
    mod = load_script(monkeypatch)
    _, cf, _ = fly(mod, steps=5, command=commands(mod)[mode], pos=on_reference(mod))
    names = [name for name, _ in cf.sent]
    assert names[-4:] == ["send_setpoint_manual", "send_setpoint_manual", "send_notify_setpoint_stop", "land"]
    assert cf.sent[-3] == ("send_setpoint_manual", (0.0, 0.0, 0.0, mod.HOVER_THRUST_PCT, False))


def test_nothing_low_level_is_sent_before_the_policy(monkeypatch):
    """A flight ended during the takeoff hands over without a level setpoint: no motors on the ground."""
    mod = load_script(monkeypatch)
    monkeypatch.setattr(sys.modules["common.tracking"], "hover_for", lambda *a: False)
    cf = TrajFakeCf(10, pos=moving)
    run(make(mod, cf, command=commands(mod)["body_rate"]))
    assert cf.calls_to("send_setpoint_manual") == [] and cf.calls_to("land") == [(0.0, 2.0)]


def test_tilt_trips_the_low_level_modes_only(monkeypatch, caplog):
    """60° of roll: Rate leaves the policy for the fly-back; Vel's firmware caps the tilt itself."""
    mod = load_script(monkeypatch)
    q = (math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0, 0.0)  # 60° roll
    cf = TrajFakeCf(200, pos=on_reference(mod), quat_wxyz=q)  # room for the fly-back's hover polls
    controller = run(make(mod, cf, command=commands(mod)["body_rate"]))
    # tripped before the policy's first setpoint, so no level setpoint either
    assert cf.calls_to("send_setpoint_manual") == [] and controller.stop_reason.startswith("Tilt 60°")
    assert [n for n, _ in cf.sent[-5:]] == ["send_notify_setpoint_stop", "go_to", "go_to", "send_notify_setpoint_stop",
                                           "land"]
    cf = TrajFakeCf(20, pos=on_reference(mod), quat_wxyz=q)
    run(make(mod, cf))
    assert len(cf.calls_to("send_velocity_world_setpoint")) == 20


def test_tracking_error_trips_after_ten_steps(monkeypatch):
    """1.3 m off the reference: past TrackingEnv's 0.8 m reset, so Rate stops after 10 steps."""
    mod = load_script(monkeypatch)
    cf = TrajFakeCf(100, pos=lambda n: (0.2, -0.1, 1.0))
    controller = run(make(mod, cf, command=commands(mod)["body_rate"]))
    rate_cmds = [a for a in cf.calls_to("send_setpoint_manual") if a[4] is True]
    assert len(rate_cmds) == mod.TRACK_ERR_STEPS - 1 and controller.stop_reason.startswith("Tracking error")


def test_body_rate_trips_after_three_steps(monkeypatch):
    mod = load_script(monkeypatch)

    class Spinning(TrajFakeCf):
        def packet(self, name):
            packet = super().packet(name)
            if name == "imu":
                packet["gyro.z"] = 400.0
            return packet
    cf = Spinning(100, pos=on_reference(mod))
    controller = run(make(mod, cf, command=commands(mod)["attitude"]))
    assert controller.stop_reason.startswith("Body rate")
    assert len([a for a in cf.calls_to("send_setpoint_manual") if a[:3] != (0.0, 0.0, 0.0)]) == mod.BODY_RATE_STEPS - 1


# ── Recording ──────────────────────────────────────────────────────────────────

def test_records_one_row_per_step_against_the_scored_reference(tmp_path, monkeypatch):
    """One row per policy step at 100 Hz: what the policy saw, what was sent, and the point
    TrackingEnv scores the step against (look-ahead point 1)."""
    mod = load_script(monkeypatch)
    path = str(tmp_path / "flight.parquet")
    cf = TrajFakeCf(60, pos=moving)
    controller = run(make(mod, cf, record_path=path, metadata={"uri": "fake"}))
    controller.stop()
    rows = pq.read_table(path).to_pydict()
    rec = controller.recorder
    assert rows["step"] == list(range(60))
    assert rec.obs_fields == mod.OBS_FIELDS and len(rec.obs_fields) == FRAME
    seen = controller.agent.seen
    assert [list(r) for r in zip(*(rows[f] for f in rec.obs_fields))] == [pytest.approx(s) for s in seen]
    ref = controller.reference
    for step in (0, 30, 59):
        target = ref.scored(step).tolist()
        assert [rows[f"target_{c}"][step] for c in "xyz"] == pytest.approx(target)
        assert rows["track_err_m"][step] == pytest.approx(math.dist(moving(step), target), abs=1e-5)
        assert rows["traj_dist_m"][step] == pytest.approx(ref.speed * (step + 5) * 0.01)
    assert rows["vx"] == pytest.approx([ACTION[0] * MAX_VEL] * 60) and rows["tilt_deg"] == pytest.approx([0.0] * 60)
    assert rows["gyro_x"] == pytest.approx([IMU[0]] * 60) and rows["vbat"] == pytest.approx([VBAT] * 60)
    assert rows["motor_m4_pwm"] == [MOTOR[3]] * 60
    periods = [b - a for a, b in zip(rows["control_time"], rows["control_time"][1:])]
    assert 0.009 < sum(periods) / len(periods) < 0.011
    meta = pq.read_schema(path).metadata
    assert meta[b"trajectory"] == b"lemniscate" and meta[b"traj_clock"].startswith(b"linear")
    assert meta[b"control_mode"] == b"velocity" and meta[b"max_velocity_m_s"] == b"3.0"
    assert {b"uri", b"traj_speed_m_s", b"traj_lap_length_m", b"stabilizer.controller"} <= set(meta)


def test_rate_recording_names_its_command_and_thrust_map(tmp_path, monkeypatch):
    mod = load_script(monkeypatch)
    path = str(tmp_path / "flight.parquet")
    cf = TrajFakeCf(5, pos=on_reference(mod))
    controller = run(make(mod, cf, record_path=path, command=commands(mod)["body_rate"]))
    controller.stop()
    rows = pq.read_table(path).to_pydict()
    assert rows["roll_rate_deg_s"] == pytest.approx([ACTION[0] * 180.0] * 5)
    assert rows["action_thrust"] == pytest.approx([ACTION[3]] * 5)
    meta = pq.read_schema(path).metadata
    assert meta[b"control_mode"] == b"body_rate" and meta[b"hover_thrust_pct"] == b"77.0"
    assert float(meta[b"max_thrust_scale"]) == pytest.approx(1.5387, abs=1e-4)


def test_lands_after_duration(monkeypatch):
    mod = load_script(monkeypatch)
    cf = TrajFakeCf(1000, pos=moving)
    controller = run(make(mod, cf, duration=0.2))
    assert 15 <= len(cf.calls_to("send_velocity_world_setpoint")) <= 25 and not controller.running
    # the hand-over follows the last policy setpoint directly: no gap without a setpoint
    assert [name for name, _ in cf.sent[-3:]] == ["send_velocity_world_setpoint", "send_notify_setpoint_stop", "land"]


def test_missing_gyro_block_does_not_fly(monkeypatch, caplog):
    """The gyro feeds ang_vel_b: without it the policy would see NaN, so nothing takes off."""
    mod = load_script(monkeypatch)
    real_add = FakeCf._add_config

    def add_config(self, logconf):
        if logconf.name == "imu":
            raise KeyError("Variable gyro.x not in TOC")
        real_add(self, logconf)
    monkeypatch.setattr(FakeCf, "_add_config", add_config)
    cf = TrajFakeCf(10, pos=moving)
    controller = run(make(mod, cf))
    assert not controller.running and controller.loop_thread is None
    assert cf.calls_to("takeoff") == [] and any("not flying" in m for m in caplog.messages)


# ── Landing: hand over to the high-level commander, never cut the motors ───────

def test_end_of_flight_lands_once_without_cutting_the_motors(monkeypatch):
    """The loop's end and stop() both land; land() follows the notify, so the firmware obeys it."""
    mod = load_script(monkeypatch)
    cf = TrajFakeCf(10, pos=moving)
    controller = run(make(mod, cf))
    assert [name for name, _ in cf.sent[-3:]] == ["send_velocity_world_setpoint", "send_notify_setpoint_stop", "land"]
    controller.stop()  # Ctrl+C after the loop ended: no second land
    assert cf.calls_to("land") == [(0.0, 2.0)]
    assert cf.calls_to("send_stop_setpoint") == [], "a motor cut in flight drops the drone"


class NoisyFakeCf(TrajFakeCf):
    """Kalman position variance above POS_VARIANCE_THRESHOLD from control step `bad_from` on."""

    def __init__(self, steps, bad_from, **kwargs):
        super().__init__(steps, **kwargs)
        self.bad_from = bad_from

    def packet(self, name):
        if name == "quality" and self.calls >= self.bad_from:
            return {"kalman.varPX": 1.0, "kalman.varPY": 1e-3, "kalman.varPZ": 1e-3, "pm.vbat": VBAT}
        return super().packet(name)


def test_emergency_landing_lands_without_cutting_the_motors(monkeypatch, caplog):
    mod = load_script(monkeypatch)
    cf = NoisyFakeCf(50, bad_from=5, pos=moving)
    controller = run(make(mod, cf))
    assert any("EMERGENCY LANDING" in m for m in caplog.messages)
    assert len(cf.calls_to("send_velocity_world_setpoint")) < 50
    assert [name for name, _ in cf.sent[-2:]] == ["send_notify_setpoint_stop", "land"]
    controller.stop()
    assert cf.calls_to("send_notify_setpoint_stop") == [()]
    assert cf.calls_to("land") == [(0.0, 2.0)]
    assert cf.calls_to("send_stop_setpoint") == []


def test_stale_position_lands_after_timeout(monkeypatch, caplog):
    mod = load_script(monkeypatch)
    cf = TrajFakeCf(300, pos=moving, telemetry_steps=20)
    run(make(mod, cf))
    assert any(m.startswith("Position stale") for m in caplog.messages)
    # ~0.5 s (POS_STALE_TIMEOUT_S) of 10 ms steps after the last packet
    assert 20 + 40 <= len(cf.calls_to("send_velocity_world_setpoint")) <= 20 + 60
    assert cf.calls_to("land") == [(0.0, 2.0)]


def test_takeoff_height_flag(monkeypatch):
    """--takeoff-height sets the HLC takeoff only; the default stays the trajectory's z."""
    mod = load_script(monkeypatch)
    for height, expected in ((None, mod.TRAJ_ORIGIN[2]), (0.5, 0.5)):
        cf = TrajFakeCf(3, pos=moving)
        run(make(mod, cf, takeoff_height=height))
        assert [args[0] for args in cf.calls_to("takeoff")] == [pytest.approx(expected)]
        assert ("commander.enHighLevel", "1") in cf.calls_to("set_value") and cf.calls_to("send_arming_request")
    for bad in ("0.1", "2.5"):
        monkeypatch.setattr(sys, "argv", ["exec_track", "--checkpoint", "x.pt", "--control-mode", "velocity",
                                          "--takeoff-height", bad])
        with pytest.raises(SystemExit):
            mod.main()


@pytest.mark.parametrize("traj_type, xy", [("circle", 0.8), ("lemniscate", 1.5)])
def test_goes_to_the_trajectory_start_before_the_policy(monkeypatch, traj_type, xy):
    """As TrackingEnv's spawn: the policy starts at the reference at distance 0, not where it took off."""
    mod = load_script(monkeypatch)
    cf = TrajFakeCf(3, pos=lambda n: (0.0, 0.0, 0.5))
    reference = mod.Reference(traj_type, [xy, xy, 1.0], 1.0)
    run(make(mod, cf, takeoff_height=0.5, reference=reference))
    names = [name for name, _ in cf.sent]
    assert names.index("takeoff") < names.index("go_to") < names.index("send_velocity_world_setpoint")
    (go_to,) = cf.calls_to("go_to")
    start = [xy + mod.TRAJ_ORIGIN[0], mod.TRAJ_ORIGIN[1], mod.TRAJ_ORIGIN[2]]  # circle(0) = lemniscate(0) = (1, 0, 0)
    assert list(go_to[:4]) == pytest.approx(start + [0.0])
    duration = max(mod.GOTO_MIN_DURATION_S, math.dist(start, (0.0, 0.0, 0.5)) / mod.GOTO_SPEED)
    assert go_to[4] == pytest.approx(duration)
    assert mod.hlc_waits == pytest.approx([3.0 + 1.0, duration + 1.0])  # takeoff, then go_to, each + 1 s pause


def test_policy_is_not_started_when_the_wait_lands(monkeypatch):
    """A stale position or Ctrl-C during the takeoff ends the flight before the policy runs."""
    mod = load_script(monkeypatch)
    monkeypatch.setattr(sys.modules["common.tracking"], "hover_for", lambda *a: False)
    cf = TrajFakeCf(10, pos=moving)
    run(make(mod, cf))
    assert cf.calls_to("go_to") == [] and cf.calls_to("send_velocity_world_setpoint") == []
    assert cf.calls_to("land") == [(0.0, 2.0)]


# ── Geofence: brake, fly back to SAFE_HOVER_POS, hover, land ───────────────────

class DivingFakeCf(TrajFakeCf):
    """World velocity `vel_w` (m/s) in place of the fake's slow constant one."""

    def __init__(self, steps, vel_w, **kwargs):
        super().__init__(steps, **kwargs)
        self.vel_w = vel_w

    def packet(self, name):
        packet = super().packet(name)
        if name == "posvel":
            packet.update(zip(("stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"), self.vel_w))
        return packet


def _assert_flew_back(mod, cf, brake):
    names = [name for name, _ in cf.sent]
    assert names[-5:] == ["send_notify_setpoint_stop", "go_to", "go_to", "send_notify_setpoint_stop", "land"]
    _, brake_go_to, safe_go_to = cf.calls_to("go_to")  # the first one flies to the trajectory start
    fence = mod.ARENA_FENCE
    assert list(brake_go_to) == pytest.approx(brake + [0.0, fence.brake_s])
    assert list(safe_go_to[:4]) == pytest.approx(list(fence.safe_hover_pos) + [0.0])
    dist = math.dist(brake, fence.safe_hover_pos)
    assert safe_go_to[4] == pytest.approx(max(fence.min_fly_back_s, dist / fence.fly_back_speed))
    assert cf.calls_to("land") == [(0.0, 2.0)]
    assert cf.calls_to("send_stop_setpoint") == [], "a motor cut in flight drops the drone"


def test_ceiling_breach_flies_back_to_safe_hover(monkeypatch, caplog):
    mod = load_script(monkeypatch)
    cf = TrajFakeCf(500, pos=lambda n: (0.2, -0.1, 1.0 + 0.1 * n))  # above ARENA_FENCE.hi z = 1.8 from step 9
    controller = run(make(mod, cf))
    assert len(cf.calls_to("send_velocity_world_setpoint")) == 9
    assert any("outside the fence" in m for m in caplog.messages)
    _assert_flew_back(mod, cf, brake=[0.2, -0.1, mod.ARENA_FENCE.hi[2]])
    controller.stop()  # Ctrl+C after landing: no second land
    assert cf.calls_to("land") == [(0.0, 2.0)]


def test_fast_descent_trips_before_the_floor(monkeypatch):
    """At 1.5 m/s down, the 0.1 s lookahead trips 0.1 m above ARENA_FENCE.lo z, before the drone gets there."""
    mod = load_script(monkeypatch)
    z = mod.ARENA_FENCE.lo[2] + 0.1                       # predicted z in 0.1 s is 0.05 m below the floor
    cf = DivingFakeCf(500, vel_w=(0.0, 0.0, -1.5), pos=lambda n: (0.2, -0.1, z))
    run(make(mod, cf))
    assert cf.calls_to("send_velocity_world_setpoint") == []
    _assert_flew_back(mod, cf, brake=[0.2, -0.1, z])


# ── Reference ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("traj_type, xy", [("circle", 0.8), ("lemniscate", 1.5)])
def test_trajectory_lies_inside_the_fence(monkeypatch, traj_type, xy):
    """The scales the --traj-scale help names keep the flown points REF_FENCE_MARGIN inside the fence."""
    mod = load_script(monkeypatch)
    ref = mod.Reference(traj_type, [xy, xy, 1.0], 1.0)
    assert ref.limit_check() is None
    lap_steps = round(ref.length.item() / (ref.speed * mod.CONTROL_DT))
    pts = torch.cat([ref.waypoints(step) for step in range(0, lap_steps, 7)])
    fence, margin = mod.ARENA_FENCE, mod.REF_FENCE_MARGIN - 1e-6
    assert (pts >= torch.tensor(fence.lo) + margin).all() and (pts <= torch.tensor(fence.hi) - margin).all()
    assert fence.breach(torch.tensor(fence.safe_hover_pos), torch.zeros(3)) is None


def test_default_scale_is_refused_for_the_circle(monkeypatch):
    """One TRAJ_SCALE for every shape: a 1.5 m circle would leave the y fence, so no flight starts."""
    mod = load_script(monkeypatch)
    error = mod.Reference("circle", mod.TRAJ_SCALE, 1.0).limit_check()
    assert error and "circle at scale [1.5, 1.5, 1.0] spans [-1.5, -1.5, 1.0]..[1.5, 1.5, 1.0] m" in error
    with pytest.raises(ValueError, match="closer than 0.2 m to ARENA_FENCE"):
        make(mod, TrajFakeCf(1), reference=mod.Reference("circle", mod.TRAJ_SCALE, 1.0))


def test_reference_moves_at_constant_speed(monkeypatch):
    """The scored point covers speed * 0.01 m per step, the same at the lemniscate's tips and crossing."""
    mod = load_script(monkeypatch)
    ref = mod.Reference("lemniscate", [1.5, 1.5, 1.0], 0.8)
    lap_steps = ref.length.item() / (ref.speed * mod.CONTROL_DT)
    for s in [0, int(lap_steps / 4), int(lap_steps / 2)]:
        assert (ref.scored(s + 1) - ref.scored(s)).norm().item() == pytest.approx(0.8 * 0.01, rel=0.02)
        wps = ref.waypoints(s)
        spacing = (wps[1:] - wps[:-1]).norm(dim=-1)
        assert spacing.tolist() == pytest.approx([0.8 * 5 * 0.01] * 3, rel=0.02)
    # lap after lap: one lap later the reference is back where it was
    assert torch.allclose(ref.at_distance(torch.tensor([0.3 + ref.length.item()])),
                          ref.at_distance(torch.tensor([0.3])), atol=1e-4)
    assert ref.distance(10) == pytest.approx(0.8 * 15 * 0.01)


def _training_trajectories(monkeypatch):
    """The training module, importable without Isaac Lab once its quat_apply import is stubbed."""
    for name in ("isaaclab", "isaaclab.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", types.SimpleNamespace(quat_apply=None))
    spec = importlib.util.spec_from_file_location("training_trajectories", TRAINING_TRAJECTORIES)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    return training


def test_shapes_match_training(monkeypatch):
    """common/trajectories.py copies the training shapes, which cannot be imported without Isaac Lab."""
    training = _training_trajectories(monkeypatch)
    mod = load_script(monkeypatch)
    deployed = sys.modules["common.trajectories"]
    t = torch.linspace(0.0, 4.0 * math.pi, 101)
    for name, fn in deployed.TRAJECTORIES.items():
        assert torch.equal(fn(t), training.TRAJECTORIES[name](t)), name
    assert torch.equal(deployed.lemniscate(t, c=0.3), training.lemniscate(t, c=0.3))
    assert set(mod.TRAJECTORIES) == set(deployed.TRAJECTORIES)


def _needs_linear_speed_training(training):
    if not hasattr(training, "interp_waypoints"):
        pytest.skip("this branch's training trajectories.py predates #96 (constant linear speed); "
                    "the check runs once #96 is merged under it")


def test_arc_length_helpers_match_training(monkeypatch):
    training = _training_trajectories(monkeypatch)
    _needs_linear_speed_training(training)
    load_script(monkeypatch)
    deployed = sys.modules["common.trajectories"]
    dense = deployed.reference_w(torch.linspace(0.0, 2 * math.pi, 1025), "lemniscate", [1.5, 1.5, 1.0],
                                 [0.0, 0.0, 1.0], c=0.2)[None]
    for got, expected in zip(deployed.resample_by_arc_length(dense, 256), training.resample_by_arc_length(dense, 256)):
        assert torch.equal(got, expected)
    wps, length = training.resample_by_arc_length(dense, 256)
    distance = torch.linspace(0.0, 3.0 * length.item(), 50)[None]
    assert torch.equal(deployed.interp_waypoints(wps, length, distance), training.interp_waypoints(wps, length, distance))


@pytest.mark.parametrize("traj_type", ["lemniscate", "circle"])
def test_waypoints_match_tracking_env_compute_traj(monkeypatch, traj_type):
    """Reference.waypoints(step) is TrackingEnv._compute_traj for a run without randomize_trajectory's
    rotation and phase: the lap from phase 0, 256 segments, distance = speed * (step + 5k) * dt."""
    training = _training_trajectories(monkeypatch)
    _needs_linear_speed_training(training)
    mod = load_script(monkeypatch)
    scale, origin, speed, k = [1.5, 1.5, 1.0], [0.0, 0.0, 1.0], 0.75, 256
    t = torch.linspace(0.0, 2.0 * math.pi, 4 * k + 1)[None]                       # _build_traj_waypoints, t0 = 0
    raw = training.lemniscate(t, c=torch.zeros(1, 1)) if traj_type == "lemniscate" else training.circle(t)
    dense = raw * torch.tensor(scale) + torch.tensor(origin)                       # apply_traj_transform, no rotation
    wps, length = training.resample_by_arc_length(dense, k)
    ref = mod.Reference(traj_type, scale, speed)
    dt = 0.002 * 5                                                                 # cfg.sim.dt * cfg.decimation
    offsets = 5.0 * torch.arange(1, 5).float()
    for step in (0, 1, 137, 2000):                                                 # 2000: past a lap
        time_s = (torch.tensor([[float(step)]]) + offsets[None]) * dt
        expected = training.interp_waypoints(wps, length, torch.tensor([speed]).unsqueeze(1) * time_s)[0]
        assert torch.allclose(ref.waypoints(step), expected, atol=1e-5), step


# ── Command line and the run's env.yaml ────────────────────────────────────────

TRACK_ENV_YAML = """sim:
  physics_prim_path: /physicsScene
  dt: {dt}
  render_interval: 5
decimation: 5
viewer:
  eye: !!python/tuple
  - 7.5
trajectory_type: lemniscate
future_traj_steps: 4
obs_input_hist: 2
action_input_hist: 1
traj_step_size: 5.0
traj_origin:
- 0.0
- 0.0
- 1.5
randomize_trajectory: true
traj_speed_range: !!python/tuple
- 0.5
- 1.0
traj_scale_range: !!python/tuple
- 1.5
- 3.0
reset_thres: 0.8
drone_config_path: /workspace/crazyplayground/source/CrazyPlayGround/CrazyPlayGround/controllers/crazyflie.yaml
control_mode: {mode}
"""
MODE_YAML = {
    "position": "max_position_delta: 0.25\n",
    "velocity": "max_velocity: 1.0\n",
    "attitude": "max_roll_pitch: 0.4\nmax_yaw_rate: 1.0\nmin_thrust_scale: 0.5\nmax_thrust_scale: null\n",
    "body_rate": "max_body_rate: 2.0\nmin_thrust_scale: 0.5\nmax_thrust_scale: null\n",
}


def _env_yaml(mode, dt=0.002):
    return TRACK_ENV_YAML.format(dt=dt, mode=mode) + MODE_YAML[mode]


def test_read_env_cfg_and_contract(tmp_path, monkeypatch):
    mod = load_script(monkeypatch)
    utils = sys.modules["common.utils"]
    cfg = utils.read_env_cfg(_run_dir(tmp_path, "run", _env_yaml("velocity")))
    assert cfg["sim.dt"] == 0.002 and cfg["decimation"] == 5 and cfg["control_mode"] == "velocity"
    assert cfg["traj_speed_range"] == [0.5, 1.0] and cfg["traj_origin"] == [0.0, 0.0, 1.5]
    assert cfg["randomize_trajectory"] is True and cfg["traj_step_size"] == 5.0 and cfg["max_velocity"] == 1.0
    assert "viewer.eye" not in cfg and cfg["_path"].endswith("env.yaml")
    assert utils.read_env_cfg(_run_dir(tmp_path, "rate", _env_yaml("body_rate")))["max_thrust_scale"] is None
    utils.check_env_contract(cfg, "velocity", mod.CONTROL_DT)
    with pytest.raises(ValueError, match="control_mode is 'velocity'; this script flies 'body_rate'"):
        utils.check_env_contract(cfg, "body_rate", mod.CONTROL_DT)
    with pytest.raises(ValueError, match="traj_step_size"):
        utils.check_env_contract(dict(cfg, traj_step_size=10.0), "velocity", mod.CONTROL_DT)
    slow = utils.read_env_cfg(_run_dir(tmp_path, "slow", _env_yaml("velocity", dt=0.004)))
    with pytest.raises(ValueError, match="policy step"):
        utils.check_env_contract(slow, "velocity", mod.CONTROL_DT)
    assert utils.read_env_cfg(str(tmp_path / "loose.pt")) == {}


def _main_kwargs(mod, monkeypatch, argv, widths=None):
    made = []
    monkeypatch.setattr(mod, "load_agent", lambda path, dev, width, act: (widths or []).append((width, act)) or object())
    monkeypatch.setattr(mod, "CrazyflieController",
                        lambda **kw: made.append(kw) or (_ for _ in ()).throw(_StopBeforeFlight()))
    monkeypatch.setattr(sys, "argv", ["exec_track"] + argv)
    with pytest.raises(_StopBeforeFlight):
        mod.main()
    return made[0]


@pytest.mark.parametrize("mode", ["position", "velocity", "attitude", "body_rate"])
def test_cli_builds_the_command_of_the_run(tmp_path, monkeypatch, mode):
    """The run's control_mode picks the Command and the policy width; its scales size the action."""
    mod = load_script(monkeypatch)
    ckpt = _run_dir(tmp_path, "run", _env_yaml(mode))
    widths = []
    kw = _main_kwargs(mod, monkeypatch, ["--checkpoint", ckpt], widths)
    command = kw["command"]
    assert command.mode == mode and widths == [(FRAME * 2 + command.act_dim, command.act_dim)]
    if mode == "position":
        assert command.max_position_delta == 0.25
    elif mode == "velocity":
        assert command.max_velocity == 1.0
    elif mode == "attitude":
        assert command.max_roll_pitch_deg == pytest.approx(math.degrees(0.4))
        assert command.max_yaw_rate_deg_s == pytest.approx(math.degrees(1.0))
    else:
        assert command.max_body_rate_deg_s == pytest.approx(math.degrees(2.0))
    if command.low_level:  # null max_thrust_scale, drone yaml not on this machine: the CF2.1 cap
        assert command.thrust.max_scale == pytest.approx(1.5387, abs=1e-4) and command.thrust.hover_pct == 77.0
    assert kw["reference"].speed == pytest.approx(0.75)  # middle of (0.5, 1.0)
    assert kw["metadata"]["env_yaml"].endswith("env.yaml") and kw["metadata"]["env_yaml_sha256"]


def test_cli_control_mode_flag(tmp_path, monkeypatch, caplog):
    mod = load_script(monkeypatch)
    rate = _run_dir(tmp_path, "rate", _env_yaml("body_rate"))
    loose = str(tmp_path / "loose.pt")
    # no env.yaml: the flag decides, with the TrackingEnvCfg defaults and a warning
    kw = _main_kwargs(mod, monkeypatch, ["--checkpoint", loose, "--control-mode", "velocity"])
    assert kw["command"].max_velocity == MAX_VEL and kw["reference"].speed == 1.0
    assert any("No max_velocity" in m for m in caplog.messages)
    kw = _main_kwargs(mod, monkeypatch, ["--checkpoint", rate, "--hover-thrust-pct", "70"])
    assert kw["command"].thrust.hover_pct == 70.0
    for argv in (["--checkpoint", loose],                                    # no env.yaml and no flag
                 ["--checkpoint", rate, "--control-mode", "velocity"],       # flag disagrees with env.yaml
                 ["--checkpoint", rate, "--hover-thrust-pct", "20"],         # below --min-thrust-pct
                 ["--checkpoint", rate, "--traj-speed", "0"]):
        monkeypatch.setattr(sys, "argv", ["exec_track"] + argv)
        with pytest.raises(SystemExit):
            mod.main()


def test_cli_sizes_policy_from_env_yaml(tmp_path, monkeypatch):
    """K and M come from the run's env.yaml; a flag that disagrees stops before the radio."""
    mod = load_script(monkeypatch)
    vel = ["--control-mode", "velocity"]
    ckpt = _run_dir(tmp_path, "run", ENV_YAML.format(k=2, m=1))
    widths = []
    made = _main_kwargs(mod, monkeypatch, ["--checkpoint", ckpt] + vel, widths)
    assert widths == [(FRAME * 2 + 3 * 1, 3)] and (made["obs_input_hist"], made["action_input_hist"]) == (2, 1)

    # a checkpoint outside its run: the flags decide
    made = _main_kwargs(mod, monkeypatch, ["--checkpoint", str(tmp_path / "loose.pt"), "--obs-input-hist", "4"] + vel,
                        widths)
    assert widths[-1] == (FRAME * 4, 3) and (made["obs_input_hist"], made["action_input_hist"]) == (4, 0)

    # an exported checkpoint: K and M from <name>_params/env.yaml beside it, not the K=1, M=0 fallback
    exported = _export_dir(tmp_path, "best_k3m2", ENV_YAML.format(k=3, m=2))
    made = _main_kwargs(mod, monkeypatch, ["--checkpoint", exported] + vel, widths)
    assert widths[-1] == (FRAME * 3 + 3 * 2, 3) and (made["obs_input_hist"], made["action_input_hist"]) == (3, 2)

    for bad in (["--obs-input-hist", "3"], ["--obs-input-hist", "0"], ["--action-input-hist", "-1"]):
        monkeypatch.setattr(sys, "argv", ["exec_track", "--checkpoint", ckpt] + vel + bad)
        with pytest.raises(SystemExit):
            mod.main()
    assert len(widths) == 3


def test_traj_scale_flag(tmp_path, monkeypatch, capsys):
    """--traj-scale sets the XY scale of any shape; one the fence cannot hold stops before the radio."""
    mod = load_script(monkeypatch)
    ckpt = _run_dir(tmp_path, "run", ENV_YAML.format(k=1, m=0))
    vel = ["--control-mode", "velocity"]
    for argv, expected in (([], ("lemniscate", mod.TRAJ_SCALE)),
                           (["--trajectory", "circle", "--traj-scale", "0.8"], ("circle", [0.8, 0.8, 1.0]))):
        ref = _main_kwargs(mod, monkeypatch, ["--checkpoint", ckpt] + vel + argv)["reference"]
        assert (ref.traj_type, ref.scale) == expected
    for bad in (["--trajectory", "circle"], ["--traj-scale", "2.0"], ["--traj-scale", "0"]):
        monkeypatch.setattr(sys, "argv", ["exec_track", "--checkpoint", ckpt] + vel + bad)
        with pytest.raises(SystemExit):
            mod.main()
    assert "closer than 0.2 m to ARENA_FENCE" in capsys.readouterr().err
