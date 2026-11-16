"""Tests of the single-drone trajectory-tracking velocity script against a fake Crazyflie.

They check that `traj_tracking/exec_vel.py` builds the policy input as
`TrackingEnv._get_observations` does: the last K 22-D frames, oldest first, primed with K
copies of the first one, then the last M commanded (clamped) actions, zeros before the first.
The fake radio is the one of `test_hover_scripts.py`; only the log block names differ.
"""

import importlib.util
import math
import os
import sys
import time
import types

import pytest
import torch
from cflib.crazyflie.log import LogConfig

from test_hover_scripts import (ACTION, ENV_YAML, FakeAgent, FakeCf, _Commands, _export_dir, _run_dir,
                                _save_checkpoint, _StopBeforeFlight)

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "..", "execution", "single_drone_exec", "traj_tracking", "exec_vel.py")
FRAME = 22


def load_script(monkeypatch):
    """Import the script under its own name: hover/exec_vel.py already owns `exec_vel`."""
    spec = importlib.util.spec_from_file_location("traj_exec_vel", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fast_time = types.SimpleNamespace(time=time.time, perf_counter=time.perf_counter, sleep=lambda s: None)
    monkeypatch.setattr(mod, "time", fast_time)
    monkeypatch.setattr(mod, "device", torch.device("cpu"))
    monkeypatch.setattr(LogConfig, "start", lambda logconf: logconf.fake_cf.fire(logconf))
    return mod


class TrajFakeCf(FakeCf):
    """FakeCf with the log blocks of the tracking script ("pos" and "vel" apart) and a platform."""

    def __init__(self, steps, **kwargs):
        super().__init__(steps, **kwargs)
        self.platform = _Commands(self)

    def packet(self, name):
        if name in ("pos", "vel"):
            posvel = super().packet("posvel")
            keys = ("x", "y", "z") if name == "pos" else ("vx", "vy", "vz")
            return {f"stateEstimate.{k}": posvel[f"stateEstimate.{k}"] for k in keys}
        return super().packet(name)


def moving(n):
    """Position at control step n: every frame differs, so the window order is visible."""
    return (0.2, -0.1, 1.0 + 0.01 * n)


def fly(mod, steps, k=1, m=0, action=None, pos=moving):
    """Run the control loop for `steps` steps. Returns (cf, agent)."""
    cf = TrajFakeCf(steps, pos=pos)
    mod.Crazyflie = lambda **kwargs: cf
    agent = FakeAgent(3)
    if action is not None:
        agent.action = torch.tensor([action])
    controller = mod.CrazyflieController(uri="fake", agent=agent, obs_input_hist=k, action_input_hist=m)
    controller._start_logging()
    controller.control_loop()  # returns once is_connected() goes False
    assert len(agent.seen) == steps
    return cf, agent


def test_default_input_is_one_frame(monkeypatch):
    """K=1, M=0 (a checkpoint trained without history): the input is the plain 22-D frame."""
    mod = load_script(monkeypatch)
    cf, agent = fly(mod, steps=10)
    assert all(len(seen) == FRAME for seen in agent.seen)
    sent = cf.calls_to("send_velocity_world_setpoint")
    assert [list(args) for args in sent] == [pytest.approx([a * mod.MAX_VELOCITY for a in ACTION[:3]] + [0.0])] * 10


def test_window_and_action_history(monkeypatch):
    """K=3, M=2: frames oldest first, primed with copies of the first, then the last two commands."""
    mod = load_script(monkeypatch)
    _, plain = fly(mod, steps=12)
    frames = plain.seen
    assert frames[0] != frames[1]
    _, agent = fly(mod, steps=12, k=3, m=2)
    zeros, a = [0.0] * 3, ACTION[:3]
    for i, seen in enumerate(agent.seen):
        window = [frames[max(j, 0)] for j in (i - 2, i - 1, i)]
        hist = [zeros + zeros, zeros + a][i] if i < 2 else a + a
        assert len(seen) == FRAME * 3 + 3 * 2
        assert seen == pytest.approx([v for frame in window for v in frame] + hist)


def test_history_holds_clamped_actions(monkeypatch):
    """The history holds the command actually sent, clamped to [-1, 1], as cmd_history does."""
    mod = load_script(monkeypatch)
    cf, agent = fly(mod, steps=5, m=1, action=[1.5, -2.0, 0.3])
    assert agent.seen[0][-3:] == [0.0, 0.0, 0.0]
    assert all(seen[-3:] == pytest.approx([1.0, -1.0, 0.3]) for seen in agent.seen[1:])
    sent = cf.calls_to("send_velocity_world_setpoint")[-1]
    assert list(sent[:3]) == pytest.approx([mod.MAX_VELOCITY, -mod.MAX_VELOCITY, 0.3 * mod.MAX_VELOCITY])


def test_load_agent_width(tmp_path, monkeypatch):
    mod = load_script(monkeypatch)
    path = tmp_path / "best_agent.pt"
    width = FRAME * 2 + 3 * 1
    saved = _save_checkpoint(mod, path, obs_space=width)
    loaded = mod.load_agent(str(path), torch.device("cpu"), width)
    obs = torch.randn(4, width)
    with torch.no_grad():
        act = [ag.act(obs, None, timestep=0, timesteps=1)[1]["mean_actions"] for ag in (loaded, saved)]
    assert torch.equal(*act)
    for wrong in (FRAME, FRAME * 2, FRAME * 2 + 3 * 2):
        with pytest.raises(RuntimeError, match="size mismatch"):
            mod.load_agent(str(path), torch.device("cpu"), wrong)


def test_cli_sizes_policy_from_env_yaml(tmp_path, monkeypatch):
    """K and M come from the run's env.yaml; a flag that disagrees stops before the radio."""
    mod = load_script(monkeypatch)
    ckpt = _run_dir(tmp_path, "run", ENV_YAML.format(k=2, m=1))
    widths, made = [], []
    monkeypatch.setattr(mod, "load_agent", lambda path, dev, width: widths.append(width) or object())
    monkeypatch.setattr(mod, "CrazyflieController",
                        lambda **kw: made.append(kw) or (_ for _ in ()).throw(_StopBeforeFlight()))

    monkeypatch.setattr(sys, "argv", ["exec_vel", "--checkpoint", ckpt])
    with pytest.raises(_StopBeforeFlight):
        mod.main()
    assert widths == [FRAME * 2 + 3 * 1]
    assert (made[0]["obs_input_hist"], made[0]["action_input_hist"]) == (2, 1)

    # a checkpoint outside its run: the flags decide
    loose = str(tmp_path / "loose.pt")
    monkeypatch.setattr(sys, "argv", ["exec_vel", "--checkpoint", loose, "--obs-input-hist", "4"])
    with pytest.raises(_StopBeforeFlight):
        mod.main()
    assert widths[-1] == FRAME * 4 and (made[-1]["obs_input_hist"], made[-1]["action_input_hist"]) == (4, 0)

    # an exported checkpoint: K and M from <name>_params/env.yaml beside it, not the K=1, M=0 fallback
    exported = _export_dir(tmp_path, "best_k3m2", ENV_YAML.format(k=3, m=2))
    monkeypatch.setattr(sys, "argv", ["exec_vel", "--checkpoint", exported])
    with pytest.raises(_StopBeforeFlight):
        mod.main()
    assert widths[-1] == FRAME * 3 + 3 * 2 and (made[-1]["obs_input_hist"], made[-1]["action_input_hist"]) == (3, 2)

    for bad in (["--obs-input-hist", "3"], ["--obs-input-hist", "0"], ["--action-input-hist", "-1"]):
        monkeypatch.setattr(sys, "argv", ["exec_vel", "--checkpoint", ckpt] + bad)
        with pytest.raises(SystemExit):
            mod.main()
    assert len(widths) == 3


# ── Landing: hand over to the high-level commander, never cut the motors ───────

def test_end_of_flight_lands_once_without_cutting_the_motors(monkeypatch):
    """The loop's end and stop() both land; land() follows the notify, so the firmware obeys it."""
    mod = load_script(monkeypatch)
    cf = TrajFakeCf(10, pos=moving)
    mod.Crazyflie = lambda **kwargs: cf
    controller = mod.CrazyflieController(uri="fake", agent=FakeAgent(3))
    controller._start_logging()
    controller.control_loop()
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
            return {"kalman.varPX": 1.0, "kalman.varPY": 1e-3, "kalman.varPZ": 1e-3}
        return super().packet(name)


def test_emergency_landing_lands_without_cutting_the_motors(monkeypatch, caplog):
    mod = load_script(monkeypatch)
    cf = NoisyFakeCf(50, bad_from=5, pos=moving)
    mod.Crazyflie = lambda **kwargs: cf
    controller = mod.CrazyflieController(uri="fake", agent=FakeAgent(3))
    controller._start_logging()
    controller.control_loop()
    assert any("EMERGENCY LANDING" in m for m in caplog.messages)
    assert len(cf.calls_to("send_velocity_world_setpoint")) < 50
    assert [name for name, _ in cf.sent[-2:]] == ["send_notify_setpoint_stop", "land"]
    controller.stop()
    assert cf.calls_to("send_notify_setpoint_stop") == [()]
    assert cf.calls_to("land") == [(0.0, 2.0)]
    assert cf.calls_to("send_stop_setpoint") == []


def test_takeoff_height_flag(monkeypatch):
    """--takeoff-height sets the HLC takeoff only; the default stays the trajectory's z."""
    mod = load_script(monkeypatch)
    for height, expected in ((None, mod.TRAJ_ORIGIN[2]), (0.5, 0.5)):
        cf = TrajFakeCf(3, pos=moving)
        mod.Crazyflie = lambda **kwargs: cf
        controller = mod.CrazyflieController(uri="fake", agent=FakeAgent(3), takeoff_height=height)
        controller._start_logging()
        controller.control_loop()
        assert [args[0] for args in cf.calls_to("takeoff")] == [pytest.approx(expected)]
    for bad in ("0.1", "2.5"):
        monkeypatch.setattr(sys, "argv", ["exec_vel", "--checkpoint", "x.pt", "--takeoff-height", bad])
        with pytest.raises(SystemExit):
            mod.main()


def test_goes_to_the_trajectory_start_before_the_policy(monkeypatch):
    """As TrackingEnv's spawn: the policy starts at the reference of step 0, not where it took off."""
    mod = load_script(monkeypatch)
    cf = TrajFakeCf(3, pos=lambda n: (0.0, 0.0, 0.5))
    mod.Crazyflie = lambda **kwargs: cf
    controller = mod.CrazyflieController(uri="fake", agent=FakeAgent(3), takeoff_height=0.5)
    controller._start_logging()
    controller.control_loop()
    names = [name for name, _ in cf.sent]
    assert names.index("takeoff") < names.index("go_to") < names.index("send_velocity_world_setpoint")
    (go_to,) = cf.calls_to("go_to")
    start = [mod.TRAJ_SCALE[0] + mod.TRAJ_ORIGIN[0], mod.TRAJ_ORIGIN[1], mod.TRAJ_ORIGIN[2]]  # lemniscate(0) = (1, 0, 0)
    assert list(go_to[:4]) == pytest.approx(start + [0.0])
    dist = math.dist(start, (0.0, 0.0, 0.5))
    assert go_to[4] == pytest.approx(max(mod.GOTO_MIN_DURATION_S, dist / mod.GOTO_SPEED))

