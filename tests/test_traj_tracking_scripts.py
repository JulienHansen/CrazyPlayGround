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
    monkeypatch.setattr(sys.modules["common.utils"], "time", fast_time)  # the fly-back's hover waits
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


@pytest.mark.parametrize("traj_type, xy", [("circle", 0.8), ("lemniscate", 1.5)])
def test_goes_to_the_trajectory_start_before_the_policy(monkeypatch, traj_type, xy):
    """As TrackingEnv's spawn: the policy starts at the reference of step 0, not where it took off."""
    mod = load_script(monkeypatch)
    cf = TrajFakeCf(3, pos=lambda n: (0.0, 0.0, 0.5))
    mod.Crazyflie = lambda **kwargs: cf
    scale = [xy, xy, mod.TRAJ_SCALE[2]]
    controller = mod.CrazyflieController(uri="fake", agent=FakeAgent(3), takeoff_height=0.5,
                                         traj_type=traj_type, traj_scale=scale)
    controller._start_logging()
    controller.control_loop()
    names = [name for name, _ in cf.sent]
    assert names.index("takeoff") < names.index("go_to") < names.index("send_velocity_world_setpoint")
    (go_to,) = cf.calls_to("go_to")
    start = [scale[0] + mod.TRAJ_ORIGIN[0], mod.TRAJ_ORIGIN[1], mod.TRAJ_ORIGIN[2]]  # circle(0) = lemniscate(0) = (1, 0, 0)
    assert list(go_to[:4]) == pytest.approx(start + [0.0])
    dist = math.dist(start, (0.0, 0.0, 0.5))
    assert go_to[4] == pytest.approx(max(mod.GOTO_MIN_DURATION_S, dist / mod.GOTO_SPEED))


# ── Geofence: brake, fly back to SAFE_HOVER_POS, hover, land ───────────────────

class DivingFakeCf(TrajFakeCf):
    """World velocity `vel_w` (m/s) in place of the fake's slow constant one."""

    def __init__(self, steps, vel_w, **kwargs):
        super().__init__(steps, **kwargs)
        self.vel_w = vel_w

    def packet(self, name):
        if name == "vel":
            return dict(zip(("stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"), self.vel_w))
        return super().packet(name)


def _fly_to_breach(mod, cf):
    mod.Crazyflie = lambda **kwargs: cf
    controller = mod.CrazyflieController(uri="fake", agent=FakeAgent(3))
    controller._start_logging()
    controller.control_loop()
    return controller


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
    controller = _fly_to_breach(mod, cf)
    assert len(cf.calls_to("send_velocity_world_setpoint")) == 9
    assert any("outside the fence" in m for m in caplog.messages)
    _assert_flew_back(mod, cf, brake=[0.2, -0.1, mod.ARENA_FENCE.hi[2]])
    controller.stop()  # Ctrl+C after landing: no second land
    assert cf.calls_to("land") == [(0.0, 2.0)]


def test_fast_descent_trips_before_the_floor(monkeypatch):
    """At 1.5 m/s down, the 0.1 s lookahead trips 0.1 m above ARENA_FENCE.lo z, before the drone gets there."""
    mod = load_script(monkeypatch)
    z = mod.ARENA_FENCE.lo[2] + 0.1                       # 0.3 m: predicted z in 0.1 s = 0.15 m, below the floor
    cf = DivingFakeCf(500, vel_w=(0.0, 0.0, -1.5), pos=lambda n: (0.2, -0.1, z))
    _fly_to_breach(mod, cf)
    assert cf.calls_to("send_velocity_world_setpoint") == []
    _assert_flew_back(mod, cf, brake=[0.2, -0.1, z])


@pytest.mark.parametrize("traj_type, xy", [("circle", 0.8), ("lemniscate", 1.5)])
def test_trajectory_lies_inside_the_fence(monkeypatch, traj_type, xy):
    """The scales the --traj-scale help names keep the flown waypoints REF_FENCE_MARGIN inside the fence."""
    mod = load_script(monkeypatch)
    scale = [xy, xy, mod.TRAJ_SCALE[2]]
    assert mod.trajectory_limit_checks(traj_type, scale) is None
    period = round(2 * math.pi / (mod.TRAJ_W * mod.CONTROL_DT))
    pts = torch.cat([mod.compute_waypoints_world(step, traj_type, scale) for step in range(0, period, 10)])
    fence, margin = mod.ARENA_FENCE, mod.REF_FENCE_MARGIN - 1e-6
    assert (pts >= torch.tensor(fence.lo) + margin).all() and (pts <= torch.tensor(fence.hi) - margin).all()
    assert fence.breach(torch.tensor(fence.safe_hover_pos), torch.zeros(3)) is None


def test_default_scale_is_refused_for_the_circle(monkeypatch):
    """One TRAJ_SCALE for every shape: a 1.5 m circle would leave the y fence, so no flight starts."""
    mod = load_script(monkeypatch)
    error = mod.trajectory_limit_checks("circle", mod.TRAJ_SCALE)
    assert error and "circle at scale [1.5, 1.5, 1.0] spans [-1.5, -1.5, 1.0]..[1.5, 1.5, 1.0] m" in error
    with pytest.raises(ValueError, match="closer than 0.2 m to ARENA_FENCE"):
        mod.CrazyflieController(uri="fake", agent=FakeAgent(3), traj_type="circle")


def test_traj_scale_flag(tmp_path, monkeypatch, capsys):
    """--traj-scale sets the XY scale of any shape; one the fence cannot hold stops before the radio."""
    mod = load_script(monkeypatch)
    ckpt = _run_dir(tmp_path, "run", ENV_YAML.format(k=1, m=0))
    made = []
    monkeypatch.setattr(mod, "load_agent", lambda path, dev, width: object())
    monkeypatch.setattr(mod, "CrazyflieController",
                        lambda **kw: made.append(kw) or (_ for _ in ()).throw(_StopBeforeFlight()))
    for argv, expected in (([], ("lemniscate", mod.TRAJ_SCALE)),
                           (["--trajectory", "circle", "--traj-scale", "0.8"], ("circle", [0.8, 0.8, 1.0]))):
        monkeypatch.setattr(sys, "argv", ["exec_vel", "--checkpoint", ckpt] + argv)
        with pytest.raises(_StopBeforeFlight):
            mod.main()
        assert (made[-1]["traj_type"], made[-1]["traj_scale"]) == expected
    for bad in (["--trajectory", "circle"], ["--traj-scale", "2.0"], ["--traj-scale", "0"]):
        monkeypatch.setattr(sys, "argv", ["exec_vel", "--checkpoint", ckpt] + bad)
        with pytest.raises(SystemExit):
            mod.main()
    assert len(made) == 2
    assert "closer than 0.2 m to ARENA_FENCE" in capsys.readouterr().err


def test_shapes_match_training(monkeypatch):
    """common/trajectories.py copies the training shapes, which cannot be imported without Isaac Lab."""
    for name in ("isaaclab", "isaaclab.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", types.SimpleNamespace(quat_apply=None))
    path = os.path.join(HERE, "..", "source", "CrazyPlayGround", "CrazyPlayGround", "tasks", "direct", "track",
                        "trajectories.py")
    spec = importlib.util.spec_from_file_location("training_trajectories", path)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    mod = load_script(monkeypatch)
    deployed = sys.modules["common.trajectories"]
    t = torch.linspace(0.0, 4.0 * math.pi, 101)
    for name, fn in deployed.TRAJECTORIES.items():
        assert torch.equal(fn(t), training.TRAJECTORIES[name](t)), name
    assert torch.equal(deployed.lemniscate(t, c=0.3), training.lemniscate(t, c=0.3))
    assert set(mod.TRAJECTORIES) == set(deployed.TRAJECTORIES)
