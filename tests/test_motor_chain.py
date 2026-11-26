"""Firmware motor chain (mixer, cap, thrust map, motor lag) and its use in CascadePIDController.

Expected values come from the Crazyflie firmware constants (CF2.1 brushed, 2.1+ props):
THRUST_MAX 0.12 N per motor at command 65535, ARM_LENGTH 0.046 m, THRUST2TORQUE 0.00699 m;
and from Crazyflow cf2x_P250: 31.8 g, hover at 23 192 rpm = 2429 rad/s per motor.
"""

import math
import os
import sys

import pytest
import torch

PKG = os.path.join(os.path.dirname(__file__), "..", "source", "CrazyPlayGround", "CrazyPlayGround")
sys.path.append(PKG)
from controllers import CascadePIDController, load_config  # noqa: E402
from controllers.config import resolve_max_thrust_scale, rpm_to_thrust  # noqa: E402
from controllers.motor_chain import CrazyflieMotorChain, thrust_to_omega  # noqa: E402

DT = 1.0 / 500.0
K = 0.12 / 65535.0                 # N per command unit, per motor
D = 0.046 * math.sqrt(0.5)         # lever arm about x and y [m]
YAML = os.path.join(PKG, "controllers", "crazyflie.yaml")


def chain(n=1, tau=0.0, tau_down=None):
    """Motor chain with the parameters of crazyflie.yaml; the expected values above stay
    hardcoded on purpose, as an independent check of those parameters. ``tau_down`` defaults
    to ``tau`` (symmetric lag)."""
    phys = load_config(YAML).physics
    return CrazyflieMotorChain(
        num_envs=n,
        dt=DT,
        device="cpu",
        max_thrust_total=phys.max_thrust,
        thrust_min=phys.motor.thrust_min,
        arm_length=phys.motor.arm_length,
        thrust2torque=phys.motor.thrust2torque,
        tau_up=tau,
        tau_down=tau if tau_down is None else tau_down,
        motor_cmd_max=phys.firmware.motor_cmd_max,
        control_cmd_max=phys.firmware.control_cmd_max,
    )


def wrench(c, thrust_cmd, alpha):
    return c(torch.full((alpha.shape[0], 1), float(thrust_cmd)), alpha)


def test_hover_command_gives_firmware_thrust_and_no_torque():
    thrust, torque = wrench(chain(), 36000, torch.zeros(1, 3))
    assert thrust.item() == pytest.approx(4 * 36000 * K, rel=1e-6)  # 0.2637 N
    assert thrust.item() == pytest.approx(0.2637, abs=1e-4)
    assert torch.all(torque == 0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_each_axis_command_gives_only_its_own_torque_with_positive_sign(axis):
    # +x roll = left side up, +y pitch = nose down, +z yaw = CCW from above (FLU, right-hand rule)
    alpha = torch.zeros(1, 3)
    alpha[0, axis] = 10.0
    c = chain()
    _, torque = wrench(c, 36000, alpha)
    assert torque[0, axis] > 0
    others = [i for i in range(3) if i != axis]
    assert torch.allclose(torque[0, others], torch.zeros(2), atol=1e-9)
    f = c.motor_forces[0]  # M1 front-right, M2 rear-right, M3 rear-left, M4 front-left
    if axis == 0:
        assert f[2] > f[1] and f[3] > f[0]          # left pair up
    elif axis == 1:
        assert f[1] > f[0] and f[2] > f[3]          # rear pair up
    else:
        assert f[1] > f[0] and f[3] > f[2]          # CW pair (M2, M4) up


def test_roll_torque_per_rad_s_matches_firmware():
    # P term only: Kp 250 on a 1 rad/s error -> 14 324 units -> r = 7 162 per motor
    _, torque = wrench(chain(), 36000, torch.tensor([[250.0, 0.0, 0.0]]))
    assert torque[0, 0].item() == pytest.approx(1.706e-3, rel=2e-3)


def test_int16_saturation():
    c = chain()
    big = c.motor_commands(torch.tensor([[36000.0]]), torch.tensor([[1e4, 0.0, 0.0]]))
    sat = c.motor_commands(torch.tensor([[36000.0]]), torch.tensor([[math.radians(40000.0), 0.0, 0.0]]))
    assert torch.equal(big, sat)
    assert (big[0, 2] - big[0, 1]).item() == 2 * 16383  # 2 r with r = trunc(32767 / 2)


def test_cap_keeps_differences_and_clips_negatives():
    c = chain()
    alpha = torch.tensor([[math.radians(4000.0), 0.0, 0.0]])  # u = 4000 -> r = 2000
    m = c.motor_commands(torch.tensor([[65000.0]]), alpha)
    assert m.max().item() == 65535.0
    assert (m[0, 2] - m[0, 1]).item() == pytest.approx(4000.0, abs=4)  # float -> trunc may drop a unit
    m = c.motor_commands(torch.tensor([[1000.0]]), alpha)
    assert m[0, 0].item() == 0.0 and m[0, 1].item() == 0.0
    assert m[0, 2].item() == pytest.approx(3000.0, abs=2)


def test_motor_below_thrust_min_is_off():
    thrust, _ = wrench(chain(), 5000, torch.zeros(1, 3))  # 5000 * K = 9.2 mN < 12.8 mN
    assert thrust.item() == 0.0
    thrust, _ = wrench(chain(), 8000, torch.zeros(1, 3))
    assert thrust.item() == pytest.approx(4 * 8000 * K, rel=1e-6)


def test_lag_reaches_63_percent_after_one_time_constant():
    tau = 0.072
    c = chain(tau=tau)
    c.reset(None, 0.0)
    steps = round(tau / DT)  # 36
    for _ in range(steps):
        thrust, _ = wrench(c, 36000, torch.zeros(1, 3))
    target = 4 * 36000 * K
    assert thrust.item() / target == pytest.approx(1 - math.exp(-steps * DT / tau), rel=1e-4)
    assert thrust.item() / target == pytest.approx(0.632, abs=2e-3)


def test_tau_per_env_and_zero_tau():
    c = chain(n=3, tau=0.072)
    taus = torch.tensor([0.0, 0.03, 0.09])
    c.set_tau(taus, taus, torch.tensor([0, 1, 2]))
    c.reset(None, 0.0)
    thrust, _ = wrench(c, 36000, torch.zeros(3, 3))
    target = 4 * 36000 * K
    assert thrust[0].item() == pytest.approx(target, rel=1e-6)                      # no lag
    assert thrust[1].item() == pytest.approx(target * (1 - math.exp(-DT / 0.03)), rel=1e-4)
    assert thrust[2].item() == pytest.approx(target * (1 - math.exp(-DT / 0.09)), rel=1e-4)


def test_lag_uses_tau_up_on_rise_and_tau_down_on_fall():
    tau_up, tau_down = 0.1117, 0.1029
    c = chain(tau=tau_up, tau_down=tau_down)
    target = 4 * 36000 * K
    # spin-up from 0: 63 % of the step after tau_up
    c.reset(None, 0.0)
    steps = round(tau_up / DT)
    for _ in range(steps):
        thrust, _ = wrench(c, 36000, torch.zeros(1, 3))
    assert thrust.item() / target == pytest.approx(1 - math.exp(-steps * DT / tau_up), rel=1e-4)
    # spin-down from the target to 0 (command below THRUST_MIN): 63 % of the drop after tau_down
    c.reset(None, target / 4)
    steps = round(tau_down / DT)
    for _ in range(steps):
        thrust, _ = wrench(c, 0, torch.zeros(1, 3))
    assert 1 - thrust.item() / target == pytest.approx(1 - math.exp(-steps * DT / tau_down), rel=1e-4)


def test_thrust_curve_inverse_at_hover_and_zero():
    motor = load_config(YAML).physics.motor
    hover_force = 0.0318 * 9.81 / 4  # 0.07799 N per motor
    omega = thrust_to_omega(torch.tensor([hover_force, 0.0, -0.01]), motor.rpm2thrust)
    assert omega[0].item() == pytest.approx(2429.0, rel=1e-3)  # 23 192 rpm (Crazyflow cf2x_P250)
    assert omega[1].item() == 0.0 and omega[2].item() == 0.0   # motor off, not the 2254 rpm root
    # round trip through the curve
    rpm = omega[0].item() * 60.0 / (2.0 * math.pi)
    assert rpm_to_thrust(rpm, motor.rpm2thrust) == pytest.approx(hover_force, rel=1e-5)


def test_speed_max_gives_thrust_max():
    motor = load_config(YAML).physics.motor
    rpm = motor.speed_max * 60.0 / (2.0 * math.pi)
    assert rpm_to_thrust(rpm, motor.rpm2thrust) == pytest.approx(0.12, rel=1e-3)
    assert thrust_to_omega(torch.tensor([0.12]), motor.rpm2thrust).item() == pytest.approx(motor.speed_max, rel=1e-3)


def test_bad_speed_max_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    with open(YAML, encoding="utf-8") as f:
        bad.write_text(f.read().replace("speed_max:  2983.0", "speed_max:  2618.0"))
    with pytest.raises(ValueError, match="speed_max"):
        load_config(bad)


def test_max_thrust_scale_derived_from_cap():
    phys = load_config(YAML).physics
    assert resolve_max_thrust_scale(None, phys.mass, phys.max_thrust) == pytest.approx(1.5387, abs=1e-4)
    assert resolve_max_thrust_scale(1.2, phys.mass, phys.max_thrust) == 1.2
    with pytest.raises(ValueError, match="max_thrust_scale"):
        resolve_max_thrust_scale(1.8, phys.mass, phys.max_thrust)


def test_set_motor_tau_down_defaults_to_up():
    ctrl = controller(n=2)
    tau = torch.tensor([0.1, 0.12])
    ctrl.set_motor_tau(tau, torch.tensor([0, 1]))
    expected = -torch.expm1(-DT / tau)
    assert torch.allclose(ctrl.motor_chain.blend_down.view(-1), expected, rtol=1e-5)
    assert torch.allclose(ctrl.motor_chain.blend_up.view(-1), expected, rtol=1e-5)


def test_reset_puts_motors_at_value_for_given_envs():
    c = chain(n=2, tau=0.072)
    c.reset(None, 0.0)
    c.reset(torch.tensor([1]), 0.066)
    assert torch.equal(c.motor_forces[0], torch.zeros(4))
    assert torch.allclose(c.motor_forces[1], torch.full((4,), 0.066))


# ── Controller with the repo's crazyflie.yaml ────────────────────────────────

def controller(n=1):
    return CascadePIDController.from_drone_config(load_config(YAML), num_envs=n, dt=DT)


def root_state(ang_vel_b):
    n = ang_vel_b.shape[0]
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4)
    return torch.cat([torch.zeros(n, 3), quat, torch.zeros(n, 3), ang_vel_b], dim=-1)


def test_controller_units_and_hover_reset():
    ctrl = controller(n=2)
    mg = load_config(YAML).physics.mass * 9.81
    assert ctrl.thrust_cmd_scale == pytest.approx(4 * K)
    assert ctrl.thrust_base_cmd == 36000.0
    ctrl.reset()
    assert torch.allclose(ctrl.motor_chain.motor_forces, torch.full((2, 4), mg / 4))
    # tasks send total newtons / thrust_cmd_scale; hover must give m g with no transient
    thrust, moment = ctrl(
        root_state(torch.zeros(2, 3)),
        target_body_rates=torch.zeros(2, 3),
        thrust_cmd=torch.full((2, 1), mg / ctrl.thrust_cmd_scale),
        command_level="body_rate",
        body_rates_in_body_frame=True,
    )
    assert torch.allclose(thrust, torch.full((2, 1), mg), rtol=1e-3)
    assert torch.allclose(moment, torch.zeros(2, 3), atol=1e-6)


def roll_step(tau, rate_sp=1.0, duration=0.5, substeps=20):
    """Single-axis roll plant Ixx ω̇ = τx under the controller, 500 Hz, ω held between calls."""
    ctrl = controller()
    ctrl.set_motor_tau(tau)
    ctrl.reset()
    ixx = load_config(YAML).physics.inertia.ixx
    w = torch.zeros(1, 3)
    hover_cmd = load_config(YAML).physics.mass * 9.81 / ctrl.thrust_cmd_scale
    out = []
    for _ in range(round(duration / DT)):
        _, moment = ctrl(
            root_state(w),
            target_body_rates=torch.tensor([[rate_sp, 0.0, 0.0]]),
            thrust_cmd=torch.tensor([[hover_cmd]]),
            command_level="body_rate",
            body_rates_in_body_frame=True,
        )
        for _ in range(substeps):
            w = w + (DT / substeps) * torch.tensor([[moment[0, 0].item() / ixx, 0.0, 0.0]])
        out.append(w[0, 0].item())
    return torch.tensor(out)


def test_closed_loop_rate_step_with_72ms_motors_is_stable_and_overshoots():
    w = roll_step(tau=0.072)
    overshoot = w.max().item() - 1.0
    assert torch.isfinite(w).all()
    # the single-axis model on the analysis page gives ~36 % (sim PID + firmware mixer + 72 ms lag)
    assert 0.2 < overshoot < 0.55
    assert abs(w[-50:].mean().item() - 1.0) < 0.2


def test_closed_loop_rate_step_without_lag_barely_overshoots():
    w = roll_step(tau=0.0)
    assert torch.isfinite(w).all()
    assert w.max().item() - 1.0 < 0.1
