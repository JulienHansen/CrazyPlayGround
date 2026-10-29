"""Step-response metrics checked against analytic first- and second-order responses.

These metrics are how reward shapes get compared, so each one is pinned to a value
known in closed form rather than to a regression snapshot.
"""

import math
import os
import sys

import pytest
import torch

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "source", "CrazyPlayGround", "CrazyPlayGround", "tasks", "direct", "hovering")
)
from step_response import step_response  # noqa: E402

DT, T = 0.01, 1000
START = torch.tensor([[0.0, 0.0, 1.0]])
GOAL = torch.tensor([[1.0, 0.0, 1.0]])  # d0 = 1 m along x


def episode(progress, start=START, goal=GOAL, length=T):
    """Positions that follow the given progress s(t) on the start-to-goal line."""
    pos_w = start[:, None] + progress[None, :, None] * (goal - start)[:, None]
    return step_response(pos_w, start, goal, torch.tensor([length]), DT)


def times():
    return (torch.arange(T, dtype=torch.float64) + 1) * DT


def test_first_order_response_matches_closed_form():
    tau = 0.5
    m = episode(1 - torch.exp(-times() / tau).float())
    assert m["d0"].item() == pytest.approx(1.0)
    assert m["tau"].item() == pytest.approx(tau, abs=DT)
    assert m["rise_10_90"].item() == pytest.approx(tau * math.log(9), abs=2 * DT)
    assert m["settling"].item() == pytest.approx(tau * math.log(20), abs=2 * DT)  # 5 % band of a 1 m step
    assert m["iae_norm"].item() == pytest.approx(tau, abs=DT)
    assert m["overshoot"].item() == pytest.approx(0.0, abs=1e-6)
    assert m["ss_error"].item() == pytest.approx(0.0, abs=1e-6)


def test_second_order_overshoot_matches_closed_form():
    zeta, wn = 0.5, 4.0
    wd, phi = wn * math.sqrt(1 - zeta**2), math.acos(zeta)
    t = times()
    s = 1 - torch.exp(-zeta * wn * t) / math.sqrt(1 - zeta**2) * torch.sin(wd * t + phi)
    m = episode(s.float())
    assert m["overshoot"].item() == pytest.approx(math.exp(-zeta * math.pi / math.sqrt(1 - zeta**2)), abs=1e-3)


def test_never_reaching_the_goal_gives_nan_not_zero():
    m = episode(torch.full((T,), 0.5))
    for k in ("tau", "rise_10_90", "settling"):
        assert math.isnan(m[k].item()), k
    assert m["ss_error"].item() == pytest.approx(0.5, abs=1e-6)


def test_displacement_shorter_than_d0_min_gives_nan_except_steady_state_error():
    m = episode(torch.zeros(T), goal=START + torch.tensor([[0.05, 0.0, 0.0]]))
    for k in ("tau", "rise_10_90", "overshoot", "settling", "iae_norm"):
        assert math.isnan(m[k].item()), k
    assert m["ss_error"].item() == pytest.approx(0.05, abs=1e-6)


def test_rows_past_length_are_ignored():
    n = 300
    s = 1 - torch.exp(-times() / 0.5).float()
    ref = episode(s[:n].clone().repeat(T // n + 1)[:T], length=n)  # same first n rows, junk after
    s_nan = s.clone()
    s_nan[n:] = float("nan")
    m = episode(s_nan, length=n)
    for k in ref:
        assert m[k].item() == pytest.approx(ref[k].item(), nan_ok=True), k
