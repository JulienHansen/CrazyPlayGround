"""Unit tests of the geofence in execution/single_drone_exec/common/utils.py."""

import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "execution", "single_drone_exec"))
from common.utils import ARENA_FENCE, Fence  # noqa: E402

FENCE = Fence(lo=(-2.0, -1.0, 0.3), hi=(2.0, 1.0, 1.8), lookahead_s=0.1)
STILL = torch.zeros(3)


def test_inside_is_no_breach():
    assert FENCE.breach(torch.tensor([0.0, 0.0, 1.0]), STILL) is None
    assert FENCE.breach(torch.tensor([2.0, -1.0, 0.3]), STILL) is None  # on the faces is still inside


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("side", ("lo", "hi"))
def test_each_face_trips(axis, side):
    pos = torch.tensor([0.0, 0.0, 1.0])
    pos[axis] = getattr(FENCE, side)[axis] + (-0.01 if side == "lo" else 0.01)
    assert FENCE.breach(pos, STILL).startswith("Position [")


def test_lookahead_trips_before_the_face():
    pos = torch.tensor([0.0, 0.0, 0.4])
    assert FENCE.breach(pos, torch.tensor([0.0, 0.0, -0.9])) is None  # 0.31 m in 0.1 s
    assert FENCE.breach(pos, torch.tensor([0.0, 0.0, -1.5])).startswith("Position in 0.1 s")  # 0.25 m


def test_clamp_moves_onto_the_box():
    assert FENCE.clamp(torch.tensor([3.0, -0.5, 0.1])).tolist() == pytest.approx([2.0, -0.5, 0.3])
    inside = torch.tensor([0.1, 0.2, 1.0])
    assert torch.equal(FENCE.clamp(inside), inside)


def test_arena_fence_holds_its_safe_hover_point():
    assert ARENA_FENCE.breach(torch.tensor(ARENA_FENCE.safe_hover_pos), STILL) is None
