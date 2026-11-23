"""Reference trajectories for the tracking scripts, as TrackingEnv computes them.

The shape functions are copies of `tasks/direct/track/trajectories.py`, which cannot be
imported here: it imports Isaac Lab. Keep them identical (tests/test_traj_tracking_scripts.py
checks it); to fly another shape, copy its function from there and add it to TRAJECTORIES.

All shapes share the training signature:
    t: (...) time parameter, rad
    returns (..., 3) unit-amplitude positions, scaled and placed by `reference_w`.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


# ── Shapes (verbatim from tasks/direct/track/trajectories.py) ──────────────────

def lemniscate(t: Tensor, c: float = 0.0) -> Tensor:
    """Figure-8 (lemniscate of Bernoulli) in the XY plane.

    ``c`` adds a Z coupling so the figure-8 gains a vertical component.
    """
    denom = 1.0 + torch.sin(t) ** 2
    x = torch.cos(t) / denom
    y = torch.sin(t) * torch.cos(t) / denom
    z = c * torch.sin(2.0 * t)
    return torch.stack([x, y, z], dim=-1)


def circle(t: Tensor) -> Tensor:
    """Unit circle in the XY plane."""
    x = torch.cos(t)
    y = torch.sin(t)
    z = torch.zeros_like(t)
    return torch.stack([x, y, z], dim=-1)


TRAJECTORIES = {
    "lemniscate": lemniscate,
    "circle": circle,
}


# ── Placement ──────────────────────────────────────────────────────────────────

def unit_shape(t: Tensor, traj_type: str, c: float = 0.0) -> Tensor:
    """Unit-amplitude shape at `t`; `c` is used by the lemniscate only, as in TrackingEnv._compute_traj."""
    fn = TRAJECTORIES[traj_type]
    return fn(t, c=c) if traj_type == "lemniscate" else fn(t)


def reference_w(t: Tensor, traj_type: str, scale: Sequence[float], origin: Sequence[float],
                c: float = 0.0) -> Tensor:
    """Reference position: unit_shape(t) * scale + origin (TrackingEnv's transform, no rotation).

    Args:
        t: (...) time parameter, rad.
        traj_type: a key of TRAJECTORIES.
        scale: (x, y, z) scale, m.
        origin: (x, y, z) trajectory centre, world frame, m.
        c: lemniscate z coupling.

    Returns:
        (..., 3) world frame, m.
    """
    scale = torch.tensor(scale, dtype=t.dtype, device=t.device)
    origin = torch.tensor(origin, dtype=t.dtype, device=t.device)
    return unit_shape(t, traj_type, c) * scale + origin
