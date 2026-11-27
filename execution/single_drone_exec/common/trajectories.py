"""Reference trajectories for the tracking scripts, as TrackingEnv computes them.

The shape functions and the arc-length helpers are copies of `tasks/direct/track/trajectories.py`,
which cannot be imported here: it imports Isaac Lab. Keep them identical
(tests/test_traj_tracking_scripts.py checks it); to fly another shape, copy its function from
there and add it to TRAJECTORIES.

All shapes share the training signature:
    t: (...) time parameter, rad
    returns (..., 3) unit-amplitude positions, scaled and placed by `reference_w`.

The reference moves at a constant linear speed, as in TrackingEnv (#96): `build_lap` resamples
one lap into TRAJ_NUM_WAYPOINTS equal-length segments, and `interp_waypoints` gives the point a
given distance along it.
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


# ── Arc-length resampling (verbatim from tasks/direct/track/trajectories.py) ───

def resample_by_arc_length(points: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Resample a densely sampled lap into ``k`` segments of equal length.

    Args:
        points: [N, M+1, 3] dense samples of one lap, first to last, m (already scaled).
        k:      number of equal segments.

    Returns:
        waypoints: [N, k+1, 3] points at arc lengths 0, L/k, ..., L, m. The first and last are
                   the first and last dense samples.
        length:    [N] lap length L, m.
    """
    m = points.shape[1] - 1
    seg_norm = torch.linalg.norm(points[:, 1:] - points[:, :-1], dim=-1)                  # [N, M]
    cum_norm = torch.cat([torch.zeros_like(seg_norm[:, :1]), torch.cumsum(seg_norm, dim=1)], dim=1)  # [N, M+1]
    length = cum_norm[:, -1]
    targets = torch.linspace(0.0, 1.0, k + 1, device=points.device) * length.unsqueeze(1)  # [N, k+1]
    # dense segment holding each target; the last target (= L) falls in the last segment
    idx = (torch.searchsorted(cum_norm, targets.contiguous(), right=True) - 1).clamp(0, m - 1)
    c0, c1 = cum_norm.gather(1, idx), cum_norm.gather(1, idx + 1)
    frac = ((targets - c0) / (c1 - c0).clamp_min(1e-12)).clamp(0.0, 1.0).unsqueeze(-1)  # [N, k+1, 1]
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, 3)
    p0 = points.gather(1, gather_idx)
    p1 = points.gather(1, gather_idx + 1)
    return p0 + frac * (p1 - p0), length


def interp_waypoints(waypoints: Tensor, length: Tensor, distance: Tensor) -> Tensor:
    """Point at a given distance along a closed lap of equally spaced waypoints, lap after lap.

    The point is the linear interpolation of the two waypoints around ``distance``, so it moves
    at a constant speed when ``distance`` grows linearly in time.

    Args:
        waypoints: [N, k+1, 3] waypoints spaced ``length / k`` apart along the curve, m
                   (closed: the last one equals the first).
        length:    [N] lap length, m.
        distance:  [N, S] distance travelled from waypoint 0, m (>= 0).

    Returns:
        [N, S, 3] positions, m.
    """
    k = waypoints.shape[1] - 1
    length = length.unsqueeze(1)                     # [N, 1]
    lap = torch.floor(distance / length)             # [N, S] completed laps
    u = (distance - lap * length) * (k / length)     # [N, S] position in segments
    idx = torch.floor(u).long().clamp(0, k - 1)
    frac = (u - idx).clamp(0.0, 1.0).unsqueeze(-1)   # [N, S, 1]
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, 3)
    p0 = waypoints.gather(1, gather_idx)
    p1 = waypoints.gather(1, gather_idx + 1)
    return p0 + frac * (p1 - p0)


# ── One lap, as TrackingEnv._build_traj_waypoints ──────────────────────────────

TRAJ_NUM_WAYPOINTS = 256  # equal-length segments per lap (tracking_env.TRAJ_NUM_WAYPOINTS)


def build_lap(traj_type: str, scale: Sequence[float], origin: Sequence[float], c: float = 0.0,
              t0: float = 0.0, k: int = TRAJ_NUM_WAYPOINTS) -> tuple[Tensor, Tensor]:
    """One lap from `t0`, resampled into `k` equal-length segments (no rotation, as the deploy flies).

    Same steps as TrackingEnv._build_traj_waypoints: 4k+1 dense samples over [t0, t0 + 2π],
    scaled and placed, then resampled by arc length.

    Args:
        traj_type: a key of TRAJECTORIES.
        scale: (x, y, z) scale, m.
        origin: (x, y, z) trajectory centre, world frame, m.
        c: lemniscate z coupling.
        t0: phase of the first waypoint, rad.
        k: number of segments.

    Returns:
        waypoints: (k+1, 3) world frame, m; the last equals the first.
        length: () lap length, m.
    """
    u = torch.linspace(0.0, 2.0 * math.pi, 4 * k + 1)
    dense_w = reference_w(t0 + u, traj_type, scale, origin, c=c)  # [4k+1, 3]
    waypoints_w, length = resample_by_arc_length(dense_w.unsqueeze(0), k)
    return waypoints_w[0], length[0]
