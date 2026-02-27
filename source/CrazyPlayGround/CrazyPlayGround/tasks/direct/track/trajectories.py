# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-torch trajectory functions for trajectory-tracking environments.

All functions share the same signature:
    t : Tensor[N, steps]  — time parameter (radians)
    returns Tensor[N, steps, 3]  — unnormalised positions (unit-amplitude)
"""

from __future__ import annotations

import torch
from torch import Tensor

from isaaclab.utils.math import quat_apply


# ---------------------------------------------------------------------------
# Trajectory functions
# ---------------------------------------------------------------------------

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


def vertical_circle(t: Tensor) -> Tensor:
    """Unit circle in the XZ plane."""
    x = torch.cos(t)
    y = torch.zeros_like(t)
    z = torch.sin(t)
    return torch.stack([x, y, z], dim=-1)


def helix(t: Tensor, pitch: float = 0.3) -> Tensor:
    """Circle in XY with a linear Z rise per revolution.

    Z rises by ``pitch`` per full revolution (2π in t).
    """
    x = torch.cos(t)
    y = torch.sin(t)
    z = pitch * t / (2.0 * torch.pi)
    # Centre z around zero so the trajectory starts near the origin
    z = z - z.mean(dim=-1, keepdim=True)
    return torch.stack([x, y, z], dim=-1)


def sine_wave(t: Tensor, freq: float = 2.0) -> Tensor:
    """Sinusoidal motion along X, constant Y=0."""
    x = torch.sin(t)
    y = torch.zeros_like(t)
    z = torch.sin(freq * t) * 0.3
    return torch.stack([x, y, z], dim=-1)


def epitrochoid(t: Tensor, r: float = 0.5, d: float = 0.7) -> Tensor:
    """Epitrochoid (rose-like) curve in the XY plane.

    With r=0.5, d=0.7 produces a nice petal pattern.
    """
    R = 1.0
    x = (R + r) * torch.cos(t) - d * torch.cos((R + r) / r * t)
    y = (R + r) * torch.sin(t) - d * torch.sin((R + r) / r * t)
    # Normalise amplitude to unit range
    amp = float(R + r + d)
    x = x / amp
    y = y / amp
    z = torch.zeros_like(t)
    return torch.stack([x, y, z], dim=-1)


def trefoil(t: Tensor) -> Tensor:
    """3-D trefoil knot."""
    x = torch.sin(t) + 2.0 * torch.sin(2.0 * t)
    y = torch.cos(t) - 2.0 * torch.cos(2.0 * t)
    z = -torch.sin(3.0 * t)
    # Normalise to unit amplitude
    amp = 3.0
    return torch.stack([x / amp, y / amp, z / amp], dim=-1)


def polygon(t: Tensor, n: int = 4) -> Tensor:
    """Smooth approximation of a regular n-gon using Fourier components.

    The first harmonic gives a circle; adding the n-th harmonic creates
    corner-like features without discontinuities.
    """
    x = torch.cos(t) + 0.2 * torch.cos((n - 1) * t)
    y = torch.sin(t) - 0.2 * torch.sin((n - 1) * t)
    z = torch.zeros_like(t)
    amp = 1.2
    return torch.stack([x / amp, y / amp, z], dim=-1)


# ---------------------------------------------------------------------------
# Registry — name → callable
# ---------------------------------------------------------------------------

TRAJECTORIES: dict[str, callable] = {
    "lemniscate": lemniscate,
    "circle": circle,
    "vertical_circle": vertical_circle,
    "helix": helix,
    "sine_wave": sine_wave,
    "epitrochoid": epitrochoid,
    "trefoil": trefoil,
    "polygon": polygon,
}


# ---------------------------------------------------------------------------
# Transform helper
# ---------------------------------------------------------------------------

def apply_traj_transform(
    pos: Tensor,
    scale: Tensor,
    rot_quat: Tensor,
    offset: Tensor,
) -> Tensor:
    """Apply scale, rotation and translation to a batch of trajectory positions.

    Args:
        pos:      [N, steps, 3]  — raw trajectory positions
        scale:    [N, 3]         — per-env scale in x, y, z
        rot_quat: [N, 4]         — per-env rotation quaternion (w, x, y, z)
        offset:   [N, 3]         — per-env world-frame origin

    Returns:
        [N, steps, 3] transformed positions in world frame.
    """
    N, steps, _ = pos.shape

    # Scale
    pos_scaled = pos * scale.unsqueeze(1)  # [N, steps, 3]

    # Rotate — quat_rotate expects [M, 3] so flatten then reshape
    pos_flat = pos_scaled.reshape(N * steps, 3)
    rot_expanded = rot_quat.unsqueeze(1).expand(N, steps, 4).reshape(N * steps, 4)
    pos_rot = quat_apply(rot_expanded, pos_flat)  # [N*steps, 3]
    pos_rot = pos_rot.reshape(N, steps, 3)

    # Translate — offset is [N, 3], broadcast over steps
    pos_world = pos_rot + offset.unsqueeze(1)  # [N, 1, 3] → [N, steps, 3]

    return pos_world
