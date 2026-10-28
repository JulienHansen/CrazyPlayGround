"""Camera detection model for the pursuit-evasion task.

The pursuer carries a forward-looking camera rigidly mounted on its body,
optical axis along body +x. Instead of rendering images, the detector
projects the evader's position through a pinhole model and returns what an
object detector would output: a visibility flag, the normalised image
coordinates of the bounding-box centre and its normalised width.

The ideal detector (all noise terms 0) is exact. The noise terms are there to
move towards a realistic detector later without changing the observation
layout.

Camera frame convention (derived from the body frame, FLU):

* ``u`` (image right)  = -y_b / x_b, normalised by ``tan(hfov / 2)``
* ``v`` (image up)     = +z_b / x_b, normalised by ``tan(vfov / 2)``

so ``u, v`` lie in [-1, 1] whenever the target is inside the field of view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaaclab.utils.math import subtract_frame_transforms


@dataclass
class DetectorCfg:
    """Parameters of the camera and of the detector error model."""

    hfov_deg: float = 90.0
    """Horizontal field of view."""
    vfov_deg: float = 70.0
    """Vertical field of view."""
    max_range: float = 6.0
    """Beyond this distance the target is never detected."""
    target_radius: float = 0.06
    """Radius of the sphere that stands for the target drone when computing its apparent size."""

    # Error model (all zero = ideal detector).
    pixel_noise_std: float = 0.0
    """Std of the noise added to ``u`` and ``v`` (normalised image units)."""
    size_noise_rel_std: float = 0.0
    """Relative std of the multiplicative noise on the apparent size."""
    miss_prob: float = 0.0
    """Probability that a target inside the field of view is not detected at a given step."""


class Detector:
    """Pinhole-camera detector, batched over environments.

    Args:
        cfg: Camera and error-model parameters.
        device: Torch device of the outputs.
    """

    def __init__(self, cfg: DetectorCfg, device: torch.device | str):
        self.cfg = cfg
        self.device = device
        self._tan_half_hfov = math.tan(math.radians(cfg.hfov_deg) / 2.0)
        self._tan_half_vfov = math.tan(math.radians(cfg.vfov_deg) / 2.0)
        self._hfov = math.radians(cfg.hfov_deg)

    def size_to_range(self, size_norm: torch.Tensor) -> torch.Tensor:
        """Invert the apparent-size model: distance implied by a normalised box width.

        This is the only way the reward recovers a range: it uses what the
        detector sees, not the simulator's ground truth.

        Args:
            size_norm: Normalised box width, shape ``(num_envs,)``. Must be > 0.

        Returns:
            Estimated distance to the target, shape ``(num_envs,)``, m.
        """
        half_angle = 0.5 * size_norm.clamp_min(1e-6) * self._hfov
        return self.cfg.target_radius / torch.tan(half_angle)

    def __call__(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_w: torch.Tensor,
        target_pos_w: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Detect the target.

        Args:
            cam_pos_w: Camera (pursuer) position, shape ``(num_envs, 3)``, world frame, m.
            cam_quat_w: Camera (pursuer) orientation, shape ``(num_envs, 4)``, WXYZ, world frame.
            target_pos_w: Target (evader) position, shape ``(num_envs, 3)``, world frame, m.

        Returns:
            A dict of tensors of shape ``(num_envs,)``:

            * ``visible``: 1.0 when detected, else 0.0.
            * ``u``, ``v``: normalised image coordinates of the box centre, 0 when not visible.
            * ``size``: box width over the image width, 0 when not visible.
            * ``range``: ground-truth distance (for logging only, never observed).
        """
        rel_b, _ = subtract_frame_transforms(cam_pos_w, cam_quat_w, target_pos_w)
        x, y, z = rel_b.unbind(-1)
        dist = torch.linalg.norm(rel_b, dim=-1)

        in_front = x > 1e-3
        x_safe = torch.where(in_front, x, torch.ones_like(x))
        u = (-y / x_safe) / self._tan_half_hfov
        v = (z / x_safe) / self._tan_half_vfov
        visible = in_front & (u.abs() <= 1.0) & (v.abs() <= 1.0) & (dist <= self.cfg.max_range)

        # Angular diameter of a sphere of radius r at distance d, over the horizontal FOV.
        size = 2.0 * torch.atan(self.cfg.target_radius / dist.clamp_min(self.cfg.target_radius)) / self._hfov

        if self.cfg.pixel_noise_std > 0.0:
            u = u + torch.randn_like(u) * self.cfg.pixel_noise_std
            v = v + torch.randn_like(v) * self.cfg.pixel_noise_std
        if self.cfg.size_noise_rel_std > 0.0:
            size = size * (1.0 + torch.randn_like(size) * self.cfg.size_noise_rel_std).clamp_min(0.1)
        if self.cfg.miss_prob > 0.0:
            visible = visible & (torch.rand_like(u) >= self.cfg.miss_prob)

        vis_f = visible.float()
        return {
            "visible": vis_f,
            "u": u.clamp(-1.0, 1.0) * vis_f,
            "v": v.clamp(-1.0, 1.0) * vis_f,
            "size": size * vis_f,
            "range": dist,
        }
