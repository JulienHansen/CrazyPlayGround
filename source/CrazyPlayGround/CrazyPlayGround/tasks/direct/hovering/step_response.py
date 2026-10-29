"""Reward-independent step-response metrics for hovering episodes.

Each episode is read as the response to a step input in control-theory terms: the
setpoint jumps from the spawn point p0 to the goal g. In this module "step" alone means a
policy step (one row of the trajectory); the jump itself is the *displacement* g - p0, of
length d0 = |g - p0| and direction u = (g - p0) / d0. The progress
s(t) = (p(t) - p0) . u / d0 goes from 0 to 1, and d(t) = |g - p(t)| is the distance left.
Pure torch with no Isaac Lab imports, so the CPU test suite can load it on its own.
"""

from __future__ import annotations

import torch


def step_response(
    pos_w: torch.Tensor,
    start_pos_w: torch.Tensor,
    goal_pos_w: torch.Tensor,
    length: torch.Tensor,
    dt: float,
    band_frac: float = 0.05,
    band_min: float = 0.02,
    ss_window_s: float = 2.0,
    d0_min: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Step-response metrics of E episodes of up to T policy steps.

    A threshold that is never crossed, and every normalised metric of a displacement
    shorter than ``d0_min``, is NaN rather than 0, so a mean never counts a failure as
    a perfect score.

    Args:
        pos_w: Drone position after each policy step, (E, T, 3), world frame, m. Row k
            is at t = (k + 1) * dt.
        start_pos_w: Spawn position p0, (E, 3), world frame, m.
        goal_pos_w: Goal position g, (E, 3), world frame, m.
        length: Valid rows per episode, (E,), long; rows >= length are ignored.
        dt: Policy step, s.
        band_frac: Settling band as a fraction of d0.
        band_min: Lower bound of the settling band, m.
        ss_window_s: Window at the end of the episode for the steady-state error, s.
        d0_min: Displacements shorter than this get NaN normalised metrics, m.

    Returns:
        Dict of (E,) tensors: ``d0`` [m], displacement length; ``tau`` [s], first time s >= 0.632;
        ``rise_10_90`` [s], time from s >= 0.1 to s >= 0.9; ``overshoot`` [-],
        max(s) - 1 clamped at 0; ``settling`` [s], time of the first sample after the
        last one outside the band max(band_frac * d0, band_min), 0 if never outside and
        NaN if still outside at the end; ``ss_error`` [m], mean d over the last
        ``ss_window_s``; ``iae_norm`` [s], integral of d dt divided by d0.
    """
    T = pos_w.shape[1]
    t = (torch.arange(T, device=pos_w.device) + 1) * dt  # (T,)
    valid = torch.arange(T, device=pos_w.device) < length[:, None]  # (E, T)
    t_end = length * dt  # time of the last valid sample

    displacement_w = goal_pos_w - start_pos_w
    d0 = torch.linalg.norm(displacement_w, dim=1)
    d0_safe = d0.clamp_min(1e-9)
    u = displacement_w / d0_safe[:, None]
    s = ((pos_w - start_pos_w[:, None]) * u[:, None]).sum(-1) / d0_safe[:, None]
    d = torch.linalg.norm(goal_pos_w[:, None] - pos_w, dim=-1)

    inf, nan = float("inf"), float("nan")

    def first_time(level: float) -> torch.Tensor:
        t_cross = torch.where(valid & (s >= level), t, inf).amin(1)
        return torch.where(torch.isinf(t_cross), nan, t_cross)

    t_10, tau, t_90 = first_time(0.1), first_time(0.632), first_time(0.9)
    overshoot = (torch.where(valid, s, -inf).amax(1) - 1).clamp_min(0)

    band = torch.clamp_min(band_frac * d0, band_min)
    t_last_out = torch.where(valid & (d > band[:, None]), t, 0.0).amax(1)  # 0: never outside
    settling = torch.where(t_last_out >= t_end, nan, torch.where(t_last_out > 0, t_last_out + dt, 0.0))

    in_window = valid & (t > (t_end - ss_window_s)[:, None])
    # where, not a product: rows past length may hold NaN, and NaN * 0 is NaN
    ss_error = torch.where(in_window, d, 0.0).sum(1) / in_window.sum(1)  # 0/0 = NaN for an empty episode

    iae_norm = torch.where(valid, d, 0.0).sum(1) * dt / d0_safe

    too_short = d0 < d0_min
    out = {"tau": tau, "rise_10_90": t_90 - t_10, "overshoot": overshoot, "settling": settling, "iae_norm": iae_norm}
    out = {k: torch.where(too_short, nan, v) for k, v in out.items()}
    return {"d0": d0, **out, "ss_error": ss_error}
