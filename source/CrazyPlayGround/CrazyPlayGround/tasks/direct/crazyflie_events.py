"""Isaac Lab events shared by the Crazyflie envs.

``set_total_mass`` (startup) sets the PhysX mass of every Crazyflie to ``drone.mass`` of
``crazyflie.yaml``. Without it the yaml mass only drives the controller and PhysX flies the USD mass.

The Crazyflie USD is not in this repo (``isaaclab_assets.CRAZYFLIE_CFG``, downloaded by Isaac Lab),
so the mass is changed at runtime here instead of in the USD. ``spawn.mass_props`` can't do it:
``@apply_nested`` would give each of the 5 bodies the full mass.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from CrazyPlayGround.controllers import load_config

if TYPE_CHECKING:
    from isaaclab.envs import DirectMARLEnv, DirectRLEnv

logger = logging.getLogger(__name__)


def set_total_mass(
    env: DirectRLEnv | DirectMARLEnv,
    env_ids: torch.Tensor | None,
    mass: float | None = None,
) -> None:
    """Scale every articulation in the scene (all Crazyflies) to a total mass.

    Body masses and inertias are multiplied by ``mass / Σ masses``, keeping the USD mass distribution.

    Args:
        env: The environment.
        env_ids: Envs to update; None updates all.
        mass: Total mass [kg]; None reads ``drone.mass`` from ``env.cfg.drone_config_path``.
    """
    if mass is None:
        mass = load_config(env.cfg.drone_config_path).physics.mass
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    for name, asset in env.scene.articulations.items():
        view = asset.root_physx_view
        masses = view.get_masses()      # (num_envs, num_bodies) [kg], CPU
        inertias = view.get_inertias()  # (num_envs, num_bodies, 9) [kg·m²], CPU
        usd_total = masses[env_ids].sum(dim=1, keepdim=True)  # (len(env_ids), 1)
        ratio = mass / usd_total
        masses[env_ids] = masses[env_ids] * ratio
        inertias[env_ids] = inertias[env_ids] * ratio.unsqueeze(-1)
        view.set_masses(masses, env_ids)
        view.set_inertias(inertias, env_ids)
        # reset-time mass DR draws around the defaults
        asset.data.default_mass[env_ids] = masses[env_ids].to(asset.data.default_mass.device)
        asset.data.default_inertia[env_ids] = inertias[env_ids].to(asset.data.default_inertia.device)
        logger.info(
            "set_total_mass: %s total mass %.5f kg (USD) -> %.5f kg (%d envs)",
            name, usd_total.mean().item(), mass, len(env_ids),
        )


@configclass
class CrazyflieEventCfg:
    """Events of every Crazyflie env: ``events: CrazyflieEventCfg = CrazyflieEventCfg()`` in the env cfg."""

    set_total_mass = EventTerm(func=set_total_mass, mode="startup")
