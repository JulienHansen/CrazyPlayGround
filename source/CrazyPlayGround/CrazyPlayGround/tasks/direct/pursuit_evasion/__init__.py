# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

for _level, _cfg in (
    ("Rate", "RatePursuitEvasionEnvCfg"),
    ("Att", "AttPursuitEvasionEnvCfg"),
    ("Vel", "VelPursuitEvasionEnvCfg"),
):
    gym.register(
        id=f"{_level}-PursuitEvasion",
        entry_point=f"{__name__}.pursuit_evasion_env:PursuitEvasionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.pursuit_evasion_env:{_cfg}",
            "orcax_cfg_entry_point": f"{agents.__name__}:orcax_ppo_cfg.yaml",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        },
    )
