# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Teleoperation",
    entry_point=f"{__name__}.teleop_env:TeleoperationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teleop_env:TeleoperationEnvCfg",
    },
)
