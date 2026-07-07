# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `CrazyPlayGround` extension.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

# Snapshot the registry before importing the extension so we can list exactly the
# environments that CrazyPlayGround registers, regardless of how they are named.
_tasks_before = set(gym.registry.keys())
import CrazyPlayGround.tasks  # noqa: F401

_crazyplayground_tasks = sorted(set(gym.registry.keys()) - _tasks_before)


def main():
    """Print all environments registered in `CrazyPlayGround` extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # add every environment registered by the CrazyPlayGround extension
    for index, task_id in enumerate(_crazyplayground_tasks):
        task_spec = gym.registry[task_id]
        table.add_row(
            [index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs.get("env_cfg_entry_point", "-")]
        )

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
