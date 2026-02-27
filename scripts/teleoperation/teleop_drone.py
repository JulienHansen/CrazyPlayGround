# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperation script for Crazyflie drone.

Usage:
    python scripts/teleoperation/teleop_drone.py --input=gamepad --mode=velocity
    python scripts/teleoperation/teleop_drone.py --input=keyboard --mode=position
    python scripts/teleoperation/teleop_drone.py --input=both  # Fallback: gamepad -> keyboard
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Add the teleoperation directory to path for input_handlers import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Crazyflie drone.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--input",
    type=str,
    default="both",
    choices=["gamepad", "keyboard", "both"],
    help="Input method: gamepad, keyboard, or both (fallback)",
)
parser.add_argument(
    "--mode",
    type=str,
    default="velocity",
    choices=["position", "velocity", "attitude"],
    help="Initial control mode",
)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import CrazyPlayGround.tasks  # noqa: F401

# Import after simulation app is running
from input_handlers import GamepadHandler, KeyboardHandler, BaseInputHandler


def create_input_handler(input_type: str, device: torch.device) -> BaseInputHandler | None:
    """Create and initialize input handler based on type.

    Args:
        input_type: One of "gamepad", "keyboard", or "both"
        device: Torch device for action tensors

    Returns:
        Initialized input handler or None if none available
    """
    if input_type == "gamepad":
        handler = GamepadHandler(device)
        if handler.initialize():
            return handler
        print("[Teleop] Gamepad not available")
        return None

    elif input_type == "keyboard":
        handler = KeyboardHandler(device)
        if handler.initialize():
            return handler
        print("[Teleop] Keyboard not available")
        return None

    elif input_type == "both":
        # Try gamepad first, fallback to keyboard
        gamepad = GamepadHandler(device)
        if gamepad.initialize():
            print("[Teleop] Using gamepad input")
            return gamepad

        keyboard = KeyboardHandler(device)
        if keyboard.initialize():
            print("[Teleop] Gamepad not available, using keyboard input")
            return keyboard

        print("[Teleop] No input method available")
        return None

    return None


def print_controls(input_type: str) -> None:
    """Print control instructions."""
    print("\n" + "=" * 60)
    print("TELEOPERATION CONTROLS")
    print("=" * 60)

    if input_type in ("gamepad", "both"):
        print("\nGAMEPAD:")
        print("  Velocity Mode:")
        print("    Left Stick Y  : Forward/Backward")
        print("    Left Stick X  : Left/Right")
        print("    Right Stick Y : Up/Down")
        print("    Right Stick X : Yaw")
        print("  Attitude Mode:")
        print("    Left Stick X  : Roll")
        print("    Left Stick Y  : Pitch")
        print("    Right Stick X : Yaw Rate")
        print("    RT - LT       : Thrust")
        print("  Buttons:")
        print("    A : Velocity mode")
        print("    B : Attitude mode")
        print("    Y : Position mode")
        print("    Start : Reset")
        print("    Back  : Quit")

    if input_type in ("keyboard", "both"):
        print("\nKEYBOARD:")
        print("  Movement (Position/Velocity):")
        print("    Z/W/Up    : Forward")
        print("    S/Down    : Backward")
        print("    Q/A/Left  : Left")
        print("    D/Right   : Right")
        print("    Space     : Up")
        print("    Shift     : Down")
        print("  Mode Switch:")
        print("    1 : Position mode")
        print("    2 : Velocity mode")
        print("    3 : Attitude mode")
        print("  Control:")
        print("    R   : Reset")
        print("    Esc : Quit")

    print("=" * 60 + "\n")


def main():
    """Main teleoperation loop."""
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        "Teleoperation",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )

    # Create environment
    env = gym.make("Teleoperation", cfg=env_cfg)

    print(f"[INFO] Gym observation space: {env.observation_space}")
    print(f"[INFO] Gym action space: {env.action_space}")

    # Create input handler
    handler = create_input_handler(args_cli.input, env.unwrapped.device)
    if handler is None:
        print("[ERROR] No input method available, exiting")
        env.close()
        return

    # Set initial control mode
    handler.current_mode = args_cli.mode
    env.unwrapped.set_control_mode(args_cli.mode)

    # Print controls
    print_controls(args_cli.input)

    print(f"[INFO] Starting teleoperation in {args_cli.mode.upper()} mode")
    print(f"[INFO] Input method: {args_cli.input}")

    # Reset environment
    env.reset()

    # Main loop
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Get input state
                input_state = handler.update()

                # Check for quit
                if input_state.quit_requested:
                    print("[INFO] Quit requested")
                    break

                # Check for reset
                if input_state.reset_requested:
                    print("[INFO] Reset requested")
                    env.reset()
                    continue

                # Check for mode switch
                if input_state.mode_switch is not None and input_state.mode_switch != handler.current_mode:
                    print(f"[INFO] Switching to {input_state.mode_switch.upper()} mode")
                    handler.current_mode = input_state.mode_switch
                    env.unwrapped.set_control_mode(input_state.mode_switch)

                # Convert input to action tensor
                action = input_state.to_action_tensor(env.unwrapped.device)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)

                # Auto-reset if terminated (safety limit reached)
                if terminated.any():
                    print("[INFO] Safety limit reached, auto-resetting")
                    env.reset()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        handler.cleanup()
        env.close()
        print("[INFO] Teleoperation ended")


if __name__ == "__main__":
    # Run main function
    main()
    # Close sim app
    simulation_app.close()
