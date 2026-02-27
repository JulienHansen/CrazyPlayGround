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
parser.add_argument(
    "--debug-gamepad",
    action="store_true",
    default=False,
    help="Print raw gamepad axis values every ~0.6 s to identify correct axis mapping.",
)
parser.add_argument(
    "--rate-profile",
    type=str,
    default="none",
    choices=["none", "betaflight", "actual", "kiss", "raceflight"],
    help=(
        "Stick-to-rate profile for attitude mode. "
        "'none' (default) = angle mode: sticks set roll/pitch angle targets. "
        "Any other value = rate mode: sticks feed through the chosen firmware-style "
        "expo curve and set body-rate targets directly (bypasses the angle PID loop)."
    ),
)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

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
    debug = getattr(args_cli, "debug_gamepad", False)

    if input_type == "gamepad":
        handler = GamepadHandler(device, debug=debug)
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
        gamepad = GamepadHandler(device, debug=debug)
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
        print("  Position / Velocity mode:")
        print("    Z/W/Up    : Forward")
        print("    S/Down    : Backward")
        print("    Q/A/Left  : Left")
        print("    D/Right   : Right")
        print("    E         : Up")
        print("    Shift     : Down")
        print("  Attitude mode:")
        print("    Z/W/Up    : Pitch forward")
        print("    S/Down    : Pitch backward")
        print("    Q/A       : Roll left")
        print("    D         : Roll right")
        print("    Left/Right arrows : Yaw")
        print("    E         : Thrust up   (sticky)")
        print("    Shift     : Thrust down (sticky)")
        print("  Mode Switch:")
        print("    1 : Position mode")
        print("    2 : Velocity mode")
        print("    3 : Attitude mode")
        print("  Control:")
        print("    R   : Reset")
        print("    Esc : Quit")

    print("=" * 60 + "\n")


def _update_follow_camera(env, offset: tuple = (-2.0, 0.0, 0.8)) -> None:
    """Move the Isaac Sim viewport camera to follow the drone.

    The camera sits at `offset` metres behind/above the drone in its body frame,
    but only the yaw is tracked — roll and pitch are ignored so the view stays
    stable during aggressive manoeuvres.

    Args:
        env:    The unwrapped teleoperation environment.
        offset: (back, right, up) in the drone's body frame [m].
                Negative X = behind the drone.
    """
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        return

    # Drone state (first env only)
    pos = env._robot.data.root_pos_w[0].cpu()          # [3]
    quat = env._robot.data.root_quat_w[0].cpu()        # [w, x, y, z]

    qw, qx, qy, qz = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()

    # Extract yaw angle only for a smooth follow camera
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    cy, sy = math.cos(yaw), math.sin(yaw)
    ox, oy, oz = offset  # body-frame offset: back, right, up

    # Rotate horizontal offset by yaw, keep vertical as-is
    eye_x = pos[0].item() + cy * ox - sy * oy
    eye_y = pos[1].item() + sy * ox + cy * oy
    eye_z = pos[2].item() + oz

    import numpy as np
    set_camera_view(
        eye=np.array([eye_x, eye_y, eye_z]),
        target=pos.numpy(),
    )


def main():
    """Main teleoperation loop."""
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        "Teleoperation",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.rate_profile = args_cli.rate_profile

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
                    handler.reset()
                    continue

                # Check for mode switch
                if input_state.mode_switch is not None and input_state.mode_switch != handler.current_mode:
                    print(f"[INFO] Switching to {input_state.mode_switch.upper()} mode")
                    handler.current_mode = input_state.mode_switch
                    env.unwrapped.set_control_mode(input_state.mode_switch)
                    # Re-poll so the action this frame already uses the new mode's
                    # thrust logic (avoids a zero-thrust glitch on the switch frame)
                    input_state = handler.update()

                # Convert input to action tensor
                action = input_state.to_action_tensor(env.unwrapped.device)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)

                # Update third-person follow camera
                _update_follow_camera(env.unwrapped)

                # Auto-reset if terminated (safety limit reached)
                if terminated.any():
                    print("[INFO] Safety limit reached, auto-resetting")
                    env.reset()
                    handler.reset()

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
