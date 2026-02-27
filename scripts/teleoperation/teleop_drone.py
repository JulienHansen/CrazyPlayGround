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


# ── Follow-camera state ───────────────────────────────────────────────────────
# /OmniverseKit_Persp is a read-only built-in prim; we create our own camera.
_FOLLOW_CAM_PATH  = "/World/FollowCamera"
_follow_cam_ready: bool  = False
_smooth_eye       = None   # np.ndarray | None  — smoothed camera position
_smooth_target    = None   # np.ndarray | None  — smoothed look-at point

# Smoothing factor (per simulation step at ~100 Hz).
# Lower = more lag (smoother), higher = snappier.
# 0.12 gives ~80 ms time-constant — similar to Liftoff's follow cam.
_CAM_ALPHA = 0.12


def _ensure_follow_camera() -> bool:
    """Create /World/FollowCamera once and switch the active viewport to it."""
    global _follow_cam_ready
    if _follow_cam_ready:
        return True
    try:
        import omni.usd
        import omni.kit.viewport.utility
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(_FOLLOW_CAM_PATH).IsValid():
            cam = UsdGeom.Camera.Define(stage, _FOLLOW_CAM_PATH)
            cam.CreateFocalLengthAttr(18.147)        # ~70° horizontal FOV
            cam.CreateHorizontalApertureAttr(24.0)

        vp = omni.kit.viewport.utility.get_active_viewport()
        if vp is None:
            return False
        vp.set_active_camera(_FOLLOW_CAM_PATH)
        _follow_cam_ready = True
        return True
    except Exception:
        return False


def _update_follow_camera(env, offset: tuple = (-3.0, 0.0, 0.8)) -> None:
    """Liftoff-style smooth follow camera.

    - Horizon stays level (world Z-up, no camera roll or pitch tilt).
    - Camera position smoothly lags behind the drone (exponential filter).
    - Only yaw is tracked for the camera's orbital position; the drone's
      tilt is visible in the frame because the *drone body* tilts, not the cam.
    - Wide FOV and close distance give a good sense of speed, like Liftoff.

    Args:
        env:    Unwrapped teleoperation environment.
        offset: (back, right, up) in the drone's yaw frame [m].
    """
    global _smooth_eye, _smooth_target

    if not _ensure_follow_camera():
        return

    try:
        import omni.usd
        from pxr import Gf, UsdGeom
    except ImportError:
        return

    import numpy as np

    pos  = env._robot.data.root_pos_w[0].cpu().numpy().astype(float)
    quat = env._robot.data.root_quat_w[0].cpu().numpy().astype(float)  # [w,x,y,z]
    qw, qx, qy, qz = quat

    # Extract yaw only — camera orbits horizontally around the drone
    yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))

    # Desired camera eye: offset rotated by yaw only (horizon stays level)
    ox, oy, oz = offset
    desired_eye = np.array([
        pos[0] + cy*ox - sy*oy,
        pos[1] + sy*ox + cy*oy,
        pos[2] + oz,
    ])
    desired_target = pos.copy()

    # Exponential smoothing — gives the "camera lags behind" Liftoff feel
    if _smooth_eye is None:
        _smooth_eye    = desired_eye.copy()
        _smooth_target = desired_target.copy()
    else:
        _smooth_eye    = _CAM_ALPHA * desired_eye    + (1.0 - _CAM_ALPHA) * _smooth_eye
        _smooth_target = _CAM_ALPHA * desired_target + (1.0 - _CAM_ALPHA) * _smooth_target

    # Build orthonormal camera frame with world Z-up (no roll, no pitch in view)
    forward = _smooth_target - _smooth_eye
    fwd_len = np.linalg.norm(forward)
    if fwd_len < 1e-6:
        return
    forward /= fwd_len

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right_len = np.linalg.norm(right)
    if right_len < 1e-6:
        return
    right /= right_len

    up   = np.cross(right, forward)
    up  /= np.linalg.norm(up) + 1e-9
    back = -forward

    # USD 4×4 row-vector transform
    gf_mat = Gf.Matrix4d(
        right[0], right[1], right[2], 0.0,
        up[0],    up[1],    up[2],    0.0,
        back[0],  back[1],  back[2],  0.0,
        _smooth_eye[0], _smooth_eye[1], _smooth_eye[2], 1.0,
    )

    try:
        stage  = omni.usd.get_context().get_stage()
        prim   = stage.GetPrimAtPath(_FOLLOW_CAM_PATH)
        if prim.IsValid():
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(gf_mat)
    except Exception:
        pass


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

    # When the primary handler is a gamepad, also init a secondary keyboard
    # handler so that R (reset) and Escape (quit) always work from the keyboard
    # regardless of which device is driving the drone.
    kb_hotkeys: KeyboardHandler | None = None
    if not isinstance(handler, KeyboardHandler):
        kb_hotkeys = KeyboardHandler(env.unwrapped.device)
        if not kb_hotkeys.initialize():
            kb_hotkeys = None
        else:
            print("[Teleop] Secondary keyboard hotkeys active (R=reset, Esc=quit)")

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
                # Get input state from primary handler
                input_state = handler.update()

                # Merge hotkeys from secondary keyboard handler (if active)
                if kb_hotkeys is not None:
                    kb_state = kb_hotkeys.update()
                    if kb_state.reset_requested:
                        input_state.reset_requested = True
                    if kb_state.quit_requested:
                        input_state.quit_requested = True

                # Check for quit
                if input_state.quit_requested:
                    print("[INFO] Quit requested")
                    break

                # Check for reset
                if input_state.reset_requested:
                    print("[INFO] Reset requested")
                    env.reset()
                    handler.reset()
                    # Clear camera smoothing so it snaps to the new position
                    global _smooth_eye, _smooth_target
                    _smooth_eye = _smooth_target = None
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

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        if kb_hotkeys is not None:
            kb_hotkeys.cleanup()
        handler.cleanup()
        env.close()
        print("[INFO] Teleoperation ended")


if __name__ == "__main__":
    # Run main function
    main()
    # Close sim app
    simulation_app.close()
