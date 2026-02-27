# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad input handler using pygame."""

from __future__ import annotations

import torch

from .base_handler import BaseInputHandler, ControlMode, InputState


class GamepadHandler(BaseInputHandler):
    """Gamepad input handler using pygame.

    Stick mappings:

    Velocity mode:
        Left stick Y  : Vx (forward/backward)
        Left stick X  : Vy (left/right)
        Right stick Y : Vz (up/down)
        Right stick X : Yaw rate

    Attitude mode:
        Left stick X  : Roll
        Left stick Y  : Pitch
        Right stick X : Yaw rate
        RT - LT       : Thrust

    Button mappings:
        A (0)     : Switch to velocity mode
        B (1)     : Switch to attitude mode
        Y (3)     : Switch to position mode
        Start (7) : Reset
        Back (6)  : Quit
    """

    # Deadzone for analog sticks
    DEADZONE = 0.15

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._pygame = None
        self._joystick = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize pygame and gamepad."""
        try:
            import pygame

            self._pygame = pygame

            # Initialize pygame joystick module
            pygame.init()
            pygame.joystick.init()

            # Check for connected joysticks
            if pygame.joystick.get_count() == 0:
                print("[GamepadHandler] No gamepad connected")
                return False

            # Use first joystick
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()

            print(f"[GamepadHandler] Initialized: {self._joystick.get_name()}")
            print(f"  Axes: {self._joystick.get_numaxes()}")
            print(f"  Buttons: {self._joystick.get_numbuttons()}")
            print(f"  Hats: {self._joystick.get_numhats()}")

            self._initialized = True
            return True

        except ImportError:
            print("[GamepadHandler] pygame not installed. Install with: pip install pygame")
            return False
        except Exception as e:
            print(f"[GamepadHandler] Initialization failed: {e}")
            return False

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to analog input."""
        if abs(value) < self.DEADZONE:
            return 0.0
        # Remap value to full range after deadzone
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.DEADZONE) / (1.0 - self.DEADZONE)

    def _get_axis(self, axis_id: int) -> float:
        """Get axis value with deadzone applied."""
        if self._joystick is None or axis_id >= self._joystick.get_numaxes():
            return 0.0
        return self._apply_deadzone(self._joystick.get_axis(axis_id))

    def _get_button(self, button_id: int) -> bool:
        """Get button state."""
        if self._joystick is None or button_id >= self._joystick.get_numbuttons():
            return False
        return self._joystick.get_button(button_id)

    def _get_trigger_value(self) -> float:
        """Get combined trigger value (RT - LT).

        Common axis mappings:
        - Xbox: LT = axis 4, RT = axis 5 (0 to 1 range on some, -1 to 1 on others)
        - PS4: L2 = axis 4, R2 = axis 5

        Returns value in range [-1, 1] where positive is RT, negative is LT.
        """
        if self._joystick is None:
            return 0.0

        num_axes = self._joystick.get_numaxes()

        # Try standard trigger axes (4 and 5)
        if num_axes >= 6:
            lt = self._joystick.get_axis(4)  # Left trigger
            rt = self._joystick.get_axis(5)  # Right trigger

            # Some controllers report triggers as -1 to 1 (need to remap to 0 to 1)
            # Others report as 0 to 1 directly
            # Handle both cases by checking the idle state
            # If idle value is around -1, remap; if around 0, use as-is
            if lt < -0.5:  # Likely -1 to 1 range
                lt = (lt + 1.0) / 2.0
            if rt < -0.5:
                rt = (rt + 1.0) / 2.0

            return rt - lt

        return 0.0

    def update(self) -> InputState:
        """Poll gamepad and return current input state."""
        state = InputState()

        if not self._initialized or self._joystick is None:
            return state

        # Process pygame events to update joystick state
        self._pygame.event.pump()

        # Standard axis mapping:
        # Axis 0: Left stick X
        # Axis 1: Left stick Y (inverted: push up = negative)
        # Axis 2: Right stick X
        # Axis 3: Right stick Y (inverted)
        # Axis 4: Left trigger
        # Axis 5: Right trigger

        left_x = self._get_axis(0)
        left_y = -self._get_axis(1)  # Invert Y axis
        right_x = self._get_axis(2)
        right_y = -self._get_axis(3)  # Invert Y axis

        if self._current_mode == "velocity":
            # Velocity mode
            state.vx = left_y   # Left stick Y = forward/backward
            state.vy = -left_x  # Left stick X = left/right (inverted for intuitive control)
            state.vz = right_y  # Right stick Y = up/down
            state.yaw_rate = right_x

        elif self._current_mode == "attitude":
            # Attitude mode
            state.roll = left_x     # Left stick X = roll
            state.pitch = left_y    # Left stick Y = pitch
            state.yaw_rate = right_x
            state.thrust = self._get_trigger_value()  # RT - LT

        elif self._current_mode == "position":
            # Position mode (same as velocity, but interpreted as position delta)
            state.vx = left_y
            state.vy = -left_x
            state.vz = right_y
            state.yaw_rate = right_x

        # Button mappings (Xbox style)
        # A=0, B=1, X=2, Y=3, LB=4, RB=5, Back=6, Start=7
        if self._get_button(0):  # A - Velocity mode
            state.mode_switch = "velocity"
        elif self._get_button(1):  # B - Attitude mode
            state.mode_switch = "attitude"
        elif self._get_button(3):  # Y - Position mode
            state.mode_switch = "position"

        if self._get_button(7):  # Start - Reset
            state.reset_requested = True
        if self._get_button(6):  # Back - Quit
            state.quit_requested = True

        return state

    def cleanup(self) -> None:
        """Clean up pygame resources."""
        if self._initialized:
            if self._joystick is not None:
                self._joystick.quit()
            self._pygame.joystick.quit()
            self._pygame.quit()
            self._initialized = False

    def is_available(self) -> bool:
        """Check if gamepad is available."""
        return self._initialized and self._joystick is not None
