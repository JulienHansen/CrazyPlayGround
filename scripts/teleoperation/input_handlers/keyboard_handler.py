# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard input handler using carb.input (Isaac Sim)."""

from __future__ import annotations

from typing import Set

import torch

from .base_handler import BaseInputHandler, ControlMode, InputState


class KeyboardHandler(BaseInputHandler):
    """Keyboard input handler using carb.input for Isaac Sim.

    Key mappings:
        Position mode (ZQSD/Arrows):
            Z / Up Arrow    : Forward (+X)
            S / Down Arrow  : Backward (-X)
            Q / Left Arrow  : Left (+Y)
            D / Right Arrow : Right (-Y)
            Space / Page Up : Up (+Z)
            Left Shift / Page Down : Down (-Z)

        Mode switching:
            1 : Position mode
            2 : Velocity mode
            3 : Attitude mode

        Control:
            R : Reset
            Escape : Quit
    """

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._carb_input = None
        self._keyboard = None
        self._input_interface = None
        self._pressed_keys: Set[int] = set()
        self._key_codes: dict = {}

    def initialize(self) -> bool:
        """Initialize keyboard input using carb.input."""
        try:
            import carb.input

            self._carb_input = carb.input
            self._input_interface = carb.input.acquire_input_interface()

            # Get keyboard device
            self._keyboard = self._input_interface.get_keyboard()
            if self._keyboard is None:
                print("[KeyboardHandler] No keyboard found")
                return False

            # Cache key codes for faster lookup
            self._key_codes = {
                "W": carb.input.KeyboardInput.W,
                "Z": carb.input.KeyboardInput.Z,
                "S": carb.input.KeyboardInput.S,
                "A": carb.input.KeyboardInput.A,
                "Q": carb.input.KeyboardInput.Q,
                "D": carb.input.KeyboardInput.D,
                "UP": carb.input.KeyboardInput.UP,
                "DOWN": carb.input.KeyboardInput.DOWN,
                "LEFT": carb.input.KeyboardInput.LEFT,
                "RIGHT": carb.input.KeyboardInput.RIGHT,
                "SPACE": carb.input.KeyboardInput.SPACE,
                "LEFT_SHIFT": carb.input.KeyboardInput.LEFT_SHIFT,
                "PAGE_UP": carb.input.KeyboardInput.PAGE_UP,
                "PAGE_DOWN": carb.input.KeyboardInput.PAGE_DOWN,
                "1": carb.input.KeyboardInput.KEY_1,
                "2": carb.input.KeyboardInput.KEY_2,
                "3": carb.input.KeyboardInput.KEY_3,
                "R": carb.input.KeyboardInput.R,
                "ESCAPE": carb.input.KeyboardInput.ESCAPE,
            }

            print("[KeyboardHandler] Initialized successfully")
            return True

        except ImportError:
            print("[KeyboardHandler] carb.input not available (not running in Isaac Sim)")
            return False
        except Exception as e:
            print(f"[KeyboardHandler] Initialization failed: {e}")
            return False

    def _is_key_pressed(self, key_name: str) -> bool:
        """Check if a key is currently pressed."""
        if self._input_interface is None or self._keyboard is None:
            return False

        key_code = self._key_codes.get(key_name)
        if key_code is None:
            return False

        return self._input_interface.get_keyboard_value(self._keyboard, key_code)

    def update(self) -> InputState:
        """Poll keyboard and return current input state."""
        state = InputState()

        if self._input_interface is None:
            return state

        # Movement keys (ZQSD/Arrows for position mode)
        # Forward/backward (X axis)
        if self._is_key_pressed("Z") or self._is_key_pressed("W") or self._is_key_pressed("UP"):
            state.vx = 1.0
        elif self._is_key_pressed("S") or self._is_key_pressed("DOWN"):
            state.vx = -1.0

        # Left/right (Y axis)
        if self._is_key_pressed("Q") or self._is_key_pressed("A") or self._is_key_pressed("LEFT"):
            state.vy = 1.0
        elif self._is_key_pressed("D") or self._is_key_pressed("RIGHT"):
            state.vy = -1.0

        # Up/down (Z axis)
        if self._is_key_pressed("SPACE") or self._is_key_pressed("PAGE_UP"):
            state.vz = 1.0
        elif self._is_key_pressed("LEFT_SHIFT") or self._is_key_pressed("PAGE_DOWN"):
            state.vz = -1.0

        # Mode switching
        if self._is_key_pressed("1"):
            state.mode_switch = "position"
        elif self._is_key_pressed("2"):
            state.mode_switch = "velocity"
        elif self._is_key_pressed("3"):
            state.mode_switch = "attitude"

        # Control
        if self._is_key_pressed("R"):
            state.reset_requested = True
        if self._is_key_pressed("ESCAPE"):
            state.quit_requested = True

        return state

    def cleanup(self) -> None:
        """Clean up keyboard handler."""
        self._input_interface = None
        self._keyboard = None
        self._carb_input = None

    def is_available(self) -> bool:
        """Check if keyboard is available."""
        return self._input_interface is not None and self._keyboard is not None
