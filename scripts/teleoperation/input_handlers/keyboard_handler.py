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

    Position / velocity mode:
        Z / W / Up    : Forward (+X)
        S / Down      : Backward (-X)
        Q / A / Left  : Left (+Y)
        D / Right     : Right (-Y)
        E / Page Up   : Up (+Z)
        Left Shift / Page Down : Down (-Z)

    Attitude mode (keys are reused with mode-specific meaning):
        Z / W / Up    : Pitch forward
        S / Down      : Pitch backward
        Q / A         : Roll left
        D             : Roll right
        Left Arrow    : Yaw left
        Right Arrow   : Yaw right
        E             : Increase thrust (sticky)
        Left Shift    : Decrease thrust (sticky)

    Mode switching:
        1 : Position mode
        2 : Velocity mode
        3 : Attitude mode

    Control:
        R      : Reset
        Escape : Quit
    """

    # Hover fraction: 1 / max_thrust_scale (= 1/1.8 ≈ 0.556).
    HOVER_THRUST: float = 1.0 / 1.8

    # Thrust offset applied while E or Shift is held.
    # E → HOVER + THRUST_DELTA (drone rises)
    # Shift → HOVER - THRUST_DELTA (drone descends)
    # Neither → HOVER (drone holds altitude)
    THRUST_DELTA: float = 0.2

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._carb_input = None
        self._keyboard = None
        self._input_interface = None
        self._keyboard_sub = None
        self._pressed_keys: Set[str] = set()

    # ── Mode setter override (resets sticky thrust on entering attitude) ──────

    @BaseInputHandler.current_mode.setter
    def current_mode(self, mode: ControlMode) -> None:
        self._current_mode = mode

    def reset(self) -> None:
        """No persistent state to reset for keyboard handler."""
        pass

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Initialize keyboard input using carb.input."""
        try:
            import carb.input
            import omni.appwindow

            self._carb_input = carb.input
            self._input_interface = carb.input.acquire_input_interface()

            # get_keyboard() lives on the app window, not on the input interface
            appwindow = omni.appwindow.get_default_app_window()
            self._keyboard = appwindow.get_keyboard()
            if self._keyboard is None:
                print("[KeyboardHandler] No keyboard found")
                return False

            # Subscribe to keyboard events for reliable key-state tracking
            self._keyboard_sub = self._input_interface.subscribe_to_keyboard_events(
                self._keyboard, self._on_keyboard_event
            )

            print("[KeyboardHandler] Initialized successfully")
            return True

        except ImportError:
            print("[KeyboardHandler] carb.input not available (not running in Isaac Sim)")
            return False
        except Exception as e:
            print(f"[KeyboardHandler] Initialization failed: {e}")
            return False

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Track key-press/release events to maintain the set of held keys."""
        import carb.input

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed_keys.add(event.input.name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed_keys.discard(event.input.name)
        return True

    def _is_key_pressed(self, key_name: str) -> bool:
        """Check if a key is currently held down.

        Args:
            key_name: carb.input.KeyboardInput enum member name,
                      e.g. "W", "SPACE", "LEFT_SHIFT", "UP", "KEY_1".
        """
        return key_name in self._pressed_keys

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self) -> InputState:
        """Poll keyboard and return current input state."""
        state = InputState()

        if self._input_interface is None:
            return state

        if self._current_mode == "attitude":
            self._update_attitude(state)
        else:
            self._update_position_velocity(state)

        # Mode switching and control signals are always active
        if self._is_key_pressed("KEY_1"):
            state.mode_switch = "position"
        elif self._is_key_pressed("KEY_2"):
            state.mode_switch = "velocity"
        elif self._is_key_pressed("KEY_3"):
            state.mode_switch = "attitude"

        if self._is_key_pressed("R"):
            state.reset_requested = True
        if self._is_key_pressed("ESCAPE"):
            state.quit_requested = True

        return state

    def _update_position_velocity(self, state: InputState) -> None:
        """Fill movement axes for position / velocity modes."""
        # Forward / backward (X)
        if self._is_key_pressed("Z") or self._is_key_pressed("W") or self._is_key_pressed("UP"):
            state.vx = 1.0
        elif self._is_key_pressed("S") or self._is_key_pressed("DOWN"):
            state.vx = -1.0

        # Left / right (Y)
        if self._is_key_pressed("Q") or self._is_key_pressed("A") or self._is_key_pressed("LEFT"):
            state.vy = 1.0
        elif self._is_key_pressed("D") or self._is_key_pressed("RIGHT"):
            state.vy = -1.0

        # Up / down (Z)
        if self._is_key_pressed("E") or self._is_key_pressed("PAGE_UP"):
            state.vz = 1.0
        elif self._is_key_pressed("LEFT_SHIFT") or self._is_key_pressed("PAGE_DOWN"):
            state.vz = -1.0

    def _update_attitude(self, state: InputState) -> None:
        """Fill attitude axes and update sticky thrust for attitude mode."""
        # Pitch (Z/W/Up = forward, S/Down = backward)
        if self._is_key_pressed("Z") or self._is_key_pressed("W") or self._is_key_pressed("UP"):
            state.pitch = 1.0
        elif self._is_key_pressed("S") or self._is_key_pressed("DOWN"):
            state.pitch = -1.0

        # Roll (Q/A = left, D = right)
        if self._is_key_pressed("Q") or self._is_key_pressed("A"):
            state.roll = -1.0
        elif self._is_key_pressed("D"):
            state.roll = 1.0

        # Yaw rate (arrow keys only, freed from left/right movement)
        if self._is_key_pressed("LEFT"):
            state.yaw_rate = -1.0
        elif self._is_key_pressed("RIGHT"):
            state.yaw_rate = 1.0

        # Thrust: momentary control relative to hover.
        # E held → rise, Shift held → descend, neither → hold altitude.
        if self._is_key_pressed("E"):
            state.thrust = min(1.0, self.HOVER_THRUST + self.THRUST_DELTA)
        elif self._is_key_pressed("LEFT_SHIFT"):
            state.thrust = max(0.0, self.HOVER_THRUST - self.THRUST_DELTA)
        else:
            state.thrust = self.HOVER_THRUST

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Clean up keyboard handler."""
        if (
            self._input_interface is not None
            and self._keyboard is not None
            and self._keyboard_sub is not None
        ):
            self._input_interface.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None
        self._input_interface = None
        self._keyboard = None
        self._carb_input = None
        self._pressed_keys.clear()

    def is_available(self) -> bool:
        """Check if keyboard is available."""
        return self._input_interface is not None and self._keyboard is not None
