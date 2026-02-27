# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad input handler using pygame — FPV Mode 2 layout.

Axis layout (AETR — standard for FPV radio controllers on Linux):
    ax0  Aileron  (right stick X) →  Roll            (spring, -=left,  +=right)
    ax1  Elevator (right stick Y) →  Pitch           (spring, -=fwd,   +=back)
    ax2  Throttle (left  stick Y) →  Thrust / Vz     (non-spring, -=bottom, +=top)
    ax3  Rudder   (left  stick X) →  Yaw rate        (spring, -=left,  +=right)

Velocity / Position mode:
    Left  stick Y  : Vz  (up/down)
    Left  stick X  : Yaw rate
    Right stick Y  : Vx  (forward/backward)
    Right stick X  : Vy  (left/right)

Attitude mode:
    Left  stick Y  : Thrust  (non-spring: bottom=min, centre=hover, top=max)
    Left  stick X  : Yaw rate
    Right stick Y  : Pitch
    Right stick X  : Roll

Button mappings (Xbox style):
    A (0)     : Switch to velocity mode
    B (1)     : Switch to attitude mode
    Y (3)     : Switch to position mode
    Start (7) : Reset
    Back  (6) : Quit
"""

from __future__ import annotations

import torch

from .base_handler import BaseInputHandler, ControlMode, InputState


class GamepadHandler(BaseInputHandler):

    DEADZONE = 0.10

    # Hover fraction (must match KeyboardHandler.HOVER_THRUST)
    _HOVER = 1.0 / 1.8
    # Thrust excursion above/below hover when stick is at full deflection
    _THRUST_DELTA = 0.35

    def __init__(self, device: torch.device, debug: bool = False):
        super().__init__(device)
        self._pygame = None
        self._joystick = None
        self._initialized = False
        self._debug = debug
        self._debug_counter = 0
        self._axis_rest: list[float] = []   # per-axis rest offsets

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Initialize pygame and pick the first real gamepad (>= 4 axes)."""
        try:
            import pygame

            self._pygame = pygame
            pygame.init()
            pygame.joystick.init()

            count = pygame.joystick.get_count()
            if count == 0:
                print("[GamepadHandler] No gamepad connected")
                return False

            print(f"[GamepadHandler] Scanning {count} device(s):")
            for i in range(count):
                joy = pygame.joystick.Joystick(i)
                joy.init()
                axes = joy.get_numaxes()
                print(f"  [{i}] {joy.get_name()}  axes={axes}  buttons={joy.get_numbuttons()}")
                if axes >= 4:
                    self._joystick = joy
                    break
                joy.quit()

            if self._joystick is None:
                print("[GamepadHandler] No suitable gamepad found (need >= 4 axes)")
                return False

            print(f"[GamepadHandler] Using: {self._joystick.get_name()}")

            # ── Auto-calibration ─────────────────────────────────────────────
            # Pump several frames so axis values have time to settle, then
            # record the rest position of every axis.  Spring-return sticks
            # (yaw, roll, pitch) should be centred; the throttle (non-spring)
            # can be anywhere — its rest value is used only to decide whether
            # to apply calibration (skipped when |rest| >= 0.7).
            print("[GamepadHandler] Calibrating — centre all SPRING sticks (roll/pitch/yaw) …")
            for _ in range(30):
                pygame.event.pump()
            n = self._joystick.get_numaxes()
            self._axis_rest = [self._joystick.get_axis(i) for i in range(n)]
            print(f"[GamepadHandler] Rest offsets: "
                  f"{' '.join(f'ax{i}={v:+.2f}' for i, v in enumerate(self._axis_rest))}")

            self._initialized = True
            return True

        except ImportError:
            print("[GamepadHandler] pygame not installed.  pip install pygame")
            return False
        except Exception as e:
            print(f"[GamepadHandler] Initialization failed: {e}")
            return False

    # ── Axis helpers ──────────────────────────────────────────────────────────

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.DEADZONE:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.DEADZONE) / (1.0 - self.DEADZONE)

    def _get_axis(self, axis_id: int) -> float:
        """Return calibrated + deadzone-filtered axis value in [-1, 1].

        Spring-return axes (rest ≈ 0) get a rest-offset subtracted so they
        centre precisely at 0.  Non-spring axes (throttle, rest ≈ ±1) are
        passed through as-is so their full [-1, +1] travel is preserved.
        """
        if self._joystick is None or axis_id >= self._joystick.get_numaxes():
            return 0.0
        raw = self._joystick.get_axis(axis_id)
        # Only subtract the rest offset for spring-return sticks (rest near 0).
        # Skip calibration for throttle-style axes whose rest is near ±1
        # (subtracting would shift the range and clip one direction of travel).
        if axis_id < len(self._axis_rest) and abs(self._axis_rest[axis_id]) < 0.7:
            raw -= self._axis_rest[axis_id]
        return self._apply_deadzone(raw)

    def _get_button(self, button_id: int) -> bool:
        if self._joystick is None or button_id >= self._joystick.get_numbuttons():
            return False
        return bool(self._joystick.get_button(button_id))

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self) -> InputState:
        """Poll gamepad and return current input state."""
        state = InputState()

        if not self._initialized or self._joystick is None:
            return state

        self._pygame.event.pump()

        # ── Debug print ───────────────────────────────────────────────────────
        if self._debug:
            self._debug_counter += 1
            if self._debug_counter >= 60:
                self._debug_counter = 0
                n = self._joystick.get_numaxes()
                raw = [f"ax{i}={self._joystick.get_axis(i):+.2f}" for i in range(n)]
                cal = [f"ax{i}={self._get_axis(i):+.2f}" for i in range(n)]
                print(f"[Gamepad RAW] {' | '.join(raw)}")
                print(f"[Gamepad CAL] {' | '.join(cal)}")

        # AETR axes (calibrated + deadzone applied)
        roll_input   =  self._get_axis(0)   # ax0 Aileron  right X:  -=left,  +=right
        pitch_input  = -self._get_axis(1)   # ax1 Elevator right Y:  up=-raw → positive=fwd
        thrust_input =  self._get_axis(2)   # ax2 Throttle left  Y:  -=bottom, +=top
        yaw_input    =  self._get_axis(3)   # ax3 Rudder   left  X:  -=left,  +=right

        if self._current_mode == "velocity":
            state.vx       =  pitch_input   # right Y: forward/backward
            state.vy       = -roll_input    # right X: left/right (inverted)
            state.vz       =  thrust_input  # left  Y: up/down
            state.yaw_rate =  yaw_input     # left  X: yaw

        elif self._current_mode == "attitude":
            state.roll     =  roll_input
            state.pitch    =  pitch_input
            state.yaw_rate =  yaw_input
            # Thrust: spring-centred stick → centre = hover, up = rise, down = descend
            # Matches KeyboardHandler behaviour (no input = hold altitude).
            thrust = self._HOVER + thrust_input * self._THRUST_DELTA
            state.thrust = float(max(0.0, min(1.0, thrust)))

        elif self._current_mode == "position":
            state.vx       =  pitch_input
            state.vy       = -roll_input
            state.vz       =  thrust_input
            state.yaw_rate =  yaw_input

        # ── Buttons (Xbox style) ──────────────────────────────────────────────
        if self._get_button(0):   state.mode_switch = "velocity"
        elif self._get_button(1): state.mode_switch = "attitude"
        elif self._get_button(3): state.mode_switch = "position"

        if self._get_button(7):   state.reset_requested = True
        if self._get_button(6):   state.quit_requested  = True

        return state

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        if self._initialized:
            if self._joystick is not None:
                self._joystick.quit()
            self._pygame.joystick.quit()
            self._pygame.quit()
            self._initialized = False

    def is_available(self) -> bool:
        return self._initialized and self._joystick is not None
