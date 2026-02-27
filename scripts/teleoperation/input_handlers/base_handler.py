# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for input handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch


ControlMode = Literal["position", "velocity", "attitude"]


@dataclass
class InputState:
    """State of input at a given moment.

    Action format (7D): [vx, vy, vz, roll, pitch, yaw_rate, thrust]
    """

    # Movement axes (used for position/velocity modes)
    vx: float = 0.0  # Forward/backward
    vy: float = 0.0  # Left/right
    vz: float = 0.0  # Up/down

    # Attitude axes (used for attitude mode)
    roll: float = 0.0
    pitch: float = 0.0
    yaw_rate: float = 0.0
    thrust: float = 0.0  # -1 to 1, will be mapped to thrust range

    # Control signals
    mode_switch: Optional[ControlMode] = None
    reset_requested: bool = False
    quit_requested: bool = False

    def to_action_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert input state to 7D action tensor."""
        return torch.tensor(
            [[self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw_rate, self.thrust]],
            dtype=torch.float32,
            device=device,
        )


class BaseInputHandler(ABC):
    """Abstract base class for input handlers."""

    def __init__(self, device: torch.device):
        """Initialize the input handler.

        Args:
            device: Torch device for action tensors.
        """
        self.device = device
        self._current_mode: ControlMode = "velocity"

    @property
    def current_mode(self) -> ControlMode:
        """Get current control mode."""
        return self._current_mode

    @current_mode.setter
    def current_mode(self, mode: ControlMode) -> None:
        """Set current control mode."""
        self._current_mode = mode

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the input handler.

        Returns:
            True if initialization successful, False otherwise.
        """
        pass

    @abstractmethod
    def update(self) -> InputState:
        """Poll input devices and return current state.

        Returns:
            InputState with current input values and control signals.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this input method is available.

        Returns:
            True if the input device is connected/available.
        """
        pass

    def get_mode_display_name(self) -> str:
        """Get display name for current mode."""
        mode_names = {
            "position": "Position (ZQSD)",
            "velocity": "Velocity (Stick)",
            "attitude": "Attitude (Stick+Trigger)",
        }
        return mode_names.get(self._current_mode, self._current_mode)
