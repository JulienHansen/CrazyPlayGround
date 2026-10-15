# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Patch skrl's TensorBoard writer to also forward scalars to Weights & Biases.

skrl.utils.tensorboard.SummaryWriter writes events via the low-level
tensorboard EventFileWriter directly instead of torch.utils.tensorboard.SummaryWriter.
wandb's ``sync_tensorboard=True`` only patches the latter, so scalars logged through
skrl's writer (e.g. the ``Episode_Reward/*`` extras) never reach W&B even though they
show up correctly in local TensorBoard logs. This patches skrl's writer to call
``wandb.log`` directly whenever a W&B run is active.
"""

try:
    from skrl.utils.tensorboard import SummaryWriter as _SkrlSummaryWriter
except ImportError:
    _SkrlSummaryWriter = None

if _SkrlSummaryWriter is not None and not getattr(_SkrlSummaryWriter, "_wandb_patched", False):
    _original_add_scalar = _SkrlSummaryWriter.add_scalar

    def _add_scalar_with_wandb(self, *, tag: str, value: float, timestep: int) -> None:
        _original_add_scalar(self, tag=tag, value=value, timestep=timestep)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({tag: value}, step=timestep)
        except ImportError:
            pass

    _SkrlSummaryWriter.add_scalar = _add_scalar_with_wandb
    _SkrlSummaryWriter._wandb_patched = True
