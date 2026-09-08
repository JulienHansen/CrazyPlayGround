"""V(s, h) history-state critic with critic-side CNN+GRU.

The critic has its **own** CNN encoder and GRU, separate from the actor's.
It processes the same observation-action stream (o, a) as the actor, but
with independent weights. The GRU output is concatenated with the privileged
state vector and fed through an MLP to produce the value estimate.

This is the Baisero-Amato / Lyu history-state critic baseline. Unlike V(s,z)
which uses stop-gradient on the actor's hidden state, this critic maintains
its own recurrence and receives full gradient flow through its CNN+GRU.

The model exposes RNN specification metadata so that the PPO_RNN_SH agent
can manage the critic's hidden state (reset on episode boundaries, BPTT
over sequences, etc.).
"""

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import gymnasium
import numpy as np
from gymnasium import spaces

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, Model


class HistoryStateCriticModel(DeterministicMixin, Model):
    """V(s, h) critic: value = MLP(concat(state, GRU(CNN(o), a))).

    The critic maintains its own CNN+GRU pipeline over the learner's
    observation-action stream. The GRU hidden state encodes the critic's
    own summary of the history h = (o_0, a_0, ..., o_t).

    Parameters
    ----------
    observation_space : gymnasium.Space
        The privileged state space (e.g. 35-dim flat Box).
    action_space : gymnasium.Space
        The action space (used to determine action_dim for GRU input).
    image_space : gymnasium.Space or tuple
        Shape of the image observation (C, H, W). Must be provided so the
        critic CNN can be sized correctly.
    past_actions_size : int
        Number of past action dimensions concatenated with CNN features
        before the GRU (same as actor's obs_num_past_actions * action_dim).
    rnn : dict
        RNN config: hidden_size, num_layers, sequence_length.
    num_envs : int
        Number of parallel environments (for hidden state sizing).
    cnn_feature_size : int
        Output size of the CNN linear projection.
    layers : tuple of int
        MLP head layer sizes after concat(state, gru_output).
    """

    def __init__(
        self,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        clip_actions: bool = False,
        image_channels: int = 2,
        image_height: int = 64,
        image_width: int = 64,
        past_actions_size: int = 4,
        rnn: Optional[Mapping[str, Any]] = None,
        num_envs: int = 1,
        cnn_feature_size: int = 128,
        layers: Sequence[int] = (256, 128, 64),
        opp_branch: bool = False,
        opp_image_channels: Optional[int] = None,
        opp_past_actions_size: Optional[int] = None,
        opp_id_dim: int = 0,
        opp_id_num: int = 0,
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self._past_actions_size = past_actions_size
        self._cnn_feature_size = cnn_feature_size
        self._opp_branch = opp_branch
        self._opp_id_dim = opp_id_dim
        # Learned embedding e(k_t) of the opponent-pool identifier (paper
        # Prop. 2, joint history-state critic). For SHH this mirrors the
        # SZZ wiring in vsh_critic.py — same Embedding(opp_id_num, opp_id_dim)
        # concatenated into the MLP head.
        self.opp_id_embedding: Optional[nn.Embedding] = None
        if opp_id_dim > 0:
            if opp_id_num <= 0:
                raise ValueError(
                    f"opp_id_dim={opp_id_dim} requires opp_id_num > 0 (got {opp_id_num})."
                )
            self.opp_id_embedding = nn.Embedding(opp_id_num, opp_id_dim)

        # RNN configuration
        rnn_cfg = rnn or {}
        self._rnn_hidden_size = int(rnn_cfg.get("hidden_size", 256))
        self._rnn_num_layers = int(rnn_cfg.get("num_layers", 1))
        self._rnn_sequence_length = int(rnn_cfg.get("sequence_length", 16))
        self._rnn_num_envs = max(int(num_envs), 1)

        # --- Learner branch: CNN + GRU ---
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.cnn_linear = nn.Sequential(
            nn.LazyLinear(cnn_feature_size),
            nn.ELU(),
        )

        gru_input_size = cnn_feature_size + past_actions_size
        self.input_ln = nn.LayerNorm(gru_input_size)

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._rnn_num_layers,
            batch_first=True,
        )
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # --- Opponent branch: separate CNN + GRU (only for V(s,h,h^opp)) ---
        if opp_branch:
            opp_ch = opp_image_channels if opp_image_channels is not None else image_channels
            opp_pa = opp_past_actions_size if opp_past_actions_size is not None else past_actions_size
            self._opp_past_actions_size = opp_pa
            self._opp_image_shape = (opp_ch, image_height, image_width)

            self.opp_cnn = nn.Sequential(
                nn.Conv2d(opp_ch, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.opp_cnn_linear = nn.Sequential(
                nn.LazyLinear(cnn_feature_size),
                nn.ELU(),
            )
            opp_gru_input_size = cnn_feature_size + opp_pa
            self.opp_input_ln = nn.LayerNorm(opp_gru_input_size)
            self.opp_gru = nn.GRU(
                input_size=opp_gru_input_size,
                hidden_size=self._rnn_hidden_size,
                num_layers=self._rnn_num_layers,
                batch_first=True,
            )
            for name, param in self.opp_gru.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        # MLP head: concat(state, gru_learner [, gru_opp] [, e(k_t)]) -> value
        state_size = self.num_observations
        mlp_input_size = state_size + self._rnn_hidden_size
        if opp_branch:
            mlp_input_size += self._rnn_hidden_size  # opponent GRU output
        if opp_id_dim > 0:
            mlp_input_size += opp_id_dim

        modules: list[nn.Module] = []
        in_dim = mlp_input_size
        for out_dim in layers:
            modules.extend([nn.Linear(in_dim, out_dim), nn.ELU()])
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*modules)

        # Store image shape for unflattening
        self._image_shape = (image_channels, image_height, image_width)
        self._image_size = image_channels * image_height * image_width

    def get_specification(self) -> Mapping[str, Any]:
        sizes = [(self._rnn_num_layers, self._rnn_num_envs, self._rnn_hidden_size)]
        if self._opp_branch:
            sizes.append((self._rnn_num_layers, self._rnn_num_envs, self._rnn_hidden_size))
        return {
            "rnn": {
                "sizes": sizes,
                "sequence_length": self._rnn_sequence_length,
            }
        }

    def _run_gru_branch(
        self,
        gru: nn.GRU,
        x: torch.Tensor,
        h0: torch.Tensor,
        has_seq: bool,
        seq_len: int,
        terminated: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a GRU branch with BPTT and mid-sequence done resets.

        Returns (features, h_final) where features is (batch*seq, hidden) or (batch, hidden).
        """
        batch_size = x.shape[0]
        if has_seq and seq_len > 1:
            outputs = []
            h = h0
            for t in range(seq_len):
                if terminated is not None and t > 0:
                    done_prev = terminated[:, t - 1]
                    if done_prev.dim() > 1:
                        done_prev = done_prev[..., 0]
                    done_prev = done_prev.bool()
                    if done_prev.any():
                        h = h.clone()
                        h[:, done_prev] = 0.0
                out_t, h = gru(x[:, t : t + 1, :], h)
                outputs.append(out_t)
            rnn_out = torch.cat(outputs, dim=1)
            features = rnn_out.reshape(batch_size * seq_len, -1)
        else:
            if terminated is not None:
                done = terminated
                if done.dim() > 1:
                    done = done[..., 0]
                if done.dim() > 1:
                    done = done[:, -1]
                done = done.reshape(-1).bool()
                if done.any():
                    h0 = h0.clone()
                    h0[:, done] = 0.0
            rnn_out, h = gru(x, h0)
            features = rnn_out[:, -1, :]
        return features, h

    def compute(self, inputs: Mapping[str, Any], role: str = ""):
        """Forward pass.

        Expected inputs:
        - "states": preprocessed privileged state (batch, state_dim) or (batch, seq, state_dim)
        - "critic_image": image tensor (batch, C, H, W) or (batch, seq, C, H, W)
        - "critic_past_actions": past actions (batch, A) or (batch, seq, A)
        - "rnn": list of hidden state tensors [h_learner, h_opp?] for critic GRUs
        - "terminated": done flags for mid-sequence hidden state resets
        - "opp_image": opponent image (batch, C, H, W) or (batch, seq, C, H, W) [only V(s,h,h^opp)]
        - "opp_prev_action": opponent prev action (batch, A) or (batch, seq, A) [only V(s,h,h^opp)]
        """
        states = inputs.get("states")
        critic_image = inputs.get("critic_image")
        critic_past_actions = inputs.get("critic_past_actions")

        # Handle sequence dimension for BPTT training
        has_seq = states.dim() == 3
        if has_seq:
            batch_size, seq_len, state_dim = states.shape
        else:
            batch_size = states.shape[0]
            seq_len = 1
            state_dim = states.shape[-1]

        # If no image provided, use zeros (fallback for initial eval)
        if critic_image is None:
            flat_batch = batch_size * seq_len if has_seq else batch_size
            critic_image = torch.zeros(
                flat_batch, *self._image_shape,
                device=states.device, dtype=states.dtype,
            )
        if critic_past_actions is None:
            flat_batch = batch_size * seq_len if has_seq else batch_size
            critic_past_actions = torch.zeros(
                flat_batch, self._past_actions_size,
                device=states.device, dtype=states.dtype,
            )

        # Flatten seq dim for CNN
        if has_seq and critic_image.dim() == 5:
            critic_image = critic_image.reshape(batch_size * seq_len, *self._image_shape)
        if has_seq and critic_past_actions.dim() == 3:
            critic_past_actions = critic_past_actions.reshape(batch_size * seq_len, -1)

        # CNN forward (learner branch)
        cnn_out = self.cnn(critic_image)
        cnn_features = self.cnn_linear(cnn_out)

        if self._past_actions_size > 0:
            features = torch.cat([cnn_features, critic_past_actions], dim=-1)
        else:
            features = cnn_features
        features = self.input_ln(features)

        # Reshape for GRU: (batch, seq, gru_input_size)
        x = features.reshape(batch_size, seq_len, -1)

        # --- RNN hidden state handling ---
        rnn_states = inputs.get("rnn", None)
        if isinstance(rnn_states, torch.Tensor):
            rnn_states = [rnn_states]
        if rnn_states and len(rnn_states) > 0 and x.dim() == 3:
            h_batch = rnn_states[0].shape[1]
            if x.shape[0] != h_batch and x.shape[1] == h_batch:
                x = x.transpose(0, 1)
        if not rnn_states or len(rnn_states) == 0:
            h0 = torch.zeros(
                self._rnn_num_layers, x.shape[0], self._rnn_hidden_size,
                device=x.device, dtype=x.dtype,
            )
        else:
            h0 = rnn_states[0]
            if h0.dim() == 2:
                h0 = h0.unsqueeze(0)
            h0 = h0.contiguous()

        terminated = inputs.get("terminated", None)
        gru_features, h_learner = self._run_gru_branch(self.gru, x, h0, has_seq, seq_len, terminated)

        # --- Opponent branch (V(s,h,h^opp)) ---
        rnn_outputs = [h_learner]
        opp_gru_features = None
        if self._opp_branch:
            opp_image = inputs.get("opp_image")
            opp_prev_action = inputs.get("opp_prev_action")

            if opp_image is None:
                flat_batch = batch_size * seq_len if has_seq else batch_size
                opp_image = torch.zeros(
                    flat_batch, *self._opp_image_shape,
                    device=states.device, dtype=states.dtype,
                )
            if opp_prev_action is None:
                flat_batch = batch_size * seq_len if has_seq else batch_size
                opp_prev_action = torch.zeros(
                    flat_batch, self._opp_past_actions_size,
                    device=states.device, dtype=states.dtype,
                )

            if has_seq and opp_image.dim() == 5:
                opp_image = opp_image.reshape(batch_size * seq_len, *self._opp_image_shape)
            if has_seq and opp_prev_action.dim() == 3:
                opp_prev_action = opp_prev_action.reshape(batch_size * seq_len, -1)

            # Sanitize opponent inputs at point-of-use. skrl's stored rollout
            # memory can hand back non-finite slots for opp_image during BPTT
            # (uninitialised opponent-observation entries), which previously
            # propagated NaN through the entire opp-branch and corrupted the
            # value -> advantage -> actor chain (invalid_state=1). Replacing
            # non-finite pixels with 0 keeps the branch finite; real opponent
            # frames are unaffected.
            # Defensive guard: opp inputs should be finite once
            # expose_opponent_obs=True populates them, but keep a cheap
            # nan_to_num as belt-and-suspenders against transient NaN.
            opp_image = torch.nan_to_num(opp_image, nan=0.0, posinf=0.0, neginf=0.0)
            opp_prev_action = torch.nan_to_num(opp_prev_action, nan=0.0, posinf=0.0, neginf=0.0)

            opp_cnn_out = self.opp_cnn(opp_image)
            opp_cnn_features = self.opp_cnn_linear(opp_cnn_out)

            if self._opp_past_actions_size > 0:
                opp_features = torch.cat([opp_cnn_features, opp_prev_action], dim=-1)
            else:
                opp_features = opp_cnn_features
            opp_features = self.opp_input_ln(opp_features)

            x_opp = opp_features.reshape(batch_size, seq_len, -1)

            # Opponent hidden state
            if rnn_states and len(rnn_states) > 1:
                h0_opp = rnn_states[1]
                if h0_opp.dim() == 2:
                    h0_opp = h0_opp.unsqueeze(0)
                h0_opp = h0_opp.contiguous()
            else:
                h0_opp = torch.zeros(
                    self._rnn_num_layers, x_opp.shape[0], self._rnn_hidden_size,
                    device=x_opp.device, dtype=x_opp.dtype,
                )
            h0_opp = torch.nan_to_num(h0_opp, nan=0.0, posinf=0.0, neginf=0.0)

            opp_gru_features, h_opp = self._run_gru_branch(
                self.opp_gru, x_opp, h0_opp, has_seq, seq_len, terminated
            )
            rnn_outputs.append(h_opp)

        # Flatten states for MLP if needed
        if has_seq:
            states_flat = states.reshape(batch_size * seq_len, state_dim)
        else:
            states_flat = states

        # MLP head: concat(state, gru_learner [, gru_opp] [, e(k_t)]) -> value
        parts = [states_flat, gru_features]
        if opp_gru_features is not None:
            parts.append(opp_gru_features)

        if self._opp_id_dim > 0:
            opp_id = inputs.get("opp_id")
            if opp_id is None:
                # Initial-eval / bootstrap fallback: zero embedding output.
                opp_emb = torch.zeros(
                    states_flat.shape[0], self._opp_id_dim,
                    device=states_flat.device, dtype=states_flat.dtype,
                )
            else:
                flat_ids = opp_id.to(torch.long).reshape(-1)
                # If id is per-env (length batch_size) but states are
                # (batch_size * seq_len, ...), broadcast by repeating each id
                # across the sequence. Inputs from BPTT may already be flat
                # per-step (length batch_size * seq_len); detect by shape.
                if flat_ids.shape[0] == batch_size and has_seq and seq_len > 1:
                    flat_ids = flat_ids.unsqueeze(1).expand(batch_size, seq_len).reshape(-1)
                opp_emb = self.opp_id_embedding(flat_ids)
                if opp_emb.shape[0] != states_flat.shape[0]:
                    # Defensive: pad/trim to match if there's a residual mismatch.
                    opp_emb = opp_emb[: states_flat.shape[0]]
            parts.append(opp_emb)

        combined = torch.cat(parts, dim=-1)
        value = self.net(combined)

        return value, {"rnn": rnn_outputs}


def history_state_critic_model(
    observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    clip_actions: bool = False,
    image_channels: int = 2,
    image_height: int = 64,
    image_width: int = 64,
    past_actions_size: int = 4,
    rnn: Optional[Mapping[str, Any]] = None,
    num_envs: int = 1,
    cnn_feature_size: int = 128,
    layers: Sequence[int] = (256, 128, 64),
    opp_branch: bool = False,
    opp_image_channels: Optional[int] = None,
    opp_past_actions_size: Optional[int] = None,
    opp_id_dim: int = 0,
    opp_id_num: int = 0,
    return_source: bool = False,
    *args,
    **kwargs,
) -> Union[Model, str]:
    """Factory function for the V(s, h) / V(s, h, h^opp [, e(k)]) critic.

    Called by the skrl Runner when the YAML config specifies
    ``class: HistoryStateCriticMixin``.
    """
    rnn_cfg = rnn or {}
    opp_str = " + opp_branch" if opp_branch else ""
    e_k_str = f" + e_k({opp_id_dim} from {opp_id_num} ids)" if opp_id_dim > 0 else ""
    if return_source:
        return (
            f"HistoryStateCriticModel(\n"
            f"  CNN: Conv2d({image_channels}→32, k=8,s=4) → Conv2d(32→64, k=4,s=2) → "
            f"Conv2d(64→64, k=3,s=1) → Linear({cnn_feature_size})\n"
            f"  GRU: input={cnn_feature_size}+{past_actions_size}, "
            f"hidden={rnn_cfg.get('hidden_size', 256)}, "
            f"layers={rnn_cfg.get('num_layers', 1)}, "
            f"seq_len={rnn_cfg.get('sequence_length', 16)}{opp_str}{e_k_str}\n"
            f"  MLP: state({observation_space}) + gru_out → {list(layers)} → 1\n"
            f")"
        )

    return HistoryStateCriticModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        image_channels=image_channels,
        image_height=image_height,
        image_width=image_width,
        past_actions_size=past_actions_size,
        rnn=rnn,
        num_envs=num_envs,
        cnn_feature_size=cnn_feature_size,
        layers=layers,
        opp_branch=opp_branch,
        opp_image_channels=opp_image_channels,
        opp_past_actions_size=opp_past_actions_size,
        opp_id_dim=opp_id_dim,
        opp_id_num=opp_id_num,
    )
