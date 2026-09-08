from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import gymnasium

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, Model
from skrl.utils.spaces.torch import compute_space_size


class VshCriticModel(DeterministicMixin, Model):
    """V(s, z) / V(s, z, z^opp [, e(k)]) memory-state critic.

    Value = MLP(concat(state, z_theta [, z_opp] [, e(k)])).

    The critic receives the privileged state vector **and** the detached hidden
    state ``z_theta`` from the actor's GRU.  Optionally, when
    ``z_opp_dim > 0``, it also receives the opponent's detached RNN hidden
    state ``z_opp`` for joint memory-state estimation. When ``opp_id_dim > 0``
    and ``opp_id_num > 0``, an additional learned ``nn.Embedding(opp_id_num,
    opp_id_dim)`` indexes the active opponent pool member k_t — required by
    Proposition 2 in paper v22 §3.3 (joint critic Markov reduction needs the
    identifier to be injective on the pool).

    GRU hidden states are bounded by tanh (values in [-1, 1]) so no
    additional preprocessing is needed.
    """

    def __init__(
        self,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        state_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        clip_actions: bool = False,
        actor_hidden_size: int = 128,
        z_opp_dim: int = 0,
        opp_id_dim: int = 0,
        opp_id_num: int = 0,
        layers: Sequence[int] = (256, 128, 64),
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, state_space=state_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self._z_dim = actor_hidden_size
        self._z_opp_dim = z_opp_dim
        self._opp_id_dim = opp_id_dim

        # skrl 2.x has a dedicated state_space for privileged info; size from it when
        # given (1.4.3 passed the privileged space in as observation_space instead).
        # Storing it also matters because Model.init_state_dict() samples `states` from
        # self.state_space -- if unset, the dry-run compute() receives states=None.
        state_size = compute_space_size(state_space) if state_space is not None else self.num_observations
        input_size = state_size + actor_hidden_size + z_opp_dim + opp_id_dim

        self.opp_id_embedding: Optional[nn.Embedding] = None
        if opp_id_dim > 0:
            if opp_id_num <= 0:
                raise ValueError(
                    f"opp_id_dim={opp_id_dim} requires opp_id_num > 0 (got {opp_id_num})."
                )
            self.opp_id_embedding = nn.Embedding(opp_id_num, opp_id_dim)

        modules: list[nn.Module] = []
        in_dim = input_size
        for out_dim in layers:
            modules.extend([nn.Linear(in_dim, out_dim), nn.ELU()])
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*modules)

    def compute(self, inputs: Mapping[str, Any], role: str = ""):
        states = inputs.get("states")  # preprocessed privileged state (skrl 2.x)
        if states is None:
            # dry-run init or a config that routes the privileged vector as the
            # observation: fall back so the model can still be initialised.
            states = inputs.get("observations")
        # Accept both new (z_theta) and legacy (actor_hidden) key names
        z_theta = inputs.get("z_theta")
        if z_theta is None:
            z_theta = inputs.get("actor_hidden")

        # Fallback: if no z_theta provided (e.g. during initial eval), use zeros
        if z_theta is None:
            z_theta = torch.zeros(
                *states.shape[:-1],
                self._z_dim,
                device=states.device,
                dtype=states.dtype,
            )

        parts = [states, z_theta]

        # Optional opponent hidden state z^opp for V(s,z,z^opp)
        if self._z_opp_dim > 0:
            z_opp = inputs.get("z_opp")
            if z_opp is None:
                z_opp = torch.zeros(
                    *states.shape[:-1],
                    self._z_opp_dim,
                    device=states.device,
                    dtype=states.dtype,
                )
            parts.append(z_opp)

        # Optional learned opponent-identifier embedding e(k_t) — see Prop. 2.
        if self._opp_id_dim > 0:
            opp_id = inputs.get("opp_id")
            if opp_id is None:
                # Initial-eval fallback: zeros mean "embedding for opp id 0"
                # is unused — instead, route through the zero vector path so
                # the critic produces a well-defined output without grad.
                opp_emb = torch.zeros(
                    *states.shape[:-1],
                    self._opp_id_dim,
                    device=states.device,
                    dtype=states.dtype,
                )
            else:
                flat_ids = opp_id.to(torch.long).reshape(-1)
                opp_emb = self.opp_id_embedding(flat_ids)
                opp_emb = opp_emb.view(*states.shape[:-1], self._opp_id_dim)
            parts.append(opp_emb)

        if len({p.shape[0] for p in parts}) > 1:
            raise RuntimeError("VshCriticModel batch mismatch: "
                               + str([tuple(p.shape) for p in parts]))
        x = torch.cat(parts, dim=-1)
        return self.net(x), {}


def vsh_critic_model(
    observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    state_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    clip_actions: bool = False,
    actor_hidden_size: int = 128,
    z_opp_dim: int = 0,
    opp_id_dim: int = 0,
    opp_id_num: int = 0,
    layers: Sequence[int] = (256, 128, 64),
    return_source: bool = False,
    *args,
    **kwargs,
) -> Union[Model, str]:
    """Factory function for the V(s, z) / V(s, z, z^opp [, e(k)]) critic.

    Called by the skrl Runner when the YAML config specifies
    ``class: VshCriticMixin`` or ``class: SzCriticMixin``.
    """
    z_opp_str = f" + z_opp({z_opp_dim})" if z_opp_dim > 0 else ""
    opp_id_str = f" + e_k({opp_id_dim} from {opp_id_num} ids)" if opp_id_dim > 0 else ""
    if return_source:
        return (
            f"VshCriticModel(\n"
            f"  Input: state({observation_space}) + z_theta({actor_hidden_size}){z_opp_str}{opp_id_str}\n"
            f"  MLP: {list(layers)} → 1\n"
            f")"
        )

    return VshCriticModel(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        actor_hidden_size=actor_hidden_size,
        z_opp_dim=z_opp_dim,
        opp_id_dim=opp_id_dim,
        opp_id_num=opp_id_num,
        layers=layers,
    )


# Aliases using the paper's notation: V(s, z) memory-state critic
SzCriticModel = VshCriticModel
sz_critic_model = vsh_critic_model
