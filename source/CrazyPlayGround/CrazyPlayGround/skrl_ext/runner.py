"""skrl Runner that knows the memory-state / asymmetric critic components.

Stock ``skrl.utils.runner.torch.Runner`` resolves component names from a fixed map
(`_component`) that contains no recurrent model instantiator and no PPO_RNN variants,
so the YAML path cannot reach them. This subclass adds:

    GaussianRNNMixin   -> gaussian_rnn_model   (vector-observation GRU actor)
    VshCriticMixin     -> vsh_critic_model     (V(s,z) memory-state critic)
    SzCriticMixin      -> alias of VshCriticMixin
    PPO_RNN_SZ         -> PPO_RNN_SZ agent + its default config

Ported from MemoryStateCritic (commit 8a54ee6), which targets skrl 1.4.3.
"""

from typing import Type

from skrl.utils.runner.torch import Runner as _SkrlRunner

from .models.gaussian_rnn import gaussian_rnn_model
from .models.vsh_critic import vsh_critic_model
from .agents.ppo_rnn_sz import PPO_RNN_SZ, PPO_RNN_SZ_DEFAULT_CONFIG


class MemoryStateRunner(_SkrlRunner):
    """Runner with the recurrent actor and privileged-critic components registered."""

    _EXTRA = {
        "gaussianrnnmixin": gaussian_rnn_model,
        "vshcriticmixin": vsh_critic_model,
        "szcriticmixin": vsh_critic_model,
        "ppo_rnn_sz": (PPO_RNN_SZ, PPO_RNN_SZ_DEFAULT_CONFIG),
        "ppo_rnn_vsh": (PPO_RNN_SZ, PPO_RNN_SZ_DEFAULT_CONFIG),
    }

    def _component(self, name: str) -> Type:
        key = str(name).lower()
        if key in self._EXTRA:
            return self._EXTRA[key]
        return super()._component(name)
