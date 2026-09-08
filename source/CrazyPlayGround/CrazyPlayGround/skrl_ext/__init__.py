"""Asymmetric / memory-state critic extensions for skrl.

Ported from MemoryStateCritic (https://github.com/LouetteArthur/MemoryStateCritic,
commit 8a54ee6) which targets the bundled skrl 1.4.3 fork. CrazyPlayGround runs
skrl 2.1.0, so the constructors here are adapted to the 2.x keyword-only
`Model`/`Agent` signatures and to 2.x's native observation/state split.

Critic variants (paper Table 1):
    V(o)      symmetric, no privileged state
    V(s)      privileged state only            -- biased with a history-dependent actor
    V(s,z)    state + DETACHED actor GRU hidden -- the memory-state critic
    V(s,h)    state + critic-side recurrent history
"""

from .models.vsh_critic import VshCriticModel, vsh_critic_model  # noqa: F401
from .models.gaussian_rnn import GaussianRNNModel, gaussian_rnn_model  # noqa: F401
from .agents.ppo_rnn_sz import PPO_RNN_SZ, PPO_RNN_SZ_DEFAULT_CONFIG  # noqa: F401
from .runner import MemoryStateRunner  # noqa: F401,E402
