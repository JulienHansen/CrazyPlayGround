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

import copy
from typing import Any, Type

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

    # ------------------------------------------------------------------
    def _generate_agent(self, env, cfg: dict[str, Any], models):
        """Instantiate PPO_RNN_SZ.

        Stock `_generate_agent` branches on a hardcoded list of agent names and builds
        the config from a dataclass registered as ``{agent_class}_CFG``. Our ported
        agent is not in that list and carries a plain dict default config, so the base
        implementation leaves `agent_cfg` unbound and raises UnboundLocalError. This
        override reproduces the single-agent path for our agent only and delegates
        everything else upstream.
        """
        agent_class = str(cfg.get("agent", {}).get("class", "")).lower()
        if agent_class not in ("ppo_rnn_sz", "ppo_rnn_vsh"):
            return super()._generate_agent(env, cfg, models)

        device = env.device
        num_envs = env.num_envs
        observation_space = env.observation_space
        state_space = env.state_space
        action_space = env.action_space

        # memory (mirrors the base implementation)
        memory_cfg = dict(cfg["memory"])
        memory_class = self._component(memory_cfg.pop("class"))
        if memory_cfg.get("memory_size", -1) < 0:
            memory_cfg["memory_size"] = cfg["agent"]["rollouts"]
        memory = memory_class(num_envs=num_envs, device=device, **self._process_cfg(memory_cfg))

        # agent config: dict default merged with the YAML
        agent_cfg = copy.deepcopy(PPO_RNN_SZ_DEFAULT_CONFIG)
        user_cfg = self._process_cfg({k: v for k, v in cfg["agent"].items() if k != "class"})
        experiment = {**agent_cfg.get("experiment", {}), **(user_cfg.pop("experiment", {}) or {})}
        agent_cfg.update(user_cfg)
        agent_cfg["experiment"] = experiment

        # skrl's _process_cfg only resolves class names for keys it knows about, so our
        # custom `critic_state_preprocessor` (and friends) can arrive as plain strings.
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        from skrl.resources.schedulers.torch import KLAdaptiveLR

        _resolve = {"RunningStandardScaler": RunningStandardScaler, "KLAdaptiveLR": KLAdaptiveLR,
                    "None": None, "null": None}
        for key in ("state_preprocessor", "critic_state_preprocessor",
                    "value_preprocessor", "learning_rate_scheduler"):
            val = agent_cfg.get(key)
            if isinstance(val, str):
                if val not in _resolve:
                    raise ValueError(f"Unsupported '{key}' value: {val!r}")
                agent_cfg[key] = _resolve[val]

        # preprocessor sizes. `state_preprocessor` normalises the ACTOR observation
        # (skrl-1.x field name kept by the ported agent); `critic_state_preprocessor`
        # normalises the privileged state.
        if agent_cfg.get("state_preprocessor"):
            agent_cfg["state_preprocessor_kwargs"] = {
                **(agent_cfg.get("state_preprocessor_kwargs") or {}),
                "size": observation_space, "device": device}
        if agent_cfg.get("critic_state_preprocessor"):
            agent_cfg["critic_state_preprocessor_kwargs"] = {
                **(agent_cfg.get("critic_state_preprocessor_kwargs") or {}),
                "size": state_space, "device": device}
        if agent_cfg.get("value_preprocessor"):
            agent_cfg["value_preprocessor_kwargs"] = {
                **(agent_cfg.get("value_preprocessor_kwargs") or {}),
                "size": 1, "device": device}

        return PPO_RNN_SZ(
            models=models["agent"],
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=agent_cfg,
        )
