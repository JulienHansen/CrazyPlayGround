"""PPO_RNN_VSH: PPO_RNN with asymmetric actor-critic, V(s, z), and V(s, h) support.

This is a modified copy of ``skrl.agents.torch.ppo.PPO_RNN`` that adds:

1. Asymmetric actor-critic support (same pattern as ``PPO_ASYM``): separate
   preprocessor path for critic states, ``critic_states`` memory tensor,
   extraction from env infos in ``record_transition``.

2. Optional V(s, z) memory-state critic mode: when ``sz_critic: True`` (or
   legacy ``vsh_critic: True``) in the config, the detached actor GRU hidden
   state ``z_theta`` is stored in memory and passed to the critic at every
   call site.  The critic is expected to be a ``SzCriticModel``.

3. Optional V(s, h) history-state critic mode: when ``sh_critic: True``, the
   critic has its own CNN+GRU that processes the learner's observation-action
   stream with separate weights from the actor.  The critic model is expected
   to be a ``HistoryStateCriticModel`` that exposes RNN specification so its
   hidden state is managed automatically by the existing value-RNN infra.
   The agent extracts image and past_actions from the flat observation at
   each call site and passes them to the critic.

Differences from upstream ``skrl.PPO_RNN``:
- ``critic_state_preprocessor`` / ``critic_state_preprocessor_kwargs`` fields
- ``sz_critic`` / ``sz_z_dim`` fields (default: disabled)
- ``sh_critic`` / ``sh_image_shape`` / ``sh_past_actions_size`` fields
- Memory tensor ``z_theta`` (only when ``sz_critic`` is True)
- ``z_theta`` or image inputs threaded through ``self.value.act`` at 3 call sites
"""

from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium
import numpy as np
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.spaces.torch import compute_space_size, flatten_tensorized_space, unflatten_tensorized_space


# fmt: off
# [start-config-dict-torch]
PPO_RNN_VSH_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "critic_state_preprocessor": None,      # separate preprocessor for critic states (asymmetric actor-critic)
    "critic_state_preprocessor_kwargs": {}, # critic state preprocessor's kwargs (e.g. {"size": env.state_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "sz_critic": False,             # V(s,z) memory-state critic: pass detached actor GRU z_theta to critic
    "sz_z_dim": 128,                # dimension of actor GRU hidden state z (must match actor RNN hidden_size)
    "sz_z_opp_dim": 0,              # V(s,z,z^opp): opponent z dim (0 = disabled, >0 = joint)
    "opp_id_dim": 0,                # V(..., e(k_t)) joint critic: learned opponent-identifier embedding dim (0 = disabled)
    "opp_id_num": 0,                # Number of unique opponent ids the embedding spans (required when opp_id_dim > 0)
    "sh_critic": False,             # V(s,h) history-state critic: critic has own CNN+GRU over learner's (o,a)
    "sh_image_shape": [2, 64, 64],  # (C, H, W) of the image in the flat observation
    "sh_past_actions_size": 4,      # number of past-action dims in the flat observation
    "sh_opp_branch": False,         # V(s,h,h^opp): enable opponent CNN+GRU branch in critic
    "sh_opp_image_shape": None,     # (C, H, W) of opponent image (defaults to sh_image_shape)
    "sh_opp_past_actions_size": 4,  # opponent past-action dims
    # Legacy aliases (backward compatibility with existing configs/checkpoints)
    "vsh_critic": None,             # deprecated: use sz_critic
    "vsh_actor_hidden_size": None,  # deprecated: use sz_z_dim

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class PPO_RNN_VSH(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        state_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Proximal Policy Optimization (PPO) with support for Recurrent Neural Networks (RNN, GRU, LSTM, etc.)

        https://arxiv.org/abs/1707.06347

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(PPO_RNN_VSH_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._critic_state_preprocessor = self.cfg["critic_state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # V(s,z) memory-state critic: support both new and legacy config keys
        self._sz_critic = self.cfg["sz_critic"] or bool(self.cfg.get("vsh_critic"))
        self._sz_z_dim = self.cfg["sz_z_dim"] if self.cfg.get("vsh_actor_hidden_size") is None else self.cfg["vsh_actor_hidden_size"]
        self._sz_z_opp_dim = int(self.cfg.get("sz_z_opp_dim", 0))
        self._opp_id_dim = int(self.cfg.get("opp_id_dim", 0))
        self._opp_id_num = int(self.cfg.get("opp_id_num", 0))

        # V(s,h) history-state critic: critic has own CNN+GRU
        self._sh_critic = self.cfg["sh_critic"]
        if self._sh_critic:
            self._sh_image_shape = tuple(self.cfg["sh_image_shape"])  # (C, H, W)
            self._sh_image_size = int(np.prod(self._sh_image_shape))
            self._sh_past_actions_size = self.cfg["sh_past_actions_size"]
        self._sh_opp_branch = bool(self.cfg.get("sh_opp_branch", False))
        if self._sh_opp_branch:
            opp_shape = self.cfg.get("sh_opp_image_shape") or self.cfg.get("sh_image_shape", [2, 64, 64])
            self._sh_opp_image_shape = tuple(opp_shape)
            self._sh_opp_image_size = int(np.prod(self._sh_opp_image_shape))
            self._sh_opp_past_actions_size = int(self.cfg.get("sh_opp_past_actions_size", 4))

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self._learning_rate
                )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        # set up separate critic state preprocessor for asymmetric actor-critic
        if self._critic_state_preprocessor:
            self._critic_state_preprocessor = self._critic_state_preprocessor(**self.cfg["critic_state_preprocessor_kwargs"])
            self.checkpoint_modules["critic_state_preprocessor"] = self._critic_state_preprocessor
        else:
            # Fall back to state_preprocessor if critic_state_preprocessor not specified
            self._critic_state_preprocessor = None

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

        # Dict critic support: normalize the "state" component of Dict critic states
        # (used by Geles-style unbiased critic where critic receives image + past_actions + state)
        self._state_component_scaler = None
        if self.value is not None:
            val_obs_space = getattr(self.value, "observation_space", None)
            if isinstance(val_obs_space, gymnasium.spaces.Dict) and "state" in val_obs_space.spaces:
                from skrl.resources.preprocessors.torch import RunningStandardScaler

                self._state_component_scaler = RunningStandardScaler(
                    size=val_obs_space["state"], device=device
                )
                self.checkpoint_modules["state_component_scaler"] = self._state_component_scaler

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # Determine critic state size from config or fall back to observation space
        critic_pp_kwargs = self.cfg.get("critic_state_preprocessor_kwargs") or {}
        critic_state_size = critic_pp_kwargs.get("size", self.observation_space)
        # Dict critic states (Geles-style): compute flat size for memory storage
        if isinstance(critic_state_size, gymnasium.spaces.Dict):
            critic_state_size = compute_space_size(critic_state_size)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="critic_states", size=critic_state_size, dtype=torch.float32)

            # V(s,z) memory-state critic: store actor GRU hidden state z_theta
            if self._sz_critic:
                self.memory.create_tensor(
                    name="z_theta", size=self._sz_z_dim, dtype=torch.float32
                )

            # V(s,z,z^opp): store opponent hidden state
            if self._sz_z_opp_dim > 0:
                self.memory.create_tensor(
                    name="z_opp", size=self._sz_z_opp_dim, dtype=torch.float32
                )

            # V(..., e(k_t)): store the opponent-identifier integer k_t.
            # Stored as float32 to fit the memory API; cast to long inside the
            # critic before the embedding lookup. K is always small enough
            # (< 2^24) that float32 is exact.
            if self._opp_id_dim > 0:
                self.memory.create_tensor(name="opp_id", size=1, dtype=torch.float32)

            # V(s,h,h^opp): store opponent image and prev_action
            if self._sh_opp_branch:
                self.memory.create_tensor(
                    name="opp_image", size=int(np.prod(self._sh_opp_image_shape)), dtype=torch.float32
                )
                self.memory.create_tensor(
                    name="opp_prev_action", size=self._sh_opp_past_actions_size, dtype=torch.float32
                )

            # tensors sampled during training
            self._tensors_names = [
                "states",
                "actions",
                "terminated",
                "truncated",
                "log_prob",
                "values",
                "returns",
                "advantages",
                "critic_states",
            ]
            if self._sz_critic:
                self._tensors_names.append("z_theta")
            if self._sz_z_opp_dim > 0:
                self._tensors_names.append("z_opp")
            if self._opp_id_dim > 0:
                self._tensors_names.append("opp_id")
            if self._sh_opp_branch:
                self._tensors_names.append("opp_image")
                self._tensors_names.append("opp_prev_action")

        # RNN specifications
        self._rnn = False  # flag to indicate whether RNN is available
        self._rnn_tensors_names = []  # used for sampling during training
        self._rnn_final_states = {"policy": [], "value": []}
        self._rnn_initial_states = {"policy": [], "value": []}
        self._rnn_sequence_length = self.policy.get_specification().get("rnn", {}).get("sequence_length", 1)

        # policy
        for i, size in enumerate(self.policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn = True
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(
                    name=f"rnn_policy_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True
                )
                self._rnn_tensors_names.append(f"rnn_policy_{i}")
            # default RNN states
            self._rnn_initial_states["policy"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # value
        if self.value is not None:
            if self.policy is self.value:
                self._rnn_initial_states["value"] = self._rnn_initial_states["policy"]
            else:
                for i, size in enumerate(self.value.get_specification().get("rnn", {}).get("sizes", [])):
                    self._rnn = True
                    # create tensors in memory
                    if self.memory is not None:
                        self.memory.create_tensor(
                            name=f"rnn_value_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True
                        )
                        self._rnn_tensors_names.append(f"rnn_value_{i}")
                    # default RNN states
                    self._rnn_initial_states["value"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None
        self._current_critic_states = None
        self._current_next_critic_states = None

        # Log critic architecture for experiment verification
        critic_mode = "V(s)"
        critic_input_dim = critic_state_size if isinstance(critic_state_size, int) else "dict"
        if self._sz_critic:
            critic_mode = f"V(s,z) — z_dim={self._sz_z_dim}"
            if isinstance(critic_state_size, int):
                critic_input_dim = f"{critic_state_size} + {self._sz_z_dim} = {critic_state_size + self._sz_z_dim}"
        if self._sh_critic:
            critic_mode = "V(s,h) — critic-side CNN+GRU"
        if self._sz_z_opp_dim > 0:
            critic_mode += f", z_opp_dim={self._sz_z_opp_dim}"
        logger.info(f"[PPO_RNN_VSH] Critic mode: {critic_mode} | critic_state_dim: {critic_input_dim}")

    def _extract_sh_inputs(self, flat_obs: torch.Tensor) -> dict:
        """Extract image and past_actions from the flat observation for V(s,h) critic.

        The flat observation layout (from gymnasium Dict sorted keys) is:
        [image_flat, past_actions]. This matches the flatten order used by
        skrl's ``flatten_tensorized_space`` which sorts Dict keys alphabetically.
        """
        if not self._sh_critic:
            return {}
        # Handle 2D (batch, flat) and 3D (batch, seq, flat)
        orig_shape = flat_obs.shape
        if flat_obs.dim() == 3:
            batch, seq, flat_dim = orig_shape
            flat_obs_2d = flat_obs.reshape(batch * seq, flat_dim)
        else:
            flat_obs_2d = flat_obs

        image = flat_obs_2d[:, : self._sh_image_size].reshape(-1, *self._sh_image_shape)
        past_actions = flat_obs_2d[:, self._sh_image_size : self._sh_image_size + self._sh_past_actions_size]

        if len(orig_shape) == 3:
            # Reshape back to (batch, seq, ...) for BPTT
            image = image.reshape(batch, seq, *self._sh_image_shape)
            past_actions = past_actions.reshape(batch, seq, -1)

        return {"critic_image": image, "critic_past_actions": past_actions}

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        rnn = {"rnn": self._rnn_initial_states["policy"]} if self._rnn else {}

        # sample random actions (random_timesteps=0 in all configs, so this is never hit)
        if timestep < self._random_timesteps:
            # skrl 2.x has no random_act; sample the action space directly.
            return (
                torch.tensor(
                    np.array([self.action_space.sample() for _ in range(states.shape[0])]),
                    device=self.device, dtype=torch.float32,
                ),
                None,
                {},
            )

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, outputs = self.policy.act(
                {"observations": self._state_preprocessor(states), **rnn}, role="policy"
            )
            log_prob = outputs["log_prob"]
            self._current_log_prob = log_prob

        if self._rnn:
            self._rnn_final_states["policy"] = outputs.get("rnn", [])

        # skrl 2.x trainer contract: `actions, outputs = agent.act(...)`
        return actions, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            self._current_next_states = next_states

            # Extract critic states from infos for asymmetric actor-critic
            critic_states = infos.get("critic_states", None) if isinstance(infos, dict) else None
            next_critic_states = infos.get("next_critic_states", None) if isinstance(infos, dict) else None
            # Current critic states should align with current observations
            self._current_critic_states = critic_states if critic_states is not None else states
            # Next critic states are used for bootstrapping
            self._current_next_critic_states = (
                next_critic_states if next_critic_states is not None else next_states
            )

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # V(s,z): extract detached actor hidden state z_theta for critic and storage
            if self._sz_critic and self._rnn_final_states["policy"]:
                # z_theta shape: (num_layers, batch, hidden_size) — take last layer
                self._current_z_theta = self._rnn_final_states["policy"][0][-1].detach()
            else:
                self._current_z_theta = None

            # V(s,z,z^opp): extract opponent z from infos
            self._current_z_opp = None
            if self._sz_z_opp_dim > 0 and isinstance(infos, dict):
                self._current_z_opp = infos.get("z_opp")

            # V(..., e(k_t)): extract opponent identifier integer from infos
            self._current_opp_id = None
            if self._opp_id_dim > 0 and isinstance(infos, dict):
                self._current_opp_id = infos.get("opp_id")

            # V(s,h,h^opp): extract opponent image and prev_action from infos
            self._current_opp_image = None
            self._current_opp_prev_action = None
            if self._sh_opp_branch and isinstance(infos, dict):
                self._current_opp_image = infos.get("opp_image")
                self._current_opp_prev_action = infos.get("opp_prev_action")

            # compute values using critic states (asymmetric actor-critic)
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
                if self._critic_state_preprocessor is not None:
                    value_input = self._critic_state_preprocessor(self._current_critic_states)
                elif isinstance(self._current_critic_states, dict):
                    # Dict critic states (Geles-style): normalize state component
                    value_input = self._current_critic_states
                    if self._state_component_scaler is not None:
                        value_input = {**value_input, "state": self._state_component_scaler(value_input["state"])}
                else:
                    value_input = self._state_preprocessor(states)
                critic_kwargs = {"states": value_input, **rnn}
                if self._current_z_theta is not None:
                    critic_kwargs["z_theta"] = self._current_z_theta
                if self._current_z_opp is not None:
                    critic_kwargs["z_opp"] = self._current_z_opp
                if self._current_opp_id is not None:
                    critic_kwargs["opp_id"] = self._current_opp_id
                critic_kwargs.update(self._extract_sh_inputs(states))
                if self._current_opp_image is not None:
                    critic_kwargs["opp_image"] = self._current_opp_image
                if self._current_opp_prev_action is not None:
                    critic_kwargs["opp_prev_action"] = self._current_opp_prev_action
                values, outputs = self.value.act(critic_kwargs, role="value")
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # package RNN states
            rnn_states = {}
            if self._rnn:
                rnn_states.update(
                    {f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])}
                )
                if self.policy is not self.value:
                    rnn_states.update(
                        {f"rnn_value_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["value"])}
                    )

            # V(s,z): include actor hidden state z_theta in stored samples
            sz_kwargs = {}
            if self._current_z_theta is not None:
                sz_kwargs["z_theta"] = self._current_z_theta
            if self._current_z_opp is not None:
                sz_kwargs["z_opp"] = self._current_z_opp
            if self._current_opp_id is not None:
                # Stored as (num_envs, 1) float32 to match the memory layout.
                sz_kwargs["opp_id"] = self._current_opp_id.to(torch.float32).reshape(-1, 1)
            if self._current_opp_image is not None:
                # Flatten each env's opponent image to a 1D vector for memory
                # storage; use the tensor's own batch dim (the agent has no
                # `num_envs` attribute).
                sz_kwargs["opp_image"] = self._current_opp_image.reshape(
                    self._current_opp_image.shape[0], -1
                )
            if self._current_opp_prev_action is not None:
                sz_kwargs["opp_prev_action"] = self._current_opp_prev_action

            # Flatten Dict critic states for memory storage
            critic_states_for_memory = (
                flatten_tensorized_space(self._current_critic_states)
                if isinstance(self._current_critic_states, dict)
                else self._current_critic_states
            )

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
                critic_states=critic_states_for_memory,
                **rnn_states,
                **sz_kwargs,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                    critic_states=critic_states_for_memory,
                    **rnn_states,
                    **sz_kwargs,
                )

        # update RNN states
        if self._rnn:
            self._rnn_final_states["value"] = (
                self._rnn_final_states["policy"] if self.policy is self.value else outputs.get("rnn", [])
            )

            # reset states if the episodes have ended
            finished_episodes = (terminated | truncated).nonzero(as_tuple=False)
            if finished_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    rnn_state[:, finished_episodes[:, 0]] = 0
                if self.policy is not self.value:
                    for rnn_state in self._rnn_final_states["value"]:
                        rnn_state[:, finished_episodes[:, 0]] = 0

            self._rnn_initial_states = self._rnn_final_states

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
            if self._critic_state_preprocessor is not None and self._current_next_critic_states is not None:
                value_input = self._critic_state_preprocessor(self._current_next_critic_states.float())
            elif isinstance(self._current_next_critic_states, dict):
                value_input = self._current_next_critic_states
                if self._state_component_scaler is not None:
                    value_input = {**value_input, "state": self._state_component_scaler(value_input["state"])}
            else:
                value_input = self._state_preprocessor(self._current_next_states.float())
            bootstrap_kwargs = {"states": value_input, **rnn}
            # V(s,z): use latest actor hidden state z_theta for bootstrapping
            if self._sz_critic and self._rnn_final_states["policy"]:
                bootstrap_kwargs["z_theta"] = self._rnn_final_states["policy"][0][-1].detach()
            # V(s,z,z^opp): use latest z_opp for bootstrapping
            if self._current_z_opp is not None:
                bootstrap_kwargs["z_opp"] = self._current_z_opp
            if self._current_opp_id is not None:
                bootstrap_kwargs["opp_id"] = self._current_opp_id
            # V(s,h): extract image+past_actions from the latest next_states
            bootstrap_kwargs.update(self._extract_sh_inputs(self._current_next_states))
            # V(s,h,h^opp): use latest opponent image/action for bootstrapping
            if self._current_opp_image is not None:
                bootstrap_kwargs["opp_image"] = self._current_opp_image
            if self._current_opp_prev_action is not None:
                bootstrap_kwargs["opp_prev_action"] = self._current_opp_prev_action
            last_values, _ = self.value.act(bootstrap_kwargs, role="value")
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(
            names=self._tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length
        )

        rnn_policy, rnn_value = {}, {}
        if self._rnn:
            sampled_rnn_batches = self.memory.sample_all(
                names=self._rnn_tensors_names,
                mini_batches=self._mini_batches,
                sequence_length=self._rnn_sequence_length,
            )

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for i, sampled in enumerate(sampled_batches):
                sampled_states = sampled[0]
                sampled_actions = sampled[1]
                sampled_terminated = sampled[2]
                sampled_truncated = sampled[3]
                sampled_log_prob = sampled[4]
                sampled_values = sampled[5]
                sampled_returns = sampled[6]
                sampled_advantages = sampled[7]
                sampled_critic_states = sampled[8]
                # Dynamic extraction based on _tensors_names ordering
                _extra_idx = 9
                sampled_z_theta = None
                sampled_z_opp = None
                sampled_opp_id = None
                sampled_opp_image = None
                sampled_opp_prev_action = None
                if self._sz_critic:
                    sampled_z_theta = sampled[_extra_idx]
                    _extra_idx += 1
                if self._sz_z_opp_dim > 0:
                    sampled_z_opp = sampled[_extra_idx]
                    _extra_idx += 1
                if self._opp_id_dim > 0:
                    sampled_opp_id = sampled[_extra_idx]
                    _extra_idx += 1
                if self._sh_opp_branch:
                    sampled_opp_image = sampled[_extra_idx]
                    _extra_idx += 1
                    sampled_opp_prev_action = sampled[_extra_idx]
                    _extra_idx += 1

                # ---- BPTT: reshape data into sequences ----
                seq_len = self._rnn_sequence_length
                use_bptt = self._rnn and seq_len > 1

                if use_bptt:
                    N = sampled_states.shape[0]
                    num_seq = N // seq_len

                    # States → 3D for BPTT through GRU
                    sampled_states_3d = sampled_states.reshape(num_seq, seq_len, -1)
                    # Terminated → 3D for mid-sequence done resets
                    done_3d = (sampled_terminated | sampled_truncated).reshape(num_seq, seq_len, -1)

                    # RNN initial states: take only the first h per sequence
                    if self.policy is self.value:
                        rnn_policy = {
                            "rnn": [s[::seq_len].transpose(0, 1) for s in sampled_rnn_batches[i]],
                            "terminated": done_3d,
                        }
                        rnn_value = rnn_policy
                    else:
                        rnn_policy = {
                            "rnn": [
                                s[::seq_len].transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "policy" in n
                            ],
                            "terminated": done_3d,
                        }
                        rnn_value = {
                            "rnn": [
                                s[::seq_len].transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "value" in n
                            ],
                            "terminated": done_3d,
                        }
                elif self._rnn:
                    # Stored-state (seq_len=1): pass all h, terminated as 2D
                    if self.policy is self.value:
                        rnn_policy = {
                            "rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]],
                            "terminated": sampled_terminated | sampled_truncated,
                        }
                        rnn_value = rnn_policy
                    else:
                        rnn_policy = {
                            "rnn": [
                                s.transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "policy" in n
                            ],
                            "terminated": sampled_terminated | sampled_truncated,
                        }
                        rnn_value = {
                            "rnn": [
                                s.transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "value" in n
                            ],
                            "terminated": sampled_terminated | sampled_truncated,
                        }

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    policy_input = sampled_states_3d if use_bptt else sampled_states
                    policy_input = self._state_preprocessor(policy_input, train=not epoch)

                    _, _policy_out = self.policy.act(
                        {"observations": policy_input, "taken_actions": sampled_actions, **rnn_policy},
                        role="policy",
                    )
                    next_log_prob = _policy_out["log_prob"]

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    if self._state_component_scaler is not None:
                        # Dict critic: unflatten from memory, normalize state component
                        critic_dict = unflatten_tensorized_space(
                            self.value.observation_space, sampled_critic_states
                        )
                        critic_dict["state"] = self._state_component_scaler(
                            critic_dict["state"], train=not epoch
                        )
                        critic_input = critic_dict
                    elif self._critic_state_preprocessor is not None:
                        critic_input = self._critic_state_preprocessor(sampled_critic_states, train=not epoch)
                    else:
                        critic_input = sampled_states
                    # V(s,h): reshape critic states to 3D during BPTT so the
                    # critic's RNN can process sequences properly.
                    if self._sh_critic and use_bptt:
                        critic_input = critic_input.reshape(num_seq, seq_len, -1)
                    value_kwargs = {"states": critic_input, **rnn_value}
                    if sampled_z_theta is not None:
                        value_kwargs["z_theta"] = sampled_z_theta
                    if sampled_z_opp is not None:
                        value_kwargs["z_opp"] = sampled_z_opp
                    if sampled_opp_id is not None:
                        # Stored as (N, 1) float; the critic casts to long
                        # internally. For BPTT, reshape to (num_seq, seq_len, 1)
                        # so the embedding lookup applies pointwise per step.
                        opp_id = sampled_opp_id
                        if use_bptt:
                            opp_id = opp_id.reshape(num_seq, seq_len, 1)
                        value_kwargs["opp_id"] = opp_id
                    # V(s,h): extract image+past_actions from sampled states for BPTT
                    sh_source = sampled_states_3d if use_bptt else sampled_states
                    value_kwargs.update(self._extract_sh_inputs(sh_source))
                    # V(s,h,h^opp): pass opponent image and prev_action
                    if sampled_opp_image is not None:
                        opp_img = sampled_opp_image
                        if use_bptt:
                            opp_img = opp_img.reshape(num_seq, seq_len, *self._sh_opp_image_shape)
                        else:
                            opp_img = opp_img.reshape(-1, *self._sh_opp_image_shape)
                        value_kwargs["opp_image"] = opp_img
                    if sampled_opp_prev_action is not None:
                        opp_pa = sampled_opp_prev_action
                        if use_bptt:
                            opp_pa = opp_pa.reshape(num_seq, seq_len, -1)
                        value_kwargs["opp_prev_action"] = opp_pa
                    predicted_values, _ = self.value.act(value_kwargs, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])


# Aliases using the paper's notation
PPO_RNN_SZ = PPO_RNN_VSH  # V(s, z) memory-state critic
PPO_RNN_SZ_DEFAULT_CONFIG = PPO_RNN_VSH_DEFAULT_CONFIG
PPO_RNN_SH = PPO_RNN_VSH  # V(s, h) history-state critic (same agent, different config)
PPO_RNN_SH_DEFAULT_CONFIG = PPO_RNN_VSH_DEFAULT_CONFIG
