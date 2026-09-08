from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import gymnasium

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import flatten_tensorized_space, unflatten_tensorized_space


_ACTIVATION_MAP = {
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
}


def _expand_activations(activations: Union[str, Sequence[str], None], num_layers: int) -> Sequence[Optional[str]]:
    if activations is None:
        return [None] * num_layers
    if isinstance(activations, str):
        return [activations] * num_layers
    if isinstance(activations, (list, tuple)):
        if len(activations) == 0:
            return [None] * num_layers
        if len(activations) == 1:
            return list(activations) * num_layers
        if len(activations) == num_layers:
            return list(activations)
        raise ValueError(f"Activations length ({len(activations)}) doesn't match layers ({num_layers})")
    raise ValueError(f"Invalid or unsupported activations definition: {activations}")


def _parse_linear_size(layer: Any) -> int:
    if isinstance(layer, (int, float)):
        return int(layer)
    if isinstance(layer, dict):
        if "linear" not in layer:
            raise ValueError(f"Invalid or unsupported layer definition: {layer}")
        value = layer["linear"]
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, list):
            return int(value[0])
        if isinstance(value, dict):
            if "out_features" in value:
                return int(value["out_features"])
            if "features" in value:
                return int(value["features"])
        raise ValueError(f"Invalid or unsupported 'linear' layer definition: {value}")
    raise ValueError(f"Invalid or unsupported layer definition: {layer}")


def _build_mlp(input_size: int, layers: Sequence[Any], activations: Union[str, Sequence[str], None]) -> Tuple[nn.Module, int]:
    if not layers:
        return nn.Identity(), input_size
    activations = _expand_activations(activations, len(layers))
    modules: list[nn.Module] = []
    in_features = input_size
    for layer, activation in zip(layers, activations):
        out_features = _parse_linear_size(layer)
        modules.append(nn.Linear(in_features, out_features))
        if activation:
            activation_cls = _ACTIVATION_MAP.get(str(activation).lower())
            if activation_cls is None:
                raise ValueError(f"Unsupported activation: {activation}")
            if activation_cls is nn.Softmax:
                modules.append(activation_cls(dim=-1))
            else:
                modules.append(activation_cls())
        in_features = out_features
    return nn.Sequential(*modules), in_features


class GaussianRNNModel(GaussianMixin, Model):
    def __init__(
        self,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        initial_log_std: float = 0.0,
        fixed_log_std: bool = False,
        network: Sequence[Mapping[str, Any]] = (),
        rnn: Optional[Mapping[str, Any]] = None,
        output: Union[str, Sequence[str]] = "",
        num_envs: int = 1,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std, reduction=reduction)

        # Pre-RNN MLP (optional)
        layers: Sequence[Any] = []
        activations: Union[str, Sequence[str], None] = None
        if network:
            if len(network) != 1:
                raise ValueError("GaussianRNNModel only supports a single network container")
            layers = network[0].get("layers", [])
            activations = network[0].get("activations", None)
        self.mlp, mlp_out = _build_mlp(self.num_observations, layers, activations)

        # RNN configuration
        rnn_cfg = rnn or {}
        rnn_type = str(rnn_cfg.get("type", "gru")).lower()
        if rnn_type != "gru":
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        self._rnn_hidden_size = int(rnn_cfg.get("hidden_size", mlp_out))
        self._rnn_num_layers = int(rnn_cfg.get("num_layers", 1))
        self._rnn_sequence_length = int(rnn_cfg.get("sequence_length", 1))
        self._rnn_num_envs = max(int(num_envs), 1)
        rnn_dropout = float(rnn_cfg.get("dropout", 0.0))

        self.rnn = nn.GRU(
            input_size=mlp_out,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._rnn_num_layers,
            dropout=rnn_dropout if self._rnn_num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.output_layer = nn.Linear(self._rnn_hidden_size, self.num_actions)
        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions,), fill_value=float(initial_log_std)),
            requires_grad=not fixed_log_std,
        )

    def get_specification(self) -> Mapping[str, Any]:
        return {
            "rnn": {
                "sizes": [(self._rnn_num_layers, self._rnn_num_envs, self._rnn_hidden_size)],
                "sequence_length": self._rnn_sequence_length,
            }
        }

    def compute(self, inputs, role: str = ""):
        # skrl 2.x: "observations" is what the actor may see; "states" is the
        # PRIVILEGED state (1.4.3 used "states" for the observation). Reading
        # "states" here would silently feed the actor privileged information.
        states = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        if not torch.is_tensor(states):
            states = flatten_tensorized_space(states)

        x = states
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = self.mlp(x)
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            batch, seq_len, feat = x.shape
            x = x.reshape(batch * seq_len, feat)
            x = self.mlp(x)
            x = x.reshape(batch, seq_len, -1)
        else:
            x = x.view(x.shape[0], -1)
            x = self.mlp(x)
            x = x.unsqueeze(1)

        rnn_states = inputs.get("rnn", None)
        if isinstance(rnn_states, torch.Tensor):
            rnn_states = [rnn_states]
        if rnn_states and x.dim() == 3:
            batch_from_state = rnn_states[0].shape[1]
            if x.shape[0] != batch_from_state and x.shape[1] == batch_from_state:
                x = x.transpose(0, 1)
        if not rnn_states:
            h0 = torch.zeros(
                self._rnn_num_layers, x.shape[0], self._rnn_hidden_size, device=x.device, dtype=x.dtype
            )
        else:
            h0 = rnn_states[0]
            if h0.dim() == 2:
                h0 = h0.unsqueeze(0)

        terminated = inputs.get("terminated", None)
        if terminated is not None:
            done = terminated
            if done.dim() > 1:
                done = done[..., 0]
            if done.dim() > 1:
                done = done[:, -1]
            done = done.reshape(-1).bool()
            if done.any():
                h0[:, done] = 0.0

        rnn_out, h_n = self.rnn(x, h0)
        features = rnn_out[:, -1, :]
        output = self.output_layer(features)
        # skrl 2.x compute() returns (output, outputs_dict) and takes log_std from the
        # dict; 1.4.3 returned the 3-tuple (output, log_std, outputs).
        return output, {"log_std": self.log_std_parameter, "rnn": [h_n]}


def gaussian_rnn_model(
    observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    clip_actions: bool = False,
    clip_log_std: bool = True,
    min_log_std: float = -20,
    max_log_std: float = 2,
    reduction: str = "sum",
    initial_log_std: float = 0.0,
    fixed_log_std: bool = False,
    network: Sequence[Mapping[str, Any]] = (),
    rnn: Optional[Mapping[str, Any]] = None,
    output: Union[str, Sequence[str]] = "",
    return_source: bool = False,
    num_envs: int = 1,
    *args,
    **kwargs,
) -> Union[Model, str]:
    """Instantiate a Gaussian RNN (GRU) model"""
    # skrl 2.x removed convert_deprecated_parameters, so the pre-1.3.0 implicit-kwargs
    # path no longer exists: the network must be specified explicitly.
    if not network:
        raise ValueError(
            "gaussian_rnn_model requires an explicit `network` specification under skrl 2.x"
        )

    if return_source:
        rnn_cfg = rnn or {}
        return (
            "GaussianRNNModel("
            f"network_layers={[n.get('layers', []) for n in (network or [])]}, "
            f"rnn={rnn_cfg}, "
            f"output={output})"
        )

    return GaussianRNNModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        clip_log_std=clip_log_std,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
        reduction=reduction,
        initial_log_std=initial_log_std,
        fixed_log_std=fixed_log_std,
        network=network,
        rnn=rnn,
        output=output,
        num_envs=num_envs,
    )
