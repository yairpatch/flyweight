from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bf16 import BF16Tensor
from .matrix import load_matrix
from .float_tensor import FloatTensor
from .q4 import np
from .tensor_container import ColiTensorFile


@dataclass(frozen=True, slots=True)
class GatedDeltaResult:
    output: list[float]


class GatedDeltaState:
    """Depthwise convolution and recurrent matrix state for one linear layer."""

    def __init__(
        self,
        conv_dim: int,
        conv_kernel_size: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
    ):
        self.conv_dim = conv_dim
        self.conv_kernel_size = conv_kernel_size
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.tokens = 0
        self.cuda_conv_state: Any = None
        self.cuda_recurrent_state: Any = None
        if np is not None:
            self.conv_state: Any = np.zeros(
                (conv_dim, conv_kernel_size), dtype=np.float32
            )
            self.recurrent_state: Any = np.zeros(
                (num_value_heads, key_head_dim, value_head_dim),
                dtype=np.float32,
            )
        else:
            self.conv_state = [
                [0.0] * conv_kernel_size for _ in range(conv_dim)
            ]
            self.recurrent_state = [
                [
                    [0.0] * value_head_dim
                    for _ in range(key_head_dim)
                ]
                for _ in range(num_value_heads)
            ]

    def clear(self) -> None:
        self.tokens = 0
        if self.cuda_conv_state is not None:
            self.cuda_conv_state.fill(0.0)
            self.cuda_recurrent_state.fill(0.0)
        if np is not None:
            self.conv_state.fill(0.0)
            self.recurrent_state.fill(0.0)
            return
        for channel in self.conv_state:
            channel[:] = [0.0] * self.conv_kernel_size
        for head in self.recurrent_state:
            for row in head:
                row[:] = [0.0] * self.value_head_dim


class QwenGatedDeltaLayer:
    """Executable Qwen3.5/3.6 Gated DeltaNet token mixer for one token."""

    def __init__(self, layer_file: Path | str):
        self.layer_file = Path(layer_file)
        container = ColiTensorFile(self.layer_file)
        metadata = container.metadata
        self.layer = int(metadata["layer"])
        self.hidden_size = int(metadata["hidden_size"])
        self.num_key_heads = int(metadata["num_key_heads"])
        self.num_value_heads = int(metadata["num_value_heads"])
        self.key_head_dim = int(metadata["key_head_dim"])
        self.value_head_dim = int(metadata["value_head_dim"])
        self.conv_kernel_size = int(metadata["conv_kernel_size"])
        self.rms_norm_eps = float(metadata["rms_norm_eps"])
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.input_norm = BF16Tensor.from_container(
            container, "input_layernorm.weight"
        )
        self.in_proj_qkv = load_matrix(container, "in_proj_qkv.weight")
        self.in_proj_z = load_matrix(container, "in_proj_z.weight")
        self.in_proj_b = load_matrix(container, "in_proj_b.weight")
        self.in_proj_a = load_matrix(container, "in_proj_a.weight")
        self.conv1d = BF16Tensor.from_container(container, "conv1d.weight")
        self.dt_bias = BF16Tensor.from_container(container, "dt_bias")
        self.a_log = FloatTensor.from_container(container, "A_log")
        self.norm = FloatTensor.from_container(container, "norm.weight")
        self.out_proj = load_matrix(container, "out_proj.weight")
        self._input_norm_weights = self.input_norm.values()
        self._conv_weights = self.conv1d.values()
        self._dt_bias = self.dt_bias.values()
        self._a_log = self.a_log.values()
        self._norm_weights = self.norm.values()
        self._validate()

    @classmethod
    def from_model_directory(
        cls, root: Path | str, layer: int
    ) -> "QwenGatedDeltaLayer":
        return cls(Path(root) / "linear_layers" / f"layer-{layer:03d}.coli")

    def new_state(self) -> GatedDeltaState:
        return GatedDeltaState(
            self.conv_dim,
            self.conv_kernel_size,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
        )

    def forward(
        self, hidden: list[float], state: GatedDeltaState
    ) -> GatedDeltaResult:
        if len(hidden) != self.hidden_size:
            raise ValueError(
                f"expected hidden width {self.hidden_size}, got {len(hidden)}"
            )
        self._validate_state(state)
        from .cuda import active_cuda

        accelerator = active_cuda()
        if accelerator is not None:
            output = accelerator.gated_delta(self, hidden, state)
            state.tokens += 1
            return GatedDeltaResult(output=output)
        normalized = _rms_norm(
            hidden, self._input_norm_weights, self.rms_norm_eps
        )
        from .cuda import active_cuda

        accelerator = active_cuda()
        if accelerator is not None:
            mixed_qkv, z, beta_logits, decay_logits = (
                accelerator.bf16_matvec_many(
                    [
                        self.in_proj_qkv,
                        self.in_proj_z,
                        self.in_proj_b,
                        self.in_proj_a,
                    ],
                    normalized,
                )
            )
        else:
            mixed_qkv = self.in_proj_qkv.matvec(normalized)
            z = self.in_proj_z.matvec(normalized)
            beta_logits = self.in_proj_b.matvec(normalized)
            decay_logits = self.in_proj_a.matvec(normalized)
        if np is not None:
            core = self._forward_numpy(
                mixed_qkv, z, beta_logits, decay_logits, state
            )
        else:
            core = self._forward_python(
                mixed_qkv, z, beta_logits, decay_logits, state
            )
        state.tokens += 1
        return GatedDeltaResult(output=self.out_proj.matvec(core))

    def forward_residual(
        self, hidden: list[float], state: GatedDeltaState
    ) -> GatedDeltaResult:
        result = self.forward(hidden, state)
        return GatedDeltaResult(
            output=[residual + value for residual, value in zip(hidden, result.output)]
        )

    def _forward_numpy(
        self,
        mixed_qkv: list[float],
        z: list[float],
        beta_logits: list[float],
        decay_logits: list[float],
        state: GatedDeltaState,
    ) -> list[float]:
        mixed = np.asarray(mixed_qkv, dtype=np.float32)
        state.conv_state[:, :-1] = state.conv_state[:, 1:]
        state.conv_state[:, -1] = mixed
        weights = np.asarray(self._conv_weights, dtype=np.float32).reshape(
            self.conv_dim, self.conv_kernel_size
        )
        convolved = (state.conv_state * weights).sum(axis=1)
        convolved = convolved / (1.0 + np.exp(-convolved))
        queries = convolved[: self.key_dim].reshape(
            self.num_key_heads, self.key_head_dim
        )
        keys = convolved[self.key_dim : self.key_dim * 2].reshape(
            self.num_key_heads, self.key_head_dim
        )
        values = convolved[self.key_dim * 2 :].reshape(
            self.num_value_heads, self.value_head_dim
        )
        repeats = self.num_value_heads // self.num_key_heads
        queries = np.repeat(queries, repeats, axis=0)
        keys = np.repeat(keys, repeats, axis=0)
        queries = queries / np.sqrt(
            (queries * queries).sum(axis=1, keepdims=True) + 1e-6
        )
        keys = keys / np.sqrt(
            (keys * keys).sum(axis=1, keepdims=True) + 1e-6
        )
        queries *= self.key_head_dim**-0.5
        beta = 1.0 / (
            1.0 + np.exp(-np.asarray(beta_logits, dtype=np.float32))
        )
        decay = -np.exp(np.asarray(self._a_log, dtype=np.float32)) * np.logaddexp(
            0.0,
            np.asarray(decay_logits, dtype=np.float32)
            + np.asarray(self._dt_bias, dtype=np.float32),
        )
        state.recurrent_state *= np.exp(decay)[:, None, None]
        memory = np.einsum("hkv,hk->hv", state.recurrent_state, keys)
        delta = (values - memory) * beta[:, None]
        state.recurrent_state += keys[:, :, None] * delta[:, None, :]
        core = np.einsum("hkv,hk->hv", state.recurrent_state, queries)
        inverse_rms = 1.0 / np.sqrt(
            (core * core).mean(axis=1, keepdims=True) + self.rms_norm_eps
        )
        gates = np.asarray(z, dtype=np.float32).reshape(
            self.num_value_heads, self.value_head_dim
        )
        silu_gates = gates / (1.0 + np.exp(-gates))
        core = (
            core
            * inverse_rms
            * np.asarray(self._norm_weights, dtype=np.float32)[None, :]
            * silu_gates
        )
        return core.reshape(-1).tolist()

    def _forward_python(
        self,
        mixed_qkv: list[float],
        z: list[float],
        beta_logits: list[float],
        decay_logits: list[float],
        state: GatedDeltaState,
    ) -> list[float]:
        convolved = []
        for channel, value in enumerate(mixed_qkv):
            history = state.conv_state[channel]
            history.pop(0)
            history.append(value)
            start = channel * self.conv_kernel_size
            total = sum(
                sample * weight
                for sample, weight in zip(
                    history,
                    self._conv_weights[start : start + self.conv_kernel_size],
                )
            )
            convolved.append(_silu(total))
        queries = _reshape_heads(
            convolved[: self.key_dim], self.num_key_heads, self.key_head_dim
        )
        keys = _reshape_heads(
            convolved[self.key_dim : self.key_dim * 2],
            self.num_key_heads,
            self.key_head_dim,
        )
        values = _reshape_heads(
            convolved[self.key_dim * 2 :],
            self.num_value_heads,
            self.value_head_dim,
        )
        repeats = self.num_value_heads // self.num_key_heads
        queries = [head for head in queries for _ in range(repeats)]
        keys = [head for head in keys for _ in range(repeats)]
        queries = [
            [value / math.sqrt(sum(item * item for item in head) + 1e-6)
             / math.sqrt(self.key_head_dim) for value in head]
            for head in queries
        ]
        keys = [
            [value / math.sqrt(sum(item * item for item in head) + 1e-6)
             for value in head]
            for head in keys
        ]
        output: list[float] = []
        for head in range(self.num_value_heads):
            beta = _sigmoid(beta_logits[head])
            decay = math.exp(
                -math.exp(self._a_log[head])
                * _softplus(decay_logits[head] + self._dt_bias[head])
            )
            matrix = state.recurrent_state[head]
            for row in matrix:
                for column in range(self.value_head_dim):
                    row[column] *= decay
            memory = [
                sum(
                    matrix[row][column] * keys[head][row]
                    for row in range(self.key_head_dim)
                )
                for column in range(self.value_head_dim)
            ]
            delta = [
                (value - remembered) * beta
                for value, remembered in zip(values[head], memory)
            ]
            for row in range(self.key_head_dim):
                for column in range(self.value_head_dim):
                    matrix[row][column] += keys[head][row] * delta[column]
            core = [
                sum(
                    matrix[row][column] * queries[head][row]
                    for row in range(self.key_head_dim)
                )
                for column in range(self.value_head_dim)
            ]
            inverse_rms = 1.0 / math.sqrt(
                sum(value * value for value in core) / len(core)
                + self.rms_norm_eps
            )
            gate_start = head * self.value_head_dim
            output.extend(
                value
                * inverse_rms
                * self._norm_weights[column]
                * _silu(z[gate_start + column])
                for column, value in enumerate(core)
            )
        return output

    def _validate_state(self, state: GatedDeltaState) -> None:
        expected = (
            self.conv_dim,
            self.conv_kernel_size,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
        )
        actual = (
            state.conv_dim,
            state.conv_kernel_size,
            state.num_value_heads,
            state.key_head_dim,
            state.value_head_dim,
        )
        if actual != expected:
            raise ValueError(f"state geometry {actual} does not match layer {expected}")

    def _validate(self) -> None:
        expected = {
            "input norm": (self.hidden_size,),
            "qkv projection": (self.conv_dim, self.hidden_size),
            "z projection": (self.value_dim, self.hidden_size),
            "beta projection": (self.num_value_heads, self.hidden_size),
            "decay projection": (self.num_value_heads, self.hidden_size),
            "convolution": (self.conv_dim, 1, self.conv_kernel_size),
            "dt bias": (self.num_value_heads,),
            "A log": (self.num_value_heads,),
            "gated norm": (self.value_head_dim,),
            "output projection": (self.hidden_size, self.value_dim),
        }
        actual = {
            "input norm": self.input_norm.shape,
            "qkv projection": self.in_proj_qkv.shape,
            "z projection": self.in_proj_z.shape,
            "beta projection": self.in_proj_b.shape,
            "decay projection": self.in_proj_a.shape,
            "convolution": self.conv1d.shape,
            "dt bias": self.dt_bias.shape,
            "A log": self.a_log.shape,
            "gated norm": self.norm.shape,
            "output projection": self.out_proj.shape,
        }
        mismatches = [
            f"{name} {actual[name]} != {shape}"
            for name, shape in expected.items()
            if actual[name] != shape
        ]
        if mismatches:
            raise ValueError(f"invalid Gated DeltaNet layer: {', '.join(mismatches)}")
        if self.num_value_heads % self.num_key_heads:
            raise ValueError("value heads must be divisible by key heads")


def _rms_norm(
    vector: list[float], weights: list[float], epsilon: float
) -> list[float]:
    inverse_rms = 1.0 / math.sqrt(
        sum(value * value for value in vector) / len(vector) + epsilon
    )
    return [
        value * inverse_rms * (1.0 + weight)
        for value, weight in zip(vector, weights)
    ]


def _reshape_heads(
    vector: list[float], heads: int, width: int
) -> list[list[float]]:
    return [vector[head * width : (head + 1) * width] for head in range(heads)]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _silu(value: float) -> float:
    return value * _sigmoid(value)


def _softplus(value: float) -> float:
    if value > 20.0:
        return value
    if value < -20.0:
        return math.exp(value)
    return math.log1p(math.exp(value))

