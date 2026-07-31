from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bf16 import BF16Tensor
from .matrix import load_matrix
from .tensor_container import ColiTensorFile


@dataclass(frozen=True, slots=True)
class AttentionResult:
    output: list[float]
    attention_weights: tuple[tuple[float, ...], ...]


class AttentionKVCache:
    """Per-layer grouped-query key/value cache for incremental decoding."""

    def __init__(self, num_key_value_heads: int, head_dim: int):
        if num_key_value_heads <= 0 or head_dim <= 0:
            raise ValueError("cache dimensions must be positive")
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.keys: list[list[list[float]]] = [
            [] for _ in range(num_key_value_heads)
        ]
        self.values: list[list[list[float]]] = [
            [] for _ in range(num_key_value_heads)
        ]
        self.tokens = 0
        self.cuda_keys: Any = None
        self.cuda_values: Any = None
        self.cuda_key_scales: Any = None
        self.cuda_value_scales: Any = None
        self.cuda_cache_type: str | None = None
        self.cuda_capacity = 0

    @property
    def length(self) -> int:
        return self.tokens

    def append(
        self, keys: list[list[float]], values: list[list[float]]
    ) -> None:
        if len(keys) != self.num_key_value_heads:
            raise ValueError("key head count does not match cache")
        if len(values) != self.num_key_value_heads:
            raise ValueError("value head count does not match cache")
        for key, value in zip(keys, values):
            if len(key) != self.head_dim or len(value) != self.head_dim:
                raise ValueError("key/value head width does not match cache")
        for head, (key, value) in enumerate(zip(keys, values)):
            self.keys[head].append(list(key))
            self.values[head].append(list(value))
        self.tokens += 1

    def clear(self) -> None:
        self.tokens = 0
        for keys, values in zip(self.keys, self.values):
            keys.clear()
            values.clear()
        # Device storage is owned by this logical cache. Drop every device
        # reference as well so prefix-cache eviction and state clearing can
        # release VRAM and cannot reuse arrays from a previous cache mode.
        self.cuda_keys = None
        self.cuda_values = None
        self.cuda_key_scales = None
        self.cuda_value_scales = None
        self.cuda_cache_type = None
        self.cuda_capacity = 0


class QwenFullAttentionLayer:
    """Executable Qwen3.5/3.6 full-attention token mixer for one token."""

    def __init__(self, layer_file: Path | str):
        self.layer_file = Path(layer_file)
        container = ColiTensorFile(self.layer_file)
        metadata = container.metadata
        self.layer = int(metadata["layer"])
        self.hidden_size = int(metadata["hidden_size"])
        self.num_attention_heads = int(metadata["num_attention_heads"])
        self.num_key_value_heads = int(metadata["num_key_value_heads"])
        self.head_dim = int(metadata["head_dim"])
        self.rotary_dim = int(metadata["rotary_dim"])
        self.rope_theta = float(metadata["rope_theta"])
        self.rms_norm_eps = float(metadata["rms_norm_eps"])
        self.input_norm = BF16Tensor.from_container(
            container, "input_layernorm.weight"
        )
        self.q_projection = load_matrix(container, "q_proj.weight")
        self.k_projection = load_matrix(container, "k_proj.weight")
        self.v_projection = load_matrix(container, "v_proj.weight")
        self.o_projection = load_matrix(container, "o_proj.weight")
        self.q_norm = BF16Tensor.from_container(container, "q_norm.weight")
        self.k_norm = BF16Tensor.from_container(container, "k_norm.weight")
        self._input_norm_weights = self.input_norm.values()
        self._q_norm_weights = self.q_norm.values()
        self._k_norm_weights = self.k_norm.values()
        self._validate()

    @classmethod
    def from_model_directory(
        cls, root: Path | str, layer: int
    ) -> "QwenFullAttentionLayer":
        return cls(Path(root) / "attention_layers" / f"layer-{layer:03d}.coli")

    def new_cache(self) -> AttentionKVCache:
        return AttentionKVCache(self.num_key_value_heads, self.head_dim)

    def forward(
        self,
        hidden: list[float],
        position: int,
        cache: AttentionKVCache,
    ) -> AttentionResult:
        if len(hidden) != self.hidden_size:
            raise ValueError(
                f"expected hidden width {self.hidden_size}, got {len(hidden)}"
            )
        self._validate_cache(cache, position)
        from .cuda import active_cuda

        accelerator = active_cuda()
        if accelerator is not None:
            output, weights = accelerator.full_attention(
                self, hidden, position, cache, residual=False
            )
            return AttentionResult(output=output, attention_weights=weights)
        normalized = _rms_norm(
            hidden, self._input_norm_weights, self.rms_norm_eps
        )
        projected_queries = self.q_projection.matvec(normalized)
        projected_keys = self.k_projection.matvec(normalized)
        projected_values = self.v_projection.matvec(normalized)

        queries: list[list[float]] = []
        gates: list[float] = []
        query_stride = self.head_dim * 2
        for head in range(self.num_attention_heads):
            start = head * query_stride
            query = projected_queries[start : start + self.head_dim]
            gates.extend(
                projected_queries[
                    start + self.head_dim : start + query_stride
                ]
            )
            normalized_query = _rms_norm(
                query, self._q_norm_weights, self.rms_norm_eps
            )
            queries.append(self._apply_rope(normalized_query, position))

        keys: list[list[float]] = []
        values: list[list[float]] = []
        for head in range(self.num_key_value_heads):
            start = head * self.head_dim
            key = projected_keys[start : start + self.head_dim]
            normalized_key = _rms_norm(
                key, self._k_norm_weights, self.rms_norm_eps
            )
            keys.append(self._apply_rope(normalized_key, position))
            values.append(projected_values[start : start + self.head_dim])
        cache.append(keys, values)

        groups = self.num_attention_heads // self.num_key_value_heads
        scale = self.head_dim**-0.5
        attention_output: list[float] = []
        all_weights: list[tuple[float, ...]] = []
        for query_head, query in enumerate(queries):
            kv_head = query_head // groups
            scores = [
                sum(q * k for q, k in zip(query, key)) * scale
                for key in cache.keys[kv_head]
            ]
            weights = _softmax(scores)
            all_weights.append(tuple(weights))
            for column in range(self.head_dim):
                attention_output.append(
                    sum(
                        weight * value[column]
                        for weight, value in zip(weights, cache.values[kv_head])
                    )
                )

        gated_output = [
            value * _sigmoid(gate)
            for value, gate in zip(attention_output, gates)
        ]
        output = self.o_projection.matvec(gated_output)
        return AttentionResult(output=output, attention_weights=tuple(all_weights))

    def forward_residual(
        self,
        hidden: list[float],
        position: int,
        cache: AttentionKVCache,
        *,
        return_attention_weights: bool = True,
    ) -> AttentionResult:
        from .cuda import active_cuda

        accelerator = active_cuda()
        if accelerator is not None:
            if len(hidden) != self.hidden_size:
                raise ValueError(
                    f"expected hidden width {self.hidden_size}, got {len(hidden)}"
                )
            self._validate_cache(cache, position)
            output, weights = accelerator.full_attention(
                self,
                hidden,
                position,
                cache,
                residual=True,
                return_attention_weights=return_attention_weights,
            )
            return AttentionResult(output=output, attention_weights=weights)
        result = self.forward(hidden, position, cache)
        return AttentionResult(
            output=[residual + value for residual, value in zip(hidden, result.output)],
            attention_weights=result.attention_weights,
        )

    def _apply_rope(self, vector: list[float], position: int) -> list[float]:
        rotary = vector[: self.rotary_dim]
        passthrough = vector[self.rotary_dim :]
        half = self.rotary_dim // 2
        first = rotary[:half]
        second = rotary[half:]
        frequencies = [
            position / (self.rope_theta ** (2 * index / self.rotary_dim))
            for index in range(half)
        ]
        cosines = [math.cos(value) for value in frequencies]
        sines = [math.sin(value) for value in frequencies]
        rotated = [
            first[index] * cosines[index] - second[index] * sines[index]
            for index in range(half)
        ] + [
            second[index] * cosines[index] + first[index] * sines[index]
            for index in range(half)
        ]
        return rotated + passthrough

    def _validate_cache(self, cache: AttentionKVCache, position: int) -> None:
        if cache.num_key_value_heads != self.num_key_value_heads:
            raise ValueError("cache key/value head count does not match layer")
        if cache.head_dim != self.head_dim:
            raise ValueError("cache head dimension does not match layer")
        if position != cache.length:
            raise ValueError(
                f"incremental position {position} must equal cache length {cache.length}"
            )

    def _validate(self) -> None:
        expected = {
            "input norm": (self.hidden_size,),
            "q projection": (
                self.num_attention_heads * self.head_dim * 2,
                self.hidden_size,
            ),
            "k projection": (
                self.num_key_value_heads * self.head_dim,
                self.hidden_size,
            ),
            "v projection": (
                self.num_key_value_heads * self.head_dim,
                self.hidden_size,
            ),
            "o projection": (
                self.hidden_size,
                self.num_attention_heads * self.head_dim,
            ),
            "q norm": (self.head_dim,),
            "k norm": (self.head_dim,),
        }
        actual = {
            "input norm": self.input_norm.shape,
            "q projection": self.q_projection.shape,
            "k projection": self.k_projection.shape,
            "v projection": self.v_projection.shape,
            "o projection": self.o_projection.shape,
            "q norm": self.q_norm.shape,
            "k norm": self.k_norm.shape,
        }
        mismatches = [
            f"{name} {actual[name]} != {shape}"
            for name, shape in expected.items()
            if actual[name] != shape
        ]
        if mismatches:
            raise ValueError(f"invalid attention layer: {', '.join(mismatches)}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by key/value heads")
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim:
            raise ValueError("invalid rotary dimension")
        if self.rotary_dim % 2:
            raise ValueError("rotary dimension must be even")


def _rms_norm(
    vector: list[float], weights: list[float], epsilon: float
) -> list[float]:
    variance = sum(value * value for value in vector) / len(vector)
    inverse_rms = 1.0 / math.sqrt(variance + epsilon)
    return [
        value * inverse_rms * (1.0 + weight)
        for value, weight in zip(vector, weights)
    ]


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
