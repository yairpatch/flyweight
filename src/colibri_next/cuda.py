from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .bf16 import BF16Tensor
    from .kernels import Q4SwiGLUExpert
    from .q4 import Q4BlockTensor


THREADS_PER_BLOCK = 256


def _group_selected_experts(
    selected_ids: list[list[int]],
) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for token, route in enumerate(selected_ids):
        for rank, expert_id in enumerate(route):
            groups.setdefault(int(expert_id), []).append((token, rank))
    return groups


def _use_expert_major_prefill(tokens: int) -> bool:
    mode = os.environ.get("COLIBRI_EXPERT_MAJOR_PREFILL", "auto")
    if mode not in {"0", "1", "auto"}:
        raise ValueError(
            "COLIBRI_EXPERT_MAJOR_PREFILL must be 'auto', '0', or '1'"
        )
    minimum = int(os.environ.get("COLIBRI_EXPERT_MAJOR_MIN_TOKENS", "128"))
    if minimum <= 0:
        raise ValueError("COLIBRI_EXPERT_MAJOR_MIN_TOKENS must be positive")
    return mode == "1" or (mode == "auto" and tokens >= minimum)


class CudaUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class _CacheEntry:
    owner: object
    arrays: tuple[Any, ...]
    byte_size: int
    protected: bool


@dataclass(slots=True)
class _PendingPrefetch:
    event: Any
    pinned_buffers: tuple[Any, ...]


class CudaAccelerator:
    """CuPy CUDA execution with a bounded packed-weight cache."""

    def __init__(self, *, cache_mib: int = 8192, device_id: int = 0):
        if cache_mib <= 0:
            raise ValueError("cache_mib must be positive")
        os.environ.setdefault(
            "CUPY_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "colibri-next-cupy-cache"),
        )
        try:
            import cupy as cp
        except (ImportError, OSError) as error:
            raise CudaUnavailableError(
                "CUDA execution requires CuPy; install cupy-cuda13x or "
                "the CuPy wheel matching your CUDA runtime"
            ) from error
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
        except Exception as error:
            raise CudaUnavailableError(f"CUDA initialization failed: {error}") from error
        if device_id < 0 or device_id >= device_count:
            raise CudaUnavailableError(
                f"CUDA device {device_id} unavailable; found {device_count} device(s)"
            )
        self.cp = cp
        self.device_id = device_id
        self.cache_limit_bytes = cache_mib * 1024 * 1024
        self.cache_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.device_resident_decode_tokens = 0
        self.batched_prefill_tokens = 0
        self.batched_moe_tokens = 0
        self.batched_expert_groups = 0
        self.q8_grouped_moe_calls = 0
        self.q8_moe_enabled = os.environ.get("COLIBRI_Q8_MOE") == "1"
        self.expert_prefetch_enabled = (
            os.environ.get("COLIBRI_EXPERT_PREFETCH") == "1"
            and os.environ.get("COLIBRI_DISABLE_EXPERT_PREFETCH") != "1"
        )
        self.expert_prefetch_budget = max(
            0, int(os.environ.get("COLIBRI_EXPERT_PREFETCH_BUDGET", "2"))
        )
        self.expert_prefetch_requests = 0
        self.expert_prefetch_hits = 0
        self.expert_prefetch_waits = 0
        self.expert_prefetch_uses = 0
        self.expert_prefetch_bytes = 0
        self._cache: OrderedDict[tuple[str, int], _CacheEntry] = OrderedDict()
        self._unused_prefetches: set[tuple[str, int]] = set()
        self._pending_prefetches: OrderedDict[
            tuple[str, int], _PendingPrefetch
        ] = OrderedDict()
        self._rope_cache: dict[tuple[int, float, int], tuple[Any, Any]] = {}
        with cp.cuda.Device(device_id):
            self._prefetch_stream = cp.cuda.Stream(non_blocking=True)
            self._bf16_kernel = cp.RawKernel(_KERNEL_SOURCE, "bf16_matvec")
            self._bf16_matmul_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "bf16_matmul"
            )
            self._q4_matmul_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q4_matmul"
            )
            self._rms_norm_rows_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "rms_norm_rows"
            )
            self._rms_norm_kernel = cp.RawKernel(_KERNEL_SOURCE, "rms_norm")
            self._route_topk_kernel = cp.RawKernel(_KERNEL_SOURCE, "route_topk")
            self._route_topk_rows_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "route_topk_rows"
            )
            self._q4_kernel = cp.RawKernel(_KERNEL_SOURCE, "q4_matvec")
            self._silu_mul_kernel = cp.RawKernel(_KERNEL_SOURCE, "silu_mul")
            self._scaled_add_kernel = cp.RawKernel(_KERNEL_SOURCE, "scaled_add")
            self._q4_batched_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q4_batched_matvec"
            )
            self._silu_mul_batched_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "silu_mul_batched"
            )
            self._q4_weighted_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q4_batched_weighted_matvec"
            )
            self._quantize_q8_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "quantize_q8_blocks"
            )
            self._q4_q8_batched_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q4_q8_batched_matvec"
            )
            self._q4_q8_weighted_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q4_q8_batched_weighted_matvec"
            )
            self._delta_conv_sequence_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "delta_conv_sequence"
            )
            self._delta_recurrent_sequence_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "delta_recurrent_sequence"
            )

    @property
    def device_name(self) -> str:
        properties = self.cp.cuda.runtime.getDeviceProperties(self.device_id)
        name = properties["name"]
        return name.decode() if isinstance(name, bytes) else str(name)

    def device_vector(self, vector: object) -> Any:
        with self.cp.cuda.Device(self.device_id):
            return self.cp.asarray(vector, dtype=self.cp.float32)

    def device_to_host(self, vector: Any) -> list[float]:
        return vector.get().tolist()

    def rms_norm_device(
        self,
        hidden: Any,
        weights: list[float],
        epsilon: float,
        *,
        one_centered: bool = True,
    ) -> Any:
        cp = self.cp
        device_weights = self._float32_array(weights)
        output = cp.empty_like(hidden)
        self._rms_norm_kernel(
            (1,),
            (THREADS_PER_BLOCK,),
            (
                hidden,
                device_weights,
                output,
                hidden.size,
                cp.float32(epsilon),
                int(one_centered),
            ),
        )
        return output

    def matrix_matvec_device(self, tensor: Any, vector: Any) -> Any:
        with self.cp.cuda.Device(self.device_id):
            return self._matrix_matvec_device(tensor, vector)

    def matrix_matmul_device(
        self, tensor: Any, vectors: Any, *, protected: bool = True
    ) -> Any:
        from .q4 import Q4BlockTensor

        if vectors.ndim != 2 or vectors.shape[1] != tensor.shape[1]:
            raise ValueError("matrix batch width does not match weight tensor")
        rows, columns = tensor.shape
        tokens = int(vectors.shape[0])
        cp = self.cp
        output = cp.empty((tokens, rows), dtype=cp.float32)
        if isinstance(tensor, Q4BlockTensor):
            packed, scales = self._q4_arrays(tensor, protected=protected)
            self._q4_matmul_kernel(
                (rows, tokens),
                (THREADS_PER_BLOCK,),
                (packed, scales, vectors, output, rows, columns, tokens),
            )
            return output
        (weights,) = self._cached_arrays(
            "bf16",
            tensor,
            len(tensor.data),
            lambda: (
                cp.asarray(memoryview(tensor.data), dtype=cp.uint8).view(cp.uint16),
            ),
            protected=True,
        )
        self._bf16_matmul_kernel(
            (rows, tokens),
            (THREADS_PER_BLOCK,),
            (weights, vectors, output, rows, columns, tokens),
        )
        return output

    def rms_norm_rows_device(
        self,
        hidden: Any,
        weights: list[float],
        epsilon: float,
        *,
        one_centered: bool = True,
    ) -> Any:
        if hidden.ndim != 2:
            raise ValueError("batched RMSNorm requires a rank-2 matrix")
        cp = self.cp
        output = cp.empty_like(hidden)
        device_weights = self._float32_array(weights)
        self._rms_norm_rows_kernel(
            (int(hidden.shape[0]),),
            (THREADS_PER_BLOCK,),
            (
                hidden,
                device_weights,
                output,
                int(hidden.shape[0]),
                int(hidden.shape[1]),
                cp.float32(epsilon),
                int(one_centered),
            ),
        )
        return output

    def bf16_matvec(self, tensor: BF16Tensor, vector: list[float]) -> list[float]:
        rows, columns = tensor.shape
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            (weights,) = self._cached_arrays(
                "bf16",
                tensor,
                len(tensor.data),
                lambda: (cp.asarray(memoryview(tensor.data), dtype=cp.uint8).view(cp.uint16),),
                protected=True,
            )
            input_vector = cp.asarray(vector, dtype=cp.float32)
            output = cp.empty(rows, dtype=cp.float32)
            self._bf16_kernel(
                (rows,),
                (THREADS_PER_BLOCK,),
                (weights, input_vector, output, rows, columns),
            )
            return output.get().tolist()

    def bf16_matvec_many(
        self,
        tensors: list[BF16Tensor],
        vector: list[float],
    ) -> list[list[float]]:
        if not tensors:
            return []
        columns = tensors[0].shape[1]
        if any(len(tensor.shape) != 2 or tensor.shape[1] != columns for tensor in tensors):
            raise ValueError("BF16 matrices must have the same input width")
        if len(vector) != columns:
            raise ValueError(f"expected vector width {columns}, got {len(vector)}")
        cp = self.cp
        row_counts = [tensor.shape[0] for tensor in tensors]
        with cp.cuda.Device(self.device_id):
            input_vector = cp.asarray(vector, dtype=cp.float32)
            combined = cp.empty(sum(row_counts), dtype=cp.float32)
            offset = 0
            for tensor, rows in zip(tensors, row_counts):
                (weights,) = self._cached_arrays(
                    "bf16",
                    tensor,
                    len(tensor.data),
                    lambda tensor=tensor: (
                        cp.asarray(
                            memoryview(tensor.data), dtype=cp.uint8
                        ).view(cp.uint16),
                    ),
                    protected=True,
                )
                self._bf16_kernel(
                    (rows,),
                    (THREADS_PER_BLOCK,),
                    (
                        weights,
                        input_vector,
                        combined[offset : offset + rows],
                        rows,
                        columns,
                    ),
                )
                offset += rows
            values = combined.get().tolist()
        outputs: list[list[float]] = []
        offset = 0
        for rows in row_counts:
            outputs.append(values[offset : offset + rows])
            offset += rows
        return outputs

    def full_attention(
        self,
        layer: Any,
        hidden: Any,
        position: int,
        cache: Any,
        *,
        residual: bool,
        return_attention_weights: bool = True,
        return_device: bool = False,
    ) -> Any:
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            hidden_device = cp.asarray(hidden, dtype=cp.float32)
            input_norm_weights = self._float32_array(
                layer._input_norm_weights
            )
            inverse_rms = cp.float32(1.0) / cp.sqrt(
                cp.mean(hidden_device * hidden_device) + layer.rms_norm_eps
            )
            normalized = hidden_device * inverse_rms * (
                cp.float32(1.0) + input_norm_weights
            )
            projected_queries = self._matrix_matvec_device(
                layer.q_projection, normalized
            )
            projected_keys = self._matrix_matvec_device(
                layer.k_projection, normalized
            )
            projected_values = self._matrix_matvec_device(
                layer.v_projection, normalized
            )

            query_gate = projected_queries.reshape(
                layer.num_attention_heads, 2, layer.head_dim
            )
            queries = query_gate[:, 0, :]
            gates = query_gate[:, 1, :]
            query_norm_weights = self._float32_array(layer._q_norm_weights)
            queries *= cp.float32(1.0) / cp.sqrt(
                cp.mean(queries * queries, axis=1, keepdims=True)
                + layer.rms_norm_eps
            )
            queries *= cp.float32(1.0) + query_norm_weights[None, :]

            keys = projected_keys.reshape(
                layer.num_key_value_heads, layer.head_dim
            )
            key_norm_weights = self._float32_array(layer._k_norm_weights)
            keys *= cp.float32(1.0) / cp.sqrt(
                cp.mean(keys * keys, axis=1, keepdims=True)
                + layer.rms_norm_eps
            )
            keys *= cp.float32(1.0) + key_norm_weights[None, :]
            cosines, sines = self._rope_factors(
                layer.rotary_dim, layer.rope_theta, position
            )
            queries = self._apply_rope_device(
                queries, layer.rotary_dim, cosines, sines
            )
            keys = self._apply_rope_device(
                keys, layer.rotary_dim, cosines, sines
            )
            values = projected_values.reshape(
                layer.num_key_value_heads, layer.head_dim
            )
            self._append_attention_cache(cache, keys, values)

            groups = layer.num_attention_heads // layer.num_key_value_heads
            grouped_queries = queries.reshape(
                layer.num_key_value_heads, groups, layer.head_dim
            )
            cached_keys = cache.cuda_keys[:, : cache.tokens, :]
            cached_values = cache.cuda_values[:, : cache.tokens, :]
            scores = cp.einsum(
                "hgd,htd->hgt", grouped_queries, cached_keys
            ) * cp.float32(layer.head_dim**-0.5)
            scores -= cp.max(scores, axis=2, keepdims=True)
            attention_weights = cp.exp(scores)
            attention_weights /= cp.sum(
                attention_weights, axis=2, keepdims=True
            )
            context = cp.einsum(
                "hgt,htd->hgd", attention_weights, cached_values
            ).reshape(layer.num_attention_heads, layer.head_dim)
            gated = context * (
                cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-gates))
            )
            output = self._matrix_matvec_device(
                layer.o_projection, gated.reshape(-1)
            )
            if residual:
                output += hidden_device
            if return_device:
                return output, (attention_weights if return_attention_weights else None)
            output_values = output.get().tolist()
            if not return_attention_weights:
                return output_values, ()
            weight_values = attention_weights.reshape(
                layer.num_attention_heads, cache.tokens
            ).get().tolist()
            return output_values, tuple(
                tuple(head_weights) for head_weights in weight_values
            )

    def full_attention_sequence(
        self, layer: Any, hidden: Any, position: int, cache: Any
    ) -> Any:
        outputs = []
        for offset in range(int(hidden.shape[0])):
            output, _ = self.full_attention(
                layer,
                hidden[offset],
                position + offset,
                cache,
                residual=True,
                return_attention_weights=False,
                return_device=True,
            )
            outputs.append(output)
        return self.cp.stack(outputs)

    def _rope_factors(
        self, rotary_dim: int, rope_theta: float, position: int
    ) -> tuple[Any, Any]:
        key = (rotary_dim, rope_theta, position)
        factors = self._rope_cache.get(key)
        if factors is not None:
            return factors
        cp = self.cp
        half = rotary_dim // 2
        indices = cp.arange(half, dtype=cp.float32)
        frequencies = cp.float32(position) / cp.power(
            cp.float32(rope_theta),
            cp.float32(2.0) * indices / cp.float32(rotary_dim),
        )
        factors = cp.cos(frequencies), cp.sin(frequencies)
        self._rope_cache[key] = factors
        return factors

    def _apply_rope_device(
        self, vectors: Any, rotary_dim: int, cosines: Any, sines: Any
    ) -> Any:
        cp = self.cp
        half = rotary_dim // 2
        output = cp.empty_like(vectors)
        first = vectors[:, :half]
        second = vectors[:, half:rotary_dim]
        output[:, :half] = first * cosines - second * sines
        output[:, half:rotary_dim] = second * cosines + first * sines
        if rotary_dim < vectors.shape[1]:
            output[:, rotary_dim:] = vectors[:, rotary_dim:]
        return output

    def _append_attention_cache(
        self, cache: Any, keys: Any, values: Any
    ) -> None:
        cp = self.cp
        if cache.cuda_keys is None:
            cache.cuda_capacity = max(256, cache.tokens + 1)
            cache.cuda_keys = cp.empty(
                (
                    cache.num_key_value_heads,
                    cache.cuda_capacity,
                    cache.head_dim,
                ),
                dtype=cp.float32,
            )
            cache.cuda_values = cp.empty_like(cache.cuda_keys)
        elif cache.tokens >= cache.cuda_capacity:
            capacity = cache.cuda_capacity * 2
            grown_keys = cp.empty(
                (cache.num_key_value_heads, capacity, cache.head_dim),
                dtype=cp.float32,
            )
            grown_values = cp.empty_like(grown_keys)
            grown_keys[:, : cache.tokens, :] = cache.cuda_keys[
                :, : cache.tokens, :
            ]
            grown_values[:, : cache.tokens, :] = cache.cuda_values[
                :, : cache.tokens, :
            ]
            cache.cuda_keys = grown_keys
            cache.cuda_values = grown_values
            cache.cuda_capacity = capacity
        cache.cuda_keys[:, cache.tokens, :] = keys
        cache.cuda_values[:, cache.tokens, :] = values
        cache.tokens += 1

    def gated_delta(
        self,
        layer: Any,
        hidden: Any,
        state: Any,
        *,
        return_device: bool = False,
    ) -> Any:
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            hidden_device = cp.asarray(hidden, dtype=cp.float32)
            input_norm_weights = self._float32_array(
                layer._input_norm_weights
            )
            inverse_rms = cp.float32(1.0) / cp.sqrt(
                cp.mean(hidden_device * hidden_device) + layer.rms_norm_eps
            )
            normalized = hidden_device * inverse_rms * (
                cp.float32(1.0) + input_norm_weights
            )
            mixed_qkv, z, beta_logits, decay_logits = [
                self._matrix_matvec_device(tensor, normalized)
                for tensor in (
                    layer.in_proj_qkv,
                    layer.in_proj_z,
                    layer.in_proj_b,
                    layer.in_proj_a,
                )
            ]

            if state.cuda_conv_state is None:
                state.cuda_conv_state = cp.asarray(
                    state.conv_state, dtype=cp.float32
                )
                state.cuda_recurrent_state = cp.asarray(
                    state.recurrent_state, dtype=cp.float32
                )
            conv_state = state.cuda_conv_state
            recurrent_state = state.cuda_recurrent_state
            conv_state[:, :-1] = conv_state[:, 1:]
            conv_state[:, -1] = mixed_qkv
            conv_weights = self._float32_array(layer._conv_weights).reshape(
                layer.conv_dim, layer.conv_kernel_size
            )
            convolved = cp.sum(conv_state * conv_weights, axis=1)
            convolved = convolved / (
                cp.float32(1.0) + cp.exp(-convolved)
            )

            queries = convolved[: layer.key_dim].reshape(
                layer.num_key_heads, layer.key_head_dim
            )
            keys = convolved[layer.key_dim : layer.key_dim * 2].reshape(
                layer.num_key_heads, layer.key_head_dim
            )
            values = convolved[layer.key_dim * 2 :].reshape(
                layer.num_value_heads, layer.value_head_dim
            )
            repeats = layer.num_value_heads // layer.num_key_heads
            queries = cp.repeat(queries, repeats, axis=0)
            keys = cp.repeat(keys, repeats, axis=0)
            queries = queries / cp.sqrt(
                cp.sum(queries * queries, axis=1, keepdims=True)
                + cp.float32(1e-6)
            )
            keys = keys / cp.sqrt(
                cp.sum(keys * keys, axis=1, keepdims=True)
                + cp.float32(1e-6)
            )
            queries *= cp.float32(layer.key_head_dim**-0.5)

            beta = cp.float32(1.0) / (
                cp.float32(1.0) + cp.exp(-beta_logits)
            )
            a_log = self._float32_array(layer._a_log)
            dt_bias = self._float32_array(layer._dt_bias)
            decay = -cp.exp(a_log) * cp.logaddexp(
                cp.float32(0.0), decay_logits + dt_bias
            )
            recurrent_state *= cp.exp(decay)[:, None, None]
            memory = cp.einsum("hkv,hk->hv", recurrent_state, keys)
            delta = (values - memory) * beta[:, None]
            recurrent_state += keys[:, :, None] * delta[:, None, :]
            core = cp.einsum("hkv,hk->hv", recurrent_state, queries)
            core_inverse_rms = cp.float32(1.0) / cp.sqrt(
                cp.mean(core * core, axis=1, keepdims=True)
                + layer.rms_norm_eps
            )
            gates = z.reshape(
                layer.num_value_heads, layer.value_head_dim
            )
            silu_gates = gates / (
                cp.float32(1.0) + cp.exp(-gates)
            )
            norm_weights = self._float32_array(layer._norm_weights)
            core = (
                core
                * core_inverse_rms
                * norm_weights[None, :]
                * silu_gates
            )
            output = self._matrix_matvec_device(
                layer.out_proj, core.reshape(-1)
            )
            if return_device:
                return output
            return output.get().tolist()

    def gated_delta_sequence(
        self, layer: Any, hidden: Any, state: Any
    ) -> Any:
        cp = self.cp
        tokens = int(hidden.shape[0])
        normalized = self.rms_norm_rows_device(
            hidden, layer._input_norm_weights, layer.rms_norm_eps
        )
        mixed_qkv, z, beta_logits, decay_logits = [
            self.matrix_matmul_device(tensor, normalized)
            for tensor in (
                layer.in_proj_qkv,
                layer.in_proj_z,
                layer.in_proj_b,
                layer.in_proj_a,
            )
        ]
        if state.cuda_conv_state is None:
            state.cuda_conv_state = cp.asarray(
                state.conv_state, dtype=cp.float32
            )
            state.cuda_recurrent_state = cp.asarray(
                state.recurrent_state, dtype=cp.float32
            )
        if os.environ.get("COLIBRI_FUSED_DELTA_PREFILL", "1") != "0":
            return self._gated_delta_sequence_fused(
                layer,
                hidden,
                state,
                mixed_qkv,
                z,
                beta_logits,
                decay_logits,
            )
        conv_state = state.cuda_conv_state
        recurrent_state = state.cuda_recurrent_state
        conv_weights = self._float32_array(layer._conv_weights).reshape(
            layer.conv_dim, layer.conv_kernel_size
        )
        a_log = self._float32_array(layer._a_log)
        dt_bias = self._float32_array(layer._dt_bias)
        norm_weights = self._float32_array(layer._norm_weights)
        cores = cp.empty((tokens, layer.value_dim), dtype=cp.float32)
        repeats = layer.num_value_heads // layer.num_key_heads
        for token in range(tokens):
            conv_state[:, :-1] = conv_state[:, 1:]
            conv_state[:, -1] = mixed_qkv[token]
            convolved = cp.sum(conv_state * conv_weights, axis=1)
            convolved = convolved / (cp.float32(1.0) + cp.exp(-convolved))
            queries = convolved[: layer.key_dim].reshape(
                layer.num_key_heads, layer.key_head_dim
            )
            keys = convolved[layer.key_dim : layer.key_dim * 2].reshape(
                layer.num_key_heads, layer.key_head_dim
            )
            values = convolved[layer.key_dim * 2 :].reshape(
                layer.num_value_heads, layer.value_head_dim
            )
            queries = cp.repeat(queries, repeats, axis=0)
            keys = cp.repeat(keys, repeats, axis=0)
            queries = queries / cp.sqrt(
                cp.sum(queries * queries, axis=1, keepdims=True)
                + cp.float32(1e-6)
            )
            keys = keys / cp.sqrt(
                cp.sum(keys * keys, axis=1, keepdims=True)
                + cp.float32(1e-6)
            )
            queries *= cp.float32(layer.key_head_dim**-0.5)
            beta = cp.float32(1.0) / (
                cp.float32(1.0) + cp.exp(-beta_logits[token])
            )
            decay = -cp.exp(a_log) * cp.logaddexp(
                cp.float32(0.0), decay_logits[token] + dt_bias
            )
            recurrent_state *= cp.exp(decay)[:, None, None]
            memory = cp.einsum("hkv,hk->hv", recurrent_state, keys)
            delta = (values - memory) * beta[:, None]
            recurrent_state += keys[:, :, None] * delta[:, None, :]
            core = cp.einsum("hkv,hk->hv", recurrent_state, queries)
            inverse_rms = cp.float32(1.0) / cp.sqrt(
                cp.mean(core * core, axis=1, keepdims=True)
                + layer.rms_norm_eps
            )
            gates = z[token].reshape(
                layer.num_value_heads, layer.value_head_dim
            )
            silu_gates = gates / (cp.float32(1.0) + cp.exp(-gates))
            cores[token] = (
                core * inverse_rms * norm_weights[None, :] * silu_gates
            ).reshape(-1)
        state.tokens += tokens
        output = self.matrix_matmul_device(layer.out_proj, cores)
        return output + hidden

    def _gated_delta_sequence_fused(
        self,
        layer: Any,
        hidden: Any,
        state: Any,
        mixed_qkv: Any,
        z: Any,
        beta_logits: Any,
        decay_logits: Any,
    ) -> Any:
        """Run the token-sequential DeltaNet recurrence in two CUDA launches."""
        cp = self.cp
        tokens = int(hidden.shape[0])
        conv_weights = self._float32_array(layer._conv_weights).reshape(
            layer.conv_dim, layer.conv_kernel_size
        )
        convolved = cp.empty((tokens, layer.conv_dim), dtype=cp.float32)
        self._delta_conv_sequence_kernel(
            (layer.conv_dim,),
            (1,),
            (
                mixed_qkv,
                conv_weights,
                state.cuda_conv_state,
                convolved,
                tokens,
                layer.conv_dim,
                layer.conv_kernel_size,
            ),
        )
        cores = cp.empty((tokens, layer.value_dim), dtype=cp.float32)
        self._delta_recurrent_sequence_kernel(
            (layer.num_value_heads,),
            (THREADS_PER_BLOCK,),
            (
                convolved,
                z,
                beta_logits,
                decay_logits,
                self._float32_array(layer._a_log),
                self._float32_array(layer._dt_bias),
                self._float32_array(layer._norm_weights),
                state.cuda_recurrent_state,
                cores,
                tokens,
                layer.num_key_heads,
                layer.num_value_heads,
                layer.key_head_dim,
                layer.value_head_dim,
                cp.float32(layer.rms_norm_eps),
            ),
        )
        state.tokens += tokens
        output = self.matrix_matmul_device(layer.out_proj, cores)
        return output + hidden

    def _matrix_matvec_device(self, tensor: Any, vector: Any) -> Any:
        from .q4 import Q4BlockTensor

        rows, columns = tensor.shape
        cp = self.cp
        output = cp.empty(rows, dtype=cp.float32)
        if isinstance(tensor, Q4BlockTensor):
            packed, scales = self._q4_arrays(tensor, protected=True)
            self._q4_kernel(
                (rows,),
                (THREADS_PER_BLOCK,),
                (packed, scales, vector, output, rows, columns),
            )
            return output
        (weights,) = self._cached_arrays(
            "bf16",
            tensor,
            len(tensor.data),
            lambda: (
                cp.asarray(memoryview(tensor.data), dtype=cp.uint8).view(cp.uint16),
            ),
            protected=True,
        )
        self._bf16_kernel(
            (rows,),
            (THREADS_PER_BLOCK,),
            (weights, vector, output, rows, columns),
        )
        return output

    def _float32_array(self, values: list[float]) -> Any:
        cp = self.cp
        (array,) = self._cached_arrays(
            "float32",
            values,
            len(values) * 4,
            lambda: (cp.asarray(values, dtype=cp.float32),),
            protected=True,
        )
        return array

    def q4_matvec(self, tensor: Q4BlockTensor, vector: list[float]) -> list[float]:
        rows, columns = tensor.shape
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            packed, scales = self._q4_arrays(tensor)
            input_vector = cp.asarray(vector, dtype=cp.float32)
            output = cp.empty(rows, dtype=cp.float32)
            self._q4_kernel(
                (rows,),
                (THREADS_PER_BLOCK,),
                (packed, scales, input_vector, output, rows, columns),
            )
            return output.get().tolist()

    def q4_swiglu(
        self,
        gate_up: Q4BlockTensor,
        down: Q4BlockTensor,
        vector: list[float],
    ) -> list[float]:
        gate_rows, hidden_size = gate_up.shape
        intermediate_size = gate_rows // 2
        output_size, down_columns = down.shape
        if gate_rows % 2 or down_columns != intermediate_size:
            raise ValueError("invalid Q4 SwiGLU tensor geometry")
        if len(vector) != hidden_size:
            raise ValueError(
                f"expected vector width {hidden_size}, got {len(vector)}"
            )
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            input_vector = cp.asarray(vector, dtype=cp.float32)
            output = self._q4_swiglu_device(gate_up, down, input_vector)
            return output.get().tolist()

    def q4_moe(
        self,
        experts: list[Q4SwiGLUExpert],
        routing_weights: list[float],
        shared_expert: Q4SwiGLUExpert,
        shared_weight: float,
        vector: list[float],
    ) -> list[float]:
        if len(experts) != len(routing_weights):
            raise ValueError("expert and routing-weight counts must match")
        shared_expert.validate()
        hidden_size = shared_expert.hidden_size
        if len(vector) != hidden_size:
            raise ValueError(
                f"expected vector width {hidden_size}, got {len(vector)}"
            )
        all_experts = [*experts, shared_expert]
        for expert in all_experts:
            expert.validate()
        geometry = (
            shared_expert.gate_up.shape,
            shared_expert.down.shape,
        )
        if all(
            (expert.gate_up.shape, expert.down.shape) == geometry
            for expert in all_experts
        ):
            with self.cp.cuda.Device(self.device_id):
                return self._q4_grouped_moe(
                    all_experts,
                    [*routing_weights, shared_weight],
                    vector,
                )
        cp = self.cp
        blocks = (hidden_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        with cp.cuda.Device(self.device_id):
            input_vector = cp.asarray(vector, dtype=cp.float32)
            output = cp.zeros(hidden_size, dtype=cp.float32)
            for expert, weight in zip(experts, routing_weights):
                expert_output = self._q4_swiglu_device(
                    expert.gate_up, expert.down, input_vector
                )
                self._scaled_add_kernel(
                    (blocks,),
                    (THREADS_PER_BLOCK,),
                    (output, expert_output, cp.float32(weight), hidden_size),
                )
            shared_output = self._q4_swiglu_device(
                shared_expert.gate_up,
                shared_expert.down,
                input_vector,
                protect_weights=True,
            )
            self._scaled_add_kernel(
                (blocks,),
                (THREADS_PER_BLOCK,),
                (output, shared_output, cp.float32(shared_weight), hidden_size),
            )
            return output.get().tolist()

    def q4_moe_device(
        self,
        experts: list[Q4SwiGLUExpert],
        routing_weights: Any,
        shared_expert: Q4SwiGLUExpert,
        shared_weight: Any,
        vector: Any,
    ) -> Any:
        all_experts = [*experts, shared_expert]
        geometry = (shared_expert.gate_up.shape, shared_expert.down.shape)
        if not all(
            (expert.gate_up.shape, expert.down.shape) == geometry
            for expert in all_experts
        ):
            raise ValueError(
                "device-resident MoE requires uniform expert geometry"
            )
        cp = self.cp
        weights = cp.concatenate(
            (
                cp.asarray(routing_weights, dtype=cp.float32).reshape(-1),
                cp.asarray(shared_weight, dtype=cp.float32).reshape(1),
            )
        )
        return self._q4_grouped_moe(
            all_experts, weights, vector, return_device=True
        )

    def moe_residual_device(
        self, layer: Any, hidden: Any, route_state: Any | None = None
    ) -> Any:
        if layer.expert_device == "cpu":
            return self._moe_residual_cpu(layer, hidden, route_state)
        cp = self.cp
        normalized = self.rms_norm_device(
            hidden,
            layer._post_attention_norm_weights,
            layer.rms_norm_eps,
        )
        router_logits = self._matrix_matvec_device(layer.router, normalized)
        selected = cp.empty(layer.top_k, dtype=cp.int32)
        routing_weights = cp.empty(layer.top_k, dtype=cp.float32)
        self._route_topk_kernel(
            (1,),
            (THREADS_PER_BLOCK,),
            (
                router_logits,
                selected,
                routing_weights,
                router_logits.size,
                layer.top_k,
            ),
            shared_mem=int(router_logits.size) * 4,
        )
        selected_ids = selected.get().tolist()
        if route_state is not None:
            route_state.last_selected_experts = tuple(
                int(expert_id) for expert_id in selected_ids
            )
        shared_logit = self._matrix_matvec_device(
            layer.shared_gate, normalized
        )[0]
        shared_weight = cp.float32(1.0) / (
            cp.float32(1.0) + cp.exp(-shared_logit)
        )
        experts = [layer._expert(int(expert_id)) for expert_id in selected_ids]
        output = self.q4_moe_device(
            experts,
            routing_weights,
            layer.shared_expert,
            shared_weight,
            normalized,
        )
        return hidden + output

    def _moe_residual_cpu(
        self, layer: Any, hidden: Any, route_state: Any | None
    ) -> Any:
        """Offload one token's routed experts to the host CPU backend.

        The token mixer, router, and residual all stay on the GPU; only the
        Q4 expert MLPs (whose weights are deliberately kept off the device)
        run on the CPU, so just the hidden vector crosses the PCIe boundary.
        """
        result = layer.forward_residual(hidden.get().tolist())
        if route_state is not None:
            route_state.last_selected_experts = result.selected_experts
        return self.device_vector(result.output)

    def _moe_sequence_cpu(
        self, layer: Any, hidden: Any, route_state: Any | None
    ) -> Any:
        """Host-CPU offload of a prefill sequence's routed experts."""
        rows = hidden.get().tolist()
        outputs: list[list[float]] = []
        routes: list[tuple[int, ...]] = []
        for row in rows:
            result = layer.forward_residual(row)
            outputs.append(result.output)
            routes.append(result.selected_experts)
        if route_state is not None and routes:
            route_state.sequence_selected_experts = tuple(routes)
            route_state.last_selected_experts = routes[-1]
        return self.device_vector(outputs)

    def moe_sequence_device(
        self, layer: Any, hidden: Any, route_state: Any | None = None
    ) -> Any:
        if layer.expert_device == "cpu":
            return self._moe_sequence_cpu(layer, hidden, route_state)
        cp = self.cp
        tokens = int(hidden.shape[0])
        normalized = self.rms_norm_rows_device(
            hidden,
            layer._post_attention_norm_weights,
            layer.rms_norm_eps,
        )
        router_logits = self.matrix_matmul_device(layer.router, normalized)
        selected = cp.empty((tokens, layer.top_k), dtype=cp.int32)
        routing_weights = cp.empty(
            (tokens, layer.top_k), dtype=cp.float32
        )
        self._route_topk_rows_kernel(
            (tokens,),
            (THREADS_PER_BLOCK,),
            (
                router_logits,
                selected,
                routing_weights,
                tokens,
                int(router_logits.shape[1]),
                layer.top_k,
            ),
            shared_mem=int(router_logits.shape[1]) * 4,
        )
        selected_ids = selected.get().tolist()
        if route_state is not None and selected_ids:
            route_state.sequence_selected_experts = tuple(
                tuple(int(expert_id) for expert_id in route)
                for route in selected_ids
            )
            route_state.last_selected_experts = (
                route_state.sequence_selected_experts[-1]
            )
        shared_logits = self.matrix_matmul_device(
            layer.shared_gate, normalized
        ).reshape(-1)
        shared_weights = cp.float32(1.0) / (
            cp.float32(1.0) + cp.exp(-shared_logits)
        )
        if not _use_expert_major_prefill(tokens):
            outputs = []
            for token in range(tokens):
                experts = [
                    layer._expert(int(expert_id))
                    for expert_id in selected_ids[token]
                ]
                output = self.q4_moe_device(
                    experts,
                    routing_weights[token],
                    layer.shared_expert,
                    shared_weights[token],
                    normalized[token],
                )
                outputs.append(hidden[token] + output)
            self.batched_moe_tokens += tokens
            return cp.stack(outputs)
        outputs = cp.zeros_like(hidden)
        expert_groups = _group_selected_experts(selected_ids)
        for expert_id, assignments in expert_groups.items():
            token_indices = cp.asarray(
                [token for token, _ in assignments], dtype=cp.int32
            )
            route_indices = cp.asarray(
                [route for _, route in assignments], dtype=cp.int32
            )
            expert = layer._expert(expert_id)
            expert_output = self._q4_swiglu_sequence_device(
                expert, normalized[token_indices], protect_weights=False
            )
            weights = routing_weights[token_indices, route_indices]
            outputs[token_indices] += expert_output * weights[:, None]
        shared_output = self._q4_swiglu_sequence_device(
            layer.shared_expert, normalized, protect_weights=True
        )
        outputs += shared_output * shared_weights[:, None]
        self.batched_moe_tokens += tokens
        self.batched_expert_groups += len(expert_groups)
        return hidden + outputs

    def _q4_swiglu_sequence_device(
        self,
        expert: Q4SwiGLUExpert,
        vectors: Any,
        *,
        protect_weights: bool,
    ) -> Any:
        cp = self.cp
        gate_up = self.matrix_matmul_device(
            expert.gate_up, vectors, protected=protect_weights
        )
        intermediate_size = expert.intermediate_size
        gates = gate_up[:, :intermediate_size]
        activated = (
            gates
            / (cp.float32(1.0) + cp.exp(-gates))
            * gate_up[:, intermediate_size:]
        )
        return self.matrix_matmul_device(
            expert.down, activated, protected=protect_weights
        )

    def _q4_grouped_moe(
        self,
        experts: list[Q4SwiGLUExpert],
        weights: Any,
        vector: Any,
        *,
        return_device: bool = False,
    ) -> Any:
        cp = self.cp
        expert_count = len(experts)
        gate_rows, hidden_size = experts[0].gate_up.shape
        intermediate_size = gate_rows // 2
        output_size = experts[0].down.shape[0]
        gate_packed = []
        gate_scales = []
        down_packed = []
        down_scales = []
        for index, expert in enumerate(experts):
            protected = index + 1 == expert_count
            packed, scales = self._q4_arrays(
                expert.gate_up, protected=protected
            )
            gate_packed.append(packed.data.ptr)
            gate_scales.append(scales.data.ptr)
            packed, scales = self._q4_arrays(
                expert.down, protected=protected
            )
            down_packed.append(packed.data.ptr)
            down_scales.append(scales.data.ptr)
        with cp.cuda.Device(self.device_id):
            addresses = cp.asarray(
                [gate_packed, gate_scales, down_packed, down_scales],
                dtype=cp.uint64,
            )
            input_vector = cp.asarray(vector, dtype=cp.float32)
            gate_output = cp.empty(expert_count * gate_rows, dtype=cp.float32)
            use_q8 = (
                self.q8_moe_enabled
                and hidden_size % 32 == 0
                and intermediate_size % 32 == 0
            )
            if use_q8:
                hidden_blocks = hidden_size // 32
                quantized_input = cp.empty(hidden_size, dtype=cp.int8)
                input_scales = cp.empty(hidden_blocks, dtype=cp.float16)
                self._quantize_q8_kernel(
                    (hidden_blocks,),
                    (32,),
                    (input_vector, quantized_input, input_scales, hidden_size),
                )
                self._q4_q8_batched_kernel(
                    (gate_rows, expert_count),
                    (THREADS_PER_BLOCK,),
                    (
                        addresses[0],
                        addresses[1],
                        quantized_input,
                        input_scales,
                        gate_output,
                        gate_rows,
                        hidden_size,
                        expert_count,
                        1,
                    ),
                )
            else:
                self._q4_batched_kernel(
                    (gate_rows, expert_count),
                    (THREADS_PER_BLOCK,),
                    (
                        addresses[0],
                        addresses[1],
                        input_vector,
                        gate_output,
                        gate_rows,
                        hidden_size,
                        expert_count,
                    ),
                )
            activated = cp.empty(
                expert_count * intermediate_size, dtype=cp.float32
            )
            activation_count = expert_count * intermediate_size
            activation_blocks = (
                activation_count + THREADS_PER_BLOCK - 1
            ) // THREADS_PER_BLOCK
            self._silu_mul_batched_kernel(
                (activation_blocks,),
                (THREADS_PER_BLOCK,),
                (
                    gate_output,
                    activated,
                    intermediate_size,
                    activation_count,
                ),
            )
            output = cp.empty(output_size, dtype=cp.float32)
            device_weights = cp.asarray(weights, dtype=cp.float32)
            if use_q8:
                intermediate_blocks = activation_count // 32
                quantized_activated = cp.empty(activation_count, dtype=cp.int8)
                activated_scales = cp.empty(intermediate_blocks, dtype=cp.float16)
                self._quantize_q8_kernel(
                    (intermediate_blocks,),
                    (32,),
                    (
                        activated,
                        quantized_activated,
                        activated_scales,
                        activation_count,
                    ),
                )
                self._q4_q8_weighted_kernel(
                    (output_size,),
                    (THREADS_PER_BLOCK,),
                    (
                        addresses[2],
                        addresses[3],
                        quantized_activated,
                        activated_scales,
                        device_weights,
                        output,
                        output_size,
                        intermediate_size,
                        expert_count,
                    ),
                )
                self.q8_grouped_moe_calls += 1
            else:
                self._q4_weighted_kernel(
                    (output_size,),
                    (THREADS_PER_BLOCK,),
                    (
                        addresses[2],
                        addresses[3],
                        activated,
                        device_weights,
                        output,
                        output_size,
                        intermediate_size,
                        expert_count,
                    ),
                )
            if return_device:
                return output
            return output.get().tolist()

    def _q4_swiglu_device(
        self,
        gate_up: Q4BlockTensor,
        down: Q4BlockTensor,
        input_vector: Any,
        *,
        protect_weights: bool = False,
    ) -> Any:
        cp = self.cp
        gate_rows, hidden_size = gate_up.shape
        intermediate_size = gate_rows // 2
        output_size = down.shape[0]
        gate_packed, gate_scales = self._q4_arrays(
            gate_up, protected=protect_weights
        )
        down_packed, down_scales = self._q4_arrays(
            down, protected=protect_weights
        )
        gate_output = cp.empty(gate_rows, dtype=cp.float32)
        self._q4_kernel(
            (gate_rows,),
            (THREADS_PER_BLOCK,),
            (
                gate_packed,
                gate_scales,
                input_vector,
                gate_output,
                gate_rows,
                hidden_size,
            ),
        )
        activated = cp.empty(intermediate_size, dtype=cp.float32)
        activation_blocks = (
            intermediate_size + THREADS_PER_BLOCK - 1
        ) // THREADS_PER_BLOCK
        self._silu_mul_kernel(
            (activation_blocks,),
            (THREADS_PER_BLOCK,),
            (gate_output, activated, intermediate_size),
        )
        output = cp.empty(output_size, dtype=cp.float32)
        self._q4_kernel(
            (output_size,),
            (THREADS_PER_BLOCK,),
            (
                down_packed,
                down_scales,
                activated,
                output,
                output_size,
                intermediate_size,
            ),
        )
        return output

    def _q4_arrays(
        self, tensor: Q4BlockTensor, *, protected: bool = False
    ) -> tuple[Any, Any]:
        cp = self.cp
        byte_size = len(tensor.packed) + len(tensor.scales)
        key = ("q4", id(tensor))
        arrays = self._cached_arrays(
            "q4",
            tensor,
            byte_size,
            lambda: (
                cp.asarray(memoryview(tensor.packed), dtype=cp.uint8),
                cp.asarray(memoryview(tensor.scales), dtype=cp.uint8).view(cp.float16),
            ),
            protected=protected,
        )
        if key in self._unused_prefetches:
            self._unused_prefetches.remove(key)
            self.expert_prefetch_uses += 1
        self._wait_for_prefetch(key)
        return arrays

    def prefetch_moe(
        self, layer: Any, selected_experts: tuple[int, ...]
    ) -> None:
        if (
            not self.expert_prefetch_enabled
            or layer.expert_device != "cuda"
            or not selected_experts
            or self.expert_prefetch_budget == 0
        ):
            return
        for expert_id in selected_experts[: self.expert_prefetch_budget]:
            expert = layer._experts.get(int(expert_id))
            if expert is None:
                continue
            self.prefetch_q4(expert.gate_up)
            self.prefetch_q4(expert.down)

    def prefetch_q4(self, tensor: Q4BlockTensor) -> bool:
        if not self.expert_prefetch_enabled:
            return False
        import numpy as np

        cp = self.cp
        self._reap_prefetches()
        key = ("q4", id(tensor))
        entry = self._cache.get(key)
        if entry is not None and entry.owner is tensor:
            self.expert_prefetch_hits += 1
            self._cache.move_to_end(key)
            return False
        byte_size = len(tensor.packed) + len(tensor.scales)
        if byte_size > self.cache_limit_bytes or not self._reserve_cache(
            byte_size, allow_protected=False
        ):
            return False
        pinned_packed = cp.cuda.alloc_pinned_memory(len(tensor.packed))
        pinned_scales = cp.cuda.alloc_pinned_memory(len(tensor.scales))
        host_packed = np.frombuffer(
            pinned_packed, dtype=np.uint8, count=len(tensor.packed)
        )
        host_scales = np.frombuffer(
            pinned_scales, dtype=np.uint8, count=len(tensor.scales)
        )
        host_packed[:] = np.frombuffer(tensor.packed, dtype=np.uint8)
        host_scales[:] = np.frombuffer(tensor.scales, dtype=np.uint8)
        with cp.cuda.Device(self.device_id), self._prefetch_stream:
            packed = cp.empty(len(tensor.packed), dtype=cp.uint8)
            scales_bytes = cp.empty(len(tensor.scales), dtype=cp.uint8)
            packed.set(host_packed, stream=self._prefetch_stream)
            scales_bytes.set(host_scales, stream=self._prefetch_stream)
            event = cp.cuda.Event()
            event.record(self._prefetch_stream)
        arrays = (packed, scales_bytes.view(cp.float16))
        self._cache[key] = _CacheEntry(tensor, arrays, byte_size, False)
        self.cache_bytes += byte_size
        self.cache_misses += 1
        self.expert_prefetch_requests += 1
        self.expert_prefetch_bytes += byte_size
        self._pending_prefetches[key] = _PendingPrefetch(
            event, (pinned_packed, pinned_scales, host_packed, host_scales)
        )
        self._unused_prefetches.add(key)
        return True

    def _wait_for_prefetch(self, key: tuple[str, int]) -> None:
        pending = self._pending_prefetches.get(key)
        if pending is None:
            return
        if not pending.event.done:
            self.cp.cuda.get_current_stream().wait_event(pending.event)
            self.expert_prefetch_waits += 1
        self._reap_prefetches()

    def _reap_prefetches(self) -> None:
        completed = [
            key
            for key, pending in self._pending_prefetches.items()
            if pending.event.done
        ]
        for key in completed:
            self._pending_prefetches.pop(key, None)

    def clear(self) -> None:
        self._prefetch_stream.synchronize()
        self._pending_prefetches.clear()
        self._unused_prefetches.clear()
        self._cache.clear()
        self._rope_cache.clear()
        self.cache_bytes = 0
        with self.cp.cuda.Device(self.device_id):
            self.cp.get_default_memory_pool().free_all_blocks()

    def stats(self) -> dict[str, int | str]:
        protected_entries = [
            entry for entry in self._cache.values() if entry.protected
        ]
        return {
            "device": "cuda",
            "device_name": self.device_name,
            "cache_limit_mib": self.cache_limit_bytes // (1024 * 1024),
            "cache_used_mib": self.cache_bytes // (1024 * 1024),
            "cache_entries": len(self._cache),
            "cache_protected_mib": sum(
                entry.byte_size for entry in protected_entries
            ) // (1024 * 1024),
            "cache_protected_entries": len(protected_entries),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_evictions": self.cache_evictions,
            "device_resident_decode_tokens": self.device_resident_decode_tokens,
            "batched_prefill_tokens": self.batched_prefill_tokens,
            "batched_moe_tokens": self.batched_moe_tokens,
            "batched_expert_groups": self.batched_expert_groups,
            "expert_major_prefill": os.environ.get(
                "COLIBRI_EXPERT_MAJOR_PREFILL", "auto"
            ),
            "expert_major_min_tokens": int(
                os.environ.get("COLIBRI_EXPERT_MAJOR_MIN_TOKENS", "128")
            ),
            "q8_grouped_moe_calls": self.q8_grouped_moe_calls,
            "q8_moe_enabled": str(self.q8_moe_enabled).lower(),
            "expert_prefetch_enabled": str(
                self.expert_prefetch_enabled
            ).lower(),
            "expert_prefetch_budget": self.expert_prefetch_budget,
            "expert_prefetch_requests": self.expert_prefetch_requests,
            "expert_prefetch_hits": self.expert_prefetch_hits,
            "expert_prefetch_waits": self.expert_prefetch_waits,
            "expert_prefetch_uses": self.expert_prefetch_uses,
            "expert_prefetch_mib": self.expert_prefetch_bytes // (1024 * 1024),
            "expert_prefetch_pending": len(self._pending_prefetches),
        }

    def _reserve_cache(
        self, byte_size: int, *, allow_protected: bool
    ) -> bool:
        self._reap_prefetches()
        while self._cache and self.cache_bytes + byte_size > self.cache_limit_bytes:
            eviction_key = next(
                (
                    candidate_key
                    for candidate_key, candidate in self._cache.items()
                    if not candidate.protected
                    and candidate_key not in self._pending_prefetches
                ),
                None,
            )
            if eviction_key is None and allow_protected:
                eviction_key = next(
                    (
                        candidate_key
                        for candidate_key in self._cache
                        if candidate_key not in self._pending_prefetches
                    ),
                    None,
                )
            if eviction_key is None:
                return False
            evicted = self._cache.pop(eviction_key)
            self._unused_prefetches.discard(eviction_key)
            self.cache_bytes -= evicted.byte_size
            self.cache_evictions += 1
        return self.cache_bytes + byte_size <= self.cache_limit_bytes

    def _cached_arrays(
        self,
        kind: str,
        owner: object,
        byte_size: int,
        upload: Callable[[], tuple[Any, ...]],
        *,
        protected: bool = False,
    ) -> tuple[Any, ...]:
        key = (kind, id(owner))
        entry = self._cache.get(key)
        if entry is not None and entry.owner is owner:
            self.cache_hits += 1
            if protected and not entry.protected:
                entry.protected = True
            self._cache.move_to_end(key)
            return entry.arrays
        self.cache_misses += 1
        can_cache = byte_size <= self.cache_limit_bytes and self._reserve_cache(
            byte_size, allow_protected=True
        )
        arrays = upload()
        if can_cache:
            self._cache[key] = _CacheEntry(
                owner, arrays, byte_size, protected
            )
            self.cache_bytes += byte_size
        return arrays


_accelerator: CudaAccelerator | None = None


def configure_cuda(*, cache_mib: int = 8192, device_id: int = 0) -> CudaAccelerator:
    global _accelerator
    if _accelerator is not None:
        _accelerator.clear()
    _accelerator = CudaAccelerator(cache_mib=cache_mib, device_id=device_id)
    return _accelerator


def disable_cuda() -> None:
    global _accelerator
    if _accelerator is not None:
        _accelerator.clear()
    _accelerator = None


def active_cuda() -> CudaAccelerator | None:
    return _accelerator


_KERNEL_SOURCE = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float block_reduce_sum(float value) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
    }
    return value;
}

extern "C" __global__
void delta_conv_sequence(
    const float* mixed_qkv,
    const float* weights,
    float* state,
    float* output,
    const int tokens,
    const int channels,
    const int kernel_size
) {
    const int channel = blockIdx.x;
    if (channel >= channels || threadIdx.x != 0) return;
    float* channel_state = state + channel * kernel_size;
    const float* channel_weights = weights + channel * kernel_size;
    for (int token = 0; token < tokens; ++token) {
        for (int index = 0; index + 1 < kernel_size; ++index) {
            channel_state[index] = channel_state[index + 1];
        }
        channel_state[kernel_size - 1] = mixed_qkv[token * channels + channel];
        float value = 0.0f;
        for (int index = 0; index < kernel_size; ++index) {
            value += channel_state[index] * channel_weights[index];
        }
        output[token * channels + channel] = value / (1.0f + expf(-value));
    }
}

extern "C" __global__
void delta_recurrent_sequence(
    const float* convolved,
    const float* gates,
    const float* beta_logits,
    const float* decay_logits,
    const float* a_log,
    const float* dt_bias,
    const float* norm_weights,
    float* state,
    float* output,
    const int tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float epsilon
) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    if (head >= value_heads) return;
    const int key_head = head / (value_heads / key_heads);
    const int total_key_dim = key_heads * key_dim;
    const int conv_width = total_key_dim * 2 + value_heads * value_dim;
    __shared__ float query_inverse_norm;
    __shared__ float key_inverse_norm;
    __shared__ float decay_scale;
    __shared__ float beta;
    __shared__ float core_values[256];
    for (int token = 0; token < tokens; ++token) {
        const float* row = convolved + token * conv_width;
        const int key_offset = key_head * key_dim;
        if (lane == 0) {
            float query_square = 0.0f;
            float key_square = 0.0f;
            for (int index = 0; index < key_dim; ++index) {
                const float query = row[key_offset + index];
                const float key = row[total_key_dim + key_offset + index];
                query_square += query * query;
                key_square += key * key;
            }
            query_inverse_norm = rsqrtf(query_square + 1.0e-6f)
                * rsqrtf((float)key_dim);
            key_inverse_norm = rsqrtf(key_square + 1.0e-6f);
            beta = 1.0f / (1.0f + expf(-beta_logits[token * value_heads + head]));
            const float softplus_input =
                decay_logits[token * value_heads + head] + dt_bias[head];
            const float softplus = softplus_input > 20.0f
                ? softplus_input : log1pf(expf(softplus_input));
            decay_scale = expf(-expf(a_log[head]) * softplus);
        }
        __syncthreads();
        float core = 0.0f;
        if (lane < value_dim) {
            float memory = 0.0f;
            for (int index = 0; index < key_dim; ++index) {
                const float key = row[total_key_dim + key_offset + index]
                    * key_inverse_norm;
                const int state_index =
                    (head * key_dim + index) * value_dim + lane;
                state[state_index] *= decay_scale;
                memory += state[state_index] * key;
            }
            const float value = row[
                total_key_dim * 2 + head * value_dim + lane
            ];
            const float delta = (value - memory) * beta;
            for (int index = 0; index < key_dim; ++index) {
                const float key = row[total_key_dim + key_offset + index]
                    * key_inverse_norm;
                const int state_index =
                    (head * key_dim + index) * value_dim + lane;
                state[state_index] += key * delta;
            }
        }
        __syncthreads();
        if (lane < value_dim) {
            for (int index = 0; index < key_dim; ++index) {
                const float query = row[key_offset + index] * query_inverse_norm;
                core += state[(head * key_dim + index) * value_dim + lane]
                    * query;
            }
        }
        core_values[lane] = lane < value_dim ? core : 0.0f;
        __syncthreads();
        __shared__ float core_inverse_rms;
        if (lane == 0) {
            float core_square = 0.0f;
            for (int index = 0; index < value_dim; ++index) {
                core_square += core_values[index] * core_values[index];
            }
            core_inverse_rms = rsqrtf(core_square / (float)value_dim + epsilon);
        }
        __syncthreads();
        if (lane < value_dim) {
            const int gate_index =
                token * value_heads * value_dim + head * value_dim + lane;
            const float gate = gates[gate_index];
            const float silu_gate = gate / (1.0f + expf(-gate));
            output[gate_index] = core * core_inverse_rms
                * norm_weights[lane] * silu_gate;
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int pack_signed_chars(
    const int first,
    const int second,
    const int third,
    const int fourth
) {
    return (first & 255)
        | ((second & 255) << 8)
        | ((third & 255) << 16)
        | ((fourth & 255) << 24);
}

__device__ __forceinline__ int q4_q8_dot_block(
    const unsigned char* packed,
    const signed char* vector
) {
    int dot = 0;
    #pragma unroll
    for (int group = 0; group < 8; ++group) {
        const unsigned char first = packed[group * 2];
        const unsigned char second = packed[group * 2 + 1];
        const int weights = pack_signed_chars(
            (first & 15) - 8,
            (first >> 4) - 8,
            (second & 15) - 8,
            (second >> 4) - 8
        );
        const int activations = *((const int*)(vector + group * 4));
        dot = __dp4a(weights, activations, dot);
    }
    return dot;
}

extern "C" __global__
void quantize_q8_blocks(
    const float* input,
    signed char* output,
    __half* scales,
    const int elements
) {
    const int lane = threadIdx.x;
    const int index = blockIdx.x * 32 + lane;
    float value = index < elements ? input[index] : 0.0f;
    float maximum = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    if (lane == 0) scales[blockIdx.x] = __float2half(scale);
    if (index < elements) {
        const int quantized = max(-127, min(127, __float2int_rn(value / scale)));
        output[index] = (signed char)quantized;
    }
}

extern "C" __global__
void q4_q8_batched_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const signed char* vectors,
    const __half* vector_scales,
    float* output,
    const int rows,
    const int columns,
    const int expert_count,
    const int vector_count
) {
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    if (row >= rows || expert >= expert_count) return;
    const unsigned char* packed =
        (const unsigned char*)packed_addresses[expert];
    const __half* scales = (const __half*)scale_addresses[expert];
    const int blocks_per_row = columns >> 5;
    const int vector_index = vector_count == 1 ? 0 : expert;
    float partial = 0.0f;
    for (int block = threadIdx.x; block < blocks_per_row; block += blockDim.x) {
        const int weight_block = row * blocks_per_row + block;
        const int vector_block = vector_index * blocks_per_row + block;
        const int dot = q4_q8_dot_block(
            packed + weight_block * 16,
            vectors + vector_block * 32
        );
        partial += (float)dot
            * __half2float(scales[weight_block])
            * __half2float(vector_scales[vector_block]);
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[expert * rows + row] = partial;
}

extern "C" __global__
void q4_q8_batched_weighted_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const signed char* vectors,
    const __half* vector_scales,
    const float* routing_weights,
    float* output,
    const int rows,
    const int columns,
    const int expert_count
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int blocks_per_row = columns >> 5;
    float partial = 0.0f;
    for (int expert = 0; expert < expert_count; ++expert) {
        const unsigned char* packed =
            (const unsigned char*)packed_addresses[expert];
        const __half* scales = (const __half*)scale_addresses[expert];
        const int vector_offset = expert * blocks_per_row;
        float expert_partial = 0.0f;
        for (int block = threadIdx.x; block < blocks_per_row; block += blockDim.x) {
            const int weight_block = row * blocks_per_row + block;
            const int vector_block = vector_offset + block;
            const int dot = q4_q8_dot_block(
                packed + weight_block * 16,
                vectors + vector_block * 32
            );
            expert_partial += (float)dot
                * __half2float(scales[weight_block])
                * __half2float(vector_scales[vector_block]);
        }
        partial += routing_weights[expert] * expert_partial;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

__device__ __forceinline__ float block_reduce_max(float value) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    __shared__ float warp_maxima[8];
    if (lane == 0) warp_maxima[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warp_maxima[lane] : -3.402823466e+38F;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        }
    }
    return value;
}

extern "C" __global__
void rms_norm(
    const float* input,
    const float* weights,
    float* output,
    const int elements,
    const float epsilon,
    const int one_centered
) {
    float partial = 0.0f;
    for (int index = threadIdx.x; index < elements; index += blockDim.x) {
        partial += input[index] * input[index];
    }
    partial = block_reduce_sum(partial);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(partial / (float)elements + epsilon);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < elements; index += blockDim.x) {
        const float weight = one_centered ? 1.0f + weights[index] : weights[index];
        output[index] = input[index] * inverse_rms * weight;
    }
}

extern "C" __global__
void route_topk(
    const float* logits,
    int* selected,
    float* routing_weights,
    const int experts,
    const int top_k
) {
    extern __shared__ float probabilities[];
    float local_maximum = -3.402823466e+38F;
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        local_maximum = fmaxf(local_maximum, logits[index]);
    }
    local_maximum = block_reduce_max(local_maximum);
    __shared__ float maximum;
    if (threadIdx.x == 0) maximum = local_maximum;
    __syncthreads();
    float local_sum = 0.0f;
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        const float probability = expf(logits[index] - maximum);
        probabilities[index] = probability;
        local_sum += probability;
    }
    local_sum = block_reduce_sum(local_sum);
    __shared__ float denominator;
    if (threadIdx.x == 0) denominator = local_sum;
    __syncthreads();
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        probabilities[index] /= denominator;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float selected_total = 0.0f;
        for (int rank = 0; rank < top_k; ++rank) {
            int best_index = 0;
            float best_value = -1.0f;
            for (int expert = 0; expert < experts; ++expert) {
                if (probabilities[expert] > best_value) {
                    best_value = probabilities[expert];
                    best_index = expert;
                }
            }
            selected[rank] = best_index;
            routing_weights[rank] = best_value;
            selected_total += best_value;
            probabilities[best_index] = -1.0f;
        }
        for (int rank = 0; rank < top_k; ++rank) {
            routing_weights[rank] /= selected_total;
        }
    }
}

extern "C" __global__
void route_topk_rows(
    const float* logits,
    int* selected,
    float* routing_weights,
    const int rows,
    const int experts,
    const int top_k
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float probabilities[];
    const int logits_offset = row * experts;
    const int output_offset = row * top_k;
    float local_maximum = -3.402823466e+38F;
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        local_maximum = fmaxf(local_maximum, logits[logits_offset + index]);
    }
    local_maximum = block_reduce_max(local_maximum);
    __shared__ float maximum;
    if (threadIdx.x == 0) maximum = local_maximum;
    __syncthreads();
    float local_sum = 0.0f;
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        const float probability = expf(logits[logits_offset + index] - maximum);
        probabilities[index] = probability;
        local_sum += probability;
    }
    local_sum = block_reduce_sum(local_sum);
    __shared__ float denominator;
    if (threadIdx.x == 0) denominator = local_sum;
    __syncthreads();
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        probabilities[index] /= denominator;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float selected_total = 0.0f;
        for (int rank = 0; rank < top_k; ++rank) {
            int best_index = 0;
            float best_value = -1.0f;
            for (int expert = 0; expert < experts; ++expert) {
                if (probabilities[expert] > best_value) {
                    best_value = probabilities[expert];
                    best_index = expert;
                }
            }
            selected[output_offset + rank] = best_index;
            routing_weights[output_offset + rank] = best_value;
            selected_total += best_value;
            probabilities[best_index] = -1.0f;
        }
        for (int rank = 0; rank < top_k; ++rank) {
            routing_weights[output_offset + rank] /= selected_total;
        }
    }
}

extern "C" __global__
void bf16_matvec(
    const unsigned short* weights,
    const float* vector,
    float* output,
    const int rows,
    const int columns
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float partial = 0.0f;
    const int start = row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned int bits = ((unsigned int)weights[start + column]) << 16;
        partial += __uint_as_float(bits) * vector[column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void bf16_matmul(
    const unsigned short* weights,
    const float* vectors,
    float* output,
    const int rows,
    const int columns,
    const int tokens
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= rows || token >= tokens) return;
    float partial = 0.0f;
    const int weight_start = row * columns;
    const int vector_start = token * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned int bits = ((unsigned int)weights[weight_start + column]) << 16;
        partial += __uint_as_float(bits) * vectors[vector_start + column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[token * rows + row] = partial;
}

extern "C" __global__
void q4_matmul(
    const unsigned char* packed,
    const __half* scales,
    const float* vectors,
    float* output,
    const int rows,
    const int columns,
    const int tokens
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= rows || token >= tokens) return;
    float partial = 0.0f;
    const int weight_start = row * columns;
    const int vector_start = token * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = weight_start + column;
        const int block = index >> 5;
        const int within_block = index & 31;
        const unsigned char byte = packed[block * 16 + (within_block >> 1)];
        const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
        const float weight = (float)(nibble - 8) * __half2float(scales[block]);
        partial += weight * vectors[vector_start + column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[token * rows + row] = partial;
}

extern "C" __global__
void rms_norm_rows(
    const float* input,
    const float* weights,
    float* output,
    const int rows,
    const int columns,
    const float epsilon,
    const int one_centered
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int start = row * columns;
    float partial = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = input[start + column];
        partial += value * value;
    }
    partial = block_reduce_sum(partial);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(partial / (float)columns + epsilon);
    }
    __syncthreads();
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float weight = one_centered ? 1.0f + weights[column] : weights[column];
        output[start + column] = input[start + column] * inverse_rms * weight;
    }
}

extern "C" __global__
void silu_mul(
    const float* gate_up,
    float* output,
    const int elements
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float gate = gate_up[index];
    const float exponential = gate >= 0.0f ? expf(-gate) : expf(gate);
    const float sigmoid = gate >= 0.0f
        ? 1.0f / (1.0f + exponential)
        : exponential / (1.0f + exponential);
    output[index] = gate * sigmoid * gate_up[elements + index];
}

extern "C" __global__
void scaled_add(
    float* target,
    const float* source,
    const float scale,
    const int elements
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) target[index] += scale * source[index];
}

extern "C" __global__
void q4_batched_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const float* vector,
    float* output,
    const int rows,
    const int columns,
    const int expert_count
) {
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    if (row >= rows || expert >= expert_count) return;
    const unsigned char* packed =
        (const unsigned char*)packed_addresses[expert];
    const __half* scales = (const __half*)scale_addresses[expert];
    float partial = 0.0f;
    const int start = row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = start + column;
        const int block = index >> 5;
        const int within_block = index & 31;
        const unsigned char byte = packed[block * 16 + (within_block >> 1)];
        const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
        const float weight = (float)(nibble - 8) * __half2float(scales[block]);
        partial += weight * vector[column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[expert * rows + row] = partial;
}

extern "C" __global__
void silu_mul_batched(
    const float* gate_up,
    float* output,
    const int intermediate_size,
    const int elements
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int expert = index / intermediate_size;
    const int within_expert = index - expert * intermediate_size;
    const int gate_offset = expert * intermediate_size * 2;
    const float gate = gate_up[gate_offset + within_expert];
    const float exponential = gate >= 0.0f ? expf(-gate) : expf(gate);
    const float sigmoid = gate >= 0.0f
        ? 1.0f / (1.0f + exponential)
        : exponential / (1.0f + exponential);
    output[index] = gate * sigmoid
        * gate_up[gate_offset + intermediate_size + within_expert];
}

extern "C" __global__
void q4_batched_weighted_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const float* vectors,
    const float* routing_weights,
    float* output,
    const int rows,
    const int columns,
    const int expert_count
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float partial = 0.0f;
    const int start = row * columns;
    for (int expert = 0; expert < expert_count; ++expert) {
        const unsigned char* packed =
            (const unsigned char*)packed_addresses[expert];
        const __half* scales = (const __half*)scale_addresses[expert];
        const float* vector = vectors + expert * columns;
        const float route = routing_weights[expert];
        for (int column = threadIdx.x; column < columns; column += blockDim.x) {
            const int index = start + column;
            const int block = index >> 5;
            const int within_block = index & 31;
            const unsigned char byte = packed[block * 16 + (within_block >> 1)];
            const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
            const float weight =
                (float)(nibble - 8) * __half2float(scales[block]);
            partial += route * weight * vector[column];
        }
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void q4_matvec(
    const unsigned char* packed,
    const __half* scales,
    const float* vector,
    float* output,
    const int rows,
    const int columns
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float partial = 0.0f;
    const int start = row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = start + column;
        const int block = index >> 5;
        const int within_block = index & 31;
        const unsigned char byte = packed[block * 16 + (within_block >> 1)];
        const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
        const float weight = (float)(nibble - 8) * __half2float(scales[block]);
        partial += weight * vector[column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}
"""
