from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
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


def ctypes_address(array: object) -> int:
    import ctypes

    return ctypes.cast(array, ctypes.c_void_p).value


def ctypes_element(array: object) -> int:
    """Value of the first pointer stored in a single-entry ctypes array."""
    import ctypes

    return ctypes.cast(array[0], ctypes.c_void_p).value


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


def _use_batched_attention_prefill(tokens: int) -> bool:
    """Limit the quadratic attention workspace for long server prompts.

    The fully batched path materializes attention scores for every prompt
    token against every cached key. That is fast for short prompts but can
    require multiple gigabytes for extension conversations. Long prompts are
    therefore split into bounded attention tiles unless explicitly forced.
    """
    mode = os.environ.get("COLIBRI_BATCHED_ATTENTION_PREFILL", "auto")
    if mode not in {"0", "1", "auto"}:
        raise ValueError(
            "COLIBRI_BATCHED_ATTENTION_PREFILL must be '0', '1', or 'auto'"
        )
    if mode == "0" or (mode == "1"):
        return mode == "1"
    limit = int(os.environ.get("COLIBRI_BATCHED_ATTENTION_MAX_TOKENS", "512"))
    if limit <= 0:
        raise ValueError("COLIBRI_BATCHED_ATTENTION_MAX_TOKENS must be positive")
    return tokens <= limit


def _attention_prefill_chunk_size() -> int:
    size = int(os.environ.get("COLIBRI_ATTENTION_PREFILL_CHUNK", "256"))
    if size <= 0:
        raise ValueError("COLIBRI_ATTENTION_PREFILL_CHUNK must be positive")
    return size


def _kv_cache_type(cache: Any) -> str:
    cache_type = getattr(cache, "cuda_cache_type", None)
    if cache_type is None:
        accelerator = active_cuda()
        cache_type = getattr(accelerator, "kv_cache_type", None)
        if cache_type is None:
            cache_type = os.environ.get("COLIBRI_KV_CACHE_TYPE", "f32").lower()
        if cache_type not in {"f32", "q8"}:
            raise ValueError("COLIBRI_KV_CACHE_TYPE must be 'f32' or 'q8'")
        cache.cuda_cache_type = cache_type
    return cache_type


def _quantize_kv(values: Any, cp: Any) -> tuple[Any, Any]:
    """Symmetrically quantize [..., head_dim] values with row scales."""
    scales = cp.max(cp.abs(values), axis=-1) / cp.float32(127.0)
    scales = cp.maximum(scales, cp.float32(1e-8))
    quantized = cp.clip(
        cp.rint(values / scales[..., None]), -127, 127
    ).astype(cp.int8)
    return quantized, scales


def _dequantize_kv(values: Any, scales: Any, cp: Any) -> Any:
    return values.astype(cp.float32) * scales[..., None]


class CudaUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class _CacheEntry:
    owner: object
    arrays: tuple[Any, ...]
    byte_size: int
    protected: bool
    priority_until: int = 0


@dataclass(slots=True)
class _PendingPrefetch:
    event: Any
    pinned_buffers: tuple[Any, ...]


@dataclass(slots=True)
class _InFlightBuffers:
    event: Any
    arrays: tuple[Any, ...]


class CudaAccelerator:
    """CuPy CUDA execution with a bounded packed-weight cache."""

    def __init__(
        self,
        *,
        cache_mib: int = 8192,
        device_id: int = 0,
        kv_cache_type: str | None = None,
    ):
        if cache_mib <= 0:
            raise ValueError("cache_mib must be positive")
        if kv_cache_type is not None and kv_cache_type not in {"f32", "q8"}:
            raise ValueError("kv_cache_type must be 'f32' or 'q8'")
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
        self.kv_cache_type = kv_cache_type
        self.cache_limit_bytes = cache_mib * 1024 * 1024
        self.cache_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.cache_eviction_scans = 0
        self.cache_eviction_entries_examined = 0
        self.cache_priority_promotions = 0
        self._cache_priority_epoch = 0
        self.profiling = False
        self._profile_events: dict[str, list[tuple[Any, Any]]] = {}
        self._profile_host_seconds: dict[str, float] = {}
        self._profile_host_calls: dict[str, int] = {}
        self.profile_upload_seconds = 0.0
        self.profile_upload_calls = 0
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
        self._expert_load_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="colibri-expert-load"
        )
        self._pending_expert_loads: dict[tuple[int, int], Future[Any]] = {}
        self.expert_load_requests = 0
        self.expert_load_completions = 0
        self._inflight_buffers: list[_InFlightBuffers] = []
        self._moe_address_cache: OrderedDict[
            tuple[int, ...], Any
        ] = OrderedDict()
        self._resident_moe_table: dict[str, Any] | None = None
        self.moe_address_cache_hits = 0
        self.moe_address_cache_misses = 0
        self._rope_cache: dict[tuple[int, float, int], tuple[Any, Any]] = {}
        self._moe_transfer_cache: dict[int, tuple[Any, Any, Any]] = {}
        self._v2_weight_cache: dict[tuple[int, int], Any] = {}
        self._delta_segment_state = "unknown"
        self._native_moe_state = "unknown"
        self._delta_segment_tables: dict[tuple, dict] = {}
        self._delta_segment_scratch_cache: dict | None = None
        with cp.cuda.Device(device_id):
            self._prefetch_stream = cp.cuda.Stream(non_blocking=True)
            self._bf16_kernel = cp.RawKernel(_KERNEL_SOURCE, "bf16_matvec")
            self._bf16_matmul_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "bf16_matmul"
            )
            self._q4_matmul_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q4_matmul"
            )
            self._bf16_matmul_small_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "bf16_matmul_small"
            )
            self._q4_matmul_small_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q4_matmul_small"
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
            self._q8_transposed_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q8_matvec_transposed_warp"
            )
            self._q4k_transposed_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q4k_matvec_transposed"
            )
            self._q5k_transposed_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q5k_matvec_transposed"
            )
            self._q6k_transposed_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q6k_matvec_transposed"
            )
            self._q5k_swiglu_transposed_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q5k_swiglu_transposed"
            )
            self._q6k_accumulate_transposed_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q6k_accumulate_transposed"
            )
            self._q5k_grouped_swiglu_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q5k_grouped_swiglu"
            )
            self._q6k_grouped_accumulate_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q6k_grouped_accumulate"
            )
            self._q8_grouped_accumulate_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q8_grouped_accumulate"
            )
            self._nvfp4_grouped_swiglu_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "nvfp4_grouped_swiglu"
            )
            self._nvfp4_grouped_accumulate_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "nvfp4_grouped_accumulate"
            )
            self._nvfp4_matvec_transposed_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "nvfp4_matvec_transposed"
            )
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
            self._q4_selected_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q4_selected_batched_matvec"
            )
            self._q4_selected_weighted_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "q4_selected_weighted_matvec"
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
            self._delta_conv_step_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "delta_conv_step"
            )
            self._delta_recurrent_sequence_kernel = cp.RawKernel(
                _KERNEL_SOURCE, "delta_recurrent_sequence"
            )

    def enable_profiling(self, enabled: bool = True) -> None:
        self.profiling = enabled
        if enabled:
            self._profile_events.clear()
            self._profile_host_seconds.clear()
            self._profile_host_calls.clear()
            self.profile_upload_seconds = 0.0
            self.profile_upload_calls = 0

    def profile_start(self) -> Any | None:
        if not self.profiling:
            return None
        event = self.cp.cuda.Event()
        event.record()
        return event

    def profile_end(self, name: str, start: Any | None) -> None:
        if start is None:
            return
        end = self.cp.cuda.Event()
        end.record()
        self._profile_events.setdefault(name, []).append((start, end))

    def profile_host(self, name: str, seconds: float) -> None:
        if not self.profiling:
            return
        self._profile_host_seconds[name] = (
            self._profile_host_seconds.get(name, 0.0) + seconds
        )
        self._profile_host_calls[name] = self._profile_host_calls.get(name, 0) + 1

    def _profile_stats(self) -> dict[str, Any]:
        regions = {}
        for name, events in self._profile_events.items():
            milliseconds = sum(
                self.cp.cuda.get_elapsed_time(start, end)
                for start, end in events
            )
            regions[name] = {
                "calls": len(events),
                "milliseconds": milliseconds,
            }
        return {
            "regions": regions,
            "host_regions": {
                name: {
                    "calls": self._profile_host_calls[name],
                    "milliseconds": seconds * 1000.0,
                }
                for name, seconds in self._profile_host_seconds.items()
            },
            "weight_upload_calls": self.profile_upload_calls,
            "weight_upload_milliseconds": self.profile_upload_seconds * 1000.0,
        }

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
        small = tokens <= 8
        if isinstance(tensor, Q4BlockTensor):
            packed, scales = self._q4_arrays(tensor, protected=protected)
            kernel = (
                self._q4_matmul_small_kernel if small else self._q4_matmul_kernel
            )
            kernel(
                (rows,) if small else (rows, tokens),
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
        kernel = (
            self._bf16_matmul_small_kernel if small else self._bf16_matmul_kernel
        )
        kernel(
            (rows,) if small else (rows, tokens),
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
            normalized = self.rms_norm_device(
                hidden_device, layer._input_norm_weights, layer.rms_norm_eps
            )
            combined = self._combined_qkv(layer)
            if combined is not None:
                projected = self._matrix_matvec_device(combined, normalized)
                offsets = layer._combined_qkv_offsets
                projected_queries = projected[: offsets[0]]
                projected_keys = projected[offsets[0] : offsets[1]]
                projected_values = projected[offsets[1] :]
            else:
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
            queries = cp.ascontiguousarray(query_gate[:, 0, :])
            gates = query_gate[:, 1, :]
            queries = self.rms_norm_rows_device(
                queries, layer._q_norm_weights, layer.rms_norm_eps
            )

            keys = projected_keys.reshape(
                layer.num_key_value_heads, layer.head_dim
            )
            keys = self.rms_norm_rows_device(
                keys, layer._k_norm_weights, layer.rms_norm_eps
            )
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
            if _kv_cache_type(cache) == "q8":
                cached_keys = _dequantize_kv(
                    cache.cuda_keys[:, : cache.tokens, :],
                    cache.cuda_key_scales[:, : cache.tokens],
                    cp,
                )
                cached_values = _dequantize_kv(
                    cache.cuda_values[:, : cache.tokens, :],
                    cache.cuda_value_scales[:, : cache.tokens],
                    cp,
                )
            else:
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
        tokens = int(hidden.shape[0])
        if _use_batched_attention_prefill(tokens):
            return self._full_attention_sequence_batched(
                layer, hidden, position, cache
            )
        # Keep prompt prefill batched, but bound the quadratic attention
        # workspace. The previous fallback processed one token at a time,
        # which was memory-safe but made long extension conversations slow.
        outputs = []
        chunk_size = _attention_prefill_chunk_size()
        for start in range(0, tokens, chunk_size):
            outputs.append(
                self._full_attention_sequence_batched(
                    layer,
                    hidden[start : start + chunk_size],
                    position + start,
                    cache,
                )
            )
        return self.cp.concatenate(outputs, axis=0)

    def _full_attention_sequence_batched(
        self, layer: Any, hidden: Any, position: int, cache: Any
    ) -> Any:
        """Prefill every prompt token's attention in a handful of launches.

        Numerically matches the per-token :meth:`full_attention` path (same
        RMSNorm, q/k norm, RoPE, GQA and causal masking) but computes all
        tokens at once instead of looping in Python.
        """
        cp = self.cp
        eps = layer.rms_norm_eps
        heads = layer.num_attention_heads
        kv_heads = layer.num_key_value_heads
        head_dim = layer.head_dim
        with cp.cuda.Device(self.device_id):
            tokens = int(hidden.shape[0])
            normalized = self.rms_norm_rows_device(
                hidden, layer._input_norm_weights, eps
            )
            projected_queries = self.matrix_matmul_device(
                layer.q_projection, normalized
            )
            projected_keys = self.matrix_matmul_device(
                layer.k_projection, normalized
            )
            projected_values = self.matrix_matmul_device(
                layer.v_projection, normalized
            )

            query_gate = projected_queries.reshape(
                tokens, heads, 2, head_dim
            )
            queries = query_gate[:, :, 0, :]
            gates = query_gate[:, :, 1, :]
            query_norm_weights = self._float32_array(layer._q_norm_weights)
            queries = (
                queries
                * (
                    cp.float32(1.0)
                    / cp.sqrt(
                        cp.mean(queries * queries, axis=2, keepdims=True) + eps
                    )
                )
                * (cp.float32(1.0) + query_norm_weights)
            )

            keys = projected_keys.reshape(tokens, kv_heads, head_dim)
            key_norm_weights = self._float32_array(layer._k_norm_weights)
            keys = (
                keys
                * (
                    cp.float32(1.0)
                    / cp.sqrt(
                        cp.mean(keys * keys, axis=2, keepdims=True) + eps
                    )
                )
                * (cp.float32(1.0) + key_norm_weights)
            )
            values = projected_values.reshape(tokens, kv_heads, head_dim)

            queries = self._apply_rope_sequence(
                queries, layer.rotary_dim, layer.rope_theta, position
            )
            keys = self._apply_rope_sequence(
                keys, layer.rotary_dim, layer.rope_theta, position
            )
            self._append_attention_cache_sequence(cache, keys, values)

            groups = heads // kv_heads
            grouped_queries = queries.reshape(
                tokens, kv_heads, groups, head_dim
            )
            if _kv_cache_type(cache) == "q8":
                cached_keys = _dequantize_kv(
                    cache.cuda_keys[:, : cache.tokens, :],
                    cache.cuda_key_scales[:, : cache.tokens],
                    cp,
                )
                cached_values = _dequantize_kv(
                    cache.cuda_values[:, : cache.tokens, :],
                    cache.cuda_value_scales[:, : cache.tokens],
                    cp,
                )
            else:
                cached_keys = cache.cuda_keys[:, : cache.tokens, :]
                cached_values = cache.cuda_values[:, : cache.tokens, :]
            scores = cp.einsum(
                "tkgd,ksd->tkgs", grouped_queries, cached_keys
            ) * cp.float32(head_dim**-0.5)
            span = int(cache.tokens)
            query_positions = cp.arange(position, position + tokens)[:, None]
            key_positions = cp.arange(span)[None, :]
            allowed = key_positions <= query_positions
            scores = cp.where(
                allowed[:, None, None, :], scores, cp.float32(-cp.inf)
            )
            scores -= cp.max(scores, axis=3, keepdims=True)
            attention_weights = cp.exp(scores)
            attention_weights /= cp.sum(
                attention_weights, axis=3, keepdims=True
            )
            context = cp.einsum(
                "tkgs,ksd->tkgd", attention_weights, cached_values
            ).reshape(tokens, heads, head_dim)
            gated = context * (
                cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-gates))
            )
            output = self.matrix_matmul_device(
                layer.o_projection, gated.reshape(tokens, heads * head_dim)
            )
            return output + hidden

    def _apply_rope_sequence(
        self, vectors: Any, rotary_dim: int, rope_theta: float, position: int
    ) -> Any:
        cp = self.cp
        tokens = int(vectors.shape[0])
        half = rotary_dim // 2
        positions = cp.arange(position, position + tokens, dtype=cp.float32)
        indices = cp.arange(half, dtype=cp.float32)
        frequencies = positions[:, None] / cp.power(
            cp.float32(rope_theta),
            cp.float32(2.0) * indices / cp.float32(rotary_dim),
        )
        cosines = cp.cos(frequencies)[:, None, :]
        sines = cp.sin(frequencies)[:, None, :]
        output = cp.empty_like(vectors)
        first = vectors[:, :, :half]
        second = vectors[:, :, half:rotary_dim]
        output[:, :, :half] = first * cosines - second * sines
        output[:, :, half:rotary_dim] = second * cosines + first * sines
        if rotary_dim < vectors.shape[2]:
            output[:, :, rotary_dim:] = vectors[:, :, rotary_dim:]
        return output

    def _append_attention_cache_sequence(
        self, cache: Any, keys: Any, values: Any
    ) -> None:
        cp = self.cp
        cache_type = _kv_cache_type(cache)
        tokens = int(keys.shape[0])
        required = cache.tokens + tokens
        if cache.cuda_keys is None:
            cache.cuda_capacity = max(256, required)
            dtype = cp.int8 if cache_type == "q8" else cp.float32
            cache.cuda_keys = cp.empty(
                (cache.num_key_value_heads, cache.cuda_capacity, cache.head_dim),
                dtype=dtype,
            )
            cache.cuda_values = cp.empty_like(cache.cuda_keys)
            if cache_type == "q8":
                cache.cuda_key_scales = cp.empty(
                    (cache.num_key_value_heads, cache.cuda_capacity),
                    dtype=cp.float32,
                )
                cache.cuda_value_scales = cp.empty_like(cache.cuda_key_scales)
        elif required > cache.cuda_capacity:
            capacity = cache.cuda_capacity
            while capacity < required:
                capacity *= 2
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
            if cache_type == "q8":
                grown_key_scales = cp.empty(
                    (cache.num_key_value_heads, capacity), dtype=cp.float32
                )
                grown_value_scales = cp.empty_like(grown_key_scales)
                grown_key_scales[:, : cache.tokens] = cache.cuda_key_scales[
                    :, : cache.tokens
                ]
                grown_value_scales[:, : cache.tokens] = cache.cuda_value_scales[
                    :, : cache.tokens
                ]
                cache.cuda_key_scales = grown_key_scales
                cache.cuda_value_scales = grown_value_scales
            cache.cuda_keys = grown_keys
            cache.cuda_values = grown_values
            cache.cuda_capacity = capacity
        base = cache.tokens
        keys = cp.transpose(keys, (1, 0, 2))
        values = cp.transpose(values, (1, 0, 2))
        if cache_type == "q8":
            quantized_keys, key_scales = _quantize_kv(keys, cp)
            quantized_values, value_scales = _quantize_kv(values, cp)
            cache.cuda_keys[:, base : base + tokens, :] = quantized_keys
            cache.cuda_values[:, base : base + tokens, :] = quantized_values
            cache.cuda_key_scales[:, base : base + tokens] = key_scales
            cache.cuda_value_scales[:, base : base + tokens] = value_scales
        else:
            cache.cuda_keys[:, base : base + tokens, :] = keys
            cache.cuda_values[:, base : base + tokens, :] = values
        cache.tokens += tokens

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
        cache_type = _kv_cache_type(cache)
        if cache.cuda_keys is None:
            cache.cuda_capacity = max(256, cache.tokens + 1)
            dtype = cp.int8 if cache_type == "q8" else cp.float32
            cache.cuda_keys = cp.empty(
                (
                    cache.num_key_value_heads,
                    cache.cuda_capacity,
                    cache.head_dim,
                ),
                dtype=dtype,
            )
            cache.cuda_values = cp.empty_like(cache.cuda_keys)
            if cache_type == "q8":
                cache.cuda_key_scales = cp.empty(
                    (cache.num_key_value_heads, cache.cuda_capacity),
                    dtype=cp.float32,
                )
                cache.cuda_value_scales = cp.empty_like(cache.cuda_key_scales)
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
            if cache_type == "q8":
                grown_key_scales = cp.empty(
                    (cache.num_key_value_heads, capacity), dtype=cp.float32
                )
                grown_value_scales = cp.empty_like(grown_key_scales)
                grown_key_scales[:, : cache.tokens] = cache.cuda_key_scales[
                    :, : cache.tokens
                ]
                grown_value_scales[:, : cache.tokens] = cache.cuda_value_scales[
                    :, : cache.tokens
                ]
                cache.cuda_key_scales = grown_key_scales
                cache.cuda_value_scales = grown_value_scales
            cache.cuda_keys = grown_keys
            cache.cuda_values = grown_values
            cache.cuda_capacity = capacity
        if cache_type == "q8":
            quantized_keys, key_scales = _quantize_kv(keys, cp)
            quantized_values, value_scales = _quantize_kv(values, cp)
            cache.cuda_keys[:, cache.tokens, :] = quantized_keys
            cache.cuda_values[:, cache.tokens, :] = quantized_values
            cache.cuda_key_scales[:, cache.tokens] = key_scales
            cache.cuda_value_scales[:, cache.tokens] = value_scales
        else:
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
            if os.environ.get("COLIBRI_FUSED_DELTA_DECODE", "1") != "0":
                return self._gated_delta_decode_fused(
                    layer, hidden, state, return_device=return_device
                )
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

    def _gated_delta_decode_fused(
        self, layer: Any, hidden: Any, state: Any, *, return_device: bool
    ) -> Any:
        """Single-token DeltaNet step in ~6 launches.

        Reuses the fused prefill conv/recurrence kernels with tokens=1 and a
        row-concatenated input projection, replacing the ~25 small elementwise
        launches of the portable path whose Python dispatch dominates decode
        once the MoE experts run off-GPU.
        """
        cp = self.cp
        hidden_device = cp.asarray(hidden, dtype=cp.float32)
        normalized = self.rms_norm_device(
            hidden_device, layer._input_norm_weights, layer.rms_norm_eps
        )
        qz, ba = self._combined_in_proj(layer)
        if qz is not None:
            projected = self._matrix_matvec_device(qz, normalized)
            gates = self._matrix_matvec_device(ba, normalized)
            mixed_qkv = projected[: layer.conv_dim]
            z = projected[layer.conv_dim :]
            beta_logits = gates[: layer.num_value_heads]
            decay_logits = gates[layer.num_value_heads :]
        else:
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
        conv_weights = self._float32_array(layer._conv_weights).reshape(
            layer.conv_dim, layer.conv_kernel_size
        )
        convolved = cp.empty((1, layer.conv_dim), dtype=cp.float32)
        conv_blocks = (
            layer.conv_dim + THREADS_PER_BLOCK - 1
        ) // THREADS_PER_BLOCK
        self._delta_conv_step_kernel(
            (conv_blocks,),
            (THREADS_PER_BLOCK,),
            (
                mixed_qkv,
                conv_weights,
                state.cuda_conv_state,
                convolved,
                layer.conv_dim,
                layer.conv_kernel_size,
            ),
        )
        cores = cp.empty((1, layer.value_dim), dtype=cp.float32)
        self._delta_recurrent_sequence_kernel(
            (layer.num_value_heads,),
            (THREADS_PER_BLOCK,),
            (
                convolved,
                z.reshape(1, -1),
                beta_logits.reshape(1, -1),
                decay_logits.reshape(1, -1),
                self._float32_array(layer._a_log),
                self._float32_array(layer._dt_bias),
                self._float32_array(layer._norm_weights),
                state.cuda_recurrent_state,
                cores,
                1,
                layer.num_key_heads,
                layer.num_value_heads,
                layer.key_head_dim,
                layer.value_head_dim,
                cp.float32(layer.rms_norm_eps),
            ),
        )
        output = self._matrix_matvec_device(
            layer.out_proj, cores.reshape(-1)
        )
        if return_device:
            return output
        return output.get().tolist()

    def _combined_in_proj(self, layer: Any) -> tuple[Any, Any]:
        """Row-concatenate the DeltaNet input projections once per layer.

        Returns ``(qz, ba)``: the fused qkv+z projection and the fused
        beta+decay projection, or ``(None, None)`` when the tensor types
        cannot be concatenated. The pairs are combined separately because
        checkpoints quantize the large qkv/z projections while keeping the
        tiny b/a projections at BF16.
        """
        cached = getattr(layer, "_combined_in_proj_cache", None)
        if cached is not None:
            return cached

        def concatenate(first: Any, second: Any) -> Any:
            from .bf16 import BF16Tensor
            from .q4 import Q4BlockTensor

            columns = first.shape[1]
            if second.shape[1] != columns:
                return None
            rows = first.shape[0] + second.shape[0]
            if isinstance(first, BF16Tensor) and isinstance(
                second, BF16Tensor
            ):
                return BF16Tensor(
                    shape=(rows, columns), data=first.data + second.data
                )
            if (
                columns % 32 == 0
                and isinstance(first, Q4BlockTensor)
                and isinstance(second, Q4BlockTensor)
            ):
                return Q4BlockTensor(
                    shape=(rows, columns),
                    packed=first.packed + second.packed,
                    scales=first.scales + second.scales,
                )
            return None

        qz = concatenate(layer.in_proj_qkv, layer.in_proj_z)
        ba = concatenate(layer.in_proj_b, layer.in_proj_a)
        cached = (qz, ba) if qz is not None and ba is not None else (None, None)
        layer._combined_in_proj_cache = cached
        return cached

    def _combined_qkv(self, layer: Any) -> Any:
        """Row-concatenate the attention Q/K/V projections once per layer."""
        if hasattr(layer, "_combined_qkv_cache"):
            return layer._combined_qkv_cache
        from .bf16 import BF16Tensor
        from .q4 import Q4BlockTensor

        tensors = (layer.q_projection, layer.k_projection, layer.v_projection)
        columns = tensors[0].shape[1]
        rows = sum(tensor.shape[0] for tensor in tensors)
        aligned = columns % 32 == 0 and all(
            tensor.shape[1] == columns for tensor in tensors
        )
        if all(isinstance(tensor, BF16Tensor) for tensor in tensors):
            combined = BF16Tensor(
                shape=(rows, columns),
                data=b"".join(tensor.data for tensor in tensors),
            )
        elif aligned and all(
            isinstance(tensor, Q4BlockTensor) for tensor in tensors
        ):
            combined = Q4BlockTensor(
                shape=(rows, columns),
                packed=b"".join(tensor.packed for tensor in tensors),
                scales=b"".join(tensor.scales for tensor in tensors),
            )
        else:
            combined = None
        layer._combined_qkv_offsets = (
            tensors[0].shape[0],
            tensors[0].shape[0] + tensors[1].shape[0],
        )
        layer._combined_qkv_cache = combined
        return combined

    def gated_delta_sequence(
        self, layer: Any, hidden: Any, state: Any
    ) -> Any:
        cp = self.cp
        tokens = int(hidden.shape[0])
        normalized = self.rms_norm_rows_device(
            hidden, layer._input_norm_weights, layer.rms_norm_eps
        )
        qz, ba = self._combined_in_proj(layer)
        if qz is not None:
            projected = self.matrix_matmul_device(qz, normalized)
            gates = self.matrix_matmul_device(ba, normalized)
            mixed_qkv = cp.ascontiguousarray(projected[:, : layer.conv_dim])
            z = cp.ascontiguousarray(projected[:, layer.conv_dim :])
            beta_logits = cp.ascontiguousarray(
                gates[:, : layer.num_value_heads]
            )
            decay_logits = cp.ascontiguousarray(
                gates[:, layer.num_value_heads :]
            )
        else:
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

    def _matrix_matvec_device(
        self, tensor: Any, vector: Any, *, protected: bool = True
    ) -> Any:
        from .q4 import Q4BlockTensor

        rows, columns = tensor.shape
        cp = self.cp
        output = cp.empty(rows, dtype=cp.float32)
        if isinstance(tensor, Q4BlockTensor):
            packed, scales = self._q4_arrays(tensor, protected=protected)
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
            protected=protected,
        )
        self._bf16_kernel(
            (rows,),
            (THREADS_PER_BLOCK,),
            (weights, vector, output, rows, columns),
        )
        return output

    def _float32_array(self, values: list[float]) -> Any:
        cp = self.cp
        elements = int(getattr(values, "size", len(values)))
        (array,) = self._cached_arrays(
            "float32",
            values,
            elements * 4,
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

    def q8_matvec_transposed(
        self,
        raw: bytes,
        input_size: int,
        output_size: int,
        vector: Any,
        *,
        return_device: bool = False,
        cache_weight: bool = False,
        protect_weight: bool = False,
    ) -> Any:
        """Multiply a GGML Q8_0 [input, output] tensor by a vector.

        GGUF's Qwen matrices use the first dimension as the input width. The
        quantized bytes therefore remain laid out as ``matrix[input, output]``
        even though this operation emits one value per output column.
        """
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Q8 matvec dimensions must be positive")
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            if cache_weight:
                (packed,) = self._cached_arrays(
                    "v2_q8",
                    raw,
                    len(raw),
                    lambda: (cp.asarray(memoryview(raw), dtype=cp.uint8),),
                    protected=protect_weight,
                )
            else:
                packed = cp.asarray(memoryview(raw), dtype=cp.uint8)
            input_device = cp.asarray(vector, dtype=cp.float32).reshape(-1)
            if int(input_device.size) != input_size:
                raise ValueError("Q8 matvec input width does not match tensor")
            output = cp.empty(output_size, dtype=cp.float32)
            blocks = (input_size * output_size + 31) // 32
            if int(packed.size) < blocks * 34:
                raise ValueError("Q8 tensor byte length is too small")
            self._q8_transposed_kernel(
                ((output_size + 7) // 8,),
                (THREADS_PER_BLOCK,),
                (packed, input_device, output, input_size, output_size),
            )
            return output if return_device else output.get().tolist()

    def route_topk_device(self, logits: Any, top_k: int) -> tuple[Any, Any]:
        """Select and renormalize routed experts in one CUDA launch."""
        cp = self.cp
        values = cp.asarray(logits, dtype=cp.float32).reshape(-1)
        experts = int(values.size)
        if top_k <= 0 or top_k > experts:
            raise ValueError("top-k must be between one and the expert count")
        selected = cp.empty(top_k, dtype=cp.int32)
        routing_weights = cp.empty(top_k, dtype=cp.float32)
        self._route_topk_kernel(
            (1,),
            (THREADS_PER_BLOCK,),
            (values, selected, routing_weights, experts, top_k),
            shared_mem=experts * 4,
        )
        return selected, routing_weights

    def q4k_matvec_transposed(
        self, raw: bytes, input_size: int, output_size: int, vector: Any, *,
        return_device: bool = False, cache_weight: bool = False,
    ) -> Any:
        return self._k_matvec_transposed(
            raw, input_size, output_size, vector, self._q4k_transposed_kernel,
            144, return_device=return_device, cache_weight=cache_weight,
        )

    def q5k_matvec_transposed(
        self, raw: bytes, input_size: int, output_size: int, vector: Any, *,
        return_device: bool = False, cache_weight: bool = False,
    ) -> Any:
        return self._k_matvec_transposed(
            raw, input_size, output_size, vector, self._q5k_transposed_kernel,
            176, return_device=return_device, cache_weight=cache_weight,
        )

    def q6k_matvec_transposed(
        self, raw: bytes, input_size: int, output_size: int, vector: Any, *,
        return_device: bool = False, cache_weight: bool = False,
    ) -> Any:
        return self._k_matvec_transposed(
            raw, input_size, output_size, vector, self._q6k_transposed_kernel,
            210, return_device=return_device, cache_weight=cache_weight,
        )

    def nvfp4_matvec_transposed(
        self, raw: bytes, input_size: int, output_size: int, vector: Any, *,
        return_device: bool = False, cache_weight: bool = False,
    ) -> Any:
        return self._k_matvec_transposed(
            raw, input_size, output_size, vector, self._nvfp4_matvec_transposed_kernel,
            18, return_device=return_device, cache_weight=cache_weight,
        )

    def _k_matvec_transposed(
        self, raw: bytes, input_size: int, output_size: int, vector: Any,
        kernel: Any, bytes_per_block: int, *, return_device: bool,
        cache_weight: bool,
    ) -> Any:
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            if cache_weight:
                (packed,) = self._cached_arrays(
                    "v2_k",
                    raw,
                    len(raw),
                    lambda: (cp.asarray(memoryview(raw), dtype=cp.uint8),),
                )
            else:
                packed = cp.asarray(memoryview(raw), dtype=cp.uint8)
            input_device = cp.asarray(vector, dtype=cp.float32).reshape(-1)
            if int(input_device.size) != input_size:
                raise ValueError("K-block matvec input width does not match tensor")
            output = cp.empty(output_size, dtype=cp.float32)
            blocks = (input_size * output_size + 255) // 256
            if int(packed.size) < blocks * bytes_per_block:
                raise ValueError("K-block tensor byte length is too small")
            kernel(
                (output_size,), (THREADS_PER_BLOCK,),
                (packed, input_device, output, input_size, output_size),
            )
            return output if return_device else output.get().tolist()

    def q5k_q6k_swiglu_accumulate(
        self,
        gate_raw: bytes,
        up_raw: bytes,
        down_raw: bytes,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        vector: Any,
        output: Any,
        weights: Any,
        weight_index: int,
    ) -> None:
        """Execute one Q5_K/Q5_K/Q6_K expert in two CUDA launches."""
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            packed = []
            for raw in (gate_raw, up_raw, down_raw):
                (array,) = self._cached_arrays(
                    "v2_k", raw, len(raw),
                    lambda raw=raw: (
                        cp.asarray(memoryview(raw), dtype=cp.uint8),
                    ),
                )
                packed.append(array)
            input_device = cp.asarray(vector, dtype=cp.float32).reshape(-1)
            activated = cp.empty(intermediate_size, dtype=cp.float32)
            self._q5k_swiglu_transposed_kernel(
                (intermediate_size,),
                (THREADS_PER_BLOCK,),
                (
                    packed[0], packed[1], input_device, activated,
                    input_size, intermediate_size,
                ),
            )
            self._q6k_accumulate_transposed_kernel(
                (output_size,),
                (THREADS_PER_BLOCK,),
                (
                    packed[2], activated, output, weights, weight_index,
                    intermediate_size, output_size,
                ),
            )

    def prefetch_v2_k_weights(self, raws: tuple[bytes, ...]) -> Any:
        """Upload one expert's K-quantized weights on the transfer stream."""
        cp = self.cp
        event = cp.cuda.Event()
        with cp.cuda.Device(self.device_id), self._prefetch_stream:
            for raw in raws:
                key = ("v2_k", id(raw))
                entry = self._cache.get(key)
                if entry is not None and entry.owner is raw:
                    self.expert_prefetch_hits += 1
                self.expert_prefetch_requests += 1
                self.expert_prefetch_bytes += len(raw)
                self._cached_arrays(
                    "v2_k", raw, len(raw),
                    lambda raw=raw: (
                        cp.asarray(memoryview(raw), dtype=cp.uint8),
                    ),
                )
            event.record(self._prefetch_stream)
        return event

    def _cached_v2_expert_group(
        self, group: tuple[bytes, bytes, bytes]
    ) -> tuple[Any, Any, Any]:
        """Cache one expert as a single staged gate/up/down allocation."""
        import numpy as np

        cp = self.cp
        sizes = tuple(len(raw) for raw in group)
        total = sum(sizes)

        def upload() -> tuple[Any, Any, Any]:
            pinned = cp.cuda.alloc_pinned_memory(total)
            host = np.frombuffer(pinned, dtype=np.uint8, count=total)
            offset = 0
            for raw, size in zip(group, sizes):
                host[offset:offset + size] = np.frombuffer(
                    raw, dtype=np.uint8, count=size
                )
                offset += size
            device = cp.empty(total, dtype=cp.uint8)
            device.set(host, stream=cp.cuda.get_current_stream())
            self._retain_until_stream_complete((pinned,))
            gate_end = sizes[0]
            up_end = gate_end + sizes[1]
            return (
                device[:gate_end],
                device[gate_end:up_end],
                device[up_end:],
            )

        arrays = self._cached_arrays(
            "v2_expert", group[0], total, upload
        )
        return arrays[0], arrays[1], arrays[2]

    def prefetch_v2_expert_groups(
        self, groups: list[tuple[bytes, bytes, bytes]]
    ) -> Any | None:
        """Upload route-hinted expert bundles on the transfer stream."""
        if not self.expert_prefetch_enabled or not groups:
            return None
        cp = self.cp
        event = cp.cuda.Event()
        prefetched_keys = []
        with cp.cuda.Device(self.device_id), self._prefetch_stream:
            for group in groups:
                key = ("v2_expert", id(group[0]))
                prefetched_keys.append(key)
                entry = self._cache.get(key)
                self.expert_prefetch_requests += 1
                self.expert_prefetch_bytes += sum(len(raw) for raw in group)
                if entry is not None and entry.owner is group[0]:
                    self.expert_prefetch_hits += 1
                self._cached_v2_expert_group(group)
            event.record(self._prefetch_stream)
        for key in prefetched_keys:
            self._pending_prefetches[key] = _PendingPrefetch(event, ())
        return event

    def q5k_q6k_grouped_swiglu_accumulate(
        self,
        groups: list[tuple[bytes, bytes, bytes]],
        input_size: int,
        intermediate_size: int,
        output_size: int,
        vector: Any,
        output: Any,
        weights: Any,
        *,
        down_ggml_type: int = 14,
        gate_ggml_type: int = 13,
    ) -> None:
        """Execute selected Q5/Q6/NVFP4 gate/up experts with grouped Q6/Q8/NVFP4 down."""
        if not groups:
            return
        if gate_ggml_type == 40:
            gate_up_kernel = self._nvfp4_grouped_swiglu_kernel
        elif gate_ggml_type == 14:
            gate_up_kernel = self._q5k_grouped_swiglu_kernel
        else:
            gate_up_kernel = self._q5k_grouped_swiglu_kernel
        if down_ggml_type == 8:
            down_kernel = self._q8_grouped_accumulate_kernel
        elif down_ggml_type == 40:
            down_kernel = self._nvfp4_grouped_accumulate_kernel
        else:
            down_kernel = self._q6k_grouped_accumulate_kernel
        self.batched_moe_tokens += 1
        self.batched_expert_groups += len(groups)
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            gate_arrays, up_arrays, down_arrays = [], [], []
            cache_started = time.perf_counter()
            for group in groups:
                uploaded = self._cached_v2_expert_group(group)
                gate_arrays.append(uploaded[0])
                up_arrays.append(uploaded[1])
                down_arrays.append(uploaded[2])
            self.profile_host(
                "routed_weight_cache", time.perf_counter() - cache_started
            )
            pointers_started = time.perf_counter()
            gate_ptrs = cp.asarray(
                [array.data.ptr for array in gate_arrays], dtype=cp.uint64
            )
            up_ptrs = cp.asarray(
                [array.data.ptr for array in up_arrays], dtype=cp.uint64
            )
            down_ptrs = cp.asarray(
                [array.data.ptr for array in down_arrays], dtype=cp.uint64
            )
            activated = cp.empty(
                (len(groups), intermediate_size), dtype=cp.float32
            )
            self.profile_host(
                "routed_pointer_setup", time.perf_counter() - pointers_started
            )
            gate_up_profile = self.profile_start()
            gate_up_kernel(
                (intermediate_size, len(groups)),
                (THREADS_PER_BLOCK,),
                (
                    gate_ptrs, up_ptrs, cp.asarray(vector, dtype=cp.float32),
                    activated, input_size, intermediate_size, len(groups),
                ),
            )
            self.profile_end("routed_gate_up", gate_up_profile)
            down_profile = self.profile_start()
            down_kernel(
                (output_size,),
                (THREADS_PER_BLOCK,),
                (
                    down_ptrs, activated, output, weights,
                    intermediate_size, output_size, len(groups),
                ),
            )
            self.profile_end("routed_down", down_profile)

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
        resident_table = None
        if os.environ.get("COLIBRI_MOE_LAYER_RESIDENT", "0") == "1":
            resident_table = self._resident_moe_table_for(layer)
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
        if resident_table is not None:
            shared_logit = self._matrix_matvec_device(
                layer.shared_gate, normalized
            )[0]
            shared_weight = cp.float32(1.0) / (
                cp.float32(1.0) + cp.exp(-shared_logit)
            )
            if route_state is not None:
                route_state.last_selected_experts = ()
            output = self._resident_moe_device(
                resident_table,
                selected,
                routing_weights,
                shared_weight,
                normalized,
            )
            return hidden + output
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

        The whole MoE block runs on the host in NumPy plus the fused native
        expert kernel, so the hidden vector crosses the PCIe boundary exactly
        once in each direction and no per-layer GPU round trips are issued for
        the router. The router matvec deliberately uses ``einsum`` (not ``@``)
        because BLAS thread fan-out on a tiny per-token matvec costs far more
        than the arithmetic.
        """
        from .native import active_native

        backend = active_native()
        if backend is None:
            # CUDA remains active for the rest of the decoder, so explicitly
            # suppress accelerator dispatch in this CPU fallback. Otherwise
            # Q4BlockTensor/Q4SwiGLUExpert would send the supposedly CPU
            # offloaded expert work back through the global CUDA accelerator.
            result = layer.forward_residual(
                hidden.get().tolist(), allow_cuda=False
            )
            if route_state is not None:
                route_state.last_selected_experts = result.selected_experts
            return self.device_vector(result.output)
        import numpy as np

        router_np, gate_np, norm_np = self._host_moe_weights(layer)
        host, output_host, output_device = self._moe_transfer_buffers(
            int(hidden.size)
        )
        hidden.get(out=host)
        inverse_rms = 1.0 / np.sqrt(
            float(np.mean(host * host)) + layer.rms_norm_eps
        )
        normalized = (host * (inverse_rms * norm_np)).astype(np.float32)
        logits = np.einsum("eh,h->e", router_np, normalized)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        selected = np.argpartition(probabilities, -layer.top_k)[-layer.top_k :]
        selected = selected[np.argsort(probabilities[selected])[::-1]]
        weights = probabilities[selected]
        weights /= weights.sum()
        selected_ids = [int(expert_id) for expert_id in selected]
        if route_state is not None:
            route_state.last_selected_experts = tuple(selected_ids)
        shared_logit = float(gate_np @ normalized)
        shared_weight = 1.0 / (1.0 + np.exp(-shared_logit))
        backend.q4_moe(
            [layer._expert(expert_id) for expert_id in selected_ids],
            weights.tolist(),
            layer.shared_expert,
            shared_weight,
            normalized,
            as_array=True,
            out=output_host,
        )
        output_host += host
        output_device.set(output_host)
        return output_device

    def _moe_transfer_buffers(self, hidden_size: int) -> tuple[Any, Any, Any]:
        """Pinned host staging plus a reused device output buffer.

        Reusing one device buffer across offloaded layers is safe because the
        blocking ``hidden.get`` at the top of the next offloaded layer
        synchronizes the stream after every consumer of the previous output
        has been enqueued.
        """
        buffers = self._moe_transfer_cache.get(hidden_size)
        if buffers is None:
            import numpy as np

            cp = self.cp
            input_memory = cp.cuda.alloc_pinned_memory(hidden_size * 4)
            output_memory = cp.cuda.alloc_pinned_memory(hidden_size * 4)
            buffers = (
                np.frombuffer(
                    input_memory, dtype=np.float32, count=hidden_size
                ),
                np.frombuffer(
                    output_memory, dtype=np.float32, count=hidden_size
                ),
                cp.empty(hidden_size, dtype=cp.float32),
            )
            self._moe_transfer_cache[hidden_size] = buffers
        return buffers

    def _moe_sequence_cpu(
        self, layer: Any, hidden: Any, route_state: Any | None
    ) -> Any:
        """Host-CPU offload of a prefill sequence's routed experts."""
        from .native import active_native

        backend = active_native()
        if backend is None:
            rows = hidden.get().tolist()
            outputs: list[list[float]] = []
            routes: list[tuple[int, ...]] = []
            for row in rows:
                result = layer.forward_residual(row)
                outputs.append(result.output)
                routes.append(result.selected_experts)
            if route_state is not None and routes:
                route_state.sequence_selected_experts += tuple(routes)
                route_state.last_selected_experts = routes[-1]
            return self.device_vector(outputs)
        import numpy as np

        router_np, gate_np, norm_np = self._host_moe_weights(layer)
        host = hidden.get()
        inverse_rms = 1.0 / np.sqrt(
            np.mean(host * host, axis=1, keepdims=True) + layer.rms_norm_eps
        )
        normalized = (host * inverse_rms * norm_np).astype(np.float32)
        if normalized.shape[0] < 16:
            # Tiny speculative-verify batches: BLAS thread fan-out on a small
            # matmul costs far more than the arithmetic, so use einsum like
            # the single-token path does.
            logits = np.einsum("th,eh->te", normalized, router_np)
            shared_logits = np.einsum("th,h->t", normalized, gate_np)
        else:
            logits = normalized @ router_np.T
            shared_logits = normalized @ gate_np
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        shared_weights = (1.0 / (1.0 + np.exp(-shared_logits))).astype(
            np.float32
        )
        tokens = int(host.shape[0])
        top_k = layer.top_k
        selected = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
        selected_probabilities = np.take_along_axis(
            probabilities, selected, axis=1
        )
        order = np.argsort(-selected_probabilities, axis=1)
        selected = np.take_along_axis(selected, order, axis=1)
        weights = np.take_along_axis(selected_probabilities, order, axis=1)
        weights = (weights / weights.sum(axis=1, keepdims=True)).astype(
            np.float32
        )
        if route_state is not None:
            routes = tuple(
                tuple(int(expert_id) for expert_id in row) for row in selected
            )
            route_state.sequence_selected_experts += routes
            route_state.last_selected_experts = routes[-1]
        if getattr(backend, "_grouped_moe", False):
            # Expert-major: sort assignments by expert so each unique expert's
            # weights are streamed from RAM once per call, then append the
            # shared expert (which every token uses) as the final group.
            flat_experts = selected.ravel()
            flat_order = np.argsort(flat_experts, kind="stable")
            unique_ids = np.unique(flat_experts)
            assignment_expert = np.searchsorted(
                unique_ids, flat_experts[flat_order]
            ).astype(np.int32)
            assignment_token = (flat_order // top_k).astype(np.int32)
            assignment_weight = weights.ravel()[flat_order]
            shared_index = len(unique_ids)
            assignment_expert = np.concatenate(
                (
                    assignment_expert,
                    np.full(tokens, shared_index, dtype=np.int32),
                )
            )
            assignment_token = np.concatenate(
                (assignment_token, np.arange(tokens, dtype=np.int32))
            )
            assignment_weight = np.concatenate(
                (assignment_weight, shared_weights)
            )
            experts = [
                layer._expert(int(expert_id)) for expert_id in unique_ids
            ]
            experts.append(layer.shared_expert)
            outputs = backend.q4_moe_grouped(
                experts,
                assignment_expert,
                assignment_token,
                assignment_weight,
                normalized,
            )
            outputs += host
        else:
            outputs = np.empty_like(host)
            for token in range(tokens):
                selected_ids = [int(expert_id) for expert_id in selected[token]]
                output = backend.q4_moe(
                    [layer._expert(expert_id) for expert_id in selected_ids],
                    weights[token].tolist(),
                    layer.shared_expert,
                    float(shared_weights[token]),
                    normalized[token],
                    as_array=True,
                )
                outputs[token] = output + host[token]
        return self.device_vector(outputs)

    def delta_segment_ready(self) -> bool:
        """Whether the native C decode driver can run DeltaNet+MoE segments."""
        if self._delta_segment_state == "off":
            return False
        if self._delta_segment_state == "ready":
            return True
        if os.environ.get("COLIBRI_C_DECODE", "1") == "0":
            self._delta_segment_state = "off"
            return False
        from .native import active_native

        backend = active_native()
        if backend is None or not getattr(backend, "_gpu_driver", False):
            self._delta_segment_state = "off"
            return False
        include_dirs = []
        cuda_path = self.cp.cuda.get_cuda_path()
        if cuda_path:
            include_dirs.append(str(Path(cuda_path) / "include"))
        try:
            ready = backend.gpu_prepare(
                _KERNEL_SOURCE, self.device_id, include_dirs
            )
        except RuntimeError:
            ready = False
        self._delta_segment_state = "ready" if ready else "off"
        return ready

    def _native_q4_moe_backend(self) -> Any | None:
        if self._native_moe_state == "off":
            return None
        from .native import active_native

        backend = active_native()
        if self._native_moe_state == "ready":
            return backend
        if os.environ.get("COLIBRI_NATIVE_CUDA_MOE", "0") == "0":
            self._native_moe_state = "off"
            return None
        if backend is None or not getattr(backend, "_gpu_driver", False):
            self._native_moe_state = "off"
            return None
        include_dirs = []
        cuda_path = self.cp.cuda.get_cuda_path()
        if cuda_path:
            include_dirs.append(str(Path(cuda_path) / "include"))
        try:
            ready = backend.gpu_prepare(
                _KERNEL_SOURCE, self.device_id, include_dirs
            )
        except RuntimeError:
            ready = False
        self._native_moe_state = "ready" if ready else "off"
        return backend if ready else None

    def delta_moe_segment(
        self, layers: list[Any], layer_states: list[Any], hidden: Any
    ) -> Any | None:
        """Run consecutive CPU-offloaded DeltaNet layers through the C driver.

        Returns the new hidden device array, or None when any layer cannot be
        pointer-resolved (caller falls back to the per-layer Python path).
        """
        from .native import active_native

        backend = active_native()
        key = (id(layers[0]), id(layer_states[0]), len(layers))
        entry = self._delta_segment_tables.get(key)
        if entry is None or not all(
            entry["states"][index] is layer_states[index].token_mixer_state.cuda_conv_state
            for index in range(len(layers))
        ):
            stale = self._delta_segment_tables.pop(key, None)
            if stale is not None:
                for handle in stale.get("graphs", ()):
                    backend.delta_graph_destroy(handle)
            entry = self._build_delta_segment(layers, layer_states, backend)
            if entry is None:
                return None
            self._delta_segment_tables[key] = entry
        scratch_hidden = entry["params_refs"]["hidden"]
        scratch_hidden[...] = hidden
        backend.delta_moe_segment(
            entry["params"], entry["table"], len(layers)
        )
        for layer_state in layer_states:
            layer_state.token_mixer_state.tokens += 1
        return scratch_hidden

    def _build_delta_segment(
        self, layers: list[Any], layer_states: list[Any], backend: Any
    ) -> dict | None:
        from .bf16 import BF16Tensor
        from .native import DeltaLayerStruct, DeltaParamsStruct
        from .q4 import Q4BlockTensor

        cp = self.cp
        first = layers[0].token_mixer
        qz_rows = first.conv_dim + first.value_dim
        ba_rows = 2 * first.num_value_heads
        params_refs = self._delta_segment_scratch(first, qz_rows, ba_rows)
        table = (DeltaLayerStruct * len(layers))()
        keep_alive: list[Any] = []
        states: list[Any] = []
        for index, (layer, layer_state) in enumerate(
            zip(layers, layer_states)
        ):
            mixer = layer.token_mixer
            moe = layer.moe
            mixer_state = layer_state.token_mixer_state
            if (
                mixer.conv_dim != first.conv_dim
                or mixer.value_dim != first.value_dim
                or mixer_state.cuda_conv_state is None
                or len(moe._experts) != moe.expert_count
            ):
                return None
            qz, ba = self._combined_in_proj(mixer)
            if (
                not isinstance(qz, Q4BlockTensor)
                or not isinstance(ba, BF16Tensor)
                or not isinstance(mixer.out_proj, Q4BlockTensor)
            ):
                return None
            in_packed, in_scales = self._q4_arrays(qz, protected=True)
            (ba_weights,) = self._cached_arrays(
                "bf16",
                ba,
                len(ba.data),
                lambda ba=ba: (
                    cp.asarray(memoryview(ba.data), dtype=cp.uint8).view(
                        cp.uint16
                    ),
                ),
                protected=True,
            )
            out_packed, out_scales = self._q4_arrays(
                mixer.out_proj, protected=True
            )
            input_norm = self._float32_array(mixer._input_norm_weights)
            conv_weights = self._float32_array(mixer._conv_weights)
            a_log = self._float32_array(mixer._a_log)
            dt_bias = self._float32_array(mixer._dt_bias)
            delta_norm = self._float32_array(mixer._norm_weights)
            post_norm = self._float32_array(
                moe._post_attention_norm_weights
            )
            router_gate_tensor = getattr(moe, "_router_gate_cache", None)
            if router_gate_tensor is None:
                router_gate_tensor = BF16Tensor(
                    shape=(
                        moe.router.shape[0] + 1,
                        moe.router.shape[1],
                    ),
                    data=moe.router.data + moe.shared_gate.data,
                )
                moe._router_gate_cache = router_gate_tensor
            (router_gate,) = self._cached_arrays(
                "bf16",
                router_gate_tensor,
                len(router_gate_tensor.data),
                lambda t=router_gate_tensor: (
                    cp.asarray(memoryview(t.data), dtype=cp.uint8).view(
                        cp.uint16
                    ),
                ),
                protected=True,
            )
            experts = [
                moe._expert(expert_id)
                for expert_id in range(moe.expert_count)
            ]
            pointer_arrays = backend._expert_pointer_arrays(experts)
            shared_pointers = backend._expert_pointer_arrays(
                [moe.shared_expert]
            )
            struct = table[index]
            struct.qz_packed = in_packed.data.ptr
            struct.qz_scales = in_scales.data.ptr
            struct.ba_weights = ba_weights.data.ptr
            struct.out_proj_packed = out_packed.data.ptr
            struct.out_proj_scales = out_scales.data.ptr
            struct.input_norm = input_norm.data.ptr
            struct.conv_weights = conv_weights.data.ptr
            struct.a_log = a_log.data.ptr
            struct.dt_bias = dt_bias.data.ptr
            struct.delta_norm = delta_norm.data.ptr
            struct.conv_state = mixer_state.cuda_conv_state.data.ptr
            struct.recurrent_state = mixer_state.cuda_recurrent_state.data.ptr
            struct.router_gate = router_gate.data.ptr
            struct.post_attention_norm = post_norm.data.ptr
            struct.expert_gate_packed = ctypes_address(pointer_arrays[0])
            struct.expert_gate_scales = ctypes_address(pointer_arrays[1])
            struct.expert_down_packed = ctypes_address(pointer_arrays[2])
            struct.expert_down_scales = ctypes_address(pointer_arrays[3])
            # The shared-expert fields are direct data pointers, not pointer
            # tables: dereference the single-entry arrays.
            struct.shared_gate_up_packed = ctypes_element(shared_pointers[0])
            struct.shared_gate_up_scales = ctypes_element(shared_pointers[1])
            struct.shared_down_packed = ctypes_element(shared_pointers[2])
            struct.shared_down_scales = ctypes_element(shared_pointers[3])
            keep_alive.append(
                (
                    in_packed, in_scales, ba_weights, out_packed, out_scales,
                    input_norm, conv_weights, a_log, dt_bias, delta_norm,
                    post_norm, router_gate, router_gate_tensor,
                    pointer_arrays, shared_pointers, experts,
                )
            )
            states.append(mixer_state.cuda_conv_state)
        params = DeltaParamsStruct(
            hidden_size=first.hidden_size,
            conv_dim=first.conv_dim,
            conv_kernel=first.conv_kernel_size,
            value_dim=first.value_dim,
            num_key_heads=first.num_key_heads,
            num_value_heads=first.num_value_heads,
            key_head_dim=first.key_head_dim,
            value_head_dim=first.value_head_dim,
            qz_rows=qz_rows,
            ba_rows=ba_rows,
            num_experts=layers[0].moe.expert_count,
            top_k=layers[0].moe.top_k,
            moe_intermediate=layers[0].moe.shared_expert.intermediate_size,
            rms_norm_eps=float(first.rms_norm_eps),
            hidden=params_refs["hidden"].data.ptr,
            normalized=params_refs["normalized"].data.ptr,
            projected=params_refs["projected"].data.ptr,
            gates=params_refs["gates"].data.ptr,
            convolved=params_refs["convolved"].data.ptr,
            cores=params_refs["cores"].data.ptr,
            mixed=params_refs["mixed"].data.ptr,
            moe_normalized=params_refs["moe_normalized"].data.ptr,
            router_logits=params_refs["router_logits"].data.ptr,
            hidden_host=params_refs["hidden_host"].ctypes.data,
            normalized_host=params_refs["normalized_host"].ctypes.data,
            moe_host=params_refs["moe_host"].ctypes.data,
            logits_host=params_refs["logits_host"].ctypes.data,
            bundle_floats=params_refs["bundle_floats"],
        )
        graphs: list[int] = []
        # Replaying a different graph exec per layer measured ~5x slower than
        # plain launches (exec upload thrash), so graphs stay opt-in.
        build_graphs = os.environ.get("COLIBRI_SEG_GRAPHS") == "1"
        for index in range(len(layers)):
            handle = (
                backend.delta_graph_build(params, table[index])
                if build_graphs
                else 0
            )
            table[index].graph = handle
            graphs.append(handle)
        return {
            "params": params,
            "params_refs": params_refs,
            "table": table,
            "keep_alive": keep_alive,
            "states": states,
            "graphs": graphs,
        }

    def _delta_segment_scratch(
        self, mixer: Any, qz_rows: int, ba_rows: int
    ) -> dict:
        if self._delta_segment_scratch_cache is not None:
            return self._delta_segment_scratch_cache
        import numpy as np

        cp = self.cp
        hidden_size = mixer.hidden_size

        def pinned(size: int) -> Any:
            memory = cp.cuda.alloc_pinned_memory(size * 4)
            return np.frombuffer(memory, dtype=np.float32, count=size)

        # mixed | moe_normalized | router_logits live in one allocation (as
        # do their pinned host mirrors) so the C driver can pull all three
        # back in a single transfer per layer.
        bundle_floats = hidden_size * 2 + 1024
        bundle_device = cp.empty(bundle_floats, dtype=cp.float32)
        bundle_host = pinned(bundle_floats)
        self._delta_segment_scratch_cache = {
            "hidden": cp.empty(hidden_size, dtype=cp.float32),
            "normalized": cp.empty(hidden_size, dtype=cp.float32),
            "projected": cp.empty(qz_rows, dtype=cp.float32),
            "gates": cp.empty(ba_rows, dtype=cp.float32),
            "convolved": cp.empty(mixer.conv_dim, dtype=cp.float32),
            "cores": cp.empty(mixer.value_dim, dtype=cp.float32),
            "mixed": bundle_device[:hidden_size],
            "moe_normalized": bundle_device[hidden_size : hidden_size * 2],
            "router_logits": bundle_device[hidden_size * 2 :],
            "bundle_device": bundle_device,
            "hidden_host": bundle_host[:hidden_size],
            "normalized_host": bundle_host[hidden_size : hidden_size * 2],
            "logits_host": bundle_host[hidden_size * 2 :],
            "bundle_host": bundle_host,
            "moe_host": pinned(hidden_size),
            "bundle_floats": bundle_floats,
        }
        return self._delta_segment_scratch_cache

    def _host_moe_weights(self, layer: Any) -> tuple[Any, Any, Any]:
        """Cache the layer's router/gate/norm weights as float32 host arrays."""
        cached = getattr(layer, "_host_moe_weights_cache", None)
        if cached is not None:
            return cached
        import numpy as np

        def _bf16_matrix(tensor: Any) -> Any:
            raw = np.frombuffer(tensor.data, dtype="<u2")
            return (
                (raw.astype(np.uint32) << 16)
                .view(np.float32)
                .reshape(tensor.shape)
            )

        cached = (
            np.ascontiguousarray(_bf16_matrix(layer.router)),
            np.ascontiguousarray(_bf16_matrix(layer.shared_gate).reshape(-1)),
            1.0
            + np.asarray(
                layer._post_attention_norm_weights, dtype=np.float32
            ),
        )
        layer._host_moe_weights_cache = cached
        return cached

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
            routes = tuple(
                tuple(int(expert_id) for expert_id in route)
                for route in selected_ids
            )
            route_state.sequence_selected_experts += routes
            route_state.last_selected_experts = routes[-1]
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
        device_arrays: list[Any] = []
        address_key = tuple(
            value
            for expert in experts
            for value in (id(expert.gate_up), id(expert.down))
        )
        for index, expert in enumerate(experts):
            protected = index + 1 == expert_count
            packed, scales = self._q4_arrays(
                expert.gate_up, protected=protected
            )
            device_arrays.extend((packed, scales))
            gate_packed.append(packed.data.ptr)
            gate_scales.append(scales.data.ptr)
            packed, scales = self._q4_arrays(
                expert.down, protected=protected
            )
            device_arrays.extend((packed, scales))
            down_packed.append(packed.data.ptr)
            down_scales.append(scales.data.ptr)
        with cp.cuda.Device(self.device_id):
            addresses = self._moe_address_cache.get(address_key)
            if addresses is None:
                self.moe_address_cache_misses += 1
                addresses = cp.asarray(
                    [gate_packed, gate_scales, down_packed, down_scales],
                    dtype=cp.uint64,
                )
                if self._moe_address_table_cacheable(address_key):
                    self._moe_address_cache[address_key] = addresses
            else:
                self.moe_address_cache_hits += 1
                self._moe_address_cache.move_to_end(address_key)
            input_vector = cp.asarray(vector, dtype=cp.float32)
            gate_output = cp.empty(expert_count * gate_rows, dtype=cp.float32)
            use_q8 = (
                self.q8_moe_enabled
                and hidden_size % 32 == 0
                and intermediate_size % 32 == 0
            )
            native_backend = (
                None if use_q8 else self._native_q4_moe_backend()
            )
            if native_backend is not None:
                activation_count = expert_count * intermediate_size
                activated = cp.empty(activation_count, dtype=cp.float32)
                output = cp.empty(output_size, dtype=cp.float32)
                device_weights = cp.asarray(weights, dtype=cp.float32)
                native_backend.gpu_q4_moe(
                    addresses[0].data.ptr,
                    addresses[1].data.ptr,
                    addresses[2].data.ptr,
                    addresses[3].data.ptr,
                    device_weights.data.ptr,
                    input_vector.data.ptr,
                    gate_output.data.ptr,
                    activated.data.ptr,
                    output.data.ptr,
                    cp.cuda.get_current_stream().ptr,
                    expert_count,
                    hidden_size,
                    intermediate_size,
                )
                if return_device:
                    self._retain_until_stream_complete(tuple(device_arrays))
                    return output
                return output.get().tolist()
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
                self._retain_until_stream_complete(tuple(device_arrays))
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
        self._reap_expert_loads()
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

    def _release_resident_moe_table(self) -> None:
        table = self._resident_moe_table
        if table is None:
            return
        for tensor_id in table["tensor_ids"]:
            entry = self._cache.get(("q4", tensor_id))
            if entry is not None:
                entry.protected = False
        self._resident_moe_table = None

    def _resident_moe_table_for(self, layer: Any) -> dict[str, Any] | None:
        if layer.expert_device != "cuda":
            return None
        current = self._resident_moe_table
        if current is not None and current["layer"] is layer:
            return current
        self._release_resident_moe_table()
        experts = [
            layer._expert(index) for index in range(layer.expert_count)
        ] + [layer.shared_expert]
        gate_packed: list[int] = []
        gate_scales: list[int] = []
        down_packed: list[int] = []
        down_scales: list[int] = []
        arrays: list[Any] = []
        tensor_ids: list[int] = []
        for expert in experts:
            gate, scales = self._q4_arrays(expert.gate_up, protected=True)
            down, down_scale = self._q4_arrays(expert.down, protected=True)
            arrays.extend((gate, scales, down, down_scale))
            tensor_ids.extend(
                (id(expert.gate_up), id(expert.down))
            )
            gate_packed.append(gate.data.ptr)
            gate_scales.append(scales.data.ptr)
            down_packed.append(down.data.ptr)
            down_scales.append(down_scale.data.ptr)
        if not all(
            ("q4", tensor_id) in self._cache for tensor_id in tensor_ids
        ):
            self._release_resident_moe_table()
            return None
        cp = self.cp
        table = {
            "layer": layer,
            "expert_count": layer.expert_count,
            "top_k": layer.top_k,
            "gate_rows": experts[0].gate_up.shape[0],
            "hidden_size": experts[0].hidden_size,
            "intermediate_size": experts[0].intermediate_size,
            "arrays": tuple(arrays),
            "tensor_ids": tuple(tensor_ids),
            "gate_addresses": cp.asarray(
                gate_packed, dtype=cp.uint64
            ),
            "gate_scale_addresses": cp.asarray(
                gate_scales, dtype=cp.uint64
            ),
            "down_addresses": cp.asarray(
                down_packed, dtype=cp.uint64
            ),
            "down_scale_addresses": cp.asarray(
                down_scales, dtype=cp.uint64
            ),
        }
        self._resident_moe_table = table
        return table

    def _resident_moe_device(
        self,
        table: dict[str, Any],
        selected: Any,
        routing_weights: Any,
        shared_weight: Any,
        vector: Any,
    ) -> Any:
        cp = self.cp
        selected_count = table["top_k"] + 1
        selected_all = cp.empty(selected_count, dtype=cp.int32)
        selected_all[:-1] = selected
        selected_all[-1] = table["expert_count"]
        weights = cp.empty(selected_count, dtype=cp.float32)
        weights[:-1] = routing_weights
        weights[-1] = shared_weight
        gate_output = cp.empty(
            selected_count * table["gate_rows"], dtype=cp.float32
        )
        activated = cp.empty(
            selected_count * table["intermediate_size"], dtype=cp.float32
        )
        output = cp.empty(table["hidden_size"], dtype=cp.float32)
        self._q4_selected_kernel(
            (table["gate_rows"], selected_count),
            (THREADS_PER_BLOCK,),
            (
                table["gate_addresses"],
                table["gate_scale_addresses"],
                selected_all,
                vector,
                gate_output,
                table["gate_rows"],
                table["hidden_size"],
                selected_count,
            ),
        )
        activation_count = activated.size
        self._silu_mul_batched_kernel(
            ((activation_count + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,),
            (THREADS_PER_BLOCK,),
            (
                gate_output,
                activated,
                table["intermediate_size"],
                activation_count,
            ),
        )
        self._q4_selected_weighted_kernel(
            (table["hidden_size"],),
            (THREADS_PER_BLOCK,),
            (
                table["down_addresses"],
                table["down_scale_addresses"],
                selected_all,
                activated,
                weights,
                output,
                table["hidden_size"],
                table["intermediate_size"],
                selected_count,
            ),
        )
        return output

    def _reap_expert_loads(self) -> None:
        completed = [
            key for key, future in self._pending_expert_loads.items()
            if future.done()
        ]
        for key in completed:
            future = self._pending_expert_loads.pop(key)
            try:
                layer, expert = future.result()
            except Exception:
                continue
            self.expert_load_completions += 1
            self.prefetch_q4(expert.gate_up)
            self.prefetch_q4(expert.down)

    def _schedule_expert_load(self, layer: Any, expert_id: int) -> None:
        key = (id(layer), int(expert_id))
        if key in self._pending_expert_loads:
            return

        def load() -> Any:
            expert = layer._expert(int(expert_id))
            return layer, expert

        self._pending_expert_loads[key] = self._expert_load_executor.submit(load)
        self.expert_load_requests += 1

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
        self._reap_expert_loads()
        for expert_id in selected_experts[: self.expert_prefetch_budget]:
            expert = layer._experts.get(int(expert_id))
            if expert is None:
                self._schedule_expert_load(layer, int(expert_id))
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
            entry.priority_until = max(
                entry.priority_until, self._cache_priority_epoch + 8
            )
            self.cache_priority_promotions += 1
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
        self._cache[key] = _CacheEntry(
            tensor, arrays, byte_size, False, self._cache_priority_epoch + 8
        )
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

    def _reap_inflight_buffers(self) -> None:
        self._inflight_buffers = [
            entry for entry in self._inflight_buffers if not entry.event.done
        ]

    def _retain_until_stream_complete(self, arrays: tuple[Any, ...]) -> None:
        if not arrays:
            return
        with self.cp.cuda.Device(self.device_id):
            event = self.cp.cuda.Event()
            event.record(self.cp.cuda.get_current_stream())
        self._inflight_buffers.append(_InFlightBuffers(event, arrays))

    def _moe_address_table_cacheable(self, address_key: tuple[int, ...]) -> bool:
        return all(
            self._cache.get(("q4", tensor_id)) is not None
            for tensor_id in address_key
        )

    def _invalidate_moe_address_tables(self, tensor_id: int) -> None:
        stale = [
            key for key in self._moe_address_cache if tensor_id in key
        ]
        for key in stale:
            self._moe_address_cache.pop(key, None)

    def clear(self) -> None:
        self._expert_load_executor.shutdown(
            wait=True, cancel_futures=True
        )
        self._pending_expert_loads.clear()
        with self.cp.cuda.Device(self.device_id):
            self.cp.cuda.get_current_stream().synchronize()
            self._prefetch_stream.synchronize()
        self._pending_prefetches.clear()
        self._inflight_buffers.clear()
        self._release_resident_moe_table()
        self._moe_address_cache.clear()
        self._unused_prefetches.clear()
        self._cache.clear()
        self._rope_cache.clear()
        self._moe_transfer_cache.clear()
        self._v2_weight_cache.clear()
        self._delta_segment_tables.clear()
        self._delta_segment_scratch_cache = None
        self._profile_events.clear()
        self._profile_host_seconds.clear()
        self._profile_host_calls.clear()
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
            "cache_eviction_scans": self.cache_eviction_scans,
            "cache_eviction_entries_examined": (
                self.cache_eviction_entries_examined
            ),
            "cache_priority_promotions": self.cache_priority_promotions,
            "cache_priority_entries": sum(
                1
                for entry in self._cache.values()
                if entry.priority_until > self._cache_priority_epoch
            ),
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
            "expert_load_pending": len(self._pending_expert_loads),
            "expert_load_requests": self.expert_load_requests,
            "expert_load_completions": self.expert_load_completions,
            "inflight_buffer_groups": len(self._inflight_buffers),
            "moe_address_cache_entries": len(self._moe_address_cache),
            "moe_address_cache_hits": self.moe_address_cache_hits,
            "moe_address_cache_misses": self.moe_address_cache_misses,
            "resident_moe_layer": (
                self._resident_moe_table["layer"].layer
                if self._resident_moe_table is not None
                else -1
            ),
            "profile": self._profile_stats() if self.profiling else None,
        }

    def _reserve_cache(
        self, byte_size: int, *, allow_protected: bool
    ) -> bool:
        self._reap_prefetches()
        self._reap_inflight_buffers()
        self._cache_priority_epoch += 1
        while self._cache and self.cache_bytes + byte_size > self.cache_limit_bytes:
            self.cache_eviction_scans += 1
            eviction_key = None
            priority_fallback = None
            for candidate_key, candidate in self._cache.items():
                self.cache_eviction_entries_examined += 1
                if (
                    candidate.protected
                    or candidate_key in self._pending_prefetches
                ):
                    continue
                if candidate.priority_until <= self._cache_priority_epoch:
                    eviction_key = candidate_key
                    break
                if priority_fallback is None:
                    priority_fallback = candidate_key
            if eviction_key is None:
                eviction_key = priority_fallback
            if eviction_key is None and allow_protected:
                priority_fallback = None
                for candidate_key, candidate in self._cache.items():
                    self.cache_eviction_entries_examined += 1
                    if (
                        not candidate.protected
                        or candidate_key in self._pending_prefetches
                    ):
                        continue
                    if candidate.priority_until <= self._cache_priority_epoch:
                        eviction_key = candidate_key
                        break
                    if priority_fallback is None:
                        priority_fallback = candidate_key
                if eviction_key is None:
                    eviction_key = priority_fallback
            if eviction_key is None:
                return False
            evicted = self._cache.pop(eviction_key)
            if eviction_key[0] == "q4":
                self._invalidate_moe_address_tables(eviction_key[1])
                if (
                    self._resident_moe_table is not None
                    and eviction_key[1]
                    in self._resident_moe_table["tensor_ids"]
                ):
                    self._release_resident_moe_table()
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
            if kind == "q4":
                entry.priority_until = 0
            self._cache.move_to_end(key)
            return entry.arrays
        self.cache_misses += 1
        can_cache = byte_size <= self.cache_limit_bytes and self._reserve_cache(
            byte_size, allow_protected=True
        )
        if self.profiling:
            started = time.perf_counter()
            arrays = upload()
            self.profile_upload_seconds += time.perf_counter() - started
            self.profile_upload_calls += 1
        else:
            arrays = upload()
        if can_cache:
            self._cache[key] = _CacheEntry(
                owner, arrays, byte_size, protected
            )
            self.cache_bytes += byte_size
        return arrays


_accelerator: CudaAccelerator | None = None


def configure_cuda(
    *, cache_mib: int = 8192, device_id: int = 0, kv_cache_type: str | None = None
) -> CudaAccelerator:
    global _accelerator
    if _accelerator is not None:
        _accelerator.clear()
    _accelerator = CudaAccelerator(
        cache_mib=cache_mib,
        device_id=device_id,
        kv_cache_type=kv_cache_type,
    )
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
void delta_conv_step(
    const float* mixed_qkv,
    const float* weights,
    float* state,
    float* output,
    const int channels,
    const int kernel_size
) {
    // Single-token variant: no cross-token recurrence, so one thread per
    // channel instead of one single-thread block per channel.
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    float* channel_state = state + channel * kernel_size;
    const float* channel_weights = weights + channel * kernel_size;
    for (int index = 0; index + 1 < kernel_size; ++index) {
        channel_state[index] = channel_state[index + 1];
    }
    channel_state[kernel_size - 1] = mixed_qkv[channel];
    float value = 0.0f;
    for (int index = 0; index < kernel_size; ++index) {
        value += channel_state[index] * channel_weights[index];
    }
    output[channel] = value / (1.0f + expf(-value));
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
void bf16_matmul_small(
    const unsigned short* weights,
    const float* vectors,
    float* output,
    const int rows,
    const int columns,
    const int tokens
) {
    // Small token batches (speculative verify): read each weight element
    // once and accumulate every token in registers, instead of re-streaming
    // the weight matrix per token like the (rows, tokens)-grid kernel.
    const int row = blockIdx.x;
    if (row >= rows || tokens > 8) return;
    float partial[8];
    for (int token = 0; token < 8; ++token) partial[token] = 0.0f;
    const int weight_start = row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned int bits =
            ((unsigned int)weights[weight_start + column]) << 16;
        const float weight = __uint_as_float(bits);
        for (int token = 0; token < tokens; ++token) {
            partial[token] += weight * vectors[token * columns + column];
        }
    }
    for (int token = 0; token < tokens; ++token) {
        const float total = block_reduce_sum(partial[token]);
        if (threadIdx.x == 0) output[token * rows + row] = total;
        __syncthreads();
    }
}

extern "C" __global__
void q4_matmul_small(
    const unsigned char* packed,
    const __half* scales,
    const float* vectors,
    float* output,
    const int rows,
    const int columns,
    const int tokens
) {
    const int row = blockIdx.x;
    if (row >= rows || tokens > 8) return;
    float partial[8];
    for (int token = 0; token < 8; ++token) partial[token] = 0.0f;
    const int weight_start = row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = weight_start + column;
        const int block = index >> 5;
        const int within_block = index & 31;
        const unsigned char byte = packed[block * 16 + (within_block >> 1)];
        const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
        const float weight = (float)(nibble - 8) * __half2float(scales[block]);
        for (int token = 0; token < tokens; ++token) {
            partial[token] += weight * vectors[token * columns + column];
        }
    }
    for (int token = 0; token < tokens; ++token) {
        const float total = block_reduce_sum(partial[token]);
        if (threadIdx.x == 0) output[token * rows + row] = total;
        __syncthreads();
    }
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
void q4_selected_batched_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const int* selected_ids,
    const float* vector,
    float* output,
    const int rows,
    const int columns,
    const int selected_count
) {
    const int row = blockIdx.x;
    const int selected = blockIdx.y;
    if (row >= rows || selected >= selected_count) return;
    const int expert = selected_ids[selected];
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
        partial += (float)(nibble - 8)
            * __half2float(scales[block]) * vector[column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[selected * rows + row] = partial;
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
void q4_silu_batched(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const float* vector,
    float* output,
    const int intermediate_size,
    const int columns,
    const int expert_count
) {
    const int intermediate = blockIdx.x;
    const int expert = blockIdx.y;
    if (intermediate >= intermediate_size || expert >= expert_count) return;
    const unsigned char* packed =
        (const unsigned char*)packed_addresses[expert];
    const __half* scales = (const __half*)scale_addresses[expert];
    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    const int gate_row = intermediate;
    const int up_row = intermediate_size + intermediate;
    const int gate_start = gate_row * columns;
    const int up_start = up_row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int gate_index = gate_start + column;
        const int gate_block = gate_index >> 5;
        const int gate_within = gate_index & 31;
        const unsigned char gate_byte =
            packed[gate_block * 16 + (gate_within >> 1)];
        const int gate_nibble = (gate_within & 1)
            ? (gate_byte >> 4) : (gate_byte & 15);
        gate_partial += (float)(gate_nibble - 8)
            * __half2float(scales[gate_block]) * vector[column];

        const int up_index = up_start + column;
        const int up_block = up_index >> 5;
        const int up_within = up_index & 31;
        const unsigned char up_byte =
            packed[up_block * 16 + (up_within >> 1)];
        const int up_nibble = (up_within & 1)
            ? (up_byte >> 4) : (up_byte & 15);
        up_partial += (float)(up_nibble - 8)
            * __half2float(scales[up_block]) * vector[column];
    }
    gate_partial = block_reduce_sum(gate_partial);
    up_partial = block_reduce_sum(up_partial);
    if (threadIdx.x == 0) {
        const float exponential = gate_partial >= 0.0f
            ? expf(-gate_partial) : expf(gate_partial);
        const float sigmoid = gate_partial >= 0.0f
            ? 1.0f / (1.0f + exponential)
            : exponential / (1.0f + exponential);
        output[expert * intermediate_size + intermediate] =
            gate_partial * sigmoid * up_partial;
    }
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
void q4_selected_weighted_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const int* selected_ids,
    const float* vectors,
    const float* routing_weights,
    float* output,
    const int rows,
    const int columns,
    const int selected_count
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float partial = 0.0f;
    const int start = row * columns;
    for (int selected = 0; selected < selected_count; ++selected) {
        const int expert = selected_ids[selected];
        const unsigned char* packed =
            (const unsigned char*)packed_addresses[expert];
        const __half* scales = (const __half*)scale_addresses[expert];
        const float* vector = vectors + selected * columns;
        const float route = routing_weights[selected];
        for (int column = threadIdx.x; column < columns; column += blockDim.x) {
            const int index = start + column;
            const int block = index >> 5;
            const int within_block = index & 31;
            const unsigned char byte = packed[block * 16 + (within_block >> 1)];
            const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
            const float weight = (float)(nibble - 8)
                * __half2float(scales[block]);
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

extern "C" __global__
void q8_matvec_transposed(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        // GGML dimension 0 is contiguous: each logical output row contains
        // input_size values even though GGUF reports [input, output].
        const int absolute = row * input_size + input;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float scale = __half2float(
            *((const __half*)(packed + block * 34))
        );
        const signed char value = *((const signed char*)(packed + block * 34 + 2 + within));
        partial += ((float)value * scale) * vector[input];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void q8_matvec_transposed_warp(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = lane; input < input_size; input += 32) {
        const int absolute = row * input_size + input;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float scale = __half2float(
            *((const __half*)(packed + block * 34))
        );
        const signed char value = *((const signed char*)(
            packed + block * 34 + 2 + within
        ));
        partial += ((float)value * scale) * vector[input];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) output[row] = partial;
}

__device__ __forceinline__ void q5k_scale_min(
    const unsigned char* scales, int index, int* scale, int* minimum
) {
    if (index < 4) {
        *scale = scales[index] & 63;
        *minimum = scales[index + 4] & 63;
    } else {
        *scale = (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4);
        *minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
    }
}

__device__ __forceinline__ float q5k_value(
    const unsigned char* packed, int absolute
) {
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 176;
    const float d = __half2float(*((const __half*)(base)));
    const float dmin = __half2float(*((const __half*)(base + 2)));
    const unsigned char* scales = base + 4;
    const int group = within / 64;
    const int offset = within & 63;
    const int sub = offset / 32;
    const int qindex = group * 32 + (offset & 31);
    const unsigned char low = base[48 + qindex];
    const unsigned char high = base[16 + (offset & 31)];
    const int bit = (high >> (2 * group + sub)) & 1;
    const int quant = ((offset < 32) ? (low & 15) : (low >> 4)) + 16 * bit;
    int scale, minimum;
    q5k_scale_min(scales, group * 2 + sub, &scale, &minimum);
    return d * (float)scale * (float)quant - dmin * (float)minimum;
}

extern "C" __global__
void q5k_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += q5k_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void q5k_swiglu_transposed(
    const unsigned char* gate_packed,
    const unsigned char* up_packed,
    const float* vector,
    float* activated,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += q5k_value(gate_packed, absolute) * value;
        up += q5k_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[row] = (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void q5k_grouped_swiglu(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const float* vector,
    float* activated,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    if (row >= output_size || expert >= experts) return;
    const unsigned char* gate_packed =
        (const unsigned char*)gate_ptrs[expert];
    const unsigned char* up_packed =
        (const unsigned char*)up_ptrs[expert];
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += q5k_value(gate_packed, absolute) * value;
        up += q5k_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[expert * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

__device__ __forceinline__ float q4k_value(
    const unsigned char* packed, int absolute
) {
    // Q4_K: like Q5_K but no qh block, so ql starts at +16 and quants are 4-bit.
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 144;
    const float d = __half2float(*((const __half*)(base)));
    const float dmin = __half2float(*((const __half*)(base + 2)));
    const unsigned char* scales = base + 4;
    const int group = within / 64;
    const int offset = within & 63;
    const int sub = offset / 32;
    const unsigned char low = base[16 + group * 32 + (offset & 31)];
    const int quant = (offset < 32) ? (low & 15) : (low >> 4);
    int scale, minimum;
    q5k_scale_min(scales, group * 2 + sub, &scale, &minimum);
    return d * (float)scale * (float)quant - dmin * (float)minimum;
}

extern "C" __global__
void q4k_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += q4k_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

__device__ __forceinline__ float q6k_value(
    const unsigned char* packed, int absolute
) {
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 210;
    const unsigned char* ql = base;
    const unsigned char* qh = base + 128;
    const signed char* scales = (const signed char*)(base + 192);
    const float d = __half2float(*((const __half*)(base + 208)));
    const int half = within / 128;
    const int offset = within & 127;
    const int lane = offset / 32;
    const int l = offset & 31;
    const int qindex = l + ((lane == 0 || lane == 2) ? 0 : 32);
    const unsigned char qbyte = ql[half * 64 + qindex];
    const unsigned char high = qh[half * 32 + l];
    const int nibble = (lane == 0 || lane == 1) ? (qbyte & 15) : (qbyte >> 4);
    const int quant = (nibble | (((high >> (lane * 2)) & 3) << 4)) - 32;
    const int scale_index = half * 8 + (l / 16) + lane * 2;
    return d * (float)scales[scale_index] * (float)quant;
}

extern "C" __global__
void q6k_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += q6k_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void q6k_accumulate_transposed(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const float* weights,
    const int weight_index,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += q6k_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] += weights[weight_index] * partial;
}

extern "C" __global__
void q6k_grouped_accumulate(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        float combined = 0.0f;
        for (int expert = 0; expert < experts; ++expert) {
            const unsigned char* packed =
                (const unsigned char*)down_ptrs[expert];
            combined += weights[expert]
                * q6k_value(packed, row * input_size + input)
                * activated[expert * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] += partial;
}

extern "C" __global__
void q8_grouped_accumulate(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const int absolute = row * input_size + input;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        float combined = 0.0f;
        for (int expert = 0; expert < experts; ++expert) {
            const unsigned char* packed =
                (const unsigned char*)down_ptrs[expert];
            const float scale = __half2float(
                *((const __half*)(packed + block * 34))
            );
            const signed char value = *((const signed char*)(
                packed + block * 34 + 2 + within
            ));
            combined += weights[expert] * ((float)value * scale)
                * activated[expert * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] += partial;
}

// ---- NVFP4 kernels (GGML type 40) -----------------------------------------
__device__ __forceinline__ float ue4m3_to_float(unsigned char bits) {
    const int s = (bits >> 7) & 1;
    const int e = (bits >> 3) & 0xF;
    const int m = bits & 7;
    float val;
    if (e == 0) val = ldexpf(m / 8.0f, -6);
    else if (e == 0xF) val = (m == 0) ? __int_as_float(0x7f800000) : __int_as_float(0x7fffffff);
    else val = ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -val : val;
}
__device__ __forceinline__ float nvfp4_value(
    const unsigned char* packed, int absolute
) {
    const int sub = absolute / 16;
    const int within = absolute & 15;
    const unsigned char* base = packed + sub * 9;
    const float scale = ue4m3_to_float(base[0]);
    const unsigned char nibble_pair = base[1 + (within >> 1)];
    const int val = (within & 1) ? (nibble_pair >> 4) : (nibble_pair & 0x0F);
    const float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return scale * lut[val];
}

extern "C" __global__
void nvfp4_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += nvfp4_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void nvfp4_grouped_swiglu(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const float* vector,
    float* activated,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    if (row >= output_size || expert >= experts) return;
    const unsigned char* gate_packed = (const unsigned char*)gate_ptrs[expert];
    const unsigned char* up_packed = (const unsigned char*)up_ptrs[expert];
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += nvfp4_value(gate_packed, absolute) * value;
        up += nvfp4_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[expert * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void nvfp4_grouped_accumulate(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        float combined = 0.0f;
        for (int expert = 0; expert < experts; ++expert) {
            const unsigned char* packed = (const unsigned char*)down_ptrs[expert];
            combined += weights[expert]
                * nvfp4_value(packed, row * input_size + input)
                * activated[expert * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] += partial;
}

extern "C" __global__
void kv_attention(
    const float* query,
    const float* keys,
    const float* values,
    float* output,
    const int heads,
    const int kv_heads,
    const int head_dim,
    const int tokens,
    const int capacity,
    const float scale
) {
    const int head = blockIdx.x;
    if (head >= heads || threadIdx.x != 0) return;
    const int group = heads / kv_heads;
    const int kv_head = head / group;
    const float* q = query + head * head_dim;
    float maximum = -3.402823466e+38F;
    for (int token = 0; token < tokens; ++token) {
        float score = 0.0f;
        const float* k = keys + (kv_head * capacity + token) * head_dim;
        for (int d = 0; d < head_dim; ++d) score += q[d] * k[d];
        maximum = fmaxf(maximum, score * scale);
    }
    float denominator = 0.0f;
    for (int d = 0; d < head_dim; ++d) output[head * head_dim + d] = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        float score = 0.0f;
        const float* k = keys + (kv_head * capacity + token) * head_dim;
        for (int d = 0; d < head_dim; ++d) score += q[d] * k[d];
        const float weight = expf(score * scale - maximum);
        denominator += weight;
        const float* v = values + (kv_head * capacity + token) * head_dim;
        for (int d = 0; d < head_dim; ++d) output[head * head_dim + d] += weight * v[d];
    }
    for (int d = 0; d < head_dim; ++d) output[head * head_dim + d] /= denominator;
}

extern "C" __global__
void kv_append(
    const float* current_keys,
    const float* current_values,
    float* cache_keys,
    float* cache_values,
    const int kv_heads,
    const int head_dim,
    const int position,
    const int capacity
) {
    const int head = blockIdx.x;
    if (head >= kv_heads) return;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        cache_keys[(head * capacity + position) * head_dim + d] =
            current_keys[head * head_dim + d];
        cache_values[(head * capacity + position) * head_dim + d] =
            current_values[head * head_dim + d];
    }
}
"""
