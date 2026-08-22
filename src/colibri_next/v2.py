"""Opt-in bindings for the native Colibrì v2 ABI.

The module is intentionally small: model bytes stay memory mapped in the
native process and Python only owns handles and request-level data.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:  # optional: only used to keep the logits path off the Python interpreter
    import numpy as _numpy
except ImportError:  # pragma: no cover - numpy is normally present
    _numpy = None


class V2Error(RuntimeError):
    pass


_EXPERT_MODE_ALIASES = {
    "cpu": "cpu",
    "auto": "auto",
    "hybrid": "legacy-hybrid",
    "resident": "resident",
    "gpu": "legacy-paging",
    "legacy-paging": "legacy-paging",
    "legacy-hybrid": "legacy-hybrid",
}

_SPECIAL_TOKEN_PATTERN = re.compile(r"<[^<>]+>")


def _gguf_byte_decoder() -> dict[int, int]:
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    inverse = {value: value for value in direct}
    extra = 0
    for value in range(256):
        if value not in direct:
            inverse[256 + extra] = value
            extra += 1
    return inverse


_GGUF_BYTE_DECODER = _gguf_byte_decoder()


def _resolve_expert_mode(value: str) -> tuple[str, int, bool]:
    try:
        resolved = _EXPERT_MODE_ALIASES[value]
    except (KeyError, TypeError) as error:
        choices = "', '".join(_EXPERT_MODE_ALIASES)
        raise ValueError(f"expert_mode must be one of '{choices}'") from error
    native_mode = {
        "resident": 0,
        "legacy-paging": 0,
        "cpu": 1,
        "auto": 2,
        "legacy-hybrid": 2,
    }[resolved]
    return resolved, native_mode, resolved == "resident"


def _normalize_prefill_cache_seed(value: int | str) -> tuple[int, bool]:
    if value == "auto":
        return 0, True
    if value == "off":
        return 0, False
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("prefill_cache_seed must be 'auto', 'off', or an integer")
    if not 0 <= value <= 256:
        raise ValueError("prefill_cache_seed must be between 0 and 256")
    return value, False


class _Info(ctypes.Structure):
    _fields_ = [
        ("gguf_version", ctypes.c_uint32),
        ("tensor_count", ctypes.c_uint64),
        ("metadata_count", ctypes.c_uint64),
        ("file_size", ctypes.c_uint64),
        ("alignment", ctypes.c_uint32),
        ("architecture", ctypes.c_char * 64),
        ("name", ctypes.c_char * 128),
        ("format", ctypes.c_char * 32),
    ]


class _TensorInfo(ctypes.Structure):
    _fields_ = [
        ("dimensions", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * 4),
        ("ggml_type", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("name", ctypes.c_char * 192),
    ]


class _HfQuantOption(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 8),
        ("arena_bytes", ctypes.c_uint64),
        ("cache_bytes", ctypes.c_uint64),
        ("cache_path", ctypes.c_char * 512),
        ("unavailable", ctypes.c_char * 128),
    ]


class _ModelConfig(ctypes.Structure):
    _fields_ = [
        ("architecture", ctypes.c_char * 64),
        ("hidden_size", ctypes.c_uint32),
        ("layer_count", ctypes.c_uint32),
        ("attention_heads", ctypes.c_uint32),
        ("attention_kv_heads", ctypes.c_uint32),
        ("context_length", ctypes.c_uint32),
        ("intermediate_size", ctypes.c_uint32),
        ("expert_count", ctypes.c_uint32),
        ("expert_used_count", ctypes.c_uint32),
        ("vocabulary_size", ctypes.c_uint32),
        ("rotary_dimension", ctypes.c_uint32),
        ("full_attention_interval", ctypes.c_uint32),
        ("sliding_window", ctypes.c_uint32),
        ("sliding_window_pattern_length", ctypes.c_uint32),
        ("rms_norm_epsilon", ctypes.c_float),
        ("rope_freq_base", ctypes.c_float),
        ("eos_token_id", ctypes.c_uint32),
        ("eot_token_id", ctypes.c_uint32),
        ("bos_token_id", ctypes.c_uint32),
        # DeepSeek-V4 geometry; zero on every other architecture.
        ("q_lora_rank", ctypes.c_uint32),
        ("kv_lora_rank", ctypes.c_uint32),
        ("output_lora_rank", ctypes.c_uint32),
        ("output_group_count", ctypes.c_uint32),
        ("indexer_head_count", ctypes.c_uint32),
        ("indexer_key_length", ctypes.c_uint32),
        ("indexer_top_k", ctypes.c_uint32),
        ("hyper_connection_count", ctypes.c_uint32),
        ("sinkhorn_iterations", ctypes.c_uint32),
        ("expert_shared_count", ctypes.c_uint32),
        ("hash_layer_count", ctypes.c_uint32),
        ("compress_ratios_length", ctypes.c_uint32),
        ("sinkhorn_epsilon", ctypes.c_float),
        ("compress_rope_freq_base", ctypes.c_float),
        ("rope_scaling_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("rope_original_context_length", ctypes.c_uint32),
        ("draft_block_size", ctypes.c_uint32),
        ("target_layers_length", ctypes.c_uint32),
        ("mask_token_id", ctypes.c_uint32),
        # Head-output transforms; zero when the checkpoint asked for neither.
        ("logit_scale", ctypes.c_float),
        ("final_logit_softcap", ctypes.c_float),
        ("expert_group_count", ctypes.c_uint32),
        ("expert_group_used", ctypes.c_uint32),
    ]


class _Deepseek4Info(ctypes.Structure):
    _fields_ = [
        ("layers", ctypes.c_uint32),
        ("window_layers", ctypes.c_uint32),
        ("csa_layers", ctypes.c_uint32),
        ("hca_layers", ctypes.c_uint32),
        ("context_limit", ctypes.c_uint32),
        ("positions", ctypes.c_uint32),
        ("state_bytes", ctypes.c_uint64),
        ("resolved_tensors", ctypes.c_uint32),
        ("forward_calls", ctypes.c_uint64),
        ("forward_nanoseconds", ctypes.c_uint64),
        ("routed_expert_nanoseconds", ctypes.c_uint64),
        ("shared_expert_nanoseconds", ctypes.c_uint64),
        ("attention_nanoseconds", ctypes.c_uint64),
        ("head_nanoseconds", ctypes.c_uint64),
        ("attention_core_nanoseconds", ctypes.c_uint64),
        ("routed_expert_bytes", ctypes.c_uint64),
        ("indexer_selections", ctypes.c_uint64),
        ("indexer_candidates", ctypes.c_uint64),
        ("expert_prefetch_bytes", ctypes.c_uint64),
        ("gpu_weight_bytes", ctypes.c_uint64),
        ("gpu_matvec_calls", ctypes.c_uint64),
        ("gpu_batches", ctypes.c_uint64),
        ("hyper_nanoseconds", ctypes.c_uint64),
        ("matvec_nanoseconds", ctypes.c_uint64),
        ("prefill_calls", ctypes.c_uint64),
        ("prefill_tokens", ctypes.c_uint64),
        ("prefill_nanoseconds", ctypes.c_uint64),
        ("expert_cache_bytes", ctypes.c_uint64),
        ("expert_cache_slots", ctypes.c_uint64),
        ("expert_cache_hits", ctypes.c_uint64),
        ("expert_cache_misses", ctypes.c_uint64),
        ("expert_cache_evictions", ctypes.c_uint64),
    ]


class _GpuInfo(ctypes.Structure):
    _fields_ = [
        ("available", ctypes.c_int32),
        ("device", ctypes.c_int32),
        ("compute_major", ctypes.c_int32),
        ("compute_minor", ctypes.c_int32),
        ("total_memory", ctypes.c_uint64),
        ("free_memory", ctypes.c_uint64),
    ]


class _MemoryPlan(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "budget",
            "static_weights",
            "kv_state",
            "workspace",
            "active_experts",
            "staging",
            "unused",
        )
    ]


class _QwenRuntimeOptions(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_int32),
        ("moe_device", ctypes.c_int32),
        ("mtp_drafts", ctypes.c_uint32),
        ("expert_top_k", ctypes.c_uint32),
        ("context_limit", ctypes.c_uint64),
        ("gpu_cache_bytes", ctypes.c_uint64),
        ("expert_top_p", ctypes.c_float),
        ("cache_type_k", ctypes.c_int32),
        ("cache_type_v", ctypes.c_int32),
        ("prefill_checkpoint_interval", ctypes.c_uint32),
        ("prefill_checkpoint_slots", ctypes.c_uint32),
        ("parallel_sequences", ctypes.c_uint32),
        ("prompt_cache_mib", ctypes.c_uint32),
        ("swa_full", ctypes.c_uint32),
        ("prefill_cache_seed", ctypes.c_uint32),
        ("expert_paging", ctypes.c_uint32),
        ("cpu_prefetch_mib", ctypes.c_uint32),
        ("cpu_prefetch_auto", ctypes.c_uint32),
        ("next_layer_prefetch", ctypes.c_uint32),
        ("cpu_threads", ctypes.c_uint32),
        ("hybrid_prefill_cpu", ctypes.c_uint32),
        ("immutable_residency", ctypes.c_uint32),
        ("prefill_cache_seed_auto", ctypes.c_uint32),
        ("strict_resident", ctypes.c_uint32),
        ("dense_requant", ctypes.c_uint32),
        ("prefill_expert_stream_mib", ctypes.c_int32),
    ]


class _QwenRuntimeInfo(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "layers",
            "deltanet_layers",
            "attention_layers",
            "swa_layers",
            "sliding_window",
            "swa_full",
            "hidden_size",
            "expert_count",
            "expert_used_count",
            "mtp_available",
            "mtp_enabled",
            "mtp_drafts",
            "mtp_layer",
        )
    ] + [
        ("context_limit", ctypes.c_uint64),
        ("static_tensor_bytes", ctypes.c_uint64),
        ("expert_tensor_bytes", ctypes.c_uint64),
        ("gpu_allocated_bytes", ctypes.c_uint64),
        ("workspace_bytes", ctypes.c_uint64),
        ("state_bytes", ctypes.c_uint64),
        ("expert_staging_bytes", ctypes.c_uint64),
        ("expert_cache_bytes", ctypes.c_uint64),
        ("expert_cache_slots", ctypes.c_uint64),
        ("expert_cache_hits", ctypes.c_uint64),
        ("expert_cache_misses", ctypes.c_uint64),
        ("expert_cache_evictions", ctypes.c_uint64),
        ("expert_cache_admissions", ctypes.c_uint64),
        ("expert_cache_rejections", ctypes.c_uint64),
        ("expert_cache_prompt_bypasses", ctypes.c_uint64),
        ("prefix_cache_hits", ctypes.c_uint64),
        ("prefix_cache_misses", ctypes.c_uint64),
        ("prefix_cache_reused_tokens", ctypes.c_uint64),
        ("mtp_tensor_bytes", ctypes.c_uint64),
        ("mtp_draft_tokens", ctypes.c_uint64),
        ("mtp_accepted_tokens", ctypes.c_uint64),
        ("mtp_rejected_tokens", ctypes.c_uint64),
        ("mtp_draft_nanoseconds", ctypes.c_uint64),
        ("mtp_verify_nanoseconds", ctypes.c_uint64),
        ("mtp_rollback_nanoseconds", ctypes.c_uint64),
        ("decode_calls", ctypes.c_uint64),
        ("decode_nanoseconds", ctypes.c_uint64),
        ("route_wait_nanoseconds", ctypes.c_uint64),
        ("expert_page_nanoseconds", ctypes.c_uint64),
        ("tail_wait_nanoseconds", ctypes.c_uint64),
        ("position", ctypes.c_uint64),
        ("device", ctypes.c_int32),
        ("moe_device", ctypes.c_int32),
        ("cuda_ready", ctypes.c_int32),
        ("decode_ready", ctypes.c_int32),
        ("route_expert_sum", ctypes.c_uint64),
        ("expert_compute_nanoseconds", ctypes.c_uint64),
        ("prefix_cache_reprefilled_tokens", ctypes.c_uint64),
        ("prefix_cache_last_prompt_tokens", ctypes.c_uint64),
        ("prefix_cache_last_reused_tokens", ctypes.c_uint64),
        ("prefix_cache_last_lcp_live", ctypes.c_uint64),
        ("prefix_cache_last_lcp_snapshot", ctypes.c_uint64),
        ("prompt_cache_entries", ctypes.c_uint64),
        ("prompt_cache_used_bytes", ctypes.c_uint64),
        ("prefill_cache_seeded_experts", ctypes.c_uint64),
        ("prefill_cache_seed_nanoseconds", ctypes.c_uint64),
        ("direct_paging", ctypes.c_uint64),
        ("paging_registration_nanoseconds", ctypes.c_uint64),
        ("host_available_bytes", ctypes.c_uint64),
        ("cpu_prefetch_experts", ctypes.c_uint64),
        ("cpu_prefetch_bytes", ctypes.c_uint64),
        ("cpu_prefetch_nanoseconds", ctypes.c_uint64),
        ("cpu_prefetch_pages", ctypes.c_uint64),
        ("cpu_prefetch_cold_pages", ctypes.c_uint64),
        ("cpu_prefetch_loaded_pages", ctypes.c_uint64),
        ("cpu_prefetch_auto_skips", ctypes.c_uint64),
        ("cpu_prefetch_last_budget_bytes", ctypes.c_uint64),
        ("prefill_calls", ctypes.c_uint64),
        ("prefill_tokens", ctypes.c_uint64),
        ("prefill_nanoseconds", ctypes.c_uint64),
        ("prefill_route_wait_nanoseconds", ctypes.c_uint64),
        ("prefill_expert_nanoseconds", ctypes.c_uint64),
        ("prefill_direct_quant", ctypes.c_uint64),
        ("prefill_direct_quant_width", ctypes.c_uint64),
        ("prefill_profile", ctypes.c_uint64),
        ("prefill_gpu_core_nanoseconds", ctypes.c_uint64),
        ("prefill_gpu_router_nanoseconds", ctypes.c_uint64),
        ("prefill_gpu_transfer_nanoseconds", ctypes.c_uint64),
        ("expert_history_loaded_entries", ctypes.c_uint64),
        ("expert_history_saves", ctypes.c_uint64),
        ("next_layer_prefetch_predictions", ctypes.c_uint64),
        ("next_layer_prefetch_hits", ctypes.c_uint64),
        ("next_layer_prefetch_bytes", ctypes.c_uint64),
        ("next_layer_prefetch_trained_pairs", ctypes.c_uint64),
        ("nvfp4_tensor_core_moe_calls", ctypes.c_uint64),
        ("nvfp4_tensor_core_moe_fallbacks", ctypes.c_uint64),
        ("nvfp4_tensor_core_moe_last_status", ctypes.c_int64),
        ("host_ffn_layers", ctypes.c_uint64),
        ("host_ffn_bytes", ctypes.c_uint64),
        ("dense_host_nanoseconds", ctypes.c_uint64),
        ("expert_cache_deferred_admissions", ctypes.c_uint64),
        ("expert_residency_epochs", ctypes.c_uint64),
        ("expert_residency_frozen", ctypes.c_uint64),
        ("prefill_cache_seed_bytes", ctypes.c_uint64),
        ("prefill_cache_seed_selected_experts", ctypes.c_uint64),
        ("prefill_cache_seed_hits", ctypes.c_uint64),
        ("prefill_cache_seed_avoided_misses", ctypes.c_uint64),
        ("prefill_cache_seed_auto_skips", ctypes.c_uint64),
        ("prefill_cache_seed_budget_stops", ctypes.c_uint64),
        ("sampling_gpu_topk_calls", ctypes.c_uint64),
        ("sampling_gpu_topk_bytes", ctypes.c_uint64),
        ("sampling_full_download_bytes", ctypes.c_uint64),
        ("sampling_nanoseconds", ctypes.c_uint64),
        ("route_recurrence_observations", ctypes.c_uint64),
        ("route_recurrence_prev_hits", ctypes.c_uint64),
        ("route_recurrence_window_hits", ctypes.c_uint64),
        ("route_recurrence_layer_samples", ctypes.c_uint64),
        ("route_recurrence_window_experts", ctypes.c_uint64),
        ("route_recurrence_resident", ctypes.c_uint64),
        ("route_recurrence_miss_in_window", ctypes.c_uint64),
        ("route_recurrence_miss_cold", ctypes.c_uint64),
        ("resolved_cache_type_k", ctypes.c_int32),
        ("resolved_cache_type_v", ctypes.c_int32),
        ("grammar_constrained_steps", ctypes.c_uint64),
        ("grammar_rejected_candidates", ctypes.c_uint64),
        ("grammar_empty_candidate_sets", ctypes.c_uint64),
        ("multi_decode_batches", ctypes.c_uint64),
        ("multi_decode_tokens", ctypes.c_uint64),
        ("prefill_streamed_bytes", ctypes.c_uint64),
    ]


# KV cache precision codes shared with the native runtime. "auto" is resolved
# natively at prepare time, where the context limit and every attention layer's
# head_dim are known -- turbo needs a power-of-two head_dim and raises
# otherwise, so it cannot be picked blind. See v2_runtime.cpp.
#
# "auto" is deliberately NOT the default. It selects turbo4 above 32K, which
# changes generated tokens, and the regime it switches on in -- long context --
# is exactly where KV quantization damage concentrates (retrieval and
# instruction adherence degrade well before fluency does). No long-context
# quality benchmark has been run, so this stays opt-in, matching llama.cpp
# (-ctk/-ctv) and vLLM, which both ship an unquantized KV default.
CACHE_TYPE_CODES = {
    "f32": 0, "f16": 1, "bf16": 2, "q8_0": 3, "turbo3": 4, "turbo4": 5,
    "auto": 6,
}
CACHE_TYPE_NAMES = {code: name for name, code in CACHE_TYPE_CODES.items()}


_TokenCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p)


class _QwenTaskEvent(ctypes.Structure):
    _fields_ = [
        ("task_id", ctypes.c_uint64),
        ("token", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
    ]


TASK_EVENT_TOKEN = 0
TASK_EVENT_DONE = 1
TASK_EVENT_ERROR = 2
TASK_EVENT_PREFILL = 3
NATIVE_ABI_VERSION = 5
# "Let the runtime pick the size" for the host prompt cache, carried through the
# option struct as a size. Lives here rather than in the CLI so a caller that is
# not the CLI -- and the runtimes that read it -- can say the same thing.
AUTO_PROMPT_CACHE_MIB = (1 << 32) - 1

_cached_library: ctypes.CDLL | None = None


# Mirrors ColibriBackend in native/include/colibri_backend.hpp.
_BACKENDS = {"auto": -1, "cuda": 0, "cpu": 1}


def _library() -> ctypes.CDLL:
    global _cached_library
    if _cached_library is not None:
        return _cached_library
    root = Path(__file__).with_name("_native")
    # Prefer the current platform's library extension first. A stale .so left in
    # _native (e.g. from a Linux build) must not shadow the Windows .dll, or
    # ctypes raises WinError 193 trying to load an ELF image.
    if os.name == "nt":
        names = ("colibri_v2.dll", "colibri_v2.dylib", "colibri_v2.so")
    elif sys.platform == "darwin":
        names = ("colibri_v2.dylib", "colibri_v2.so", "colibri_v2.dll")
    else:
        names = ("colibri_v2.so", "colibri_v2.dylib", "colibri_v2.dll")
    for name in names:
        path = root / name
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            try:
                lib.colibri_v2_last_error.restype = ctypes.c_char_p
                lib.colibri_v2_version.argtypes = []
                lib.colibri_v2_version.restype = ctypes.c_uint32
                lib.colibri_v2_runtime_options_size.argtypes = []
                lib.colibri_v2_runtime_options_size.restype = ctypes.c_uint64
                lib.colibri_v2_runtime_info_size.argtypes = []
                lib.colibri_v2_runtime_info_size.restype = ctypes.c_uint64
                version = int(lib.colibri_v2_version())
                native_options_size = int(lib.colibri_v2_runtime_options_size())
                native_info_size = int(lib.colibri_v2_runtime_info_size())
                expected_options_size = ctypes.sizeof(_QwenRuntimeOptions)
                expected_info_size = ctypes.sizeof(_QwenRuntimeInfo)
                if (
                    version != NATIVE_ABI_VERSION
                    or native_options_size != expected_options_size
                    or native_info_size != expected_info_size
                ):
                    raise V2Error(
                        "native v2 ABI mismatch: "
                        f"library version={version}, options={native_options_size}, "
                        f"info={native_info_size}; Python expects "
                        f"version={NATIVE_ABI_VERSION}, options={expected_options_size}, "
                        f"info={expected_info_size}. Rebuild the native library."
                    )
                lib.colibri_v2_model_config.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_ModelConfig),
                ]
                lib.colibri_v2_model_config.restype = ctypes.c_int
                lib.colibri_v2_model_chat_template.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                lib.colibri_v2_model_chat_template.restype = ctypes.c_int
                lib.colibri_v2_model_attach_mtp.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                ]
                lib.colibri_v2_model_attach_mtp.restype = ctypes.c_int
                lib.colibri_v2_hf_quant_options.argtypes = [
                    ctypes.c_char_p,
                    ctypes.POINTER(_HfQuantOption),
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint32),
                ]
                lib.colibri_v2_hf_quant_options.restype = ctypes.c_int
                lib.colibri_v2_model_attention_window.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint32),
                ]
                lib.colibri_v2_model_attention_window.restype = ctypes.c_int
                lib.colibri_v2_model_compress_ratios.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_int32,
                ]
                lib.colibri_v2_model_compress_ratios.restype = ctypes.c_int
                lib.colibri_v2_model_target_layers.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_int32,
                ]
                lib.colibri_v2_model_target_layers.restype = ctypes.c_int
                lib.colibri_v2_dspark_encode.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                ]
                lib.colibri_v2_dspark_encode.restype = ctypes.c_int
                lib.colibri_v2_dspark_runtime_create.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
                ]
                lib.colibri_v2_dspark_runtime_create.restype = ctypes.c_int
                lib.colibri_v2_dspark_runtime_free.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_dspark_runtime_reset.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_dspark_runtime_reset.restype = ctypes.c_int
                lib.colibri_v2_dspark_inject.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                ]
                lib.colibri_v2_dspark_inject.restype = ctypes.c_int
                lib.colibri_v2_dspark_cached.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                ]
                lib.colibri_v2_dspark_cached.restype = ctypes.c_int
                lib.colibri_v2_dspark_heads.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.c_uint32, ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_uint32),
                ]
                lib.colibri_v2_dspark_heads.restype = ctypes.c_int
                lib.colibri_v2_dspark_attention.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                ]
                lib.colibri_v2_dspark_attention.restype = ctypes.c_int
                lib.colibri_v2_dspark_attention_stage.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float),
                    ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                ]
                lib.colibri_v2_dspark_attention_stage.restype = ctypes.c_int
                lib.colibri_v2_dspark_ffn_stage.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float),
                    ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                ]
                lib.colibri_v2_dspark_ffn_stage.restype = ctypes.c_int
                lib.colibri_v2_dspark_decode_hidden.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                ]
                lib.colibri_v2_dspark_decode_hidden.restype = ctypes.c_int
                lib.colibri_v2_pretokenize.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                lib.colibri_v2_pretokenize.restype = ctypes.c_int
                lib.colibri_v2_quant_supported.argtypes = [ctypes.c_uint32]
                lib.colibri_v2_quant_supported.restype = ctypes.c_int
                _float_p = ctypes.POINTER(ctypes.c_float)
                lib.colibri_v2_deepseek4_hyper_connection.argtypes = [
                    _float_p, _float_p, _float_p, _float_p,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_float, ctypes.c_float, _float_p,
                    _float_p, _float_p, _float_p, _float_p, _float_p, _float_p,
                ]
                lib.colibri_v2_deepseek4_hyper_connection.restype = ctypes.c_int
                lib.colibri_v2_matvec.argtypes = [
                    ctypes.c_void_p, ctypes.c_char_p, _float_p,
                    ctypes.c_int32, _float_p, ctypes.c_int32,
                ]
                lib.colibri_v2_matvec.restype = ctypes.c_int
                lib.colibri_v2_grouped_matvec.argtypes = [
                    ctypes.c_void_p, ctypes.c_char_p, _float_p,
                    ctypes.c_int32, _float_p, ctypes.c_int32, ctypes.c_int32,
                ]
                lib.colibri_v2_grouped_matvec.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_attention.argtypes = [
                    _float_p, _float_p, _float_p, ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_float, _float_p,
                ]
                lib.colibri_v2_deepseek4_attention.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_rope.argtypes = [
                    _float_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_int32,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_int32,
                ]
                lib.colibri_v2_deepseek4_rope.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_head.argtypes = [
                    _float_p, _float_p, _float_p, _float_p,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float,
                    _float_p, _float_p,
                ]
                lib.colibri_v2_deepseek4_head.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_runtime_create.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
                ]
                lib.colibri_v2_deepseek4_runtime_create.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_runtime_free.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_deepseek4_runtime_free.restype = None
                lib.colibri_v2_deepseek4_runtime_reset.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_deepseek4_runtime_reset.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_forward.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32, _float_p,
                ]
                lib.colibri_v2_deepseek4_forward.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_capture_layers.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                ]
                lib.colibri_v2_deepseek4_capture_layers.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_captured.argtypes = [
                    ctypes.c_void_p, _float_p, ctypes.c_uint64,
                ]
                lib.colibri_v2_deepseek4_captured.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_lm_head.argtypes = [
                    ctypes.c_void_p, _float_p, ctypes.c_uint32,
                    _float_p, ctypes.c_uint64,
                ]
                lib.colibri_v2_deepseek4_lm_head.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_prefill.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                ]
                lib.colibri_v2_deepseek4_prefill.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_forward_batch.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                    _float_p, ctypes.c_uint64,
                ]
                lib.colibri_v2_deepseek4_forward_batch.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_forward_batch_capture.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                    _float_p, ctypes.c_uint64, _float_p, ctypes.c_uint64,
                ]
                lib.colibri_v2_deepseek4_forward_batch_capture.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_snapshot.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
                lib.colibri_v2_deepseek4_snapshot.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_restore.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                lib.colibri_v2_deepseek4_restore.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_snapshot_free.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_deepseek4_runtime_info.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(_Deepseek4Info),
                ]
                lib.colibri_v2_deepseek4_runtime_info.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_half_round_trip.argtypes = [ctypes.c_float]
                lib.colibri_v2_deepseek4_half_round_trip.restype = ctypes.c_float
                lib.colibri_v2_deepseek4_gather_block.argtypes = [
                    _float_p, _float_p, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    _float_p, _float_p, ctypes.POINTER(ctypes.c_int32),
                ]
                lib.colibri_v2_deepseek4_gather_block.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_visible_keys.argtypes = [
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_int32, ctypes.POINTER(ctypes.c_uint8),
                    ctypes.POINTER(ctypes.c_int32),
                ]
                lib.colibri_v2_deepseek4_visible_keys.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_runtime_gpu_attach.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_deepseek4_runtime_gpu_attach.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_runtime_gpu.argtypes = [
                    ctypes.c_void_p, ctypes.c_int32,
                ]
                lib.colibri_v2_deepseek4_runtime_gpu.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_runtime_gpu_share.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                ]
                lib.colibri_v2_deepseek4_runtime_gpu_share.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_gpu_matvec_check.argtypes = [
                    ctypes.c_void_p, ctypes.c_char_p, _float_p, ctypes.c_int32,
                    ctypes.c_int32, _float_p, _float_p, ctypes.c_int32,
                    ctypes.c_int32, ctypes.POINTER(ctypes.c_double),
                ]
                lib.colibri_v2_deepseek4_gpu_matvec_check.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_indexer_key.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                    _float_p, ctypes.c_int32,
                ]
                lib.colibri_v2_deepseek4_indexer_key.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_indexer_scores.argtypes = [
                    _float_p, _float_p, _float_p, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_int32, _float_p,
                ]
                lib.colibri_v2_deepseek4_indexer_scores.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_top_k.argtypes = [
                    _float_p, ctypes.c_int32, ctypes.c_int32,
                    ctypes.POINTER(ctypes.c_uint8),
                ]
                lib.colibri_v2_deepseek4_top_k.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_compress.argtypes = [
                    _float_p, _float_p, ctypes.c_int32, ctypes.c_int32, _float_p,
                ]
                lib.colibri_v2_deepseek4_compress.restype = ctypes.c_int
                lib.colibri_v2_expert_matvec.argtypes = [
                    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, _float_p,
                    ctypes.c_int32, _float_p, ctypes.c_int32,
                ]
                lib.colibri_v2_expert_matvec.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_router.argtypes = [
                    _float_p, _float_p, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_float, ctypes.c_float, ctypes.c_int32,
                    ctypes.POINTER(ctypes.c_int32), _float_p,
                ]
                lib.colibri_v2_deepseek4_router.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_swiglu.argtypes = [
                    _float_p, _float_p, ctypes.c_int32, ctypes.c_float, _float_p,
                ]
                lib.colibri_v2_deepseek4_swiglu.restype = ctypes.c_int
                lib.colibri_v2_deepseek4_rms_norm.argtypes = [
                    _float_p, _float_p, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_float, _float_p,
                ]
                lib.colibri_v2_deepseek4_rms_norm.restype = ctypes.c_int
                lib.colibri_v2_qwen_validate.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_qwen_validate.restype = ctypes.c_int
                lib.colibri_v2_qwen_tensor_role.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.POINTER(_TensorInfo),
                ]
                lib.colibri_v2_qwen_tensor_role.restype = ctypes.c_int
                lib.colibri_v2_qwen_layer_tensor.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.c_char_p,
                    ctypes.POINTER(_TensorInfo),
                ]
                lib.colibri_v2_qwen_layer_tensor.restype = ctypes.c_int
                lib.colibri_v2_qwen_embedding.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_uint64,
                ]
                lib.colibri_v2_qwen_embedding.restype = ctypes.c_int
                lib.colibri_v2_qwen_lm_head.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                ]
                lib.colibri_v2_qwen_lm_head.restype = ctypes.c_int
                lib.colibri_v2_qwen_token_text.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.c_char_p,
                    ctypes.c_uint64,
                ]
                lib.colibri_v2_qwen_token_text.restype = ctypes.c_int
                lib.colibri_v2_token_id.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_uint32),
                ]
                lib.colibri_v2_token_id.restype = ctypes.c_int
                lib.colibri_v2_tokenize.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                lib.colibri_v2_tokenize.restype = ctypes.c_int
                lib.colibri_v2_tensor_read.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint64,
                    ctypes.c_void_p,
                    ctypes.c_uint64,
                ]
                lib.colibri_v2_tensor_read.restype = ctypes.c_int
                lib.colibri_v2_tensor_read_slice.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_void_p,
                    ctypes.c_uint64,
                ]
                lib.colibri_v2_tensor_read_slice.restype = ctypes.c_int
                lib.colibri_v2_tensor_view.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                lib.colibri_v2_tensor_view.restype = ctypes.c_int
                lib.colibri_v2_qwen_runtime_create.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_QwenRuntimeOptions),
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                lib.colibri_v2_qwen_runtime_create.restype = ctypes.c_int
                lib.colibri_v2_qwen_runtime_destroy.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_qwen_runtime_info.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_QwenRuntimeInfo),
                ]
                lib.colibri_v2_qwen_runtime_info.restype = ctypes.c_int
                lib.colibri_v2_qwen_runtime_reset.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_qwen_runtime_reset.restype = ctypes.c_int
                lib.colibri_v2_qwen_runtime_cancel.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_qwen_runtime_cancel.restype = ctypes.c_int
                lib.colibri_v2_qwen_runtime_dump_kv.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p,
                ]
                lib.colibri_v2_qwen_runtime_dump_kv.restype = ctypes.c_int
                lib.colibri_v2_qwen_runtime_prepare.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_qwen_runtime_prepare.restype = ctypes.c_int
                lib.colibri_v2_qwen_runtime_synchronize.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_qwen_runtime_synchronize.restype = ctypes.c_int
                lib.colibri_v2_qwen_runtime_decode.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint32),
                ]
                lib.colibri_v2_qwen_runtime_decode.restype = ctypes.c_int
                lib.colibri_v2_qwen_runtime_generate.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    _TokenCallback,
                    ctypes.c_void_p,
                ]
                lib.colibri_v2_qwen_runtime_generate.restype = ctypes.c_int
                lib.colibri_v2_qwen_task_submit.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                lib.colibri_v2_qwen_task_submit.restype = ctypes.c_int
                lib.colibri_v2_qwen_task_submit_sampling.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_uint64,
                    ctypes.c_float,
                    ctypes.c_uint32,
                    ctypes.c_float,
                    ctypes.c_uint64,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                lib.colibri_v2_qwen_task_submit_sampling.restype = ctypes.c_int
                lib.colibri_v2_qwen_task_submit_penalties.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_uint64,
                    ctypes.c_float,
                    ctypes.c_uint32,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_uint32,
                    ctypes.c_uint64,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                lib.colibri_v2_qwen_task_submit_penalties.restype = ctypes.c_int
                lib.colibri_v2_qwen_task_submit_grammar.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_uint64,
                    ctypes.c_float,
                    ctypes.c_uint32,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_uint32,
                    ctypes.c_uint64,
                    ctypes.c_uint32,
                    ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                lib.colibri_v2_qwen_task_submit_grammar.restype = ctypes.c_int
                lib.colibri_v2_qwen_engine_step.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_QwenTaskEvent),
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                lib.colibri_v2_qwen_engine_step.restype = ctypes.c_int
                lib.colibri_v2_qwen_task_cancel.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint64,
                ]
                lib.colibri_v2_qwen_task_cancel.restype = ctypes.c_int
                lib.colibri_v2_gpu_probe.argtypes = [
                    ctypes.c_int32,
                    ctypes.POINTER(_GpuInfo),
                ]
                lib.colibri_v2_gpu_probe.restype = ctypes.c_int
                lib.colibri_v2_gpu_available.restype = ctypes.c_int
                lib.colibri_backend_select.argtypes = [ctypes.c_int]
                lib.colibri_backend_select.restype = ctypes.c_int
                lib.colibri_backend_active.restype = ctypes.c_int
                lib.colibri_v2_gpu_init.argtypes = [ctypes.c_int32]
                lib.colibri_v2_gpu_init.restype = ctypes.c_int
                lib.colibri_v2_gpu_compile.argtypes = [
                    ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_char_p),
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_char_p,
                    ctypes.c_int32,
                ]
                lib.colibri_v2_gpu_compile.restype = ctypes.c_int
                lib.colibri_v2_memory_plan.argtypes = [
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.POINTER(_MemoryPlan),
                ]
                lib.colibri_v2_memory_plan.restype = ctypes.c_int
                lib.colibri_v2_gpu_rms_norm.argtypes = [
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_int32,
                    ctypes.c_float,
                    ctypes.c_int32,
                ]
                lib.colibri_v2_gpu_rms_norm.restype = ctypes.c_int
                lib.colibri_v2_gpu_q4_matvec.argtypes = [
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_int32,
                    ctypes.c_int32,
                ]
                lib.colibri_v2_gpu_q4_matvec.restype = ctypes.c_int
                lib.colibri_v2_gpu_dense_projection.argtypes = [
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_float,
                    ctypes.c_int32,
                ]
                lib.colibri_v2_gpu_dense_projection.restype = ctypes.c_int
                lib.colibri_v2_gpu_dense_residual.argtypes = [
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_float,
                    ctypes.c_int32,
                ]
                lib.colibri_v2_gpu_dense_residual.restype = ctypes.c_int
                lib.colibri_v2_gpu_attention.argtypes = [
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_float,
                ]
                lib.colibri_v2_gpu_attention.restype = ctypes.c_int
                lib.colibri_v2_gpu_decoder_attention_step.argtypes = [
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_float,
                    ctypes.c_int32,
                ]
                lib.colibri_v2_gpu_decoder_attention_step.restype = ctypes.c_int
                lib.colibri_v2_kv_cache_create.argtypes = [
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                lib.colibri_v2_kv_cache_create.restype = ctypes.c_int
                lib.colibri_v2_kv_cache_destroy.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_kv_cache_reset.argtypes = [ctypes.c_void_p]
                lib.colibri_v2_kv_cache_reset.restype = ctypes.c_int
                lib.colibri_v2_kv_cache_position.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_int32),
                ]
                lib.colibri_v2_kv_cache_position.restype = ctypes.c_int
                lib.colibri_v2_gpu_decoder_attention_cached.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_float,
                    ctypes.c_int32,
                ]
                lib.colibri_v2_gpu_decoder_attention_cached.restype = ctypes.c_int
            except AttributeError as error:
                # A library in _native that predates a C API change fails
                # here with a bare ctypes AttributeError naming the missing
                # symbol. Say what that actually means: the checked-in
                # binaries go stale whenever the native API grows.
                raise V2Error(
                    f"native v2 library at {path} is out of date ({error}); "
                    "rebuild it with python -m colibri_next.native_build"
                ) from error
            _cached_library = lib
            return lib
    raise V2Error(
        "native v2 library is not built; run python -m colibri_next.native_build"
    )


# int(*)(void* user, uint32_t processed, uint32_t total); non-zero aborts.
_BAILING_PROGRESS = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32
)


class BailingRuntime:
    """Host execution for BailingMoE3 (Ling 3.0) checkpoints.

    Straightforward f32 CPU inference: correct rather than fast, and the thing
    the eventual quantized and GPU paths are checked against. Needs a model
    opened with ``COLIBRI_HF_QUANT=F32``.

    The runtime owns a position: ``eval`` continues from wherever the last call
    left off, so a prompt is one call and each generated token is another.
    ``reset`` starts a new sequence.
    """

    def __init__(self, model: "V2Model", capacity: int = 4096):
        self._model = model
        self._lib = model._lib
        self._handle = ctypes.c_void_p()
        self._vocabulary = int(model.config["vocabulary_size"])
        self._lib.colibri_v2_bailing_create.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)
        ]
        self._lib.colibri_v2_bailing_eval.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float)
        ]
        self._check(
            self._lib.colibri_v2_bailing_create(
                model._handle, ctypes.c_uint32(capacity), ctypes.byref(self._handle)
            )
        )
        self._logits = (ctypes.c_float * self._vocabulary)()
        # Built on first unseeded sample and reused; see sample().
        self._generator: Any = None
        # Keeps the progress thunk alive for as long as the native side holds
        # its pointer; see set_progress.
        self._progress: Any = None
        self.uses_gpu = self._query_uses_gpu()

    def _query_uses_gpu(self) -> bool:
        """Whether creation actually got the device.

        Creation picks the GPU whenever there is one and falls back to the host
        silently, so the environment does not say which path this runtime is
        on -- only the runtime does.
        """
        try:
            entry = self._lib.colibri_v2_bailing_uses_gpu
        except AttributeError:
            # An older shared library; the host path is the safe thing to claim.
            return False
        entry.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        value = ctypes.c_int(0)
        self._check(entry(self._handle, ctypes.byref(value)))
        return bool(value.value)

    def _check(self, status: int) -> None:
        if status:
            message = self._lib.colibri_v2_last_error() or b"native v2 error"
            raise V2Error(message.decode(errors="replace"))

    def set_progress(self, callback: Any) -> None:
        """Watch a prompt as it is evaluated, and optionally stop it.

        `callback(processed, total)` runs once per internal tile -- every 128
        tokens -- and returning False abandons the call, which then raises. A
        prompt is otherwise one uninterruptible call that can run for minutes:
        nothing to report while it does, and no way to drop a request whose
        client has already gone. Pass None to clear.
        """
        entry = self._lib.colibri_v2_bailing_set_progress
        entry.argtypes = [ctypes.c_void_p, _BAILING_PROGRESS, ctypes.c_void_p]
        if callback is None:
            self._progress = None
            self._check(entry(self._handle, _BAILING_PROGRESS(0), None))
            return

        def trampoline(_user, processed, total) -> int:
            try:
                return 0 if callback(int(processed), int(total)) is not False else 1
            except Exception:
                # A raising watcher must not unwind through the native frame.
                return 1

        # Held on the runtime: the native side keeps the pointer, and a
        # collected thunk would be a call into freed memory.
        self._progress = _BAILING_PROGRESS(trampoline)
        self._check(entry(self._handle, self._progress, None))

    def _eval_into_buffer(self, tokens: Sequence[int]) -> None:
        if not tokens:
            raise ValueError("eval needs at least one token")
        buffer = (ctypes.c_uint32 * len(tokens))(*tokens)
        self._check(
            self._lib.colibri_v2_bailing_eval(
                self._handle, buffer, ctypes.c_uint32(len(tokens)), self._logits
            )
        )

    def eval(self, tokens: Sequence[int]) -> Any:
        """Advance by `tokens` and return the logits for the last one.

        Returns a numpy array when numpy is available and a list otherwise.
        The distinction matters: converting this vocabulary (157k floats) to a
        Python list costs ~7 ms, which was larger than the entire GPU forward
        pass it was reporting on.
        """
        self._eval_into_buffer(tokens)
        if _numpy is not None:
            # Copy: the buffer is reused by the next call, so handing back a
            # view would alias.
            return _numpy.frombuffer(self._logits, dtype=_numpy.float32).copy()
        return list(self._logits)

    def reset(self) -> None:
        self._check(self._lib.colibri_v2_bailing_reset(self._handle))

    def save_state(self) -> bytes:
        """The live sequence's cache, so it can be restored instead of re-run.

        Sized by the tokens the runtime actually holds, not by its capacity: a
        32-token side-call snapshots a few hundred kilobytes.
        """
        save = self._lib.colibri_v2_bailing_cache_save
        save.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        length = ctypes.c_uint64()
        self._check(save(self._handle, None, 0, ctypes.byref(length)))
        buffer = ctypes.create_string_buffer(length.value)
        self._check(
            save(self._handle, buffer, length.value, ctypes.byref(length))
        )
        return buffer.raw[: length.value]

    def load_state(self, snapshot: bytes) -> None:
        """Restore a sequence saved by ``save_state``."""
        load = self._lib.colibri_v2_bailing_cache_load
        load.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        self._check(load(self._handle, snapshot, len(snapshot)))

    def eval_into(self, tokens: Sequence[int]) -> None:
        """Advance the caches, leaving the logits in the internal buffer.

        The server path never wants the logits as Python objects -- it samples
        from them and discards them -- and materializing 157k floats per token
        costs more than the model does.
        """
        self._eval_into_buffer(tokens)

    def sample(self, config: Any = None) -> int:
        """Pick the next token from the logits currently in the buffer.

        Greedy when no temperature is set. Sampling runs on the ctypes buffer
        through numpy rather than a Python list, for the same reason as above.
        """
        temperature = float(getattr(config, "temperature", 0.0) or 0.0)
        top_k = int(getattr(config, "top_k", 0) or 0)
        top_p = float(getattr(config, "top_p", 0.0) or 0.0)
        if _numpy is None or temperature <= 0.0:
            if _numpy is not None:
                view = _numpy.frombuffer(self._logits, dtype=_numpy.float32)
                return int(view.argmax())
            return max(range(self._vocabulary), key=self._logits.__getitem__)

        # Everything below runs on the CANDIDATES, not on the vocabulary.
        #
        # This used to widen all 157k logits to float64, mask all but top_k of
        # them to -inf, and then argsort, cumsum and `choice` over the whole
        # array -- of which at most `top_k` entries could ever be drawn. It cost
        # ~5 ms per token against a ~7 ms token, so asking for any temperature
        # at all took decode from 146 to 88 tok/s while the GPU sat idle.
        #
        # Selecting first is exact, not an approximation: the discarded entries
        # have probability zero, so they cannot enter the top_p prefix and
        # cannot be drawn.
        view = _numpy.frombuffer(self._logits, dtype=_numpy.float32)
        if top_k > 0 and top_k < view.size:
            candidates = _numpy.argpartition(view, -top_k)[-top_k:]
        else:
            # No top_k, so top_p has to consider the whole distribution. Take a
            # generous prefix and widen it only if its mass has not reached
            # top_p -- which keeps this exact while almost always touching a few
            # hundred entries rather than the vocabulary. A top_p of 1.0 (or
            # none) needs everything, and says so immediately.
            candidates = _numpy.arange(view.size)
            if 0.0 < top_p < 1.0:
                full = _numpy.exp((view - view.max()) / temperature)
                mass = full.sum()
                width = 256
                while width < view.size:
                    prefix = _numpy.argpartition(view, -width)[-width:]
                    if full[prefix].sum() >= top_p * mass:
                        candidates = prefix
                        break
                    width *= 2

        values = view[candidates].astype(_numpy.float32) / temperature
        values -= values.max()
        probabilities = _numpy.exp(values)
        total = probabilities.sum()
        if not _numpy.isfinite(total) or total <= 0.0:
            return int(candidates[int(_numpy.argmax(values))])
        probabilities /= total

        order = _numpy.argsort(probabilities)[::-1]
        candidates = candidates[order]
        probabilities = probabilities[order]
        if 0.0 < top_p < 1.0:
            cumulative = _numpy.cumsum(probabilities)
            # Keep the smallest prefix whose mass reaches top_p, always at
            # least one token.
            keep = int(_numpy.searchsorted(cumulative, top_p) + 1)
            candidates = candidates[:keep]
            probabilities = probabilities[:keep]
            probabilities = probabilities / probabilities.sum()

        seed = getattr(config, "seed", None)
        if seed is not None:
            generator = _numpy.random.default_rng(seed)
        else:
            # One generator for the runtime rather than one per token:
            # constructing a Generator seeds it from the OS entropy pool, which
            # is not something to do 150 times a second.
            generator = self._generator
            if generator is None:
                generator = self._generator = _numpy.random.default_rng()
        return int(candidates[generator.choice(probabilities.size, p=probabilities)])

    def generate(
        self,
        tokens: Sequence[int],
        max_tokens: int = 32,
        stop: Sequence[int] | None = None,
    ) -> list[int]:
        """Greedy decode. Returns only the generated ids, not the prompt."""
        terminators = set(stop or ())
        produced: list[int] = []
        # The prompt goes through in one call; every step after is one token,
        # which is what makes the caches worth having.
        #
        # The argmax runs over the ctypes buffer directly rather than a copy:
        # a Python-level max() over 157k logits costs ~2 ms and the copy ~7 ms,
        # which together dwarfed the model.
        step = tokens
        for _ in range(max_tokens):
            self._eval_into_buffer(step)
            if _numpy is not None:
                view = _numpy.frombuffer(self._logits, dtype=_numpy.float32)
                best = int(view.argmax())
            else:
                best = max(range(self._vocabulary), key=self._logits.__getitem__)
            produced.append(best)
            if best in terminators:
                break
            step = [best]
        return produced

    def close(self) -> None:
        if getattr(self, "_handle", None) and self._handle:
            self._lib.colibri_v2_bailing_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "BailingRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class V2Model:
    def __init__(self, path: str | Path, *, mtp_model: str | Path | None = None):
        self.path = Path(path)
        self.mtp_model_path = Path(mtp_model) if mtp_model is not None else None
        self._lib = _library()
        self._handle = ctypes.c_void_p()
        self._tensor_catalog: dict[str, dict[str, object]] | None = None
        try:
            self._check(
                self._lib.colibri_v2_model_open(
                    str(self.path).encode(), ctypes.byref(self._handle)
                )
            )
            if self.mtp_model_path is not None:
                self._check(
                    self._lib.colibri_v2_model_attach_mtp(
                        self._handle, str(self.mtp_model_path).encode()
                    )
                )
        except Exception:
            self.close()
            raise
        self._architecture = str(self.info["architecture"])

    @staticmethod
    def hf_quant_options(path: str | Path) -> list[dict[str, object]]:
        """What a safetensors checkpoint could be loaded as, without loading it.

        One entry per quantization, each with the exact ``arena_bytes`` the load
        would produce and, when a cache for it is already on disk,
        ``cache_bytes`` and ``cache_path`` -- which is the difference between
        opening in a second and repacking the whole checkpoint.

        Raises V2Error for anything that is not a readable HF checkpoint this
        runtime understands, GGUF files included: there is nothing to choose.
        """
        lib = _library()
        buffer = (_HfQuantOption * 16)()
        count = ctypes.c_uint32()
        status = lib.colibri_v2_hf_quant_options(
            str(path).encode(), buffer, len(buffer), ctypes.byref(count)
        )
        if status:
            message = lib.colibri_v2_last_error() or b"native v2 error"
            raise V2Error(message.decode(errors="replace"))
        return [
            {
                "name": buffer[index].name.decode(),
                "arena_bytes": int(buffer[index].arena_bytes),
                "cache_bytes": int(buffer[index].cache_bytes),
                "cache_path": buffer[index].cache_path.decode(),
                "unavailable": buffer[index].unavailable.decode(),
            }
            for index in range(int(count.value))
        ]

    def _check(self, status: int) -> None:
        if status:
            message = self._lib.colibri_v2_last_error() or b"native v2 error"
            raise V2Error(message.decode(errors="replace"))

    def close(self) -> None:
        if self._handle:
            self._lib.colibri_v2_model_close(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "V2Model":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def info(self) -> dict[str, object]:
        value = _Info()
        self._check(self._lib.colibri_v2_model_info(self._handle, ctypes.byref(value)))
        return {
            "gguf_version": value.gguf_version,
            "tensor_count": value.tensor_count,
            "metadata_count": value.metadata_count,
            "file_size": value.file_size,
            "alignment": value.alignment,
            "architecture": value.architecture.decode(errors="replace"),
            "name": value.name.decode(errors="replace"),
            "format": value.format.decode(errors="replace"),
        }

    def tensors(self) -> Iterator[dict[str, object]]:
        if self._tensor_catalog is not None:
            yield from (dict(tensor) for tensor in self._tensor_catalog.values())
            return
        count = int(self.info["tensor_count"])
        for index in range(count):
            value = _TensorInfo()
            self._check(
                self._lib.colibri_v2_tensor_info(
                    self._handle, index, ctypes.byref(value)
                )
            )
            yield {
                "index": index,
                "name": value.name.decode(errors="replace"),
                "shape": tuple(value.shape[: value.dimensions]),
                "ggml_type": value.ggml_type,
                "offset": value.offset,
                "size": value.size,
            }

    def _ensure_tensor_catalog(self) -> dict[str, dict[str, object]]:
        if self._tensor_catalog is None:
            catalog = {}
            count = int(self.info["tensor_count"])
            for index in range(count):
                value = _TensorInfo()
                self._check(
                    self._lib.colibri_v2_tensor_info(
                        self._handle, index, ctypes.byref(value)
                    )
                )
                tensor = {
                    "index": index,
                    "name": value.name.decode(errors="replace"),
                    "shape": tuple(value.shape[: value.dimensions]),
                    "ggml_type": value.ggml_type,
                    "offset": value.offset,
                    "size": value.size,
                }
                catalog[tensor["name"]] = tensor
            self._tensor_catalog = catalog
        return self._tensor_catalog

    @property
    def config(self) -> dict[str, int | float | str | tuple[bool, ...] | tuple[int, ...]]:
        value = _ModelConfig()
        self._check(
            self._lib.colibri_v2_model_config(self._handle, ctypes.byref(value))
        )
        float_fields = {
            "rms_norm_epsilon",
            "rope_freq_base",
            "sinkhorn_epsilon",
            "compress_rope_freq_base",
            "rope_scaling_factor",
            "yarn_beta_fast",
            "yarn_beta_slow",
            "logit_scale",
            "final_logit_softcap",
            "expert_group_count",
            "expert_group_used",
        }
        config = {
            field: (
                getattr(value, field).decode(errors="replace")
                if field == "architecture"
                else float(getattr(value, field))
                if field in float_fields
                else int(getattr(value, field))
            )
            for field, _ in _ModelConfig._fields_
        }
        windows: list[int] = []
        for layer in range(value.layer_count):
            window = ctypes.c_uint32()
            self._check(
                self._lib.colibri_v2_model_attention_window(
                    self._handle, layer, ctypes.byref(window)
                )
            )
            windows.append(int(window.value))
        config["attention_windows"] = tuple(windows)
        config["sliding_window_pattern"] = tuple(bool(window) for window in windows)
        config["compress_ratios"] = self.compress_ratios
        config["target_layers"] = self.target_layers
        return config

    @property
    def target_layers(self) -> tuple[int, ...]:
        count = self._lib.colibri_v2_model_target_layers(self._handle, None, 0)
        if count < 0:
            self._check(count)
        if not count:
            return ()
        buffer = (ctypes.c_uint32 * count)()
        written = self._lib.colibri_v2_model_target_layers(
            self._handle, buffer, count
        )
        if written < 0:
            self._check(written)
        return tuple(int(buffer[index]) for index in range(written))

    def dspark_encode(self, features):
        import numpy as np

        values = np.ascontiguousarray(features, dtype=np.float32).reshape(-1)
        output = np.empty(int(self.config["hidden_size"]), dtype=np.float32)
        self._check(self._lib.colibri_v2_dspark_encode(
            self._handle,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), values.size,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size,
        ))
        return output

    def unsupported_quant_types(self) -> dict[int, list[str]]:
        """Tensors whose GGML type no backend can decode, keyed by type.

        Unsloth's dynamic quants mix a different type per tensor, so a
        checkpoint can be mostly decodable and still carry a handful of tensors
        in a format the runtime has no kernel for. Empty means the whole file
        is executable as far as its weight formats go.
        """
        offenders: dict[int, list[str]] = {}
        for tensor in self.tensors():
            kind = int(tensor["ggml_type"])
            if not self._lib.colibri_v2_quant_supported(kind):
                offenders.setdefault(kind, []).append(str(tensor["name"]))
        return offenders

    def pretokenize(self, text: str) -> tuple[str, ...]:
        """Split `text` the way this model's pre-tokenizer does, before BPE."""
        raw = text.encode("utf-8")
        count = ctypes.c_uint64()
        self._check(
            self._lib.colibri_v2_pretokenize(
                self._handle, raw, None, 0, ctypes.byref(count)
            )
        )
        offsets = (ctypes.c_uint64 * count.value)()
        self._check(
            self._lib.colibri_v2_pretokenize(
                self._handle, raw, offsets, count.value, ctypes.byref(count)
            )
        )
        return tuple(
            raw[offsets[index] : offsets[index + 1]].decode("utf-8", errors="replace")
            for index in range(count.value - 1)
        )

    @property
    def compress_ratios(self) -> tuple[int, ...]:
        """DeepSeek-V4 per-layer attention kinds; empty on other architectures.

        0 selects sliding-window attention, 4 compressed sparse attention over
        4:1 compressed tokens, and 128 heavily compressed attention.
        """
        count = self._lib.colibri_v2_model_compress_ratios(self._handle, None, 0)
        if count < 0:
            self._check(count)
        if count == 0:
            return ()
        buffer = (ctypes.c_uint32 * count)()
        written = self._lib.colibri_v2_model_compress_ratios(self._handle, buffer, count)
        if written < 0:
            self._check(written)
        return tuple(int(value) for value in buffer[:written])

    @property
    def chat_template(self) -> str | None:
        """Return the model-authored Jinja chat template embedded in GGUF."""
        length = ctypes.c_uint64()
        self._check(
            self._lib.colibri_v2_model_chat_template(
                self._handle, None, 0, ctypes.byref(length)
            )
        )
        if length.value == 0:
            return None
        output = ctypes.create_string_buffer(length.value + 1)
        self._check(
            self._lib.colibri_v2_model_chat_template(
                self._handle, output, len(output), ctypes.byref(length)
            )
        )
        return output.raw[: length.value].decode("utf-8", errors="strict")

    def tensor(self, name: str) -> dict[str, object]:
        if self._tensor_catalog is not None:
            try:
                return {
                    key: value
                    for key, value in self._tensor_catalog[name].items()
                    if key != "index"
                }
            except KeyError as error:
                raise V2Error(f"tensor not found: {name}") from error
        value = _TensorInfo()
        self._check(
            self._lib.colibri_v2_tensor_find(
                self._handle, name.encode(), ctypes.byref(value)
            )
        )
        return {
            "name": value.name.decode(errors="replace"),
            "shape": tuple(value.shape[: value.dimensions]),
            "ggml_type": value.ggml_type,
            "offset": value.offset,
            "size": value.size,
        }

    @staticmethod
    def _read_buffer(size: int) -> tuple[memoryview, object]:
        storage = bytearray(size)
        pointer = (ctypes.c_ubyte * size).from_buffer(storage)
        return memoryview(storage).toreadonly(), pointer

    def read_tensor(self, name: str) -> memoryview:
        try:
            info = self._ensure_tensor_catalog()[name]
        except KeyError as error:
            raise V2Error(f"tensor not found: {name}") from error
        output, pointer = self._read_buffer(int(info["size"]))
        self._check(
            self._lib.colibri_v2_tensor_read(
                self._handle, int(info["index"]), pointer, len(output)
            )
        )
        return output

    def read_tensor_slice(self, name: str, offset: int, size: int) -> memoryview:
        if offset < 0 or size < 0:
            raise ValueError("tensor slice offset and size must be non-negative")
        try:
            info = self._ensure_tensor_catalog()[name]
        except KeyError as error:
            raise V2Error(f"tensor not found: {name}") from error
        if offset + size > int(info["size"]):
            raise ValueError("tensor slice is outside the tensor")
        output, pointer = self._read_buffer(size)
        self._check(
            self._lib.colibri_v2_tensor_read_slice(
                self._handle, int(info["index"]), offset, pointer, size
            )
        )
        return output

    def view_tensor(self, name: str) -> memoryview:
        try:
            info = self._ensure_tensor_catalog()[name]
        except KeyError as error:
            raise V2Error(f"tensor not found: {name}") from error
        return self.view_tensor_slice(name, 0, int(info["size"]))

    def view_tensor_slice(self, name: str, offset: int, size: int) -> memoryview:
        if offset < 0 or size < 0:
            raise ValueError("tensor view offset and size must be non-negative")
        try:
            info = self._ensure_tensor_catalog()[name]
        except KeyError as error:
            raise V2Error(f"tensor not found: {name}") from error
        if offset + size > int(info["size"]):
            raise ValueError("tensor view is outside the tensor")
        if size == 0:
            return memoryview(b"")
        pointer = ctypes.c_void_p()
        self._check(
            self._lib.colibri_v2_tensor_view(
                self._handle,
                int(info["index"]),
                offset,
                size,
                ctypes.byref(pointer),
            )
        )
        address = pointer.value
        if address is None:
            raise V2Error("native tensor view returned a null address")
        array = (ctypes.c_ubyte * size).from_address(address)
        return memoryview(array).cast("B").toreadonly()

    def validate_qwen(self) -> None:
        self._check(self._lib.colibri_v2_qwen_validate(self._handle))

    def qwen_tensor(self, role: str) -> dict[str, object]:
        value = _TensorInfo()
        self._check(
            self._lib.colibri_v2_qwen_tensor_role(
                self._handle, role.encode(), ctypes.byref(value)
            )
        )
        return {
            "name": value.name.decode(errors="replace"),
            "shape": tuple(value.shape[: value.dimensions]),
            "ggml_type": value.ggml_type,
            "offset": value.offset,
            "size": value.size,
        }

    def qwen_layer_tensor(self, layer: int, role: str) -> dict[str, object]:
        value = _TensorInfo()
        self._check(
            self._lib.colibri_v2_qwen_layer_tensor(
                self._handle, layer, role.encode(), ctypes.byref(value)
            )
        )
        return {
            "name": value.name.decode(errors="replace"),
            "shape": tuple(value.shape[: value.dimensions]),
            "ggml_type": value.ggml_type,
            "offset": value.offset,
            "size": value.size,
        }

    def qwen_embedding(self, token: int, elements: int) -> list[float]:
        output = (ctypes.c_float * elements)()
        self._check(
            self._lib.colibri_v2_qwen_embedding(self._handle, token, output, elements)
        )
        return list(output)

    def qwen_lm_head(self, hidden: list[float], vocabulary: int) -> list[float]:
        values = (ctypes.c_float * len(hidden))(*hidden)
        logits = (ctypes.c_float * vocabulary)()
        self._check(
            self._lib.colibri_v2_qwen_lm_head(
                self._handle, values, logits, vocabulary, len(hidden)
            )
        )
        return list(logits)

    def token_text(self, token: int) -> str:
        # Some GGUF vocabularies store very long strings in a single token
        # (whole words/sentences). Grow the decode buffer until it fits instead
        # of failing with "token output buffer is too small".
        capacity = 4096
        while True:
            output = ctypes.create_string_buffer(capacity)
            code = self._lib.colibri_v2_qwen_token_text(
                self._handle, int(token), output, len(output)
            )
            if code == 0:
                return output.value.decode("utf-8", errors="replace")
            if capacity >= 1 << 20:
                self._check(code)
            capacity *= 4

    def tokenize(self, text: str, capacity: int = 4096) -> list[int]:
        # Qwen uses both pipe-delimited control tokens (``<|im_start|>``)
        # and ordinary angle-delimited control tokens (``<think>`` and
        # ``</think>``).  Only treat an angle-delimited candidate as special
        # when it exists verbatim in the GGUF vocabulary.
        #
        # A candidate that is *not* in the vocabulary has to be left inside the
        # surrounding run rather than tokenized on its own.  ``<[^<>]+>`` is a
        # broad pattern -- it happily spans ordinary prose containing a ``<``
        # and a later ``>`` -- and splitting there invents a piece boundary the
        # pre-tokenizer never would, which silently blocks merges across it.
        pieces: list[int] = []
        plain_start = 0
        for match in _SPECIAL_TOKEN_PATTERN.finditer(text):
            if match.start() < plain_start:
                continue  # overlapped by a candidate already consumed
            try:
                token = self.token_id(match.group(0))
            except V2Error:
                continue
            if match.start() > plain_start:
                pieces.extend(
                    self._tokenize_plain(text[plain_start : match.start()], capacity)
                )
            pieces.append(token)
            plain_start = match.end()
        if plain_start < len(text):
            pieces.extend(self._tokenize_plain(text[plain_start:], capacity))
        return pieces

    def _tokenize_plain(self, text: str, capacity: int) -> list[int]:
        encoded = text.encode("utf-8")
        while True:
            tokens = (ctypes.c_uint32 * capacity)()
            count = ctypes.c_uint64()
            status = self._lib.colibri_v2_tokenize(
                self._handle,
                encoded,
                tokens,
                capacity,
                ctypes.byref(count),
            )
            if status == 0:
                return [int(tokens[index]) for index in range(count.value)]
            if capacity >= 1 << 20:
                self._check(status)
            capacity *= 4

    def token_id(self, text: str) -> int:
        token = ctypes.c_uint32()
        self._check(
            self._lib.colibri_v2_token_id(
                self._handle, text.encode("utf-8"), ctypes.byref(token)
            )
        )
        return int(token.value)

    def decode_token_bytes(self, tokens: list[int]) -> bytes:
        """Raw UTF-8 bytes for a token sequence (byte-level BPE unmapped).

        A single BPE token frequently holds only part of a multi-byte UTF-8
        character, so per-token *string* decoding replaces the halves with
        U+FFFD. Streaming callers must accumulate these bytes through an
        incremental UTF-8 decoder instead of decoding token-by-token (that bug
        corrupted emoji/box-drawing characters in generated files).
        """
        encoded = bytearray()
        for token in tokens:
            piece = self.token_text(token)
            # Gemma 4's generated vocabulary pieces can carry SentencePiece's
            # visible word-boundary marker. Prompt tokenization may represent
            # the same boundary as a standalone space token, so round-trip
            # tests alone do not expose this; generated text must explicitly
            # map every marker back to an ASCII space.
            if getattr(self, "_architecture", "") == "gemma4":
                piece = piece.replace("▁", " ")
            if piece.startswith("<|") and piece.endswith("|>"):
                encoded.extend(piece.encode("utf-8"))
                continue
            for character in piece:
                codepoint = ord(character)
                if codepoint not in _GGUF_BYTE_DECODER:
                    encoded.extend(character.encode("utf-8"))
                else:
                    encoded.append(_GGUF_BYTE_DECODER[codepoint])
        return bytes(encoded)

    def decode_tokens(self, tokens: list[int]) -> str:
        return self.decode_token_bytes(tokens).decode("utf-8", errors="replace")

    def native_qwen_runtime(
        self,
        *,
        device: int = 0,
        context_limit: int = 0,
        gpu_cache_bytes: int = 0,
        moe_device: str | None = None,
        expert_mode: str | None = None,
        mtp_drafts: int = 0,
        expert_top_k: int = 0,
        expert_top_p: float = 0.0,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        prefill_checkpoint_interval: int = 256,
        prefill_checkpoint_slots: int = 4,
        parallel_sequences: int = 1,
        prompt_cache_mib: int = 0,
        swa_full: bool = False,
        prefill_cache_seed: int | str | None = None,
        expert_paging: str = "auto",
        cpu_prefetch_mib: int = 0,
        cpu_prefetch_auto: bool = False,
        next_layer_prefetch: int = 0,
        cpu_threads: int = 0,
        hybrid_prefill: str = "split",
        expert_residency: str | None = None,
        dense_requant: str = "auto",
        prefill_expert_stream_mib: int = -1,
    ) -> "V2QwenRuntime":
        return V2QwenRuntime(
            self,
            device=device,
            context_limit=context_limit,
            gpu_cache_bytes=gpu_cache_bytes,
            moe_device=moe_device,
            expert_mode=expert_mode,
            mtp_drafts=mtp_drafts,
            expert_top_k=expert_top_k,
            expert_top_p=expert_top_p,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            prefill_checkpoint_interval=prefill_checkpoint_interval,
            prefill_checkpoint_slots=prefill_checkpoint_slots,
            parallel_sequences=parallel_sequences,
            prompt_cache_mib=prompt_cache_mib,
            swa_full=swa_full,
            prefill_cache_seed=prefill_cache_seed,
            expert_paging=expert_paging,
            cpu_prefetch_mib=cpu_prefetch_mib,
            cpu_prefetch_auto=cpu_prefetch_auto,
            next_layer_prefetch=next_layer_prefetch,
            cpu_threads=cpu_threads,
            hybrid_prefill=hybrid_prefill,
            expert_residency=expert_residency,
            dense_requant=dense_requant,
            prefill_expert_stream_mib=prefill_expert_stream_mib,
        )

    def native_runtime(self, **options: Any) -> "V2QwenRuntime":
        """Create the native runtime for any supported GGUF architecture.

        ``expert_mode="auto"`` is the default for routed models.
        ``native_qwen_runtime`` remains as a compatibility alias.
        """
        return self.native_qwen_runtime(**options)

    @staticmethod
    def select_backend(backend: str = "auto") -> str:
        """Choose where the runtime executes. Returns the backend selected.

        Must be called before a runtime is prepared: allocations belong to
        whichever backend was active when they were made.

        "auto" prefers CUDA and falls back to the CPU backend when no driver
        is present, which is the behaviour a caller almost always wants -- the
        alternative is a hard failure on a machine that could have run the
        model slowly.
        """
        lib = _library()
        if backend not in _BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected one of "
                + ", ".join(sorted(_BACKENDS))
            )
        if backend == "auto":
            # Ask the CUDA backend whether a driver is present before settling.
            lib.colibri_backend_select(_BACKENDS["cuda"])
            backend = "cuda" if lib.colibri_v2_gpu_available() == 1 else "cpu"
        if lib.colibri_backend_select(_BACKENDS[backend]) != 0:
            raise V2Error(f"native backend {backend!r} is unavailable")
        return backend

    @staticmethod
    def active_backend() -> str:
        lib = _library()
        value = int(lib.colibri_backend_active())
        for name, code in _BACKENDS.items():
            if name != "auto" and code == value:
                return name
        return "unknown"

    @staticmethod
    def gpu_info(device: int = 0) -> dict[str, int]:
        lib = _library()
        value = _GpuInfo()
        status = lib.colibri_v2_gpu_probe(device, ctypes.byref(value))
        if status:
            message = lib.colibri_v2_last_error() or b"native v2 GPU probe failed"
            raise V2Error(message.decode(errors="replace"))
        return {field: int(getattr(value, field)) for field, _ in _GpuInfo._fields_}

    @staticmethod
    def memory_plan(
        budget: int,
        static_weights: int,
        kv_state: int,
        workspace: int,
        active_experts: int,
        staging: int,
    ) -> dict[str, int]:
        lib = _library()
        value = _MemoryPlan()
        status = lib.colibri_v2_memory_plan(
            budget,
            static_weights,
            kv_state,
            workspace,
            active_experts,
            staging,
            ctypes.byref(value),
        )
        if status:
            message = lib.colibri_v2_last_error() or b"native v2 memory planner failed"
            raise V2Error(message.decode(errors="replace"))
        return {field: int(getattr(value, field)) for field, _ in _MemoryPlan._fields_}

    @staticmethod
    def gpu_rms_norm(
        input_ptr: int,
        weights_ptr: int,
        output_ptr: int,
        size: int,
        epsilon: float,
        one_centered: bool = True,
    ) -> None:
        lib = _library()
        status = lib.colibri_v2_gpu_rms_norm(
            input_ptr, weights_ptr, output_ptr, size, epsilon, int(one_centered)
        )
        if status:
            message = lib.colibri_v2_last_error() or b"native v2 RMSNorm failed"
            raise V2Error(message.decode(errors="replace"))

    @staticmethod
    def gpu_prepare(
        kernel_source: str, device: int = 0, options: list[str] | None = None
    ) -> str:
        lib = _library()
        if lib.colibri_v2_gpu_available() != 1:
            raise V2Error("CUDA driver or NVRTC is unavailable")
        status = lib.colibri_v2_gpu_init(device)
        if status:
            raise V2Error(f"native v2 CUDA initialization failed with status {status}")
        if options is None:
            include_dirs: list[Path] = []
            for variable in ("CUDA_PATH", "CUDA_HOME"):
                value = os.environ.get(variable)
                if value:
                    include_dirs.append(Path(value) / "include")
                    include_dirs.append(
                        Path(value) / "targets" / "x86_64-linux" / "include"
                    )
            try:
                import cupy

                cuda_path = cupy.cuda.get_cuda_path()
                if cuda_path:
                    include_dirs.append(Path(cuda_path) / "include")
                    include_dirs.append(
                        Path(cuda_path) / "targets" / "x86_64-linux" / "include"
                    )
            except (ImportError, RuntimeError):
                pass
            include_dirs.extend(
                (
                    Path("/usr/local/cuda/include"),
                    Path("/opt/cuda/include"),
                    Path("/opt/cuda/targets/x86_64-linux/include"),
                )
            )
            options = [
                f"-I{directory}"
                for directory in dict.fromkeys(
                    str(path)
                    for path in include_dirs
                    if (path / "cuda_fp16.h").is_file()
                )
            ]
        encoded = [
            (option.encode() if isinstance(option, str) else option)
            for option in (options or [])
        ]
        option_array = (ctypes.c_char_p * len(encoded))(*encoded)
        log = ctypes.create_string_buffer(16384)
        status = lib.colibri_v2_gpu_compile(
            kernel_source.encode(), option_array, len(encoded), device, log, len(log)
        )
        if status:
            detail = log.value.decode(errors="replace").strip()
            raise V2Error(f"native v2 CUDA compilation failed ({status}): {detail}")
        return log.value.decode(errors="replace")

    @staticmethod
    def gpu_q4_matvec(
        packed_ptr: int,
        scales_ptr: int,
        input_ptr: int,
        output_ptr: int,
        rows: int,
        columns: int,
        stream_ptr: int = 0,
    ) -> None:
        lib = _library()
        status = lib.colibri_v2_gpu_q4_matvec(
            packed_ptr, scales_ptr, input_ptr, output_ptr, stream_ptr, rows, columns
        )
        if status:
            message = lib.colibri_v2_last_error() or b"native v2 Q4 matvec failed"
            raise V2Error(message.decode(errors="replace"))

    @staticmethod
    def gpu_dense_projection(
        input_ptr: int,
        norm_weights_ptr: int,
        normalized_ptr: int,
        packed_ptr: int,
        scales_ptr: int,
        projection_ptr: int,
        rows: int,
        columns: int,
        epsilon: float,
        one_centered: bool = True,
    ) -> None:
        lib = _library()
        status = lib.colibri_v2_gpu_dense_projection(
            input_ptr,
            norm_weights_ptr,
            normalized_ptr,
            packed_ptr,
            scales_ptr,
            projection_ptr,
            rows,
            columns,
            epsilon,
            int(one_centered),
        )
        if status:
            message = (
                lib.colibri_v2_last_error() or b"native v2 dense projection failed"
            )
            raise V2Error(message.decode(errors="replace"))

    @staticmethod
    def gpu_dense_residual(
        input_ptr: int,
        norm_weights_ptr: int,
        normalized_ptr: int,
        packed_ptr: int,
        scales_ptr: int,
        output_ptr: int,
        rows: int,
        columns: int,
        epsilon: float,
        one_centered: bool = True,
    ) -> None:
        lib = _library()
        status = lib.colibri_v2_gpu_dense_residual(
            input_ptr,
            norm_weights_ptr,
            normalized_ptr,
            packed_ptr,
            scales_ptr,
            output_ptr,
            rows,
            columns,
            epsilon,
            int(one_centered),
        )
        if status:
            message = lib.colibri_v2_last_error() or b"native v2 dense residual failed"
            raise V2Error(message.decode(errors="replace"))

    @staticmethod
    def gpu_attention(
        query_ptr: int,
        keys_ptr: int,
        values_ptr: int,
        output_ptr: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        tokens: int,
        scale: float,
    ) -> None:
        lib = _library()
        status = lib.colibri_v2_gpu_attention(
            query_ptr,
            keys_ptr,
            values_ptr,
            output_ptr,
            heads,
            kv_heads,
            head_dim,
            tokens,
            scale,
        )
        if status:
            message = lib.colibri_v2_last_error() or b"native v2 attention failed"
            raise V2Error(message.decode(errors="replace"))

    @staticmethod
    def gpu_decoder_attention_step(
        input_ptr: int,
        norm_weights_ptr: int,
        normalized_ptr: int,
        qkv_packed_ptr: int,
        qkv_scales_ptr: int,
        qkv_ptr: int,
        cache_keys_ptr: int,
        cache_values_ptr: int,
        attention_output_ptr: int,
        out_packed_ptr: int,
        out_scales_ptr: int,
        output_ptr: int,
        hidden_size: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        position: int,
        capacity: int,
        epsilon: float,
        one_centered: bool = True,
    ) -> None:
        lib = _library()
        status = lib.colibri_v2_gpu_decoder_attention_step(
            input_ptr,
            norm_weights_ptr,
            normalized_ptr,
            qkv_packed_ptr,
            qkv_scales_ptr,
            qkv_ptr,
            cache_keys_ptr,
            cache_values_ptr,
            attention_output_ptr,
            out_packed_ptr,
            out_scales_ptr,
            output_ptr,
            hidden_size,
            heads,
            kv_heads,
            head_dim,
            position,
            capacity,
            epsilon,
            int(one_centered),
        )
        if status:
            message = (
                lib.colibri_v2_last_error()
                or b"native v2 decoder attention step failed"
            )
            raise V2Error(message.decode(errors="replace"))


def _constraint_specification(
    tools: Sequence[Mapping[str, Any]] | None,
    response_format: Mapping[str, Any] | None,
) -> bytes | None:
    """The sampler-constraint spec as the native library reads it.

    The bare tool array is the historical wire form and stays whenever it
    suffices, so an older native library keeps parsing what it always did;
    the object form exists only to carry the response constraint beside it.
    """
    if response_format:
        document: Any = {
            "tools": list(tools or []),
            "response_format": dict(response_format),
        }
    elif tools:
        document = list(tools)
    else:
        return None
    return json.dumps(document).encode("utf-8")


class V2QwenRuntime:
    """Owns a native Qwen or Gemma 4 execution plan and its CUDA state.

    The model must outlive this object.  ``decode_ready`` is deliberately
    exposed so callers cannot confuse the catalog/planning milestone with the
    upcoming one-call native token executor.
    """

    def __init__(
        self,
        model: V2Model,
        *,
        device: int = 0,
        context_limit: int = 0,
        gpu_cache_bytes: int = 0,
        moe_device: str | None = None,
        expert_mode: str | None = None,
        mtp_drafts: int = 0,
        expert_top_k: int = 0,
        expert_top_p: float = 0.0,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        prefill_checkpoint_interval: int = 256,
        prefill_checkpoint_slots: int = 4,
        parallel_sequences: int = 1,
        prompt_cache_mib: int = 0,
        swa_full: bool = False,
        prefill_cache_seed: int | str | None = None,
        expert_paging: str = "auto",
        cpu_prefetch_mib: int = 0,
        cpu_prefetch_auto: bool = False,
        next_layer_prefetch: int = 0,
        cpu_threads: int = 0,
        hybrid_prefill: str = "split",
        expert_residency: str | None = None,
        dense_requant: str = "auto",
        prefill_expert_stream_mib: int = -1,
    ):
        # gpu_cache_bytes is the total CUDA budget (base allocations + expert
        # cache). 0 = auto-fit to free VRAM; any positive value is an exact
        # manual budget.
        if device < 0 or context_limit < 0 or gpu_cache_bytes < 0:
            raise ValueError("native Qwen runtime options must be non-negative")
        # Mid-prefill recurrent-state checkpoints: interval=0 disables them (only
        # the end-of-prompt snapshot is kept). slots is the total snapshot pool.
        if prefill_checkpoint_interval < 0 or prefill_checkpoint_slots < 0:
            raise ValueError("prefill checkpoint options must be non-negative")
        # Independent decode slots (llama.cpp --parallel): each is a full KV +
        # DeltaNet state arena, so side-requests don't evict the main conversation.
        if parallel_sequences < 1:
            raise ValueError("parallel_sequences must be >= 1")
        # Host RAM budget (MiB) for spilling evicted slot state so recycled
        # conversations restore from RAM instead of reprefilling; 0 disables.
        if prompt_cache_mib < 0:
            raise ValueError("prompt_cache_mib must be non-negative")
        if expert_mode is not None and moe_device is not None:
            if _resolve_expert_mode(expert_mode)[0] != _resolve_expert_mode(
                moe_device
            )[0]:
                raise ValueError("expert_mode and moe_device select different policies")
        requested_expert_mode = expert_mode or moe_device or "auto"
        resolved_expert_mode, native_expert_mode, strict_resident = (
            _resolve_expert_mode(requested_expert_mode)
        )
        legacy_policy = resolved_expert_mode in {
            "legacy-hybrid",
            "legacy-paging",
        }
        if prefill_cache_seed is None:
            prefill_cache_seed = "off" if legacy_policy else "auto"
        if expert_residency is None:
            # Decode must be allowed to replace seeded experts. Freezing residency
            # for the whole request left the device cache at whatever the prefill
            # seed guessed -- ~99% of decode expert lookups missed and fell to the
            # CPU. See the expert cache notes in the native runtime.
            expert_residency = "mutable"
        prefill_cache_seed_count, prefill_cache_seed_auto = (
            _normalize_prefill_cache_seed(prefill_cache_seed)
        )
        if expert_paging not in {"auto", "staged", "direct"}:
            raise ValueError("expert_paging must be 'auto', 'staged', or 'direct'")
        if cpu_prefetch_mib < 0:
            raise ValueError("cpu_prefetch_mib must be non-negative")
        if cpu_prefetch_mib and cpu_prefetch_auto:
            raise ValueError("cpu_prefetch_mib and cpu_prefetch_auto are mutually exclusive")
        if not 0 <= next_layer_prefetch <= 64:
            raise ValueError("next_layer_prefetch must be between 0 and 64")
        if cpu_threads < 0:
            raise ValueError('cpu_threads must be non-negative (0 = automatic)')
        if hybrid_prefill not in {"split", "cpu"}:
            raise ValueError("hybrid_prefill must be 'split' or 'cpu'")
        if expert_residency not in {"mutable", "immutable"}:
            raise ValueError("expert_residency must be 'mutable' or 'immutable'")
        if dense_requant not in {"auto", "q8", "off"}:
            raise ValueError("dense_requant must be 'auto', 'q8', or 'off'")
        if prefill_expert_stream_mib < -1 or prefill_expert_stream_mib > 65536:
            raise ValueError(
                "prefill_expert_stream_mib must be -1 (auto), 0 (off), or a "
                "budget in MiB")
        if next_layer_prefetch and mtp_drafts:
            raise ValueError("next_layer_prefetch does not support MTP yet")
        effective_hybrid_prefill = (
            "cpu" if resolved_expert_mode == "auto" else hybrid_prefill
        )
        effective_expert_residency = expert_residency
        if mtp_drafts < 0 or mtp_drafts > 8:
            raise ValueError("mtp_drafts must be between 0 and 8")
        if expert_top_k < 0:
            raise ValueError("expert_top_k must be non-negative (0 = model default)")
        if not 0.0 <= expert_top_p <= 1.0:
            raise ValueError("expert_top_p must be within [0, 1] (0 = disabled)")
        # KV cache precision per llama.cpp's -ctk/-ctv (Phase 1: f32, f16).
        # turbo3/turbo4 are TurboQuant (arXiv:2504.19874): a fixed rotation plus
        # a Lloyd-Max codebook, at 3.5 and 4.5 bits per value. They need a
        # head_dim that is a power of two between 32 and 512.
        cache_types = CACHE_TYPE_CODES
        if cache_type_k not in cache_types or cache_type_v not in cache_types:
            raise ValueError(
                "cache_type_k/v must be one of " + ", ".join(sorted(cache_types))
            )
        self.model, self._lib = model, model._lib
        self.parallel_sequences = parallel_sequences
        self.prompt_cache_mib = prompt_cache_mib
        self.requested_expert_mode = requested_expert_mode
        self.expert_mode = resolved_expert_mode
        self._handle = ctypes.c_void_p()
        options = _QwenRuntimeOptions(
            device,
            native_expert_mode,
            mtp_drafts,
            expert_top_k,
            context_limit,
            gpu_cache_bytes,
            expert_top_p,
            cache_types[cache_type_k],
            cache_types[cache_type_v],
            prefill_checkpoint_interval,
            prefill_checkpoint_slots,
            parallel_sequences,
            prompt_cache_mib,
            int(swa_full),
            prefill_cache_seed_count,
            {"auto": 0, "staged": 1, "direct": 2}[expert_paging],
            cpu_prefetch_mib,
            int(cpu_prefetch_auto),
            next_layer_prefetch,
            cpu_threads,
            int(effective_hybrid_prefill == "cpu"),
            int(effective_expert_residency == "immutable"),
            int(prefill_cache_seed_auto),
            int(strict_resident),
            {"auto": 0, "q8": 1, "off": 2}[dense_requant],
            prefill_expert_stream_mib,
        )
        model._check(
            self._lib.colibri_v2_qwen_runtime_create(
                model._handle, ctypes.byref(options), ctypes.byref(self._handle)
            )
        )

    def close(self) -> None:
        if self._handle:
            self._lib.colibri_v2_qwen_runtime_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "V2QwenRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def info(self) -> dict[str, object]:
        value = _QwenRuntimeInfo()
        self.model._check(
            self._lib.colibri_v2_qwen_runtime_info(self._handle, ctypes.byref(value))
        )
        boolean_fields = {"cuda_ready", "decode_ready"}
        result: dict[str, object] = {
            field: bool(getattr(value, field))
            if field in boolean_fields
            else int(getattr(value, field))
            for field, _ in _QwenRuntimeInfo._fields_
        }
        native_mode = int(result["moe_device"])
        fallback_reason = ""
        resolved_mode = self.expert_mode
        if self.expert_mode in {"auto", "legacy-hybrid"} and native_mode == 1:
            resolved_mode = "cpu"
            fallback_reason = (
                "automatic GPU expert working set did not fit; using CPU experts"
            )
        result.update(
            {
                "requested_expert_mode": self.requested_expert_mode,
                "expert_mode": resolved_mode,
                "expert_mode_alias": (
                    self.requested_expert_mode
                    if self.requested_expert_mode != self.expert_mode
                    else ""
                ),
                "expert_fallback_reason": fallback_reason,
                "cache_type_k": CACHE_TYPE_NAMES.get(
                    int(result["resolved_cache_type_k"]), "?"
                ),
                "cache_type_v": CACHE_TYPE_NAMES.get(
                    int(result["resolved_cache_type_v"]), "?"
                ),
            }
        )
        return result

    def reset(self) -> None:
        self.model._check(self._lib.colibri_v2_qwen_runtime_reset(self._handle))

    def cancel(self) -> None:
        self.model._check(self._lib.colibri_v2_qwen_runtime_cancel(self._handle))

    def dump_kv(self, layer: int, path: str) -> None:
        """Write one attention layer's live KV window for bench_turboquant.

        Decodes whatever precision the cache runs at to f32, so the dump shows
        the distribution the codec would really see. Requires at least one
        decoded token, and the layer must be an attention layer rather than a
        Gated DeltaNet one.
        """
        self.model._check(
            self._lib.colibri_v2_qwen_runtime_dump_kv(
                self._handle, ctypes.c_uint32(layer), str(path).encode("utf-8")
            )
        )

    def prepare(self) -> None:
        """Allocate native CUDA arenas and upload persistent model weights."""
        self.model._check(self._lib.colibri_v2_qwen_runtime_prepare(self._handle))

    def synchronize(self) -> None:
        self.model._check(self._lib.colibri_v2_qwen_runtime_synchronize(self._handle))

    def decode(self, input_token: int) -> int:
        """Run one complete greedy token step inside native C++/CUDA."""
        output = ctypes.c_uint32()
        self.model._check(
            self._lib.colibri_v2_qwen_runtime_decode(
                self._handle, input_token, ctypes.byref(output)
            )
        )
        return int(output.value)

    def generate(self, prompt_tokens: list[int], max_tokens: int, callback) -> None:
        """Run prompt ingestion and greedy decode inside the native token loop.

        Runs to `max_tokens` and does NOT stop at end-of-turn: the loop has no
        stop set, so a caller that wants one returns False from `callback` --
        which is how a fixed-length benchmark and a real generation can share
        it. Left to run past the model's own end token, generation continues
        into invented turns (`<|im_end|><|im_start|>Human...`), so anything
        showing output to a person wants the callback to stop.

        The cooperative engine (`task_submit`) takes a stop-token list instead
        and is what the server uses.
        """
        if not prompt_tokens:
            raise ValueError("prompt_tokens must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        values = (ctypes.c_uint32 * len(prompt_tokens))(*prompt_tokens)
        callback_error: list[BaseException] = []

        def receive(token: int, _user: int) -> int:
            try:
                return 0 if callback(int(token)) is not False else 1
            except BaseException as error:
                callback_error.append(error)
                return 1

        callback_ref = _TokenCallback(receive)
        self.model._check(
            self._lib.colibri_v2_qwen_runtime_generate(
                self._handle,
                values,
                len(prompt_tokens),
                max_tokens,
                callback_ref,
                None,
            )
        )
        if callback_error:
            raise callback_error[0]

    def task_submit(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        stop_tokens: tuple[int, ...] | list[int] = (),
        *,
        temperature: float = 0.0,
        top_k: int = 20,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        penalty_window: int = 64,
        seed: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> int:
        """Queue a request on the cooperative engine; returns its task id.

        `tools` constrains the sampler while a tool call is open, so a required
        parameter cannot be skipped: each entry is `{"name": str, "parameters":
        [{"name": str, "required": bool}]}`. `response_format` constrains the
        visible answer to one JSON value: `{"shape": "object" | "array" |
        "value", "thinking_open": bool}`. Omitting both samples freely.

        The wire spec stays the bare tool array unless a response format is
        present, so an older native library keeps parsing what it always did.
        """
        if not prompt_tokens:
            raise ValueError("prompt_tokens must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        values = (ctypes.c_uint32 * len(prompt_tokens))(*prompt_tokens)
        stops = (ctypes.c_uint32 * len(stop_tokens))(*stop_tokens) if stop_tokens else None
        task_id = ctypes.c_uint64()
        specification = _constraint_specification(tools, response_format)
        self.model._check(
            self._lib.colibri_v2_qwen_task_submit_grammar(
                self._handle,
                values,
                len(prompt_tokens),
                max_tokens,
                stops,
                len(stop_tokens),
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
                penalty_window,
                0 if seed is None else seed & ((1 << 64) - 1),
                int(seed is not None),
                specification,
                ctypes.byref(task_id),
            )
        )
        return int(task_id.value)

    def engine_step(self, capacity: int = 256) -> list[tuple[int, int, int]]:
        """Run one engine scheduling cycle; returns (task_id, token, kind) events."""
        events = (_QwenTaskEvent * capacity)()
        count = ctypes.c_uint64()
        self.model._check(
            self._lib.colibri_v2_qwen_engine_step(
                self._handle, events, capacity, ctypes.byref(count)
            )
        )
        return [
            (int(e.task_id), int(e.token), int(e.kind))
            for e in events[: count.value]
        ]

    def task_cancel(self, task_id: int) -> None:
        self.model._check(self._lib.colibri_v2_qwen_task_cancel(self._handle, task_id))


class V2KvCache:
    def __init__(
        self,
        cache_keys_ptr: int,
        cache_values_ptr: int,
        capacity: int,
        kv_heads: int,
        head_dim: int,
    ):
        self._lib = _library()
        self._handle = ctypes.c_void_p()
        status = self._lib.colibri_v2_kv_cache_create(
            cache_keys_ptr,
            cache_values_ptr,
            capacity,
            kv_heads,
            head_dim,
            ctypes.byref(self._handle),
        )
        if status:
            message = (
                self._lib.colibri_v2_last_error()
                or b"native v2 KV cache creation failed"
            )
            raise V2Error(message.decode(errors="replace"))

    def close(self) -> None:
        if self._handle:
            self._lib.colibri_v2_kv_cache_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "V2KvCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def position(self) -> int:
        value = ctypes.c_int32()
        status = self._lib.colibri_v2_kv_cache_position(
            self._handle, ctypes.byref(value)
        )
        if status:
            raise V2Error("native v2 KV cache position query failed")
        return int(value.value)

    def reset(self) -> None:
        status = self._lib.colibri_v2_kv_cache_reset(self._handle)
        if status:
            raise V2Error("native v2 KV cache reset failed")

    def decoder_attention_step(
        self,
        input_ptr: int,
        norm_weights_ptr: int,
        normalized_ptr: int,
        qkv_packed_ptr: int,
        qkv_scales_ptr: int,
        qkv_ptr: int,
        attention_output_ptr: int,
        out_packed_ptr: int,
        out_scales_ptr: int,
        output_ptr: int,
        hidden_size: int,
        heads: int,
        epsilon: float,
        one_centered: bool = True,
    ) -> None:
        status = self._lib.colibri_v2_gpu_decoder_attention_cached(
            self._handle,
            input_ptr,
            norm_weights_ptr,
            normalized_ptr,
            qkv_packed_ptr,
            qkv_scales_ptr,
            qkv_ptr,
            attention_output_ptr,
            out_packed_ptr,
            out_scales_ptr,
            output_ptr,
            hidden_size,
            heads,
            epsilon,
            int(one_centered),
        )
        if status:
            message = (
                self._lib.colibri_v2_last_error()
                or b"native v2 cached decoder step failed"
            )
            raise V2Error(message.decode(errors="replace"))
