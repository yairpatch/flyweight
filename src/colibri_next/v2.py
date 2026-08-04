"""Opt-in bindings for the native Colibrì v2 ABI.

The module is intentionally small: model bytes stay memory mapped in the
native process and Python only owns handles and request-level data.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator


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
    ]


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
        return config

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
        pieces: list[int] = []
        position = 0
        for match in _SPECIAL_TOKEN_PATTERN.finditer(text):
            if match.start() > position:
                pieces.extend(
                    self._tokenize_plain(text[position : match.start()], capacity)
                )
            candidate = match.group(0)
            try:
                pieces.append(self.token_id(candidate))
            except V2Error:
                pieces.extend(self._tokenize_plain(candidate, capacity))
            position = match.end()
        if position < len(text):
            pieces.extend(self._tokenize_plain(text[position:], capacity))
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
            expert_residency = "mutable" if legacy_policy else "immutable"
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
        if next_layer_prefetch and mtp_drafts:
            raise ValueError("next_layer_prefetch does not support MTP yet")
        effective_hybrid_prefill = (
            "cpu" if resolved_expert_mode == "auto" else hybrid_prefill
        )
        effective_expert_residency = (
            "immutable" if resolved_expert_mode == "auto" else expert_residency
        )
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
        cache_types = {
            "f32": 0, "f16": 1, "bf16": 2, "q8_0": 3, "turbo3": 4, "turbo4": 5,
        }
        if cache_type_k not in cache_types or cache_type_v not in cache_types:
            raise ValueError(
                "cache_type_k/v must be one of " + ", ".join(sorted(cache_types))
            )
        self.model, self._lib = model, model._lib
        self.parallel_sequences = parallel_sequences
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
        """Run prompt ingestion and greedy decode inside the native token loop."""
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
        seed: int | None = None,
    ) -> int:
        """Queue a request on the cooperative engine; returns its task id."""
        if not prompt_tokens:
            raise ValueError("prompt_tokens must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        values = (ctypes.c_uint32 * len(prompt_tokens))(*prompt_tokens)
        stops = (ctypes.c_uint32 * len(stop_tokens))(*stop_tokens) if stop_tokens else None
        task_id = ctypes.c_uint64()
        self.model._check(
            self._lib.colibri_v2_qwen_task_submit_sampling(
                self._handle,
                values,
                len(prompt_tokens),
                max_tokens,
                stops,
                len(stop_tokens),
                temperature,
                top_k,
                top_p,
                0 if seed is None else seed & ((1 << 64) - 1),
                int(seed is not None),
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
