from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .attention import QwenFullAttentionLayer
from .attention_converter import QwenAttentionConverter
from .causal_lm import QwenForCausalLM
from .cache import LayeredExpertCache
from .converter import QwenCheckpointConverter, QwenSafetensorCheckpoint
from .cuda import active_cuda, configure_cuda, disable_cuda
from .decoder import QwenDecoderStack
from .gated_delta import QwenGatedDeltaLayer
from .gated_delta_converter import QwenGatedDeltaConverter
from .generation import TextGenerator
from .hardware import HardwareTopology, available_ram_bytes, probe_hardware
from .kernels import Q4SwiGLUExpert
from .model_io_converter import QwenModelIOConverter
from .models import model_spec
from .moe import QwenMoELayer
from .moe_converter import QwenMoELayerConverter
from .native import active_native
from .planner import PlacementPlanner
from .predictor import TransitionPredictor
from .sampling import SamplingConfig
from .server import InferenceService, serve as serve_http
from .residency import ResidencyManager
from .runtime import ToyMoERuntime
from .storage import ExpertStore
from .tokenizer import HuggingFaceTokenizer
from .tokenizer_converter import TokenizerAssetsConverter
from .validation import (
    TransformersReference,
    diagnose_hidden_states,
    diagnose_layer_components,
    validate_against_reference,
)
from .v2 import V2Model
from .v2_qwen import (
    QwenDeltaLayer,
    QwenFullAttentionLayer as V2QwenFullAttentionLayer,
    QwenMoELayer as V2QwenMoELayer,
    QwenV2Decoder,
)


def _steady_state_counters(start, end):
    """Report native runtime counter deltas over the measured window only.

    The raw ``runtime`` counters accumulate across prompt ingestion (cold,
    all-miss) and warmup, which swamps the steady-state decode signal.  Diffing
    a snapshot taken at the warmup/measured boundary against the final counters
    isolates the timed iterations.
    """
    if start is None:
        return None
    fields = (
        "decode_calls", "decode_nanoseconds", "route_wait_nanoseconds",
        "expert_page_nanoseconds", "tail_wait_nanoseconds",
        "expert_compute_nanoseconds",
        "expert_cache_hits", "expert_cache_misses", "expert_cache_evictions",
    )
    delta = {field: end[field] - start[field] for field in fields}
    calls = delta["decode_calls"] or 1
    lookups = delta["expert_cache_hits"] + delta["expert_cache_misses"]
    return {
        "decode_calls": delta["decode_calls"],
        "route_wait_ns_per_token": delta["route_wait_nanoseconds"] / calls,
        "expert_page_ns_per_token": delta["expert_page_nanoseconds"] / calls,
        "tail_wait_ns_per_token": delta["tail_wait_nanoseconds"] / calls,
        "expert_compute_ns_per_token": (
            delta["expert_compute_nanoseconds"] / calls
        ),
        "decode_ns_per_token": delta["decode_nanoseconds"] / calls,
        "expert_cache_hits": delta["expert_cache_hits"],
        "expert_cache_misses": delta["expert_cache_misses"],
        "expert_cache_evictions": delta["expert_cache_evictions"],
        "expert_cache_hit_rate": delta["expert_cache_hits"] / (lookups or 1),
    }


def _validate_mtp_cache_types(
    mtp_drafts: int, cache_type_k: str, cache_type_v: str
) -> None:
    if mtp_drafts and (cache_type_k != "f32" or cache_type_v != "f32"):
        raise SystemExit(
            "--mtp-drafts currently requires --cache-type-k f32 "
            "and --cache-type-v f32"
        )


def _benchmark_native_prefill(runtime, prompt_tokens: list[int]) -> tuple[int, float]:
    """Run the production batched prefill path and return its first token."""
    first_tokens: list[int] = []
    started = time.perf_counter()
    runtime.generate(prompt_tokens, 1, first_tokens.append)
    elapsed = time.perf_counter() - started
    if len(first_tokens) != 1:
        raise RuntimeError("native prefill did not produce exactly one token")
    return first_tokens[0], elapsed


def _drop_file_cache(path: Path) -> None:
    """Best-effort eviction of clean GGUF pages for reproducible cold A/B runs."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("--cold-cache requires POSIX_FADV_DONTNEED support")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colibri-next",
        description="Hierarchical expert residency prototype for sparse MoE inference",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect = subcommands.add_parser(
        "inspect-hardware", help="probe memory tiers and compute devices"
    )
    inspect.add_argument("--storage-path", type=Path, default=Path("."))
    inspect.add_argument("--save", type=Path)

    plan = subcommands.add_parser(
        "plan", help="create a hardware-adaptive model placement plan"
    )
    plan.add_argument("--model", default="qwen3.6-35b-a3b")
    plan.add_argument("--context", type=int, default=32_768)
    plan.add_argument("--expert-bits", type=int, choices=(4, 8, 16), default=4)
    plan.add_argument("--hardware", type=Path)
    plan.add_argument("--storage-path", type=Path, default=Path("."))
    plan.add_argument("--save", type=Path)

    inspect_model = subcommands.add_parser(
        "inspect-model", help="inspect a local sharded Qwen safetensors checkpoint"
    )
    inspect_model.add_argument("source", type=Path)

    convert = subcommands.add_parser(
        "convert-qwen", help="create a runtime manifest and optional expert files"
    )
    convert.add_argument("source", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--extract-experts", action="store_true")
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--quantization", choices=("q4", "bf16"), default="q4")

    benchmark = subcommands.add_parser(
        "benchmark-expert", help="execute and benchmark one converted Q4 expert"
    )
    benchmark.add_argument("path", type=Path)
    benchmark.add_argument("--iterations", type=int, default=5)

    convert_layers = subcommands.add_parser(
        "convert-moe-layers", help="convert routers, norms, and shared experts"
    )
    convert_layers.add_argument("source", type=Path)
    convert_layers.add_argument("output", type=Path)
    convert_layers.add_argument("--overwrite", action="store_true")

    benchmark_layer = subcommands.add_parser(
        "benchmark-moe-layer", help="execute one complete converted MoE block"
    )
    benchmark_layer.add_argument("root", type=Path)
    benchmark_layer.add_argument("--layer", type=int, default=0)
    benchmark_layer.add_argument("--iterations", type=int, default=3)
    benchmark_layer.add_argument("--residual", action="store_true")

    convert_attention = subcommands.add_parser(
        "convert-attention-layers", help="convert Qwen full-attention layers"
    )
    convert_attention.add_argument("source", type=Path)
    convert_attention.add_argument("output", type=Path)
    convert_attention.add_argument("--overwrite", action="store_true")
    convert_attention.add_argument("--quantization", choices=("bf16", "q4"), default="bf16")

    benchmark_attention = subcommands.add_parser(
        "benchmark-attention-layer", help="execute incremental full attention"
    )
    benchmark_attention.add_argument("root", type=Path)
    benchmark_attention.add_argument("--layer", type=int, default=3)
    benchmark_attention.add_argument("--tokens", type=int, default=8)

    convert_linear = subcommands.add_parser(
        "convert-linear-layers", help="convert Qwen Gated DeltaNet layers"
    )
    convert_linear.add_argument("source", type=Path)
    convert_linear.add_argument("output", type=Path)
    convert_linear.add_argument("--overwrite", action="store_true")
    convert_linear.add_argument("--quantization", choices=("bf16", "q4"), default="bf16")

    benchmark_linear = subcommands.add_parser(
        "benchmark-linear-layer", help="execute incremental Gated DeltaNet"
    )
    benchmark_linear.add_argument("root", type=Path)
    benchmark_linear.add_argument("--layer", type=int, default=0)
    benchmark_linear.add_argument("--tokens", type=int, default=8)

    benchmark_decoder = subcommands.add_parser(
        "benchmark-decoder", help="execute the complete converted decoder stack"
    )
    benchmark_decoder.add_argument("root", type=Path)
    benchmark_decoder.add_argument("--tokens", type=int, default=1)

    convert_model_io = subcommands.add_parser(
        "convert-model-io", help="convert embeddings, final norm, and LM head"
    )
    convert_model_io.add_argument("source", type=Path)
    convert_model_io.add_argument("output", type=Path)
    convert_model_io.add_argument("--overwrite", action="store_true")
    convert_model_io.add_argument("--quantization", choices=("bf16", "q4"), default="bf16")

    benchmark_logits = subcommands.add_parser(
        "benchmark-logits", help="execute token IDs through the complete model"
    )
    benchmark_logits.add_argument("root", type=Path)
    benchmark_logits.add_argument("--token-ids", default="0")
    benchmark_logits.add_argument("--rows-per-chunk", type=int, default=4096)
    _add_device_arguments(benchmark_logits)

    validate = subcommands.add_parser(
        "validate-transformers",
        help="compare converted-model logits with a Transformers checkpoint",
    )
    validate.add_argument("root", type=Path, help="converted Colibri model")
    validate.add_argument("source", type=Path, help="Hugging Face checkpoint")
    validate.add_argument("--token-ids", required=True)
    validate.add_argument("--generate-tokens", type=int, default=4)
    validate.add_argument("--top-k", type=int, default=10)
    validate.add_argument("--reference-device", default="cpu")
    validate.add_argument("--reference-dtype", default="auto")
    validate.add_argument("--reference-offload-dir", type=Path)
    validate.add_argument("--reference-gpu-memory-mib", type=int)
    validate.add_argument("--reference-cpu-memory-mib", type=int)
    validate.add_argument(
        "--layerwise",
        action="store_true",
        help="compare one-token hidden states instead of generation logits",
    )
    validate.add_argument(
        "--component-layer",
        type=int,
        help="compare internal stages of one full-attention layer",
    )
    validate.add_argument("--trust-remote-code", action="store_true")
    validate.add_argument("--rows-per-chunk", type=int, default=4096)
    _add_device_arguments(validate)

    convert_tokenizer = subcommands.add_parser(
        "convert-tokenizer", help="copy tokenizer and generation assets"
    )
    convert_tokenizer.add_argument("source", type=Path)
    convert_tokenizer.add_argument("output", type=Path)
    convert_tokenizer.add_argument("--overwrite", action="store_true")

    convert_mtp = subcommands.add_parser(
        "convert-mtp", help="extract the multi-token-prediction head (BF16)"
    )
    convert_mtp.add_argument("source", type=Path)
    convert_mtp.add_argument("output", type=Path)
    convert_mtp.add_argument("--overwrite", action="store_true")

    generate_text = subcommands.add_parser(
        "generate-text", help="generate decoded text from a prompt"
    )
    generate_text.add_argument("root", type=Path)
    generate_text.add_argument("--prompt", required=True)
    generate_text.add_argument("--system")
    generate_text.add_argument("--max-new-tokens", type=int, default=8)
    generate_text.add_argument("--temperature", type=float, default=0.0)
    generate_text.add_argument("--top-k", type=int, default=20)
    generate_text.add_argument("--top-p", type=float, default=0.95)
    generate_text.add_argument("--seed", type=int)
    generate_text.add_argument("--enable-thinking", action="store_true")
    generate_text.add_argument("--rows-per-chunk", type=int, default=4096)
    _add_device_arguments(generate_text)
    _add_expert_preload_argument(generate_text, default="none")
    _add_cpu_moe_argument(generate_text)

    generate_v2 = subcommands.add_parser(
        "generate-text-v2", help="generate with the opt-in native v2 runtime"
    )
    generate_v2.add_argument("model", type=Path)
    generate_v2.add_argument("--prompt", required=True)
    generate_v2.add_argument("--max-new-tokens", type=int, default=8)
    generate_v2.add_argument("--context-window", type=int, default=4096)

    inspect_v2 = subcommands.add_parser(
        "inspect-gguf-v2", help="inspect a GGUF using the native v2 reader"
    )
    inspect_v2.add_argument("model", type=Path)

    benchmark_v2 = subcommands.add_parser(
        "benchmark-v2", help="benchmark the real Qwen v2 CUDA decoder"
    )
    benchmark_v2.add_argument("model", type=Path)
    benchmark_v2.add_argument("--tokens", default="0")
    benchmark_v2.add_argument("--prompt")
    benchmark_v2.add_argument("--chat", action="store_true")
    benchmark_v2.add_argument("--warmup", type=int, default=3)
    benchmark_v2.add_argument("--iterations", type=int, default=10)
    benchmark_v2.add_argument("--gpu-cache-mib", type=int, default=0)
    benchmark_v2.add_argument("--context", type=int, default=2048)
    benchmark_v2.add_argument(
        "--runtime", choices=("native", "cupy-reference"), default="native",
    )
    benchmark_v2.add_argument(
        "--moe-device", choices=("gpu", "cpu", "hybrid"), default="gpu",
        help="execute routed experts by GPU streaming or native CPU kernels",
    )
    benchmark_v2.add_argument("--mtp-drafts", type=int, default=0)
    benchmark_v2.add_argument(
        "--cache-type-k", choices=("f32", "f16", "bf16", "q8_0"),
        default="f16",
    )
    benchmark_v2.add_argument(
        "--cache-type-v", choices=("f32", "f16", "bf16", "q8_0"),
        default="f16",
    )
    benchmark_v2.add_argument("--expert-top-k", type=int, default=0)
    benchmark_v2.add_argument("--expert-top-p", type=float, default=0.0)
    benchmark_v2.add_argument(
        "--parallel", type=int, default=1, dest="parallel_sequences",
        help="allocate this many independent sequence slots",
    )
    benchmark_v2.add_argument("--prompt-cache-mib", type=int, default=0)
    benchmark_v2.add_argument(
        "--prefill-cache-seed", type=int, default=0,
        help="bulk-load this many prompt-hot experts per layer before decode",
    )
    benchmark_v2.add_argument(
        "--expert-paging", choices=("auto", "staged", "direct"), default="auto",
        help="hybrid expert transfer policy (auto uses direct DMA only with host-memory headroom)",
    )
    benchmark_v2.add_argument(
        "--cpu-prefetch-mib", type=int, default=0,
        help="host page-cache budget for prompt-hot CPU/hybrid experts (0 = off)",
    )
    benchmark_v2.add_argument(
        "--cold-cache", action="store_true",
        help="best-effort eviction of clean GGUF pages before a cold-start A/B run",
    )

    probe_qwen_v2 = subcommands.add_parser(
        "probe-qwen-v2", help="run one real Qwen v2 block and routed MoE from GGUF"
    )
    probe_qwen_v2.add_argument("model", type=Path)
    probe_qwen_v2.add_argument("--layer", type=int, default=0)
    probe_qwen_v2.add_argument("--token-id", type=int, default=0)
    probe_qwen_v2.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    stack_qwen_v2 = subcommands.add_parser(
        "probe-qwen-stack-v2", help="run one token through the complete Qwen v2 block stack"
    )
    stack_qwen_v2.add_argument("model", type=Path)
    stack_qwen_v2.add_argument("--token-id", type=int, default=0)
    stack_qwen_v2.add_argument("--prompt")
    stack_qwen_v2.add_argument("--chat", action="store_true", help="wrap prompt in the Qwen user/assistant chat template")
    stack_qwen_v2.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    stack_qwen_v2.add_argument(
        "--gpu-cache-mib", type=int, default=0,
        help="bounded CUDA weight cache in MiB (0 selects from free VRAM)",
    )

    native_qwen_v2 = subcommands.add_parser(
        "probe-qwen-native-v2",
        aliases=("probe-native-v2",),
        help="run the one-call C++/CUDA native v2 decoder (Qwen or Gemma 4)",
    )
    native_qwen_v2.add_argument("model", type=Path)
    native_qwen_v2.add_argument("--token-id", type=int, default=0)
    native_qwen_v2.add_argument("--prompt")
    native_qwen_v2.add_argument("--chat", action="store_true")
    native_qwen_v2.add_argument("--generate-tokens", type=int, default=2)
    native_qwen_v2.add_argument("--context", type=int, default=2048)
    native_qwen_v2.add_argument(
        "--gpu-cache-mib", type=int, default=0,
        help="total v2 CUDA budget in MiB (0 = auto-fit to free VRAM)")
    native_qwen_v2.add_argument(
        "--moe-device", choices=("gpu", "cpu", "hybrid"), default=None,
        help="expert backend (default: CPU for Gemma 4, GPU otherwise)",
    )
    native_qwen_v2.add_argument("--mtp-drafts", type=int, default=0)
    stack_qwen_v2.add_argument(
        "--profile", action="store_true",
        help="synchronize token boundaries and report prompt/decode timings",
    )
    stack_qwen_v2.add_argument("--top-k", type=int, default=10)
    stack_qwen_v2.add_argument("--generate-tokens", type=int, default=0)

    serve_command = subcommands.add_parser(
        "serve", help="run a persistent OpenAI-compatible local server"
    )
    serve_command.add_argument("root", type=Path)
    serve_command.add_argument("--host", default="127.0.0.1")
    serve_command.add_argument("--port", type=int, default=8000)
    serve_command.add_argument("--model-name")
    serve_command.add_argument(
        "--kv-cache-type",
        choices=("f32", "q8"),
        default=None,
        help="attention KV-cache storage type",
    )
    serve_command.add_argument(
        "--strict-model",
        action="store_true",
        help="reject request model IDs that differ from the loaded model name",
    )
    serve_command.add_argument(
        "--api-key", default=os.environ.get("COLIBRI_API_KEY")
    )
    serve_command.add_argument("--cors-origin", default="*")
    serve_command.add_argument(
        "--context-window",
        type=int,
        default=4096,
        help="maximum combined prompt and generated tokens",
    )
    serve_command.add_argument(
        "--max-new-tokens",
        type=int,
        help="optional output ceiling below the context window",
    )
    serve_command.add_argument("--rows-per-chunk", type=int, default=4096)
    _add_device_arguments(serve_command)
    _add_expert_preload_argument(serve_command, default="auto")
    _add_cpu_moe_argument(serve_command)

    serve_v2 = subcommands.add_parser(
        "serve-v2", help="run the OpenAI-compatible server on native v2 GGUF"
    )
    serve_v2.add_argument("model", type=Path)
    serve_v2.add_argument("--host", default="127.0.0.1")
    serve_v2.add_argument("--port", type=int, default=8000)
    serve_v2.add_argument("--model-name")
    serve_v2.add_argument("--device", type=int, default=0)
    serve_v2.add_argument(
        "--strict-model",
        action="store_true",
        help="reject request model IDs that differ from the loaded model name",
    )
    serve_v2.add_argument(
        "--mtp-drafts", type=int, default=0,
        help="native MTP draft depth (0 disables speculative decoding)",
    )
    serve_v2.add_argument(
        "--api-key", default=os.environ.get("COLIBRI_API_KEY")
    )
    serve_v2.add_argument("--cors-origin", default="*")
    serve_v2.add_argument("--context-window", type=int, default=32_768)
    # Agentic clients (Claude Code etc.) request large output budgets and treat
    # them as an upper bound; the service clamps requests to this ceiling.
    serve_v2.add_argument("--max-new-tokens", type=int, default=4096)
    serve_v2.add_argument(
        "--gpu-cache-mib",
        type=int,
        default=0,
        help="total native v2 CUDA allocation budget in MiB (0 = auto-fit to free VRAM)",
    )
    serve_v2.add_argument(
        "--moe-device",
        choices=("gpu", "cpu", "hybrid"),
        default="hybrid",
        help="routed-expert execution policy",
    )
    serve_v2.add_argument(
        "--cache-type-k",
        choices=("f32", "f16", "bf16", "q8_0"),
        default="f16",
        help="KV cache K precision (llama.cpp -ctk); f16 halves KV VRAM",
    )
    serve_v2.add_argument(
        "--cache-type-v",
        choices=("f32", "f16", "bf16", "q8_0"),
        default="f16",
        help="KV cache V precision (llama.cpp -ctv); f16 halves KV VRAM",
    )
    serve_v2.add_argument(
        "--prefill-checkpoint-interval",
        type=int,
        default=256,
        help="position of the first mid-prefill recurrent-state checkpoint; "
        "later ones are geometric (interval<<k). 0 disables checkpointing so a "
        "diverging prefix (agentic clients injecting reminders/tool results) "
        "reprefills from token 0 instead of resuming near the divergence point.",
    )
    serve_v2.add_argument(
        "--prefill-checkpoint-slots",
        type=int,
        default=4,
        help="total prefix-reuse snapshot slots (one reserved for the exact "
        "end-of-prompt snapshot); more slots widen checkpoint coverage at the "
        "cost of GPU VRAM shared with the expert cache",
    )
    serve_v2.add_argument(
        "--parallel",
        type=int,
        default=1,
        dest="parallel_sequences",
        help="independent decode slots (llama.cpp --parallel); each has its own "
        "KV cache so interleaved side-requests (subagents, title/quota calls) "
        "don't evict the main conversation's prefix. Costs ~1 KV cache per slot "
        "(~20 KB/token); 1 = single-sequence (default)",
    )
    serve_v2.add_argument(
        "--prompt-cache-mib",
        type=int,
        default=0,
        help="host RAM budget (MiB) for the prompt cache (llama.cpp-style): when "
        "a slot is recycled its state is spilled to RAM and restored on a later "
        "matching request instead of reprefilling cold. Needs --parallel >= 2; "
        "each cached conversation costs one slot's state (~20 KB/token). 0 = off",
    )
    serve_v2.add_argument(
        "--swa-full",
        action="store_true",
        help="allocate full-size caches for sliding-attention layers; uses more "
        "VRAM but preserves unrestricted prefix-cache rollback",
    )
    serve_v2.add_argument(
        "--prefill-cache-seed", type=int, default=0,
        help="experimental Qwen prompt-trained expert seed per layer (0 = off)",
    )
    serve_v2.add_argument(
        "--expert-paging", choices=("auto", "staged", "direct"), default="auto",
        help="hybrid expert transfer policy; direct trades startup time/RAM pinning for throughput",
    )
    serve_v2.add_argument(
        "--cpu-prefetch-mib", type=int, default=0,
        help="host page-cache budget for prompt-hot CPU/hybrid experts (0 = off)",
    )

    create = subcommands.add_parser("create-demo", help="create deterministic experts")
    create.add_argument("path", type=Path)
    create.add_argument("--layers", type=int, default=6)
    create.add_argument("--experts", type=int, default=12)
    create.add_argument("--width", type=int, default=16)

    run = subcommands.add_parser("run", help="run the toy MoE scheduler")
    run.add_argument("path", type=Path)
    run.add_argument("--tokens", type=int, default=32)
    run.add_argument("--cache-experts", type=int, default=4)
    run.add_argument("--top-k", type=int, default=2)
    run.add_argument("--prefetch", type=int, default=2)
    run.add_argument("--warmup", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect-gguf-v2":
        with V2Model(args.model) as model:
            print(json.dumps({"model": model.info, "tensors": list(model.tensors())}, indent=2))
        return 0
    if args.command == "generate-text-v2":
        with V2Model(args.model) as model, model.session(args.context_window) as session:
            session.prompt(list(args.prompt.encode("utf-8")))
            generated = [session.decode() for _ in range(args.max_new_tokens)]
            print(json.dumps({"text": "", "generated_tokens": generated, "execution": "native-v2", "stats": session.stats}, indent=2))
        return 0
    if args.command == "benchmark-v2":
        if args.iterations < 3:
            raise SystemExit("benchmark-v2 requires at least 3 measured iterations")
        if args.warmup < 1:
            raise SystemExit("benchmark-v2 requires at least 1 warmup iteration")
        if args.context <= 0:
            raise SystemExit("benchmark-v2 requires a positive context")
        if args.parallel_sequences < 1:
            raise SystemExit("benchmark-v2 requires --parallel >= 1")
        if args.prompt_cache_mib < 0:
            raise SystemExit("benchmark-v2 requires --prompt-cache-mib >= 0")
        if args.cpu_prefetch_mib < 0:
            raise SystemExit("benchmark-v2 requires --cpu-prefetch-mib >= 0")
        if args.expert_top_k < 0:
            raise SystemExit("benchmark-v2 requires --expert-top-k >= 0")
        if not 0 <= args.prefill_cache_seed <= 256:
            raise SystemExit(
                "benchmark-v2 requires --prefill-cache-seed within [0, 256]"
            )
        if not 0.0 <= args.expert_top_p <= 1.0:
            raise SystemExit("benchmark-v2 requires --expert-top-p within [0, 1]")
        _validate_mtp_cache_types(
            args.mtp_drafts, args.cache_type_k, args.cache_type_v
        )
        if args.runtime == "native":
            if args.cold_cache:
                _drop_file_cache(args.model)
            with V2Model(args.model) as model:
                cache_mib = args.gpu_cache_mib  # 0 = auto-fit to free VRAM
                prompt_text = args.prompt
                if args.chat:
                    if prompt_text is None:
                        raise SystemExit("--chat requires --prompt")
                    prompt_text = (
                        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
                        "<|im_start|>assistant\n<think>\n"
                    )
                prompt_tokens = (
                    model.tokenize(prompt_text)
                    if prompt_text is not None
                    else [int(value) for value in args.tokens.split(",") if value.strip()]
                )
                if not prompt_tokens:
                    raise SystemExit("benchmark prompt must contain at least one token")
                if len(prompt_tokens) + args.warmup + args.iterations > args.context:
                    raise SystemExit(
                        "benchmark prompt, warmup, and measured tokens exceed --context"
                    )
                with model.native_qwen_runtime(
                    context_limit=args.context,
                    gpu_cache_bytes=cache_mib * 1024**2,
                    moe_device=args.moe_device,
                    mtp_drafts=args.mtp_drafts,
                    expert_top_k=args.expert_top_k,
                    expert_top_p=args.expert_top_p,
                    cache_type_k=args.cache_type_k,
                    cache_type_v=args.cache_type_v,
                    parallel_sequences=args.parallel_sequences,
                    prompt_cache_mib=args.prompt_cache_mib,
                    prefill_cache_seed=args.prefill_cache_seed,
                    expert_paging=args.expert_paging,
                    cpu_prefetch_mib=args.cpu_prefetch_mib,
                ) as runtime:
                    prepare_started = time.perf_counter()
                    runtime.prepare()
                    prepare_seconds = time.perf_counter() - prepare_started
                    current_token, prompt_seconds = _benchmark_native_prefill(
                        runtime, prompt_tokens
                    )
                    if args.mtp_drafts:
                        warm_outputs: list[int] = []
                        warm_started = time.perf_counter()
                        runtime.generate(
                            prompt_tokens,
                            args.warmup + 1,
                            warm_outputs.append,
                        )
                        warm_elapsed = time.perf_counter() - warm_started
                        measured_prompt = [
                            *prompt_tokens,
                            *warm_outputs[:-1],
                        ]
                        measured_outputs: list[int] = []
                        measured_started = 0.0

                        def receive_measured(token: int) -> None:
                            nonlocal measured_started
                            measured_outputs.append(token)
                            if len(measured_outputs) == 1:
                                measured_started = time.perf_counter()

                        steady_start = runtime.info
                        runtime.generate(
                            measured_prompt,
                            args.iterations + 1,
                            receive_measured,
                        )
                        measured_total = time.perf_counter() - measured_started
                        runtime_info = runtime.info
                        generated = [
                            *warm_outputs[1:],
                            *measured_outputs[1:],
                        ]
                        steady_state = _steady_state_counters(
                            steady_start, runtime_info
                        )
                        average = measured_total / args.iterations
                        print(json.dumps({
                            "execution": (
                                f"native-v2-cpp-cuda-{args.moe_device}-moe-mtp"
                                if args.moe_device != "gpu"
                                else "native-v2-cpp-cuda-mtp"
                            ),
                            "measurement": "aggregate native generation wall time",
                            "cold_cache": args.cold_cache,
                            "device": runtime_info["device"],
                            "prepare_seconds": prepare_seconds,
                            "prompt_tokens": len(prompt_tokens),
                            "prompt_seconds": prompt_seconds,
                            "prompt_tokens_per_second": (
                                len(prompt_tokens) / prompt_seconds
                            ),
                            "first_token_latency_seconds": prompt_seconds,
                            "warmup_decode_seconds": warm_elapsed,
                            "warmup_iterations": args.warmup,
                            "iterations": args.iterations,
                            "decode_batch_seconds": measured_total,
                            "decode_median_seconds": average,
                            "decode_variance_seconds2": 0.0,
                            "decode_tokens_per_second": (
                                args.iterations / measured_total
                            ),
                            "decode_median_tokens_per_second": 1.0 / average,
                            "generated_tokens": generated,
                            "generated_text": model.decode_tokens(generated),
                            "steady_state": steady_state,
                            "runtime": runtime_info,
                        }, indent=2))
                        return 0
                    generated = []
                    warmup_seconds = []
                    measured_seconds = []
                    steady_start = None
                    for step in range(args.warmup + args.iterations):
                        if step == args.warmup:
                            steady_start = runtime.info
                        step_started = time.perf_counter()
                        current_token = runtime.decode(current_token)
                        elapsed = time.perf_counter() - step_started
                        generated.append(current_token)
                        (warmup_seconds if step < args.warmup else measured_seconds).append(elapsed)
                    runtime_info = runtime.info
                measured_total = sum(measured_seconds)
                steady_state = _steady_state_counters(steady_start, runtime_info)
                print(json.dumps({
                    "execution": (
                        f"native-v2-cpp-cuda-{args.moe_device}-moe"
                        if args.moe_device != "gpu" else "native-v2-cpp-cuda"
                    ),
                    "measurement": "batched prefill plus steady single-token decode",
                    "cold_cache": args.cold_cache,
                    "device": runtime_info["device"],
                    "prepare_seconds": prepare_seconds,
                    "prompt_tokens": len(prompt_tokens),
                    "prompt_seconds": prompt_seconds,
                    "prompt_tokens_per_second": len(prompt_tokens) / prompt_seconds,
                    "first_token_latency_seconds": prompt_seconds,
                    "first_warm_decode_seconds": warmup_seconds[0],
                    "warmup_iterations": args.warmup,
                    "iterations": args.iterations,
                    "decode_seconds": measured_seconds,
                    "decode_median_seconds": statistics.median(measured_seconds),
                    "decode_variance_seconds2": statistics.pvariance(measured_seconds),
                    "decode_tokens_per_second": args.iterations / measured_total,
                    "decode_median_tokens_per_second": 1.0 / statistics.median(measured_seconds),
                    "generated_tokens": generated,
                    "generated_text": model.decode_tokens(generated),
                    "steady_state": steady_state,
                    "runtime": runtime_info,
                }, indent=2))
            return 0
        cuda_enabled = False
        try:
            with V2Model(args.model) as model:
                decoder = QwenV2Decoder(model)
                state = decoder.new_state()
                cache_mib = args.gpu_cache_mib
                if cache_mib <= 0:
                    free_mib = V2Model.gpu_info()["free_memory"] // (1024 * 1024)
                    cache_mib = max(1024, min(6144, free_mib - 4096))
                accelerator = configure_cuda(cache_mib=cache_mib)
                cuda_enabled = True
                prompt_text = args.prompt
                if args.chat:
                    if prompt_text is None:
                        raise SystemExit("--chat requires --prompt")
                    prompt_text = (
                        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
                        "<|im_start|>assistant\n<think>\n"
                    )
                prompt_tokens = (
                    model.tokenize(prompt_text)
                    if prompt_text is not None
                    else [int(value) for value in args.tokens.split(",") if value.strip()]
                )
                if not prompt_tokens:
                    raise SystemExit("benchmark prompt must contain at least one token")
                prompt_started = time.perf_counter()
                for token in prompt_tokens[:-1]:
                    decoder.forward_token_cuda(token, state, accelerator)
                accelerator.cp.cuda.runtime.deviceSynchronize()
                prompt_seconds = time.perf_counter() - prompt_started

                current_token = prompt_tokens[-1]
                generated = []
                warmup_seconds = []
                measured_seconds = []
                for step in range(args.warmup + args.iterations):
                    step_started = time.perf_counter()
                    hidden, _ = decoder.forward_token_cuda(
                        current_token, state, accelerator
                    )
                    logits = decoder.logits_cuda(hidden, accelerator)
                    current_token = int(accelerator.cp.argmax(logits).get())
                    elapsed = time.perf_counter() - step_started
                    generated.append(current_token)
                    if step < args.warmup:
                        warmup_seconds.append(elapsed)
                    else:
                        measured_seconds.append(elapsed)
                measured_total = sum(measured_seconds)
                output = {
                    "execution": "native-v2-cuda",
                    "device": accelerator.device_name,
                    "prompt_tokens": len(prompt_tokens),
                    "prompt_tokens_processed": max(0, len(prompt_tokens) - 1),
                    "prompt_seconds": prompt_seconds,
                    "prompt_tokens_per_second": (
                        (len(prompt_tokens) - 1) / prompt_seconds
                        if prompt_seconds and len(prompt_tokens) > 1 else 0.0
                    ),
                    "first_token_latency_seconds": warmup_seconds[0],
                    "warmup_iterations": args.warmup,
                    "iterations": args.iterations,
                    "decode_seconds": measured_seconds,
                    "decode_median_seconds": statistics.median(measured_seconds),
                    "decode_variance_seconds2": statistics.pvariance(measured_seconds),
                    "decode_tokens_per_second": (
                        args.iterations / measured_total if measured_total else 0.0
                    ),
                    "generated_tokens": generated,
                    "generated_text": model.decode_tokens(generated),
                    "cuda_stats": accelerator.stats(),
                }
                print(json.dumps(output, indent=2))
        finally:
            if cuda_enabled:
                disable_cuda()
        return 0
    if args.command == "probe-qwen-v2":
        cuda_enabled = False
        with V2Model(args.model) as model:
            model.validate_qwen()
            width = int(model.config["hidden_size"])
            hidden = model.qwen_embedding(args.token_id, width)
            try:
                model.qwen_layer_tensor(args.layer, "attention_q")
            except Exception:
                block = QwenDeltaLayer(model, args.layer)
                state = block.new_state()
                if args.device == "cuda":
                    accelerator = configure_cuda(cache_mib=512)
                    cuda_enabled = True
                    hidden = block.forward_cuda(hidden, state, accelerator).get().tolist()
                else:
                    hidden = block.forward_residual(hidden, state)
                block_kind = "deltanet"
            else:
                block = V2QwenFullAttentionLayer(model, args.layer)
                state = block.new_state()
                if args.device == "cuda":
                    accelerator = configure_cuda(cache_mib=512)
                    cuda_enabled = True
                    hidden = block.forward_cuda(hidden, state, accelerator).get().tolist()
                else:
                    hidden = block.forward_residual(hidden, state)
                block_kind = "attention"
            mixer_hidden = list(hidden)
            moe = V2QwenMoELayer(model, args.layer)
            if args.device == "cuda":
                output_device, experts, weights = moe.forward_cuda(hidden, accelerator)
                output = output_device.get().tolist()
            else:
                output, experts, weights = moe.forward_residual(hidden)
            print(json.dumps({
                "execution": f"native-v2-{args.device}",
                "device": args.device,
                "layer": args.layer,
                "block": block_kind,
                "token_id": args.token_id,
                "hidden_size": len(output),
                "experts": experts,
                "weights": weights,
                "output_first": output[:8],
                "output_min": min(output),
                "output_max": max(output),
                "mixer_first": mixer_hidden[:8],
                "mixer_min": min(mixer_hidden),
                "mixer_max": max(mixer_hidden),
                "shared_first": (moe.last_cuda_shared if args.device == "cuda" else moe.last_cpu_shared).get().tolist()[:8]
                if args.device == "cuda" else (moe.last_cpu_shared.tolist()[:8]),
                "routed_first": (moe.last_cuda_routed if args.device == "cuda" else moe.last_cpu_routed).get().tolist()[:8]
                if args.device == "cuda" else (moe.last_cpu_routed.tolist()[:8]),
                "expert_gate_first": (moe.last_cuda_expert[0] if args.device == "cuda" else moe.last_cpu_expert[0]).get().tolist()[:8]
                if args.device == "cuda" else moe.last_cpu_expert[0].tolist()[:8],
                "expert_activated_first": (moe.last_cuda_expert[1] if args.device == "cuda" else moe.last_cpu_expert[1]).get().tolist()[:8]
                if args.device == "cuda" else moe.last_cpu_expert[1].tolist()[:8],
                "expert_down_first": (moe.last_cuda_expert[2] if args.device == "cuda" else moe.last_cpu_expert[2]).get().tolist()[:8]
                if args.device == "cuda" else moe.last_cpu_expert[2].tolist()[:8],
            }, indent=2))
        if cuda_enabled:
            disable_cuda()
        return 0
    if args.command in ("probe-qwen-native-v2", "probe-native-v2"):
        if args.generate_tokens < 0:
            raise SystemExit("--generate-tokens must be non-negative")
        if args.context <= 0 or args.gpu_cache_mib < 0:
            raise SystemExit("--context must be positive and --gpu-cache-mib non-negative")
        with V2Model(args.model) as model:
            moe_device = args.moe_device or (
                "hybrid" if model.info["architecture"] == "gemma4" else "gpu"
            )
            prompt_text = args.prompt
            if args.chat:
                if prompt_text is None:
                    raise SystemExit("--chat requires --prompt")
                from .v2_server import NativeV2Tokenizer
                prompt_text = NativeV2Tokenizer(model).format_messages(
                    [{"role": "user", "content": prompt_text}],
                    enable_thinking=True,
                )
            prompt_tokens = (
                model.tokenize(prompt_text)
                if prompt_text is not None else [args.token_id]
            )
            if not prompt_tokens:
                raise SystemExit("native prompt must contain at least one token")
            generated: list[int] = []
            step_seconds: list[float] = []
            started = time.perf_counter()
            with model.native_runtime(
                context_limit=args.context,
                gpu_cache_bytes=args.gpu_cache_mib * 1024**2,
                moe_device=moe_device,
                mtp_drafts=args.mtp_drafts,
            ) as runtime:
                prepare_started = time.perf_counter()
                runtime.prepare()
                prepare_seconds = time.perf_counter() - prepare_started
                next_token = 0
                for token in prompt_tokens:
                    step_started = time.perf_counter()
                    next_token = runtime.decode(token)
                    step_seconds.append(time.perf_counter() - step_started)
                for index in range(args.generate_tokens):
                    generated.append(next_token)
                    if index + 1 < args.generate_tokens:
                        step_started = time.perf_counter()
                        next_token = runtime.decode(next_token)
                        step_seconds.append(time.perf_counter() - step_started)
                runtime_info = runtime.info
            print(json.dumps({
                "execution": (
                    f"native-v2-cpp-cuda-{moe_device}-moe"
                    if moe_device != "gpu" else "native-v2-cpp-cuda"
                ),
                "prompt": prompt_text,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated,
                "generated_pieces": [model.token_text(token) for token in generated],
                "generated_text": model.decode_tokens(generated),
                "prepare_seconds": prepare_seconds,
                "step_seconds": step_seconds,
                "decode_median_seconds": statistics.median(step_seconds),
                "decode_tokens_per_second": (
                    1.0 / statistics.median(step_seconds)
                    if step_seconds and statistics.median(step_seconds) else 0.0
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "runtime": runtime_info,
            }, indent=2))
        return 0
    if args.command == "probe-qwen-stack-v2":
        cuda_enabled = False
        with V2Model(args.model) as model:
            decoder = QwenV2Decoder(model)
            state = decoder.new_state()
            accelerator = None
            if args.device == "cuda":
                cache_mib = args.gpu_cache_mib
                if cache_mib <= 0:
                    free_mib = V2Model.gpu_info()["free_memory"] // (1024 * 1024)
                    cache_mib = max(1024, min(6144, free_mib - 4096))
                accelerator = configure_cuda(cache_mib=cache_mib)
                accelerator.enable_profiling(args.profile)
                cuda_enabled = True
            generated = []
            prompt_text = args.prompt
            if args.chat:
                if prompt_text is None:
                    raise SystemExit("--chat requires --prompt")
                prompt_text = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            prompt_tokens = model.tokenize(prompt_text) if prompt_text is not None else [args.token_id]
            started = time.perf_counter()
            prompt_step_seconds = []
            decode_step_seconds = []
            for prompt_token in prompt_tokens[:-1]:
                step_started = time.perf_counter()
                if args.device == "cuda":
                    decoder.forward_token_cuda(prompt_token, state, accelerator)
                    if args.profile:
                        accelerator.cp.cuda.runtime.deviceSynchronize()
                else:
                    decoder.forward_token(prompt_token, state)
                if args.profile:
                    prompt_step_seconds.append(time.perf_counter() - step_started)
            current_token = prompt_tokens[-1]
            values, routes, logits = None, (), None
            for step in range(args.generate_tokens + 1):
                step_started = time.perf_counter()
                if args.device == "cuda":
                    hidden, routes = decoder.forward_token_cuda(current_token, state, accelerator)
                    logits_device = decoder.logits_cuda(hidden, accelerator)
                    next_token = int(accelerator.cp.argmax(logits_device).get())
                    if step == args.generate_tokens:
                        values = hidden.get().tolist()
                        top = accelerator.cp.argsort(logits_device)[-args.top_k:][::-1]
                        top_tokens = top.get().tolist()
                        top_logits = logits_device[top].get().tolist()
                else:
                    values, routes = decoder.forward_token(current_token, state)
                    logits = model.qwen_lm_head(values, int(model.config["vocabulary_size"]))
                    next_token = max(range(len(logits)), key=logits.__getitem__)
                    if step == args.generate_tokens:
                        top_tokens = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:args.top_k]
                        top_logits = [logits[index] for index in top_tokens]
                if step < args.generate_tokens:
                    generated.append(next_token)
                    current_token = next_token
                if args.profile:
                    if accelerator is not None:
                        accelerator.cp.cuda.runtime.deviceSynchronize()
                    decode_step_seconds.append(time.perf_counter() - step_started)
            print(json.dumps({
                "execution": f"native-v2-{args.device}",
                "device": args.device,
                "token_id": args.token_id,
                "prompt": prompt_text,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated,
                "generated_pieces": [model.token_text(token) for token in generated],
                "generated_text": model.decode_tokens(generated),
                "layers": decoder.layers,
                "hidden_size": len(values),
                "routes_recorded": len(routes),
                "top_tokens": top_tokens,
                "top_logits": top_logits,
                "hidden_first": values[:8],
                "hidden_min": min(values),
                "hidden_max": max(values),
                "elapsed_seconds": time.perf_counter() - started,
                "profile": {
                    "prompt_step_seconds": prompt_step_seconds,
                    "decode_step_seconds": decode_step_seconds,
                    "cold_prompt_seconds": prompt_step_seconds[0] if prompt_step_seconds else 0.0,
                    "warm_prompt_median_seconds": statistics.median(prompt_step_seconds[1:]) if len(prompt_step_seconds) > 1 else 0.0,
                    "decode_median_seconds": statistics.median(decode_step_seconds) if decode_step_seconds else 0.0,
                } if args.profile else None,
                "cuda_stats": accelerator.stats() if accelerator is not None else None,
            }, indent=2))
        if cuda_enabled:
            disable_cuda()
        return 0
    if args.command == "inspect-hardware":
        topology = probe_hardware(args.storage_path)
        output = topology.to_json()
        if args.save:
            args.save.write_text(output, encoding="utf-8")
        print(output, end="")
        return 0

    if args.command == "plan":
        topology = (
            HardwareTopology.from_json_file(args.hardware)
            if args.hardware
            else probe_hardware(args.storage_path)
        )
        placement = PlacementPlanner(topology).plan(
            model_spec(args.model),
            context_length=args.context,
            expert_bits=args.expert_bits,
        )
        output = placement.to_json()
        if args.save:
            args.save.write_text(output, encoding="utf-8")
        print(output, end="")
        return 0 if placement.supported else 2

    if args.command == "inspect-model":
        print(json.dumps(QwenSafetensorCheckpoint(args.source).inspect(), indent=2))
        return 0

    if args.command == "convert-qwen":
        def progress(completed: int, total: int) -> None:
            if completed == total or completed % 64 == 0:
                print(
                    f"\rExtracted experts: {completed}/{total}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

        manifest = QwenCheckpointConverter(args.source).convert(
            args.output,
            extract_experts=args.extract_experts,
            quantization=args.quantization,
            overwrite=args.overwrite,
            progress=progress if args.extract_experts else None,
        )
        if args.extract_experts:
            print(file=sys.stderr)
        print(
            json.dumps(
                {
                    "manifest": str((args.output / "manifest.json").resolve()),
                    "architecture": manifest["architecture"],
                    "expert_storage": manifest["expert_storage"]["mode"],
                    "expected_expert_files": manifest["model"]["layers"]
                    * manifest["model"]["experts_per_layer"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "convert-moe-layers":
        def layer_progress(completed: int, total: int) -> None:
            print(
                f"\rConverted MoE layers: {completed}/{total}",
                end="",
                file=sys.stderr,
                flush=True,
            )

        storage = QwenMoELayerConverter(args.source).convert(
            args.output,
            overwrite=args.overwrite,
            progress=layer_progress,
        )
        print(file=sys.stderr)
        print(json.dumps(storage, indent=2))
        return 0

    if args.command == "benchmark-moe-layer":
        if args.iterations <= 0:
            raise ValueError("iterations must be positive")
        layer = QwenMoELayer.from_model_directory(args.root, args.layer)
        hidden = [math.sin(index * 0.17) for index in range(layer.hidden_size)]
        started = time.perf_counter()
        result = None
        for _ in range(args.iterations):
            result = (
                layer.forward_residual(hidden)
                if args.residual
                else layer.forward(hidden)
            )
        elapsed = time.perf_counter() - started
        assert result is not None
        print(
            json.dumps(
                {
                    "layer": args.layer,
                    "iterations": args.iterations,
                    "seconds": elapsed,
                    "layers_per_second": args.iterations / elapsed if elapsed else 0.0,
                    "selected_experts": result.selected_experts,
                    "routing_weights": result.routing_weights,
                    "output_checksum": round(sum(result.output), 6),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "convert-tokenizer":
        storage = TokenizerAssetsConverter(args.source).convert(
            args.output, overwrite=args.overwrite
        )
        print(json.dumps(storage, indent=2))
        return 0

    if args.command == "convert-mtp":
        from .mtp_converter import QwenMtpConverter

        storage = QwenMtpConverter(args.source).convert(
            args.output, overwrite=args.overwrite
        )
        print(json.dumps(storage, indent=2))
        return 0

    if args.command == "serve":
        _configure_execution(args)
        print(f"Loading model from {args.root}...", file=sys.stderr, flush=True)
        service = InferenceService.from_model_directory(
            args.root,
            model_name=args.model_name,
            rows_per_chunk=args.rows_per_chunk,
            max_new_tokens=args.max_new_tokens or args.context_window,
            context_window=args.context_window,
            api_key=args.api_key,
            cors_origin=args.cors_origin,
            expert_preload=args.expert_preload,
            cpu_moe_layers=args.cpu_moe_layers,
            strict_model=args.strict_model,
            expert_preload_progress=_expert_preload_progress
            if args.expert_preload != "none"
            else None,
        )
        if args.expert_preload != "none":
            print(file=sys.stderr)
        print(
            f"Serving {service.model_name} at http://{args.host}:{args.port}",
            file=sys.stderr,
            flush=True,
        )
        try:
            serve_http(service, host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("Server stopped.", file=sys.stderr)
        return 0

    if args.command == "serve-v2":
        from .v2_server import NativeV2InferenceService

        if args.gpu_cache_mib < 0:
            raise SystemExit("--gpu-cache-mib must be >= 0 (0 = auto-fit to free VRAM)")
        if args.cpu_prefetch_mib < 0:
            raise SystemExit("--cpu-prefetch-mib must be >= 0")
        if not 0 <= args.prefill_cache_seed <= 256:
            raise SystemExit("--prefill-cache-seed must be within [0, 256]")
        _validate_mtp_cache_types(
            args.mtp_drafts, args.cache_type_k, args.cache_type_v
        )
        print(
            f"Loading native v2 GGUF from {args.model}...",
            file=sys.stderr,
            flush=True,
        )
        service = NativeV2InferenceService(
            args.model,
            model_name=args.model_name,
            device=args.device,
            context_window=args.context_window,
            max_new_tokens=args.max_new_tokens,
            gpu_cache_mib=args.gpu_cache_mib,
                moe_device=args.moe_device,
                mtp_drafts=args.mtp_drafts,
                cache_type_k=args.cache_type_k,
                cache_type_v=args.cache_type_v,
                prefill_checkpoint_interval=args.prefill_checkpoint_interval,
                prefill_checkpoint_slots=args.prefill_checkpoint_slots,
                parallel_sequences=args.parallel_sequences,
                prompt_cache_mib=args.prompt_cache_mib,
                swa_full=args.swa_full,
                prefill_cache_seed=args.prefill_cache_seed,
                expert_paging=args.expert_paging,
                cpu_prefetch_mib=args.cpu_prefetch_mib,
                api_key=args.api_key,
            cors_origin=args.cors_origin,
            strict_model=args.strict_model,
        )
        print(
            f"Serving {service.model_name} at http://{args.host}:{args.port} "
            f"(native v2, {args.moe_device} MoE)",
            file=sys.stderr,
            flush=True,
        )
        try:
            serve_http(service, host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("Server stopped.", file=sys.stderr)
        finally:
            service.close()
        return 0

    if args.command == "generate-text":
        _configure_execution(args)
        tokenizer = HuggingFaceTokenizer.from_model_directory(args.root)
        model = QwenForCausalLM.from_model_directory(
            args.root, rows_per_chunk=args.rows_per_chunk
        )
        model.configure_moe_placement(args.cpu_moe_layers)
        model.preload_experts(
            mode=args.expert_preload,
            available_ram_bytes=available_ram_bytes()
            if args.expert_preload == "auto"
            else None,
            progress=_expert_preload_progress
            if args.expert_preload != "none"
            else None,
        )
        if args.expert_preload != "none":
            print(file=sys.stderr)
        prompt_tokens = tokenizer.encode_chat(
            args.prompt,
            system=args.system,
            enable_thinking=args.enable_thinking,
        )
        print(
            f"Processing {len(prompt_tokens)} prompt tokens and up to "
            f"{args.max_new_tokens} new tokens...",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        result = TextGenerator(model, tokenizer).generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            sampling=SamplingConfig(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            ),
            system=args.system,
            enable_thinking=args.enable_thinking,
        )
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "text": result.text,
                    "prompt_tokens": len(result.prompt_ids),
                    "generated_tokens": list(result.generated_ids),
                    "stopped_on_eos": result.stopped_on_eos,
                    "state_tokens": result.state_tokens,
                    "seconds": elapsed,
                    "execution": _execution_stats(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "convert-model-io":
        storage = QwenModelIOConverter(args.source).convert(
            args.output,
            overwrite=args.overwrite,
            quantization=args.quantization,
        )
        print(json.dumps(storage, indent=2))
        return 0

    if args.command == "benchmark-logits":
        _configure_execution(args)
        token_ids = [
            int(value.strip())
            for value in args.token_ids.split(",")
            if value.strip()
        ]
        if not token_ids:
            raise ValueError("token-ids must contain at least one integer")
        model = QwenForCausalLM.from_model_directory(
            args.root, rows_per_chunk=args.rows_per_chunk
        )
        state = model.new_state()
        started = time.perf_counter()
        results = model.forward_ids(token_ids, state)
        elapsed = time.perf_counter() - started
        result = results[-1]
        top_token = result.greedy_token
        print(
            json.dumps(
                {
                    "input_tokens": token_ids,
                    "tokens": len(token_ids),
                    "seconds": elapsed,
                    "tokens_per_second": len(token_ids) / elapsed if elapsed else 0.0,
                    "state_tokens": state.tokens,
                    "vocab_size": model.vocab_size,
                    "top_token": top_token,
                    "top_logit": result.logits[top_token],
                    "logits_checksum": round(sum(result.logits), 6),
                    "execution": _execution_stats(),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "validate-transformers":
        _configure_execution(args)
        token_ids = [
            int(value.strip())
            for value in args.token_ids.split(",")
            if value.strip()
        ]
        model = QwenForCausalLM.from_model_directory(
            args.root, rows_per_chunk=args.rows_per_chunk
        )
        reference = TransformersReference(
            args.source,
            device=args.reference_device,
            dtype=args.reference_dtype,
            trust_remote_code=args.trust_remote_code,
            offload_dir=args.reference_offload_dir,
            max_gpu_memory_mib=args.reference_gpu_memory_mib,
            max_cpu_memory_mib=args.reference_cpu_memory_mib,
        )
        if args.component_layer is not None:
            if len(token_ids) != 1:
                raise ValueError("component validation requires one token ID")
            comparisons = diagnose_layer_components(
                model, reference, token_ids[0], args.component_layer
            )
            print(
                json.dumps(
                    {
                        "input_token": token_ids[0],
                        "layer": args.component_layer,
                        "comparisons": [asdict(item) for item in comparisons],
                    },
                    indent=2,
                )
            )
        elif args.layerwise:
            if len(token_ids) != 1:
                raise ValueError("layerwise validation requires one token ID")
            comparisons = diagnose_hidden_states(model, reference, token_ids[0])
            print(
                json.dumps(
                    {
                        "input_token": token_ids[0],
                        "comparisons": [asdict(item) for item in comparisons],
                    },
                    indent=2,
                )
            )
        else:
            report = validate_against_reference(
                model,
                reference,
                token_ids,
                generate_tokens=args.generate_tokens,
                top_k=args.top_k,
            )
            print(json.dumps(report.to_dict(), indent=2))
        return 0

    if args.command == "benchmark-decoder":
        if args.tokens <= 0:
            raise ValueError("tokens must be positive")
        decoder = QwenDecoderStack.from_model_directory(args.root)
        state = decoder.new_state()
        started = time.perf_counter()
        result = None
        for position in range(args.tokens):
            hidden = [
                math.sin((position + 1) * (index + 1) * 0.017)
                for index in range(decoder.hidden_size)
            ]
            result = decoder.forward_token(hidden, state)
        elapsed = time.perf_counter() - started
        assert result is not None
        print(
            json.dumps(
                {
                    "layers": len(decoder.layers),
                    "tokens": args.tokens,
                    "seconds": elapsed,
                    "tokens_per_second": args.tokens / elapsed if elapsed else 0.0,
                    "state_tokens": state.tokens,
                    "selected_experts": [
                        list(layer.selected_experts)
                        for layer in result.layer_results
                    ],
                    "output_checksum": round(sum(result.output), 6),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "convert-linear-layers":
        def linear_progress(completed: int, total: int) -> None:
            print(
                f"\rConverted linear layers: {completed}/{total}",
                end="",
                file=sys.stderr,
                flush=True,
            )

        storage = QwenGatedDeltaConverter(args.source).convert(
            args.output,
            overwrite=args.overwrite,
            quantization=args.quantization,
            progress=linear_progress,
        )
        if storage["layer_count"]:
            print(file=sys.stderr)
        print(json.dumps(storage, indent=2))
        return 0

    if args.command == "benchmark-linear-layer":
        if args.tokens <= 0:
            raise ValueError("tokens must be positive")
        layer = QwenGatedDeltaLayer.from_model_directory(args.root, args.layer)
        state = layer.new_state()
        started = time.perf_counter()
        result = None
        for position in range(args.tokens):
            hidden = [
                math.sin((position + 1) * (index + 1) * 0.017)
                for index in range(layer.hidden_size)
            ]
            result = layer.forward_residual(hidden, state)
        elapsed = time.perf_counter() - started
        assert result is not None
        print(
            json.dumps(
                {
                    "layer": args.layer,
                    "tokens": args.tokens,
                    "seconds": elapsed,
                    "tokens_per_second": args.tokens / elapsed if elapsed else 0.0,
                    "state_tokens": state.tokens,
                    "output_checksum": round(sum(result.output), 6),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "convert-attention-layers":
        def attention_progress(completed: int, total: int) -> None:
            print(
                f"\rConverted attention layers: {completed}/{total}",
                end="",
                file=sys.stderr,
                flush=True,
            )

        storage = QwenAttentionConverter(args.source).convert(
            args.output,
            overwrite=args.overwrite,
            quantization=args.quantization,
            progress=attention_progress,
        )
        if storage["layer_count"]:
            print(file=sys.stderr)
        print(json.dumps(storage, indent=2))
        return 0

    if args.command == "benchmark-attention-layer":
        if args.tokens <= 0:
            raise ValueError("tokens must be positive")
        layer = QwenFullAttentionLayer.from_model_directory(args.root, args.layer)
        cache = layer.new_cache()
        started = time.perf_counter()
        result = None
        for position in range(args.tokens):
            hidden = [
                math.sin((position + 1) * (index + 1) * 0.017)
                for index in range(layer.hidden_size)
            ]
            result = layer.forward_residual(hidden, position, cache)
        elapsed = time.perf_counter() - started
        assert result is not None
        print(
            json.dumps(
                {
                    "layer": args.layer,
                    "tokens": args.tokens,
                    "seconds": elapsed,
                    "tokens_per_second": args.tokens / elapsed if elapsed else 0.0,
                    "cache_length": cache.length,
                    "output_checksum": round(sum(result.output), 6),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "benchmark-expert":
        if args.iterations <= 0:
            raise ValueError("iterations must be positive")
        expert = Q4SwiGLUExpert.from_file(args.path)
        hidden = [math.sin(index * 0.17) for index in range(expert.hidden_size)]
        started = time.perf_counter()
        output = []
        for _ in range(args.iterations):
            output = expert.forward(hidden)
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "path": str(args.path.resolve()),
                    "engine": expert.engine,
                    "hidden_size": expert.hidden_size,
                    "intermediate_size": expert.intermediate_size,
                    "iterations": args.iterations,
                    "seconds": elapsed,
                    "experts_per_second": args.iterations / elapsed if elapsed else 0.0,
                    "output_checksum": round(sum(output), 6),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "create-demo":
        store = ExpertStore.create_demo(
            args.path,
            layers=args.layers,
            experts_per_layer=args.experts,
            width=args.width,
        )
        print(
            json.dumps(
                {
                    "path": str(store.root),
                    "layers": store.layers,
                    "experts_per_layer": store.experts_per_layer,
                    "width": store.width,
                    "expert_bytes": store.expert_byte_size(),
                },
                indent=2,
            )
        )
        return 0

    store = ExpertStore(args.path)
    cache = LayeredExpertCache(store.expert_byte_size() * args.cache_experts)
    predictor = TransitionPredictor()
    with ResidencyManager(store, cache) as residency:
        runtime = ToyMoERuntime(
            residency,
            predictor,
            top_k=args.top_k,
            prefetch_budget=args.prefetch,
        )
        if args.warmup:
            runtime.run(list(range(args.warmup)))
        started = time.perf_counter()
        outputs = runtime.run(list(range(args.warmup, args.warmup + args.tokens)))
        elapsed = time.perf_counter() - started
        report = {
            "tokens": args.tokens,
            "seconds": elapsed,
            "tokens_per_second": args.tokens / elapsed if elapsed else 0.0,
            "output_checksum": round(sum(sum(output) for output in outputs), 6),
            **residency.stats(),
        }
        print(json.dumps(report, indent=2))
    return 0


def _add_expert_preload_argument(
    parser: argparse.ArgumentParser, *, default: str
) -> None:
    parser.add_argument(
        "--expert-preload",
        choices=("none", "auto", "all"),
        default=default,
        help="preload routed experts into RAM before inference",
    )


def _add_cpu_moe_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cpu-moe-layers",
        type=int,
        default=0,
        help="run the first N MoE blocks on the native CPU backend",
    )


def _expert_preload_progress(completed: int, total: int) -> None:
    print(
        f"\rPreloaded experts: {completed}/{total}",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _add_device_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--gpu-cache-mib", type=int, default=8192)


def _configure_execution(args: argparse.Namespace) -> None:
    if args.device == "cuda":
        accelerator = configure_cuda(
            cache_mib=args.gpu_cache_mib,
            device_id=args.cuda_device,
            kv_cache_type=getattr(args, "kv_cache_type", "f32"),
        )
        print(
            f"CUDA enabled: {accelerator.device_name} "
            f"({args.gpu_cache_mib} MiB weight cache)",
            file=sys.stderr,
            flush=True,
        )
        return
    disable_cuda()


def _execution_stats() -> dict[str, int | str]:
    accelerator = active_cuda()
    stats = accelerator.stats() if accelerator is not None else {"device": "cpu"}
    native = active_native()
    if native is not None:
        stats["native_cpu_features"] = ",".join(native.features)
    return stats


if __name__ == "__main__":
    raise SystemExit(main())
