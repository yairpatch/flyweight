from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from .server import serve as serve_http
from .v2 import V2Model


EXPERT_MODE_CHOICES = (
    "cpu", "auto", "resident", "hybrid", "gpu",
    "legacy-paging", "legacy-hybrid",
)
KV_TYPES = ("auto", "f32", "f16", "bf16", "q8_0", "turbo3", "turbo4")


def _steady_state_counters(start, end):
    if start is None:
        return None
    fields = (
        "decode_calls", "decode_nanoseconds", "route_wait_nanoseconds",
        "expert_page_nanoseconds", "tail_wait_nanoseconds",
        "expert_compute_nanoseconds", "expert_cache_hits",
        "expert_cache_misses", "expert_cache_evictions",
    )
    # Opt-in probe (COLIBRI_ROUTE_RECURRENCE); absent from older runtimes and
    # all-zero when it is off, so it is reported only once it has samples.
    recurrence_fields = (
        "route_recurrence_observations", "route_recurrence_prev_hits",
        "route_recurrence_window_hits", "route_recurrence_layer_samples",
        "route_recurrence_window_experts", "route_recurrence_resident",
        "route_recurrence_miss_in_window", "route_recurrence_miss_cold",
    )
    delta = {field: end[field] - start[field] for field in fields}
    recurrence = {
        field: end.get(field, 0) - start.get(field, 0)
        for field in recurrence_fields
    }
    calls = delta["decode_calls"] or 1
    lookups = delta["expert_cache_hits"] + delta["expert_cache_misses"]
    routes = recurrence["route_recurrence_observations"]
    summary = {
        "decode_calls": delta["decode_calls"],
        "route_wait_ns_per_token": delta["route_wait_nanoseconds"] / calls,
        "expert_page_ns_per_token": delta["expert_page_nanoseconds"] / calls,
        "tail_wait_ns_per_token": delta["tail_wait_nanoseconds"] / calls,
        "expert_compute_ns_per_token": delta["expert_compute_nanoseconds"] / calls,
        "decode_ns_per_token": delta["decode_nanoseconds"] / calls,
        "expert_cache_hits": delta["expert_cache_hits"],
        "expert_cache_misses": delta["expert_cache_misses"],
        "expert_cache_evictions": delta["expert_cache_evictions"],
        "expert_cache_hit_rate": delta["expert_cache_hits"] / (lookups or 1),
    }
    if routes:
        summary.update({
            "route_recurrence_observations": routes,
            "route_recurrence_prev_rate":
                recurrence["route_recurrence_prev_hits"] / routes,
            "route_recurrence_window_rate":
                recurrence["route_recurrence_window_hits"] / routes,
            "route_recurrence_window_experts_per_layer":
                recurrence["route_recurrence_window_experts"] /
                (recurrence["route_recurrence_layer_samples"] or 1),
            "route_recurrence_resident_rate":
                recurrence["route_recurrence_resident"] / routes,
            "route_recurrence_miss_in_window_rate":
                recurrence["route_recurrence_miss_in_window"] / routes,
            "route_recurrence_miss_cold_rate":
                recurrence["route_recurrence_miss_cold"] / routes,
        })
    return summary


def _benchmark_native_prefill(runtime, prompt_tokens: list[int]) -> tuple[int, float]:
    first_tokens: list[int] = []
    started = time.perf_counter()
    runtime.generate(prompt_tokens, 1, first_tokens.append)
    elapsed = time.perf_counter() - started
    if len(first_tokens) != 1:
        raise RuntimeError("native prefill did not produce exactly one token")
    return first_tokens[0], elapsed


def _drop_file_cache(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("--cold-cache requires POSIX_FADV_DONTNEED support")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def _prefill_cache_seed(value: str) -> int | str:
    normalized = value.lower()
    if normalized in {"auto", "off"}:
        return normalized
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected auto, off, or an integer within [0, 256]"
        ) from error
    if not 0 <= count <= 256:
        raise argparse.ArgumentTypeError(
            "expected auto, off, or an integer within [0, 256]"
        )
    return count


def _add_runtime_options(
    parser: argparse.ArgumentParser, *, serving: bool, show_help: bool = True
) -> None:
    hidden = None if show_help else argparse.SUPPRESS
    parser.add_argument("--gpu-cache-mib", type=int, default=0, help=hidden)
    parser.add_argument(
        "--expert-mode", "--moe-device", dest="expert_mode",
        choices=EXPERT_MODE_CHOICES, default="auto", help=hidden,
    )
    parser.add_argument("--hybrid-prefill", choices=("split", "cpu"), default="split", help=hidden)
    parser.add_argument("--expert-residency", choices=("mutable", "immutable"), help=hidden)
    parser.add_argument(
        "--dense-requant", choices=("auto", "q8", "off"), default="auto",
        help=("BF16 dense-weight GPU policy (default: auto from GPU pressure)"
              if show_help else argparse.SUPPRESS),
    )
    parser.add_argument("--mtp-drafts", type=int, default=0, help=hidden)
    parser.add_argument(
        "--mtp-model", type=Path,
        help=(("optional MTP-only GGUF overlay; when omitted, use the draft "
               "head embedded in the target model")
              if show_help else argparse.SUPPRESS),
    )
    parser.add_argument("--cpu-threads", type=int, default=0, help=hidden)
    parser.add_argument("--cache-type-k", choices=KV_TYPES, default="f16", help=hidden)
    parser.add_argument("--cache-type-v", choices=KV_TYPES, default="f16", help=hidden)
    parser.add_argument("--parallel", type=int, default=1, dest="parallel_sequences", help=hidden)
    parser.add_argument("--prompt-cache-mib", type=int, default=0, help=hidden)
    parser.add_argument("--swa-full", action="store_true", help=hidden)
    parser.add_argument("--prefill-cache-seed", type=_prefill_cache_seed, default=None, help=hidden)
    parser.add_argument("--expert-paging", choices=("auto", "staged", "direct"), default="auto", help=hidden)
    parser.add_argument("--cpu-prefetch-mib", type=int, default=0, help=hidden)
    parser.add_argument("--cpu-prefetch-auto", action="store_true", help=hidden)
    parser.add_argument("--next-layer-prefetch", type=int, default=0, help=hidden)
    if serving:
        parser.add_argument("--prefill-checkpoint-interval", type=int, default=256, help=hidden)
        parser.add_argument("--prefill-checkpoint-slots", type=int, default=4, help=hidden)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colibri-next",
        description="Native GGUF inference for Qwen and Gemma models",
    )
    commands = parser.add_subparsers(
        dest="command", required=True,
        metavar="{serve,generate,inspect,benchmark}",
    )

    inspect = commands.add_parser(
        "inspect", aliases=("inspect-gguf", "inspect-gguf-v2"),
        help="show model metadata",
    )
    inspect.add_argument("model", type=Path)

    generate = commands.add_parser(
        "generate", aliases=("generate-text-v2",),
        help="generate one response locally",
    )
    generate.add_argument("model", type=Path)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--system")
    generate.add_argument(
        "--max-tokens", "--max-new-tokens", dest="max_new_tokens", type=int,
        default=64, help="maximum generated tokens (default: 64)",
    )
    generate.add_argument(
        "--context", "--context-window", dest="context_window", type=int,
        default=32768, help="maximum prompt + output tokens (default: 32768)",
    )
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-k", type=int, default=20)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--enable-thinking", action="store_true")
    generate.add_argument("--device", type=int, default=0, help=argparse.SUPPRESS)
    generate.add_argument(
        "--backend", choices=("auto", "cuda", "cpu"), default="auto",
        help="execution backend; auto uses CUDA when a device is present",
    )
    _add_runtime_options(generate, serving=True, show_help=False)

    benchmark = commands.add_parser("benchmark", aliases=("benchmark-v2",), help="measure local inference speed")
    benchmark.add_argument("model", type=Path)
    benchmark.add_argument("--tokens", default="0")
    benchmark.add_argument("--prompt")
    benchmark.add_argument("--chat", action="store_true")
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--iterations", type=int, default=10)
    benchmark.add_argument("--context", type=int, default=2048)
    benchmark.add_argument("--expert-top-k", type=int, default=0)
    benchmark.add_argument("--expert-top-p", type=float, default=0.0)
    benchmark.add_argument("--cold-cache", action="store_true")
    _add_runtime_options(benchmark, serving=False)

    probe = commands.add_parser(
        "probe", aliases=("probe-native", "probe-native-v2", "probe-qwen-native-v2"),
    )
    probe.add_argument("model", type=Path)
    probe.add_argument("--token-id", type=int, default=0)
    probe.add_argument("--prompt")
    probe.add_argument("--chat", action="store_true")
    probe.add_argument("--generate-tokens", type=int, default=2)
    probe.add_argument("--context", type=int, default=2048)
    _add_runtime_options(probe, serving=False)

    serve = commands.add_parser(
        "serve", aliases=("serve-v2",),
        help="start the OpenAI/Anthropic-compatible server",
    )
    serve.add_argument("model", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model-name")
    serve.add_argument("--device", type=int, default=0)
    # Separate from --device, which selects *which* GPU. This selects whether a
    # GPU is used at all.
    serve.add_argument(
        "--backend",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="execution backend; auto uses CUDA when a driver is present and "
             "falls back to the CPU backend otherwise",
    )
    serve.add_argument("--strict-model", action="store_true")
    serve.add_argument("--api-key", default=os.environ.get("COLIBRI_API_KEY"))
    serve.add_argument("--cors-origin", default="*")
    serve.add_argument(
        "--context", "--context-window", dest="context_window", type=int,
        default=32768, help="maximum prompt + output tokens (default: 32768)",
    )
    serve.add_argument(
        "--max-tokens", "--max-new-tokens", dest="max_new_tokens", type=int,
        default=4096, help="maximum generated tokens per request (default: 4096)",
    )
    serve.add_argument(
        "--concurrency", "--max-concurrent-requests",
        dest="max_concurrent_requests", type=int, default=64,
        help="maximum simultaneous inference requests (default: 64)",
    )
    serve.add_argument("--max-connections", type=int, default=128, help=argparse.SUPPRESS)
    serve.add_argument("--request-timeout-seconds", type=float, default=30.0, help=argparse.SUPPRESS)
    serve.add_argument("--sse-keepalive-seconds", type=float, default=10.0, help=argparse.SUPPRESS)
    _add_runtime_options(serve, serving=True, show_help=False)
    return parser


def _validate_runtime_args(args: argparse.Namespace) -> None:
    for name in ("gpu_cache_mib", "cpu_prefetch_mib", "cpu_threads", "prompt_cache_mib"):
        if getattr(args, name, 0) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if getattr(args, "parallel_sequences", 1) < 1:
        raise SystemExit("--parallel must be at least 1")
    if getattr(args, "max_concurrent_requests", 1) < 1:
        raise SystemExit("--max-concurrent-requests must be at least 1")
    if getattr(args, "max_connections", 1) < 1:
        raise SystemExit("--max-connections must be at least 1")
    if getattr(args, "request_timeout_seconds", 1.0) <= 0:
        raise SystemExit("--request-timeout-seconds must be positive")
    if getattr(args, "sse_keepalive_seconds", 1.0) <= 0:
        raise SystemExit("--sse-keepalive-seconds must be positive")
    if args.cpu_prefetch_mib and args.cpu_prefetch_auto:
        raise SystemExit("use either --cpu-prefetch-mib or --cpu-prefetch-auto")
    if not 0 <= args.next_layer_prefetch <= 64:
        raise SystemExit("--next-layer-prefetch must be within [0, 64]")


def _runtime_options(args: argparse.Namespace) -> dict[str, object]:
    names = (
        "device", "expert_mode", "mtp_drafts", "cache_type_k", "cache_type_v",
        "prefill_checkpoint_interval", "prefill_checkpoint_slots",
        "parallel_sequences", "prompt_cache_mib", "swa_full",
        "prefill_cache_seed", "expert_paging", "cpu_prefetch_mib",
        "cpu_prefetch_auto", "next_layer_prefetch", "cpu_threads",
        "hybrid_prefill", "expert_residency", "expert_top_k", "expert_top_p",
        "dense_requant",
    )
    options = {name: getattr(args, name) for name in names if hasattr(args, name)}
    options["gpu_cache_bytes"] = args.gpu_cache_mib * 1024**2
    return options


def _prompt_tokens(model: V2Model, args: argparse.Namespace) -> list[int]:
    if args.prompt is None:
        return [int(value) for value in getattr(args, "tokens", "0").split(",") if value.strip()]
    text = args.prompt
    if getattr(args, "chat", False):
        from .v2_server import NativeV2Tokenizer
        text = NativeV2Tokenizer(model).format_messages(
            [{"role": "user", "content": text}], enable_thinking=True,
        )
    return model.tokenize(text)


def _benchmark_native_generate(
    runtime: object, prompt: list[int], tokens: int,
) -> tuple[list[int], list[float], float]:
    """Time one complete generate call, retaining callback arrival times.

    MTP may commit several tokens in one verifier round, so timing repeated
    ``decode()`` calls bypasses the feature entirely. Callback intervals keep
    round boundaries visible without pretending each committed token was a
    separate native invocation.
    """
    generated: list[int] = []
    arrivals: list[float] = []
    started = time.perf_counter()

    def receive(token: int) -> None:
        generated.append(token)
        arrivals.append(time.perf_counter() - started)

    runtime.generate(prompt, tokens, receive)
    return generated, arrivals, time.perf_counter() - started


def _benchmark(args: argparse.Namespace) -> int:
    _validate_runtime_args(args)
    if args.iterations < 3 or args.warmup < 1 or args.context <= 0:
        raise SystemExit("benchmark requires warmup >= 1, iterations >= 3, and context > 0")
    if args.expert_top_k < 0 or not 0 <= args.expert_top_p <= 1:
        raise SystemExit("invalid expert routing limit")
    if args.cold_cache:
        _drop_file_cache(args.model)
    with V2Model(args.model, mtp_model=args.mtp_model) as model:
        prompt = _prompt_tokens(model, args)
        if not prompt:
            raise SystemExit("benchmark prompt must contain at least one token")
        if len(prompt) + args.warmup + args.iterations > args.context:
            raise SystemExit("benchmark exceeds --context")
        options = _runtime_options(args)
        options["context_limit"] = args.context
        with model.native_runtime(**options) as runtime:
            started = time.perf_counter()
            runtime.prepare()
            prepare_seconds = time.perf_counter() - started
            request_start = runtime.info
            all_generated, arrivals, total_seconds = _benchmark_native_generate(
                runtime, prompt, 1 + args.warmup + args.iterations,
            )
            info = runtime.info
        # Token zero is the pending result of prompt evaluation. Both paths
        # then warm and measure the same following output positions.
        generated = all_generated[1:]
        measured_start = arrivals[args.warmup]
        measured_total = arrivals[-1] - measured_start
        callback_intervals = [
            arrivals[index] - arrivals[index - 1]
            for index in range(args.warmup + 1, len(arrivals))
        ]
        mtp_suffix = "-mtp" if args.mtp_drafts else ""
        print(json.dumps({
            "execution": f"native-v2-{info['expert_mode']}{mtp_suffix}",
            "prepare_seconds": prepare_seconds,
            "prompt_tokens": len(prompt),
            "prompt_and_first_token_seconds": arrivals[0],
            "request_seconds": total_seconds,
            "decode_seconds": callback_intervals,
            "decode_median_seconds": statistics.median(callback_intervals),
            "decode_tokens_per_second": (
                args.iterations / measured_total if measured_total > 0 else 0.0
            ),
            "generated_tokens": generated,
            "generated_text": model.decode_tokens(generated),
            "request_counters": _steady_state_counters(request_start, info),
            "runtime": info,
        }, indent=2))
    return 0


# Knobs for Qwen-specific placement, paging, KV formats or drafting. DeepSeek-V4
# has its own CPU/hybrid placement and half-precision compressed state, so these
# are reported rather than accepted and ignored.
_DEEPSEEK4_UNSUPPORTED = (
    "gpu_cache_mib", "expert_mode", "hybrid_prefill",
    "expert_residency", "dense_requant", "mtp_drafts", "mtp_model",
    "cache_type_k", "cache_type_v", "prompt_cache_mib", "swa_full",
    "prefill_cache_seed", "expert_paging", "cpu_prefetch_mib",
    "cpu_prefetch_auto", "next_layer_prefetch", "cpu_threads",
    "prefill_checkpoint_interval", "prefill_checkpoint_slots",
)


def _architecture(model_path: Path) -> str | None:
    """The checkpoint's architecture, or None if it cannot be read.

    Used only to choose a service; a model that will not open is left to fail
    where it is loaded for real, which reports the reason.
    """
    from .v2 import V2Error
    try:
        with V2Model(model_path) as model:
            return str(model.config["architecture"])
    except (V2Error, OSError, KeyError):
        return None


def _deepseek4_service(args: argparse.Namespace, command: str):
    """Build the DeepSeek-V4 service, refusing options it cannot honour.

    The comparison is against this parser's own defaults, so a flag left alone
    passes and one the caller actually typed does not.
    """
    from .deepseek4_server import NativeDeepseek4InferenceService

    baseline = _parser().parse_args(
        [command, str(args.model)] + (["--prompt", ""] if command == "generate" else [])
    )
    requested = [
        name for name in _DEEPSEEK4_UNSUPPORTED
        if getattr(args, name, None) != getattr(baseline, name, None)
    ]
    if requested:
        raise SystemExit(
            "the DeepSeek-V4 runtime does not support "
            + ", ".join("--" + name.replace("_", "-") for name in sorted(requested))
            + " yet; it uses its dedicated CPU/hybrid runtime with half-precision caches"
        )
    # The dense half goes to the GPU when there is one and nothing said
    # otherwise; the routed experts stay on the CPU whatever happens, because
    # they are 90 GiB against 12 of VRAM.
    backend = getattr(args, "backend", "auto")
    device = None
    if backend != "cpu":
        from .v2 import V2Model as _V2Model
        try:
            available = bool(_V2Model.gpu_info()["available"])
        except Exception:
            available = False
        if available:
            device = int(getattr(args, "device", 0) or 0)
        elif backend == "cuda":
            raise SystemExit("no CUDA device is available")
    return NativeDeepseek4InferenceService(
        args.model,
        device=device,
        model_name=getattr(args, "model_name", None),
        context_window=args.context_window,
        max_new_tokens=args.max_new_tokens,
        parallel_sequences=args.parallel_sequences,
        api_key=getattr(args, "api_key", None),
        cors_origin=getattr(args, "cors_origin", "*"),
        strict_model=getattr(args, "strict_model", False),
        max_concurrent_requests=getattr(args, "max_concurrent_requests", 64),
        request_timeout_seconds=getattr(args, "request_timeout_seconds", 30.0),
        sse_keepalive_seconds=getattr(args, "sse_keepalive_seconds", 10.0),
    )


def _generate(args: argparse.Namespace) -> int:
    _validate_runtime_args(args)
    if _architecture(args.model) == "deepseek4":
        service = _deepseek4_service(args, "generate")
    else:
        from .v2_server import NativeV2InferenceService
        service = NativeV2InferenceService(
            args.model,
            mtp_model_path=args.mtp_model,
            context_window=args.context_window,
            max_new_tokens=args.max_new_tokens,
            gpu_cache_mib=args.gpu_cache_mib,
            **{key: value for key, value in _runtime_options(args).items()
               if key not in {"gpu_cache_bytes", "device", "expert_top_k", "expert_top_p"}},  # type: ignore[arg-type]
        )
    try:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        response = service.chat_completion({
            "model": service.model_name,
            "messages": messages,
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
            "enable_thinking": args.enable_thinking,
        })
        print(response["choices"][0]["message"]["content"])
    finally:
        service.close()
    return 0


def _serve(args: argparse.Namespace) -> int:
    _validate_runtime_args(args)
    from .v2 import V2Model
    if _architecture(args.model) == "deepseek4":
        return _serve_http(args, _deepseek4_service(args, "serve"))
    # Before the service builds a runtime: allocations belong to whichever
    # backend was active when they were made.
    selected = V2Model.select_backend(getattr(args, "backend", "auto"))
    if selected == "cpu":
        print(
            "[colibri] running on the CPU backend; decode will be far slower "
            "than a GPU",
            file=sys.stderr,
        )
    from .v2_server import NativeV2InferenceService
    service = NativeV2InferenceService(
        args.model,
        mtp_model_path=args.mtp_model,
        model_name=args.model_name,
        device=args.device,
        context_window=args.context_window,
        max_new_tokens=args.max_new_tokens,
        gpu_cache_mib=args.gpu_cache_mib,
        expert_mode=args.expert_mode,
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
        cpu_prefetch_auto=args.cpu_prefetch_auto,
        next_layer_prefetch=args.next_layer_prefetch,
        cpu_threads=args.cpu_threads,
        hybrid_prefill=args.hybrid_prefill,
        expert_residency=args.expert_residency,
        dense_requant=args.dense_requant,
        api_key=args.api_key,
        cors_origin=args.cors_origin,
        strict_model=args.strict_model,
        max_concurrent_requests=args.max_concurrent_requests,
        request_timeout_seconds=args.request_timeout_seconds,
        sse_keepalive_seconds=args.sse_keepalive_seconds,
    )
    return _serve_http(args, service)


def _serve_http(args: argparse.Namespace, service) -> int:
    try:
        print(f"Serving {service.model_name} at http://{args.host}:{args.port}", file=sys.stderr)
        serve_http(
            service,
            host=args.host,
            port=args.port,
            max_connections=args.max_connections,
        )
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"inspect", "inspect-gguf-v2", "inspect-gguf"}:
        with V2Model(args.model) as model:
            print(json.dumps({"model": model.info, "config": model.config,
                              "tensors": list(model.tensors())}, indent=2))
        return 0
    if args.command in {"benchmark-v2", "benchmark"}:
        return _benchmark(args)
    if args.command in {"generate", "generate-text-v2"}:
        return _generate(args)
    if args.command in {"serve-v2", "serve"}:
        return _serve(args)
    if args.command in {"probe", "probe-native-v2", "probe-qwen-native-v2", "probe-native"}:
        _validate_runtime_args(args)
        with V2Model(args.model, mtp_model=args.mtp_model) as model:
            prompt = _prompt_tokens(model, args)
            options = _runtime_options(args)
            options["context_limit"] = args.context
            with model.native_runtime(**options) as runtime:
                runtime.prepare()
                output: list[int] = []
                runtime.generate(prompt, args.generate_tokens, output.append)
                info = runtime.info
            print(json.dumps({"generated_tokens": output,
                              "generated_text": model.decode_tokens(output),
                              "runtime": info}, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
