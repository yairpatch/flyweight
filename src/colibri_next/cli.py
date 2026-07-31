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
KV_TYPES = ("f32", "f16", "bf16", "q8_0", "turbo3", "turbo4")


def _steady_state_counters(start, end):
    if start is None:
        return None
    fields = (
        "decode_calls", "decode_nanoseconds", "route_wait_nanoseconds",
        "expert_page_nanoseconds", "tail_wait_nanoseconds",
        "expert_compute_nanoseconds", "expert_cache_hits",
        "expert_cache_misses", "expert_cache_evictions",
    )
    delta = {field: end[field] - start[field] for field in fields}
    calls = delta["decode_calls"] or 1
    lookups = delta["expert_cache_hits"] + delta["expert_cache_misses"]
    return {
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


def _add_runtime_options(parser: argparse.ArgumentParser, *, serving: bool) -> None:
    parser.add_argument("--gpu-cache-mib", type=int, default=0)
    parser.add_argument(
        "--expert-mode", "--moe-device", dest="expert_mode",
        choices=EXPERT_MODE_CHOICES, default="auto",
    )
    parser.add_argument("--hybrid-prefill", choices=("split", "cpu"), default="split")
    parser.add_argument("--expert-residency", choices=("mutable", "immutable"))
    parser.add_argument("--mtp-drafts", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--cache-type-k", choices=KV_TYPES, default="f16")
    parser.add_argument("--cache-type-v", choices=KV_TYPES, default="f16")
    parser.add_argument("--parallel", type=int, default=1, dest="parallel_sequences")
    parser.add_argument("--prompt-cache-mib", type=int, default=0)
    parser.add_argument("--swa-full", action="store_true")
    parser.add_argument("--prefill-cache-seed", type=_prefill_cache_seed, default=None)
    parser.add_argument("--expert-paging", choices=("auto", "staged", "direct"), default="auto")
    parser.add_argument("--cpu-prefetch-mib", type=int, default=0)
    parser.add_argument("--cpu-prefetch-auto", action="store_true")
    parser.add_argument("--next-layer-prefetch", type=int, default=0)
    if serving:
        parser.add_argument("--prefill-checkpoint-interval", type=int, default=256)
        parser.add_argument("--prefill-checkpoint-slots", type=int, default=4)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colibri-next",
        description="Native GGUF inference for Qwen and Gemma models",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect-gguf-v2", aliases=("inspect-gguf",))
    inspect.add_argument("model", type=Path)

    generate = commands.add_parser("generate", aliases=("generate-text-v2",))
    generate.add_argument("model", type=Path)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--system")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--context-window", type=int, default=32768)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-k", type=int, default=20)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--enable-thinking", action="store_true")
    _add_runtime_options(generate, serving=True)

    benchmark = commands.add_parser("benchmark-v2", aliases=("benchmark",))
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
        "probe-native-v2", aliases=("probe-qwen-native-v2", "probe-native"),
    )
    probe.add_argument("model", type=Path)
    probe.add_argument("--token-id", type=int, default=0)
    probe.add_argument("--prompt")
    probe.add_argument("--chat", action="store_true")
    probe.add_argument("--generate-tokens", type=int, default=2)
    probe.add_argument("--context", type=int, default=2048)
    _add_runtime_options(probe, serving=False)

    serve = commands.add_parser("serve-v2", aliases=("serve",))
    serve.add_argument("model", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model-name")
    serve.add_argument("--device", type=int, default=0)
    serve.add_argument("--strict-model", action="store_true")
    serve.add_argument("--api-key", default=os.environ.get("COLIBRI_API_KEY"))
    serve.add_argument("--cors-origin", default="*")
    serve.add_argument("--context-window", type=int, default=32768)
    serve.add_argument("--max-new-tokens", type=int, default=4096)
    _add_runtime_options(serve, serving=True)
    return parser


def _validate_runtime_args(args: argparse.Namespace) -> None:
    for name in ("gpu_cache_mib", "cpu_prefetch_mib", "cpu_threads", "prompt_cache_mib"):
        if getattr(args, name, 0) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if getattr(args, "parallel_sequences", 1) < 1:
        raise SystemExit("--parallel must be at least 1")
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


def _benchmark(args: argparse.Namespace) -> int:
    _validate_runtime_args(args)
    if args.iterations < 3 or args.warmup < 1 or args.context <= 0:
        raise SystemExit("benchmark requires warmup >= 1, iterations >= 3, and context > 0")
    if args.expert_top_k < 0 or not 0 <= args.expert_top_p <= 1:
        raise SystemExit("invalid expert routing limit")
    if args.cold_cache:
        _drop_file_cache(args.model)
    with V2Model(args.model) as model:
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
            current, prompt_seconds = _benchmark_native_prefill(runtime, prompt)
            generated: list[int] = []
            warmup_seconds: list[float] = []
            measured_seconds: list[float] = []
            steady_start = None
            for index in range(args.warmup + args.iterations):
                if index == args.warmup:
                    steady_start = runtime.info
                started = time.perf_counter()
                current = runtime.decode(current)
                elapsed = time.perf_counter() - started
                generated.append(current)
                (warmup_seconds if index < args.warmup else measured_seconds).append(elapsed)
            info = runtime.info
        measured_total = sum(measured_seconds)
        print(json.dumps({
            "execution": f"native-v2-{info['expert_mode']}",
            "prepare_seconds": prepare_seconds,
            "prompt_tokens": len(prompt),
            "prompt_seconds": prompt_seconds,
            "prompt_tokens_per_second": len(prompt) / prompt_seconds,
            "first_warm_decode_seconds": warmup_seconds[0],
            "decode_seconds": measured_seconds,
            "decode_median_seconds": statistics.median(measured_seconds),
            "decode_tokens_per_second": args.iterations / measured_total,
            "generated_tokens": generated,
            "generated_text": model.decode_tokens(generated),
            "steady_state": _steady_state_counters(steady_start, info),
            "runtime": info,
        }, indent=2))
    return 0


def _generate(args: argparse.Namespace) -> int:
    _validate_runtime_args(args)
    from .v2_server import NativeV2InferenceService
    service = NativeV2InferenceService(
        args.model,
        context_window=args.context_window,
        max_new_tokens=args.max_new_tokens,
        gpu_cache_mib=args.gpu_cache_mib,
        **{key: value for key, value in _runtime_options(args).items()
           if key not in {"gpu_cache_bytes", "device", "expert_top_k", "expert_top_p"}},
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
    from .v2_server import NativeV2InferenceService
    service = NativeV2InferenceService(
        args.model,
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
        api_key=args.api_key,
        cors_origin=args.cors_origin,
        strict_model=args.strict_model,
    )
    try:
        print(f"Serving {service.model_name} at http://{args.host}:{args.port}", file=sys.stderr)
        serve_http(service, host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"inspect-gguf-v2", "inspect-gguf"}:
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
    if args.command in {"probe-native-v2", "probe-qwen-native-v2", "probe-native"}:
        _validate_runtime_args(args)
        with V2Model(args.model) as model:
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
