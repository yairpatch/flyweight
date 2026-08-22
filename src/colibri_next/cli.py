from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from .server import serve as serve_http
from .v2 import AUTO_PROMPT_CACHE_MIB as _AUTO_PROMPT_CACHE_MIB, V2Model


EXPERT_MODE_CHOICES = (
    "cpu", "auto", "resident", "hybrid", "gpu",
    "legacy-paging", "legacy-hybrid",
)
KV_TYPES = ("auto", "f32", "f16", "bf16", "q8_0", "turbo3", "turbo4")
AUTO_PROMPT_CACHE_MIB = _AUTO_PROMPT_CACHE_MIB
# Offered when a safetensors checkpoint is opened, smallest first. GGUF models
# carry their own quantization and are never asked about.
QUANT_CHOICES = ("ask", "IQ2_XS", "Q2_K", "IQ3_XXS", "Q3_K", "IQ4_XS", "Q4_K",
                 "Q5_K", "Q6_K",
                 "Q8_0", "F32")
# What the loader packs when nothing says otherwise, and so what the prompt
# offers as its default.
DEFAULT_QUANT = "Q6_K"


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


def _prompt_cache_budget(value: str) -> int:
    normalized = value.lower()
    if normalized == "auto":
        return AUTO_PROMPT_CACHE_MIB
    if normalized == "off":
        return 0
    try:
        budget = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected auto, off, or a non-negative MiB budget"
        ) from error
    if not 0 <= budget < AUTO_PROMPT_CACHE_MIB:
        raise argparse.ArgumentTypeError(
            "expected auto, off, or a non-negative MiB budget"
        )
    return budget


def _add_quant_option(parser: argparse.ArgumentParser) -> None:
    """The quantization a safetensors checkpoint is packed to on first open."""
    parser.add_argument(
        "--quant", choices=QUANT_CHOICES, default=None,
        help="quantization for a safetensors checkpoint; 'ask' prompts, and is "
             f"the default on a terminal (otherwise {DEFAULT_QUANT})",
    )
    parser.add_argument(
        "--imatrix", type=Path, default=None,
        help="importance matrix (llama.cpp imatrix.dat) that weights IQ "
             "packing; an imatrix.dat beside the checkpoint is used "
             "automatically, and 'off' disables that",
    )


def _format_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{value:,.0f} B"
        value /= 1024.0
    raise AssertionError("unreachable")


def _quant_menu(options: list[dict[str, object]], default: str) -> str:
    """The menu text. Separate from the asking so it can be tested as a value."""
    lines = []
    for index, option in enumerate(options, start=1):
        name = str(option["name"])
        # An option this checkpoint cannot use keeps its number, so the numbering
        # does not depend on the model and a habit does not misfire.
        if str(option.get("unavailable", "")):
            lines.append(f"  {index}) {name:<7} {'--':>11}"
                         f"   unavailable: {option['unavailable']}")
            continue
        cached = int(option["cache_bytes"]) > 0
        note = ("cached, opens immediately" if cached
                else f"packs on first open, writes {_format_bytes(int(option['arena_bytes']))}")
        mark = "  [default]" if name == default else ""
        lines.append(
            f"  {index}) {name:<7} {_format_bytes(int(option['arena_bytes'])):>11}"
            f"   {note}{mark}"
        )
    return "\n".join(lines)


def _quant_from_answer(
    answer: str, options: list[dict[str, object]], default: str
) -> str | None:
    """A menu answer resolved to a quantization name, or None if it is not one.

    Accepts the number, the name in any case, and the name with the underscore
    left out -- "q4k" is what a hurried typist writes.
    """
    text = answer.strip()
    if not text:
        return default
    # An unavailable option is not an answer: picking it would only fail at
    # load, and the menu already says why.
    names = [str(option["name"]) for option in options
             if not str(option.get("unavailable", ""))]
    numbered = [str(option["name"]) for option in options]
    if text.isdigit():
        index = int(text)
        chosen = numbered[index - 1] if 1 <= index <= len(numbered) else None
        return chosen if chosen in names else None
    folded = text.upper().replace("_", "").replace("-", "")
    for name in names:
        if folded == name.upper().replace("_", ""):
            return name
    return None


def _unavailable_reason(text: str, options: list[dict[str, object]]) -> str:
    """Why `text` named an option that cannot be chosen, or empty."""
    text = text.strip()
    selected: dict[str, object] | None = None
    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(options):
            selected = options[index - 1]
    else:
        folded = text.upper().replace("_", "").replace("-", "")
        for option in options:
            if folded == str(option["name"]).upper().replace("_", ""):
                selected = option
                break
    if selected is None:
        return ""
    reason = str(selected.get("unavailable", ""))
    return f"{selected['name']} is unavailable here: {reason}" if reason else ""


def _resolve_quant(args: argparse.Namespace) -> None:
    """Settle the quantization before anything opens the model.

    The loader reads COLIBRI_HF_QUANT, so a choice made here is published as
    that variable -- which also means an explicitly exported one wins over the
    prompt, since the caller has already answered the question.

    Prompting is for a person at a terminal. A pipe, a service manager or CI
    gets the loader's default, which is what it got before this existed.
    """
    # Published the same way the quantization is: the loader reads the
    # variable, and an explicitly exported one wins over the flag.
    imatrix = getattr(args, "imatrix", None)
    if imatrix is not None and "COLIBRI_HF_IMATRIX" not in os.environ:
        os.environ["COLIBRI_HF_IMATRIX"] = str(imatrix)

    requested = getattr(args, "quant", None)
    if requested and requested != "ask":
        os.environ["COLIBRI_HF_QUANT"] = requested
        return
    if requested is None:
        if os.environ.get("COLIBRI_HF_QUANT"):
            return
        if not (sys.stdin.isatty() and sys.stderr.isatty()):
            return
    model = getattr(args, "model", None)
    if model is None or not Path(model).is_dir():
        return

    from .v2 import V2Error

    try:
        options = V2Model.hf_quant_options(model)
    except (V2Error, OSError):
        # Not a checkpoint this runtime can describe -- a GGUF, an unsupported
        # architecture, an unreadable directory. Whatever it is, the load path
        # reports it far better than a menu could, so leave it to fail there.
        return
    if not options:
        return

    print(f"\n{Path(model).name} is a safetensors checkpoint. "
          "Choose how to quantize it:", file=sys.stderr)
    print(_quant_menu(options, DEFAULT_QUANT), file=sys.stderr)
    for _ in range(3):
        try:
            # The question goes to stderr rather than through input()'s own
            # prompt, which writes to stdout -- `inspect` prints JSON there.
            print(f"quantization [{DEFAULT_QUANT}]: ", end="", flush=True,
                  file=sys.stderr)
            answer = input()
        except EOFError:
            # stdin closed under us. Answering for the caller is better than
            # failing here; the loader's own default is the answer.
            print(file=sys.stderr)
            break
        except KeyboardInterrupt:
            # Interrupting a question means stop, not "pick the default".
            print(file=sys.stderr)
            raise SystemExit(130)
        chosen = _quant_from_answer(answer, options, DEFAULT_QUANT)
        if chosen:
            os.environ["COLIBRI_HF_QUANT"] = chosen
            return
        # Picking an unavailable option deserves its reason, not the generic
        # rejection: "not one of the options" against a menu that plainly
        # shows the number reads as a broken prompt.
        unavailable = _unavailable_reason(answer, options)
        if unavailable:
            print(unavailable, file=sys.stderr)
            continue
        print(f"not one of the options; enter 1-{len(options)} or a name",
              file=sys.stderr)
    print(f"using {DEFAULT_QUANT}", file=sys.stderr)
    os.environ["COLIBRI_HF_QUANT"] = DEFAULT_QUANT


def _add_runtime_options(
    parser: argparse.ArgumentParser, *, serving: bool, show_help: bool = True,
    cache_default: int = 0, show_cache_help: bool = False,
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
    parser.add_argument(
        "--cache", "--prompt-cache-mib", dest="prompt_cache_mib",
        type=_prompt_cache_budget, default=cache_default,
        metavar="{auto,off,MIB}",
        help=("host prompt cache budget: auto, off, or MiB (default: auto)"
              if show_help or show_cache_help else argparse.SUPPRESS),
    )
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
        description="Native GGUF inference for Qwen, Laguna, Muse, DeepSeek-V4 and Gemma models",
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
    _add_quant_option(inspect)

    generate = commands.add_parser(
        "generate", aliases=("generate-text-v2",),
        help="generate one response locally",
    )
    generate.add_argument("model", type=Path)
    _add_quant_option(generate)
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
    _add_quant_option(benchmark)
    benchmark.add_argument("--tokens", default="0")
    benchmark.add_argument("--prompt")
    benchmark.add_argument("--chat", action="store_true")
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument(
        "--iterations", type=int, default=128,
        help="measured decode tokens after warmup (default: 128)",
    )
    benchmark.add_argument("--context", type=int, default=2048)
    benchmark.add_argument("--expert-top-k", type=int, default=0)
    benchmark.add_argument("--expert-top-p", type=float, default=0.0)
    benchmark.add_argument("--temperature", type=float, default=0.0)
    benchmark.add_argument("--top-k", type=int, default=0)
    benchmark.add_argument("--top-p", type=float, default=0.0)
    benchmark.add_argument("--cold-cache", action="store_true")
    _add_runtime_options(benchmark, serving=False)

    probe = commands.add_parser(
        "probe", aliases=("probe-native", "probe-native-v2", "probe-qwen-native-v2"),
    )
    probe.add_argument("model", type=Path)
    _add_quant_option(probe)
    probe.add_argument("--token-id", type=int, default=0)
    probe.add_argument("--prompt")
    probe.add_argument("--chat", action="store_true")
    probe.add_argument("--generate-tokens", type=int, default=2)
    probe.add_argument("--context", type=int, default=2048)
    _add_runtime_options(probe, serving=False)

    imatrix = commands.add_parser(
        "imatrix",
        help="gather an importance matrix over calibration text",
    )
    imatrix.add_argument("model", type=Path)
    _add_quant_option(imatrix)
    imatrix.add_argument(
        "--text", type=Path, required=True,
        help="calibration text; general prose plus some code is the usual mix",
    )
    imatrix.add_argument(
        "--output", type=Path, default=Path("imatrix.dat"),
        help="where to write the matrix (llama.cpp legacy .dat layout)",
    )
    imatrix.add_argument(
        "--chunk", type=int, default=512,
        help="tokens per prefill chunk (default: 512)",
    )
    imatrix.add_argument(
        "--max-chunks", type=int, default=0,
        help="cap on chunks processed; 0 means the whole file",
    )
    imatrix.add_argument("--context", type=int, default=2048)
    _add_runtime_options(imatrix, serving=False, show_help=False)

    serve = commands.add_parser(
        "serve", aliases=("serve-v2",),
        help="start the OpenAI/Anthropic-compatible server",
    )
    serve.add_argument("model", type=Path)
    _add_quant_option(serve)
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
    serve.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high", "xhigh"),
        default=None,
        help="default reasoning effort for checkpoints that grade it "
             "(Qwen3.5 reads low/medium/xhigh and defaults to xhigh, its "
             "maximum); a request may override it per call",
    )
    # Server-wide sampling defaults, one flag per setting in sampling.SETTINGS
    # so this list cannot fall behind the sampler. Each applies when a request
    # does not carry its own value, and an absent flag changes nothing, giving
    # request > flag > generation_config.json beside the model > built-in.
    #
    # The penalties matter here because they are not a nicety on a low-bit
    # checkpoint -- they are what keeps it from looping, and they apply to
    # greedy decode, which is the default. A client that cannot send extra body
    # fields, and several coding harnesses cannot, has no other way to set them.
    from .sampling import SERVER_SETTINGS
    for setting in SERVER_SETTINGS:
        built_in = _sampling_defaults().get(setting.name)
        serve.add_argument(
            f"--{setting.name.replace('_', '-')}",
            type=setting.kind, default=None,
            help=f"default {setting.help}"
                 + (f" (default: {built_in})" if built_in is not None else ""),
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
    serve.add_argument("--max-tool-call-tokens", type=int, default=0, help=argparse.SUPPRESS)
    _add_runtime_options(
        serve, serving=True, show_help=False,
        cache_default=AUTO_PROMPT_CACHE_MIB, show_cache_help=True,
    )
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
    if getattr(args, "max_tool_call_tokens", 0) < 0:
        raise SystemExit("--max-tool-call-tokens must be non-negative")
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


def _benchmark_bailing_generate(runtime, prompt, tokens, config):
    """Time the same eval/sample loop used by the Bailing HTTP generator."""
    generated: list[int] = []
    arrivals: list[float] = []
    started = time.perf_counter()
    step = prompt
    runtime.reset()
    for _ in range(tokens):
        runtime.eval_into(step)
        token = runtime.sample(config)
        generated.append(token)
        arrivals.append(time.perf_counter() - started)
        step = [token]
    return generated, arrivals, time.perf_counter() - started


def _benchmark(args: argparse.Namespace) -> int:
    _validate_runtime_args(args)
    if args.iterations < 3 or args.warmup < 1 or args.context <= 0:
        raise SystemExit("benchmark requires warmup >= 1, iterations >= 3, and context > 0")
    if args.expert_top_k < 0 or not 0 <= args.expert_top_p <= 1:
        raise SystemExit("invalid expert routing limit")
    if args.temperature < 0 or args.top_k < 0 or not 0 <= args.top_p <= 1:
        raise SystemExit("invalid sampling options")
    if args.cold_cache:
        _drop_file_cache(args.model)
    with V2Model(args.model, mtp_model=args.mtp_model) as model:
        prompt = _prompt_tokens(model, args)
        if not prompt:
            raise SystemExit("benchmark prompt must contain at least one token")
        if len(prompt) + args.warmup + args.iterations > args.context:
            raise SystemExit("benchmark exceeds --context")
        if str(model.config["architecture"]) == "bailingmoe3":
            from .v2 import BailingRuntime

            started = time.perf_counter()
            runtime = BailingRuntime(model, capacity=args.context)
            prepare_seconds = time.perf_counter() - started
            try:
                sampling = SimpleNamespace(
                    temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, seed=None,
                )
                all_generated, arrivals, total_seconds = _benchmark_bailing_generate(
                    runtime, prompt, 1 + args.warmup + args.iterations, sampling,
                )
            finally:
                runtime.close()
            info = {"expert_mode": "bailing-gpu-or-host"}
            request_counters = None
        else:
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
            request_counters = _steady_state_counters(request_start, info)
        # Token zero is the pending result of prompt evaluation. Both paths
        # then warm and measure the same following output positions.
        generated = all_generated[1:]
        measured_start = arrivals[args.warmup]
        measured_total = arrivals[-1] - measured_start
        callback_intervals = [
            arrivals[index] - arrivals[index - 1]
            for index in range(args.warmup + 1, len(arrivals))
        ]
        # Short decode windows can materially overstate sustained server speed.
        # Expose the tail and latency distribution so decay is not hidden in a
        # single optimistic aggregate.
        sorted_intervals = sorted(callback_intervals)
        def percentile(fraction: float) -> float:
            return sorted_intervals[min(
                len(sorted_intervals) - 1,
                int((len(sorted_intervals) - 1) * fraction),
            )]
        window = min(32, max(1, len(callback_intervals) // 2))
        first_window = callback_intervals[:window]
        last_window = callback_intervals[-window:]
        mtp_suffix = "-mtp" if args.mtp_drafts else ""
        print(json.dumps({
            "execution": f"native-v2-{info['expert_mode']}{mtp_suffix}",
            "prepare_seconds": prepare_seconds,
            "prompt_tokens": len(prompt),
            "prompt_and_first_token_seconds": arrivals[0],
            "request_seconds": total_seconds,
            "decode_seconds": callback_intervals,
            "decode_median_seconds": statistics.median(callback_intervals),
            "decode_p90_seconds": percentile(0.90),
            "decode_p99_seconds": percentile(0.99),
            "decode_tokens_per_second": (
                args.iterations / measured_total if measured_total > 0 else 0.0
            ),
            "decode_first_window_tokens_per_second": (
                len(first_window) / sum(first_window) if sum(first_window) > 0 else 0.0
            ),
            "decode_last_window_tokens_per_second": (
                len(last_window) / sum(last_window) if sum(last_window) > 0 else 0.0
            ),
            "generated_tokens": generated,
            "generated_text": model.decode_tokens(generated),
            "request_counters": request_counters,
            "runtime": info,
        }, indent=2))
    return 0


# Knobs for Qwen-specific placement, paging, KV formats or drafting. DeepSeek-V4
# has its own CPU/hybrid placement and half-precision compressed state, so these
# are reported rather than accepted and ignored.
_DEEPSEEK4_UNSUPPORTED = (
    "gpu_cache_mib", "expert_mode", "hybrid_prefill",
    "expert_residency", "dense_requant",
    "cache_type_k", "cache_type_v", "prompt_cache_mib", "swa_full",
    "prefill_cache_seed", "expert_paging", "cpu_prefetch_mib",
    "cpu_prefetch_auto", "next_layer_prefetch", "cpu_threads",
    "prefill_checkpoint_interval", "prefill_checkpoint_slots",
)


def _stop_tokens(model: V2Model) -> set[int]:
    """The ids that end a turn, as the runtime's raw generate loop will not.

    `V2QwenRuntime.generate` has no stop set of its own; without this a probe
    keeps decoding past the model's end token and prints an invented next turn.
    """
    config = model.config
    unset = 0xFFFFFFFF
    stops = set()
    for key in ("eos_token_id", "eot_token_id"):
        value = config.get(key)
        if isinstance(value, int) and value != unset:
            stops.add(value)
    return stops


def _architecture(model_path: Path) -> str | None:
    """The checkpoint's architecture, or None if it cannot be read.

    Used only to choose a service; a model that will not open is left to fail
    where it is loaded for real, which reports the reason.
    """
    # HF directories already expose their architecture in a tiny config file.
    # Opening V2Model here mapped/parsed the multi-gigabyte quantized arena and
    # the 12 MB tokenizer, only to close both and repeat the work in the service.
    if model_path.is_dir():
        try:
            config = json.loads((model_path / "config.json").read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        model_type = config.get("model_type")
        architectures = config.get("architectures") or ()
        if model_type in {"bailing_hybrid", "bailingmoe3", "bailing-hybrid"} or (
            "BailingMoeV3ForCausalLM" in architectures
        ):
            return "bailingmoe3"
        return model_type if isinstance(model_type, str) else None

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
        dspark_model_path=getattr(args, "mtp_model", None),
        dspark_drafts=getattr(args, "mtp_drafts", 0),
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
        max_tool_call_tokens=getattr(args, "max_tool_call_tokens", 0),
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


def _sampling_defaults() -> dict[str, float | int]:
    from .sampling import defaults
    return defaults()


def _sampling_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    """The sampling flags the caller actually passed.

    Only the ones given: an absent flag must leave whatever the checkpoint's own
    generation_config.json says, rather than overwrite it with an argparse
    default that the caller never chose.
    """
    from .sampling import SERVER_SETTINGS, from_values
    given = {
        setting.name: getattr(args, setting.name)
        for setting in SERVER_SETTINGS
        if getattr(args, setting.name, None) is not None
    }
    # Check the ranges here rather than letting the service do it: the service
    # validates after building the runtime, so a typo would cost a full model
    # load before saying so.
    try:
        from_values(given)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"invalid sampling default: {error}") from error
    return given


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
        max_tool_call_tokens=args.max_tool_call_tokens,
        reasoning_effort=getattr(args, "reasoning_effort", None),
        generation_defaults=_sampling_overrides(args),
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


def _gather_imatrix(args: argparse.Namespace) -> int:
    """Prefill calibration text and write the accumulated importance matrix.

    The capture hooks cover the dense prefill path and the CPU expert path,
    so the run pins routed experts to the CPU and switches the streamed GPU
    expert GEMM off -- a matrix that silently missed every routed layer would
    be worse than an error. Decode contributes nothing: each chunk runs as
    prefill plus a single discarded token, which is also how llama.cpp
    gathers its matrices.
    """
    import struct

    _validate_runtime_args(args)
    os.environ["COLIBRI_IMATRIX"] = "1"
    os.environ["COLIBRI_PREFILL_EXPERT_STREAM_MIB"] = "0"
    args.expert_mode = "cpu"
    args.mtp_drafts = 0

    text = args.text.read_text(encoding="utf-8")
    if not text.strip():
        print("calibration text is empty", file=sys.stderr)
        return 2
    chunk = max(32, int(args.chunk))
    with V2Model(args.model) as model:
        tokens = model.tokenize(text, capacity=len(text.encode()) + 16)
        pieces = [tokens[i:i + chunk] for i in range(0, len(tokens), chunk)]
        # A short tail skews the statistics of every channel it touches
        # without adding coverage; llama.cpp drops it too.
        pieces = [piece for piece in pieces if len(piece) >= 32]
        if args.max_chunks:
            pieces = pieces[: args.max_chunks]
        if not pieces:
            print("calibration text is shorter than one chunk", file=sys.stderr)
            return 2
        options = _runtime_options(args)
        options["context_limit"] = max(args.context, chunk + 8)
        with model.native_runtime(**options) as runtime:
            runtime.prepare()
            for index, piece in enumerate(pieces):
                runtime.reset()
                runtime.generate(piece, 1, lambda _token: False)
                print(f"\r[imatrix] chunk {index + 1}/{len(pieces)}",
                      end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)
            entries = runtime.imatrix_entries()

    if not entries:
        print("no activations were captured; is this a supported "
              "architecture for the native Qwen runtime?", file=sys.stderr)
        return 1
    # The legacy llama.cpp layout the packer's loader reads back: per entry a
    # name, a call count, and per-channel values. Values are means, so the
    # file does not depend on how much text was run beyond its distribution.
    blob = struct.pack("<i", len(entries))
    for name, sums, rows in entries:
        encoded = name.encode()
        mean = [value / max(1, rows) for value in sums]
        blob += struct.pack("<i", len(encoded)) + encoded
        blob += struct.pack("<ii", len(pieces), len(mean))
        blob += struct.pack(f"<{len(mean)}f", *mean)
    blob += struct.pack("<i", len(pieces))
    dataset = str(args.text).encode()
    blob += struct.pack("<i", len(dataset)) + dataset
    args.output.write_bytes(blob)
    total = sum(len(piece) for piece in pieces)
    print(f"wrote {args.output}: {len(entries)} tensors over {total} tokens "
          f"({len(pieces)} chunks)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _resolve_quant(args)
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
    if args.command == "imatrix":
        return _gather_imatrix(args)
    if args.command in {"probe", "probe-native-v2", "probe-qwen-native-v2", "probe-native"}:
        _validate_runtime_args(args)
        with V2Model(args.model, mtp_model=args.mtp_model) as model:
            prompt = _prompt_tokens(model, args)
            options = _runtime_options(args)
            options["context_limit"] = args.context
            with model.native_runtime(**options) as runtime:
                runtime.prepare()
                output: list[int] = []
                stops = _stop_tokens(model)

                def collect(token: int) -> bool:
                    if token in stops:
                        return False
                    output.append(token)
                    return True

                runtime.generate(prompt, args.generate_tokens, collect)
                info = runtime.info
            print(json.dumps({"generated_tokens": output,
                              "generated_text": model.decode_tokens(output),
                              "runtime": info}, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
