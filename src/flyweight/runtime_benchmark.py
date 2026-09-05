"""Repeatable native-v2 baseline and comparison harness.

The harness intentionally lives outside the production CLI. Runtime changes
must be measurable before they are exposed as supported product options.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO

# `resource` is POSIX-only, and importing it at module scope made the whole
# benchmark harness -- and the `flyweight benchmark` command that imports it --
# fail on Windows with ModuleNotFoundError before running anything.
if os.name == "nt":
    resource: Any = None
else:
    import resource  # type: ignore[no-redef,assignment]

from .v2 import V2Model


PREFILL_COUNTERS = (
    "prefill_calls",
    "prefill_tokens",
    "prefill_nanoseconds",
    "prefill_route_wait_nanoseconds",
    "prefill_expert_nanoseconds",
    "prefill_gpu_core_nanoseconds",
    "prefill_gpu_router_nanoseconds",
    "prefill_gpu_transfer_nanoseconds",
    "prefix_cache_hits",
    "prefix_cache_misses",
    "prefix_cache_reused_tokens",
    "prefix_cache_reprefilled_tokens",
    "expert_cache_hits",
    "expert_cache_misses",
    "expert_cache_admissions",
    "expert_cache_evictions",
    "expert_cache_rejections",
    "expert_cache_unused_admissions",
    "expert_cache_prompt_bypasses",
    "expert_cache_deferred_admissions",
    "expert_residency_epochs",
    "prefill_cache_seeded_experts",
    "prefill_cache_seed_nanoseconds",
    "prefill_cache_seed_bytes",
    "prefill_cache_seed_selected_experts",
    "prefill_cache_seed_auto_skips",
    "prefill_cache_seed_budget_stops",
)

DECODE_COUNTERS = (
    "decode_calls",
    "decode_nanoseconds",
    "route_wait_nanoseconds",
    "expert_page_nanoseconds",
    "tail_wait_nanoseconds",
    "expert_compute_nanoseconds",
    "expert_cache_hits",
    "expert_cache_misses",
    "expert_cache_admissions",
    "expert_cache_evictions",
    "expert_cache_rejections",
    "expert_cache_unused_admissions",
    "expert_cache_deferred_admissions",
    "expert_residency_epochs",
    "prefill_cache_seed_hits",
    "prefill_cache_seed_avoided_misses",
    "next_layer_prefetch_predictions",
    "next_layer_prefetch_hits",
    "next_layer_prefetch_bytes",
)


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

LATENCY_METRICS = (
    "first_token_latency_seconds",
    "native_prefill_seconds",
    "decode_median_seconds",
    "decode_p95_seconds",
)


def _counter_delta(
    before: dict[str, Any], after: dict[str, Any], fields: Iterable[str]
) -> dict[str, int]:
    return {
        field: int(after.get(field, 0)) - int(before.get(field, 0))
        for field in fields
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(len(ordered) * fraction) - 1),
    )
    return float(ordered[index])


def measure_runtime_sample(
    runtime: Any,
    prompt_tokens: Sequence[int],
    *,
    warmup_decode: int,
    decode_iterations: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure production prefill separately from steady single-token decode."""
    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")
    if warmup_decode < 0:
        raise ValueError("warmup_decode must be non-negative")
    if decode_iterations <= 0:
        raise ValueError("decode_iterations must be positive")

    before_prefill = dict(runtime.info)
    first_tokens: list[int] = []
    prefill_started = clock()
    runtime.generate(list(prompt_tokens), 1, first_tokens.append)
    first_token_latency = clock() - prefill_started
    if len(first_tokens) != 1:
        raise RuntimeError("native prefill did not produce exactly one token")
    after_prefill = dict(runtime.info)
    prefill_counters = _counter_delta(
        before_prefill, after_prefill, PREFILL_COUNTERS
    )
    native_prefill_seconds = prefill_counters["prefill_nanoseconds"] / 1.0e9

    current_token = first_tokens[0]
    warmup_seconds: list[float] = []
    warmup_tokens: list[int] = []
    for _ in range(warmup_decode):
        started = clock()
        current_token = runtime.decode(current_token)
        warmup_seconds.append(clock() - started)
        warmup_tokens.append(current_token)

    before_decode = dict(runtime.info)
    decode_seconds: list[float] = []
    measured_tokens: list[int] = []
    for _ in range(decode_iterations):
        started = clock()
        current_token = runtime.decode(current_token)
        decode_seconds.append(clock() - started)
        measured_tokens.append(current_token)
    after_decode = dict(runtime.info)
    decode_counters = _counter_delta(
        before_decode, after_decode, DECODE_COUNTERS
    )
    decode_total = sum(decode_seconds)

    return {
        "prompt_tokens": len(prompt_tokens),
        "first_token": first_tokens[0],
        "first_token_latency_seconds": first_token_latency,
        "native_prefill_seconds": native_prefill_seconds,
        "final_token_and_boundary_seconds": max(
            0.0, first_token_latency - native_prefill_seconds
        ),
        "prompt_tokens_per_second": (
            len(prompt_tokens) / first_token_latency
            if first_token_latency
            else 0.0
        ),
        "native_prefill_tokens_per_second": (
            prefill_counters["prefill_tokens"] / native_prefill_seconds
            if native_prefill_seconds
            else 0.0
        ),
        "warmup_decode": warmup_decode,
        "warmup_decode_seconds": warmup_seconds,
        "decode_iterations": decode_iterations,
        "decode_seconds": decode_seconds,
        "decode_median_seconds": statistics.median(decode_seconds),
        "decode_p95_seconds": _percentile(decode_seconds, 0.95),
        "decode_tokens_per_second": (
            decode_iterations / decode_total if decode_total else 0.0
        ),
        "generated_tokens": [
            first_tokens[0],
            *warmup_tokens,
            *measured_tokens,
        ],
        "prefill_counters": prefill_counters,
        "decode_counters": decode_counters,
        "runtime": after_decode,
    }


def _process_memory() -> dict[str, int]:
    if resource is None:
        return _windows_process_memory()
    rss_bytes = 0
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        rss_bytes = int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        pass
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maximum_bytes = int(maximum if sys.platform == "darwin" else maximum * 1024)
    return {"rss_bytes": rss_bytes, "maximum_rss_bytes": maximum_bytes}


def _windows_process_memory() -> dict[str, int]:
    """The same two numbers from psapi: working set, current and peak.

    Reporting zeros instead would be a benchmark that silently stops measuring
    memory on one platform, which is worse than not running there at all -- an
    expert-offload regression shows up in exactly this field.
    """
    import ctypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
        # The prototypes are not optional. GetCurrentProcess returns the
        # pseudo-handle (HANDLE)-1, which ctypes truncates to 32 bits without a
        # declared restype -- the call then fails and reports zero memory.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
    except (AttributeError, OSError):
        ok = 0
    if not ok:
        return {"rss_bytes": 0, "maximum_rss_bytes": 0}
    return {
        "rss_bytes": int(counters.WorkingSetSize),
        "maximum_rss_bytes": int(counters.PeakWorkingSetSize),
    }


def _source_state(root: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"revision": None, "dirty": None}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return state
    state["revision"] = revision.stdout.strip()
    state["dirty"] = bool(status.stdout)
    return state


def _expand_tokens(tokens: Sequence[int], length: int) -> list[int]:
    if length <= 0:
        raise ValueError("prompt lengths must be positive")
    if not tokens:
        raise ValueError("the base prompt must contain at least one token")
    repeats = (length + len(tokens) - 1) // len(tokens)
    return list((list(tokens) * repeats)[:length])


def _write_jsonl(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()


def _summary(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    metrics = (
        "first_token_latency_seconds",
        "native_prefill_seconds",
        "prompt_tokens_per_second",
        "decode_median_seconds",
        "decode_p95_seconds",
        "decode_tokens_per_second",
    )
    result: dict[str, Any] = {"samples": len(samples), "metrics": {}}
    for metric in metrics:
        values = [float(sample[metric]) for sample in samples]
        lower_is_better = metric.endswith("_seconds")
        result["metrics"][metric] = {
            "median": statistics.median(values),
            "worst": max(values) if lower_is_better else min(values),
            "values": values,
        }
    token_sequences = [sample["generated_tokens"] for sample in samples]
    result["deterministic_tokens"] = all(
        tokens == token_sequences[0] for tokens in token_sequences[1:]
    )
    result["generated_tokens"] = token_sequences[0]
    return result


def _runtime_options(args: argparse.Namespace) -> dict[str, Any]:
    legacy_policy = args.expert_mode in {
        "hybrid",
        "gpu",
        "legacy-hybrid",
        "legacy-paging",
    }
    return {
        "device": args.device,
        "context_limit": args.context,
        "gpu_cache_bytes": args.gpu_cache_mib * 1024**2,
        "expert_mode": args.expert_mode,
        "mtp_drafts": args.mtp_drafts,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "parallel_sequences": args.parallel,
        "prompt_cache_mib": args.prompt_cache_mib,
        "prefill_cache_seed": (
            args.prefill_cache_seed
            if args.prefill_cache_seed is not None
            else ("off" if legacy_policy else "auto")
        ),
        "expert_paging": args.expert_paging,
        "cpu_threads": args.cpu_threads,
        "hybrid_prefill": args.hybrid_prefill,
        "expert_residency": (
            args.expert_residency
            if args.expert_residency is not None
            else ("mutable" if legacy_policy else "immutable")
        ),
    }


def run_baseline(args: argparse.Namespace, stream: TextIO) -> int:
    if args.samples <= 0 or args.sample_warmup < 0:
        raise ValueError("samples must be positive and sample-warmup non-negative")
    if args.context <= 0 or args.gpu_cache_mib < 0:
        raise ValueError("context and GPU cache budget are invalid")
    prompt_lengths = [
        int(value) for value in args.prompt_lengths.split(",") if value.strip()
    ]
    if not prompt_lengths:
        raise ValueError("at least one prompt length is required")
    maximum_required = max(prompt_lengths) + args.warmup_decode + args.decode_iterations
    if maximum_required > args.context:
        raise ValueError("prompt and decode tokens exceed the configured context")

    root = Path(__file__).resolve().parents[2]
    selected_environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("FLYWEIGHT_") or key in {"OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES"}
    }
    model_path = args.model.resolve()
    with V2Model(model_path) as model:
        base_tokens = (
            model.tokenize(args.prompt)
            if args.prompt is not None
            else [int(value) for value in args.tokens.split(",") if value.strip()]
        )
        metadata = {
            "kind": "metadata",
            "schema": 1,
            "label": args.label,
            "created_unix_ns": time.time_ns(),
            "model": {
                **model.info,
                "path": str(model_path),
                "size": model_path.stat().st_size,
                "mtime_ns": model_path.stat().st_mtime_ns,
            },
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
            },
            "source": _source_state(root),
            "environment": selected_environment,
            "configuration": {
                **_runtime_options(args),
                "prompt_lengths": prompt_lengths,
                "warmup_decode": args.warmup_decode,
                "decode_iterations": args.decode_iterations,
                "samples": args.samples,
                "sample_warmup": args.sample_warmup,
            },
        }
        try:
            metadata["gpu_before"] = V2Model.gpu_info(args.device)
        except Exception as error:
            metadata["gpu_before_error"] = str(error)
        _write_jsonl(stream, metadata)

        for prompt_length in prompt_lengths:
            prompt_tokens = _expand_tokens(base_tokens, prompt_length)
            measured: list[dict[str, Any]] = []
            total_runs = args.sample_warmup + args.samples
            for run_index in range(total_runs):
                excluded = run_index < args.sample_warmup
                with model.native_runtime(**_runtime_options(args)) as runtime:
                    prepare_started = time.perf_counter()
                    runtime.prepare()
                    prepare_seconds = time.perf_counter() - prepare_started
                    sample = measure_runtime_sample(
                        runtime,
                        prompt_tokens,
                        warmup_decode=args.warmup_decode,
                        decode_iterations=args.decode_iterations,
                    )
                sample.update(
                    {
                        "kind": "sample",
                        "schema": 1,
                        "label": args.label,
                        "case": f"prompt-{prompt_length}",
                        "sample": (
                            run_index - args.sample_warmup if not excluded else run_index
                        ),
                        "excluded_warmup": excluded,
                        "prepare_seconds": prepare_seconds,
                        "process_memory": _process_memory(),
                    }
                )
                try:
                    sample["gpu_after_release"] = V2Model.gpu_info(args.device)
                except Exception as error:
                    sample["gpu_after_release_error"] = str(error)
                _write_jsonl(stream, sample)
                if not excluded:
                    measured.append(sample)
            _write_jsonl(
                stream,
                {
                    "kind": "summary",
                    "schema": 1,
                    "label": args.label,
                    "case": f"prompt-{prompt_length}",
                    **_summary(measured),
                },
            )
    return 0


def _load_samples(path: Path) -> dict[str, list[dict[str, Any]]]:
    cases: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if record.get("kind") != "sample" or record.get("excluded_warmup"):
                continue
            cases.setdefault(str(record["case"]), []).append(record)
    if not cases:
        raise ValueError(f"{path} contains no measured samples")
    return cases


def _regression_percent(baseline: float, candidate: float, lower_is_better: bool) -> float:
    if baseline == 0.0:
        if not lower_is_better or candidate == 0.0:
            return 0.0
        # A percentage is undefined here, but a finite sentinel keeps the
        # comparison valid JSON and correctly fails ordinary regression gates.
        return 100.0
    direction = candidate - baseline if lower_is_better else baseline - candidate
    return direction / baseline * 100.0


def compare_baselines(
    baseline_path: Path,
    candidate_path: Path,
    *,
    max_prefill_regression_pct: float,
    max_decode_regression_pct: float,
    require_token_match: bool = True,
) -> dict[str, Any]:
    baseline_cases = _load_samples(baseline_path)
    candidate_cases = _load_samples(candidate_path)
    missing = sorted(set(baseline_cases) ^ set(candidate_cases))
    result: dict[str, Any] = {
        "kind": "comparison",
        "schema": 1,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "passed": not missing,
        "missing_cases": missing,
        "cases": {},
    }
    for case in sorted(set(baseline_cases) & set(candidate_cases)):
        baseline = _summary(baseline_cases[case])
        candidate = _summary(candidate_cases[case])
        token_match = (
            baseline["deterministic_tokens"]
            and candidate["deterministic_tokens"]
            and baseline["generated_tokens"] == candidate["generated_tokens"]
        )
        checks: list[dict[str, Any]] = []
        for metric in LATENCY_METRICS:
            before = float(baseline["metrics"][metric]["median"])
            after = float(candidate["metrics"][metric]["median"])
            limit = (
                max_decode_regression_pct
                if metric.startswith("decode_")
                else max_prefill_regression_pct
            )
            regression = _regression_percent(before, after, True)
            checks.append(
                {
                    "metric": metric,
                    "baseline_median": before,
                    "candidate_median": after,
                    "regression_pct": regression,
                    "limit_pct": limit,
                    "passed": regression <= limit,
                }
            )
        throughput_before = float(
            baseline["metrics"]["decode_tokens_per_second"]["median"]
        )
        throughput_after = float(
            candidate["metrics"]["decode_tokens_per_second"]["median"]
        )
        throughput_regression = _regression_percent(
            throughput_before, throughput_after, False
        )
        checks.append(
            {
                "metric": "decode_tokens_per_second",
                "baseline_median": throughput_before,
                "candidate_median": throughput_after,
                "regression_pct": throughput_regression,
                "limit_pct": max_decode_regression_pct,
                "passed": throughput_regression <= max_decode_regression_pct,
            }
        )
        case_passed = all(check["passed"] for check in checks) and (
            token_match or not require_token_match
        )
        result["cases"][case] = {
            "passed": case_passed,
            "token_match": token_match,
            "checks": checks,
        }
        result["passed"] = result["passed"] and case_passed
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record and compare native-v2 CUDA runtime baselines"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="write JSONL benchmark samples")
    run.add_argument("model", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--label", default="baseline")
    prompt = run.add_mutually_exclusive_group()
    prompt.add_argument("--prompt")
    prompt.add_argument("--tokens", default="1,2,3,4,5,6,7,8")
    run.add_argument("--prompt-lengths", default="256,1024,4096")
    run.add_argument("--samples", type=int, default=5)
    run.add_argument("--sample-warmup", type=int, default=1)
    run.add_argument("--warmup-decode", type=int, default=3)
    run.add_argument("--decode-iterations", type=int, default=10)
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--context", type=int, default=8192)
    run.add_argument("--gpu-cache-mib", type=int, default=0)
    run.add_argument(
        "--expert-mode", "--moe-device",
        dest="expert_mode",
        choices=(
            "cpu", "auto", "resident", "hybrid", "gpu",
            "legacy-paging", "legacy-hybrid",
        ),
        default="auto",
    )
    run.add_argument("--hybrid-prefill", choices=("split", "cpu"), default="split")
    run.add_argument(
        "--expert-residency", choices=("mutable", "immutable"), default=None
    )
    run.add_argument("--mtp-drafts", type=int, default=0)
    run.add_argument("--cache-type-k", choices=("f32", "f16", "bf16", "q8_0"), default="f16")
    run.add_argument("--cache-type-v", choices=("f32", "f16", "bf16", "q8_0"), default="f16")
    run.add_argument("--parallel", type=int, default=1)
    run.add_argument("--prompt-cache-mib", type=int, default=0)
    run.add_argument("--prefill-cache-seed", type=_prefill_cache_seed, default=None)
    run.add_argument("--expert-paging", choices=("auto", "staged", "direct"), default="auto")
    run.add_argument("--cpu-threads", type=int, default=0)

    compare = commands.add_parser("compare", help="compare two JSONL baselines")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--max-prefill-regression-pct", type=float, default=3.0)
    compare.add_argument("--max-decode-regression-pct", type=float, default=3.0)
    compare.add_argument("--allow-token-mismatch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        if args.output is None:
            return run_baseline(args, sys.stdout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            return run_baseline(args, stream)
    result = compare_baselines(
        args.baseline,
        args.candidate,
        max_prefill_regression_pct=args.max_prefill_regression_pct,
        max_decode_regression_pct=args.max_decode_regression_pct,
        require_token_match=not args.allow_token_mismatch,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
