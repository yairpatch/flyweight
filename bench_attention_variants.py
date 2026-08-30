#!/usr/bin/env python3
"""Time the decode attention kernels against each other, and against the wall.

bench_attention_kernel.py answers "what limits one kernel"; this answers "which
kernel, at this shape". It launches the candidates on the same buffers and, with
--check, against a float64 host reference, so a kernel that is fast because it
quietly changed the arithmetic is visible as an error column rather than as a
win.

Two things it does deliberately:

  * No model. The corpus is compiled straight out of the headers, so a kernel
    experiment does not need 8 GB of weights resident to run, and the KV under
    test is the only thing on the device.
  * Interleaved rounds. This box drifts more than 10% in clock over a few
    seconds, which is larger than most of the differences worth measuring, so
    every kernel is timed inside every round and the reported number is the best
    round -- the least-throttled one.

The roofline is the same kernel with nothing shared: `--kv-heads 8 --share 1`
reads every byte exactly once and measures 483 GB/s on this card, which is what
the sharing variants are trying to reach.

Two flags matter for correctness rather than speed. `--capacity` larger than
`--tokens` leaves dead slots, which is the only way a wrong ring-slot
computation shows up -- with a full ring, attention is permutation invariant and
reads the right *set* of rows however it addresses them. `--first` then starts
the window mid-ring so it wraps.
"""
from __future__ import annotations

import argparse
import ctypes
import pathlib
import time

import numpy

from flyweight.v2 import _library

ROOT = pathlib.Path(__file__).resolve().parent
HEAD_DIM = 256


def corpus(path: pathlib.Path) -> str:
    """The CUDA source out of a header that holds it as one spliced literal."""
    text = path.read_text(encoding="utf-8")
    opener = 'R"FLYWEIGHT_CUDA('
    start = text.index(opener) + len(opener)
    end = text.rindex(')FLYWEIGHT_CUDA";')
    return text[start:end].replace(')FLYWEIGHT_CUDA"\nR"FLYWEIGHT_CUDA(', "")


class Gpu:
    def __init__(self) -> None:
        self.library = library = _library()
        if library.flyweight_gpu_init(0) != 0:
            raise SystemExit("gpu init failed")
        library.flyweight_gpu_alloc.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        library.flyweight_gpu_upload_sync.argtypes = [
            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
        library.flyweight_gpu_download.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        library.flyweight_gpu_launch_named.argtypes = [
            ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]
        library.flyweight_gpu_compile.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int32,
            ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
        library.flyweight_gpu_attention_f16_cublas.argtypes = (
            [ctypes.c_uint64] * 7 + [ctypes.c_int32] * 6 + [ctypes.c_float])
        source = ("#define FLYWEIGHT_MMQ_ROW_WARPS 4\n"
                  "#define FLYWEIGHT_MMQ_ROW_FRAGS 2\n"
                  + corpus(ROOT / "native/include/flyweight_v2_qwen_kernels.hpp")
                  + corpus(ROOT / "native/include/flyweight_v2_native_kernels.hpp"))
        options = [b"-I/opt/cuda/include", b"-I/opt/cuda/include/cccl"]
        array = (ctypes.c_char_p * len(options))(*options)
        log = ctypes.create_string_buffer(65536)
        status = library.flyweight_gpu_compile(
            source.encode(), array, len(options), 0, log, 65536)
        if status != 0:
            print(log.value.decode(errors="replace")[:6000])
            raise SystemExit(f"corpus compile failed: {status}")
        self.stream = ctypes.c_uint64()
        library.flyweight_gpu_stream_create(ctypes.byref(self.stream))
        self.start, self.end = ctypes.c_uint64(), ctypes.c_uint64()
        library.flyweight_gpu_timed_event_create(ctypes.byref(self.start))
        library.flyweight_gpu_timed_event_create(ctypes.byref(self.end))

    def alloc(self, size: int) -> ctypes.c_uint64:
        value = ctypes.c_uint64()
        if self.library.flyweight_gpu_alloc(size, ctypes.byref(value)):
            raise SystemExit("device allocation failed")
        return value

    def upload(self, pointer: ctypes.c_uint64, data: numpy.ndarray) -> None:
        buffer = numpy.ascontiguousarray(data)
        if self.library.flyweight_gpu_upload_sync(
            pointer, buffer.ctypes.data, buffer.nbytes
        ):
            raise SystemExit("upload failed")

    def download(self, pointer: ctypes.c_uint64,
                 like: numpy.ndarray) -> numpy.ndarray:
        out = numpy.empty_like(like)
        if self.library.flyweight_gpu_download(
            out.ctypes.data, pointer, out.nbytes, 0
        ):
            raise SystemExit("download failed")
        return out

    def sync(self) -> None:
        self.library.flyweight_gpu_stream_sync(self.stream)

    def elapsed_us(self, run, iterations: int) -> float:
        run(3)
        self.sync()
        self.library.flyweight_gpu_event_record(self.start, self.stream)
        run(iterations)
        self.library.flyweight_gpu_event_record(self.end, self.stream)
        self.sync()
        milliseconds = ctypes.c_float()
        self.library.flyweight_gpu_event_elapsed(
            self.start, self.end, ctypes.byref(milliseconds))
        return milliseconds.value / iterations * 1000


def quantize_q8(rows: numpy.ndarray) -> numpy.ndarray:
    """[..., head_dim] float32 -> the cache's [f16 scale | 32 int8] blocks."""
    flat = rows.reshape(-1, 32)
    scale = numpy.abs(flat).max(axis=1) / 127.0
    scale = numpy.where(scale == 0, 1.0, scale).astype(numpy.float16)
    quantized = numpy.rint(
        flat / scale.astype(numpy.float32)[:, None]
    ).clip(-127, 127).astype(numpy.int8)
    packed = numpy.empty((flat.shape[0], 34), dtype=numpy.uint8)
    packed[:, :2] = scale.view(numpy.uint8).reshape(-1, 2)
    packed[:, 2:] = quantized.view(numpy.uint8)
    return packed.reshape(rows.shape[:-1] + (rows.shape[-1] // 32 * 34,))


def dequantize_q8(packed: numpy.ndarray, head_dim: int) -> numpy.ndarray:
    blocks = packed.reshape(-1, 34)
    scale = blocks[:, :2].copy().view(numpy.float16).astype(numpy.float32)
    values = blocks[:, 2:].view(numpy.int8).astype(numpy.float32)
    return (values * scale).reshape(packed.shape[:-1] + (head_dim,))


def reference(query, keys, values, heads, kv_heads, scale, first, tokens):
    """Attention in float64 over exactly the rows the window covers.

    Token t lives in slot (first + t) % capacity. With capacity == tokens every
    slot is live and attention is permutation invariant, so a wrong slot still
    gives the right answer -- only a cache with dead slots tests the addressing.
    """
    share = heads // kv_heads
    capacity = keys.shape[1]
    order = (first + numpy.arange(tokens)) % capacity
    keys, values = keys[:, order], values[:, order]
    out = numpy.empty((heads, keys.shape[-1]), dtype=numpy.float64)
    for head in range(heads):
        kv = head // share
        scores = keys[kv].astype(numpy.float64) @ query[head].astype(numpy.float64)
        scores *= scale
        scores -= scores.max()
        weights = numpy.exp(scores)
        weights /= weights.sum()
        out[head] = weights @ values[kv].astype(numpy.float64)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=49152)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--share", type=int, default=8,
                        help="query heads per KV head; 1 measures the roofline")
    parser.add_argument("--kv-type", choices=("f16", "bf16", "q8"), default="f16")
    parser.add_argument("--tile", type=int, default=256,
                        help="tile width for the grouped-rows kernel (256/512)")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--check", action="store_true",
                        help="verify against a host reference; use few tokens")
    parser.add_argument("--first", type=int, default=0,
                        help="ring start slot; non-zero exercises the wrap")
    parser.add_argument("--capacity", type=int, default=0,
                        help="cache slots; > tokens leaves dead slots a wrong "
                             "slot computation would read")
    parser.add_argument("--no-merge", action="store_true",
                        help="time the tiles kernel alone, without its merge")
    arguments = parser.parse_args()

    gpu = Gpu()
    tokens, kv_heads, share = arguments.tokens, arguments.kv_heads, arguments.share
    heads = kv_heads * share
    capacity = arguments.capacity if arguments.capacity else arguments.tokens
    quantized = arguments.kv_type == "q8"
    scale = 0.0625
    merge = not arguments.no_merge

    rng = numpy.random.default_rng(7)
    query = rng.standard_normal((heads, HEAD_DIM), dtype=numpy.float32)
    keys = rng.standard_normal((kv_heads, capacity, HEAD_DIM), dtype=numpy.float32)
    values = rng.standard_normal((kv_heads, capacity, HEAD_DIM), dtype=numpy.float32)
    if quantized:
        key_host, value_host = quantize_q8(keys), quantize_q8(values)
        key_exact = dequantize_q8(key_host, HEAD_DIM)
        value_exact = dequantize_q8(value_host, HEAD_DIM)
    else:
        precision = numpy.float16  # bf16 rounds below, it has no numpy dtype
        key_host = keys.astype(precision)
        value_host = values.astype(precision)
        if arguments.kv_type == "bf16":
            key_host = keys.view(numpy.uint32) >> 16
            key_host = key_host.astype(numpy.uint16)
            value_host = values.view(numpy.uint32) >> 16
            value_host = value_host.astype(numpy.uint16)
            key_exact = (key_host.astype(numpy.uint32) << 16).view(numpy.float32)
            value_exact = (value_host.astype(numpy.uint32) << 16).view(numpy.float32)
        else:
            key_exact = key_host.astype(numpy.float32)
            value_exact = value_host.astype(numpy.float32)

    device_query = gpu.alloc(query.nbytes)
    device_keys = gpu.alloc(key_host.nbytes)
    device_values = gpu.alloc(value_host.nbytes)
    gpu.upload(device_query, query)
    gpu.upload(device_keys, key_host)
    gpu.upload(device_values, value_host)
    kv_bytes = key_host.nbytes
    output = numpy.zeros((heads, HEAD_DIM), dtype=numpy.float32)
    device_output = gpu.alloc(output.nbytes)
    expected = (reference(query, key_exact, value_exact, heads, kv_heads, scale,
                          arguments.first, tokens)
                if arguments.check else None)

    candidates = [
        (f"kv_attention_fused_{arguments.kv_type}_tiles256", 1024, heads),
    ]
    if share in (4, 8):
        candidates.append((
            f"kv_attention_gqa_rows_{arguments.kv_type}_256"
            f"_s{share}_t{arguments.tile}", arguments.tile, kv_heads))

    print(f"tokens={tokens} kv_heads={kv_heads} share={share} "
          f"kv_type={arguments.kv_type} KV={2 * kv_bytes / 2**20:.0f} MiB "
          f"(K+V){'' if merge else ', tiles only'}")
    print(f"{'kernel':>40} {'best us':>9} {'GB/s':>7} {'max err':>10} "
          f"{'spread':>7}")

    plans = []
    for name, tile_tokens, grid_x in candidates:
        tile_count = (tokens + tile_tokens - 1) // tile_tokens
        partial = gpu.alloc(heads * tile_count * (HEAD_DIM + 2) * 4)
        # Every scalar behind the argument array has to outlive the loop that
        # built it: the array holds raw addresses, and a recycled one launches
        # the kernel on whatever moved in, which shows up as a kernel that is
        # implausibly fast because its own guards sent it home.
        held = [partial,
                ctypes.c_int32(heads), ctypes.c_int32(kv_heads),
                ctypes.c_int32(HEAD_DIM), ctypes.c_int32(tokens),
                ctypes.c_int32(capacity), ctypes.c_int32(arguments.first),
                ctypes.c_float(scale), ctypes.c_int32(tile_count)]
        args = (ctypes.c_void_p * 11)(
            ctypes.addressof(device_query), ctypes.addressof(device_keys),
            ctypes.addressof(device_values),
            *[ctypes.addressof(item) for item in held[:8]])
        merge_args = (ctypes.c_void_p * 5)(
            ctypes.addressof(partial), ctypes.addressof(device_output),
            *[ctypes.addressof(held[index]) for index in (1, 3, 8)])
        held.extend((args, merge_args))

        # Bound at definition, because these are called after the loop ends.
        def run(count, name=name, args=args, merge_args=merge_args,
                grid_x=grid_x, tile_count=tile_count):
            for _ in range(count):
                if gpu.library.flyweight_gpu_launch_named(
                    name.encode(), grid_x, tile_count, 256, 0, gpu.stream, args
                ):
                    raise SystemExit(f"launch failed: {name}")
                if merge and gpu.library.flyweight_gpu_launch_named(
                    b"kv_attention_fused_merge256", heads, 1, 256, 0,
                    gpu.stream, merge_args
                ):
                    raise SystemExit("merge launch failed")
        plans.append([name, run, held, []])

    if arguments.kv_type == "f16":
        query_staged = gpu.alloc(heads * HEAD_DIM * 2)
        scores_staged = gpu.alloc(heads * tokens * 2 + 4096)

        def run_cublas(count):
            for _ in range(count):
                if gpu.library.flyweight_gpu_attention_f16_cublas(
                    device_query, query_staged, device_keys,
                    device_values, scores_staged, device_output, gpu.stream,
                    heads, kv_heads, HEAD_DIM, tokens, capacity,
                    arguments.first, scale
                ):
                    raise SystemExit("cublas declined this shape")
        plans.append([f"{arguments.kv_type} cublas", run_cublas,
                      [query_staged, scores_staged], []])

    errors = {}
    for name, run, _, _ in plans:
        if expected is None:
            errors[name] = float("nan")
            continue
        run(1)
        gpu.sync()
        produced = gpu.download(device_output, output)
        errors[name] = float(
            numpy.abs(produced.astype(numpy.float64) - expected).max())

    settle = time.perf_counter()
    while time.perf_counter() - settle < arguments.settle:
        for _, run, _, _ in plans:
            run(5)
        gpu.sync()
    for _ in range(arguments.rounds):
        for _, run, _, samples in plans:
            samples.append(gpu.elapsed_us(run, arguments.iterations))
    for name, _, _, samples in plans:
        best = min(samples)
        spread = (max(samples) - best) / best * 100
        print(f"{name:>40} {best:9.1f} "
              f"{2 * kv_bytes / (best / 1e6) / 1e9:7.0f} {errors[name]:10.2e} "
              f"{spread:6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
