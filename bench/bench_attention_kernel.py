#!/usr/bin/env python3
"""Isolate the fused attention kernel and find what actually limits it.

Both cuBLAS and the split-K kernel land near 230 GB/s of *useful* KV against a
643 GB/s roofline. Two explanations fit that number and they call for opposite
fixes, so this separates them by varying the GQA sharing factor at a fixed KV
size:

  query_heads = kv_heads      -> every byte read is used once
  query_heads = 8 * kv_heads  -> the same bytes serve 8 query heads

If DRAM traffic is the limit and L2 absorbs the sharing, both take about the
same time: the KV is read once either way. If the kernel is latency- or
ILP-bound, the 8x version takes ~8x longer, because the work scales and the
memory does not.
"""
from __future__ import annotations

import argparse
import ctypes
import time

from flyweight.v2 import V2Model, V2QwenRuntime, _library

HEAD_DIM = 256
TOKENS_PER_TILE = 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=49152)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--kv-type", choices=("f16", "q8_0"), default="f16",
                        help="which fused variant to launch; q8_0 reads a\n                             34-byte-blocked cache and dequantizes inline")
    arguments = parser.parse_args()

    model = V2Model(arguments.model)
    runtime = V2QwenRuntime(model, context_limit=512)
    runtime.prepare()

    library = _library()
    library.flyweight_gpu_alloc.argtypes = [ctypes.c_uint64,
                                          ctypes.POINTER(ctypes.c_uint64)]
    library.flyweight_gpu_launch_named.argtypes = [
        ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]
    library.flyweight_gpu_stream_sync.argtypes = [ctypes.c_uint64]

    stream = ctypes.c_uint64()
    library.flyweight_gpu_stream_create(ctypes.byref(stream))
    start, end = ctypes.c_uint64(), ctypes.c_uint64()
    library.flyweight_gpu_timed_event_create(ctypes.byref(start))
    library.flyweight_gpu_timed_event_create(ctypes.byref(end))

    tokens = arguments.tokens
    kv_heads = arguments.kv_heads
    capacity = tokens
    tile_count = (tokens + TOKENS_PER_TILE - 1) // TOKENS_PER_TILE
    # K and V laid out [kv_head][capacity][head_dim]. q8_0 stores 32 elements
    # as [f16 scale | 32 int8] = 34 bytes, so a row is (head_dim/32)*34 rather
    # than head_dim*2 -- 53% of the bytes for the same tokens.
    quantized = arguments.kv_type == "q8_0"
    row_bytes = (HEAD_DIM // 32) * 34 if quantized else HEAD_DIM * 2
    kv_bytes = kv_heads * capacity * row_bytes
    print(f"tokens={tokens}  kv_heads={kv_heads}  tiles={tile_count}  "
          f"KV={2 * kv_bytes / 2**20:.0f} MiB (K+V)")
    print(f"{'query_heads':>12} {'share':>6} {'us':>9} {'useful GB/s':>12} "
          f"{'us/query_head':>14}")

    for share in (1, 2, 4, 8):
        heads = kv_heads * share
        pointers = []
        for size in (
            heads * HEAD_DIM * 4,                              # query, f32
            kv_bytes,                                          # keys
            kv_bytes,                                          # values
            heads * tile_count * (HEAD_DIM + 2) * 4,           # partial
        ):
            value = ctypes.c_uint64()
            if library.flyweight_gpu_alloc(size, ctypes.byref(value)):
                raise SystemExit("device allocation failed")
            pointers.append(value)

        c_heads = ctypes.c_int32(heads)
        c_kv = ctypes.c_int32(kv_heads)
        c_dim = ctypes.c_int32(HEAD_DIM)
        c_tokens = ctypes.c_int32(tokens)
        c_capacity = ctypes.c_int32(capacity)
        c_first = ctypes.c_int32(0)
        c_scale = ctypes.c_float(0.0625)
        args = (ctypes.c_void_p * 11)(
            *[ctypes.addressof(p) for p in pointers],
            ctypes.addressof(c_heads), ctypes.addressof(c_kv),
            ctypes.addressof(c_dim), ctypes.addressof(c_tokens),
            ctypes.addressof(c_capacity), ctypes.addressof(c_first),
            ctypes.addressof(c_scale))
        name = (b"kv_attention_fused_q8_tiles256" if quantized
                else b"kv_attention_fused_f16_tiles256")

        def run(count: int) -> None:
            for _ in range(count):
                if library.flyweight_gpu_launch_named(
                    name, heads, tile_count, 256, 0, stream, args
                ):
                    raise SystemExit(f"launch failed: {name!r}")

        # Seconds of continuous load, not a few launches: an idle GPU on this
        # box runs kernels at a fraction of sustained clock, which silently
        # made whichever configuration was measured first look terrible.
        settle = time.perf_counter()
        while time.perf_counter() - settle < 3.0:
            run(20)
            library.flyweight_gpu_stream_sync(stream)
        library.flyweight_gpu_stream_sync(stream)
        library.flyweight_gpu_event_record(start, stream)
        run(arguments.iterations)
        library.flyweight_gpu_event_record(end, stream)
        library.flyweight_gpu_stream_sync(stream)
        elapsed = ctypes.c_float()
        library.flyweight_gpu_event_elapsed(start, end, ctypes.byref(elapsed))
        per_call_us = elapsed.value / arguments.iterations * 1000

        useful = 2 * kv_bytes  # K and V each read once, if sharing is free
        print(f"{heads:12d} {share:5d}x {per_call_us:9.1f} "
              f"{useful / (per_call_us / 1e6) / 1e9:12.0f} "
              f"{per_call_us / heads:14.2f}")
        for value in pointers:
            library.flyweight_gpu_free(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
