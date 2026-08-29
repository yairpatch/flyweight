#!/usr/bin/env python3
"""Achieved bandwidth of one dense matvec kernel, in isolation.

Times a single quantized matvec back to back on its own stream, so the number
is the kernel's own throughput: no surrounding decode structure, no launch gaps
from neighbouring kernels, no reduction across layers. Compare it against the
card's achievable DRAM bandwidth to tell a slow kernel from a slow schedule.

A runtime is prepared first only to get the kernel module compiled; the buffers
below are allocated and launched against directly.

CAUTION for the codebook formats (IQ2_XXS, IQ3_XXS, ...): the weight buffer is
never initialized, so every lane decodes the same grid entry and the codebook
load degenerates into a broadcast. That flatters those kernels by 5-11x against
their real divergent-index behaviour, and it is exactly what hid a pathological
__constant__ lookup for two rounds of tuning. The K-quants have no codebook and
are unaffected. Fill the buffer with representative weights before trusting an
IQ number here.
"""

import argparse
import ctypes

from flyweight.v2 import V2Model, V2QwenRuntime, _library

SUPERBLOCK_BYTES = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 16: 66}
KERNELS = {
    10: "q2k_q8_matvec_transposed_warp",
    11: "q3k_q8_matvec_transposed_warp",
    12: "q4k_q8_matvec_transposed_warp",
    13: "q5k_q8_matvec_transposed_warp",
    14: "q6k_q8_matvec_transposed_warp",
    16: "iq2xxs_q8_matvec_transposed_warp",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--type", type=int, action="append", dest="types")
    parser.add_argument("--rows", type=int, default=17408)
    parser.add_argument("--columns", type=int, default=5120)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--rows-per-block", type=int, default=1)
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
    library.flyweight_gpu_event_record.argtypes = [ctypes.c_uint64,
                                                 ctypes.c_uint64]

    rows, columns = arguments.rows, arguments.columns
    stream = ctypes.c_uint64()
    library.flyweight_gpu_stream_create(ctypes.byref(stream))
    start, end = ctypes.c_uint64(), ctypes.c_uint64()
    library.flyweight_gpu_timed_event_create(ctypes.byref(start))
    library.flyweight_gpu_timed_event_create(ctypes.byref(end))

    for quant in arguments.types or [12]:
        weight_bytes = (columns // 256) * rows * SUPERBLOCK_BYTES[quant]
        pointers = []
        for size in (weight_bytes, columns, (columns // 32) * 2, rows * 4):
            value = ctypes.c_uint64()
            if library.flyweight_gpu_alloc(size, ctypes.byref(value)):
                raise SystemExit("device allocation failed")
            pointers.append(value)

        in_size, out_size = ctypes.c_int32(columns), ctypes.c_int32(rows)
        args = (ctypes.c_void_p * 6)(
            *[ctypes.addressof(p) for p in pointers],
            ctypes.addressof(in_size), ctypes.addressof(out_size))
        grid = (rows + arguments.rows_per_block - 1) // arguments.rows_per_block
        name = KERNELS[quant].encode()

        def run(count: int) -> None:
            for _ in range(count):
                if library.flyweight_gpu_launch_named(
                    name, grid, 1, arguments.block, 0, stream, args
                ):
                    raise SystemExit(f"launch failed: {name!r}")

        run(50)
        library.flyweight_gpu_stream_sync(stream)
        library.flyweight_gpu_event_record(start, stream)
        run(arguments.iterations)
        library.flyweight_gpu_event_record(end, stream)
        library.flyweight_gpu_stream_sync(stream)

        elapsed = ctypes.c_float()
        library.flyweight_gpu_event_elapsed(start, end, ctypes.byref(elapsed))
        per_call = elapsed.value / arguments.iterations
        print(f"{KERNELS[quant]:34s} {weight_bytes / 2**20:7.1f} MiB "
              f"{per_call * 1000:8.1f} us "
              f"{weight_bytes / (per_call / 1000) / 1e9:6.0f} GB/s")
        for value in pointers:
            library.flyweight_gpu_free(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
