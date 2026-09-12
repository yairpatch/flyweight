"""Achieved bandwidth of the Q8 group-decode matvec kernels, per format.

Decode is one of these launches per projection, and the whole step is weight
traffic: the matrix is read once, the activation is negligible. So the figure
that matters is bytes of packed weight divided by kernel time, and the spread
between formats says which decoder is leaving bandwidth unused rather than
which format is smaller.

The matrix is sized past L2 on purpose. A single projection of this model is
~27 MB, which a 40-series L2 can hold most of, and timing a resident matrix in
a loop measures cache bandwidth rather than the DRAM bandwidth decode actually
sees. Rows are multiplied until the weights are several hundred MB; the kernel
is one block per output row, so extra rows only add blocks.

Scales are written rather than left random for the reason the parity test
documents: a random 16-bit word is NaN or Inf about one time in sixteen.
"""

from __future__ import annotations

import ctypes
import tempfile
import time
from pathlib import Path

import numpy as np

from flyweight.v2 import V2Model, V2QwenRuntime, _library

from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf

# type -> (super-block bytes, kernel prefix, fp16 scale offsets)
FORMATS = {
    "IQ2_S": (82, "iq2s", (0,)),
    "IQ2_XXS": (66, "iq2xxs", (0,)),
    "IQ3_XXS": (98, "iq3xxs", (0,)),
    "IQ2_XS": (74, "iq2xs", (0,)),
    "IQ4_XS": (136, "iq4xs", (0,)),
    "Q2_K": (84, "q2k", (80, 82)),
    "Q3_K": (110, "q3k", (108,)),
    "Q4_K": (144, "q4k", (0, 2)),
    "Q5_K": (176, "q5k", (0, 2)),
    "Q6_K": (210, "q6k", (208,)),
}

# Formats with no *_q8_matvec_transposed_warp: decode falls to the per-element
# reference decoder for these. See the dense_matvec switch in v2_runtime.cpp.
NO_Q8_KERNEL = ("IQ4_XS",)

INPUT_SIZE = 5120          # hidden size of Qwen3.8-27B
TARGET_BYTES = 512 << 20   # weights per matrix, well past any L2
WARMUP = 5
ITERATIONS = 40
BLOCK = 128

# A single sweep is not comparable across formats: half a GB of streaming per
# kernel heats the card, clocks drop, and whatever is timed last reads slow --
# a first draft of this bench showed Q4_K at 595 GB/s in one run and 349 in the
# next purely from where it fell in the order. Sweeping several times and
# keeping each format's best gives every format a shot at the same clocks,
# since throttling only ever costs time.
PASSES = 3
COOLDOWN = 0.25


class Bench:
    def __init__(self) -> None:
        self.workspace = tempfile.TemporaryDirectory()
        path = Path(self.workspace.name) / "tiny.gguf"
        build_dense_qwen35_gguf(path, DenseQwenSpec(layers=2))
        self.model = V2Model(path)
        # Only to get the kernel module compiled; the buffers below are ours.
        self.runtime = V2QwenRuntime(self.model, context_limit=128)
        self.runtime.prepare()

        lib = _library()
        lib.flyweight_gpu_alloc.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        lib.flyweight_gpu_upload_sync.argtypes = [
            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
        lib.flyweight_gpu_launch_named.argtypes = [
            ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]
        lib.flyweight_gpu_stream_sync.argtypes = [ctypes.c_uint64]
        lib.flyweight_gpu_free.argtypes = [ctypes.c_uint64]
        self.lib = lib

    def close(self) -> None:
        self.runtime.close()
        self.model.close()
        self.workspace.cleanup()

    def alloc(self, size: int) -> int:
        pointer = ctypes.c_uint64()
        if self.lib.flyweight_gpu_alloc(size, ctypes.byref(pointer)) != 0:
            raise RuntimeError(f"device allocation of {size} bytes failed")
        return pointer.value

    def launch(self, name: str, grid: int, block: int, args: list) -> None:
        array = (ctypes.c_void_p * len(args))(
            *[ctypes.addressof(value) for value in args])
        code = self.lib.flyweight_gpu_launch_named(
            name.encode(), grid, 1, block, 0, 0, array)
        if code != 0:
            raise RuntimeError(f"launch failed: {name} ({code})")

    def sync(self) -> None:
        if self.lib.flyweight_gpu_stream_sync(0) != 0:
            raise RuntimeError("stream sync failed")


def packed_weights(label: str, size: int, seed: int) -> np.ndarray:
    superblock, _, scale_offsets = FORMATS[label]
    rng = np.random.default_rng(seed)
    blocks = size // superblock
    packed = rng.integers(0, 256, size=blocks * superblock, dtype=np.uint8)
    view = packed.reshape(blocks, superblock)
    for offset in scale_offsets:
        values = rng.uniform(0.01, 0.2, size=blocks).astype(np.float16)
        view[:, offset:offset + 2] = values.view(np.uint8).reshape(blocks, 2)
    return packed


def main() -> None:
    bench = Bench()
    try:
        # One activation, shared by every format.
        rng = np.random.default_rng(7)
        activation = rng.integers(
            -126, 127, size=INPUT_SIZE).astype(np.float32)
        activation[::32] = 127.0
        vector_f32 = bench.alloc(INPUT_SIZE * 4)
        bench.lib.flyweight_gpu_upload_sync(
            vector_f32, activation.ctypes.data_as(ctypes.c_void_p),
            INPUT_SIZE * 4)
        quantized = bench.alloc(INPUT_SIZE)
        scales = bench.alloc((INPUT_SIZE // 32) * 2)
        columns = ctypes.c_int32(INPUT_SIZE)
        bench.launch("quantize_q8_blocks", (INPUT_SIZE + 31) // 32, 32,
                     [ctypes.c_uint64(vector_f32), ctypes.c_uint64(quantized),
                      ctypes.c_uint64(scales), columns])
        bench.sync()

        print(f"input {INPUT_SIZE}, block {BLOCK}, {ITERATIONS} iterations "
              f"after {WARMUP} warmup, best of {PASSES} passes\n")
        print(f"{'format':9} {'rows':>7} {'MiB':>7} "
              f"{'q8 GB/s':>10} {'ref GB/s':>10}")

        results = {}
        reference = {}
        shapes = {}
        for _ in range(PASSES):
            for label, (superblock, prefix, _) in FORMATS.items():
                blocks_per_row = INPUT_SIZE // 256
                row_bytes = blocks_per_row * superblock
                rows = TARGET_BYTES // row_bytes
                weight_bytes = rows * row_bytes
                shapes[label] = (rows, weight_bytes)

                packed = packed_weights(
                    label, weight_bytes, seed=hash(label) % 997)
                weights = bench.alloc(weight_bytes)
                bench.lib.flyweight_gpu_upload_sync(
                    weights, packed.ctypes.data_as(ctypes.c_void_p),
                    weight_bytes)
                output = bench.alloc(rows * 4)
                row_count = ctypes.c_int32(rows)

                def time_kernel(name: str, block: int, args: list) -> float:
                    for _ in range(WARMUP):
                        bench.launch(name, rows, block, args)
                    bench.sync()
                    time.sleep(COOLDOWN)
                    start = time.perf_counter()
                    for _ in range(ITERATIONS):
                        bench.launch(name, rows, block, args)
                    bench.sync()
                    return (time.perf_counter() - start) / ITERATIONS

                def record(table: dict, elapsed: float) -> None:
                    gbps = weight_bytes / elapsed / 1e9
                    table[label] = max(table.get(label, 0.0), gbps)

                # The per-element decoder, which every format has and which is
                # what decode actually runs for the formats in NO_Q8_KERNEL.
                record(reference, time_kernel(
                    f"{prefix}_matvec_transposed", 256,
                    [ctypes.c_uint64(weights), ctypes.c_uint64(vector_f32),
                     ctypes.c_uint64(output), columns, row_count]))

                if label not in NO_Q8_KERNEL:
                    record(results, time_kernel(
                        f"{prefix}_q8_matvec_transposed_warp", BLOCK,
                        [ctypes.c_uint64(weights), ctypes.c_uint64(quantized),
                         ctypes.c_uint64(scales), ctypes.c_uint64(output),
                         columns, row_count]))

                bench.lib.flyweight_gpu_free(weights)
                bench.lib.flyweight_gpu_free(output)

        for label in FORMATS:
            rows, weight_bytes = shapes[label]
            gbps = results.get(label)
            print(f"{label:9} {rows:7d} {weight_bytes >> 20:7d} "
                  f"{'-- none --' if gbps is None else f'{gbps:8.1f}':>10} "
                  f"{reference[label]:10.1f}")

        best = max(results.values())
        print(f"\nbest Q8 kernel {best:.1f} GB/s; IQ2_S at "
              f"{results['IQ2_S'] / best * 100:.0f}% of it")
        for label in NO_Q8_KERNEL:
            print(f"{label} has no Q8 kernel and runs at "
                  f"{reference[label]:.1f} GB/s, "
                  f"{best / reference[label]:.1f}x off the best")
    finally:
        bench.close()


if __name__ == "__main__":
    main()
