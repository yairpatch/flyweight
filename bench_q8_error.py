"""How much the Q8 activation path perturbs each format's dot product.

Routing a projection through *_q8_matvec_transposed_warp quantizes the
activation to int8 in 32-value blocks, which the exact per-element decoder does
not do. That is a real numerical change, and the question when adding a format
to the Q8 path is not whether the error is zero -- it is whether the format is
any worse than the ones already running there.

So this measures relative error against the per-element decoder for every
format, on a Gaussian activation rather than the exactly-representable one the
parity test uses: the parity test is asking whether the decoders agree, this is
asking what the quantization costs. A format in line with its neighbours is as
safe as the neighbours.
"""

from __future__ import annotations

import ctypes
import tempfile
from pathlib import Path

import numpy as np

from flyweight.v2 import V2Model, V2QwenRuntime, _library

from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf

# type -> (super-block bytes, kernel prefix, fp16 scale offsets)
#
# IQ1_M is the one format whose super-block scale is not a field: its sixteen
# bits live in the top nibbles of the four scale halfwords at 48..55, so it has
# no byte offset to write and is spelled as None here and scattered below.
FORMATS = {
    "IQ1_S": (50, "iq1s", (0,)),
    "IQ1_M": (56, "iq1m", None),
    "IQ2_S": (82, "iq2s", (0,)),
    "IQ2_XXS": (66, "iq2xxs", (0,)),
    "IQ3_XXS": (98, "iq3xxs", (0,)),
    "IQ2_XS": (74, "iq2xs", (0,)),
    "Q2_K": (84, "q2k", (80, 82)),
    "Q3_K": (110, "q3k", (108,)),
    "Q4_K": (144, "q4k", (0, 2)),
    "Q5_K": (176, "q5k", (0, 2)),
    "Q6_K": (210, "q6k", (208,)),
}

INPUT_SIZE = 5120
ROWS = 4096
BLOCK = 128


def main() -> None:
    workspace = tempfile.TemporaryDirectory()
    path = Path(workspace.name) / "tiny.gguf"
    build_dense_qwen35_gguf(path, DenseQwenSpec(layers=2))
    model = V2Model(path)
    runtime = V2QwenRuntime(model, context_limit=128)
    runtime.prepare()

    lib = _library()
    lib.flyweight_gpu_alloc.argtypes = [
        ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    lib.flyweight_gpu_upload_sync.argtypes = [
        ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
    lib.flyweight_gpu_download.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
    lib.flyweight_gpu_launch_named.argtypes = [
        ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]
    lib.flyweight_gpu_stream_sync.argtypes = [ctypes.c_uint64]
    lib.flyweight_gpu_free.argtypes = [ctypes.c_uint64]

    def alloc(size: int) -> int:
        pointer = ctypes.c_uint64()
        if lib.flyweight_gpu_alloc(size, ctypes.byref(pointer)) != 0:
            raise RuntimeError("device allocation failed")
        return pointer.value

    def launch(name: str, grid: int, block: int, args: list) -> None:
        array = (ctypes.c_void_p * len(args))(
            *[ctypes.addressof(value) for value in args])
        code = lib.flyweight_gpu_launch_named(
            name.encode(), grid, 1, block, 0, 0, array)
        if code != 0:
            raise RuntimeError(f"launch failed: {name} ({code})")

    def download(pointer: int, count: int) -> np.ndarray:
        out = np.zeros(count, dtype=np.float32)
        lib.flyweight_gpu_stream_sync(0)
        lib.flyweight_gpu_download(
            out.ctypes.data_as(ctypes.c_void_p), pointer, out.nbytes, 0)
        lib.flyweight_gpu_stream_sync(0)
        return out

    try:
        # A realistic residual-stream activation, not one chosen to survive Q8.
        rng = np.random.default_rng(11)
        activation = rng.normal(0.0, 1.0, size=INPUT_SIZE).astype(np.float32)
        vector_f32 = alloc(INPUT_SIZE * 4)
        lib.flyweight_gpu_upload_sync(
            vector_f32, activation.ctypes.data_as(ctypes.c_void_p),
            INPUT_SIZE * 4)
        quantized = alloc(INPUT_SIZE)
        scales = alloc((INPUT_SIZE // 32) * 2)
        columns = ctypes.c_int32(INPUT_SIZE)
        launch("quantize_q8_blocks", (INPUT_SIZE + 31) // 32, 32,
               [ctypes.c_uint64(vector_f32), ctypes.c_uint64(quantized),
                ctypes.c_uint64(scales), columns])
        lib.flyweight_gpu_stream_sync(0)

        print(f"input {INPUT_SIZE}, {ROWS} rows, Gaussian activation\n")
        print(f"{'format':9} {'rel L2':>10} {'max rel':>10} {'corr':>9}")

        for label, (superblock, prefix, scale_offsets) in FORMATS.items():
            blocks_per_row = INPUT_SIZE // 256
            weight_bytes = ROWS * blocks_per_row * superblock

            seed = abs(hash(label)) % 997
            gen = np.random.default_rng(seed)
            blocks = weight_bytes // superblock
            packed = gen.integers(
                0, 256, size=blocks * superblock, dtype=np.uint8)
            view = packed.reshape(blocks, superblock)
            values = gen.uniform(0.01, 0.2, size=blocks).astype(np.float16)
            if scale_offsets is None:
                # IQ1_M: nibble `half` of the f16 goes in the top nibble of
                # scale halfword `half`, whose low twelve bits are sub-scales
                # and stay random.
                bits = values.view(np.uint16)
                for half in range(4):
                    word = view[:, 48 + half * 2 : 50 + half * 2].copy().view(
                        np.uint16).reshape(blocks)
                    nibble = (bits >> (4 * half)) & 0xF
                    word = (word & 0x0FFF) | (nibble << 12)
                    view[:, 48 + half * 2 : 50 + half * 2] = (
                        word.view(np.uint8).reshape(blocks, 2))
            else:
                for offset in scale_offsets:
                    view[:, offset:offset + 2] = (
                        values.view(np.uint8).reshape(blocks, 2))

            weights = alloc(weight_bytes)
            lib.flyweight_gpu_upload_sync(
                weights, packed.ctypes.data_as(ctypes.c_void_p), weight_bytes)
            exact_out = alloc(ROWS * 4)
            q8_out = alloc(ROWS * 4)
            row_count = ctypes.c_int32(ROWS)

            launch(f"{prefix}_matvec_transposed", ROWS, 256,
                   [ctypes.c_uint64(weights), ctypes.c_uint64(vector_f32),
                    ctypes.c_uint64(exact_out), columns, row_count])
            launch(f"{prefix}_q8_matvec_transposed_warp", ROWS, BLOCK,
                   [ctypes.c_uint64(weights), ctypes.c_uint64(quantized),
                    ctypes.c_uint64(scales), ctypes.c_uint64(q8_out),
                    columns, row_count])

            exact = download(exact_out, ROWS)
            approximate = download(q8_out, ROWS)
            error = approximate - exact
            rel_l2 = np.linalg.norm(error) / np.linalg.norm(exact)
            max_rel = np.max(np.abs(error)) / np.max(np.abs(exact))
            corr = float(np.corrcoef(exact, approximate)[0, 1])
            print(f"{label:9} {rel_l2:10.2e} {max_rel:10.2e} {corr:9.6f}")

            lib.flyweight_gpu_free(weights)
            lib.flyweight_gpu_free(exact_out)
            lib.flyweight_gpu_free(q8_out)
    finally:
        runtime.close()
        model.close()
        workspace.cleanup()


if __name__ == "__main__":
    main()
