"""Group-decode matvec kernels against the per-element kernels they replaced.

Every quantized projection has two GPU implementations: a reference that calls
``<type>_value()`` once per weight, and a kernel that quantizes the activation
to 32-value Q8 blocks and dots whole weight groups with DP4A. The second is
what decode actually runs, and nothing else in the suite exercises it -- the
published dense checkpoints are all several GB, so the end-to-end parity check
cannot run here.

The weights are random bytes everywhere except the fp16 super-block scales,
which are written explicitly: an arbitrary 16-bit pattern is NaN or Inf about
one time in sixteen, and a single one poisons the whole row. Every other field
-- 2/3/4/5/6-bit quants, 4/6-bit packed scales, sign selectors, codebook
indices -- decodes any byte pattern to something finite, so randomness there is
legitimate and correctness is agreement between the two decoders, not realism.

The activation is chosen to survive Q8 quantization exactly (integers in
[-127, 127] with a 127 in each block, so the block scale is 1.0), which leaves
float accumulation order as the only source of disagreement.
"""

from __future__ import annotations

import ctypes
import tempfile
import unittest
from pathlib import Path

import numpy as np

from colibri_next.v2 import V2Model, V2QwenRuntime, _library

from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf


# type -> (super-block bytes, reference kernel, prefix, fp16 scale offsets)
FORMATS = {
    "Q2_K": (84, "q2k_matvec_transposed", "q2k", (80, 82)),
    "Q3_K": (110, "q3k_matvec_transposed", "q3k", (108,)),
    "Q4_K": (144, "q4k_matvec_transposed", "q4k", (0, 2)),
    "Q5_K": (176, "q5k_matvec_transposed", "q5k", (0, 2)),
    "Q6_K": (210, "q6k_matvec_transposed", "q6k", (208,)),
    "IQ2_XXS": (66, "iq2xxs_matvec_transposed", "iq2xxs", (0,)),
    "IQ3_XXS": (98, "iq3xxs_matvec_transposed", "iq3xxs", (0,)),
}

COLUMNS = 512      # two 256-value super-blocks per row
ROWS = 96


def _cuda_available() -> bool:
    return bool(V2Model.gpu_info()["available"])


class Q8MatvecParityTests(unittest.TestCase):
    """Each group-decode kernel must agree with its per-element reference."""

    @classmethod
    def setUpClass(cls):
        if not _cuda_available():
            raise unittest.SkipTest("CUDA is unavailable")
        cls._workspace = tempfile.TemporaryDirectory()
        path = Path(cls._workspace.name) / "tiny.gguf"
        build_dense_qwen35_gguf(path, DenseQwenSpec(layers=2))
        cls._model = V2Model(path)
        # Only to get the kernel module compiled; the buffers below are ours.
        cls._runtime = V2QwenRuntime(cls._model, context_limit=128)
        cls._runtime.prepare()

        cls._lib = _library()
        cls._lib.colibri_gpu_alloc.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        cls._lib.colibri_gpu_upload_sync.argtypes = [
            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
        cls._lib.colibri_gpu_download.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        cls._lib.colibri_gpu_launch_named.argtypes = [
            ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]
        cls._lib.colibri_gpu_stream_sync.argtypes = [ctypes.c_uint64]
        # Without this the 64-bit device pointer is marshalled as a C int and
        # the memset lands somewhere else entirely.
        cls._lib.colibri_gpu_memset.argtypes = [
            ctypes.c_uint64, ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint64]
        cls._lib.colibri_gpu_free.argtypes = [ctypes.c_uint64]

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_runtime"):
            cls._runtime.close()
            cls._model.close()
            cls._workspace.cleanup()

    def _alloc(self, size: int) -> int:
        pointer = ctypes.c_uint64()
        self.assertEqual(
            self._lib.colibri_gpu_alloc(size, ctypes.byref(pointer)), 0,
            "device allocation failed")
        return pointer.value

    def _launch(self, name: str, grid: int, block: int, args: list) -> None:
        array = (ctypes.c_void_p * len(args))(
            *[ctypes.addressof(value) for value in args])
        self.assertEqual(
            self._lib.colibri_gpu_launch_named(
                name.encode(), grid, 1, block, 0, 0, array),
            0, f"launch failed: {name}")

    def _download(self, pointer: int, count: int) -> np.ndarray:
        out = np.zeros(count, dtype=np.float32)
        self.assertEqual(self._lib.colibri_gpu_stream_sync(0), 0)
        self.assertEqual(
            self._lib.colibri_gpu_download(
                out.ctypes.data_as(ctypes.c_void_p), pointer,
                out.nbytes, 0), 0)
        self.assertEqual(self._lib.colibri_gpu_stream_sync(0), 0)
        return out

    def _packed_weights(self, label: str, size: int, seed: int) -> np.ndarray:
        """Random payload with finite fp16 super-block scales.

        A uniformly random 16-bit word is NaN or Inf roughly one time in
        sixteen (all-ones exponent), and one of those propagates through the
        whole row dot, so the scale fields are written rather than drawn.
        """
        superblock, _, _, scale_offsets = FORMATS[label]
        rng = np.random.default_rng(seed)
        packed = rng.integers(0, 256, size=size, dtype=np.uint8)
        blocks = size // superblock
        for offset in scale_offsets:
            values = rng.uniform(0.01, 0.2, size=blocks).astype(np.float16)
            raw = values.view(np.uint8).reshape(blocks, 2)
            for block in range(blocks):
                base = block * superblock + offset
                packed[base:base + 2] = raw[block]
        return packed

    def _exact_q8_activation(self, seed: int) -> np.ndarray:
        """A vector whose Q8 block quantization is lossless.

        quantize_q8_blocks divides each 32-value block by absmax/127, so a
        block of integers containing a +/-127 round-trips exactly.
        """
        rng = np.random.default_rng(seed)
        vector = rng.integers(-126, 127, size=COLUMNS).astype(np.float32)
        for start in range(0, COLUMNS, 32):
            vector[start] = 127.0
        return vector

    def _run_format(self, label: str, seed: int):
        superblock, reference_kernel, prefix, _ = FORMATS[label]
        weight_bytes = (COLUMNS // 256) * ROWS * superblock

        packed = self._packed_weights(label, weight_bytes, seed)
        activation = self._exact_q8_activation(seed)

        weights = self._alloc(weight_bytes)
        self.assertEqual(self._lib.colibri_gpu_upload_sync(
            weights, packed.ctypes.data_as(ctypes.c_void_p), weight_bytes), 0)
        vector_f32 = self._alloc(COLUMNS * 4)
        self.assertEqual(self._lib.colibri_gpu_upload_sync(
            vector_f32, activation.ctypes.data_as(ctypes.c_void_p),
            COLUMNS * 4), 0)
        quantized = self._alloc(COLUMNS)
        scales = self._alloc((COLUMNS // 32) * 2)
        reference_out = self._alloc(ROWS * 4)
        group_out = self._alloc(ROWS * 4)

        columns = ctypes.c_int32(COLUMNS)
        rows = ctypes.c_int32(ROWS)
        packed_ptr = ctypes.c_uint64(weights)
        f32_ptr = ctypes.c_uint64(vector_f32)
        q8_ptr = ctypes.c_uint64(quantized)
        scale_ptr = ctypes.c_uint64(scales)

        # Reference: one block per row, a decode call per weight.
        self._launch(reference_kernel, ROWS, 256,
                     [packed_ptr, f32_ptr, ctypes.c_uint64(reference_out),
                      columns, rows])
        expected = self._download(reference_out, ROWS)

        self._launch("quantize_q8_blocks", (COLUMNS + 31) // 32, 32,
                     [f32_ptr, q8_ptr, scale_ptr, columns])
        self._launch(f"{prefix}_q8_matvec_transposed_warp", ROWS, 128,
                     [packed_ptr, q8_ptr, scale_ptr,
                      ctypes.c_uint64(group_out), columns, rows])
        actual = self._download(group_out, ROWS)
        return expected, actual

    def test_group_decode_matches_per_element_reference(self):
        for index, label in enumerate(FORMATS):
            with self.subTest(quantization=label):
                expected, actual = self._run_format(label, seed=11 + index)
                self.assertTrue(
                    np.isfinite(actual).all(), f"{label} produced non-finite output")
                # Both sides sum the same products; only the order differs, so
                # the tolerance covers float reassociation, not decode error.
                scale = max(1.0, float(np.abs(expected).max()))
                np.testing.assert_allclose(
                    actual, expected, rtol=2e-3, atol=2e-3 * scale,
                    err_msg=f"{label} group decode disagrees with per-element decode")

    def test_reference_output_is_not_degenerate(self):
        """Guard the guard: a kernel returning zeros would pass a loose compare."""
        expected, _ = self._run_format("Q4_K", seed=5)
        self.assertGreater(float(np.abs(expected).max()), 1e-3)
        self.assertGreater(int(np.count_nonzero(expected)), ROWS // 2)

    def test_lm_head_argmax_matches_matvec(self):
        """The fused LM head must pick the row the matvec's maximum names."""
        for index, label in enumerate(FORMATS):
            with self.subTest(quantization=label):
                superblock, _, prefix, _ = FORMATS[label]
                weight_bytes = (COLUMNS // 256) * ROWS * superblock
                packed = self._packed_weights(label, weight_bytes, 29 + index)
                activation = self._exact_q8_activation(29 + index)

                weights = self._alloc(weight_bytes)
                self._lib.colibri_gpu_upload_sync(
                    weights, packed.ctypes.data_as(ctypes.c_void_p), weight_bytes)
                vector_f32 = self._alloc(COLUMNS * 4)
                self._lib.colibri_gpu_upload_sync(
                    vector_f32, activation.ctypes.data_as(ctypes.c_void_p),
                    COLUMNS * 4)
                quantized = self._alloc(COLUMNS)
                scales = self._alloc((COLUMNS // 32) * 2)
                logits = self._alloc(ROWS * 4)
                winner = self._alloc(8)

                columns = ctypes.c_int32(COLUMNS)
                rows = ctypes.c_int32(ROWS)
                packed_ptr = ctypes.c_uint64(weights)
                f32_ptr = ctypes.c_uint64(vector_f32)
                q8_ptr = ctypes.c_uint64(quantized)
                scale_ptr = ctypes.c_uint64(scales)

                self._launch("quantize_q8_blocks", (COLUMNS + 31) // 32, 32,
                             [f32_ptr, q8_ptr, scale_ptr, columns])
                self._launch(f"{prefix}_q8_matvec_transposed_warp", ROWS, 128,
                             [packed_ptr, q8_ptr, scale_ptr,
                              ctypes.c_uint64(logits), columns, rows])
                projected = self._download(logits, ROWS)

                self.assertEqual(self._lib.colibri_gpu_memset(winner, 0, 8, 0), 0)
                self._launch(f"{prefix}_q8_lm_head_argmax_warp",
                             (ROWS + 7) // 8, 256,
                             [packed_ptr, q8_ptr, scale_ptr,
                              ctypes.c_uint64(winner), columns, rows])
                self.assertEqual(self._lib.colibri_gpu_stream_sync(0), 0)
                packed_winner = ctypes.c_uint64()
                self.assertEqual(self._lib.colibri_gpu_download(
                    ctypes.byref(packed_winner), winner, 8, 0), 0)
                self.assertEqual(self._lib.colibri_gpu_stream_sync(0), 0)
                # The kernels pack ~row in the low word, matching decode's
                # 0xffffffff - value unpacking.
                chosen = 0xFFFFFFFF - (packed_winner.value & 0xFFFFFFFF)
                self.assertEqual(
                    chosen, int(np.argmax(projected)),
                    f"{label} LM head chose a different row than the matvec")


if __name__ == "__main__":
    unittest.main()
