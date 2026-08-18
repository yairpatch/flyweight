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
    "IQ2_S": (82, "iq2s_matvec_transposed", "iq2s", (0,)),
    "IQ2_XS": (74, "iq2xs_matvec_transposed", "iq2xs", (0,)),
}

# The LM head has no Q8 group-decode variant for these; they never carry an
# output projection in the checkpoints that use them.
NO_Q8_LM_HEAD = {"IQ2_S", "IQ2_XS"}

COLUMNS = 512      # two 256-value super-blocks per row
ROWS = 96

# Rows per launch of the *_q8_matvec_transposed_rows kernels; COLIBRI_Q8_ROWS
# in native/include/colibri_v2_qwen_kernels.hpp.
Q8_ROW_BATCH = 8

# Formats with a batched multi-row matvec. Prefill dispatches on these; see the
# rows_kernel switch in native/src/v2_mtp_verifier.inc.
BATCHED_ROWS = ("IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_XXS")


def _cuda_available() -> bool:
    return bool(V2Model.gpu_info()["available"])


class Q8KernelHarness:
    """Device buffers and raw kernel launches, shared by the cases below.

    Not a TestCase itself so unittest does not collect it, and so the cases
    that mix it in do not inherit each other's tests.
    """

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

    def _launch(self, name: str, grid: int, block: int, args: list,
                grid_y: int = 1) -> None:
        array = (ctypes.c_void_p * len(args))(
            *[ctypes.addressof(value) for value in args])
        self.assertEqual(
            self._lib.colibri_gpu_launch_named(
                name.encode(), grid, grid_y, block, 0, 0, array),
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


class Q8MatvecParityTests(Q8KernelHarness, unittest.TestCase):
    """Each group-decode kernel must agree with its per-element reference."""

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
            if label in NO_Q8_LM_HEAD:
                continue
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


class BatchedRowsTests(Q8KernelHarness, unittest.TestCase):
    """The batched matvecs must match the single-row ones they replace.

    Prefill runs a batch of token rows through one weight matrix.
    ``<prefix>_q8_matvec_transposed_rows`` decodes each weight group once and
    dots it against every row, so unlike the single-row kernel it reads the
    matrix once per launch rather than once per row -- the whole point, and also
    the only place a row-indexing mistake can hide, since row 0 is laid out
    identically either way and would pass on its own.

    The scale rows are deliberately on the padded stride the runtime uses
    (float-sized, though the values are halves): a kernel that ignored
    ``scale_stride`` and packed the rows tightly would still get row 0 right.
    """

    def _single_row_reference(self, label: str, packed_ptr,
                              activations: np.ndarray, batch: int) -> np.ndarray:
        """Per-row quantize + single-row matvec, the path being replaced."""
        prefix = FORMATS[label][2]
        expected = np.zeros((batch, ROWS), dtype=np.float32)
        columns = ctypes.c_int32(COLUMNS)
        rows = ctypes.c_int32(ROWS)
        for row in range(batch):
            vector_f32 = self._alloc(COLUMNS * 4)
            contiguous = np.ascontiguousarray(activations[row])
            self.assertEqual(self._lib.colibri_gpu_upload_sync(
                vector_f32, contiguous.ctypes.data_as(ctypes.c_void_p),
                COLUMNS * 4), 0)
            quantized = self._alloc(COLUMNS)
            scales = self._alloc((COLUMNS // 32) * 2)
            output = self._alloc(ROWS * 4)
            self._launch("quantize_q8_blocks", (COLUMNS + 31) // 32, 32,
                         [ctypes.c_uint64(vector_f32), ctypes.c_uint64(quantized),
                          ctypes.c_uint64(scales), columns])
            self._launch(f"{prefix}_q8_matvec_transposed_warp", ROWS, 128,
                         [packed_ptr, ctypes.c_uint64(quantized),
                          ctypes.c_uint64(scales), ctypes.c_uint64(output),
                          columns, rows])
            expected[row] = self._download(output, ROWS)
        return expected

    def _upload_weights(self, label: str, seed: int):
        superblock = FORMATS[label][0]
        weight_bytes = (COLUMNS // 256) * ROWS * superblock
        packed = self._packed_weights(label, weight_bytes, seed)
        weights = self._alloc(weight_bytes)
        self.assertEqual(self._lib.colibri_gpu_upload_sync(
            weights, packed.ctypes.data_as(ctypes.c_void_p), weight_bytes), 0)
        return ctypes.c_uint64(weights)

    def test_batched_rows_match_single_row_kernel(self):
        for index, label in enumerate(BATCHED_ROWS):
            prefix = FORMATS[label][2]
            packed_ptr = self._upload_weights(label, seed=41 + index)
            # 1 and the full batch are the edges; 3 leaves the tail of the
            # unrolled row loop predicated off, which is where an unpredicated
            # accumulator would read another row's activations.
            for batch in (1, 3, Q8_ROW_BATCH):
                with self.subTest(quantization=label, batch=batch):
                    activations = np.stack([
                        self._exact_q8_activation(41 + row)
                        for row in range(batch)])
                    expected = self._single_row_reference(
                        label, packed_ptr, activations, batch)

                    # Halves per scale row: the runtime lays these out on a
                    # float-sized stride even though the kernels write __half.
                    scale_stride = (COLUMNS // 32) * 2
                    vectors_f32 = self._alloc(batch * COLUMNS * 4)
                    flat = np.ascontiguousarray(activations.reshape(-1))
                    self.assertEqual(self._lib.colibri_gpu_upload_sync(
                        vectors_f32, flat.ctypes.data_as(ctypes.c_void_p),
                        batch * COLUMNS * 4), 0)
                    quantized = self._alloc(batch * COLUMNS)
                    scales = self._alloc(batch * scale_stride * 2)
                    output = self._alloc(batch * ROWS * 4)

                    columns = ctypes.c_int32(COLUMNS)
                    rows = ctypes.c_int32(ROWS)
                    count = ctypes.c_int32(batch)
                    stride = ctypes.c_int32(scale_stride)
                    q8_ptr = ctypes.c_uint64(quantized)
                    scale_ptr = ctypes.c_uint64(scales)

                    self._launch(
                        "quantize_q8_blocks_rows", (COLUMNS + 31) // 32, 32,
                        [ctypes.c_uint64(vectors_f32), q8_ptr, scale_ptr,
                         columns, stride], grid_y=batch)
                    self._launch(
                        f"{prefix}_q8_matvec_transposed_rows", ROWS, 128,
                        [packed_ptr, q8_ptr, scale_ptr, ctypes.c_uint64(output),
                         columns, rows, count, stride])
                    actual = self._download(
                        output, batch * ROWS).reshape(batch, ROWS)

                    self.assertTrue(
                        np.isfinite(actual).all(),
                        f"batched {label} produced non-finite output")
                    # Both kernels sum the same products in the same order;
                    # only the final scale multiply is reassociated.
                    scale = max(1.0, float(np.abs(expected).max()))
                    np.testing.assert_allclose(
                        actual, expected, rtol=2e-4, atol=2e-4 * scale,
                        err_msg=f"batched {label} disagrees with the "
                                "single-row kernel")

    def test_reference_rows_are_distinct(self):
        """Guard the guard: identical rows would pass any row-indexing bug."""
        for index, label in enumerate(BATCHED_ROWS):
            with self.subTest(quantization=label):
                packed_ptr = self._upload_weights(label, seed=41 + index)
                activations = np.stack([
                    self._exact_q8_activation(41 + row) for row in range(3)])
                expected = self._single_row_reference(
                    label, packed_ptr, activations, 3)
                for row in range(1, 3):
                    self.assertGreater(
                        float(np.abs(expected[row] - expected[0]).max()), 1e-3,
                        "reference rows are too similar to detect a mix-up")


if __name__ == "__main__":
    unittest.main()
