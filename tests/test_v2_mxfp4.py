"""MXFP4 (GGML type 39) decoding.

The UD-IQ3_XXS build stores two expert down-projections as MXFP4, which is the
last weight format in the checkpoint the runtime could not read. It is 17 bytes
per 32 values: one E8M0 exponent then sixteen packed nibbles, where byte j holds
element j in its low half and element j+16 in its high half.

The codebook is the FP4 values doubled, which is why the scale is halved --
`2^(e-128)` rather than `2^(e-127)`. Getting that pairing wrong gives results
that are uniformly a factor of two out, which is easy to miss in a sum, so the
decoder is checked against an independent implementation of the same spec
reading the same bytes rather than against a plausibility bound.
"""

from __future__ import annotations

import os
import struct
import unittest

import numpy as np

from flyweight.deepseek4 import expert_matvec
from flyweight.v2 import V2Model

_CHECKPOINT_PATH = os.environ.get("DEEPSEEK4_GGUF")
# A stale path is as good as no path: the variable often outlives the file it
# named, and treating that as "configured" turns a missing checkpoint into a
# wall of errors instead of a skip.
CHECKPOINT = _CHECKPOINT_PATH if _CHECKPOINT_PATH and os.path.exists(_CHECKPOINT_PATH) else None
# The expected values here were recorded from one specific build. Pointing this
# at a different quantization would compare its numbers against the wrong
# reference and report failures that mean nothing, so require the build the
# numbers came from.
if CHECKPOINT and "IQ3_XXS" not in CHECKPOINT:
    CHECKPOINT = None

MXFP4_TENSORS = ("blk.26.ffn_down_exps.weight", "blk.42.ffn_down_exps.weight")
CODEBOOK = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32
)
BLOCK_BYTES = 17
BLOCK_ELEMENTS = 32


def _scale(exponent: int) -> float:
    """E8M0 exponent to float, halved."""
    bits = (0x00200000 << exponent) if exponent < 2 else ((exponent - 1) << 23)
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _dequantize(raw: bytes, elements: int) -> np.ndarray:
    out = np.zeros(elements, dtype=np.float32)
    for block in range(elements // BLOCK_ELEMENTS):
        base = block * BLOCK_BYTES
        scale = _scale(raw[base])
        for lane in range(16):
            byte = raw[base + 1 + lane]
            out[block * BLOCK_ELEMENTS + lane] = CODEBOOK[byte & 0x0F] * scale
            out[block * BLOCK_ELEMENTS + lane + 16] = CODEBOOK[byte >> 4] * scale
    return out


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class Mxfp4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_the_only_type_left_without_a_decoder_is_the_hash_table(self):
        # MXFP4 was the last weight format the runtime could not read. The int32
        # routing tables remain, and should: they are index data, read as
        # integers, so having no dequantization kernel is correct rather than a
        # gap. Reporting them keeps `unsupported_quant_types` honest about what
        # it means -- "no decoder" -- instead of quietly special-casing them.
        self.assertEqual(
            {kind: len(names) for kind, names in self.model.unsupported_quant_types().items()},
            {26: int(self.model.config["hash_layer_count"])},
        )

    def test_the_mxfp4_tensors_are_the_expected_ones(self):
        by_type = {}
        for tensor in self.model.tensors():
            by_type.setdefault(int(tensor["ggml_type"]), []).append(str(tensor["name"]))
        self.assertEqual(sorted(by_type.get(39, [])), sorted(MXFP4_TENSORS))

    def test_decoding_matches_an_independent_implementation(self):
        name = MXFP4_TENSORS[0]
        info = self.model.tensor(name)
        inputs, outputs = int(info["shape"][0]), int(info["shape"][1])
        self.assertEqual(inputs % BLOCK_ELEMENTS, 0)

        # One expert's first row is `inputs` values, so `inputs/32` blocks.
        row_bytes = inputs // BLOCK_ELEMENTS * BLOCK_BYTES
        raw = bytes(self.model.read_tensor_slice(name, 0, row_bytes))
        expected = _dequantize(raw, inputs)

        # A one-hot probe reads element k of the row; a random vector exercises
        # every lane and both nibble halves at once.
        rng = np.random.default_rng(21)
        probe = rng.standard_normal(inputs).astype(np.float32)
        actual = expert_matvec(self.model, name, 0, probe, outputs)[0]
        self.assertAlmostEqual(
            float(actual), float(np.dot(expected, probe)), delta=1e-2
        )

    def test_the_scale_is_halved_not_whole(self):
        # 2^(e-128), so exponent 128 is exactly 1.0. Were the scale taken as
        # 2^(e-127) every value would come out doubled.
        self.assertAlmostEqual(_scale(128), 1.0, places=6)
        self.assertAlmostEqual(_scale(129), 2.0, places=6)
        self.assertAlmostEqual(_scale(127), 0.5, places=6)

    def test_a_whole_expert_decodes_to_finite_values(self):
        name = MXFP4_TENSORS[1]
        info = self.model.tensor(name)
        inputs, outputs = int(info["shape"][0]), int(info["shape"][1])
        probe = np.ones(inputs, dtype=np.float32)
        result = expert_matvec(self.model, name, 3, probe, outputs)
        self.assertEqual(result.shape, (outputs,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertGreater(float(np.abs(result).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
