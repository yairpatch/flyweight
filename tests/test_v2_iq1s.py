"""IQ1_S (GGML type 19) decoding.

The UD-IQ1_S build stores its routed expert weights as IQ1_S, which is 50 bytes
per 256 values -- 1.5625 bits each. A block holds one fp16 scale, then per group
of 32 an 11-bit index per 8 weights: eight bits from `qs` and three more from
`qh`, pointing into a shared 2048-entry grid of eight int8 weights. `qh` also
carries that group's own scale multiplier in bits 12-14 and, in its top bit, the
sign of a delta added to every weight in the group.

Nothing about that is guessable from the block size, so the decoder is checked
against an independent implementation of the same spec reading the same bytes.
"""

from __future__ import annotations

import os
import re
import struct
import unittest

import numpy as np

from flyweight.deepseek4 import expert_matvec
from flyweight.v2 import V2Model

_CHECKPOINT_PATH = os.environ.get("DEEPSEEK4_GGUF")
CHECKPOINT = _CHECKPOINT_PATH if _CHECKPOINT_PATH and os.path.exists(_CHECKPOINT_PATH) else None
GRID_SOURCE = "/home/yair/Desktop/llama.cpp-ref/ggml/src/ggml-common.h"

BLOCK_BYTES = 50
BLOCK_ELEMENTS = 256
DELTA = 0.125


def _grid():
    src = open(GRID_SOURCE).read()
    start = src.index("GGML_TABLE_BEGIN(uint64_t, iq1s_grid, NGRID_IQ1S)")
    end = src.index("GGML_TABLE_END()", start)
    return [int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", src[start:end])]


def _dequantize(raw: bytes, elements: int, grid) -> np.ndarray:
    out = np.zeros(elements, dtype=np.float32)
    for block in range(elements // BLOCK_ELEMENTS):
        base = block * BLOCK_BYTES
        scale = np.frombuffer(raw[base : base + 2], dtype=np.float16)[0].astype(np.float32)
        qs = raw[base + 2 : base + 34]
        for group in range(8):
            qh = struct.unpack_from("<H", raw, base + 34 + group * 2)[0]
            group_scale = scale * (2 * ((qh >> 12) & 7) + 1)
            delta = -DELTA if (qh & 0x8000) else DELTA
            for part in range(4):
                entry = grid[qs[4 * group + part] | (((qh >> (3 * part)) & 7) << 8)]
                for lane in range(8):
                    weight = (entry >> (8 * lane)) & 0xFF
                    if weight > 127:
                        weight -= 256
                    out[block * 256 + group * 32 + part * 8 + lane] = (
                        group_scale * (weight + delta)
                    )
    return out


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class Iq1sTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        cls.names = [
            str(t["name"]) for t in cls.model.tensors() if int(t["ggml_type"]) == 19
        ]
        if not cls.names:
            cls.model.close()
            raise unittest.SkipTest("this checkpoint stores nothing as IQ1_S")

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_iq1s_is_decodable(self):
        self.assertNotIn(19, self.model.unsupported_quant_types())

    def test_the_only_type_left_without_a_decoder_is_the_hash_table(self):
        # Index data, not weights, so having no dequantizer is correct.
        self.assertEqual(
            {k: len(v) for k, v in self.model.unsupported_quant_types().items()},
            {26: int(self.model.config["hash_layer_count"])},
        )

    @unittest.skipUnless(os.path.exists(GRID_SOURCE), "reference grid table unavailable")
    def test_decoding_matches_an_independent_implementation(self):
        name = self.names[0]
        info = self.model.tensor(name)
        inputs, outputs = int(info["shape"][0]), int(info["shape"][1])
        self.assertEqual(inputs % BLOCK_ELEMENTS, 0)
        raw = bytes(
            self.model.read_tensor_slice(name, 0, inputs // BLOCK_ELEMENTS * BLOCK_BYTES)
        )
        expected = _dequantize(raw, inputs, _grid())
        probe = np.random.default_rng(5).standard_normal(inputs).astype(np.float32)
        actual = expert_matvec(self.model, name, 0, probe, outputs)[0]
        self.assertAlmostEqual(float(actual), float(np.dot(expected, probe)), places=4)

    def test_a_whole_expert_decodes_to_finite_values(self):
        name = self.names[0]
        info = self.model.tensor(name)
        inputs, outputs = int(info["shape"][0]), int(info["shape"][1])
        result = expert_matvec(
            self.model, name, 7, np.ones(inputs, dtype=np.float32), outputs
        )
        self.assertEqual(result.shape, (outputs,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertGreater(float(np.abs(result).sum()), 0.0)

    def test_the_routed_experts_are_what_is_stored_this_way(self):
        # 1.5 bits is spent where the weight count is largest.
        self.assertTrue(all("exps" in name for name in self.names), self.names[:3])


if __name__ == "__main__":
    unittest.main()
