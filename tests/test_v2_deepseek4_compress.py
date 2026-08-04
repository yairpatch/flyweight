"""Block compression, the shared foundation of both compressed attention kinds.

CSA pools every four tokens into one latent and HCA pools every 128, and both
use the same operation to do it. Reading the reference:

    kv     = get_rows(kv_state,    block)      -> [head_dim, ratio]
    score  = get_rows(score_state, block)      -> [head_dim, ratio]
    weights = soft_max(permute(score))         -- over the block, per channel
    comp   = sum_rows(permute(kv) * weights)
    comp   = rms_norm(comp, attn_compressor_norm)
    comp   = concat(comp_nope, rope(comp_pe, compress_rope_base, yarn))

The detail that is easy to miss, and that these tests pin, is that the softmax
runs per *channel* rather than per position: each of the head_dim channels
weights the block's tokens by its own scores, so one channel can take its value
from the first token of a block while another takes it from the last. Pooling
per position -- one weight per token, shared across channels -- would be the
natural guess and is wrong.

The score state has the block-slot position embedding
(`attn_compressor_ape`, shaped [width, ratio]) already added to it, which is
what lets a channel prefer, say, the last token of every block.

These are unit checks. Parity for a real compressed layer needs the state built
across preceding layers, which does not exist yet -- layers 0 and 1 are the only
uncompressed ones, so the first compressed block is layer 2.
"""

from __future__ import annotations

import unittest

import numpy as np

from colibri_next.deepseek4 import compress


def _reference(values: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Softmax over the block, per channel."""
    shifted = scores - scores.max(axis=0, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=0, keepdims=True)
    return (values * weights).sum(axis=0)


class CompressTests(unittest.TestCase):
    def test_matches_a_per_channel_softmax_average(self):
        rng = np.random.default_rng(11)
        for positions, width in ((4, 16), (128, 8), (4, 512)):
            with self.subTest(positions=positions, width=width):
                values = rng.standard_normal((positions, width)).astype(np.float32)
                scores = rng.standard_normal((positions, width)).astype(np.float32)
                np.testing.assert_allclose(
                    compress(values, scores), _reference(values, scores), rtol=1e-5, atol=1e-6
                )

    def test_channels_pool_independently(self):
        # Channel 0 is steered entirely to the first token, channel 1 to the
        # last. A per-position softmax could not produce this.
        values = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]], dtype=np.float32)
        scores = np.array([[100.0, 0.0], [0.0, 0.0], [0.0, 100.0]], dtype=np.float32)
        result = compress(values, scores)
        self.assertAlmostEqual(float(result[0]), 10.0, places=4)
        self.assertAlmostEqual(float(result[1]), 3.0, places=4)

    def test_equal_scores_give_a_plain_mean(self):
        rng = np.random.default_rng(12)
        values = rng.standard_normal((4, 32)).astype(np.float32)
        scores = np.full((4, 32), 2.5, dtype=np.float32)
        np.testing.assert_allclose(
            compress(values, scores), values.mean(axis=0), rtol=1e-5, atol=1e-6
        )

    def test_a_single_position_is_the_identity(self):
        rng = np.random.default_rng(13)
        values = rng.standard_normal((1, 64)).astype(np.float32)
        scores = rng.standard_normal((1, 64)).astype(np.float32)
        np.testing.assert_allclose(compress(values, scores), values[0], rtol=1e-6)

    def test_large_scores_do_not_overflow(self):
        values = np.ones((4, 8), dtype=np.float32)
        scores = np.full((4, 8), 1e4, dtype=np.float32)
        result = compress(values, scores)
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            compress(np.zeros((4, 8), np.float32), np.zeros((4, 9), np.float32))


if __name__ == "__main__":
    unittest.main()
