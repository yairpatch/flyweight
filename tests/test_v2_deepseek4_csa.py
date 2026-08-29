"""The CSA compressor: overlapped block pooling with its own rotation.

CSA compresses every four tokens into one latent, but not from those four
tokens alone. Each token projects to a state row twice the latent width -- a
"prev" half and a "cur" half -- and a block pools eight entries: the previous
four tokens' prev halves followed by its own four tokens' cur halves. That is
the overlap the architecture describes as keeping a window of the last eight
tokens at each four-token boundary. The first block has no predecessor, so those
rows read a padding entry scored -inf, which the softmax drops.

The compressed latent is rotated at the *block* index using the compressed
frequency base with YaRN -- a different rotation from the one raw tokens get at
their own positions, and the only place YaRN applies at all.

Verified against layer 2 of the published checkpoint, reachable because layers 0
and 1 are sliding-window blocks that already run. Getting the overlap wrong --
pooling four entries instead of eight, or taking the wrong half of the state
row -- still produces plausible numbers, so the comparison is against the
reference rather than against a shape check.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from flyweight.deepseek4 import rms_norm
from flyweight.deepseek4_layer import CompressedState, DeepSeek4Block
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
PROMPT = "The quick brown fox jumps over the lazy dog today"

# llama-eval-callback sums for layer 2.
REFERENCE = {
    "state_kv": -10.693863,
    "state_score_ape": -1234.675415,
    "compressed": -22.257238,
}


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class CsaCompressorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        cls.tokens = list(cls.model.tokenize(PROMPT))
        first = DeepSeek4Block(cls.model, 0)
        streams = np.stack([
            np.repeat(
                np.asarray(cls.model.qwen_embedding(t, first.n_embd), dtype=np.float32)[None, :],
                first.hc, axis=0,
            )
            for t in cls.tokens
        ])
        streams, _ = first.forward(streams, cls.tokens)
        streams, _ = DeepSeek4Block(cls.model, 1).forward(streams, cls.tokens)

        block = DeepSeek4Block(cls.model, 2)
        mixes = [block._mix(streams[p], block.hc_attn) for p in range(len(cls.tokens))]
        cls.hidden = np.stack([
            rms_norm(m.collapsed, block.attn_norm, epsilon=block.epsilon) for m in mixes
        ])
        cls.state = CompressedState(cls.model, 2, 4)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def _close(self, actual, expected, label, tolerance=0.05):
        self.assertLessEqual(
            abs(actual - expected), tolerance * max(abs(expected), 1.0),
            msg=f"{label}: {actual} vs reference {expected}",
        )

    def test_the_state_rows_are_twice_the_latent_width(self):
        values, scores = self.state.states(self.hidden)
        self.assertEqual(values.shape, (len(self.tokens), 2 * self.state.head_dim))
        self.assertEqual(scores.shape, values.shape)
        self._close(float(values.sum()), REFERENCE["state_kv"], "state_kv")

    def test_the_slot_embedding_is_added_to_the_scores(self):
        values, scores = self.state.states(self.hidden)
        self._close(
            float(scores.sum()), REFERENCE["state_score_ape"], "state_score_ape", 0.01
        )
        # The embedding repeats every `ratio` tokens, so tokens 0 and 4 share a
        # slot while 0 and 1 do not.
        self.assertEqual(self.state.ape.shape, (self.state.ratio, 2 * self.state.head_dim))

    def test_compressed_latents_match_the_reference(self):
        compressed = self.state.compress_blocks(self.hidden)
        # Ten tokens at ratio four completes two blocks; the last two are left.
        self.assertEqual(compressed.shape, (2, self.state.head_dim))
        self._close(float(compressed.sum()), REFERENCE["compressed"], "compressed", 0.03)

    def test_an_incomplete_block_is_not_compressed(self):
        for count in range(self.state.ratio):
            with self.subTest(tokens=count):
                self.assertEqual(
                    self.state.compress_blocks(self.hidden[:count]).shape,
                    (0, self.state.head_dim),
                )
        self.assertEqual(
            self.state.compress_blocks(self.hidden[: self.state.ratio]).shape,
            (1, self.state.head_dim),
        )

    def test_the_first_block_pools_only_its_own_tokens(self):
        # Its predecessor rows are padding scored -inf, so compressing the first
        # four tokens alone must equal compressing them as part of a longer run.
        alone = self.state.compress_blocks(self.hidden[: self.state.ratio])
        together = self.state.compress_blocks(self.hidden)
        np.testing.assert_allclose(alone[0], together[0], rtol=1e-5, atol=1e-5)

    def test_the_second_block_sees_the_first_blocks_tokens(self):
        # If the overlap were dropped, block 1 would depend only on tokens 4-7.
        together = self.state.compress_blocks(self.hidden)
        disturbed = self.hidden.copy()
        disturbed[0] = disturbed[0] * 3.0 + 1.0
        moved = self.state.compress_blocks(disturbed)
        self.assertFalse(
            np.allclose(moved[1], together[1], rtol=1e-4, atol=1e-4),
            msg="block 1 ignored the previous block's tokens",
        )


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class CsaVisibilityTests(unittest.TestCase):
    """Which keys a CSA query may attend to.

    The raw window and the compressed blocks overlap on purpose: a token can be
    attended directly and again through its block's summary. A block becomes
    visible once every token it covers is at or before the query.
    """

    def setUp(self):
        from flyweight.deepseek4_layer import csa_attention_latents
        self.latents = csa_attention_latents

    def test_a_block_is_visible_from_the_query_that_completes_it(self):
        raw = np.zeros((10, 4), dtype=np.float32)
        compressed = np.zeros((2, 4), dtype=np.float32)
        for position, expected in enumerate([0, 0, 0, 1, 1, 1, 1, 2, 2, 2]):
            with self.subTest(position=position):
                _, mask = self.latents(raw, compressed, position, 4, 128)
                self.assertEqual(int(mask[len(raw):].sum()), expected)

    def test_raw_keys_stay_causal(self):
        raw = np.zeros((10, 4), dtype=np.float32)
        compressed = np.zeros((2, 4), dtype=np.float32)
        for position in range(10):
            _, mask = self.latents(raw, compressed, position, 4, 128)
            self.assertEqual(int(mask[: len(raw)].sum()), position + 1)

    def test_the_window_bounds_the_raw_keys(self):
        raw = np.zeros((10, 4), dtype=np.float32)
        _, mask = self.latents(raw, np.zeros((0, 4), np.float32), 9, 4, 3)
        # Window three means positions 7, 8, 9.
        self.assertEqual(int(mask.sum()), 3)

    def test_with_no_complete_blocks_only_raw_keys_are_offered(self):
        raw = np.zeros((3, 4), dtype=np.float32)
        latents, mask = self.latents(raw, np.zeros((0, 4), np.float32), 2, 4, 128)
        self.assertEqual(len(latents), 3)
        self.assertEqual(len(mask), 3)
