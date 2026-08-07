"""Streaming state must reproduce the batch result exactly.

A runtime cannot reprocess the prompt on every token, so each block carries
state forward between decode steps. Three things persist: the raw latents the
sliding window attends to, the compressed blocks completed so far, and -- the
one that makes this more than an append-only cache -- the compressor's *partial*
block. Those rows are projected from a hidden state that is gone by the next
step, so they have to be kept rather than recomputed.

This is the property the native forward loop has to hold, checked here before
any of it is written in C++: feeding tokens one at a time must produce bit-for-
bit what feeding them together produces. Anything less means the decode path
and the prefill path disagree, which shows up as output that degrades after the
prompt rather than as an obvious failure.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from colibri_next.deepseek4_layer import CompressedState, DeepSeek4Block, LayerCache
from colibri_next.v2 import V2Model

CHECKPOINT = os.environ.get("DEEPSEEK4_GGUF")
PROMPT = "The quick brown fox jumps over the lazy dog today and tomorrow as well"


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class StreamingStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        cls.tokens = list(cls.model.tokenize(PROMPT))
        cls.block = DeepSeek4Block(cls.model, 2)      # a 4:1 layer, so it compresses
        cls.hidden = np.stack([
            np.asarray(cls.model.qwen_embedding(t, cls.block.n_embd), dtype=np.float32)
            for t in cls.tokens
        ])

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_the_prompt_spans_several_blocks(self):
        # Otherwise this proves nothing about carrying a partial block.
        self.assertGreaterEqual(len(self.tokens), 9)
        self.assertNotEqual(len(self.tokens) % self.block.ratio, 0)

    def test_streaming_compression_equals_batch(self):
        batch = self.block.compressor.compress_blocks(self.hidden)
        cache = LayerCache(self.block)
        for row in self.hidden:
            cache.append_compressor_state(row)
        streamed = (
            np.stack(cache.compressed) if cache.compressed
            else np.zeros((0, self.block.head_dim), dtype=np.float32)
        )
        self.assertEqual(streamed.shape, batch.shape)
        np.testing.assert_allclose(streamed, batch, rtol=0, atol=0)

    def test_a_block_appears_only_when_it_fills(self):
        cache = LayerCache(self.block)
        ratio = self.block.ratio
        for index, row in enumerate(self.hidden):
            cache.append_compressor_state(row)
            self.assertEqual(len(cache.compressed), (index + 1) // ratio)

    def test_the_partial_block_is_kept_not_discarded(self):
        cache = LayerCache(self.block)
        ratio = self.block.ratio
        # Feed one short of a block, then the token that completes it.
        for row in self.hidden[: ratio - 1]:
            cache.append_compressor_state(row)
        self.assertEqual(len(cache.compressed), 0)
        self.assertEqual(len(cache.state_values), ratio - 1)
        cache.append_compressor_state(self.hidden[ratio - 1])
        self.assertEqual(len(cache.compressed), 1)
        # And it matches what the batch path makes of the same tokens.
        batch = self.block.compressor.compress_blocks(self.hidden[:ratio])
        np.testing.assert_allclose(cache.compressed[0], batch[0], rtol=0, atol=0)

    def test_the_overlap_survives_streaming(self):
        # Block 1 pools the previous block's rows too, so streaming has to have
        # kept them rather than dropping them once block 0 closed.
        ratio = self.block.ratio
        cache = LayerCache(self.block)
        for row in self.hidden[: 2 * ratio]:
            cache.append_compressor_state(row)
        batch = self.block.compressor.compress_blocks(self.hidden[: 2 * ratio])
        self.assertEqual(len(cache.compressed), 2)
        np.testing.assert_allclose(cache.compressed[1], batch[1], rtol=0, atol=0)

    def test_visible_keys_grow_as_positions_arrive(self):
        cache = LayerCache(self.block)
        for position, row in enumerate(self.hidden):
            cache.append_latent(np.full(self.block.head_dim, float(position), dtype=np.float32))
            cache.append_compressor_state(row)
            keys, mask = cache.keys(position)
            self.assertEqual(len(keys), len(mask))
            # Every raw position so far is visible within the window.
            expected_raw = min(position + 1, self.block.window or position + 1)
            self.assertEqual(int(mask[: cache.positions].sum()), expected_raw)


class StreamingShapeTests(unittest.TestCase):
    """A layer with no compressor keeps no compressor state."""

    def test_a_sliding_window_layer_stores_nothing_to_compress(self):
        class Stub:
            compressor = None
            ratio = 0
            head_dim = 8
            window = 4
            layer = 0
        cache = LayerCache(Stub())
        cache.append_compressor_state(np.zeros(8, dtype=np.float32))
        self.assertEqual(cache.state_values, [])
        self.assertEqual(cache.compressed, [])


if __name__ == "__main__":
    unittest.main()
