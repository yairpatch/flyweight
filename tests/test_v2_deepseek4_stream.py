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

from flyweight.deepseek4_layer import DeepSeek4Block, LayerCache
from flyweight.v2 import V2Model

_CHECKPOINT_PATH = os.environ.get("DEEPSEEK4_GGUF")
# A stale path is as good as no path: the variable often outlives the file it
# named, and treating that as "configured" turns a missing checkpoint into a
# wall of errors instead of a skip.
CHECKPOINT = _CHECKPOINT_PATH if _CHECKPOINT_PATH and os.path.exists(_CHECKPOINT_PATH) else None
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


class NativeVisibilityTests(unittest.TestCase):
    """The native mask must agree with the Python one, which is reference-checked.

    The runtime builds this mask per query position, so it is the first piece of
    the forward loop to move into C++. Keeping both and comparing them means the
    translation is checked against something already known to be right, rather
    than against a fresh reading of the architecture.
    """

    def test_native_matches_python_across_shapes(self):
        from flyweight.deepseek4 import visible_keys
        from flyweight.deepseek4_layer import csa_attention_latents

        for raw_positions, blocks, ratio, window in (
            (10, 2, 4, 128), (10, 0, 0, 128), (376, 94, 4, 128),
            (376, 2, 128, 128), (20, 5, 4, 3), (1, 0, 4, 128),
        ):
            raw = np.zeros((raw_positions, 2), dtype=np.float32)
            compressed = np.zeros((blocks, 2), dtype=np.float32)
            for position in range(min(raw_positions, 40)):
                with self.subTest(raw=raw_positions, blocks=blocks, pos=position):
                    _, expected = csa_attention_latents(
                        raw, compressed, position, max(ratio, 1), window
                    )
                    actual = visible_keys(position, raw_positions, blocks, ratio, window)
                    np.testing.assert_array_equal(actual, expected)

    def test_a_zero_ratio_with_blocks_is_rejected(self):
        from flyweight.deepseek4 import visible_keys
        with self.assertRaises(Exception):
            visible_keys(0, 4, 2, 0, 128)


class NativeGatherTests(unittest.TestCase):
    """The overlap gather, translated and cross-checked.

    Which rows a block pools took measurement to establish -- pooling four
    entries instead of eight, or reading the wrong half of the state row, still
    produces plausible numbers. So the native version is checked against the
    Python one rather than against a rereading of the architecture.
    """

    def _python_gather(self, values, scores, head_dim, ratio, block, overlapped):
        if not overlapped:
            span = slice(block * ratio, (block + 1) * ratio)
            return values[span].copy(), scores[span].copy()
        rows = 2 * ratio
        pv = np.zeros((rows, head_dim), dtype=np.float32)
        ps = np.full((rows, head_dim), -np.inf, dtype=np.float32)
        for slot in range(ratio):
            previous = (block - 1) * ratio + slot
            if previous >= 0:
                pv[slot] = values[previous][:head_dim]
                ps[slot] = scores[previous][:head_dim]
            current = block * ratio + slot
            pv[ratio + slot] = values[current][head_dim:]
            ps[ratio + slot] = scores[current][head_dim:]
        return pv, ps

    def test_overlapped_gather_matches_python(self):
        from flyweight.deepseek4 import gather_block
        rng = np.random.default_rng(17)
        head_dim, ratio = 6, 4
        values = rng.standard_normal((16, 2 * head_dim)).astype(np.float32)
        scores = rng.standard_normal((16, 2 * head_dim)).astype(np.float32)
        for block in range(4):
            with self.subTest(block=block):
                gv, gs = gather_block(values, scores, head_dim, ratio, block, True)
                pv, ps = self._python_gather(values, scores, head_dim, ratio, block, True)
                np.testing.assert_array_equal(gv, pv)
                np.testing.assert_array_equal(gs, ps)

    def test_the_first_block_pads_its_predecessor(self):
        from flyweight.deepseek4 import gather_block
        head_dim, ratio = 5, 4
        values = np.ones((8, 2 * head_dim), dtype=np.float32)
        scores = np.ones((8, 2 * head_dim), dtype=np.float32)
        gv, gs = gather_block(values, scores, head_dim, ratio, 0, True)
        # The predecessor half is zero-valued and -inf scored, so it drops out.
        np.testing.assert_array_equal(gv[:ratio], np.zeros((ratio, head_dim), np.float32))
        self.assertTrue(np.all(np.isneginf(gs[:ratio])))
        np.testing.assert_array_equal(gv[ratio:], np.ones((ratio, head_dim), np.float32))

    def test_unoverlapped_gather_matches_python(self):
        from flyweight.deepseek4 import gather_block
        rng = np.random.default_rng(18)
        head_dim, ratio = 7, 8
        values = rng.standard_normal((24, head_dim)).astype(np.float32)
        scores = rng.standard_normal((24, head_dim)).astype(np.float32)
        for block in range(3):
            with self.subTest(block=block):
                gv, gs = gather_block(values, scores, head_dim, ratio, block, False)
                pv, ps = self._python_gather(values, scores, head_dim, ratio, block, False)
                np.testing.assert_array_equal(gv, pv)
                np.testing.assert_array_equal(gs, ps)

    def test_a_state_width_that_disagrees_with_the_kind_is_rejected(self):
        from flyweight.deepseek4 import gather_block
        values = np.zeros((8, 6), dtype=np.float32)
        with self.assertRaises(Exception):
            # Six-wide rows cannot be an overlapped layer with head_dim six.
            gather_block(values, values, 6, 4, 0, True)
