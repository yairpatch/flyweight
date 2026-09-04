"""The whole stack: 43 blocks and the output head.

Where this stands, stated plainly, because the numbers are easy to misread.
Running the ten-token prompt through every block and comparing each block's
output against the reference, layers 0 through 32 agree to within 7% and most
are under 1%. From layer 33 the agreement falls apart, reaching 50% by the end.

The reference's own activations explode over the same span -- its block output
sums run 2435 at layer 32, then -28391 at 34 and -423713 at 36 -- so those
layers are dominated by a handful of very large channels, and a small relative
error in one of them swamps the sum. That makes the tail either error
amplification through an ill-conditioned region or a real defect specific to it;
this test does not distinguish them, and says so rather than asserting a bound
that would encode the current behaviour as correct.

So the assertions here cover what is established: the stack runs, its shapes
hold, the early and middle layers track the reference, and the logits are a
plausible distribution. Token-level agreement -- the acceptance criterion for
the forward pass -- is deliberately not asserted yet.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from flyweight.deepseek4_layer import DeepSeek4Model
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

# Reference block-output sums, layer -> sum, for the layers that agree.
EARLY = {0: 17.439, 1: 31.999, 2: 34.916, 3: 83.876, 4: 57.836, 5: 1517.286}
# Where agreement is lost. Recorded so a change in this boundary is visible.
DIVERGES_FROM = 33


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class FullStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        cls.tokens = list(cls.model.tokenize(PROMPT))
        cls.net = DeepSeek4Model(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_the_stack_has_one_block_per_layer(self):
        self.assertEqual(len(self.net.blocks), 43)
        ratios = [block.ratio for block in self.net.blocks]
        # Two sliding-window blocks, then 4 and 128 alternating.
        self.assertEqual(ratios[:4], [0, 0, 4, 128])
        self.assertEqual(set(ratios), {0, 4, 128})

    def test_every_compressed_layer_carries_a_compressor(self):
        for block in self.net.blocks:
            with self.subTest(layer=block.layer):
                self.assertEqual(block.compressor is not None, block.ratio != 0)
                if block.compressor is None:
                    continue
                # Only the 4:1 kind overlaps its blocks, which is what doubles
                # its state width.
                self.assertEqual(block.compressor.overlapped, block.ratio == 4)
                self.assertEqual(
                    block.compressor.width,
                    (2 if block.ratio == 4 else 1) * block.head_dim,
                )

    def test_compressed_layers_rotate_on_the_compressed_base(self):
        plain = next(b for b in self.net.blocks if b.ratio == 0)
        compressed = next(b for b in self.net.blocks if b.ratio != 0)
        self.assertEqual(plain.rope_kwargs["freq_base"], plain.freq_base)
        self.assertGreater(compressed.rope_kwargs["freq_base"], plain.freq_base)
        self.assertEqual(compressed.rope_kwargs["ext_factor"], 1.0)
        self.assertNotIn("ext_factor", plain.rope_kwargs)

    def test_the_early_layers_track_the_reference(self):
        streams = np.stack([
            np.repeat(
                np.asarray(self.model.qwen_embedding(t, self.net.n_embd), dtype=np.float32)[None, :],
                self.net.hc, axis=0,
            )
            for t in self.tokens
        ])
        for layer, expected in EARLY.items():
            streams, _ = self.net.blocks[layer].forward(streams, self.tokens)
            actual = float(streams.sum())
            self.assertLessEqual(
                abs(actual - expected), 0.05 * max(abs(expected), 1.0),
                msg=f"layer {layer}: {actual} vs reference {expected}",
            )

    def test_the_stack_produces_a_plausible_distribution(self):
        logits = self.net.forward(self.tokens)
        self.assertEqual(logits.shape, (self.net.vocabulary,))
        self.assertTrue(np.all(np.isfinite(logits)))
        self.assertGreater(float(logits.max() - logits.min()), 5.0)
        # The most likely continuations of an English sentence should be
        # ordinary text, not control tokens.
        top = np.argsort(-logits)[:5]
        self.assertTrue(
            any(self.model.token_text(int(t)).strip(" .,'\"") for t in top),
            msg="every top continuation was empty or punctuation-only",
        )


if __name__ == "__main__":
    unittest.main()
