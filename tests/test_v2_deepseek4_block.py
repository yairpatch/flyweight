"""A whole sliding-window block, composed and compared against the reference.

Every earlier test checked one component in isolation, which cannot catch a
wiring error between them -- a mixer applied to the wrong streams, a norm using
the wrong weight, the de-rope at the wrong point. This runs layer 0 end to end
over ten positions and compares each stage against `llama-eval-callback` on the
published checkpoint, finishing at the block's own output.

Tolerances come in two kinds, for a reason visible in the numbers. Quantities
whose sums are large relative to their terms are compared relatively, and land
within the accumulated activation-quantization noise described in
tests/test_v2_deepseek4_mla.py. Two stages -- the feed-forward norm and the
routed-expert output -- sum to nearly nothing across 40960 values, so their
relative error is meaningless (20% of 0.76 is 0.15, spread over 40960 terms)
and they are bounded absolutely instead.

Expert selection is exact, not approximate: the hash table is an integer lookup,
so the chosen ids must match the reference id for id.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from colibri_next.deepseek4_layer import SlidingWindowBlock
from colibri_next.v2 import V2Model

CHECKPOINT = os.environ.get("DEEPSEEK4_GGUF")
PROMPT = "The quick brown fox jumps over the lazy dog today"

# llama-eval-callback sums for layer 0, ten positions.
RELATIVE = {
    "hc_attn_pre": (-30.565649, 0.01),
    "attention": (7949.485352, 0.01),
    "derope": (849.288086, 0.02),
    "attn_out": (88.965637, 0.03),
    "hc_attn_post": (-27.260277, 0.01),
    "hc_ffn_pre": (-4.775506, 0.02),
    "ffn_out": (56.507828, 0.03),
    "output": (17.438572, 0.03),
}
# Sums over 40960 values that cancel to nearly zero; bounded absolutely.
ABSOLUTE = {
    "ffn_norm": (0.760870, 0.5),
    "moe_out": (-3.142883, 1.5),
}
EXPERT_ID_SUM = 7583.0
WEIGHT_SUM = 15.0


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class SlidingWindowBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        cls.tokens = list(cls.model.tokenize(PROMPT))
        cls.block = SlidingWindowBlock(cls.model, 0)
        streams = np.stack([
            np.repeat(
                np.asarray(
                    cls.model.qwen_embedding(token, cls.block.n_embd), dtype=np.float32
                )[None, :],
                cls.block.hc,
                axis=0,
            )
            for token in cls.tokens
        ])
        cls.output, cls.trace = cls.block.forward(streams, cls.tokens)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_the_block_output_matches_the_reference(self):
        expected, tolerance = RELATIVE["output"]
        actual = float(self.trace.output.sum())
        self.assertLessEqual(
            abs(actual - expected), tolerance * abs(expected),
            msg=f"block output {actual} vs reference {expected}",
        )

    def test_every_stage_matches(self):
        for name, (expected, tolerance) in RELATIVE.items():
            with self.subTest(stage=name):
                actual = float(getattr(self.trace, name).sum())
                self.assertLessEqual(
                    abs(actual - expected), tolerance * abs(expected),
                    msg=f"{name}: {actual} vs reference {expected}",
                )
        for name, (expected, allowed) in ABSOLUTE.items():
            with self.subTest(stage=name):
                actual = float(getattr(self.trace, name).sum())
                self.assertLessEqual(
                    abs(actual - expected), allowed,
                    msg=f"{name}: {actual} vs reference {expected}",
                )

    def test_expert_selection_is_exact(self):
        # An integer table lookup has no tolerance to spend.
        self.assertEqual(float(self.trace.experts.sum()), EXPERT_ID_SUM)
        self.assertEqual(
            self.trace.experts.shape, (len(self.tokens), self.block.experts_used)
        )

    def test_expert_weights_sum_to_the_scale_per_token(self):
        for position in range(len(self.tokens)):
            self.assertAlmostEqual(
                float(self.trace.expert_weights[position].sum()),
                self.block.weight_scale,
                places=4,
            )
        self.assertAlmostEqual(float(self.trace.expert_weights.sum()), WEIGHT_SUM, places=3)

    def test_the_output_keeps_the_stream_layout(self):
        self.assertEqual(
            self.output.shape, (len(self.tokens), self.block.hc, self.block.n_embd)
        )

    def test_attention_is_causal(self):
        # Position 0 sees only itself, so its output is the first latent scaled
        # by a single softmax weight -- rerunning with later positions removed
        # must not change it.
        first = self.trace.attention[0]
        streams = np.stack([
            np.repeat(
                np.asarray(
                    self.model.qwen_embedding(token, self.block.n_embd), dtype=np.float32
                )[None, :],
                self.block.hc,
                axis=0,
            )
            for token in self.tokens[:1]
        ])
        _, shorter = self.block.forward(streams, self.tokens[:1])
        np.testing.assert_allclose(shorter.attention[0], first, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
