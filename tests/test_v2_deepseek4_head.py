"""The output head: collapse the streams, norm, and project to logits.

    hc_head       = sum_h streams[h] * hc_head_pre[h]
    result_norm   = rms_norm(hc_head) * output_norm.weight
    result_output = output.weight @ result_norm

The head's mixer differs from a block's: it only ever reads the streams, so it
produces the `hc` pre-weights alone rather than the pre/post/comb triple, which
is why `output_hc_fn` is [hc, hc*n_embd] with a single scale where a block's is
[(2+hc)*hc, hc*n_embd] with three.

The head is exercised on the streams a real block produced -- layer 0's output
for the ten-token prompt -- rather than on noise, so the shapes and the
stream layout are checked as they will actually be used. The logits themselves
are not compared against the reference: that needs all 43 layers, and only two
of them run today.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from flyweight.deepseek4 import head_collapse, matvec, rms_norm
from flyweight.deepseek4_layer import DeepSeek4Block, _f32
from flyweight.v2 import V2Model

_CHECKPOINT_PATH = os.environ.get("DEEPSEEK4_GGUF")
# A stale path is as good as no path: the variable often outlives the file it
# named, and treating that as "configured" turns a missing checkpoint into a
# wall of errors instead of a skip.
CHECKPOINT = _CHECKPOINT_PATH if _CHECKPOINT_PATH and os.path.exists(_CHECKPOINT_PATH) else None
PROMPT = "The quick brown fox jumps over the lazy dog today"
EPSILON = 9.999999974752427e-07


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class OutputHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        cls.tokens = list(cls.model.tokenize(PROMPT))
        block = DeepSeek4Block(cls.model, 0)
        cls.n_embd = block.n_embd
        cls.hc = block.hc
        cls.vocabulary = int(cls.model.config["vocabulary_size"])
        streams = np.stack([
            np.repeat(
                np.asarray(cls.model.qwen_embedding(t, block.n_embd), dtype=np.float32)[None, :],
                block.hc, axis=0,
            )
            for t in cls.tokens
        ])
        cls.streams, _ = block.forward(streams, cls.tokens)
        cls.fn = _f32(cls.model, "output_hc_fn.weight").reshape(
            cls.hc, cls.hc * cls.n_embd
        )
        cls.scale = _f32(cls.model, "output_hc_scale.weight")
        cls.base = _f32(cls.model, "output_hc_base.weight")
        cls.output_norm = _f32(cls.model, "output_norm.weight")

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_the_head_mixer_is_pre_only(self):
        self.assertEqual(self.fn.shape, (self.hc, self.hc * self.n_embd))
        self.assertEqual(self.scale.shape, (1,))
        self.assertEqual(self.base.shape, (self.hc,))

    def test_collapse_is_the_pre_weighted_sum_of_streams(self):
        last = self.streams[-1]
        pre, collapsed = head_collapse(
            last, self.fn, self.scale, self.base,
            rms_epsilon=EPSILON, hc_epsilon=EPSILON,
        )
        self.assertEqual(pre.shape, (self.hc,))
        np.testing.assert_allclose(
            collapsed, (last * pre[:, None]).sum(axis=0), rtol=1e-5, atol=1e-5
        )

    def test_the_gates_are_in_range(self):
        _, _ = head_collapse(
            self.streams[-1], self.fn, self.scale, self.base,
            rms_epsilon=EPSILON, hc_epsilon=EPSILON,
        )
        pre, _ = head_collapse(
            self.streams[-1], self.fn, self.scale, self.base,
            rms_epsilon=EPSILON, hc_epsilon=EPSILON,
        )
        # sigmoid plus epsilon, so strictly inside (0, 1] by a hair.
        self.assertTrue(np.all(pre > 0.0))
        self.assertTrue(np.all(pre <= 1.0 + 1e-5))

    def test_logits_have_one_entry_per_vocabulary_item(self):
        _, collapsed = head_collapse(
            self.streams[-1], self.fn, self.scale, self.base,
            rms_epsilon=EPSILON, hc_epsilon=EPSILON,
        )
        normed = rms_norm(collapsed, self.output_norm, epsilon=EPSILON)
        logits = matvec(self.model, "output.weight", normed, self.vocabulary)
        self.assertEqual(logits.shape, (self.vocabulary,))
        self.assertTrue(np.all(np.isfinite(logits)))
        # A real distribution, not a constant: the spread should be wide.
        self.assertGreater(float(logits.max() - logits.min()), 1.0)

    def test_the_head_runs_on_every_position(self):
        for position in range(len(self.tokens)):
            pre, collapsed = head_collapse(
                self.streams[position], self.fn, self.scale, self.base,
                rms_epsilon=EPSILON, hc_epsilon=EPSILON,
            )
            self.assertEqual(collapsed.shape, (self.n_embd,))
            self.assertTrue(np.all(np.isfinite(collapsed)))


if __name__ == "__main__":
    unittest.main()
