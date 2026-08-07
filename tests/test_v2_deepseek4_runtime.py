"""The native per-sequence state, and what it costs.

Sizing this from the architecture rather than from the context length is the
point. Raw latents are bounded by the 128-token sliding window on every layer,
so they need a ring of 128 rather than a cache that grows. The compressor keeps
only what a later block can still read: a 4:1 layer pools the previous block's
rows alongside its own, so two blocks' worth, and a 128:1 layer only its own.
Only the compressed caches scale with the context.

The composed model in deepseek4_layer keeps every position instead, which is
simpler and right for an oracle. Copying that into the runtime would spend a
sequence's worth of memory to hold rows nothing can read again.

The totals matter for a different reason than they usually would: at tens of
megabytes against 104 GiB of weights, cache quantization is not worth wiring up
here and context length is nearly free. What limits context is the unimplemented
lightning indexer, not the cache budget.
"""

from __future__ import annotations

import os
import unittest

from colibri_next.deepseek4 import Deepseek4Runtime
from colibri_next.v2 import V2Error, V2Model

CHECKPOINT = os.environ.get("DEEPSEEK4_GGUF")


def _expected_bytes(config, ratios, context: int) -> int:
    """The design's arithmetic, written out independently of the runtime."""
    head_dim = int(config["kv_lora_rank"])
    window = min(int(config["sliding_window"]), context)
    total = 0
    for ratio in ratios[: int(config["layer_count"])]:
        total += window * head_dim * 4                     # raw latent ring
        if not ratio:
            continue
        overlapped = ratio == 4
        width = (2 if overlapped else 1) * head_dim
        rows = (2 if overlapped else 1) * ratio
        total += (context // ratio + 1) * head_dim * 4     # compressed cache
        total += rows * width * 4 * 2                      # values and scores
    return total


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class Deepseek4RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        cls.ratios = cls.model.compress_ratios

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_it_reports_the_layer_mix(self):
        with Deepseek4Runtime(self.model, 4096) as runtime:
            info = runtime.info
            self.assertEqual(info["layers"], 43)
            self.assertEqual(info["window_layers"], 2)
            self.assertEqual(info["csa_layers"] + info["hca_layers"], 41)
            self.assertEqual(info["context_limit"], 4096)
            self.assertEqual(info["positions"], 0)

    def test_the_state_matches_the_design_arithmetic(self):
        for context in (1024, 4096, 32768):
            with self.subTest(context=context):
                with Deepseek4Runtime(self.model, context) as runtime:
                    expected = _expected_bytes(self.model.config, self.ratios, context)
                    self.assertEqual(runtime.info["state_bytes"], expected)

    def test_the_state_is_negligible_beside_the_weights(self):
        # Tens of megabytes against 104 GiB, which is what makes cache
        # quantization pointless here and context nearly free.
        #
        # 65 MiB rather than the 38 the plan estimated: that estimate assumed
        # f16 caches, matching the reference, and these buffers are f32. The
        # composed model rounds its latents to f16 precisely to stay numerically
        # with the reference, so the write path will have to round too -- at
        # which point storing f16 costs nothing and halves this. Left as f32
        # until the write path exists rather than guessed at now.
        with Deepseek4Runtime(self.model, 4096) as runtime:
            megabytes = runtime.info["state_bytes"] / 1024 / 1024
            self.assertLess(megabytes, 128)
            self.assertGreater(megabytes, 1)

    def test_raw_state_does_not_grow_with_context(self):
        # Only the compressed caches scale. Quadrupling the context must not
        # quadruple the total, because the window and compressor rings are
        # fixed.
        with Deepseek4Runtime(self.model, 4096) as small:
            with Deepseek4Runtime(self.model, 16384) as large:
                self.assertLess(
                    large.info["state_bytes"], 4 * small.info["state_bytes"]
                )

    def test_reset_clears_the_positions(self):
        with Deepseek4Runtime(self.model, 1024) as runtime:
            runtime.reset()
            self.assertEqual(runtime.info["positions"], 0)

    def test_closing_twice_is_harmless(self):
        runtime = Deepseek4Runtime(self.model, 1024)
        runtime.close()
        runtime.close()


class Deepseek4RuntimeRejectionTests(unittest.TestCase):
    def test_a_non_deepseek4_model_is_refused(self):
        import tempfile
        from pathlib import Path

        from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf

        with tempfile.TemporaryDirectory(prefix="colibri-ds4rt-") as directory:
            path = Path(directory) / "dense.gguf"
            build_dense_qwen35_gguf(path, DenseQwenSpec(layers=2))
            model = V2Model(path)
            try:
                with self.assertRaises(V2Error):
                    Deepseek4Runtime(model, 256)
            finally:
                model.close()


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class WeightPlanTests(unittest.TestCase):
    """Every weight a block reads, resolved once at creation.

    What this replaces is a linear scan by name over all 1328 descriptors, done
    per weight per token. A block reads about twenty, so a 43-layer token would
    have spent roughly 900 scans of a 1328-entry vector doing nothing but
    finding weights it already found.

    Resolving up front also turns a missing or misnamed tensor into a failure at
    creation rather than a wrong answer mid-generation.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_every_layer_resolves_its_weights(self):
        with Deepseek4Runtime(self.model, 4096) as runtime:
            info = runtime.info
            # 24 on every block -- 23 plus exactly one of the routing table or
            # the router bias -- then 4 more on a compressed layer and 6 more on
            # a 4:1 layer for the indexer.
            expected = (
                24 * info["layers"]
                + 4 * (info["csa_layers"] + info["hca_layers"])
                + 6 * info["csa_layers"]
            )
            self.assertEqual(info["resolved_tensors"], expected)

    def test_creation_is_not_quadratic_in_the_model(self):
        # One pass builds a name index the plans draw from, so creating a
        # runtime over 1328 descriptors should be quick rather than 900 scans
        # per layer.
        import time
        start = time.perf_counter()
        with Deepseek4Runtime(self.model, 4096):
            pass
        self.assertLess(time.perf_counter() - start, 1.0)


class WeightPlanFixtureTests(unittest.TestCase):
    def test_the_miniature_fixture_resolves_too(self):
        import tempfile
        from pathlib import Path

        from tests.deepseek4_gguf_fixture import DeepSeek4Spec, build_deepseek4_gguf

        with tempfile.TemporaryDirectory(prefix="colibri-ds4plan-") as directory:
            path = Path(directory) / "ds4.gguf"
            build_deepseek4_gguf(path, DeepSeek4Spec(layers=6, hash_layers=3))
            model = V2Model(path)
            try:
                with Deepseek4Runtime(model, 256) as runtime:
                    info = runtime.info
                    expected = (
                        24 * info["layers"]
                        + 4 * (info["csa_layers"] + info["hca_layers"])
                        + 6 * info["csa_layers"]
                    )
                    self.assertEqual(info["resolved_tensors"], expected)
            finally:
                model.close()
