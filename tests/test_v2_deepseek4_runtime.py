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
        # Caches are half precision, as the reference stores them; the
        # compressor's own rows stay f32, which is also what the reference does.
        total += window * head_dim * 2                     # raw latent ring
        if not ratio:
            continue
        overlapped = ratio == 4
        width = (2 if overlapped else 1) * head_dim
        rows = (2 if overlapped else 1) * ratio
        total += (context // ratio + 1) * head_dim * 2     # compressed cache
        total += rows * width * 4 * 2                      # values and scores
        if not overlapped:
            continue
        # A 4:1 layer also carries the lightning indexer's cache: the same
        # shape at 128 wide instead of 512, because it only ranks blocks.
        indexer = int(config["indexer_key_length"])
        total += (context // ratio + 1) * indexer * 2
        total += rows * (2 * indexer) * 4 * 2
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
        with Deepseek4Runtime(self.model, 4096) as runtime:
            megabytes = runtime.info["state_bytes"] / 1024 / 1024
            self.assertLess(megabytes, 64)
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


class HalfPrecisionTests(unittest.TestCase):
    """The cache storage format, checked against a known-good implementation.

    The caches hold half precision because the reference does, and the composed
    model already rounds its latents the same way -- so matching costs no
    accuracy and halves the footprint. But a hand-rolled float-to-half is easy
    to get subtly wrong at the edges, and a rounding difference in the cache
    would show up as drift that looks like an architecture bug. So it is
    compared against numpy's float16 rather than trusted.
    """

    def test_it_matches_numpy_on_ordinary_values(self):
        import numpy as np
        from colibri_next.deepseek4 import half_round_trip

        rng = np.random.default_rng(23)
        values = np.concatenate([
            rng.standard_normal(4000).astype(np.float32) * 10.0,
            rng.standard_normal(2000).astype(np.float32) * 0.001,
        ])
        for value in values:
            expected = float(np.float32(np.float16(value)))
            self.assertEqual(half_round_trip(float(value)), expected, msg=f"{value!r}")

    def test_it_matches_numpy_on_edges(self):
        import numpy as np
        from colibri_next.deepseek4 import half_round_trip

        edges = [
            0.0, -0.0, 1.0, -1.0, 65504.0, -65504.0,   # largest finite half
            65520.0, 1e30, -1e30,                       # overflow to infinity
            6.103515625e-05, 5.960464477539063e-08,     # smallest normal, smallest subnormal
            3e-08, 1e-10, -1e-10,                       # underflow territory
            2049.0, 2048.5, 0.30000001192092896,        # rounding ties
        ]
        for value in edges:
            expected = float(np.float32(np.float16(np.float32(value))))
            self.assertEqual(half_round_trip(value), expected, msg=f"{value!r}")

    def test_infinities_and_nan_survive(self):
        import math
        from colibri_next.deepseek4 import half_round_trip

        self.assertEqual(half_round_trip(math.inf), math.inf)
        self.assertEqual(half_round_trip(-math.inf), -math.inf)
        self.assertTrue(math.isnan(half_round_trip(math.nan)))


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class NativeForwardTests(unittest.TestCase):
    """The native forward against the composed model.

    The oracle is verified against llama.cpp, so agreeing with it is what makes
    the runtime trustworthy. The bar here is exact rather than approximate: both
    call the same kernels in the same order on the same weights, so any
    difference at all means the loop wired something differently, not that
    precision drifted.

    Prefill and decode are the same call. A prompt is repeated forwards and
    generation is more of them, which is what stops the two paths disagreeing --
    the failure mode that shows up as output degrading after the prompt rather
    than breaking.
    """

    PROMPT = "The quick brown fox jumps over the lazy dog today"

    @classmethod
    def setUpClass(cls):
        import numpy as np
        from colibri_next.deepseek4_layer import DeepSeek4Model

        cls.model = V2Model(CHECKPOINT)
        cls.tokens = list(cls.model.tokenize(cls.PROMPT))
        with Deepseek4Runtime(cls.model, 256) as runtime:
            for index, token in enumerate(cls.tokens):
                last = index == len(cls.tokens) - 1
                cls.native = runtime.forward(token, logits=last)
            cls.positions = runtime.info["positions"]
        cls.oracle = DeepSeek4Model(cls.model).forward(cls.tokens)
        cls.np = np

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_the_logits_match_the_oracle_exactly(self):
        self.np.testing.assert_array_equal(self.native, self.oracle)

    def test_the_prediction_matches(self):
        self.assertEqual(int(self.np.argmax(self.native)), int(self.np.argmax(self.oracle)))

    def test_the_sequence_advanced_once_per_token(self):
        self.assertEqual(self.positions, len(self.tokens))

    def test_running_past_the_context_limit_is_refused(self):
        with Deepseek4Runtime(self.model, 4) as runtime:
            for token in self.tokens[:4]:
                runtime.forward(token, logits=False)
            with self.assertRaises(V2Error):
                runtime.forward(self.tokens[4], logits=False)

    def test_reset_starts_a_fresh_sequence(self):
        with Deepseek4Runtime(self.model, 64) as runtime:
            for token in self.tokens[:3]:
                runtime.forward(token, logits=False)
            runtime.reset()
            self.assertEqual(runtime.info["positions"], 0)
            first = runtime.forward(self.tokens[0])
            runtime.reset()
            again = runtime.forward(self.tokens[0])
            self.np.testing.assert_array_equal(first, again)

    def test_a_token_outside_the_vocabulary_is_refused(self):
        with Deepseek4Runtime(self.model, 16) as runtime:
            with self.assertRaises(V2Error):
                runtime.forward(10_000_000, logits=False)


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class GenerationTests(unittest.TestCase):
    """Greedy generation, judged on what it says rather than on numbers.

    Every other check here compares against the oracle or the reference. This
    one asks whether the model answers a question correctly, which is the thing
    a numeric check cannot tell you: a stack can match an oracle exactly and
    still be wired to the wrong architecture if the oracle is too.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def _continue(self, prompt: str, limit: int) -> str:
        tokens = list(self.model.tokenize(prompt))
        with Deepseek4Runtime(self.model, 256) as runtime:
            produced = list(runtime.generate(tokens, max_tokens=limit))
        return self.model.decode_tokens(produced)

    def test_it_answers_a_factual_prompt(self):
        self.assertIn("Paris", self._continue("The capital city of France is", 6))

    def test_it_answers_arithmetic(self):
        self.assertIn("4", self._continue("2 + 2 =", 4))

    def test_generation_stops_at_the_limit(self):
        tokens = list(self.model.tokenize("Once upon a time"))
        with Deepseek4Runtime(self.model, 128) as runtime:
            produced = list(runtime.generate(tokens, max_tokens=3))
        self.assertLessEqual(len(produced), 3)

    def test_a_terminator_ends_generation(self):
        # Stopping is driven by the checkpoint's own terminator ids; forcing one
        # that the first token will hit must end it immediately.
        tokens = list(self.model.tokenize("The capital city of France is"))
        with Deepseek4Runtime(self.model, 128) as runtime:
            first = runtime.forward(tokens[-1] if len(tokens) == 1 else tokens[0])
        import numpy as np
        likely = int(np.argmax(first))
        with Deepseek4Runtime(self.model, 128) as runtime:
            produced = list(runtime.generate(tokens[:1], max_tokens=5, stop={likely}))
        self.assertEqual(produced, [])


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class StreamingTests(unittest.TestCase):
    """Streaming in the shape the server consumes.

    The delta is computed by decoding the whole run each step and taking what
    grew, not by decoding each token alone. A token is bytes rather than a
    character, so a multi-byte codepoint split across two tokens decodes to
    replacement characters if each is decoded by itself -- which is how streamed
    output ends up with stray U+FFFD in exactly the languages that need the
    bytes.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        cls.tokens = list(cls.model.tokenize("The capital city of France is"))

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_the_deltas_reassemble_the_text(self):
        with Deepseek4Runtime(self.model, 128) as runtime:
            steps = list(runtime.stream(self.model, self.tokens, max_tokens=6))
        self.assertTrue(steps[-1].finished)
        self.assertEqual("".join(step.text_delta for step in steps), steps[-1].text)

    def test_only_the_last_step_is_final(self):
        with Deepseek4Runtime(self.model, 128) as runtime:
            steps = list(runtime.stream(self.model, self.tokens, max_tokens=4))
        self.assertTrue(all(not step.finished for step in steps[:-1]))
        self.assertIsNone(steps[-1].token_id)
        self.assertTrue(all(step.token_id is not None for step in steps[:-1]))

    def test_it_streams_the_same_text_it_generates(self):
        with Deepseek4Runtime(self.model, 128) as runtime:
            streamed = list(runtime.stream(self.model, self.tokens, max_tokens=6))[-1]
        with Deepseek4Runtime(self.model, 128) as runtime:
            produced = list(runtime.generate(self.tokens, max_tokens=6))
        self.assertEqual(list(streamed.generated_ids), produced)
        self.assertEqual(streamed.text, self.model.decode_tokens(produced))

    def test_the_state_count_tracks_the_sequence(self):
        with Deepseek4Runtime(self.model, 128) as runtime:
            steps = list(runtime.stream(self.model, self.tokens, max_tokens=3))
        self.assertEqual(steps[-1].state_tokens, len(self.tokens) + len(steps[-1].generated_ids))
