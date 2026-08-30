"""The first checkpoint is pinned at the interval: the compaction floor.

Checkpoint spacing adapts to the prompt, so after a long conversation the
first mid checkpoint used to sit at prompt/4 -- tens of thousands of tokens
past the system prefix. A compacted history (or any prompt sharing only the
conversation's opening) then reused NOTHING on a recurrent arch and
reprefilled from zero, system prompt included. Pinning one checkpoint at
`prefill_checkpoint_interval` gives that traffic a reuse floor; these tests
hold it on the synthetic qwen4exp fixture (CPU backend, f32).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model
from tests.qwen4exp_gguf_fixture import build_qwen4exp_gguf

# A "system prompt" opening, a long conversation on top of it, and a
# compaction: the same opening, then a summary the cache has never seen.
OPENING = [(t * 37 + 11) % 96 for t in range(16)]
CONVERSATION = OPENING + [(t * 13 + 5) % 96 for t in range(80)]
COMPACTED = OPENING + [(t * 29 + 3) % 96 for t in range(24)]
CONTINUATION = 4
CONTEXT = 256
INTERVAL = 8


class EarlyCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen4exp.gguf"
        build_qwen4exp_gguf(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def setUp(self) -> None:
        V2Model.select_backend("cpu")

    def tearDown(self) -> None:
        V2Model.select_backend("auto")

    def _generate(self, runtime, prompt: list[int]) -> list[int]:
        out: list[int] = []
        runtime.generate(
            prompt, CONTINUATION,
            lambda t: (out.append(t) or len(out) < CONTINUATION))
        return out

    def test_a_compacted_conversation_reuses_the_pinned_checkpoint(self) -> None:
        # Cold baseline: what the compacted prompt produces on a runtime that
        # has seen nothing -- reuse through the pinned checkpoint must decode
        # identically, or the floor is a corruption, not a win.
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT,
                    prefill_checkpoint_interval=INTERVAL) as runtime:
                runtime.prepare()
                expected = self._generate(runtime, COMPACTED)
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT,
                    prefill_checkpoint_interval=INTERVAL) as runtime:
                runtime.prepare()
                self._generate(runtime, CONVERSATION)
                produced = self._generate(runtime, COMPACTED)
                info = runtime.info
        # The divergence sits at len(OPENING); the pinned checkpoint at
        # INTERVAL < len(OPENING) is the floor. Without the pin the first
        # checkpoint of the 96-token prefill sat at 96/3 = 32, past the
        # opening, and reuse was zero.
        self.assertGreaterEqual(
            info["prefix_cache_last_reused_tokens"], INTERVAL,
            "the compacted prompt found no checkpoint inside the opening")
        self.assertLess(
            info["prefix_cache_last_reused_tokens"], len(OPENING) + 1)
        self.assertEqual(produced, expected)

    def test_short_prompts_keep_their_dense_coverage(self) -> None:
        # The pin must not thin out small-prompt checkpoints: a divergence
        # late in a short conversation still finds a nearby checkpoint.
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT,
                    prefill_checkpoint_interval=INTERVAL) as runtime:
                runtime.prepare()
                self._generate(runtime, CONVERSATION)
                # Diverge late: same first 64 tokens, different tail.
                late = CONVERSATION[:64] + [(t * 31 + 7) % 96
                                            for t in range(16)]
                self._generate(runtime, late)
                info = runtime.info
        # Mids spread at max(8, 96/3) = 32: the checkpoint at 64 covers the
        # late divergence just as the old uniform spacing did.
        self.assertGreaterEqual(
            info["prefix_cache_last_reused_tokens"], 32)


if __name__ == "__main__":
    unittest.main()
