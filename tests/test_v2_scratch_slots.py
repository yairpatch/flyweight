"""Heterogeneous slot sizing: side slots need not span the whole context.

Every slot reserves its full context at prepare whether a conversation fills it
or not, and on a small card that reservation comes out of the expert cache
(plans/paged-kv-cache.md). Agentic side traffic -- title detection, quota
summaries, short subagent calls -- never fills a full-context slot, so
`scratch_context` sizes the slots past the first for what they actually hold.

The load-bearing claims, in order of how badly they fail if wrong:

  * a prompt is never routed to a slot too small to hold it;
  * a scratch slot decodes exactly what a full slot decodes;
  * the reservation actually shrinks.

Run on the synthetic qwen4exp fixture (CPU backend, f32).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Error, V2Model

PROMPT = [(t * 37 + 11) % 96 for t in range(24)]
LONG_PROMPT = [(t * 37 + 11) % 96 for t in range(96)]
CONTINUATION = 4
CONTEXT = 256
SCRATCH = 64


class ScratchSlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen4exp.gguf"
        build = __import__("tests.qwen4exp_gguf_fixture", fromlist=["x"])
        build.build_qwen4exp_gguf(cls.path)

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

    def test_scratch_slots_shrink_the_reservation(self) -> None:
        def reserved(scratch: int) -> tuple[int, int]:
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(
                        context_limit=CONTEXT, parallel_sequences=3,
                        scratch_context=scratch) as runtime:
                    runtime.prepare()
                    info = runtime.info
                    return int(info["kv_reserved_bytes"]), int(info["state_bytes"])

        symmetric, full_slot = reserved(0)
        mixed, mixed_full = reserved(SCRATCH)
        # The full-context slot is untouched; only the two side slots shrink.
        self.assertEqual(full_slot, mixed_full)
        self.assertEqual(symmetric, 3 * full_slot)
        self.assertLess(mixed, symmetric)
        self.assertGreater(mixed, full_slot,
                           "scratch slots still reserve their own state")

    def test_a_long_prompt_never_lands_on_a_scratch_slot(self) -> None:
        """The correctness claim: routing must respect capacity.

        LONG_PROMPT does not fit SCRATCH, so however the router scores the
        slots it has to pick the full one -- and the tokens have to match what
        a symmetric runtime produces.
        """
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT, parallel_sequences=3) as runtime:
                runtime.prepare()
                expected = self._generate(runtime, LONG_PROMPT)
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT, parallel_sequences=3,
                    scratch_context=SCRATCH) as runtime:
                runtime.prepare()
                full_bytes = int(runtime.info["state_bytes"])
                # A short request first, so the full slot is the used one and
                # the LRU rule would prefer a scratch slot if capacity were not
                # checked.
                self._generate(runtime, PROMPT)
                self.assertEqual(self._generate(runtime, LONG_PROMPT), expected)
                # state_bytes reports the ACTIVE slot's arena, so this is the
                # assertion that the long prompt really landed on the full
                # slot rather than producing the right tokens by luck.
                self.assertEqual(int(runtime.info["state_bytes"]), full_bytes)

    def test_a_short_request_decodes_the_same_on_a_scratch_slot(self) -> None:
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT, parallel_sequences=2) as runtime:
                runtime.prepare()
                expected = self._generate(runtime, PROMPT)
        other = [(t * 53 + 7) % 96 for t in range(40)]
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT, parallel_sequences=2,
                    scratch_context=SCRATCH) as runtime:
                runtime.prepare()
                full_bytes = int(runtime.info["state_bytes"])
                # Occupy the full slot, then send the short request, which the
                # router should put on scratch.
                self._generate(runtime, other)
                self.assertEqual(self._generate(runtime, PROMPT), expected)
                # Same tokens is the correctness claim; landing on the smaller
                # arena is what makes it a test of scratch slots at all.
                self.assertLess(int(runtime.info["state_bytes"]), full_bytes)

    def test_nonsensical_scratch_settings_are_refused(self) -> None:
        with V2Model(str(self.path)) as model:
            # Bigger than the context is the full slot under another name.
            with self.assertRaisesRegex(V2Error, "exceeds the context window"):
                with model.native_qwen_runtime(
                        context_limit=CONTEXT, parallel_sequences=2,
                        scratch_context=CONTEXT * 2) as runtime:
                    runtime.prepare()
            # With one slot there are no side slots to shrink, so accepting it
            # would report a saving that never happened.
            with self.assertRaisesRegex(V2Error, "needs --parallel 2"):
                with model.native_qwen_runtime(
                        context_limit=CONTEXT, parallel_sequences=1,
                        scratch_context=SCRATCH) as runtime:
                    runtime.prepare()


if __name__ == "__main__":
    unittest.main()
