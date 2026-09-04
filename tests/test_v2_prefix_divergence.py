"""Divergence capture: which slot a prompt landed on and WHERE it split.

The runtime records, at every admission, the routed slot, the tokens that
slot held, and a short window of token ids from each side of the point where
the new prompt stops matching the cached history -- captured before reuse
truncates that history, which is the only moment the old side still exists.
The server turns this into the "history rewritten at token N" log line, so a
capture that silently reads empty would make client-side rewrites (the
dominant cause of surprise reprefills) undiagnosable again.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model
from tests.qwen4exp_gguf_fixture import build_qwen4exp_gguf

PROMPT = [(t * 37 + 11) % 96 for t in range(24)]
CONTINUATION = 4
CONTEXT = 256


class PrefixDivergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen4exp.gguf"
        build_qwen4exp_gguf(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _generate(self, runtime, prompt: list[int]) -> list[int]:
        produced: list[int] = []
        runtime.generate(
            prompt, CONTINUATION,
            lambda t: (produced.append(t) or len(produced) < CONTINUATION))
        return produced

    def test_rewrite_is_captured_with_both_sides(self) -> None:
        V2Model.select_backend("cpu")
        try:
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(
                        context_limit=CONTEXT,
                        parallel_sequences=1) as runtime:
                    runtime.prepare()
                    self._generate(runtime, PROMPT)
                    committed = int(runtime.info["position"])
                    # Rewrite history from token 8: same opening, then a tail
                    # the cache has never seen.
                    rewritten = PROMPT[:8] + [(t + 1) % 96 for t in PROMPT[8:]]
                    self._generate(runtime, rewritten)
                    info = runtime.info
                    self.assertEqual(info["prefix_cache_last_slot"], 0)
                    # The slot held the whole first conversation at admission.
                    self.assertEqual(
                        info["prefix_cache_last_cached_tokens"], committed)
                    self.assertEqual(info["prefix_cache_last_lcp_live"], 8)
                    # Both sides of the split, as token ids. The old side runs
                    # from the split to the end of the cached history (well
                    # under the 32-token snippet window here); its first token
                    # is the one the rewrite replaced.
                    old_count = int(info["prefix_cache_last_old_count"])
                    new_count = int(info["prefix_cache_last_new_count"])
                    self.assertEqual(old_count, committed - 8)
                    self.assertEqual(
                        info["prefix_cache_last_old_tokens"][0], PROMPT[8])
                    self.assertEqual(
                        info["prefix_cache_last_old_tokens"][:len(PROMPT) - 8],
                        PROMPT[8:])
                    self.assertEqual(
                        info["prefix_cache_last_new_tokens"][:new_count],
                        rewritten[8:8 + new_count])
        finally:
            V2Model.select_backend("auto")

    def test_clean_extension_reports_no_split(self) -> None:
        V2Model.select_backend("cpu")
        try:
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(
                        context_limit=CONTEXT,
                        parallel_sequences=1) as runtime:
                    runtime.prepare()
                    produced = self._generate(runtime, PROMPT)
                    committed = int(runtime.info["position"])
                    # PROMPT plus everything generated covers the committed
                    # history whether or not the final sampled token was fed
                    # back, so appending fresh tokens is a pure extension.
                    extended = (PROMPT + produced)[:committed] + [
                        (t * 13 + 5) % 96 for t in range(4)]
                    self._generate(runtime, extended)
                    info = runtime.info
                    # Nothing was rewritten: the split sits exactly at the end
                    # of the cached history and the old side is empty.
                    self.assertEqual(
                        info["prefix_cache_last_cached_tokens"], committed)
                    self.assertEqual(
                        info["prefix_cache_last_lcp_live"], committed)
                    self.assertEqual(info["prefix_cache_last_old_count"], 0)
                    self.assertEqual(
                        info["prefix_cache_last_new_tokens"][:4],
                        extended[committed:committed + 4])
        finally:
            V2Model.select_backend("auto")


if __name__ == "__main__":
    unittest.main()
