"""Public reset() must mean the runtime, not the active slot.

Under --parallel it used to memset only the active slot's arena: sibling
slots kept their conversations and the router happily routed a post-reset
prompt back onto one, resurrecting state across what the client meant as a
clean boundary. The internal recycle paths keep the narrow semantic under a
different name; the public entry point now clears every slot, every
checkpoint, and the host prompt cache.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model
from tests.qwen4exp_gguf_fixture import build_qwen4exp_gguf

MAIN = [(t * 37 + 11) % 96 for t in range(48)]
OTHER = [(t * 29 + 3) % 96 for t in range(40)]
CONTINUATION = 4
CONTEXT = 256


class ResetSemanticsTests(unittest.TestCase):
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

    def _generate(self, runtime, prompt: list[int]) -> None:
        out: list[int] = []
        runtime.generate(
            prompt, CONTINUATION,
            lambda t: (out.append(t) or len(out) < CONTINUATION))

    def test_reset_clears_sibling_slots_too(self) -> None:
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT,
                    parallel_sequences=2) as runtime:
                runtime.prepare()
                # Two unrelated conversations, one per slot.
                self._generate(runtime, MAIN)
                self._generate(runtime, OTHER)
                runtime.reset()
                self.assertEqual(runtime.info["position"], 0)
                # Re-sending either conversation must prefill cold: reuse of
                # any token here means a slot survived the reset.
                self._generate(runtime, MAIN)
                self.assertEqual(
                    runtime.info["prefix_cache_last_reused_tokens"], 0,
                    "a sibling slot's conversation survived reset()")
                self._generate(runtime, OTHER)
                self.assertEqual(
                    runtime.info["prefix_cache_last_reused_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
