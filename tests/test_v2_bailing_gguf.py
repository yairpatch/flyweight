"""Running BailingMoE3 from a GGUF conversion rather than an HF checkpoint."""

from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from pathlib import Path

from colibri_next.v2 import BailingRuntime, V2Model
from tests import bailing_gguf_fixture as gguf
from tests import hf_safetensors_fixture as safetensors


class BailingGgufTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.hf_path = safetensors.build(root / "ling-tiny-fixture")
        cls.gguf_path = gguf.build(root / "converted", cls.hf_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_it_reads_the_architecture_and_geometry_from_gguf_metadata(self) -> None:
        with V2Model(self.gguf_path) as model:
            info = model.info
            self.assertEqual(info["format"], "gguf")
            self.assertEqual(info["architecture"], "bailingmoe3")
            config = model.config
            # Derived from the per-layer kv head counts: the GGUF carries no
            # full_attention_interval of its own.
            self.assertEqual(config["full_attention_interval"], 4)
            self.assertEqual(config["expert_group_count"], 2)
            self.assertEqual(config["expert_group_used"], 1)

    @staticmethod
    @contextlib.contextmanager
    def unquantized():
        """Open the safetensors side f32, as the conversion is written.

        The loader otherwise quantizes on the way in, and a difference of that
        size would swamp the mapping mistakes these tests exist to catch.
        """
        previous = os.environ.get("COLIBRI_HF_QUANT")
        os.environ["COLIBRI_HF_QUANT"] = "F32"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("COLIBRI_HF_QUANT", None)
            else:
                os.environ["COLIBRI_HF_QUANT"] = previous

    def test_it_tokenizes_the_same_text_the_same_way(self) -> None:
        text = "the quick brown fox abc 123"
        with V2Model(self.gguf_path) as model:
            from_gguf = model.tokenize(text)
        with V2Model(self.hf_path) as model:
            from_hf = model.tokenize(text)
        self.assertEqual(from_gguf, from_hf)

    def logits(self, path, prompt: list[int]) -> list[float]:
        with V2Model(path) as model:
            runtime = BailingRuntime(model, capacity=64)
            try:
                runtime.reset()
                return list(runtime.eval(prompt))
            finally:
                runtime.close()

    def test_a_conversion_answers_exactly_as_the_checkpoint_it_came_from(self) -> None:
        # The same weights through both loaders. Everything the conversion
        # renames, splits, transposes or exponentiates is covered by this one
        # assertion, and each of those is a mistake that produces fluent but
        # wrong text rather than an error.
        #
        # More than one token matters: a rope or recurrence mistake is exactly
        # zero at position 0 and only appears once there is history, which is
        # how a broken kernel launch passed a single-token check.
        prompt = [5, 11, 23, 4, 9, 17, 3, 8]
        for length in (1, 2, len(prompt)):
            with self.subTest(prompt_tokens=length):
                converted = self.logits(self.gguf_path, prompt[:length])
                with self.unquantized():
                    original = self.logits(self.hf_path, prompt[:length])
                self.assertEqual(len(converted), len(original))
                worst = max(abs(a - b) for a, b in zip(converted, original))
                scale = max(abs(value) for value in original)
                self.assertLess(worst, 1e-3 * max(scale, 1.0))
                self.assertEqual(
                    converted.index(max(converted)),
                    original.index(max(original)),
                )

    def test_the_resume_separator_is_the_markup_the_template_emits(self) -> None:
        # Resuming a cached conversation splices this between the reused turn
        # and the next one, and an agentic loop resumes on EVERY turn: a turn
        # that ends in a tool call is cancelled rather than finished on EOS, so
        # the separator is what closes it.
        #
        # The ChatML default was not just wrong markup, it was not markup at
        # all -- <|im_end|> is absent from this vocabulary and tokenizes as
        # five ordinary text tokens, which is a literal string of junk in the
        # middle of the conversation. So assert the separator is a single token
        # and that the template really emits it between turns.
        from colibri_next.v2_server import NativeV2Tokenizer

        with V2Model(self.gguf_path) as model:
            tokenizer = NativeV2Tokenizer(model)
            separator = tokenizer.turn_separator
            self.assertEqual(len(model.tokenize(separator)), 1, separator)
            rendered = tokenizer.format_messages([
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ])
            # What the template puts between a closed assistant turn and the
            # next user turn is exactly separator + finished_turn_separator.
            self.assertIn(
                "two" + separator + tokenizer.finished_turn_separator,
                rendered,
            )

    def test_it_decodes_the_same_continuation(self) -> None:
        prompt = [5, 11, 23, 4, 9, 17, 3, 8]

        def generate(path) -> list[int]:
            with V2Model(path) as model:
                runtime = BailingRuntime(model, capacity=64)
                try:
                    runtime.reset()
                    runtime.eval_into(prompt)
                    out = []
                    for _ in range(6):
                        out.append(runtime.sample())
                        runtime.eval_into([out[-1]])
                    return out
                finally:
                    runtime.close()

        converted = generate(self.gguf_path)
        with self.unquantized():
            original = generate(self.hf_path)
        self.assertEqual(converted, original)


if __name__ == "__main__":
    unittest.main()
