"""Native Laguna planning, execution and tokenization tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from colibri_next.v2 import V2Model
from colibri_next.v2_server import NativeV2Tokenizer

from tests.laguna_gguf_fixture import LagunaSpec, build_laguna_gguf


def _model(**kwargs) -> tuple[V2Model, LagunaSpec]:
    directory = Path(tempfile.mkdtemp(prefix="colibri-laguna-"))
    path = directory / "laguna.gguf"
    spec = build_laguna_gguf(path, **kwargs)
    return V2Model(path), spec


def _native(model: V2Model, **options):
    if not V2Model.gpu_info()["available"]:
        raise unittest.SkipTest("native CUDA runtime is unavailable")
    runtime = model.native_runtime(
        context_limit=64, mtp_drafts=0, expert_mode="cpu", **options
    )
    runtime.prepare()
    return runtime


class LagunaConfigTests(unittest.TestCase):
    def test_config_reports_the_laguna_geometry(self):
        model, spec = _model()
        try:
            config = model.config
            self.assertEqual(config["architecture"], "laguna")
            self.assertEqual(config["layer_count"], spec.layers)
            self.assertEqual(config["hidden_size"], spec.hidden)
            self.assertEqual(config["expert_count"], spec.experts)
            self.assertEqual(config["expert_used_count"], spec.experts_used)
            # The scalar head count is unset in the file and has to be widened
            # from the per-layer array, since workspaces are sized off it.
            self.assertEqual(config["attention_heads"], spec.swa_heads)
        finally:
            model.close()

    def test_sliding_window_pattern_is_period_four_starting_full(self):
        model, spec = _model()
        try:
            self.assertEqual(
                model.config["attention_windows"],
                tuple(
                    0 if layer % 4 == 0 else spec.sliding_window
                    for layer in range(spec.layers)
                ),
            )
        finally:
            model.close()

    def test_head_count_disagreeing_with_the_layout_is_rejected(self):
        # The per-layer head count is the only independent witness of the
        # implied sliding-window layout, so a contradiction must not be
        # silently accepted.
        directory = Path(tempfile.mkdtemp(prefix="colibri-laguna-bad-"))
        path = directory / "laguna.gguf"
        spec = LagunaSpec()
        build_laguna_gguf(path, spec)
        raw = bytearray(path.read_bytes())
        # Flip layer 1 (sliding) to the full-attention head count.
        needle = b"".join(
            int(spec.heads(layer)).to_bytes(4, "little") for layer in range(spec.layers)
        )
        replacement = bytearray(needle)
        replacement[4:8] = int(spec.full_heads).to_bytes(4, "little")
        self.assertIn(needle, raw)
        raw = raw.replace(needle, bytes(replacement), 1)
        path.write_bytes(bytes(raw))
        with self.assertRaises(Exception) as caught:
            V2Model(path).close()
        self.assertIn("head count", str(caught.exception))


class LagunaTokenizerTests(unittest.TestCase):
    def test_control_tokens_map_to_their_reserved_ids(self):
        model, _ = _model()
        try:
            eos = model.token_id("〈|EOS|〉")
            self.assertEqual(model.tokenize("〈|EOS|〉"), [eos])
            # And they are recognized mid-text, not just in isolation.
            tokens = model.tokenize("a〈|EOS|〉b")
            self.assertIn(eos, tokens)
            self.assertEqual(tokens[tokens.index(eos) - 1], model.token_id("a"))
        finally:
            model.close()

    def test_digits_are_split_one_per_token(self):
        # The laguna pre-tokenizer matches \p{N} a single digit at a time, so a
        # number must never merge into one piece.
        model, _ = _model()
        try:
            self.assertEqual(
                model.tokenize("123"),
                [model.token_id(digit) for digit in "123"],
            )
        finally:
            model.close()


class LagunaChatTemplateTests(unittest.TestCase):
    class _Stub:
        info = {"architecture": "laguna"}

        def token_id(self, text):
            raise KeyError(text)

    def _tokenizer(self) -> NativeV2Tokenizer:
        return NativeV2Tokenizer(self._Stub())

    def test_prompt_wraps_turns_in_role_tags(self):
        rendered = self._tokenizer().format_messages(
            [{"role": "user", "content": "hi"}], enable_thinking=False
        )
        self.assertTrue(rendered.startswith("〈|EOS|〉<system>"))
        self.assertIn("<user>hi</user>", rendered)
        self.assertTrue(rendered.endswith("<assistant><think></think>"))

    def test_thinking_leaves_the_reasoning_block_open(self):
        rendered = self._tokenizer().format_messages(
            [{"role": "user", "content": "hi"}], enable_thinking=True
        )
        self.assertTrue(rendered.endswith("<assistant><think>"))

    def test_a_system_message_replaces_the_default(self):
        rendered = self._tokenizer().format_messages(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            enable_thinking=False,
        )
        self.assertIn("<system>be terse</system>", rendered)
        self.assertNotIn("Poolside", rendered)


class LagunaNativeTests(unittest.TestCase):
    def test_runtime_plans_and_decodes(self):
        model, spec = _model()
        runtime = _native(model)
        try:
            info = runtime.info
            # Twelve of forty-eight in the real model; two of eight here.
            self.assertEqual(
                info["swa_layers"], sum(1 for l in range(spec.layers) if l % 4)
            )
            token = runtime.decode(5)
            self.assertGreaterEqual(token, 0)
            self.assertLess(token, spec.vocabulary)
        finally:
            runtime.close()
            model.close()

    def test_decode_is_deterministic_across_runtimes(self):
        model, _ = _model()
        first = _native(model)
        second = _native(model)
        try:
            self.assertEqual(
                [first.decode(t) for t in (3, 4, 5)],
                [second.decode(t) for t in (3, 4, 5)],
            )
        finally:
            first.close()
            second.close()
            model.close()


if __name__ == "__main__":
    unittest.main()
