"""Native Laguna planning, execution and tokenization tests."""

from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from colibri_next.v2 import V2Model
from colibri_next.v2_server import NativeV2Tokenizer

from tests.laguna_gguf_fixture import LagunaSpec, build_laguna_gguf


_WORKSPACES: list[tempfile.TemporaryDirectory] = []


def tearDownModule():
    # These fixtures are megabytes each and /tmp is usually a RAM-backed
    # tmpfs, so a directory per test adds up fast across repeated runs.
    for holder in _WORKSPACES:
        holder.cleanup()
    _WORKSPACES.clear()


def _workspace(prefix: str) -> Path:
    holder = tempfile.TemporaryDirectory(prefix=prefix)
    _WORKSPACES.append(holder)
    return Path(holder.name)


def _model(**kwargs) -> tuple[V2Model, LagunaSpec]:
    directory = _workspace("colibri-laguna-")
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
        directory = _workspace("colibri-laguna-bad-")
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


class LagunaTerminatorTests(unittest.TestCase):
    def test_config_exposes_the_gguf_terminator_ids(self):
        model, _ = _model()
        try:
            config = model.config
            self.assertEqual(config["eos_token_id"], 0)
            self.assertEqual(config["eot_token_id"], 2)
        finally:
            model.close()

    def test_stop_set_includes_end_of_turn_not_just_end_of_text(self):
        """A turn-ending token has to stop generation.

        Laguna closes an assistant turn with </assistant> (its eot) and does not
        emit eos in conversation, so a stop set built from eos alone runs
        straight into a hallucinated next turn.
        """
        model, _ = _model()
        try:
            tokenizer = NativeV2Tokenizer(model)
            self.assertIn(2, tokenizer.eos_token_ids)
            self.assertIn(0, tokenizer.eos_token_ids)
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

    def test_batched_prefill_matches_token_by_token(self):
        """The rows forward must agree with the one-token path.

        This is the load-bearing check on the batched prefill: it exercises the
        per-layer head count, both RoPE configurations, the softplus gate and
        sigmoid routing through a second, independently written code path, and
        any disagreement in those shows up as a different continuation.
        """
        prompt = [7, 11, 3, 29, 5, 17, 23, 2, 13, 19]
        results = {}
        for rows in ("1", "8"):
            with unittest.mock.patch.dict(os.environ, {"COLIBRI_PREFILL_ROWS": rows}):
                model, _ = _model()
                runtime = _native(model)
                try:
                    produced: list[int] = []
                    runtime.generate(prompt, 6, produced.append)
                    results[rows] = produced
                finally:
                    runtime.close()
                    model.close()
        self.assertEqual(results["1"], results["8"])

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


@unittest.skipUnless(
    os.environ.get("COLIBRI_TEST_LAGUNA_MODEL")
    and Path(os.environ["COLIBRI_TEST_LAGUNA_MODEL"]).is_file(),
    "set COLIBRI_TEST_LAGUNA_MODEL to a Laguna GGUF checkpoint",
)
class LagunaRealModelTests(unittest.TestCase):
    def test_whole_layer_placement_matches_cpu_experts(self):
        path = os.environ["COLIBRI_TEST_LAGUNA_MODEL"]
        with V2Model(path) as model:
            tokens = model.tokenize(
                "<s><user>What is the capital of France?</user><assistant>"
            )
            expected: list[int] = []
            with model.native_qwen_runtime(
                context_limit=512,
                moe_device="cpu",
                prefill_cache_seed="off",
            ) as runtime:
                runtime.prepare()
                runtime.generate(tokens, 4, expected.append)

            actual: list[int] = []
            with (
                unittest.mock.patch.dict(
                    os.environ, {"COLIBRI_LAGUNA_WHOLE_LAYERS": "auto"}
                ),
                model.native_qwen_runtime(
                    context_limit=512,
                    moe_device="auto",
                    prefill_cache_seed="auto",
                ) as runtime,
            ):
                runtime.prepare()
                runtime.generate(tokens, 4, actual.append)

            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
