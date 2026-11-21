"""Loading, configuration and tokenizer tests for the inkling2 architecture."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight import models
from flyweight.v2 import Inkling2Runtime, V2Model, V2Error
from flyweight.v2_server import NativeV2Tokenizer

from tests.inkling2_gguf_fixture import Inkling2Spec, build_inkling2_gguf


_WORKSPACES: list[tempfile.TemporaryDirectory] = []
_MODELS: list[V2Model] = []


def tearDownModule():
    for model in _MODELS:
        model.close()
    _MODELS.clear()
    for holder in _WORKSPACES:
        holder.cleanup()
    _WORKSPACES.clear()


def _model(**kwargs) -> tuple[V2Model, Inkling2Spec]:
    holder = tempfile.TemporaryDirectory(prefix="flyweight-inkling2-")
    _WORKSPACES.append(holder)
    path = Path(holder.name) / "inkling2.gguf"
    spec = build_inkling2_gguf(path, **kwargs)
    model = V2Model(path)
    _MODELS.append(model)
    return model, spec


class Inkling2ConfigTests(unittest.TestCase):
    def test_config_reports_the_looped_geometry(self):
        model, spec = _model()
        config = model.config
        self.assertEqual(config["architecture"], "inkling2")
        self.assertEqual(config["layer_count"], spec.applications)
        self.assertEqual(config["hidden_size"], spec.hidden)
        self.assertEqual(config["expert_count"], spec.experts)
        self.assertEqual(config["expert_used_count"], spec.experts_used)
        self.assertEqual(config["num_loops"], spec.num_loops)
        self.assertEqual(config["loop_core_start"], spec.core_start)
        self.assertEqual(config["loop_core_end"], spec.core_end)
        self.assertEqual(config["attn_res_block_size"], spec.block_size)
        self.assertEqual(config["conv_kernel"], 4)
        self.assertEqual(config["vocab_size_unpadded"], spec.vocabulary)

    def test_relative_attention_parameters_survive_parsing(self):
        model, _ = _model()
        config = model.config
        self.assertEqual(config["relative_length"], 8)
        self.assertEqual(config["relative_extent"], 16)
        self.assertEqual(config["log_scale_floor"], 8192)
        self.assertAlmostEqual(config["qk_clip"], 22.0, places=4)
        self.assertAlmostEqual(config["relative_clip"], 8.0, places=4)
        self.assertAlmostEqual(config["log_scale_alpha"], 0.1, places=4)
        self.assertAlmostEqual(config["final_logit_softcap"], 30.0, places=4)
        self.assertAlmostEqual(config["expert_weights_scale"], 8.0, places=4)
        self.assertAlmostEqual(config["situ_beta"], 4.0, places=4)
        self.assertAlmostEqual(config["situ_linear_beta"], 25.0, places=4)

    def test_architecture_is_marked_supported(self):
        self.assertEqual(models.architecture_support("inkling2"), "yes")

    def test_dedicated_runtime_runs_a_token(self):
        model, spec = _model()
        runtime = Inkling2Runtime(model, capacity=32)
        _MODELS.append(model)
        try:
            logits = runtime.eval([0])
            self.assertEqual(len(logits), spec.vocabulary)
            self.assertTrue(all((not isinstance(v, float)) or v == v for v in logits))
        finally:
            runtime.close()

    def test_qwen_runtime_refuses_inkling2(self):
        # The layer loop for Qwen has no loop controllers or relative bias,
        # so callers must use the dedicated runtime rather than get silent
        # nonsense from a missing Qwen-shaped tensor.
        model, _ = _model()
        with self.assertRaises(V2Error) as caught:
            model.native_runtime()
        self.assertIn("inkling2", str(caught.exception))
        self.assertIn("dedicated inkling2 runtime", str(caught.exception))


class Inkling2TokenizerTests(unittest.TestCase):
    def test_pretokenizer_uses_the_gpt4o_splitter(self):
        model, _ = _model()
        self.assertEqual(
            model.pretokenize("Hello world! 123"),
            ("Hello", " world", "!", " ", "123"),
        )

    def test_control_tokens_stay_whole(self):
        model, _ = _model()
        tokenizer = NativeV2Tokenizer(model)
        tokens = tokenizer.encode("<|end_message|>")
        self.assertEqual(tokens, [0])

    def test_embedded_chat_template_is_used(self):
        model, _ = _model()
        self.assertIn("messages", model.chat_template or "")
        tokenizer = NativeV2Tokenizer(model)
        self.assertEqual(tokenizer.chat_template_source, "gguf")

    def test_turn_separator_matches_the_template(self):
        model, _ = _model()
        tokenizer = NativeV2Tokenizer(model)
        self.assertEqual(tokenizer.turn_separator, "<|end_message|>")
        self.assertEqual(tokenizer.finished_turn_separator, "")


if __name__ == "__main__":
    unittest.main()
