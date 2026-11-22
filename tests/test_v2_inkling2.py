"""Loading, configuration and tokenizer tests for the inkling2 architecture."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight import models
from flyweight.v2 import Inkling2Runtime, V2Model, V2Error
from flyweight.v2_server import NativeV2InferenceService, NativeV2Tokenizer

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
        try:
            logits = runtime.eval([0])
            self.assertEqual(len(logits), spec.vocabulary)
            self.assertTrue(all((not isinstance(v, float)) or v == v for v in logits))
        finally:
            runtime.close()

    def test_gpu_decode_matches_cpu_on_nonzero_weights(self):
        model, spec = _model(seed=42)
        previous = V2Model.active_backend()
        try:
            V2Model.select_backend("cpu")
            with Inkling2Runtime(model, capacity=32) as cpu_rt:
                self.assertFalse(cpu_rt.uses_gpu)
                cpu_prefill = list(cpu_rt.eval([1, 2, 3, 4, 5, 6]))
                cpu_decode = list(cpu_rt.eval([7]))
            self.assertEqual(len(cpu_prefill), spec.vocabulary)
            self.assertTrue(any(abs(v) > 1e-6 for v in cpu_prefill))

            selected = V2Model.select_backend("auto")
            if selected == "cpu":
                self.skipTest("no GPU backend available in this session")
            with Inkling2Runtime(model, capacity=32) as gpu_rt:
                self.assertTrue(gpu_rt.uses_gpu)
                gpu_prefill = list(gpu_rt.eval([1, 2, 3, 4, 5, 6]))
                gpu_decode = list(gpu_rt.eval([7]))
            prefill_diff = max(abs(a - b) for a, b in zip(cpu_prefill, gpu_prefill))
            decode_diff = max(abs(a - b) for a, b in zip(cpu_decode, gpu_decode))
            self.assertLess(prefill_diff, 1e-3, f"GPU prefill vs CPU max diff {prefill_diff}")
            self.assertLess(decode_diff, 1e-3, f"GPU decode-after-prefill vs CPU max diff {decode_diff}")
        finally:
            V2Model.select_backend(previous)

    def test_service_health_and_streaming_chat(self):
        holder = tempfile.TemporaryDirectory(prefix="flyweight-inkling2-srv-")
        _WORKSPACES.append(holder)
        path = Path(holder.name) / "inkling2.gguf"
        build_inkling2_gguf(path, seed=7)
        previous = V2Model.active_backend()
        try:
            V2Model.select_backend("cpu")
            service = NativeV2InferenceService(path, context_window=32, max_new_tokens=4)
            try:
                health = service.health()
                self.assertEqual(health["execution"]["backend"], "native-v2-inkling2")
                self.assertEqual(health["execution"]["architecture"], "inkling2")
                chunks = list(service.stream_chat_completion({
                    "model": service.model_name,
                    "messages": [{"role": "user", "content": "Hello world!"}],
                    "max_tokens": 2,
                    "temperature": 0.0,
                    "stream": True,
                }))
                self.assertTrue(chunks)
            finally:
                service.close()
        finally:
            V2Model.select_backend(previous)

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
