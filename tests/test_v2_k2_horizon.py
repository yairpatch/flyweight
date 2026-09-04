"""Native K2-Horizon planning, execution and tokenization tests."""

from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from flyweight.v2 import V2Model
from tests.k2_horizon_gguf_fixture import K2HorizonSpec, build_k2_horizon_gguf

_WORKSPACES: list[tempfile.TemporaryDirectory] = []


def tearDownModule():
    for holder in _WORKSPACES:
        holder.cleanup()
    _WORKSPACES.clear()


def _workspace(prefix: str) -> Path:
    holder = tempfile.TemporaryDirectory(prefix=prefix)
    _WORKSPACES.append(holder)
    return Path(holder.name)


def _model(**kwargs) -> tuple[V2Model, K2HorizonSpec]:
    directory = _workspace("flyweight-k2-horizon-")
    path = directory / "k2_horizon.gguf"
    spec = build_k2_horizon_gguf(path, **kwargs)
    return V2Model(path), spec


def _native(model: V2Model, **options):
    if not V2Model.gpu_info()["available"]:
        raise unittest.SkipTest("native CUDA runtime is unavailable")
    runtime = model.native_runtime(
        context_limit=64, mtp_drafts=0, expert_mode="cpu", **options
    )
    runtime.prepare()
    return runtime


class K2HorizonConfigTests(unittest.TestCase):
    def test_config_reports_the_k2_horizon_geometry(self):
        model, spec = _model()
        try:
            config = model.config
            self.assertEqual(config["architecture"], "k2-horizon")
            self.assertEqual(config["layer_count"], spec.layers)
            self.assertEqual(config["hidden_size"], spec.hidden)
            self.assertEqual(config["attention_heads"], spec.heads)
            self.assertEqual(config["attention_kv_heads"], spec.kv_heads)
            self.assertEqual(config["norm_groups"], spec.norm_groups)
            self.assertEqual(config["rotary_dimension"], spec.rotary_dim)
            self.assertEqual(config["rope_freq_base"], spec.rope_freq_base)
        finally:
            model.close()


class K2HorizonTokenizerTests(unittest.TestCase):
    def test_control_tokens_map_to_their_reserved_ids(self):
        model, _ = _model()
        try:
            eos = model.token_id("<|endoftext|>")
            self.assertEqual(model.tokenize("<|endoftext|>"), [eos])
            tokens = model.tokenize("a<|endoftext|>b")
            self.assertIn(eos, tokens)
            self.assertEqual(tokens[tokens.index(eos) - 1], model.token_id("a"))
        finally:
            model.close()


class K2HorizonNativeTests(unittest.TestCase):
    def test_runtime_plans_and_decodes(self):
        model, spec = _model()
        runtime = _native(model)
        try:
            token = runtime.decode(5)
            self.assertGreaterEqual(token, 0)
            self.assertLess(token, spec.vocabulary)
        finally:
            runtime.close()
            model.close()

    def test_batched_prefill_matches_token_by_token(self):
        """Prefill with rows=8 must match single-token prefill (rows=1)."""
        prompt = [7, 11, 3, 29, 5, 17, 23, 2, 13, 19]
        results = {}
        for rows in ("1", "8"):
            with unittest.mock.patch.dict(os.environ, {"FLYWEIGHT_PREFILL_ROWS": rows}):
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


CANDIDATE_MODELS = (
    os.environ.get("FLYWEIGHT_TEST_K2_HORIZON_MODEL"),
    "/home/yair/Downloads/K2-Horizon-7B-Q8_0.gguf",
    "/home/yair/Downloads/gguf/models--IFM--K2-Horizon-7B-GGUF.gguf",
)
DEFAULT_REAL_MODEL = next((p for p in CANDIDATE_MODELS if p and os.path.isfile(p)), "")


@unittest.skipUnless(
    os.path.isfile(DEFAULT_REAL_MODEL),
    "K2-Horizon GGUF checkpoint not available",
)
class K2HorizonRealModelTests(unittest.TestCase):
    def test_real_model_completion(self):
        path = DEFAULT_REAL_MODEL
        with V2Model(path) as model:
            self.assertEqual(model.config["architecture"], "k2-horizon")
            self.assertEqual(model.config["norm_groups"], 4)
            self.assertEqual(model.config["layer_count"], 36)
            self.assertEqual(model.config["hidden_size"], 4096)
            self.assertEqual(model.config["intermediate_size"], 12288)

            tokens = model.tokenize("The capital of France is")
            self.assertGreater(len(tokens), 0)

            with model.native_runtime(context_limit=512, mtp_drafts=0) as runtime:
                runtime.prepare()
                output_tokens: list[int] = []
                runtime.generate(tokens, 10, output_tokens.append)
                text = "".join(model.token_text(t) for t in output_tokens)
                print(f"\nReal K2-Horizon output: {text!r}")
                self.assertIn("Paris", text)

    def test_real_model_thinking_stream(self):
        path = DEFAULT_REAL_MODEL
        from flyweight.v2_server import NativeV2InferenceService

        service = NativeV2InferenceService(
            path,
            model_name="k2-horizon",
            max_new_tokens=64,
            context_window=1024,
        )
        try:
            self.assertIn(
                service.v2_model.token_id("<|ifm|im_end|>"),
                service.generator.tokenizer.eos_token_ids,
            )
            prompt_ids = service.generator.prepare_messages(
                [{"role": "user", "content": "What is 2+2?"}],
                enable_thinking=True,
            )
            self.assertTrue(service._prompt_opens_thinking(prompt_ids))

            stream = service.stream_chat_completion({
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "stream": True,
                "max_tokens": 64,
            })
            reasoning_chunks = []
            content_chunks = []
            for item in stream:
                if isinstance(item, dict) and "choices" in item and item["choices"]:
                    delta = item["choices"][0].get("delta", {})
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        reasoning_chunks.append(delta["reasoning_content"])
                    if "content" in delta and delta["content"]:
                        content_chunks.append(delta["content"])

            reasoning = "".join(reasoning_chunks)
            content = "".join(content_chunks)
            print(f"\nStreamed reasoning: {reasoning!r}")
            print(f"Streamed content: {content!r}")
            self.assertGreater(len(reasoning), 0)
            self.assertNotIn("<ifm|think>", reasoning)
            self.assertNotIn("</ifm|think>", reasoning)
            self.assertNotIn("<ifm|think>", content)
            self.assertNotIn("</ifm|think>", content)
            self.assertNotIn("<|ifm|im_end|>", content)
            self.assertNotIn("<|ifm|im_end|>", reasoning)
        finally:
            service.close()

    @unittest.skipUnless(
        os.path.isfile("/home/yair/Downloads/K2-Horizon-7B-Q8_0.gguf"),
        "K2-Horizon Q8_0 checkpoint not available",
    )
    def test_q8_0_quant_completion(self):
        with V2Model("/home/yair/Downloads/K2-Horizon-7B-Q8_0.gguf") as model:
            q8_count = sum(1 for t in model.tensors() if t.get("ggml_type") == 8)
            self.assertGreater(q8_count, 200)

            tokens = model.tokenize("The capital of France is")
            with model.native_runtime(context_limit=512, mtp_drafts=0) as runtime:
                runtime.prepare()
                output_tokens: list[int] = []
                runtime.generate(tokens, 10, output_tokens.append)
                text = "".join(model.token_text(t) for t in output_tokens)
                self.assertIn("Paris", text)


class K2HorizonThinkingTests(unittest.TestCase):
    def test_thinking_budget_meters_k2_horizon_blocks(self):
        from flyweight.v2_server import _ThinkingBudget

        meter = _ThinkingBudget(budget=3, thinking_open=True)
        self.assertFalse(meter.spend("analyzing "))
        self.assertFalse(meter.spend("step 1 "))
        self.assertTrue(meter.spend("step 2 "))

        meter.close()
        self.assertFalse(meter.inside)

        meter = _ThinkingBudget(budget=2, thinking_open=False)
        self.assertFalse(meter.spend("hello "))
        self.assertFalse(meter.inside)
        self.assertFalse(meter.spend("<ifm|think_fast>first"))
        self.assertTrue(meter.inside)
        self.assertTrue(meter.spend("second"))

    def test_split_reasoning_content_k2_tags(self):
        from flyweight.server import _split_reasoning_content

        vis, rsn = _split_reasoning_content("<ifm|think>deep thought</ifm|think>Result")
        self.assertEqual(vis, "Result")
        self.assertEqual(rsn, "deep thought")

        vis, rsn = _split_reasoning_content("<ifm|think_fast>quick thought</ifm|think_fast>Result")
        self.assertEqual(vis, "Result")
        self.assertEqual(rsn, "quick thought")

        vis, rsn = _split_reasoning_content("<ifm|think_faster>lightning thought</ifm|think_faster>Result")
        self.assertEqual(vis, "Result")
        self.assertEqual(rsn, "lightning thought")

        vis, rsn = _split_reasoning_content("reasoning text\n</ifm|think>\nFinal answer")
        self.assertEqual(vis, "Final answer")
        self.assertEqual(rsn, "reasoning text")


if __name__ == "__main__":
    unittest.main()
