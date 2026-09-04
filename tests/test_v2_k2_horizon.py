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


def _mova_spec(**overrides) -> K2HorizonSpec:
    """The 36B-A4B shape in miniature: leading dense blocks, then MoE + MoVA."""
    settings = dict(
        layers=3,
        leading_dense=1,
        experts=4,
        experts_used=2,
        expert_intermediate=64,
        shared_intermediate=64,
        value_experts=4,
        value_experts_used=2,
        attention_gate=True,
    )
    settings.update(overrides)
    return K2HorizonSpec(**settings)


class K2HorizonMoVATests(unittest.TestCase):
    def test_config_reports_the_value_expert_geometry(self):
        model, spec = _model(spec=_mova_spec())
        try:
            config = model.config
            self.assertEqual(config["expert_count"], spec.experts)
            self.assertEqual(config["expert_used_count"], spec.experts_used)
            self.assertEqual(config["value_expert_count"], spec.value_experts)
            self.assertEqual(config["value_expert_used_count"], spec.value_experts_used)
        finally:
            model.close()

    def test_a_used_count_above_the_expert_count_is_rejected(self):
        model, _ = _model(spec=_mova_spec(value_experts=4, value_experts_used=8))
        try:
            with self.assertRaisesRegex(Exception, "value_expert_used_count"):
                model.native_runtime(context_limit=64, mtp_drafts=0, expert_mode="cpu")
        finally:
            model.close()

    def test_runtime_plans_and_decodes(self):
        model, spec = _model(spec=_mova_spec())
        runtime = _native(model)
        try:
            info = runtime.info
            # Every layer past the leading dense block routes its values.
            self.assertEqual(info["mova_layers"], spec.layers - spec.leading_dense)
            token = runtime.decode(5)
            self.assertGreaterEqual(token, 0)
            self.assertLess(token, spec.vocabulary)
        finally:
            runtime.close()
            model.close()

    def test_batched_prefill_matches_token_by_token(self):
        """The MoVA and gate paths differ between rows=1 and rows=8; they must agree."""
        prompt = [7, 11, 3, 29, 5, 17, 23, 2, 13, 19]
        results = {}
        for rows in ("1", "8"):
            with unittest.mock.patch.dict(os.environ, {"FLYWEIGHT_PREFILL_ROWS": rows}):
                model, _ = _model(spec=_mova_spec())
                runtime = _native(model)
                try:
                    produced: list[int] = []
                    runtime.generate(prompt, 6, produced.append)
                    results[rows] = produced
                finally:
                    runtime.close()
                    model.close()
        self.assertEqual(results["1"], results["8"])

    def test_the_attention_gate_changes_the_output(self):
        """A gate that was projected but never applied would go unnoticed."""
        produced = {}
        for gated in (False, True):
            model, _ = _model(spec=_mova_spec(attention_gate=gated), seed=7)
            runtime = _native(model)
            try:
                tokens: list[int] = []
                runtime.generate([7, 11, 3, 29, 5], 6, tokens.append)
                produced[gated] = tokens
            finally:
                runtime.close()
                model.close()
        self.assertNotEqual(produced[False], produced[True])


class K2HorizonValueCacheTests(unittest.TestCase):
    """The value experts go through their own device cache; the host path is
    the reference it must agree with."""

    def test_cached_value_experts_match_the_host_path(self):
        prompt = [7, 11, 3, 29, 5, 17, 23, 2, 13, 19]
        produced = {}
        for mode in ("cpu", "auto"):
            model, _ = _model(spec=_mova_spec(), seed=11)
            if not V2Model.gpu_info()["available"]:
                raise unittest.SkipTest("native CUDA runtime is unavailable")
            runtime = model.native_runtime(context_limit=64, mtp_drafts=0, expert_mode=mode)
            runtime.prepare()
            try:
                tokens: list[int] = []
                runtime.generate(prompt, 12, tokens.append)
                produced[mode] = (tokens, dict(runtime.info))
            finally:
                runtime.close()
                model.close()
        cpu_tokens, cpu_info = produced["cpu"]
        auto_tokens, auto_info = produced["auto"]
        self.assertEqual(cpu_tokens, auto_tokens)
        # The host path never allocates a cache; the device path must have
        # one and must actually serve routes from it once it warms.
        self.assertEqual(cpu_info["mova_cache_bytes"], 0)
        self.assertGreater(auto_info["mova_cache_bytes"], 0)
        self.assertGreater(auto_info["mova_cache_admissions"], 0)
        self.assertGreater(auto_info["mova_cache_hits"], 0)

    def test_next_layer_prefetch_predicts_value_experts(self):
        """With a prefetch budget the value router's picks train the layer-
        to-layer table and produce predictions; output must not change."""
        prompt = [7, 11, 3, 29, 5, 17, 23, 2, 13, 19]
        produced = {}
        for budget in (0, 2):
            model, _ = _model(spec=_mova_spec(), seed=11)
            if not V2Model.gpu_info()["available"]:
                raise unittest.SkipTest("native CUDA runtime is unavailable")
            runtime = model.native_runtime(
                context_limit=64, mtp_drafts=0, expert_mode="auto",
                next_layer_prefetch=budget,
            )
            runtime.prepare()
            try:
                tokens: list[int] = []
                runtime.generate(prompt, 12, tokens.append)
                produced[budget] = (tokens, dict(runtime.info))
            finally:
                runtime.close()
                model.close()
        self.assertEqual(produced[0][0], produced[2][0])
        self.assertEqual(produced[0][1]["mova_prefetch_predictions"], 0)
        self.assertGreater(produced[2][1]["mova_prefetch_predictions"], 0)

    def test_value_expert_history_is_persisted_beside_the_expert_history(self):
        model, spec = _model(spec=_mova_spec())
        directory = Path(_WORKSPACES[-1].name)
        runtime = _native(model)
        try:
            runtime.generate([7, 11, 3], 4, lambda _t: None)
        finally:
            runtime.close()
            model.close()
        histories = sorted(directory.glob("*.expert-history.mova"))
        self.assertEqual(len(histories), 1, "one value-expert history per checkpoint")
        # Fixed 44-byte header, then (frequency u32, last_used u64) per
        # (layer, value expert): the same record the feed-forward table uses.
        entries = spec.layers * spec.value_experts
        self.assertEqual(histories[0].stat().st_size, 44 + 12 * entries)


class K2HorizonDecodeAttentionTests(unittest.TestCase):
    """K2 decodes through the 128-dim attention dispatch (cuBLAS, fused
    tiles, serial ring); the three must agree on greedy tokens."""

    def _tokens(self, env: dict[str, str], prompt_tokens: int = 40,
                context: int = 256, steps: int = 24) -> list[int]:
        spec = K2HorizonSpec(heads=4, kv_heads=2, head_dim=128, rotary_dim=64)
        with unittest.mock.patch.dict(os.environ, env):
            model, _ = _model(spec=spec, seed=5)
            if not V2Model.gpu_info()["available"]:
                raise unittest.SkipTest("native CUDA runtime is unavailable")
            runtime = model.native_runtime(
                context_limit=context, mtp_drafts=0, expert_mode="cpu",
                cache_type_k="f16", cache_type_v="f16",
            )
            runtime.prepare()
            try:
                produced: list[int] = []
                prompt = [8 + (i * 7) % 80 for i in range(prompt_tokens)]
                runtime.generate(prompt, steps, produced.append)
                return produced
            finally:
                runtime.close()
                model.close()

    def test_the_three_decode_attention_paths_agree(self):
        # The window has to be long enough to hit the cuBLAS threshold; the
        # threshold is lowered so a 256-token fixture reaches it.
        cublas = self._tokens({"FLYWEIGHT_CUBLAS_ATTENTION_MIN_TOKENS": "8"})
        fused = self._tokens({"FLYWEIGHT_CUBLAS_ATTENTION": "0"})
        ring = self._tokens({"FLYWEIGHT_CUBLAS_ATTENTION": "0",
                             "FLYWEIGHT_FUSED_ATTENTION": "0"})
        self.assertEqual(fused, ring)
        self.assertEqual(cublas, ring)

    def test_tensor_core_prefill_matches_the_warp_prefill(self):
        """K2 runs the cuBLAS prefill attention ungated and gates afterwards;
        forced on, it must reproduce the warp kernel's tokens."""
        # The tensor-core tiles need a visible prefix of at least 256 tokens,
        # so the prompt is long enough for the later chunks to qualify.
        common = {"FLYWEIGHT_PREFILL_ROWS": "512", "FLYWEIGHT_CUBLAS_ATTENTION": "0"}
        warp = self._tokens({**common, "FLYWEIGHT_CUBLAS_PREFILL_ATTENTION": "0"},
                            prompt_tokens=600, context=8192, steps=8)
        tensor_core = self._tokens({**common, "FLYWEIGHT_CUBLAS_PREFILL_ATTENTION": "1"},
                                   prompt_tokens=600, context=8192, steps=8)
        self.assertEqual(tensor_core, warp)


DEFAULT_REAL_MOVA_MODEL = (
    "/home/yair/Downloads/K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf"
)
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


@unittest.skipUnless(
    os.path.isfile(
        os.environ.get("FLYWEIGHT_TEST_K2_MOVA_MODEL", DEFAULT_REAL_MOVA_MODEL)
    ),
    "K2-Horizon MoVA GGUF checkpoint not available",
)
class K2HorizonMoVARealModelTests(unittest.TestCase):
    def test_real_mova_model_completion(self):
        path = os.environ.get("FLYWEIGHT_TEST_K2_MOVA_MODEL", DEFAULT_REAL_MOVA_MODEL)
        with V2Model(path) as model:
            config = model.config
            self.assertEqual(config["architecture"], "k2-horizon")
            self.assertEqual(config["layer_count"], 48)
            self.assertEqual(config["hidden_size"], 2560)
            self.assertEqual(config["norm_groups"], 2)
            self.assertEqual(config["expert_count"], 100)
            self.assertEqual(config["expert_used_count"], 8)
            self.assertEqual(config["value_expert_count"], 64)
            self.assertEqual(config["value_expert_used_count"], 4)

            tokens = model.tokenize("The capital of France is")
            with model.native_runtime(
                context_limit=512, mtp_drafts=0, expert_mode="cpu"
            ) as runtime:
                runtime.prepare()
                # 45 of the 48 blocks are MoVA; the leading three are dense.
                self.assertEqual(runtime.info["mova_layers"], 45)
                output_tokens: list[int] = []
                runtime.generate(tokens, 8, output_tokens.append)
                text = "".join(model.token_text(t) for t in output_tokens)
                print(f"\nReal K2-Horizon MoVA output: {text!r}")
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
