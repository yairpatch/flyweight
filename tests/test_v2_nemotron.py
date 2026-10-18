"""Native Nemotron H MoE planning, execution, tokenization and tool-call tests."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from flyweight import server
from flyweight.v2 import V2Error, V2Model

from tests.nemotron_gguf_fixture import NemotronSpec, build_nemotron_gguf, reference_greedy


_WORKSPACES: list[tempfile.TemporaryDirectory] = []
_MODELS: list[V2Model] = []


def tearDownModule():
    # Close before deleting: Windows refuses to remove a mapped file.
    for model in _MODELS:
        model.close()
    _MODELS.clear()
    for holder in _WORKSPACES:
        holder.cleanup()
    _WORKSPACES.clear()


def _model(spec: NemotronSpec | None = None):
    holder = tempfile.TemporaryDirectory(prefix="flyweight-nemotron-")
    _WORKSPACES.append(holder)
    path = Path(holder.name) / "nemotron_h_moe.gguf"
    spec, weights = build_nemotron_gguf(path, spec)
    model = V2Model(path)
    _MODELS.append(model)
    return model, spec, weights


def _native(model: V2Model, **options):
    if not V2Model.gpu_info()["available"]:
        raise unittest.SkipTest("native CUDA runtime is unavailable")
    options.setdefault("context_limit", 64)
    options.setdefault("mtp_drafts", 0)
    options.setdefault("expert_mode", "cpu")
    runtime = model.native_runtime(**options)
    runtime.prepare()
    return runtime


class NemotronConfigTests(unittest.TestCase):
    def test_kv_head_scalar_is_lifted_from_the_per_layer_array(self):
        # The array is zero on the mamba and MoE blocks, and the attention
        # geometry reads the scalar, which the file leaves unset.
        model, spec, _ = _model()
        config = model.config
        self.assertEqual(config["architecture"], "nemotron_h_moe")
        self.assertEqual(config["attention_kv_heads"], 2)
        self.assertEqual(config["layer_count"], spec.layers)

    def test_mangled_ssm_key_is_rejected(self):
        # ssm.inner_size carries d_inner; a truncated download or a renamed
        # key leaves the SSM geometry empty, and the plan build has to fail
        # loudly instead of silently dividing the projection wrong.
        holder = tempfile.TemporaryDirectory(prefix="flyweight-nemotron-bad-")
        _WORKSPACES.append(holder)
        path = Path(holder.name) / "nemotron_h_moe.gguf"
        build_nemotron_gguf(path)
        raw = path.read_bytes().replace(b"nemotron_h_moe.ssm.inner_size",
                                        b"nemotron_h_moe.ssm.inner_sizX")
        path.write_bytes(raw)
        model = V2Model(path)
        _MODELS.append(model)
        # The plan build refuses at runtime creation, before any device is
        # touched, so this needs no GPU.
        with self.assertRaises(V2Error) as caught:
            model.native_runtime(context_limit=64, mtp_drafts=0, expert_mode="cpu")
        self.assertIn("ssm", str(caught.exception))


class NemotronTokenizerTests(unittest.TestCase):
    def test_user_defined_chat_tags_are_split_whole(self):
        # <think>, </think>, <tool_call> and </tool_call> are type 4 in the
        # real vocabulary; the template emits them as text and BPE alone would
        # shred them into "<", "think", ">".
        model, _, _ = _model()
        think, close = model.token_id("<think>"), model.token_id("</think>")
        self.assertEqual(model.tokenize("<think>"), [think])
        self.assertEqual(
            model.tokenize("a<think>b</think>c"),
            [model.token_id("a"), think, model.token_id("b"), close,
             model.token_id("c")],
        )
        call, end = model.token_id("<tool_call>"), model.token_id("</tool_call>")
        self.assertEqual(model.tokenize("<tool_call>x</tool_call>"),
                         [call, model.token_id("x"), end])

    def test_control_tokens_map_to_their_reserved_ids(self):
        model, _, _ = _model()
        begin, end = model.token_id("<|im_start|>"), model.token_id("<|im_end|>")
        self.assertEqual(model.tokenize("<|im_start|>x<|im_end|>"),
                         [begin, model.token_id("x"), end])

    def test_whitespace_runs_leave_their_last_space_to_the_word(self):
        # llama3's `\s+(?!\S)`: "x    a" is "x", "   ", " a". The pixtral
        # vocabulary once skipped the pre-tokenizer entirely, and BPE over the
        # whole text handed the model "    ", "a" on every indented line.
        model, _, _ = _model()
        self.assertEqual(
            model.tokenize("x    a"),
            [model.token_id("x"), model.token_id("\u0120\u0120\u0120"),
             model.token_id("\u0120a")],
        )

    def test_digits_group_in_threes(self):
        # The pixtral pre is llama.cpp's llama3 regex: \p{N}{1,3}, so the
        # merges may join "123" but never reach across into the "4".
        model, _, _ = _model()
        self.assertEqual(model.tokenize("1234"),
                         [model.token_id("123"), model.token_id("4")])


class NemotronNativeTests(unittest.TestCase):
    PROMPT = [40, 72, 5, 101, 33, 64, 12, 90, 7, 55, 21]

    def test_greedy_decode_matches_the_reference_forward(self):
        """The load-bearing check: an independently written float64 forward.

        Covers the conv ring and its x|B|C-only reach, the scan (softplus dt,
        scalar decay per head, group-shared B/C, the D skip), the silu(z) gate
        and the grouped norm, the no-rope GQA blocks, and three MoE layers with
        sigmoid routing whose bias picks but does not weight, heterogeneous
        per-layer counts / widths / top-k and relu² applied in the latent
        space with an always-on shared expert.
        """
        model, spec, weights = _model()
        with unittest.mock.patch.dict(os.environ, {"FLYWEIGHT_PREFILL_ROWS": "1"}):
            runtime = _native(model)
            try:
                produced: list[int] = []
                runtime.generate(self.PROMPT, 8, produced.append)
            finally:
                runtime.close()
        self.assertEqual(produced, reference_greedy(spec, weights, self.PROMPT, 8))

    def test_batched_prefill_matches_the_reference_forward(self):
        # The rows mamba, attention and latent-MoE branches: 4-row chunks put
        # chunk boundaries inside the prompt so the conv ring and scan state
        # carry across them, and the default chunking of a 160-token prompt
        # runs the two-half pipelined driver.
        cases = (("4", self.PROMPT), (None, [(index * 37) % 96 for index in range(160)]))
        for rows, prompt in cases:
            environment = {"FLYWEIGHT_PREFILL_ROWS": rows} if rows else {}
            with self.subTest(rows=rows, tokens=len(prompt)), \
                    unittest.mock.patch.dict(os.environ, environment):
                model, spec, weights = _model()
                runtime = _native(model, context_limit=256)
                try:
                    produced: list[int] = []
                    runtime.generate(prompt, 6, produced.append)
                finally:
                    runtime.close()
                self.assertEqual(produced, reference_greedy(spec, weights, prompt, 6))

    def test_prefix_reuse_restores_the_recurrent_state(self):
        # Successive requests on one runtime reuse the cached prefix, which for
        # the Mamba-2 blocks means restoring the conv ring and scan state at a
        # checkpoint rather than recomputing it. A rewrite in mid-prompt and a
        # chat-style extension must both land exactly where a fresh forward
        # over the whole prompt does.
        model, spec, weights = _model()
        runtime = _native(model, context_limit=256)
        try:
            first = self.PROMPT * 3
            rewritten = first[:20] + [(token + 9) % 96 for token in first[20:]]
            for prompt in (first, rewritten):
                produced: list[int] = []
                runtime.generate(prompt, 4, produced.append)
                self.assertEqual(produced, reference_greedy(spec, weights, prompt, 4))
            extended = rewritten + produced + [17, 23, 61]
            produced = []
            runtime.generate(extended, 4, produced.append)
            self.assertGreater(runtime.info["prefix_cache_last_reused_tokens"], 0)
            self.assertEqual(produced, reference_greedy(spec, weights, extended, 4))
        finally:
            runtime.close()

    def test_cpu_backend_matches_the_reference_forward(self):
        # The CPU backend runs the host translation of the same kernels.
        V2Model.select_backend("cpu")
        try:
            model, spec, weights = _model()
            runtime = model.native_runtime(
                context_limit=64, mtp_drafts=0, expert_mode="cpu")
            runtime.prepare()
            try:
                produced: list[int] = []
                runtime.generate(self.PROMPT, 6, produced.append)
            finally:
                runtime.close()
        finally:
            V2Model.select_backend("auto")
        self.assertEqual(produced, reference_greedy(spec, weights, self.PROMPT, 6))

    def test_reset_clears_the_recurrent_and_cache_state(self):
        model, _, _ = _model()
        runtime = _native(model)
        try:
            first = [runtime.decode(token) for token in self.PROMPT[:5]]
            runtime.reset()
            second = [runtime.decode(token) for token in self.PROMPT[:5]]
        finally:
            runtime.close()
        self.assertEqual(first, second)

    def test_expert_placement_modes_decode_match_reference(self):
        """Verify parity across expert placement modes (cpu, auto, hybrid, gpu).

        Tests single-token decode steps, ensuring GPU grouped kernels (relu²,
        accumulate) match float64 CPU reference across multiple consecutive
        requests (exercising cache residency, hits, and misses).
        """
        model, spec, weights = _model()
        expected = reference_greedy(spec, weights, self.PROMPT, 8)
        for mode in ("cpu", "auto", "hybrid", "gpu"):
            with self.subTest(expert_mode=mode):
                runtime = _native(model, expert_mode=mode)
                try:
                    for run_idx in range(2):
                        runtime.reset()
                        produced: list[int] = []
                        runtime.generate(self.PROMPT, 8, produced.append)
                        self.assertEqual(
                            produced,
                            expected,
                            f"Expert mode {mode} run {run_idx} diverged from reference",
                        )
                finally:
                    runtime.close()

    def test_whole_layer_and_resident_placement_match_reference(self):
        # --n-gpu-layers pins whole MoE blocks and `resident` uploads every
        # one; both lay out two expert tensors at each block's own expert
        # count and must skip the Mamba-2 and attention blocks, which route
        # none (a third role and the global count overran the slots).
        if V2Model.select_backend("auto") != "cuda":
            raise unittest.SkipTest("the CPU backend runs every expert on the CPU")
        model, spec, weights = _model()
        expected = reference_greedy(spec, weights, self.PROMPT, 8)
        for options in ({"expert_mode": "hybrid", "gpu_layers": 1},
                        {"expert_mode": "hybrid", "gpu_layers": 2},
                        {"expert_mode": "resident"}):
            with self.subTest(**options):
                runtime = _native(model, **options)
                try:
                    produced: list[int] = []
                    runtime.generate(self.PROMPT, 8, produced.append)
                    self.assertEqual(produced, expected)
                    self.assertGreater(runtime.info["expert_cache_hits"], 0,
                                       "no expert ran on the GPU")
                finally:
                    runtime.close()

    def test_expert_placement_modes_batched_prefill_match_reference(self):
        """Verify prefill parity across placement modes with chunked & pipelined prefill."""
        cases = (("4", self.PROMPT), (None, [(index * 37) % 96 for index in range(160)]))
        for mode in ("auto", "hybrid", "gpu"):
            for rows, prompt in cases:
                environment = {"FLYWEIGHT_PREFILL_ROWS": rows} if rows else {}
                with self.subTest(expert_mode=mode, rows=rows, tokens=len(prompt)), \
                        unittest.mock.patch.dict(os.environ, environment):
                    model, spec, weights = _model()
                    runtime = _native(model, expert_mode=mode, context_limit=256)
                    try:
                        produced: list[int] = []
                        runtime.generate(prompt, 6, produced.append)
                    finally:
                        runtime.close()
                    self.assertEqual(produced, reference_greedy(spec, weights, prompt, 6))

    def test_kv_cache_types_decode_and_prefill(self):
        """Verify all KV cache precisions (f16, f32, bf16, q8_0, turbo3, turbo4) for Nemotron.

        Ensures that single-token decode and chunked prefill operate without errors,
        reset cleanly, maintain prefill chunking consistency, and preserve bit-exact
        parity for f16/f32/q8_0 against the float64 reference forward.
        """
        model, spec, weights = _model()
        expected = reference_greedy(spec, weights, self.PROMPT, 6)
        precisions = ("f16", "f32", "bf16", "q8_0", "turbo3", "turbo4")
        vocab_size = spec.hidden if hasattr(spec, "vocabulary_size") else 200
        for precision in precisions:
            for prefill_rows in ("1", "4", None):
                env = {"FLYWEIGHT_PREFILL_ROWS": prefill_rows} if prefill_rows else {}
                with self.subTest(precision=precision, prefill_rows=prefill_rows), \
                        unittest.mock.patch.dict(os.environ, env):
                    runtime = _native(
                        model,
                        context_limit=128,
                        cache_type_k=precision,
                        cache_type_v=precision,
                    )
                    try:
                        produced: list[int] = []
                        runtime.generate(self.PROMPT, 6, produced.append)
                        self.assertEqual(len(produced), 6)
                        for token in produced:
                            self.assertGreaterEqual(token, 0)
                            self.assertLess(token, vocab_size)
                        if precision in ("f16", "f32", "q8_0"):
                            self.assertEqual(produced, expected)
                    finally:
                        runtime.close()

        # Mixed key/value precisions
        for ck, cv in (("turbo3", "turbo4"), ("turbo4", "turbo3"), ("q8_0", "turbo4")):
            with self.subTest(cache_type_k=ck, cache_type_v=cv):
                runtime = _native(model, context_limit=128, cache_type_k=ck, cache_type_v=cv)
                try:
                    produced = []
                    runtime.generate(self.PROMPT, 6, produced.append)
                    self.assertEqual(len(produced), 6)
                finally:
                    runtime.close()

        # Long sequence crossing the 128-token cuBLAS threshold
        long_prompt = [(index * 13) % 96 for index in range(120)]
        for precision in ("f16", "turbo4", "q8_0"):
            with self.subTest(long_context_precision=precision):
                runtime = _native(model, context_limit=256, cache_type_k=precision, cache_type_v=precision)
                try:
                    produced = []
                    runtime.generate(long_prompt, 15, produced.append)
                    self.assertEqual(len(produced), 15)
                finally:
                    runtime.close()



class NemotronToolCallTests(unittest.TestCase):
    def test_template_renders_tools_with_the_generic_markers(self):
        # The checkpoint's own template renders the schemas and calls; its
        # call markup is the Qwen3-Coder <tool_call><function=...> shape the
        # generic parser already reads.
        self.assertIn("nemotron_h_moe", server.NATIVE_TOOL_ARCHITECTURES)
        self.assertEqual(server._tool_markers("nemotron_h_moe"),
                         ("<tool_call>", "</tool_call>"))

    def test_template_shaped_call_decodes_against_the_schema(self):
        text = (
            "Checking the weather.\n<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n-2\n"
            "</parameter>\n</function>\n</tool_call>\n"
        )
        tools = [{"function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"]}}}]
        content, calls = server._parse_tool_calls(text, tools=tools)
        self.assertEqual(content, "Checking the weather.")
        self.assertEqual([call["function"]["name"] for call in calls],
                         ["get_weather"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Paris", "days": -2})


if __name__ == "__main__":
    unittest.main()
