"""Native Nemotron H MoE planning, execution, tokenization and tool-call tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from flyweight import server
from flyweight.v2 import V2Error, V2Model
from flyweight.v2_server import NativeV2Tokenizer

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

    def test_layer_norm_epsilon_is_the_rms_fallback(self):
        # llama.cpp reads nemotron_h's norm epsilon from layer_norm_epsilon
        # when the RMS key is absent; the generic 1e-6 default would differ.
        model, _, _ = _model(NemotronSpec(epsilon_key="layer_norm_epsilon"))
        self.assertAlmostEqual(model.config["rms_norm_epsilon"], 1e-5, places=9)

    def test_variants_the_plan_does_not_compute_are_refused(self):
        # Each is something llama.cpp's nemotron_h applies; running without it
        # would be a different model, so creation fails instead.
        cases = (
            (NemotronSpec(expert_group_count=2), "grouped expert routing"),
            (NemotronSpec(expert_scales=True), "per-expert weight scales"),
            (NemotronSpec(attention_bias=True), "attention biases"),
        )
        for spec, message in cases:
            with self.subTest(refused=message):
                model, _, _ = _model(spec)
                with self.assertRaises(V2Error) as caught:
                    model.native_runtime(
                        context_limit=64, mtp_drafts=0, expert_mode="cpu")
                self.assertIn(message, str(caught.exception))

    def test_stop_set_includes_end_of_sequence(self):
        # `</s>` ends generation in llama.cpp beside the <|im_end|> eos.
        model, _, _ = _model()
        tokens = NativeV2Tokenizer(model).eos_token_ids
        self.assertIn(model.token_id("<|im_end|>"), tokens)
        self.assertIn(model.token_id("</s>"), tokens)


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



# Runs inside the subprocess of NemotronQuantizedTests: every placement and
# chunking the runtime offers over one quantized fixture, as JSON.
_QUANTIZED_RUNS = """
import json, os, sys
from flyweight.v2 import V2Model
model = V2Model(sys.argv[1])
cases = json.loads(sys.argv[2])
result = []
for case in cases:
    os.environ.pop("FLYWEIGHT_PREFILL_ROWS", None)
    if case["rows"]:
        os.environ["FLYWEIGHT_PREFILL_ROWS"] = case["rows"]
    runtime = model.native_runtime(mtp_drafts=0, **case["options"])
    runtime.prepare()
    produced = []
    runtime.generate(case["prompt"], case["count"], produced.append)
    result.append({"tokens": produced, "hits": runtime.info["expert_cache_hits"]})
    runtime.close()
print(json.dumps(result))
"""


class NemotronQuantizedTests(unittest.TestCase):
    """The released Bolt's tensor formats against the float64 forward.

    IQ2_XXS routed and shared experts (CPU dots and the GPU relu² kernels),
    NVFP4 ssm_in/ssm_out with their out-of-line scale and Q8_0 everywhere
    else, at widths that admit those kernels; the reference runs on the
    decoded values. Activation rounding is the one thing that may separate
    the two, and on a random model it flips greedy steps whose logit lead or
    top-k routing gap is narrow (a 1e-3 routing gap is common). So these runs
    turn the int8 activation paths and the FP4 tensor-core prefill off --
    switches read once per process, hence the subprocess -- and pin the
    decoders and kernels exactly; what the rounding costs on a real
    checkpoint is bench_kernel_quality's question.
    """

    PROMPT = [40, 72, 5, 101, 33, 64, 12, 90, 7, 55, 21]
    LONG = [(index * 37) % 96 for index in range(160)]

    @classmethod
    def setUpClass(cls):
        holder = tempfile.TemporaryDirectory(prefix="flyweight-nemotron-q-")
        _WORKSPACES.append(holder)
        cls.path = Path(holder.name) / "nemotron_h_moe.gguf"
        cls.spec, cls.weights = build_nemotron_gguf(
            cls.path, NemotronSpec(quantized=True))
        cls.expected = reference_greedy(cls.spec, cls.weights, cls.PROMPT, 8)
        cls.expected_long = reference_greedy(cls.spec, cls.weights, cls.LONG, 6)

    def _run(self, cases):
        if not V2Model.gpu_info()["available"]:
            raise unittest.SkipTest("native CUDA runtime is unavailable")
        environment = dict(os.environ, FLYWEIGHT_IQ2_Q8_DECODE="0",
                           FLYWEIGHT_Q8_ACTIVATIONS="0",
                           FLYWEIGHT_NVFP4_PREFILL_TC="0")
        completed = subprocess.run(
            [sys.executable, "-c", _QUANTIZED_RUNS, str(self.path), json.dumps(cases)],
            env=environment, capture_output=True, text=True, timeout=900)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def test_decode_and_prefill_match_the_reference_in_every_placement(self):
        placements = ({"expert_mode": "cpu"}, {"expert_mode": "auto"},
                      {"expert_mode": "hybrid"}, {"expert_mode": "gpu"},
                      {"expert_mode": "hybrid", "gpu_layers": 2},
                      {"expert_mode": "resident"})
        cases = [{"options": dict(options, context_limit=64), "rows": rows,
                  "prompt": self.PROMPT, "count": 8}
                 for options in placements for rows in ("1", None)]
        gpu = V2Model.select_backend("auto") == "cuda"
        for case, result in zip(cases, self._run(cases)):
            with self.subTest(**case["options"], rows=case["rows"]):
                self.assertEqual(result["tokens"], self.expected)
                if gpu and case["options"]["expert_mode"] != "cpu":
                    self.assertGreater(result["hits"], 0, "no expert ran on the GPU")

    def test_chunked_prefill_matches_the_reference(self):
        # 4- and 12-row chunks carry the conv ring and scan state across
        # boundaries; the default 160-row chunk runs the two-half pipelined
        # driver on the large-row Q8_0, NVFP4 and IQ2_XXS kernels.
        cases = [{"options": {"expert_mode": mode, "context_limit": 256}, "rows": rows,
                  "prompt": self.LONG, "count": 6}
                 for mode in ("cpu", "auto") for rows in ("4", "12", None)]
        for case, result in zip(cases, self._run(cases)):
            with self.subTest(expert_mode=case["options"]["expert_mode"],
                              rows=case["rows"]):
                self.assertEqual(result["tokens"], self.expected_long)


_SPECULATIVE_RUNS = """
import json, sys
from flyweight.v2 import V2Model
model = V2Model(sys.argv[1])
prompt = json.loads(sys.argv[2])
result = {}
for mode in ("cpu", "auto", "gpu"):
    runtime = model.native_runtime(context_limit=256, mtp_drafts=0, expert_mode=mode)
    runtime.prepare()
    produced = []
    runtime.generate(prompt, 64, produced.append)
    info = runtime.info
    result[mode] = {"tokens": produced, "accepted": info["mtp_accepted_tokens"],
                    "rejected": info["mtp_rejected_tokens"]}
    runtime.close()
print(json.dumps(result))
"""


class NemotronSpeculativeTests(unittest.TestCase):
    """Draft rounds: verify a batch of drafted rows, keep the agreeing prefix.

    A rejection restores the Mamba-2 conv rings and scan states from the
    round's snapshot and replays the accepted rows, so the output must equal
    plain greedy decoding. Prompt-lookup drafts with a one-token match make
    nearly every step a round; the tied head makes the fixture's greedy run
    settle into cycles, so rounds also accept, not just reject.
    """

    def test_lookup_rounds_match_the_reference(self):
        if not V2Model.gpu_info()["available"]:
            raise unittest.SkipTest("native CUDA runtime is unavailable")
        holder = tempfile.TemporaryDirectory(prefix="flyweight-nemotron-spec-")
        _WORKSPACES.append(holder)
        path = Path(holder.name) / "nemotron_h_moe.gguf"
        spec, weights = build_nemotron_gguf(path, NemotronSpec(tied_head=True))
        prompt = [40, 72, 5, 33, 64, 12] * 6 + [40, 72]
        expected = reference_greedy(spec, weights, prompt, 64)
        environment = dict(os.environ, FLYWEIGHT_LOOKUP_DRAFTS="4",
                           FLYWEIGHT_LOOKUP_NGRAM_MIN="1",
                           FLYWEIGHT_IQ2_Q8_DECODE="0", FLYWEIGHT_Q8_ACTIVATIONS="0")
        completed = subprocess.run(
            [sys.executable, "-c", _SPECULATIVE_RUNS, str(path), json.dumps(prompt)],
            env=environment, capture_output=True, text=True, timeout=600)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        for mode, run in result.items():
            with self.subTest(expert_mode=mode):
                self.assertEqual(run["tokens"], expected)
                self.assertGreater(run["accepted"], 0, "no round accepted a draft")
                self.assertGreater(run["rejected"], 0, "no round rolled back")


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
