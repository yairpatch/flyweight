"""Native LFM2-MoE planning, execution, tokenization and tool-call tests."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from flyweight import server
from flyweight.v2 import V2Model

from tests.lfm2_gguf_fixture import Lfm2Spec, build_lfm2_gguf, reference_greedy


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


def _model(spec: Lfm2Spec | None = None):
    holder = tempfile.TemporaryDirectory(prefix="flyweight-lfm2-")
    _WORKSPACES.append(holder)
    path = Path(holder.name) / "lfm2moe.gguf"
    spec, weights = build_lfm2_gguf(path, spec)
    model = V2Model(path)
    _MODELS.append(model)
    return model, spec, weights


def _native(model: V2Model, **options):
    if not V2Model.gpu_info()["available"]:
        raise unittest.SkipTest("native CUDA runtime is unavailable")
    runtime = model.native_runtime(
        context_limit=64, mtp_drafts=0, expert_mode="cpu", **options
    )
    runtime.prepare()
    return runtime


class Lfm2ConfigTests(unittest.TestCase):
    def test_kv_head_scalar_is_lifted_from_the_per_layer_array(self):
        # The array is zero on the conv blocks, and the attention geometry
        # reads the scalar, which the file leaves unset.
        model, spec, _ = _model()
        config = model.config
        self.assertEqual(config["architecture"], "lfm2moe")
        self.assertEqual(config["attention_kv_heads"], spec.kv_heads)
        self.assertEqual(config["layer_count"], spec.layers)

    def test_missing_l_cache_is_rejected(self):
        holder = tempfile.TemporaryDirectory(prefix="flyweight-lfm2-bad-")
        _WORKSPACES.append(holder)
        path = Path(holder.name) / "lfm2moe.gguf"
        build_lfm2_gguf(path)
        raw = path.read_bytes().replace(b"lfm2moe.shortconv.l_cache", b"lfm2moe.shortconv.l_cachX")
        path.write_bytes(raw)
        with self.assertRaises(Exception) as caught:
            V2Model(path).close()
        self.assertIn("l_cache", str(caught.exception))


class Lfm2TokenizerTests(unittest.TestCase):
    def test_user_defined_think_tags_are_split_whole(self):
        model, _, _ = _model()
        think, close = model.token_id("<think>"), model.token_id("</think>")
        self.assertEqual(model.tokenize("<think>"), [think])
        tokens = model.tokenize("a<think>b</think>c")
        self.assertEqual(
            tokens,
            [model.token_id("a"), think, model.token_id("b"), close, model.token_id("c")],
        )

    def test_control_tokens_map_to_their_reserved_ids(self):
        model, _, _ = _model()
        start, end = model.token_id("<|im_start|>"), model.token_id("<|im_end|>")
        self.assertEqual(model.tokenize("<|im_start|>x<|im_end|>"),
                         [start, model.token_id("x"), end])

    def test_digits_group_in_threes_and_whole_pieces_skip_bpe(self):
        # llama3's \p{N}{1,3}, and ignore_merges: with no merge table at all,
        # a three-digit piece that is itself a vocabulary entry still comes out
        # as that one token.
        model, _, _ = _model()
        self.assertEqual(model.tokenize("1234"),
                         [model.token_id("123"), model.token_id("4")])


class Lfm2NativeTests(unittest.TestCase):
    PROMPT = [40, 72, 5, 101, 33, 64, 12, 90, 7, 55, 21]

    def test_greedy_decode_matches_the_reference_forward(self):
        """The load-bearing check: an independently written float64 forward.

        Covers the conv ring (three taps across tokens), the NEOX rope and QK
        norm on the GQA blocks, the dense lead block, sigmoid routing whose
        bias picks but does not weight, the missing shared expert, and the
        tied head behind token_embd_norm.
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

    def test_dense_lfm2_matches_the_reference_forward(self):
        # Dense LFM2 rides the same plan with every block's feed-forward dense
        # and no router at all.
        model, spec, weights = _model(Lfm2Spec(architecture="lfm2"))
        self.assertEqual(model.config["expert_count"], 0)
        runtime = _native(model)
        try:
            produced: list[int] = []
            runtime.generate(self.PROMPT, 8, produced.append)
        finally:
            runtime.close()
        self.assertEqual(produced, reference_greedy(spec, weights, self.PROMPT, 8))

    def test_batched_prefill_matches_token_by_token(self):
        # lfm2_conv_rows and the rows attention path against the decode path,
        # with a chunk boundary inside the prompt so the conv ring carries.
        results = {}
        for rows in ("1", "4"):
            with unittest.mock.patch.dict(os.environ, {"FLYWEIGHT_PREFILL_ROWS": rows}):
                model, _, _ = _model()
                runtime = _native(model)
                try:
                    produced: list[int] = []
                    runtime.generate(self.PROMPT, 6, produced.append)
                    results[rows] = produced
                finally:
                    runtime.close()
        self.assertEqual(results["1"], results["4"])

    def test_reset_clears_the_conv_ring(self):
        model, _, _ = _model()
        runtime = _native(model)
        try:
            first = [runtime.decode(token) for token in self.PROMPT[:5]]
            runtime.reset()
            second = [runtime.decode(token) for token in self.PROMPT[:5]]
        finally:
            runtime.close()
        self.assertEqual(first, second)


class Lfm2ToolCallTests(unittest.TestCase):
    def test_markers_are_the_lfm2_special_tokens(self):
        self.assertEqual(
            server._tool_markers("lfm2moe"),
            ("<|tool_call_start|>", "<|tool_call_end|>"),
        )
        self.assertIn("lfm2moe", server.NATIVE_TOOL_ARCHITECTURES)

    def test_pythonic_calls_decode_with_json_literals(self):
        text = (
            "Checking.<|tool_call_start|>[get_weather(city='Paris', days=-2, "
            'opts={"metric": true, "note": null}), ping()]<|tool_call_end|>'
        )
        content, calls = server._parse_tool_calls(text)
        self.assertEqual(content, "Checking.")
        self.assertEqual([call["function"]["name"] for call in calls], ["get_weather", "ping"])
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"city": "Paris", "days": -2, "opts": {"metric": True, "note": None}},
        )
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {})

    def test_template_string_escapes_round_trip(self):
        # The template writes strings as single-quoted literals with \\, \',
        # \n and \r escaped; the parser has to undo exactly that.
        original = "line one\nit's \\ done"
        escaped = (original.replace("\\", "\\\\").replace("'", "\\'")
                   .replace("\n", "\\n"))
        _, calls = server._parse_tool_calls(
            f"<|tool_call_start|>[write(text='{escaped}')]<|tool_call_end|>"
        )
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"text": original})

    def test_json_body_is_accepted(self):
        _, calls = server._parse_tool_calls(
            '<|tool_call_start|>[{"name": "f", "arguments": {"x": 1}}]<|tool_call_end|>'
        )
        self.assertEqual(calls[0]["function"]["name"], "f")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"x": 1})

    def test_unparseable_body_yields_no_call_and_no_leaked_markup(self):
        content, calls = server._parse_tool_calls(
            "Sure.<|tool_call_start|>[get_weather(city=<|tool_call_end|>"
        )
        self.assertEqual(calls, [])
        self.assertEqual(content, "Sure.")


if __name__ == "__main__":
    unittest.main()
