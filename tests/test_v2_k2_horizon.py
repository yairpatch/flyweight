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


DEFAULT_REAL_MODEL = "/home/yair/Downloads/gguf/models--IFM--K2-Horizon-7B-GGUF.gguf"


@unittest.skipUnless(
    os.path.isfile(os.environ.get("FLYWEIGHT_TEST_K2_HORIZON_MODEL", DEFAULT_REAL_MODEL)),
    "K2-Horizon GGUF checkpoint not available",
)
class K2HorizonRealModelTests(unittest.TestCase):
    def test_real_model_completion(self):
        path = os.environ.get("FLYWEIGHT_TEST_K2_HORIZON_MODEL", DEFAULT_REAL_MODEL)
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


if __name__ == "__main__":
    unittest.main()
