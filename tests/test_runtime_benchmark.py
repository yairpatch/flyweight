from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colibri_next.runtime_benchmark import (
    _parser,
    _runtime_options,
    _expand_tokens,
    _summary,
    compare_baselines,
    measure_runtime_sample,
)
from colibri_next.v2 import V2Model


class _FakeRuntime:
    def __init__(self) -> None:
        self._info = {
            "prefill_calls": 0,
            "prefill_tokens": 0,
            "prefill_nanoseconds": 0,
            "decode_calls": 0,
            "decode_nanoseconds": 0,
        }

    @property
    def info(self):
        return dict(self._info)

    def generate(self, prompt, max_tokens, callback):
        self._info["prefill_calls"] += 1
        self._info["prefill_tokens"] += len(prompt) - 1
        self._info["prefill_nanoseconds"] += 200_000_000
        callback(42)

    def decode(self, token):
        self._info["decode_calls"] += 1
        self._info["decode_nanoseconds"] += 100_000_000
        return token + 1


def _sample(
    *,
    first_token_latency: float = 1.0,
    native_prefill: float = 0.8,
    decode_median: float = 0.1,
    decode_throughput: float = 10.0,
    tokens: list[int] | None = None,
) -> dict:
    return {
        "first_token_latency_seconds": first_token_latency,
        "native_prefill_seconds": native_prefill,
        "prompt_tokens_per_second": 100.0,
        "decode_median_seconds": decode_median,
        "decode_p95_seconds": decode_median,
        "decode_tokens_per_second": decode_throughput,
        "generated_tokens": tokens or [7, 8, 9],
    }


class RuntimeMeasurementTests(unittest.TestCase):
    def test_hybrid_prefill_option_defaults_to_split_and_accepts_cpu(self):
        defaults = _parser().parse_args(["run", "model.gguf"])
        self.assertEqual(_runtime_options(defaults)["expert_mode"], "auto")
        self.assertEqual(_runtime_options(defaults)["hybrid_prefill"], "split")
        self.assertEqual(_runtime_options(defaults)["expert_residency"], "immutable")
        selected = _parser().parse_args(
            [
                "run", "model.gguf",
                "--hybrid-prefill", "cpu",
                "--expert-residency", "immutable",
            ]
        )
        self.assertEqual(_runtime_options(selected)["hybrid_prefill"], "cpu")
        self.assertEqual(
            _runtime_options(selected)["expert_residency"], "immutable"
        )

    def test_prefill_counter_time_is_separate_from_final_token_boundary(self):
        runtime = _FakeRuntime()
        times = iter((10.0, 10.5, 11.0, 11.1, 12.0, 12.2, 13.0, 13.4))
        result = measure_runtime_sample(
            runtime,
            [1, 2, 3],
            warmup_decode=1,
            decode_iterations=2,
            clock=lambda: next(times),
        )
        self.assertAlmostEqual(result["first_token_latency_seconds"], 0.5)
        self.assertAlmostEqual(result["native_prefill_seconds"], 0.2)
        self.assertAlmostEqual(result["final_token_and_boundary_seconds"], 0.3)
        self.assertEqual(result["prefill_counters"]["prefill_tokens"], 2)
        self.assertEqual(result["decode_counters"]["decode_calls"], 2)
        self.assertEqual(len(result["decode_seconds"]), 2)
        self.assertAlmostEqual(result["decode_seconds"][0], 0.2)
        self.assertAlmostEqual(result["decode_seconds"][1], 0.4)
        self.assertAlmostEqual(result["decode_median_seconds"], 0.3)
        self.assertAlmostEqual(result["decode_p95_seconds"], 0.4)
        self.assertEqual(result["generated_tokens"], [42, 43, 44, 45])

    def test_prompt_expansion_hits_exact_boundaries(self):
        for length in (1, 2, 63, 64, 65, 128, 129):
            with self.subTest(length=length):
                expanded = _expand_tokens([1, 2, 3], length)
                self.assertEqual(len(expanded), length)
                self.assertEqual(expanded[: min(3, length)], [1, 2, 3][:length])

    def test_summary_records_median_worst_and_determinism(self):
        summary = _summary(
            [
                _sample(first_token_latency=1.0),
                _sample(first_token_latency=1.2),
                _sample(first_token_latency=0.8),
            ]
        )
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(
            summary["metrics"]["first_token_latency_seconds"]["median"], 1.0
        )
        self.assertEqual(
            summary["metrics"]["first_token_latency_seconds"]["worst"], 1.2
        )
        self.assertTrue(summary["deterministic_tokens"])


class RuntimeComparisonTests(unittest.TestCase):
    def _write(self, path: Path, samples: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for index, sample in enumerate(samples):
                stream.write(
                    json.dumps(
                        {
                            "kind": "sample",
                            "case": "prompt-64",
                            "sample": index,
                            "excluded_warmup": False,
                            **sample,
                        }
                    )
                    + "\n"
                )

    def test_comparison_passes_matching_tokens_within_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            candidate = Path(directory) / "candidate.jsonl"
            self._write(baseline, [_sample(), _sample()])
            self._write(
                candidate,
                [
                    _sample(
                        first_token_latency=1.02,
                        native_prefill=0.81,
                        decode_median=0.102,
                        decode_throughput=9.9,
                    )
                ],
            )
            result = compare_baselines(
                baseline,
                candidate,
                max_prefill_regression_pct=3.0,
                max_decode_regression_pct=3.0,
            )
        self.assertTrue(result["passed"])
        self.assertTrue(result["cases"]["prompt-64"]["token_match"])

    def test_comparison_fails_regression_and_token_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            candidate = Path(directory) / "candidate.jsonl"
            self._write(baseline, [_sample()])
            self._write(
                candidate,
                [
                    _sample(
                        first_token_latency=1.2,
                        decode_median=0.2,
                        decode_throughput=5.0,
                        tokens=[99],
                    )
                ],
            )
            result = compare_baselines(
                baseline,
                candidate,
                max_prefill_regression_pct=3.0,
                max_decode_regression_pct=3.0,
            )
        self.assertFalse(result["passed"])
        self.assertFalse(result["cases"]["prompt-64"]["token_match"])
        self.assertTrue(
            any(
                not check["passed"]
                for check in result["cases"]["prompt-64"]["checks"]
            )
        )

    def test_comparison_ignores_excluded_warmup_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            candidate = Path(directory) / "candidate.jsonl"
            warmup = {
                "kind": "sample",
                "case": "prompt-64",
                "sample": 0,
                "excluded_warmup": True,
                **_sample(tokens=[999]),
            }
            measured = {
                "kind": "sample",
                "case": "prompt-64",
                "sample": 0,
                "excluded_warmup": False,
                **_sample(),
            }
            content = json.dumps(warmup) + "\n" + json.dumps(measured) + "\n"
            baseline.write_text(content, encoding="utf-8")
            candidate.write_text(content, encoding="utf-8")
            result = compare_baselines(
                baseline,
                candidate,
                max_prefill_regression_pct=0.0,
                max_decode_regression_pct=0.0,
            )
        self.assertTrue(result["passed"])


class NativePromptBoundaryTests(unittest.TestCase):
    def test_dense_prefill_boundaries_match_single_token_path(self):
        if not V2Model.gpu_info()["available"]:
            raise unittest.SkipTest("native CUDA runtime is unavailable")
        from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf

        directory = Path(tempfile.mkdtemp(prefix="colibri-bench-boundary-"))
        path = directory / "dense.gguf"
        build_dense_qwen35_gguf(
            path,
            DenseQwenSpec(
                hidden=32,
                layers=2,
                intermediate=64,
                vocabulary=16,
                heads=1,
                kv_heads=1,
                head_dim=32,
                value_heads=1,
                ssm_head_dim=32,
                key_heads=1,
                attention_every=2,
            ),
        )
        model = V2Model(path)
        with patch.dict("os.environ", {"COLIBRI_PREFILL_ROWS": "1"}):
            serial = model.native_runtime(context_limit=256)
            serial.prepare()
        with patch.dict("os.environ", {"COLIBRI_PREFILL_ROWS": "64"}):
            chunked = model.native_runtime(context_limit=256)
            chunked.prepare()
        try:
            for length in (1, 2, 63, 64, 65, 66, 67, 128, 129):
                prompt = _expand_tokens([1, 2, 3, 4], length)
                serial_tokens: list[int] = []
                chunked_tokens: list[int] = []
                serial.reset()
                chunked.reset()
                serial.generate(prompt, 1, serial_tokens.append)
                chunked.generate(prompt, 1, chunked_tokens.append)
                with self.subTest(length=length):
                    self.assertEqual(chunked_tokens, serial_tokens)
                    self.assertEqual(chunked.info["position"], length)
        finally:
            serial.close()
            chunked.close()
            model.close()


if __name__ == "__main__":
    unittest.main()
