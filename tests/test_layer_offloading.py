"""Tests for per-layer GPU offloading (ranges, lists, mixed, and counts)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model, parse_layer_selection
from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf
from tests.qwen4exp_gguf_fixture import Qwen4ExpSpec, build_qwen4exp_gguf


class LayerSelectionParserTests(unittest.TestCase):
    def test_parses_none_and_empty(self) -> None:
        self.assertEqual(parse_layer_selection(None), (0, 0, None))
        self.assertEqual(parse_layer_selection("auto"), (0, 0, None))
        self.assertEqual(parse_layer_selection(""), (0, 0, None))
        self.assertEqual(parse_layer_selection("   "), (0, 0, None))

    def test_parses_all(self) -> None:
        self.assertEqual(parse_layer_selection("all"), (1, 1 << 30, None))

    def test_parses_counts(self) -> None:
        self.assertEqual(parse_layer_selection(0), (1, 0, None))
        self.assertEqual(parse_layer_selection(10), (1, 10, None))
        self.assertEqual(parse_layer_selection("0"), (1, 0, None))
        self.assertEqual(parse_layer_selection("10"), (1, 10, None))

    def test_parses_ranges(self) -> None:
        self.assertEqual(parse_layer_selection("1-5"), (1, 5, b"1,2,3,4,5"))
        self.assertEqual(parse_layer_selection("0-0"), (1, 1, b"0"))
        self.assertEqual(parse_layer_selection("3-3"), (1, 1, b"3"))

    def test_parses_comma_lists(self) -> None:
        self.assertEqual(parse_layer_selection("1,3,8,10"), (1, 4, b"1,3,8,10"))
        self.assertEqual(parse_layer_selection("0,2"), (1, 2, b"0,2"))

    def test_parses_mixed_specifiers(self) -> None:
        self.assertEqual(
            parse_layer_selection("1-5,8,10-12"),
            (1, 9, b"1,2,3,4,5,8,10,11,12"),
        )
        self.assertEqual(
            parse_layer_selection(" 1 - 3 , 6 , 8 - 9 "),
            (1, 6, b"1,2,3,6,8,9"),
        )

    def test_parses_sequences_and_sets(self) -> None:
        self.assertEqual(parse_layer_selection([1, 3, 5]), (1, 3, b"1,3,5"))
        self.assertEqual(parse_layer_selection({2, 4, 6}), (1, 3, b"2,4,6"))
        self.assertEqual(parse_layer_selection(range(2, 6)), (1, 4, b"2,3,4,5"))
        self.assertEqual(parse_layer_selection([]), (1, 0, b""))

    def test_rejects_invalid_inputs(self) -> None:
        for invalid in (
            -1,
            "-1",
            "10-1",
            "1,,2",
            "abc",
            "-5",
            "5-",
            "1-2-3",
            "1.5",
            [1, -2],
            "0-",
            ",1,2",
            "1,2,",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_layer_selection(invalid)


class LayerOffloadingRuntimeTests(unittest.TestCase):
    _workspaces: list[tempfile.TemporaryDirectory] = []
    _models: list[V2Model] = []

    @classmethod
    def tearDownClass(cls) -> None:
        for model in cls._models:
            model.close()
        cls._models.clear()
        for holder in cls._workspaces:
            holder.cleanup()
        cls._workspaces.clear()

    def _dense_model(self, layers: int = 8) -> tuple[V2Model, Path]:
        holder = tempfile.TemporaryDirectory(prefix="flyweight-test-layer-dense-")
        self._workspaces.append(holder)
        path = Path(holder.name) / "dense.gguf"
        build_dense_qwen35_gguf(path, DenseQwenSpec(layers=layers))
        model = V2Model(path)
        self._models.append(model)
        return model, path

    def _moe_model(self, layers: int = 8) -> tuple[V2Model, Path]:
        holder = tempfile.TemporaryDirectory(prefix="flyweight-test-layer-moe-")
        self._workspaces.append(holder)
        path = Path(holder.name) / "moe.gguf"
        build_qwen4exp_gguf(
            path,
            Qwen4ExpSpec(
                layers=layers,
                experts=8,
                expert_intermediate=32,
                hidden=64,
                heads=4,
                kv_heads=2,
            ),
        )
        model = V2Model(path)
        self._models.append(model)
        return model, path

    def _collect_tokens(self, runtime, prompt: list[int], max_tokens: int = 4) -> list[int]:
        task_id = runtime.task_submit(prompt, max_tokens)
        tokens: list[int] = []
        for _ in range(64):
            for event_task, token, kind in runtime.engine_step():
                if event_task != task_id:
                    continue
                if kind == 0:
                    tokens.append(token)
                elif kind == 1:
                    return tokens
                elif kind == 2:
                    raise AssertionError("runtime task failed")
        raise AssertionError("runtime task did not complete")

    def test_out_of_bounds_selection_raises_value_error(self) -> None:
        model, _ = self._dense_model(layers=8)
        for out_of_bounds in ("0,8", "5-8", [0, 8], {8}, [8], "8-10"):
            with self.subTest(out_of_bounds=out_of_bounds):
                with self.assertRaises(ValueError) as ctx:
                    model.native_runtime(gpu_layers=out_of_bounds)
                self.assertIn("exceeds model layer count", str(ctx.exception).lower())

    def test_dense_specific_layer_selection(self) -> None:
        if not V2Model.gpu_info()["available"]:
            raise unittest.SkipTest("native CUDA runtime unavailable")
        model, _ = self._dense_model(layers=8)
        runtime = model.native_runtime(
            gpu_layers="1,3,5",
            context_limit=64,
            mtp_drafts=0,
        )
        runtime.prepare()
        try:
            self.assertEqual(runtime.gpu_layers, [1, 3, 5])
            self.assertTrue(runtime.is_layer_on_gpu(1))
            self.assertTrue(runtime.is_layer_on_gpu(3))
            self.assertTrue(runtime.is_layer_on_gpu(5))
            self.assertFalse(runtime.is_layer_on_gpu(0))
            self.assertFalse(runtime.is_layer_on_gpu(2))
            self.assertFalse(runtime.is_layer_on_gpu(4))
            self.assertFalse(runtime.is_layer_on_gpu(6))
            self.assertFalse(runtime.is_layer_on_gpu(7))
            with self.assertRaises(IndexError):
                runtime.is_layer_on_gpu(-1)
            with self.assertRaises(IndexError):
                runtime.is_layer_on_gpu(8)
            self.assertEqual(runtime.info["host_ffn_layers"], 5)
        finally:
            runtime.close()

    def test_dense_range_and_mixed_layer_selection(self) -> None:
        if not V2Model.gpu_info()["available"]:
            raise unittest.SkipTest("native CUDA runtime unavailable")
        model, _ = self._dense_model(layers=8)

        # Range test
        runtime_range = model.native_runtime(
            gpu_layers="2-5",
            context_limit=64,
            mtp_drafts=0,
        )
        runtime_range.prepare()
        try:
            self.assertEqual(runtime_range.gpu_layers, [2, 3, 4, 5])
            self.assertEqual(runtime_range.info["host_ffn_layers"], 4)
            for layer in range(8):
                self.assertEqual(runtime_range.is_layer_on_gpu(layer), 2 <= layer <= 5)
        finally:
            runtime_range.close()

        # Mixed test
        runtime_mixed = model.native_runtime(
            gpu_layers="0,2-4,7",
            context_limit=64,
            mtp_drafts=0,
        )
        runtime_mixed.prepare()
        try:
            self.assertEqual(runtime_mixed.gpu_layers, [0, 2, 3, 4, 7])
            self.assertEqual(runtime_mixed.info["host_ffn_layers"], 3)
            for layer in range(8):
                self.assertEqual(
                    runtime_mixed.is_layer_on_gpu(layer),
                    layer in {0, 2, 3, 4, 7},
                )
        finally:
            runtime_mixed.close()

    def test_dense_counts_and_collections(self) -> None:
        if not V2Model.gpu_info()["available"]:
            raise unittest.SkipTest("native CUDA runtime unavailable")
        model, _ = self._dense_model(layers=8)

        # Count = 0 -> all host
        r0 = model.native_runtime(gpu_layers=0, context_limit=64, mtp_drafts=0)
        r0.prepare()
        try:
            self.assertEqual(r0.gpu_layers, [])
            self.assertEqual(r0.info["host_ffn_layers"], 8)
        finally:
            r0.close()

        # Count = 4 -> layers 0..3
        r4 = model.native_runtime(gpu_layers=4, context_limit=64, mtp_drafts=0)
        r4.prepare()
        try:
            self.assertEqual(r4.gpu_layers, [0, 1, 2, 3])
            self.assertEqual(r4.info["host_ffn_layers"], 4)
        finally:
            r4.close()

        # List of layers
        r_list = model.native_runtime(gpu_layers=[0, 3, 6], context_limit=64, mtp_drafts=0)
        r_list.prepare()
        try:
            self.assertEqual(r_list.gpu_layers, [0, 3, 6])
            self.assertEqual(r_list.info["host_ffn_layers"], 5)
        finally:
            r_list.close()

    def test_dense_decode_parity_across_offloading_patterns(self) -> None:
        if not V2Model.gpu_info()["available"]:
            raise unittest.SkipTest("native CUDA runtime unavailable")
        model, _ = self._dense_model(layers=8)
        prompt = [5, 9, 3, 7]

        # Turn off lossy CPU FFN quantization so CPU-spilled feed-forward is
        # bitwise-matched with GPU resident feed-forward.
        prev_ffn_format = os.environ.get("FLYWEIGHT_HOST_FFN_FORMAT")
        os.environ["FLYWEIGHT_HOST_FFN_FORMAT"] = "off"
        try:
            # Baseline: all resident on GPU
            base_rt = model.native_runtime(gpu_layers=8, context_limit=64, mtp_drafts=0)
            base_rt.prepare()
            try:
                expected_tokens = self._collect_tokens(base_rt, prompt, 4)
            finally:
                base_rt.close()

            self.assertEqual(len(expected_tokens), 4)

            # Compare against various layer offload configurations
            test_configs = [
                0,  # all spilled to CPU
                "1-4",  # contiguous subset
                "0,2,4,6",  # alternating layers
                "1-3,6",  # mixed range + single
                [0, 7],  # boundary layers
            ]

            for config in test_configs:
                with self.subTest(config=config):
                    rt = model.native_runtime(gpu_layers=config, context_limit=64, mtp_drafts=0)
                    rt.prepare()
                    try:
                        tokens = self._collect_tokens(rt, prompt, 4)
                        self.assertEqual(tokens, expected_tokens)
                    finally:
                        rt.close()
        finally:
            if prev_ffn_format is not None:
                os.environ["FLYWEIGHT_HOST_FFN_FORMAT"] = prev_ffn_format
            else:
                os.environ.pop("FLYWEIGHT_HOST_FFN_FORMAT", None)

    def test_moe_layer_selection(self) -> None:
        if not V2Model.gpu_info()["available"]:
            raise unittest.SkipTest("native CUDA runtime unavailable")
        model, _ = self._moe_model(layers=8)

        # Mixed selection on MoE
        runtime = model.native_runtime(
            gpu_layers="1-2,5",
            context_limit=64,
            mtp_drafts=0,
        )
        runtime.prepare()
        try:
            self.assertEqual(runtime.gpu_layers, [1, 2, 5])
            self.assertTrue(runtime.is_layer_on_gpu(1))
            self.assertTrue(runtime.is_layer_on_gpu(2))
            self.assertTrue(runtime.is_layer_on_gpu(5))
            self.assertFalse(runtime.is_layer_on_gpu(0))
            self.assertFalse(runtime.is_layer_on_gpu(3))

            # Verify generation runs
            tokens = self._collect_tokens(runtime, [3, 7, 2], 3)
            self.assertEqual(len(tokens), 3)
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
