"""Native dense-model planning and offload tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from colibri_next.v2 import V2Model

from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf


SPILL_BUDGET_BYTES = 32 * 1024 * 1024


_WORKSPACES: list[tempfile.TemporaryDirectory] = []


def tearDownModule():
    # /tmp is often a RAM-backed tmpfs and these fixtures are megabytes each,
    # so a directory per test adds up fast across repeated runs.
    for holder in _WORKSPACES:
        holder.cleanup()
    _WORKSPACES.clear()


def _model(**kwargs) -> tuple[V2Model, DenseQwenSpec, Path]:
    holder = tempfile.TemporaryDirectory(prefix="colibri-dense-")
    _WORKSPACES.append(holder)
    directory = Path(holder.name)
    path = directory / "dense.gguf"
    spec = build_dense_qwen35_gguf(path, **kwargs)
    return V2Model(path), spec, path


def _native(model: V2Model, **options):
    if not V2Model.gpu_info()["available"]:
        raise unittest.SkipTest("native CUDA runtime is unavailable")
    runtime = model.native_runtime(context_limit=64, mtp_drafts=0, **options)
    runtime.prepare()
    return runtime


class DensePlanTests(unittest.TestCase):
    def test_config_reports_a_dense_model_with_no_experts(self):
        model, spec, _ = _model()
        try:
            config = model.config
            self.assertEqual(config["architecture"], "qwen35")
            self.assertEqual(config["layer_count"], spec.layers)
            self.assertEqual(config["hidden_size"], spec.hidden)
            self.assertEqual(config["intermediate_size"], spec.intermediate)
            self.assertEqual(config["expert_count"], 0)
            self.assertEqual(config["expert_used_count"], 0)
        finally:
            model.close()

    def test_tied_embedding_table_is_used_as_the_lm_head(self):
        model, _, _ = _model(tied_lm_head=True)
        try:
            self.assertEqual(model.qwen_tensor("lm_head")["name"], "token_embd.weight")
            with model.native_runtime(context_limit=32, mtp_drafts=0) as runtime:
                self.assertGreater(runtime.info["static_tensor_bytes"], 0)
        finally:
            model.close()

    def test_mtp_without_dedicated_head_norm_uses_output_norm(self):
        model, spec, _ = _model(mtp=True, mtp_head_norm=False)
        try:
            with model.native_runtime(context_limit=32, mtp_drafts=0) as runtime:
                self.assertTrue(runtime.info["mtp_available"])
                self.assertEqual(runtime.info["mtp_layer"], spec.layers)
        finally:
            model.close()


class DenseNativeTests(unittest.TestCase):
    def test_host_spilled_feed_forward_matches_the_resident_one(self):
        model, _, _ = _model()
        resident = _native(model)
        spilled = _native(model, gpu_cache_bytes=SPILL_BUDGET_BYTES)
        try:
            self.assertEqual(resident.info["host_ffn_layers"], 0)
            self.assertGreater(spilled.info["host_ffn_layers"], 0)
            self.assertEqual(spilled.decode(5), resident.decode(5))
        finally:
            resident.close()
            spilled.close()
            model.close()

    def test_spilling_shrinks_the_resident_footprint(self):
        model, spec, _ = _model()
        resident = _native(model)
        spilled = _native(model, gpu_cache_bytes=SPILL_BUDGET_BYTES)
        try:
            self.assertGreater(spilled.info["host_ffn_layers"], 0)
            self.assertLessEqual(spilled.info["host_ffn_layers"], spec.layers)
            self.assertGreater(spilled.info["host_ffn_bytes"], 0)
            self.assertLess(
                spilled.info["gpu_allocated_bytes"],
                resident.info["gpu_allocated_bytes"],
            )
        finally:
            resident.close()
            spilled.close()
            model.close()

    def test_parallel_engine_handles_spilled_dense_layers(self):
        model, _, _ = _model()
        runtime = _native(
            model, parallel_sequences=2, gpu_cache_bytes=36 * 1024 * 1024
        )
        try:
            self.assertGreater(runtime.info["host_ffn_layers"], 0)
            task_ids = {runtime.task_submit([5], 2), runtime.task_submit([11], 2)}
            completed = set()
            for _ in range(12):
                for task_id, _, kind in runtime.engine_step():
                    if kind == 1:
                        completed.add(task_id)
                if completed == task_ids:
                    break
            self.assertEqual(completed, task_ids)
        finally:
            runtime.close()
            model.close()


if __name__ == "__main__":
    unittest.main()
