"""Native dense-model planning and offload tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model

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
    holder = tempfile.TemporaryDirectory(prefix="flyweight-dense-")
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


def _run_task(runtime, prompt: list[int]) -> None:
    task_id = runtime.task_submit(prompt, 1)
    for _ in range(16):
        if any(
            event_task == task_id and kind == 1
            for event_task, _, kind in runtime.engine_step()
        ):
            return
    raise AssertionError("native engine task did not finish")


def _collect_task(runtime, task_id: int, steps: int = 64) -> list[int]:
    tokens: list[int] = []
    for _ in range(steps):
        for event_task, token, kind in runtime.engine_step():
            if event_task != task_id:
                continue
            if kind == 0:
                tokens.append(token)
            elif kind == 1:
                return tokens
            elif kind == 2:
                raise AssertionError("native engine task failed")
    raise AssertionError("native engine task did not finish")


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
    def test_single_slot_host_cache_restores_displaced_conversation(self):
        V2Model.select_backend("cpu")
        model, _, _ = _model()
        runtime = model.native_runtime(
            context_limit=512,
            mtp_drafts=0,
            parallel_sequences=1,
            prompt_cache_mib=64,
        )
        runtime.prepare()
        main = [5] * 300
        side = [11] * 300
        try:
            _run_task(runtime, main)
            _run_task(runtime, side)
            self.assertGreaterEqual(runtime.info["prompt_cache_entries"], 1)

            _run_task(runtime, [*main, 7])

            self.assertEqual(runtime.info["prefix_cache_last_reused_tokens"], 300)
        finally:
            runtime.close()
            model.close()
            V2Model.select_backend("auto")

    def test_tail_checkpoint_survives_a_last_token_divergence(self):
        # A stripped-reasoning replay re-tokenizes the newline after the
        # forced "<think>" opener, so the next turn agrees with the previous
        # prompt up to exactly its LAST token. The reserved snapshot is taken
        # one token short of the end (qwen_prompt_begin's tail target) so that
        # turn restores everything but that token, instead of falling back to
        # a mid checkpoint a quarter of the prompt behind.
        V2Model.select_backend("cpu")
        model, _, _ = _model()
        runtime = model.native_runtime(
            context_limit=512,
            mtp_drafts=0,
            parallel_sequences=1,
        )
        runtime.prepare()
        first = [5] * 300
        try:
            _run_task(runtime, first)

            _run_task(runtime, [*first[:-1], 7, 8])

            self.assertEqual(
                runtime.info["prefix_cache_last_reused_tokens"], len(first) - 1
            )
        finally:
            runtime.close()
            model.close()
            V2Model.select_backend("auto")

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

    def test_greedy_request_does_not_inherit_a_sampled_first_token(self):
        # An exact full-prompt prefix-cache hit reuses the remembered next
        # token instead of replaying the last prompt token. If a previous
        # request *sampled* that token, a later greedy request must not emit
        # it: greedy output would then depend on which request ran before.
        model, _, _ = _model()
        warmed = _native(model)
        fresh = _native(model)
        prompt = [5, 9, 3, 7, 2]
        try:
            baseline = _collect_task(fresh, fresh.task_submit(prompt, 4))
            # max_tokens=1 leaves processed_tokens exactly equal to the prompt,
            # which is the full-prompt prefix-cache hit the greedy request
            # below will take. Find a seed whose sampled token actually
            # diverges from the greedy one, so the reuse path is exercised
            # rather than trivially agreeing.
            diverged = False
            for seed in range(8):
                sampled = _collect_task(
                    warmed,
                    warmed.task_submit(prompt, 1, temperature=1.8, seed=seed),
                )
                if sampled[0] != baseline[0]:
                    diverged = True
                    break
            greedy = _collect_task(warmed, warmed.task_submit(prompt, 4))
            self.assertEqual(greedy, baseline)
            # On this fixture at temperature 1.8 at least one of the seeds is
            # expected to diverge; if none did, the assertion above proved
            # nothing about reuse and the fixture needs a new prompt.
            self.assertTrue(diverged, "no sampled seed diverged from greedy")
        finally:
            warmed.close()
            fresh.close()
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
