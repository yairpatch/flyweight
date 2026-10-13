"""Native dense-model planning and offload tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model

from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf


SPILL_BUDGET_BYTES = 32 * 1024 * 1024


_WORKSPACES: list[tempfile.TemporaryDirectory] = []


_MODELS: list[V2Model] = []


def tearDownModule():
    # /tmp is often a RAM-backed tmpfs and these fixtures are megabytes each,
    # so a directory per test adds up fast across repeated runs.
    #
    # Windows refuses to delete a file that is still mapped, so every model
    # opened here is closed before its workspace goes -- a test that forgets
    # its own close() would otherwise take the module's teardown down with it.
    # close() is idempotent, so closing early costs nothing here.
    for model in _MODELS:
        model.close()
    _MODELS.clear()
    for holder in _WORKSPACES:
        holder.cleanup()
    _WORKSPACES.clear()


def _model(**kwargs) -> tuple[V2Model, DenseQwenSpec, Path]:
    holder = tempfile.TemporaryDirectory(prefix="flyweight-dense-")
    _WORKSPACES.append(holder)
    directory = Path(holder.name)
    path = directory / "dense.gguf"
    spec = build_dense_qwen35_gguf(path, **kwargs)
    model = V2Model(path)
    _MODELS.append(model)
    return model, spec, path


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

    def test_restored_checkpoints_keep_their_recency(self):
        # A conversation displaced into the host cache, re-sent, then
        # continued must find the same checkpoint it would have without the
        # round trip. The restore used to stamp the checkpoints newest first,
        # inverting their recency, so the re-send's end-of-prompt snapshot
        # overwrote the freshest mid checkpoint (200 here) instead of the
        # oldest, and the continuation fell back to 100.
        V2Model.select_backend("cpu")
        model, _, _ = _model()
        main = [5] * 300
        diverged = [*main[:250], *([9] * 20)]

        def reuse(displace: bool) -> int:
            runtime = model.native_runtime(
                context_limit=1024, mtp_drafts=0, parallel_sequences=1,
                prompt_cache_mib=64, prefill_checkpoint_interval=64)
            runtime.prepare()
            try:
                # More than one token, so the re-send cannot take the live
                # state whole and has to restore the tail checkpoint.
                _collect_task(runtime, runtime.task_submit(main, 4))
                if displace:
                    _run_task(runtime, [11] * 300)
                _run_task(runtime, main)
                _run_task(runtime, diverged)
                return runtime.info["prefix_cache_last_reused_tokens"]
            finally:
                runtime.close()

        try:
            self.assertEqual(reuse(displace=False), 200)
            self.assertEqual(reuse(displace=True), 200)
        finally:
            V2Model.select_backend("auto")

    def _cached_runtime(self, model: V2Model, **options):
        runtime = model.native_runtime(
            context_limit=512,
            mtp_drafts=0,
            parallel_sequences=1,
            prompt_cache_mib=64,
            **options,
        )
        runtime.prepare()
        return runtime

    def _cache_directory(self) -> Path:
        holder = tempfile.TemporaryDirectory(prefix="flyweight-prompts-")
        _WORKSPACES.append(holder)
        return Path(holder.name) / "cache"

    def test_prompt_cache_survives_a_restart(self):
        # The conversation still on the slot is the one worth keeping, and
        # nothing displaced it into the host cache: saving has to spill it.
        # Restored, it must continue exactly as the live runtime did.
        V2Model.select_backend("cpu")
        model, _, _ = _model()
        directory = self._cache_directory()
        main = [5] * 300
        try:
            first = self._cached_runtime(model)
            try:
                _run_task(first, main)
                fingerprint = first.prompt_cache_fingerprint()
                self.assertEqual(first.save_prompt_cache(directory, fingerprint), 1)
                # Unchanged entries are not written again.
                self.assertEqual(first.save_prompt_cache(directory, fingerprint), 0)
                live = _collect_task(first, first.task_submit([*main, 7], 4))
            finally:
                first.close()
            self.assertEqual(len(list(directory.glob("*.fwpc"))), 1)

            second = self._cached_runtime(model)
            try:
                self.assertEqual(second.prompt_cache_fingerprint(), fingerprint)
                entries, size = second.load_prompt_cache(directory, fingerprint)
                self.assertEqual(entries, 1)
                self.assertGreater(size, 0)
                restored = _collect_task(second, second.task_submit([*main, 7], 4))
                self.assertEqual(
                    second.info["prefix_cache_last_reused_tokens"], len(main))
                self.assertEqual(restored, live)
            finally:
                second.close()
        finally:
            V2Model.select_backend("auto")

    def test_server_restores_at_startup_and_saves_at_close(self):
        from flyweight.v2_server import NativeV2InferenceService

        V2Model.select_backend("cpu")
        model, _, path = _model()
        root = self._cache_directory()

        def service(runtime):
            # The prompt cache plumbing alone, without the HTTP layer.
            instance = NativeV2InferenceService.__new__(NativeV2InferenceService)
            instance.v2_runtime = runtime
            instance._prompt_cache_dir = instance._load_prompt_cache(root, path)
            return instance

        try:
            first = self._cached_runtime(model)
            try:
                saver = service(first)
                directory = saver._prompt_cache_dir
                self.assertEqual(directory.parent, root)
                # Older configurations of the same model, oldest first.
                stale = [root / f"{path.stem}-{index:016x}" for index in range(3)]
                for age, old in enumerate(reversed(stale), start=1):
                    old.mkdir(parents=True)
                    os.utime(old, (age * 1000, age * 1000))
                _run_task(first, [5] * 300)
                saver._save_prompt_cache()
            finally:
                first.close()
            self.assertEqual(len(list(directory.glob("*.fwpc"))), 1)
            # Three configurations are kept: this one and the two newest.
            self.assertEqual(
                sorted(p.name for p in root.iterdir()),
                sorted([directory.name, stale[0].name, stale[1].name]))

            second = self._cached_runtime(model)
            try:
                service(second)
                self.assertEqual(second.info["prompt_cache_entries"], 1)
            finally:
                second.close()
        finally:
            V2Model.select_backend("auto")

    def test_prompt_cache_skips_what_it_cannot_use(self):
        V2Model.select_backend("cpu")
        model, _, _ = _model()
        directory = self._cache_directory()
        main = [5] * 300

        def damage() -> None:
            entry = next(directory.glob("*.fwpc"))
            entry.write_bytes(entry.read_bytes()[: entry.stat().st_size // 2])

        try:
            first = self._cached_runtime(model)
            try:
                _run_task(first, main)
                fingerprint = first.prompt_cache_fingerprint()
                first.save_prompt_cache(directory, fingerprint)
            finally:
                first.close()
            second = self._cached_runtime(model)
            try:
                # Another configuration's entry is left alone...
                self.assertEqual(
                    second.load_prompt_cache(directory, fingerprint + "x"), (0, 0))
                # ...and a damaged one is skipped, then replaced when the same
                # conversation is saved over it.
                damage()
                self.assertEqual(
                    second.load_prompt_cache(directory, fingerprint), (0, 0))
                self.assertEqual(second.info["prompt_cache_entries"], 0)
                _run_task(second, main)
                self.assertEqual(second.save_prompt_cache(directory, fingerprint), 1)
            finally:
                second.close()
            third = self._cached_runtime(model)
            try:
                self.assertEqual(
                    third.load_prompt_cache(directory, fingerprint)[0], 1)
            finally:
                third.close()
            # Damaged with nothing to replace it, the next save drops it.
            damage()
            fourth = self._cached_runtime(model)
            try:
                self.assertEqual(
                    fourth.load_prompt_cache(directory, fingerprint), (0, 0))
                self.assertEqual(fourth.save_prompt_cache(directory, fingerprint), 0)
                self.assertEqual(list(directory.glob("*.fwpc")), [])
            finally:
                fourth.close()
        finally:
            V2Model.select_backend("auto")

    def test_prompt_cache_fingerprint_tracks_meaning_not_placement(self):
        V2Model.select_backend("cpu")
        model, _, _ = _model()
        try:
            base = self._cached_runtime(model)
            placed = model.native_runtime(
                context_limit=256, mtp_drafts=0, parallel_sequences=2,
                prompt_cache_mib=128, cpu_threads=2)
            placed.prepare()
            quantized = self._cached_runtime(model, cache_type_k="q8_0",
                                             cache_type_v="q8_0")
            try:
                self.assertEqual(base.prompt_cache_fingerprint(),
                                 placed.prompt_cache_fingerprint())
                self.assertNotEqual(base.prompt_cache_fingerprint(),
                                    quantized.prompt_cache_fingerprint())
            finally:
                base.close()
                placed.close()
                quantized.close()
        finally:
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
