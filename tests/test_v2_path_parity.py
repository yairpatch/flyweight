"""Differential path-parity harness.

The runtime reaches the same logical computation through several independent
paths: blocking generate vs the cooperative engine, chunked rows prefill vs the
single-token loop, a fresh prefill vs live/snapshot/host-cache prefix reuse,
and solo vs interleaved multi-sequence decode. Each pair is supposed to be
bit-identical for greedy requests, and history says divergence bugs here are
silent: a path that drops an expert's output or reuses a sampled token still
produces fluent text. This file runs the same requests through every pair and
asserts token equality, so a future change that splits the paths fails a test
instead of a bisection session.

All models are tiny synthetic GGUF fixtures; every test needs CUDA and skips
without it.
"""

from __future__ import annotations

import os
import random
import tempfile
import unittest
from pathlib import Path

from colibri_next.v2 import V2Model

from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf
from tests.laguna_gguf_fixture import build_laguna_gguf


_WORKSPACES: list[tempfile.TemporaryDirectory] = []


def tearDownModule():
    for holder in _WORKSPACES:
        holder.cleanup()
    _WORKSPACES.clear()


def _workspace(prefix: str) -> Path:
    holder = tempfile.TemporaryDirectory(prefix=prefix)
    _WORKSPACES.append(holder)
    return Path(holder.name)


def _require_cuda():
    if not V2Model.gpu_info()["available"]:
        raise unittest.SkipTest("native CUDA runtime is unavailable")


def _prompts(vocabulary: int, lengths: tuple[int, ...]) -> list[list[int]]:
    # Deterministic and reproducible in a failure report; the lengths cross
    # the prefill-unit boundaries (1 = no chunk possible, 2 = below the
    # chunkable minimum, 3 = the smallest chunkable prompt, then one chunk,
    # and one prompt spanning multiple 256-row chunks plus the scalar tail).
    rng = random.Random(0xC0)
    return [[rng.randrange(vocabulary) for _ in range(n)] for n in lengths]


def _engine_run(runtime, prompt: list[int], max_tokens: int, **kw) -> list[int]:
    task_id = runtime.task_submit(prompt, max_tokens, **kw)
    tokens: list[int] = []
    for _ in range(64 + 4 * max_tokens):
        for event_task, token, kind in runtime.engine_step():
            if event_task != task_id:
                continue
            if kind == 0:
                tokens.append(token)
            elif kind == 1:
                return tokens
            elif kind == 2:
                raise AssertionError("engine task failed")
    raise AssertionError("engine task did not finish")


def _blocking_run(runtime, prompt: list[int], max_tokens: int) -> list[int]:
    tokens: list[int] = []
    runtime.generate(prompt, max_tokens, tokens.append)
    return tokens


class _ParityCase(unittest.TestCase):
    """Shared fixture plumbing; concrete classes pick the model."""

    maxDiff = None

    def _runtime(self, **options):
        _require_cuda()
        runtime = self._model.native_runtime(mtp_drafts=0, **self._options(options))
        runtime.prepare()
        self.addCleanup(runtime.close)
        return runtime

    def _options(self, overrides: dict) -> dict:
        options = dict(self._base_options)
        options.update(overrides)
        return options

    def _assert_same(self, label: str, expected: list[int], actual: list[int]):
        self.assertEqual(
            expected, actual,
            f"{label}: paths diverged (expected is the reference path)",
        )


class DensePathParityTests(_ParityCase):
    """Dense Qwen3.5 fixture: DeltaNet + attention, no experts."""

    LENGTHS = (1, 2, 3, 9, 40, 300)
    TOKENS = 6

    @classmethod
    def setUpClass(cls):
        path = _workspace("colibri-parity-dense-") / "dense.gguf"
        cls._spec = build_dense_qwen35_gguf(path)
        cls._model = V2Model(path)
        cls._base_options = {"context_limit": 512}

    @classmethod
    def tearDownClass(cls):
        cls._model.close()

    def test_blocking_and_engine_paths_agree(self):
        # The engine's emit-then-compute order is documented to keep a
        # single-task run bit-identical to blocking generate.
        for prompt in _prompts(self._spec.vocabulary, self.LENGTHS):
            with self.subTest(prompt_tokens=len(prompt)):
                blocking = _blocking_run(self._runtime(), prompt, self.TOKENS)
                engine = _engine_run(self._runtime(), prompt, self.TOKENS)
                self._assert_same(f"len={len(prompt)}", blocking, engine)

    def test_chunked_and_single_token_prefill_agree(self):
        # COLIBRI_PREFILL_ROWS=0 disables the batched rows forward, so the
        # whole prompt runs through single-token decode. Both must land on
        # the same state and the same continuation.
        for prompt in _prompts(self._spec.vocabulary, self.LENGTHS):
            with self.subTest(prompt_tokens=len(prompt)):
                chunked = _engine_run(self._runtime(), prompt, self.TOKENS)
                os.environ["COLIBRI_PREFILL_ROWS"] = "0"
                try:
                    unchunked = _engine_run(self._runtime(), prompt, self.TOKENS)
                finally:
                    del os.environ["COLIBRI_PREFILL_ROWS"]
                self._assert_same(f"len={len(prompt)}", chunked, unchunked)

    def test_live_continuation_and_exact_resend_match_fresh(self):
        prompt = _prompts(self._spec.vocabulary, (40,))[0]
        warmed = self._runtime()
        first = _engine_run(warmed, prompt, 4)
        # Exact re-send: a full-prompt prefix hit that reuses the remembered
        # greedy token instead of replaying the last prompt token.
        resent = _engine_run(warmed, prompt, 4)
        self._assert_same("exact resend", first, resent)
        # Continuation the way a client renders it: prompt + the reply the
        # runtime just generated + a new user token. The warmed runtime
        # extends live state; the fresh one prefills from scratch.
        continuation = prompt + first + [7]
        reused = _engine_run(warmed, continuation, self.TOKENS)
        self.assertGreater(
            warmed.info["prefix_cache_last_reused_tokens"], 0,
            "continuation did not exercise the reuse path",
        )
        fresh = _engine_run(self._runtime(), continuation, self.TOKENS)
        self._assert_same("live continuation", fresh, reused)

    def test_snapshot_reuse_after_divergence_matches_fresh(self):
        # Two prompts sharing a 24-token base but diverging after it: the
        # second cannot extend live state and must restore a mid-prefill
        # checkpoint. Tight interval so the tiny prompt gets checkpoints.
        vocabulary = self._spec.vocabulary
        rng = random.Random(7)
        base = [rng.randrange(vocabulary) for _ in range(24)]
        suffix_a = [rng.randrange(vocabulary) for _ in range(12)]
        suffix_b = [rng.randrange(vocabulary) for _ in range(12)]
        options = {"prefill_checkpoint_interval": 8, "prefill_checkpoint_slots": 4}
        warmed = self._runtime(**options)
        _engine_run(warmed, base + suffix_a, 4)
        diverged = _engine_run(warmed, base + suffix_b, self.TOKENS)
        self.assertGreater(
            warmed.info["prefix_cache_last_reused_tokens"], 0,
            "divergent prompt did not exercise the snapshot path",
        )
        fresh = _engine_run(self._runtime(**options), base + suffix_b, self.TOKENS)
        self._assert_same("snapshot reuse", fresh, diverged)

    def test_host_cache_restore_matches_fresh(self):
        # One GPU slot: the side request displaces the main conversation to
        # host RAM; its continuation restores the arena from RAM. The restored
        # continuation must match a runtime that never lost the slot.
        vocabulary = self._spec.vocabulary
        rng = random.Random(11)
        main = [rng.randrange(vocabulary) for _ in range(300)]
        side = [rng.randrange(vocabulary) for _ in range(300)]
        options = {"parallel_sequences": 1, "prompt_cache_mib": 64}
        warmed = self._runtime(**options)
        first = _engine_run(warmed, main, 1)
        _engine_run(warmed, side, 1)
        continuation = main + first + [7]
        restored = _engine_run(warmed, continuation, self.TOKENS)
        self.assertGreater(
            warmed.info["prefix_cache_last_reused_tokens"], 0,
            "continuation did not exercise the host-cache restore path",
        )
        fresh = self._runtime()
        _engine_run(fresh, main, 1)
        expected = _engine_run(fresh, continuation, self.TOKENS)
        self._assert_same("host-cache restore", expected, restored)

    def test_interleaved_tasks_match_solo_runs(self):
        # Two tasks share the engine round-robin but own separate KV arenas;
        # interleaving must not change either output.
        prompt_a, prompt_b = _prompts(self._spec.vocabulary, (40, 33))
        solo_a = _engine_run(self._runtime(), prompt_a, self.TOKENS)
        solo_b = _engine_run(self._runtime(), prompt_b, self.TOKENS)
        runtime = self._runtime(parallel_sequences=2)
        task_a = runtime.task_submit(prompt_a, self.TOKENS)
        task_b = runtime.task_submit(prompt_b, self.TOKENS)
        collected: dict[int, list[int]] = {task_a: [], task_b: []}
        finished: set[int] = set()
        for _ in range(64 + 8 * self.TOKENS):
            for event_task, token, kind in runtime.engine_step():
                if kind == 0:
                    collected[event_task].append(token)
                elif kind == 1:
                    finished.add(event_task)
                elif kind == 2:
                    raise AssertionError("engine task failed")
            if finished == {task_a, task_b}:
                break
        self.assertEqual(finished, {task_a, task_b}, "tasks did not finish")
        self._assert_same("interleaved task A", solo_a, collected[task_a])
        self._assert_same("interleaved task B", solo_b, collected[task_b])


class MoeBatchedDecodeParityTests(_ParityCase):
    """Qwen3.5 MoE fixture: the multi-sequence decode driver vs solo runs.

    CPU-routed experts with two sequence slots is the configuration where
    concurrent tasks share qwen_decode_multi; every scenario asserts the
    batched path actually engaged, so a gating change cannot quietly turn
    these into serial-path tests. First runs of this class caught the driver
    decoding non-Q8 shared experts as Q8_0 bytes.
    """

    TOKENS = 6
    SAMPLING = {"temperature": 0.9, "top_k": 8, "top_p": 0.95}

    @classmethod
    def setUpClass(cls):
        path = _workspace("colibri-parity-qwen-moe-") / "moe.gguf"
        cls._spec = build_dense_qwen35_gguf(
            path, DenseQwenSpec(experts=8, experts_used=2)
        )
        cls._model = V2Model(path)
        cls._base_options = {
            "context_limit": 256,
            "expert_mode": "cpu",
        }
        # The expert-history sidecar feeds earlier runs' routing back into
        # later placement -- an ambient input this parity class must not have.
        os.environ["COLIBRI_EXPERT_HISTORY"] = "off"

    @classmethod
    def tearDownClass(cls):
        cls._model.close()
        del os.environ["COLIBRI_EXPERT_HISTORY"]

    def _prompt_pair(self) -> tuple[list[int], list[int]]:
        return tuple(_prompts(self._spec.vocabulary, (40, 33)))

    def _run_pair(self, kw_a: dict, kw_b: dict) -> tuple[list[int], list[int]]:
        prompt_a, prompt_b = self._prompt_pair()
        runtime = self._runtime(parallel_sequences=2)
        task_a = runtime.task_submit(prompt_a, self.TOKENS, **kw_a)
        task_b = runtime.task_submit(prompt_b, self.TOKENS, **kw_b)
        collected: dict[int, list[int]] = {task_a: [], task_b: []}
        finished: set[int] = set()
        for _ in range(64 + 8 * self.TOKENS):
            for event_task, token, kind in runtime.engine_step():
                if kind == 0:
                    collected[event_task].append(token)
                elif kind == 1:
                    finished.add(event_task)
                elif kind == 2:
                    raise AssertionError("engine task failed")
            if finished == {task_a, task_b}:
                break
        self.assertEqual(finished, {task_a, task_b}, "tasks did not finish")
        self.assertGreater(
            runtime.info["multi_decode_batches"], 0,
            "the batched decode driver did not engage",
        )
        return collected[task_a], collected[task_b]

    def _solo(self, prompt: list[int], **kw) -> list[int]:
        return _engine_run(self._runtime(), prompt, self.TOKENS, **kw)

    def test_batched_greedy_tasks_match_solo_runs(self):
        prompt_a, prompt_b = self._prompt_pair()
        solo_a, solo_b = self._solo(prompt_a), self._solo(prompt_b)
        got_a, got_b = self._run_pair({}, {})
        self._assert_same("batched greedy A", solo_a, got_a)
        self._assert_same("batched greedy B", solo_b, got_b)

    def test_batched_sampled_tasks_match_solo_runs(self):
        # Fixed seeds: each task owns its RNG, so the batch must reproduce
        # the exact solo draw. This is the capability the batch gate used to
        # exclude -- sampled tasks decoded serially, one at a time.
        prompt_a, prompt_b = self._prompt_pair()
        solo_a = self._solo(prompt_a, seed=41, **self.SAMPLING)
        solo_b = self._solo(prompt_b, seed=42, **self.SAMPLING)
        got_a, got_b = self._run_pair(
            dict(seed=41, **self.SAMPLING), dict(seed=42, **self.SAMPLING)
        )
        self._assert_same("batched sampled A", solo_a, got_a)
        self._assert_same("batched sampled B", solo_b, got_b)

    def test_batched_mixed_tasks_match_solo_runs(self):
        prompt_a, prompt_b = self._prompt_pair()
        solo_a = self._solo(prompt_a)
        solo_b = self._solo(prompt_b, seed=42, **self.SAMPLING)
        got_a, got_b = self._run_pair({}, dict(seed=42, **self.SAMPLING))
        self._assert_same("mixed greedy A", solo_a, got_a)
        self._assert_same("mixed sampled B", solo_b, got_b)


class MoePathParityTests(_ParityCase):
    """Laguna fixture: routed experts on the CPU path, sliding windows."""

    LENGTHS = (1, 3, 9, 40)
    TOKENS = 6

    @classmethod
    def setUpClass(cls):
        path = _workspace("colibri-parity-laguna-") / "laguna.gguf"
        cls._spec = build_laguna_gguf(path)
        cls._model = V2Model(path)
        cls._base_options = {"context_limit": 64, "expert_mode": "cpu"}

    @classmethod
    def tearDownClass(cls):
        cls._model.close()

    def test_blocking_engine_and_chunk_paths_agree(self):
        for prompt in _prompts(self._spec.vocabulary, self.LENGTHS):
            with self.subTest(prompt_tokens=len(prompt)):
                blocking = _blocking_run(self._runtime(), prompt, self.TOKENS)
                engine = _engine_run(self._runtime(), prompt, self.TOKENS)
                self._assert_same(f"len={len(prompt)} engine", blocking, engine)
                os.environ["COLIBRI_PREFILL_ROWS"] = "0"
                try:
                    unchunked = _engine_run(self._runtime(), prompt, self.TOKENS)
                finally:
                    del os.environ["COLIBRI_PREFILL_ROWS"]
                self._assert_same(f"len={len(prompt)} unchunked", blocking, unchunked)


class PrefillExpertStreamParityTests(_ParityCase):
    """Qwen3.5 MoE fixture: streamed prefill experts vs the resident path.

    Prefill expert streaming stages routed experts into a scratch arena and
    runs the same grouped kernels the resident table path runs, in the same
    compact order -- so with every expert covered on both sides, the outputs
    must be bit-identical. Both sides seed the full expert set so decode
    placement is identical too; only the prefill staging differs. The
    engagement counter guards against this passing vacuously with the
    streaming path never taken.
    """

    TOKENS = 6

    @classmethod
    def setUpClass(cls):
        path = _workspace("colibri-parity-stream-") / "moe.gguf"
        # Q8_0: the grouped GPU expert kernels have no f32 entry, and f32
        # experts would silently fall both sides back to the CPU MoE.
        cls._spec = build_dense_qwen35_gguf(
            path, DenseQwenSpec(experts=8, experts_used=2), quantize="q8_0"
        )
        cls._model = V2Model(path)
        cls._base_options = {
            "context_limit": 256,
            "gpu_cache_bytes": 512 * 1024 * 1024,
            "prefill_cache_seed": 8,
            "expert_residency": "immutable",
        }
        os.environ["COLIBRI_EXPERT_HISTORY"] = "off"

    @classmethod
    def tearDownClass(cls):
        cls._model.close()
        del os.environ["COLIBRI_EXPERT_HISTORY"]

    def _generate(self, **options):
        prompt = _prompts(self._spec.vocabulary, (40,))[0]
        runtime = self._runtime(**options)
        tokens = _blocking_run(runtime, prompt, self.TOKENS)
        return tokens, int(runtime.info["prefill_streamed_bytes"])

    def test_streamed_all_matches_resident_all(self):
        resident, resident_streamed = self._generate(
            expert_mode="legacy-hybrid", hybrid_prefill="split"
        )
        self.assertEqual(resident_streamed, 0,
                         "the resident reference must not stream")
        os.environ["COLIBRI_PREFILL_EXPERT_STREAM_MIB"] = "64"
        try:
            streamed_a, streamed_bytes_a = self._generate(expert_mode="auto")
            streamed_b, streamed_bytes_b = self._generate(expert_mode="auto")
        finally:
            del os.environ["COLIBRI_PREFILL_EXPERT_STREAM_MIB"]
        self.assertGreater(streamed_bytes_a, 0,
                           "the streaming path did not engage")
        self.assertEqual(streamed_bytes_a, streamed_bytes_b,
                         "a fixed budget must stage the same bytes")
        self._assert_same("stream determinism", streamed_a, streamed_b)
        self._assert_same("streamed vs resident", resident, streamed_a)

    def test_budget_zero_is_untouched(self):
        with_default, streamed = self._generate(expert_mode="auto")
        self.assertEqual(streamed, 0, "no budget must mean no streaming")
        os.environ["COLIBRI_PREFILL_EXPERT_STREAM_MIB"] = "0"
        try:
            with_zero, streamed_zero = self._generate(expert_mode="auto")
        finally:
            del os.environ["COLIBRI_PREFILL_EXPERT_STREAM_MIB"]
        self.assertEqual(streamed_zero, 0)
        self._assert_same("budget zero", with_default, with_zero)


if __name__ == "__main__":
    unittest.main()
