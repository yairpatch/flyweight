"""Concurrent decode across slots of DIFFERENT sizes must stay correct.

The multi-sequence decode driver reads every layer offset and ring capacity
from the one mirrored layer table, while each batched sequence supplies its
own arena base. With `scratch_context` the slots have different layouts, so a
mixed batch used to address one slot's arena with the other slot's offsets --
cross-region corruption inside the arena, or writes past the end of the
smaller one. The engine now batches only same-geometry slots (the rest decode
serially in the same step) and the driver refuses a mismatched slot outright.

These pin the fix on the synthetic qwen4exp fixture (CPU backend, f32): two
tasks decoding concurrently on a full slot and a scratch slot must produce
exactly what each produces alone, and the batch path must actually have run
for the test to mean anything (the vacuous-kill-switch lesson).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model
from tests.qwen4exp_gguf_fixture import build_qwen4exp_gguf

LONG_PROMPT = [(t * 37 + 11) % 96 for t in range(96)]
SHORT_PROMPT = [(t * 29 + 7) % 96 for t in range(16)]
CONTINUATION = 8
CONTEXT = 256
SCRATCH = 64


class MultiGeometryDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen4exp.gguf"
        build_qwen4exp_gguf(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def setUp(self) -> None:
        V2Model.select_backend("cpu")

    def tearDown(self) -> None:
        V2Model.select_backend("auto")

    def _runtime(self, model):
        return model.native_qwen_runtime(
            context_limit=CONTEXT, parallel_sequences=2,
            scratch_context=SCRATCH)

    def _drain(self, runtime, tasks: dict[int, list[int]]) -> None:
        remaining = set(tasks)
        for _ in range(64 + 64 * CONTINUATION):
            for identifier, token, kind in runtime.engine_step():
                if kind == 2:
                    raise AssertionError("engine task failed")
                if kind == 0:
                    tasks[identifier].append(token)
                if kind == 1:
                    remaining.discard(identifier)
            if not remaining:
                return
        raise AssertionError("engine tasks did not finish")

    def _solo(self, prompt: list[int]) -> list[int]:
        with V2Model(str(self.path)) as model:
            with self._runtime(model) as runtime:
                runtime.prepare()
                tasks = {runtime.task_submit(prompt, CONTINUATION): []}
                self._drain(runtime, tasks)
                (tokens,) = tasks.values()
                return tokens

    def test_concurrent_mixed_slots_match_solo_runs(self) -> None:
        expected_long = self._solo(LONG_PROMPT)
        expected_short = self._solo(SHORT_PROMPT)
        self.assertEqual(len(expected_long), CONTINUATION)
        with V2Model(str(self.path)) as model:
            with self._runtime(model) as runtime:
                runtime.prepare()
                # The long prompt only fits the full slot; the short one is
                # routed to the smallest slot that fits, the scratch slot. Both
                # decode concurrently, which is the batch the driver used to
                # address with a single geometry.
                long_task = runtime.task_submit(LONG_PROMPT, CONTINUATION)
                short_task = runtime.task_submit(SHORT_PROMPT, CONTINUATION)
                tasks: dict[int, list[int]] = {long_task: [], short_task: []}
                self._drain(runtime, tasks)
                info = runtime.info
        self.assertEqual(tasks[long_task], expected_long)
        self.assertEqual(tasks[short_task], expected_short)
        # If nothing was ever batched, this test exercised only the serial
        # path and proves nothing about the mixed-geometry hazard.
        self.assertGreaterEqual(int(info.get("multi_decode_batches", 0)), 1)


if __name__ == "__main__":
    unittest.main()
