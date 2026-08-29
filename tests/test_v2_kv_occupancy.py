"""KV occupancy telemetry: what the slots reserve vs what they ever hold.

Phase 0 of plans/paged-kv-cache.md. Slots are sized for the full context at
prepare, so the interesting number is the high-water mark of the live state,
which is what a paged pool would have had to allocate instead. These pin the
accounting on the synthetic qwen4exp fixture (CPU backend, f32) so a regression
shows up without a 27 GB checkpoint: a counter that silently reads zero, or a
peak that exceeds the reservation, would make the phase-2 case unmeasurable.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model
from tests.qwen4exp_gguf_fixture import build_qwen4exp_gguf

PROMPT = [(t * 37 + 11) % 96 for t in range(24)]
CONTINUATION = 4
CONTEXT = 256


class KvOccupancyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen4exp.gguf"
        build_qwen4exp_gguf(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _runtime(self, model: V2Model, slots: int):
        return model.native_qwen_runtime(
            context_limit=CONTEXT, parallel_sequences=slots)

    def test_reservation_is_reported_before_any_request(self) -> None:
        V2Model.select_backend("cpu")
        try:
            with V2Model(str(self.path)) as model:
                with self._runtime(model, 2) as runtime:
                    runtime.prepare()
                    info = runtime.info
                    self.assertEqual(info["kv_slots"], 2)
                    self.assertEqual(
                        info["kv_reserved_bytes"],
                        2 * info["state_bytes"])
                    # No request has finished, so there is nothing to report --
                    # and the sample count is what says so, rather than the
                    # peak reading zero and looking like an empty cache.
                    self.assertEqual(info["kv_occupancy_samples"], 0)
                    self.assertEqual(info["kv_peak_live_bytes"], 0)
                    self.assertEqual(info["kv_peak_tokens"], 0)
        finally:
            V2Model.select_backend("auto")

    def test_generate_records_a_peak_below_the_reservation(self) -> None:
        V2Model.select_backend("cpu")
        try:
            with V2Model(str(self.path)) as model:
                with self._runtime(model, 2) as runtime:
                    runtime.prepare()
                    produced: list[int] = []
                    runtime.generate(
                        PROMPT, CONTINUATION,
                        lambda t: (produced.append(t)
                                   or len(produced) < CONTINUATION))
                    info = runtime.info
            self.assertGreaterEqual(info["kv_occupancy_samples"], 1)
            self.assertGreater(info["kv_peak_live_bytes"], 0)
            # The whole point of the phase: 28 tokens in a 256-token context
            # across two slots cannot need the reservation.
            self.assertLess(info["kv_peak_live_bytes"],
                            info["kv_reserved_bytes"])
            self.assertEqual(info["kv_peak_tokens_max"], info["position"])
            # One slot ran; the other is idle, so the sum is the busy slot's.
            self.assertEqual(info["kv_peak_tokens"], info["position"])
            self.assertLessEqual(info["kv_peak_tokens_max"],
                                 len(PROMPT) + CONTINUATION)
        finally:
            V2Model.select_backend("auto")

    def test_idle_slots_still_cost_their_unpageable_floor(self) -> None:
        """Two slots peak higher than one for the same conversation.

        A slot that never decodes still holds fixed-size DeltaNet conv and
        recurrent state, which no block pool can reclaim. Counting it is what
        keeps the phase-2 estimate honest -- the reclaimable figure is the
        position-indexed KV, not the whole arena.
        """
        V2Model.select_backend("cpu")
        try:
            peaks = []
            for slots in (1, 2):
                with V2Model(str(self.path)) as model:
                    with self._runtime(model, slots) as runtime:
                        runtime.prepare()
                        produced: list[int] = []
                        runtime.generate(
                            PROMPT, CONTINUATION,
                            lambda t: (produced.append(t)
                                       or len(produced) < CONTINUATION))
                        peaks.append(runtime.info["kv_peak_live_bytes"])
            self.assertGreater(peaks[1], peaks[0])
        finally:
            V2Model.select_backend("auto")

    def test_engine_samples_at_its_own_request_boundary(self) -> None:
        V2Model.select_backend("cpu")
        try:
            with V2Model(str(self.path)) as model:
                with self._runtime(model, 2) as runtime:
                    runtime.prepare()
                    task = runtime.task_submit(PROMPT, CONTINUATION)
                    for _ in range(64 + 8 * CONTINUATION):
                        done = False
                        for identifier, _token, kind in runtime.engine_step():
                            if kind == 2:
                                raise AssertionError("engine task failed")
                            if kind == 1 and identifier == task:
                                done = True
                        if done:
                            break
                    else:
                        raise AssertionError("engine task did not finish")
                    info = runtime.info
            self.assertGreaterEqual(info["kv_occupancy_samples"], 1)
            self.assertGreater(info["kv_peak_live_bytes"], 0)
            self.assertLess(info["kv_peak_live_bytes"],
                            info["kv_reserved_bytes"])
        finally:
            V2Model.select_backend("auto")


if __name__ == "__main__":
    unittest.main()
