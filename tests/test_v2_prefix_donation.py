"""Cross-slot prefix donation (plans/paged-kv-cache.md, phase 1).

A prompt that shares a live conversation's opening without continuing it used
to take that conversation's slot: it reused the prefix and then overwrote
everything behind it, so the owner's next turn reprefilled. Donation copies the
shared prefix into a spare slot instead and leaves the owner alone.

Two things have to hold, and only one of them is a performance claim:

  * the donated slot produces exactly the tokens it would have produced from a
    cold prefill -- a spliced KV or a checkpoint restored at the wrong position
    shows up here and nowhere else;
  * the donor still holds its conversation afterwards, which is the whole point.

Run on the synthetic qwen4exp fixture (CPU backend, f32), whose 256-token
context is far below the real 2048-token prefix bar -- FLYWEIGHT_PREFIX_BAR
lowers it, which is what that override exists for.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model
from tests.qwen4exp_gguf_fixture import build_qwen4exp_gguf

# A shared opening ("system prompt"), then two conversations that diverge.
SHARED = [(t * 37 + 11) % 96 for t in range(24)]
MAIN = SHARED + [(t * 13 + 5) % 96 for t in range(24)]
SIDE = SHARED + [(t * 29 + 3) % 96 for t in range(8)]
CONTINUATION = 4
CONTEXT = 256


class PrefixDonationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen4exp.gguf"
        build_qwen4exp_gguf(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def setUp(self) -> None:
        self._bar = os.environ.get("FLYWEIGHT_PREFIX_BAR")
        # Below the shared opening, so SHARED clears the bar and the divergent
        # tail keeps SIDE from looking like a continuation of MAIN.
        os.environ["FLYWEIGHT_PREFIX_BAR"] = "8"
        V2Model.select_backend("cpu")

    def tearDown(self) -> None:
        V2Model.select_backend("auto")
        if self._bar is None:
            os.environ.pop("FLYWEIGHT_PREFIX_BAR", None)
        else:
            os.environ["FLYWEIGHT_PREFIX_BAR"] = self._bar

    def _generate(self, runtime, prompt: list[int]) -> list[int]:
        out: list[int] = []
        runtime.generate(
            prompt, CONTINUATION,
            lambda t: (out.append(t) or len(out) < CONTINUATION))
        return out

    def _solo(self, prompt: list[int]) -> list[int]:
        """What this prompt produces on a runtime that has seen nothing else."""
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT,
                    prefill_checkpoint_interval=8) as runtime:
                runtime.prepare()
                return self._generate(runtime, prompt)

    def test_shared_prefix_is_donated_not_stolen(self) -> None:
        expected_side = self._solo(SIDE)
        expected_main = self._solo(MAIN)
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT, parallel_sequences=2,
                    prefill_checkpoint_interval=8) as runtime:
                runtime.prepare()
                self._generate(runtime, MAIN)
                main_slot = runtime.info["position"]

                side = self._generate(runtime, SIDE)
                info = runtime.info
                self.assertEqual(info["prefix_donations"], 1,
                                 "the side request did not trigger a donation")
                self.assertGreaterEqual(info["prefix_donated_tokens"],
                                        len(SHARED) - 8)
                # The side request rode the donated prefix instead of
                # prefilling it.
                self.assertGreaterEqual(
                    info["prefix_cache_last_reused_tokens"], len(SHARED) - 8)
                # Donated state must decode identically to a cold prefill.
                self.assertEqual(side, expected_side)
                # And the donor is still there: continuing MAIN reuses its whole
                # live slot rather than reprefilling anything.
                again = self._generate(runtime, MAIN)
                self.assertEqual(again, expected_main)
                self.assertEqual(
                    runtime.info["prefix_cache_last_reused_tokens"], len(MAIN),
                    "the donor slot was not preserved")
                self.assertEqual(runtime.info["prefix_donations"], 1,
                                 "continuing the donor should not donate again")
                self.assertGreater(main_slot, 0)

    def test_without_donation_the_side_request_evicts_the_donor(self) -> None:
        """The contrast the phase exists for, held by the kill switch.

        Same traffic with FLYWEIGHT_PREFIX_DONATE=0: the side request takes the
        slot, and the main conversation's next turn can no longer reuse all of
        itself. Asserting the gap -- not just that donation happens -- is what
        keeps this from passing on a runtime where donation quietly does
        nothing.
        """
        os.environ["FLYWEIGHT_PREFIX_DONATE"] = "0"
        try:
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(
                        context_limit=CONTEXT, parallel_sequences=2,
                        prefill_checkpoint_interval=8) as runtime:
                    runtime.prepare()
                    self._generate(runtime, MAIN)
                    self._generate(runtime, SIDE)
                    self._generate(runtime, MAIN)
                    info = runtime.info
        finally:
            os.environ.pop("FLYWEIGHT_PREFIX_DONATE", None)
        self.assertEqual(info["prefix_donations"], 0)
        self.assertLess(info["prefix_cache_last_reused_tokens"], len(MAIN),
                        "without donation the main turn should not fully reuse")

    def test_a_continuation_still_keeps_its_own_slot(self) -> None:
        """Extending a conversation is not a donation.

        The discriminator is fractional: a prompt covering most of a slot's
        committed tokens continues it, and moving that to another slot would
        copy the whole conversation for nothing.
        """
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT, parallel_sequences=2,
                    prefill_checkpoint_interval=8) as runtime:
                runtime.prepare()
                produced = self._generate(runtime, MAIN)
                extended = MAIN + produced + [7, 11]
                self._generate(runtime, extended)
                self.assertEqual(runtime.info["prefix_donations"], 0)

    def test_single_slot_never_donates(self) -> None:
        """With nowhere to donate to, the old take-the-slot behaviour stands."""
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT, parallel_sequences=1,
                    prefill_checkpoint_interval=8) as runtime:
                runtime.prepare()
                self._generate(runtime, MAIN)
                self._generate(runtime, SIDE)
                self.assertEqual(runtime.info["prefix_donations"], 0)


if __name__ == "__main__":
    unittest.main()
