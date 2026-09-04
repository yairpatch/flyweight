"""The checkpoint ladder tracks the conversation tip instead of ossifying.

Mid-prefill checkpoint targets are fractions of the WHOLE prompt, and a target
the reused prefix already covers is skipped -- so from the second turn of a
conversation on, every mid sits below `prompt_start` and is never rewritten.
When the end-of-prompt checkpoint owned one reserved slot, that left the ladder
frozen at the first prompt's positions while the tip ran away: measured live on
an agentic session, checkpoints stuck at {256, 4000, 8000} against a 38k-token
tip, and a turn that diverged 352 tokens below the tip snapped back to 25820
and reprefilled 41411 tokens.

The end-of-prompt checkpoint now rotates over the slots no mid target claims,
so a session keeps the last few prompt boundaries. These tests hold that on the
synthetic qwen4exp fixture (CPU backend, f32): the ladder must stay near the
tip over many turns, and a turn whose predecessor was itself restaged -- the
forced-close resubmit, which used to overwrite the only useful checkpoint -- must
still find one.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model
from tests.qwen4exp_gguf_fixture import build_qwen4exp_gguf

OPENING = [(t * 37 + 11) % 96 for t in range(16)]
CONTINUATION = 4
CONTEXT = 512
INTERVAL = 8
TURNS = 10
TURN_TOKENS = 12


def _turn(index: int, length: int = TURN_TOKENS) -> list[int]:
    """A turn's worth of tokens, distinct from every other turn's."""
    return [(index * 7 + t * 13 + 5) % 96 for t in range(length)]


class CheckpointLadderTests(unittest.TestCase):
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

    def _generate(self, runtime, prompt: list[int]) -> list[int]:
        out: list[int] = []
        runtime.generate(
            prompt, CONTINUATION,
            lambda t: (out.append(t) or len(out) < CONTINUATION))
        return out

    def test_the_ladder_follows_the_tip_over_a_long_conversation(self) -> None:
        # Run a conversation out, then rewrite its last turn -- what an agentic
        # client does every time it re-renders the reply it just received.
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT,
                    prefill_checkpoint_interval=INTERVAL) as runtime:
                runtime.prepare()
                prompt = list(OPENING)
                self._generate(runtime, prompt)
                for turn in range(TURNS):
                    prompt = prompt + _turn(turn)
                    self._generate(runtime, prompt)
                tip = len(prompt)
                # Diverge one turn back, inside the region the ladder must
                # cover: the same conversation with a different last turn.
                rewritten = prompt[:tip - TURN_TOKENS] + _turn(TURNS + 1)
                self._generate(runtime, rewritten)
                reused = int(runtime.info["prefix_cache_last_reused_tokens"])
        # A frozen ladder would fall back to a checkpoint from the FIRST prompt
        # (at most len(OPENING) here) and reprefill the whole conversation.
        self.assertGreater(
            reused, len(OPENING),
            "the rewritten turn fell back past the whole conversation")
        self.assertLessEqual(reused, tip - TURN_TOKENS)
        # One turn of spacing around the tip, not the length of the session.
        self.assertGreaterEqual(reused, tip - 3 * TURN_TOKENS)

    def test_a_restaged_turn_does_not_evict_its_own_boundary(self) -> None:
        # A forced thinking-block close resubmits prompt+generated as a fresh
        # prompt mid-turn. That restage takes a checkpoint of its own, past the
        # point the client will re-render -- so it must not be the checkpoint
        # that replaces the turn's real prompt boundary.
        with V2Model(str(self.path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT,
                    prefill_checkpoint_interval=INTERVAL) as runtime:
                runtime.prepare()
                prompt = list(OPENING)
                self._generate(runtime, prompt)
                for turn in range(TURNS):
                    prompt = prompt + _turn(turn)
                    self._generate(runtime, prompt)
                boundary = len(prompt)
                # The restage: everything decoded so far, resubmitted.
                restaged = prompt + _turn(TURNS, 6)
                self._generate(runtime, restaged)
                # The next turn re-renders what the restage produced, so it
                # diverges above the boundary but below the restage's own tail.
                nxt = restaged[:boundary + 2] + _turn(TURNS + 2)
                self._generate(runtime, nxt)
                reused = int(runtime.info["prefix_cache_last_reused_tokens"])
        self.assertGreaterEqual(
            reused, boundary - TURN_TOKENS,
            "the restage overwrote the turn's own prompt boundary")


if __name__ == "__main__":
    unittest.main()
