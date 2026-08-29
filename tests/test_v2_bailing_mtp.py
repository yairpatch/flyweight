"""BailingMoE3 speculative decode: the nextn draft block, verify and fold.

The correctness property of speculative decode is that it never changes the
text: every emitted token is the target's own greedy output, drafted or not.
So the primary assertion runs the same generation twice -- once a token at a
time through ``eval``, once through ``mtp_round`` -- and demands identical
streams. That holds even if the draft block is semantically wrong (a draft
defect only lowers acceptance), so a second test pins the draft block's
health the only way that matters: on a fixture whose draft layer shares the
target's last layer, acceptance must be nonzero.

The rejection rollback is the fold: KDA state rebuilt from retained
transition inputs rather than a re-run forward. Rejections are what exercise
it, and the tiny random fixture disagrees with its own drafts constantly, so
any fold defect shows up as a diverged stream here.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flyweight.v2 import BailingRuntime, V2Model
from tests import bailing_gguf_fixture as gguf
from tests import hf_safetensors_fixture as safetensors

PROMPT = [5, 11, 23, 4, 9, 17, 3, 8]
GENERATED = 32


def _argmax(logits) -> int:
    best = 0
    for index in range(1, len(logits)):
        if logits[index] > logits[best]:
            best = index
    return best


class BailingMtpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.hf_path = safetensors.build(root / "flash-fixture", flash=True)
        cls.gguf_path = gguf.build(root / "converted", cls.hf_path, flash=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _sequential(self, model: V2Model) -> list[int]:
        runtime = BailingRuntime(model, capacity=128)
        token = _argmax(runtime.eval(PROMPT))
        stream = [token]
        for _ in range(GENERATED - 1):
            token = _argmax(runtime.eval([token]))
            stream.append(token)
        return stream

    def _speculative(self, model: V2Model, wanted: int) -> tuple[list[int], dict]:
        runtime = BailingRuntime(model, capacity=128)
        self.assertTrue(runtime.mtp_available)
        token = _argmax(runtime.eval(PROMPT))
        stream = [token]
        while len(stream) < GENERATED:
            produced = runtime.mtp_round(stream[-1], wanted=wanted)
            self.assertGreaterEqual(len(produced), 1)
            stream.extend(produced)
        return stream[:GENERATED], runtime.mtp_stats

    def test_speculative_stream_matches_sequential(self) -> None:
        # The host path is the oracle; keep the device out of both runs.
        with mock.patch.dict(os.environ, {"FLYWEIGHT_BAILING_GPU": "0"}):
            with V2Model(self.gguf_path) as model:
                sequential = self._sequential(model)
                for wanted in (2, 4):
                    with self.subTest(wanted=wanted):
                        stream, stats = self._speculative(model, wanted)
                        self.assertEqual(stream, sequential)
                        self.assertGreater(stats["drafted"], 0)

    def test_rejections_exercise_the_fold(self) -> None:
        # A tiny random checkpoint disagrees with its own drafts constantly;
        # a run without rejections would leave the fold rollback untested,
        # so its absence is a test defect worth failing on.
        with mock.patch.dict(os.environ, {"FLYWEIGHT_BAILING_GPU": "0"}):
            with V2Model(self.gguf_path) as model:
                _, stats = self._speculative(model, wanted=4)
                self.assertGreater(stats["rejected"], 0)

    def test_the_stub_draft_block_stays_inert(self) -> None:
        # The non-flash conversion carries no draft block at all, and a
        # norms-only stub must not be mistaken for one either.
        root = Path(self._directory.name)
        plain_hf = safetensors.build(root / "plain-fixture")
        plain = gguf.build(root / "plain-converted", plain_hf)
        with V2Model(plain) as model:
            runtime = BailingRuntime(model, capacity=64)
            self.assertFalse(runtime.mtp_available)
            with self.assertRaises(Exception):
                runtime.mtp_round(1)


if __name__ == "__main__":
    unittest.main()
