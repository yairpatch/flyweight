"""MTP rejection rollback: the ReplaySSM-style fold against the full replay.

A rejected draft round used to re-run the whole forward over the accepted
prefix just to rebuild the DeltaNet conv/recurrent state -- the only thing the
batched verify leaves wrong, since KV rows, hidden states and verified tokens
were already computed with correct inputs. The fold instead replays only the
conv and recurrence kernels from the restored snapshot over the transition
inputs the verify pass retained, which is bitwise identical by construction:
same kernels, same launch geometry, same inputs.

The synthetic checkpoint's random draft block disagrees with the target
constantly, so every generation here is dense with rejections -- exactly what
the fold path needs exercised. Both checks would have caught a fold that
drifts: token parity against the replaced path, and the native bitwise state
comparison (COLIBRI_MTP_FOLD_CHECK) that re-runs the replay after every fold
and insists on identical state bytes.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from colibri_next.v2 import V2Model
from tests import qwen35_hf_fixture as fixture

PROMPT = [3, 9, 17, 4, 21, 33, 8, 12]
GENERATED = 48


class Qwen35MtpFoldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen35"
        fixture.build(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _generate(self, extra_env: dict[str, str]) -> tuple[list[int], dict]:
        # F32 keeps the two runs comparing the same arithmetic, and disabling
        # the adaptive gate makes every decode step a draft-and-verify round
        # instead of spending the whole short generation on calibration.
        environment = {
            "COLIBRI_HF_QUANT": "F32",
            "COLIBRI_MTP_ADAPTIVE": "0",
            **extra_env,
        }
        with mock.patch.dict(os.environ, environment):
            V2Model.select_backend("cpu")
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(
                    context_limit=256, mtp_drafts=4
                ) as runtime:
                    runtime.prepare()
                    tokens: list[int] = []
                    runtime.generate(PROMPT, GENERATED, tokens.append)
                    return tokens, runtime.info

    def test_fold_matches_full_replay(self) -> None:
        replayed, replay_info = self._generate({"COLIBRI_MTP_FOLD": "0"})
        folded, fold_info = self._generate({"COLIBRI_MTP_FOLD": "1"})
        # A run without rejections would compare the fold against nothing.
        self.assertGreater(int(replay_info["mtp_rejected_tokens"]), 0)
        self.assertGreater(int(fold_info["mtp_rejected_tokens"]), 0)
        self.assertEqual(folded, replayed)

    def test_fold_state_is_bitwise_identical(self) -> None:
        # The native check re-runs the full replay after every fold and throws
        # unless the conv and recurrent state match byte for byte.
        _, info = self._generate({"COLIBRI_MTP_FOLD_CHECK": "1"})
        self.assertGreater(int(info["mtp_rejected_tokens"]), 0)

    def test_prefill_checkpoints_survive_mtp(self) -> None:
        # --mtp-draft used to zero the prefill-checkpoint slots ("MTP manages
        # its own snapshots"), which silently turned every re-rendered turn of
        # a long conversation into a cold reprefill -- minutes per turn at 70k
        # tokens. The two never actually conflicted. This pins the fix: with
        # MTP on, a second prompt that diverges from the live tail must still
        # reuse a mid-prefill checkpoint of the shared prefix.
        environment = {"COLIBRI_HF_QUANT": "F32", "COLIBRI_MTP_ADAPTIVE": "0"}
        with mock.patch.dict(os.environ, environment):
            V2Model.select_backend("cpu")
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(
                    context_limit=256, mtp_drafts=2,
                    prefill_checkpoint_interval=8,
                ) as runtime:
                    runtime.prepare()
                    first = PROMPT * 4  # long enough to cross checkpoint targets
                    runtime.generate(first, 8, lambda token: None)
                    # Same prefix, diverging tail: exact-prefix reuse against
                    # the live sequence fails (the recurrent state has moved
                    # past the divergence), so only a checkpoint can serve it.
                    second = first[:-1] + [7, 5, 2, 11]
                    runtime.generate(second, 8, lambda token: None)
                    info = runtime.info
                    self.assertGreater(int(info["prefix_cache_hits"]), 0)
                    self.assertGreater(
                        int(info["prefix_cache_reused_tokens"]), 0)


if __name__ == "__main__":
    unittest.main()
