"""The qwen4exp nextn draft block loads, plans, and is kept out of the stack.

`build_qwen4exp_plan` had no draft-block branch at all -- the arch refused MTP
outright on the grounds that no released GGUF carried one. That was true of
UD-IQ1_S (block_count 48, zero nextn tensors) and false of UD-Q4_K_XL
(block_count 49, the block in shard 5) and of the standalone
Qwen3.8-Flash-Next-MTP GGUF. These tests pin the loading half of the fix; the
draft forward is still unimplemented and is expected to say so.

CPU backend, synthetic fixture, no real checkpoint needed.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flyweight.v2 import V2Model
from tests.qwen4exp_gguf_fixture import build_qwen4exp_gguf

PROMPT = [3, 9, 17, 4, 21, 33, 8, 12]
GENERATED = 24


class Qwen4ExpMtpPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.with_mtp = root / "qwen4exp-mtp.gguf"
        cls.without_mtp = root / "qwen4exp-plain.gguf"
        cls.spec = build_qwen4exp_gguf(cls.with_mtp, mtp=True)
        build_qwen4exp_gguf(cls.without_mtp, mtp=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def setUp(self) -> None:
        V2Model.select_backend("cpu")
        self.addCleanup(V2Model.select_backend, "auto")

    def test_draft_block_is_detected_and_excluded_from_the_stack(self):
        """block_count includes the draft block; the executed stack must not.

        `detect_mtp_layer` truncates layer_count to the draft index. If that
        regressed, the runtime would execute the draft block as layer 48 of
        the trunk -- which runs, and quietly produces wrong logits.
        """
        with V2Model(str(self.with_mtp)) as model:
            with model.native_qwen_runtime(context_limit=128) as runtime:
                info = runtime.info
                self.assertEqual(info["mtp_available"], 1)
                self.assertEqual(info["mtp_layer"], self.spec.layers)
                self.assertEqual(info["layers"], self.spec.layers)

    def test_a_checkpoint_without_a_draft_block_reports_none(self):
        """UD-IQ1_S is this case, and it is why the old refusal looked right."""
        with V2Model(str(self.without_mtp)) as model:
            with model.native_qwen_runtime(context_limit=128) as runtime:
                info = runtime.info
                self.assertEqual(info["mtp_available"], 0)
                self.assertEqual(info["layers"], self.spec.layers)

    def test_drafts_are_accepted_with_a_block_and_refused_without(self):
        """The refusal must key off the checkpoint, not the architecture.

        It used to be a blanket arch check, which was right for UD-IQ1_S and
        wrong for every file that ships a draft block.
        """
        with V2Model(str(self.with_mtp)) as model:
            with model.native_qwen_runtime(context_limit=128, mtp_drafts=2) as runtime:
                self.assertEqual(runtime.info["mtp_enabled"], 1)
                self.assertEqual(runtime.info["mtp_drafts"], 2)
        with V2Model(str(self.without_mtp)) as model:
            with self.assertRaises(Exception) as caught:
                model.native_qwen_runtime(context_limit=128, mtp_drafts=2)
        self.assertIn("no draft block", str(caught.exception))


class Qwen4ExpMtpDecodeTest(unittest.TestCase):
    """Speculative decode must be invisible in the output.

    Verify re-scores every drafted token with the target model, so under greedy
    sampling MTP-on and MTP-off have to emit the identical sequence. This is
    the only check that actually exercises the stream-space plumbing end to
    end: the draft forward, the per-stream eh_proj fusion, the cache replay in
    qwen4exp_mtp_append_pair, the widened hand-over out of the rows verify, and
    the DeltaNet fold on rejection. Any of them can be wrong in a way that
    still produces fluent text -- but not in a way that keeps it identical.

    The fixture's draft block is random, so rejections are dense, which is
    exactly what the rollback path needs exercised.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen4exp-mtp.gguf"
        cls.spec = build_qwen4exp_gguf(cls.path, mtp=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _generate(self, drafts: int) -> tuple[list[int], dict]:
        with mock.patch.dict(os.environ, {"FLYWEIGHT_MTP_ADAPTIVE": "0"}):
            V2Model.select_backend("cpu")
            try:
                with V2Model(str(self.path)) as model:
                    with model.native_qwen_runtime(
                        context_limit=256, mtp_drafts=drafts
                    ) as runtime:
                        runtime.prepare()
                        tokens: list[int] = []
                        runtime.generate(PROMPT, GENERATED, tokens.append)
                        return tokens, runtime.info
            finally:
                V2Model.select_backend("auto")

    def test_speculative_decode_matches_plain_decode(self) -> None:
        """Sweep the draft count -- one width is not enough.

        The bug this caught was invisible at drafts=1 (a single-row verify is
        just a plain decode step) and identical at 2, 3 and 4: the rollback
        restored the PLE dilated-conv ring and never replayed the accepted
        token into it, so the ring ran a token behind. Because that conv taps
        at 0/3/6/9 back, the drift did not surface until three tokens later.
        """
        plain, _ = self._generate(0)
        self.assertEqual(len(plain), GENERATED)
        for drafts in (1, 2, 3, 4):
            with self.subTest(drafts=drafts):
                drafted, info = self._generate(drafts)
                self.assertEqual(plain, drafted)
                # A run that never drafted would compare nothing.
                self.assertGreater(int(info["mtp_draft_tokens"]), 0)


class Qwen4ExpMtpSidecarTest(unittest.TestCase):
    """A trunk with no draft block gets one from a companion GGUF.

    This is the shape that matters on this box: UD-IQ1_S is the only
    Qwen3.8-Flash-Next quant that fits, and it ships `block_count 48` with zero
    nextn tensors. The standalone MTP GGUF supplies `blk.48` and nothing else
    usable -- `flyweight_v2_model_attach_mtp` appends those descriptors while
    their bytes stay in the sidecar mapping (`Tensor::source`).

    Both fixtures are built from the same seed, so the trunk's blocks are
    byte-identical between them and the sidecar contributes only the draft
    block -- exactly how the real pair relates.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.trunk = root / "trunk.gguf"
        cls.sidecar = root / "mtp.gguf"
        cls.spec = build_qwen4exp_gguf(cls.trunk, seed=9, mtp=False)
        build_qwen4exp_gguf(cls.sidecar, seed=9, mtp=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _generate(self, drafts: int, sidecar: bool) -> tuple[list[int], dict]:
        with mock.patch.dict(os.environ, {"FLYWEIGHT_MTP_ADAPTIVE": "0"}):
            V2Model.select_backend("cpu")
            try:
                with V2Model(
                    str(self.trunk),
                    mtp_model=str(self.sidecar) if sidecar else None,
                ) as model:
                    with model.native_qwen_runtime(
                        context_limit=256, mtp_drafts=drafts
                    ) as runtime:
                        runtime.prepare()
                        tokens: list[int] = []
                        runtime.generate(PROMPT, GENERATED, tokens.append)
                        return tokens, runtime.info
            finally:
                V2Model.select_backend("auto")

    def test_sidecar_supplies_the_draft_block(self) -> None:
        _, plain = self._generate(0, sidecar=False)
        self.assertEqual(plain["mtp_available"], 0)
        _, attached = self._generate(0, sidecar=True)
        self.assertEqual(attached["mtp_available"], 1)
        self.assertEqual(attached["mtp_layer"], self.spec.layers)
        # The draft block must not join the executed stack.
        self.assertEqual(attached["layers"], self.spec.layers)

    def test_sidecar_drafting_matches_the_bare_trunk(self) -> None:
        """The sidecar must change speed, never output."""
        base, _ = self._generate(0, sidecar=False)
        for drafts in (1, 2, 3):
            with self.subTest(drafts=drafts):
                drafted, info = self._generate(drafts, sidecar=True)
                self.assertEqual(base, drafted)
                self.assertGreater(int(info["mtp_draft_tokens"]), 0)


if __name__ == "__main__":
    unittest.main()
