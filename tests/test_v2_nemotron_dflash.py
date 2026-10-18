"""The DFlash + LiLiCorr drafter beside Nemotron H: the GPU drafter against a
float64 reference, its context injection through every forward path, and the
speculative rounds it feeds."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from flyweight.v2 import V2Error, V2Model

from tests.dflash_gguf_fixture import DflashSpec, ReferenceDrafter, build_dflash_gguf
from tests.nemotron_gguf_fixture import NemotronSpec, build_nemotron_gguf


_WORKSPACES: list[tempfile.TemporaryDirectory] = []
_MODELS: list[V2Model] = []

# A prompt longer than the fixture's 16-position window, and a second that
# wraps the 32-slot context ring.
PROMPT = [40, 72, 5, 33, 64, 12, 7, 9, 21, 44, 3, 17, 60, 61, 2, 8, 30, 31, 50, 11,
          70, 71, 4, 6]
LONG_PROMPT = PROMPT + [13, 14, 15, 52, 53, 54, 80, 81, 82, 90, 91, 19, 20, 22, 23, 24,
                        25, 26, 27, 28, 29]


def tearDownModule():
    for model in _MODELS:
        model.close()
    _MODELS.clear()
    for holder in _WORKSPACES:
        holder.cleanup()
    _WORKSPACES.clear()


def _pair(drafter: bool = True):
    holder = tempfile.TemporaryDirectory(prefix="flyweight-dflash-")
    _WORKSPACES.append(holder)
    target = Path(holder.name) / "nemotron_h_moe.gguf"
    spec, target_weights = build_nemotron_gguf(target, NemotronSpec(tied_head=True))
    dspec = DflashSpec(spec.hidden)
    draft = Path(holder.name) / "dflash.gguf"
    draft_weights = build_dflash_gguf(draft, dspec)
    model = V2Model(target, mtp_model=draft if drafter else None)
    _MODELS.append(model)
    return model, dspec, draft_weights, target_weights, draft


def _native(model: V2Model, **options):
    if not V2Model.gpu_info()["available"]:
        raise unittest.SkipTest("native CUDA runtime is unavailable")
    options.setdefault("context_limit", 128)
    options.setdefault("expert_mode", "cpu")
    runtime = model.native_runtime(**options)
    runtime.prepare()
    return runtime


def _decode_prompt(runtime, prompt, reference: ReferenceDrafter | None = None) -> int:
    token = None
    for tok in prompt:
        token = runtime.decode(tok)
        if reference is not None:
            features, position = runtime.captured_features()
            reference.inject(features.astype(np.float64), [position])
    return token


class DflashDrafterTests(unittest.TestCase):
    def _assert_matches(self, probe, expected):
        scale = np.abs(expected["logits"]).max()
        np.testing.assert_allclose(probe["logits"], expected["logits"], rtol=0, atol=1e-5 * scale)
        np.testing.assert_array_equal(probe["candidates"], expected["candidates"])
        np.testing.assert_allclose(probe["scores"], expected["scores"], rtol=0, atol=1e-4)
        self.assertEqual(probe["drafts"], expected["drafts"])

    def test_draft_matches_the_float64_reference(self):
        # Every injected row comes from a single-token decode, so the
        # reference sees exactly the features the drafter does; the window
        # edge (16) is inside the first prompt, the ring wrap (32) inside the
        # second.
        for prompt in (PROMPT, LONG_PROMPT):
            with self.subTest(length=len(prompt)):
                model, dspec, dw, tw, _ = _pair()
                runtime = _native(model)
                try:
                    reference = ReferenceDrafter(dspec, dw, tw)
                    token = _decode_prompt(runtime, prompt, reference)
                    probe = runtime.dflash_probe(token)
                    self.assertIsNotNone(probe)
                    self.assertEqual(len(probe["drafts"]), 3)
                    self._assert_matches(probe, reference.draft(token, len(prompt), 3))
                finally:
                    runtime.close()

    def _reference_after_decode(self, model, dspec, dw, tw, tokens, token):
        # The reference fed with features captured token by token: the rows
        # forward computes the same features to its summation order.
        runtime = _native(model)
        try:
            reference = ReferenceDrafter(dspec, dw, tw)
            self.assertEqual(_decode_prompt(runtime, tokens, reference), token)
        finally:
            runtime.close()
        return reference.draft(token, len(tokens), 3)

    def _assert_close(self, probe, expected):
        # The rows forward and the rounds' verify-and-fold sum in another
        # order than decode, so a few target features differ in the fourth
        # digit; where one crosses a Q8 rounding step in the drafter's fc the
        # toy drafter's logits move by ~1.5% (one code; measured 0.41 at a
        # scale of 27). A stale or missing context row moves them as a whole.
        scale = np.abs(expected["logits"]).max()
        np.testing.assert_allclose(probe["logits"], expected["logits"], rtol=0, atol=0.03 * scale)

    def test_prefill_injection_matches_the_reference(self):
        # A batched prefill injects its rows from the rows forward, keeping
        # only the window's worth.
        model, dspec, dw, tw, _ = _pair()
        runtime = _native(model)
        try:
            first: list[int] = []
            runtime.generate(LONG_PROMPT, 1, first.append)
            probe = runtime.dflash_probe(first[0])
        finally:
            runtime.close()
        self._assert_close(probe, self._reference_after_decode(
            model, dspec, dw, tw, LONG_PROMPT, first[0]))

    def test_rounds_keep_the_greedy_output(self):
        plain_model, *_ = _pair(drafter=False)
        model, *_ = _pair()
        outputs = []
        for m in (plain_model, model):
            runtime = _native(m)
            try:
                tokens: list[int] = []
                runtime.generate(PROMPT, 40, tokens.append)
                outputs.append((tokens, dict(runtime.info)))
            finally:
                runtime.close()
        (plain, _), (drafted, info) = outputs
        self.assertEqual(drafted, plain)
        self.assertGreater(info["mtp_draft_tokens"], 0)

    def test_context_after_rounds_matches_the_reference(self):
        # Rounds inject verify rows that may be rejected; the ring must end up
        # holding the accepted sequence's rows and nothing past them.
        model, dspec, dw, tw, _ = _pair()
        runtime = _native(model)
        try:
            produced: list[int] = []
            runtime.generate(PROMPT, 30, produced.append)
            self.assertGreater(dict(runtime.info)["mtp_rejected_tokens"], 0)
            probe = runtime.dflash_probe(produced[-1])
        finally:
            runtime.close()
        self._assert_close(probe, self._reference_after_decode(
            model, dspec, dw, tw, PROMPT + produced[:-1], produced[-1]))

    def test_reset_forgets_the_context(self):
        model, *_ = _pair()
        runtime = _native(model)
        try:
            token = _decode_prompt(runtime, PROMPT)
            self.assertIsNotNone(runtime.dflash_probe(token))
            runtime.reset()
            # No anchor row: the drafter declines rather than draft from
            # another conversation's context.
            self.assertIsNone(runtime.dflash_probe(token))
        finally:
            runtime.close()


class DflashAttachTests(unittest.TestCase):
    def test_refuses_a_target_it_was_not_built_for(self):
        from tests.dense_gguf_fixture import build_dense_qwen35_gguf
        _, _, _, _, draft = _pair()
        holder = tempfile.TemporaryDirectory(prefix="flyweight-dflash-")
        _WORKSPACES.append(holder)
        target = Path(holder.name) / "dense.gguf"
        build_dense_qwen35_gguf(target)
        with self.assertRaisesRegex(V2Error, "Nemotron H"):
            V2Model(target, mtp_model=draft)

    def test_drafter_follows_the_draft_count(self):
        model, *_ = _pair()
        runtime = _native(model, mtp_drafts=2)
        try:
            token = _decode_prompt(runtime, PROMPT)
            self.assertEqual(len(runtime.dflash_probe(token)["drafts"]), 2)
        finally:
            runtime.close()
