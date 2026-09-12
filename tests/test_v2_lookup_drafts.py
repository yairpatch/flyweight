"""Prompt-lookup drafting must be invisible in the output.

FLYWEIGHT_LOOKUP_DRAFTS=N drafts from the sequence's own history (n-gram
match) instead of a draft block, and puts the drafts through the same
verify / accept / rollback round as MTP. A checkpoint with no draft block --
UD-IQ1_S is one -- is exactly where it applies, so the fixture here is the
plain qwen4exp one. The prompt repeats itself so the history offers matches;
the fixture's random weights reject most of them, which is what the rollback
path (PLE ring + DeltaNet snapshot) needs exercised.

CPU backend, synthetic fixture, no real checkpoint needed.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flyweight.v2 import V2Model
from tests.qwen4exp_gguf_fixture import build_qwen4exp_gguf

PHRASE = [3, 9, 17, 4, 21, 33, 8, 12]
PROMPT = PHRASE + [5, 6] + PHRASE + [5, 6] + PHRASE
GENERATED = 40


class Qwen4ExpLookupDraftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen4exp-plain.gguf"
        cls.spec = build_qwen4exp_gguf(cls.path, mtp=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def setUp(self) -> None:
        V2Model.select_backend("cpu")
        self.addCleanup(V2Model.select_backend, "auto")

    @contextlib.contextmanager
    def _environment(self, drafts: int):
        # patch.dict restores the whole mapping on exit, so popping inside it
        # is how "unset" is expressed.
        with mock.patch.dict(os.environ, {"FLYWEIGHT_LOOKUP_NGRAM_MIN": "1"}):
            if drafts:
                os.environ["FLYWEIGHT_LOOKUP_DRAFTS"] = str(drafts)
            else:
                os.environ.pop("FLYWEIGHT_LOOKUP_DRAFTS", None)
            yield

    def _generate(self, drafts: int) -> tuple[list[int], dict]:
        with self._environment(drafts):
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(context_limit=256) as runtime:
                    runtime.prepare()
                    tokens: list[int] = []
                    runtime.generate(PROMPT, GENERATED, tokens.append)
                    return tokens, runtime.info

    def _engine_tokens(self, drafts: int, **sampling) -> tuple[list[int], dict]:
        with self._environment(drafts):
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(context_limit=256) as runtime:
                    runtime.prepare()
                    task_id = runtime.task_submit(PROMPT, GENERATED, **sampling)
                    tokens: list[int] = []
                    for _ in range(4096):
                        for event_task, token, kind in runtime.engine_step():
                            if event_task != task_id:
                                continue
                            if kind == 0:
                                tokens.append(token)
                            elif kind == 1:
                                return tokens, runtime.info
                            elif kind == 2:
                                raise AssertionError(
                                    "engine task failed: " + runtime.task_error(task_id)
                                )
        raise AssertionError("engine task did not finish")

    def test_no_draft_block_is_not_required(self) -> None:
        _, info = self._generate(4)
        self.assertEqual(info["mtp_available"], 0)

    def test_greedy_generate_matches_plain_decode(self) -> None:
        plain, plain_info = self._generate(0)
        self.assertEqual(len(plain), GENERATED)
        self.assertEqual(int(plain_info["mtp_draft_tokens"]), 0)
        for drafts in (1, 2, 4, 7):
            with self.subTest(drafts=drafts):
                drafted, info = self._generate(drafts)
                self.assertEqual(plain, drafted)
                self.assertGreater(int(info["mtp_draft_tokens"]), 0)

    def test_engine_greedy_with_penalty_matches(self) -> None:
        sampling = dict(repetition_penalty=1.1, penalty_window=64, forbid_tool_calls=True)
        plain, _ = self._engine_tokens(0, **sampling)
        drafted, info = self._engine_tokens(4, **sampling)
        self.assertEqual(len(plain), GENERATED)
        self.assertGreater(int(info["mtp_draft_tokens"]), 0)
        self.assertEqual(drafted, plain)

    def test_engine_sampled_matches_seed_for_seed(self) -> None:
        sampling = dict(
            temperature=0.8, top_k=20, top_p=0.95, seed=1234,
            repetition_penalty=1.1, forbid_tool_calls=True,
        )
        plain, _ = self._engine_tokens(0, **sampling)
        drafted, info = self._engine_tokens(4, **sampling)
        self.assertGreater(int(info["mtp_draft_tokens"]), 0)
        self.assertEqual(drafted, plain)
        other, _ = self._engine_tokens(4, **{**sampling, "seed": 99})
        self.assertNotEqual(other, plain)


if __name__ == "__main__":
    unittest.main()
