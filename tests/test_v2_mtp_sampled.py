"""MTP under the sampler: a drafting engine task must match a non-drafting one.

The engine used to draft only for a task with no sampler features, and every
served request has some -- the tool-markup ban (or a tool grammar) plus the
default repetition penalty -- so `--mtp-drafts` never engaged under `serve`.
The round now puts each verified row through the task's own sampler and stops
at the first row whose choice is not the draft, which makes the emitted tokens
the ones the one-token path would have produced from the same logits and
sampler state. These tests pin that equality, greedy-with-penalty and sampled.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flyweight.v2 import V2Model
from tests import qwen35_hf_fixture as fixture

PROMPT = [3, 9, 17, 4, 21, 33, 8, 12]
GENERATED = 40


class Qwen35SampledMtpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen35"
        fixture.build(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _engine_tokens(self, mtp_drafts: int, **sampling) -> tuple[list[int], dict]:
        environment = {"FLYWEIGHT_HF_QUANT": "F32", "FLYWEIGHT_MTP_ADAPTIVE": "0"}
        with mock.patch.dict(os.environ, environment):
            V2Model.select_backend("cpu")
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(
                    context_limit=256, mtp_drafts=mtp_drafts
                ) as runtime:
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

    def test_penalized_greedy_task_drafts_and_matches(self) -> None:
        # The served defaults: greedy, the 1.1 repetition penalty, the markup
        # ban of a request that declared no tools.
        sampling = dict(repetition_penalty=1.1, penalty_window=64, forbid_tool_calls=True)
        plain, plain_info = self._engine_tokens(0, **sampling)
        drafted, draft_info = self._engine_tokens(4, **sampling)
        self.assertEqual(int(plain_info["mtp_draft_tokens"]), 0)
        self.assertGreater(int(draft_info["mtp_draft_tokens"]), 0)
        self.assertEqual(len(plain), GENERATED)
        self.assertEqual(drafted, plain)
        # The penalty steers this run away from the bare greedy answer, so the
        # rows were really chosen by the sampler, not the verifier's argmax.
        bare, _ = self._engine_tokens(0, forbid_tool_calls=True)
        self.assertNotEqual(plain, bare)

    def test_accepted_rows_commit_like_the_one_token_path(self) -> None:
        # The random-weight fixture's draft head rarely agrees with its target
        # (about one draft in 150), so most rounds exercise only rejection.
        # This configuration deterministically accepts a draft, which is the
        # path where the accepted row's commit has to match.
        sampling = dict(forbid_tool_calls=True)
        plain, _ = self._engine_tokens(0, **sampling)
        drafted, draft_info = self._engine_tokens(4, **sampling)
        self.assertGreaterEqual(int(draft_info["mtp_accepted_tokens"]), 1)
        self.assertEqual(drafted, plain)

    def test_sampled_task_drafts_and_matches_seed_for_seed(self) -> None:
        sampling = dict(
            temperature=0.8, top_k=20, top_p=0.95, seed=1234,
            repetition_penalty=1.1, forbid_tool_calls=True,
        )
        plain, _ = self._engine_tokens(0, **sampling)
        drafted, draft_info = self._engine_tokens(4, **sampling)
        self.assertGreater(int(draft_info["mtp_draft_tokens"]), 0)
        self.assertEqual(len(plain), GENERATED)
        self.assertEqual(drafted, plain)
        # And the sampler was really in charge: a different seed differs.
        other, _ = self._engine_tokens(4, **{**sampling, "seed": 99})
        self.assertNotEqual(other, plain)


if __name__ == "__main__":
    unittest.main()
