"""The Q8 group-decode LM head on the paths that used to reconstruct in float.

The single-token decode quantizes the activation to Q8 blocks and runs the
group-decode head; the MTP draft step and the batched multi-sequence decode
still ran the reconstruct-in-float argmax kernel. A K-quant pack gives the
head a type that has a Q8 kernel (Q6_K here, which is also what the HF
packer keeps the head at under Q4_K), so these runs exercise the new
dispatch end to end and pin the outputs against the paths that never
changed: the non-drafting engine, and one task decoding alone.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flyweight.v2 import V2Model
from tests import qwen35_hf_fixture as fixture

import pytest

PROMPTS = ([3, 9, 17, 4, 21, 33, 8, 12], [5, 40, 2, 19, 7])
GENERATED = 24


class _Q6KFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen35"
        fixture.build(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _runtime(self, **options):
        V2Model.select_backend("cpu")
        model = V2Model(str(self.path))
        runtime = model.native_qwen_runtime(context_limit=256, **options)
        runtime.prepare()
        return model, runtime

    @staticmethod
    def _drain(runtime, task_ids: dict[int, list[int]]) -> None:
        pending = set(task_ids)
        for _ in range(8192):
            for event_task, token, kind in runtime.engine_step():
                if event_task not in task_ids:
                    continue
                if kind == 0:
                    task_ids[event_task].append(token)
                elif kind == 1:
                    pending.discard(event_task)
                elif kind == 2:
                    raise AssertionError(
                        "engine task failed: " + runtime.task_error(event_task)
                    )
            if not pending:
                return
        raise AssertionError("engine tasks did not finish")


class DraftQ8HeadTests(_Q6KFixture):
    def _engine(self, mtp_drafts: int, **sampling) -> tuple[list[int], dict]:
        with mock.patch.dict(os.environ, {
            "FLYWEIGHT_HF_QUANT": "Q6_K", "FLYWEIGHT_MTP_ADAPTIVE": "0",
        }):
            model, runtime = self._runtime(mtp_drafts=mtp_drafts)
            try:
                task_id = runtime.task_submit(PROMPTS[0], GENERATED, **sampling)
                tokens: dict[int, list[int]] = {task_id: []}
                self._drain(runtime, tokens)
                return tokens[task_id], runtime.info
            finally:
                runtime.close()
                model.close()

    @pytest.mark.slow
    def test_q6k_head_drafts_and_matches_the_one_token_path(self) -> None:
        # FLYWEIGHT_IQ2_Q8_DECODE is latched once per process, so this file
        # runs the default (Q8) head only; the float fallback is the kernel
        # the rows verifier and single-token decode have always used.
        plain, plain_info = self._engine(0, repetition_penalty=1.1, forbid_tool_calls=True)
        drafted, info = self._engine(4, repetition_penalty=1.1, forbid_tool_calls=True)
        self.assertEqual(int(plain_info["mtp_draft_tokens"]), 0)
        self.assertGreater(int(info["mtp_draft_tokens"]), 0)
        self.assertEqual(len(plain), GENERATED)
        self.assertEqual(drafted, plain)


class MultiDecodeQ8HeadTests(unittest.TestCase):
    """The batched decode's head on a K-quant fixture, against solo decodes.

    The MoE variant of the dense fixture is what engages the batched driver
    (routed experts on the CPU, two slots); packed to Q6_K its head has a Q8
    group-decode kernel, which is the new dispatch in the batched tail.
    """

    TOKENS = 12

    @classmethod
    def setUpClass(cls) -> None:
        from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf

        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.q6_path = root / "moe-q6k.gguf"
        spec = DenseQwenSpec(experts=8, experts_used=2)
        cls._spec = build_dense_qwen35_gguf(cls.q6_path, spec, quantize="q6_k")
        cls._prompts = (
            [(t * 7 + 3) % spec.vocabulary for t in range(40)],
            [(t * 11 + 5) % spec.vocabulary for t in range(33)],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _runtime(self, path, **options):
        V2Model.select_backend("cpu")
        model = V2Model(str(path))
        runtime = model.native_qwen_runtime(
            context_limit=256, expert_mode="cpu", **options
        )
        runtime.prepare()
        return model, runtime

    def _solo(self, path, prompt: list[int]) -> list[int]:
        model, runtime = self._runtime(path)
        try:
            task_id = runtime.task_submit(prompt, self.TOKENS)
            tokens: dict[int, list[int]] = {task_id: []}
            _Q6KFixture._drain(runtime, tokens)
            return tokens[task_id]
        finally:
            runtime.close()
            model.close()

    def test_q6k_pack_matches_the_runtime_decoder(self) -> None:
        # The packer's layout check: the runtime's matvec over the packed head
        # must equal a matvec over an independent decode of the same bytes,
        # written from the nibble and bit-pair placement qwen_q6_value uses.
        import ctypes

        import numpy as np

        from flyweight.v2 import _library

        def decode(packed: np.ndarray, count: int) -> np.ndarray:
            out = np.zeros(count, dtype=np.float64)
            for absolute in range(count):
                block, within = divmod(absolute, 256)
                base = block * 210
                ql = packed[base : base + 128]
                qh = packed[base + 128 : base + 192]
                scales = packed[base + 192 : base + 208].astype(np.int8)
                d = float(np.frombuffer(packed[base + 208 : base + 210].tobytes(),
                                        dtype=np.float16)[0])
                half, offset = divmod(within, 128)
                lane, slot = divmod(offset, 32)
                qindex = slot + (0 if lane in (0, 2) else 32)
                qbyte = int(ql[half * 64 + qindex])
                high = int(qh[half * 32 + slot])
                nibble = (qbyte & 15) if lane in (0, 1) else (qbyte >> 4)
                quant = (nibble | (((high >> (lane * 2)) & 3) << 4)) - 32
                out[absolute] = d * float(scales[half * 8 + slot // 16 + lane * 2]) * quant
            return out

        V2Model.select_backend("cpu")
        library = _library()
        with V2Model(str(self.q6_path)) as model:
            info = model.tensor("output.weight")
            self.assertEqual(info["ggml_type"], 14)
            inputs, outputs = int(info["shape"][0]), int(info["shape"][1])
            rng = np.random.default_rng(3)
            x = rng.standard_normal(inputs).astype(np.float32)
            y = np.zeros(outputs, dtype=np.float32)
            status = library.flyweight_v2_matvec(
                model._handle, b"output.weight",
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), inputs,
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), outputs,
            )
            self.assertEqual(status, 0)
            raw = np.frombuffer(
                model.read_tensor_slice("output.weight", 0, int(info["size"])),
                dtype=np.uint8,
            )
        weights = decode(raw, inputs * outputs).reshape(outputs, inputs)
        reference = weights @ x.astype(np.float64)
        self.assertLess(float(np.abs(y - reference).max()), 1e-4 * max(1.0, float(np.abs(reference).max())))

    @pytest.mark.slow
    def test_two_sequences_batched_match_their_solo_decodes(self) -> None:
        solo = [self._solo(self.q6_path, prompt) for prompt in self._prompts]
        model, runtime = self._runtime(self.q6_path, parallel_sequences=2)
        try:
            tokens = {
                runtime.task_submit(prompt, self.TOKENS): []
                for prompt in self._prompts
            }
            _Q6KFixture._drain(runtime, tokens)
            info = runtime.info
            batched = list(tokens.values())
        finally:
            runtime.close()
            model.close()
        # The batched driver has to have been the path taken, or this
        # compares the single-token decode with itself.
        self.assertGreater(int(info["multi_decode_batches"]), 0)
        self.assertEqual(batched, solo)


if __name__ == "__main__":
    unittest.main()
