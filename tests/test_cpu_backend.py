"""CPU backend behaviour that is easy to break without noticing.

These are not numerics tests -- native/tests/cpu_parity_contract.cpp covers
those against the emulated corpus. These pin the *configuration* decisions,
which is where the expensive mistakes have been: the runtime silently choosing
a device code path that has no device under it, and costing 13x for it.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from flyweight.v2 import V2Error, V2Model

from tests.dense_gguf_fixture import build_dense_qwen35_gguf


class CpuBackendSelectionTests(unittest.TestCase):
    def tearDown(self) -> None:
        # The backend is process-global; leaving it on CPU would slow every
        # later test in the run.
        V2Model.select_backend("auto")

    def test_select_backend_round_trips(self):
        self.assertEqual(V2Model.select_backend("cpu"), "cpu")
        self.assertEqual(V2Model.active_backend(), "cpu")

    def test_auto_resolves_to_something_concrete(self):
        selected = V2Model.select_backend("auto")
        self.assertIn(selected, {"cuda", "cpu"})
        self.assertEqual(V2Model.active_backend(), selected)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            V2Model.select_backend("gpu-ish")


class CpuExpertPlacementTests(unittest.TestCase):
    """The default expert mode must not pick a device path on the CPU backend.

    Regression: flyweight_cpu_host_register used to return success because there
    is nothing to pin, the runtime read that as "direct expert paging is
    available", and routed experts moved onto the hybrid path -- which copies
    expert weights out of the GGUF mapping into another host buffer to no
    benefit. Serving measured 0.43 tok/s against 5.67 for the same model.

    It went unnoticed because every development benchmark passed
    expert_mode="cpu" explicitly, so the default was never exercised.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = TemporaryDirectory()
        cls.model_path = Path(cls._directory.name) / "dense.gguf"
        build_dense_qwen35_gguf(cls.model_path, quantize="q8_0")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def tearDown(self) -> None:
        V2Model.select_backend("auto")

    def _prepared_runtime_mode(self, expert_mode: str) -> str:
        model = V2Model(self.model_path)
        runtime = model.native_qwen_runtime(
            context_limit=64, expert_mode=expert_mode
        )
        runtime.prepare()
        return str(runtime.info["expert_mode"])

    def test_every_expert_mode_lands_on_cpu(self):
        V2Model.select_backend("cpu")
        # "auto" is the serving default and the one that regressed. "hybrid"
        # names a device placement outright and is downgraded rather than
        # refused, which is the lenient half of the behaviour.
        for requested in ("auto", "hybrid", "cpu"):
            with self.subTest(expert_mode=requested):
                self.assertEqual(self._prepared_runtime_mode(requested), "cpu")

    def test_strict_resident_is_refused_rather_than_downgraded(self):
        # Pre-existing validation, and the strict half of the behaviour: asking
        # to pin every expert in device memory when there is no device is a
        # mistake worth reporting, not one to silently reinterpret. Pinned here
        # so the asymmetry with "hybrid" above is a decision and not a surprise.
        V2Model.select_backend("cpu")
        with self.assertRaises(V2Error):
            self._prepared_runtime_mode("resident")


class CpuCublasFallbackTests(unittest.TestCase):
    """Enabling the cuBLAS attention path must not change CPU output.

    Regression, and an expensive one. The backend substitutes host kernels at
    launch(), so anything calling cuBLAS directly is invisible to it. Those
    entry points returned success without computing, and the runtime believed
    the work was done. Attention becomes cuBLAS-eligible at 128 tokens, so every
    sequence of 128 or more silently skipped attention -- coherent output up to
    that point, garbage after.

    Phrased as "the cuBLAS toggle is a no-op on the CPU backend" because that is
    the property the fix establishes.

    Verified to fail when the guard is removed -- though by crashing rather than
    by a clean assertion, since the unguarded path dereferences cuBLAS handles
    that were never loaded. A core dump still fails the run, and only happens if
    someone deletes the guard, but do not expect a readable message.

    Two weaker versions were tried and discarded: checking the tokens for
    degeneracy (on a small fixture the corrupted output is merely different, not
    obviously broken, so it passed with the fix removed) and calling the entry
    points directly with zeroed arguments (their null checks reject those before
    the backend check is reached).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = TemporaryDirectory()
        cls.model_path = Path(cls._directory.name) / "long.gguf"
        build_dense_qwen35_gguf(cls.model_path, quantize="q8_0")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def tearDown(self) -> None:
        os.environ.pop("FLYWEIGHT_CUBLAS_ATTENTION", None)
        V2Model.select_backend("auto")

    def _generate(self, prompt_length: int) -> list[int]:
        model = V2Model(self.model_path)
        runtime = model.native_qwen_runtime(context_limit=512, expert_mode="cpu")
        runtime.prepare()
        prompt = [(index * 7 + 3) % 64 for index in range(prompt_length)]
        produced: list[int] = []
        runtime.generate(prompt, 8, produced.append)
        return produced

    def test_cublas_attention_toggle_does_not_change_cpu_output(self):
        V2Model.select_backend("cpu")
        # 140 crosses the 128-token eligibility threshold; 100 does not, and is
        # included so a failure points at the threshold rather than at the model.
        for prompt_length in (100, 140):
            with self.subTest(prompt_length=prompt_length):
                os.environ["FLYWEIGHT_CUBLAS_ATTENTION"] = "0"
                disabled = self._generate(prompt_length)
                os.environ.pop("FLYWEIGHT_CUBLAS_ATTENTION", None)
                enabled = self._generate(prompt_length)
                self.assertEqual(
                    disabled, enabled,
                    "the cuBLAS attention path changed CPU output, so it is "
                    "being taken on a backend that cannot execute it")


if __name__ == "__main__":
    unittest.main()
