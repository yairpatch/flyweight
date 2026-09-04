"""A prompt must mean the same whether it arrives whole or a token at a time.

Prefill and decode are different kernels for the same arithmetic. Decode walks
one row at a time; prefill batches rows so a weight is decoded once and reused
across 8 tokens (the rows kernel), 32 (tiled) or 64 (tensor-core MMQ). Nothing
else in the suite would notice if a batched kernel unpacked a weight format
wrongly -- the model would simply answer differently when a prompt was long.

Every quantization the loader can emit is checked, because the batched path is
selected per weight type and each type unpacks differently. The asymmetric
K-quants are the delicate ones: they reconstruct as `d*scale*q - dmin*min`, and
the batched kernels carry that minimum as a separate correction against the sum
of the activations rather than folding it into the dot product.

This runs on the CPU backend, where the same CUDA source is emulated, so it
needs no GPU.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model
from tests import qwen35_hf_fixture as fixture

# Long enough to pass the batched thresholds: more than the 8 rows the rows
# kernel takes in one launch, so the wider tiles are exercised too.
PROMPT = [3, 9, 17, 4, 21, 33, 8, 12, 5, 19, 27, 11]
STEPS = 2


class PrefillParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        # Four layers rather than the fixture's eight: three gated-delta and
        # one full attention is every block shape once, and this test runs the
        # CPU emulation of the CUDA corpus, where depth is the whole cost.
        cls._layers = fixture.LAYERS
        fixture.LAYERS = 4
        fixture.CONFIG["text_config"]["num_hidden_layers"] = 4
        cls.path = str(fixture.build(Path(cls._directory.name) / "qwen35"))
        V2Model.select_backend("cpu")

    @classmethod
    def tearDownClass(cls) -> None:
        fixture.LAYERS = cls._layers
        fixture.CONFIG["text_config"]["num_hidden_layers"] = cls._layers
        cls._directory.cleanup()
        # The backend selection is process-global; leaving the CPU shim
        # selected sent every later test file in the run through it.
        V2Model.select_backend("auto")

    @staticmethod
    def continuation(model: V2Model, whole: bool) -> list[int]:
        with model.native_qwen_runtime(context_limit=256) as runtime:
            runtime.prepare()
            if whole:
                out: list[int] = []
                runtime.generate(PROMPT, STEPS, out.append)
                return out
            token = 0
            for value in PROMPT:
                token = runtime.decode(value)
            out = [token]
            for _ in range(STEPS - 1):
                out.append(runtime.decode(out[-1]))
            return out

    def test_every_quantization_prefills_as_it_decodes(self) -> None:
        previous = os.environ.get("FLYWEIGHT_HF_QUANT")
        try:
            for quant in ("Q2_K", "IQ3_XXS", "Q3_K", "Q4_K", "Q5_K", "Q6_K"):
                with self.subTest(quant=quant):
                    os.environ["FLYWEIGHT_HF_QUANT"] = quant
                    # One open per quantization: the arena is what the choice
                    # changes, and packing the fixture is most of the cost.
                    with V2Model(self.path) as model:
                        self.assertEqual(self.continuation(model, True),
                                         self.continuation(model, False))
        finally:
            if previous is None:
                os.environ.pop("FLYWEIGHT_HF_QUANT", None)
            else:
                os.environ["FLYWEIGHT_HF_QUANT"] = previous


if __name__ == "__main__":
    unittest.main()
