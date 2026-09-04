"""Hyper-connection parity against the reference implementation.

DeepSeek-V4 replaces the residual with four parallel streams. Each block reads a
weighted collapse of them and writes its output back through a Sinkhorn-balanced
mixing matrix, so getting this wrong corrupts every later component in a way
that is hard to attribute.

The expected values below are `llama-eval-callback` output for the published
UD-IQ3_XXS checkpoint on the one-token prompt "The":

    llama-eval-callback -m <shard 1> -p "The" -n 1 -c 256 --temp 0

It prints each intermediate with a `sum`, which is what these assert against.
The chain is fully determined by the checkpoint, so it reproduces exactly.

Note what each assertion is worth. The `hc_mixes` sum is the strong one: a
16384-wide dot product per output, matching to about 3e-5 relative. On this
input the pre-weight sigmoid saturates -- all four gates come out at 1.0 -- so
`hc_pre` and the collapse agreeing mostly confirms the plumbing rather than the
arithmetic. A multi-token prompt where the gates land off the rails would test
those harder, and is worth adding when the surrounding components exist to
carry one.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from flyweight.deepseek4 import hyper_connection
from flyweight.v2 import V2Model

_CHECKPOINT_PATH = os.environ.get("DEEPSEEK4_GGUF")
# A stale path is as good as no path: the variable often outlives the file it
# named, and treating that as "configured" turns a missing checkpoint into a
# wall of errors instead of a skip.
CHECKPOINT = _CHECKPOINT_PATH if _CHECKPOINT_PATH and os.path.exists(_CHECKPOINT_PATH) else None
# The expected values here were recorded from one specific build. Pointing this
# at a different quantization would compare its numbers against the wrong
# reference and report failures that mean nothing, so require the build the
# numbers came from.
if CHECKPOINT and "IQ3_XXS" not in CHECKPOINT:
    CHECKPOINT = None

# Sums the reference reported, in graph order.
REFERENCE = {
    "embd": -0.448816,
    "hc_init": -1.795269,
    "node_4": -53.225002,      # rms_norm over the flattened streams
    "hc_mixes-0": -1582.693726,
    "pre_view": 122.691177,    # mixes[0:4]
    "pre_scaled": 254.824554,  # * hc_attn_scale[0]
    "pre_biased": 252.475830,  # + hc_attn_base[0:4]
    "hc_pre-0": 4.000004,      # sigmoid, then + hc_eps
    "hc_attn_pre-0": -1.795268,
}
RMS_EPSILON = 9.999999974752427e-07
HC_EPSILON = 9.999999974752427e-07


def _tensor(model: V2Model, name: str) -> np.ndarray:
    """Read an f32 tensor, shaped as the byte stream lays it out.

    GGUF reports [input, output] while the payload is output-major, so the
    reported shape is reversed to index it.
    """
    info = model.tensor(name)
    if int(info["ggml_type"]) != 0:
        raise AssertionError(f"{name} is not f32")
    shape = tuple(int(dimension) for dimension in info["shape"])
    count = 1
    for dimension in shape:
        count *= dimension
    raw = model.read_tensor_slice(name, 0, count * 4)
    return np.frombuffer(bytes(raw), dtype=np.float32, count=count).reshape(shape[::-1])


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class HyperConnectionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        config = cls.model.config
        cls.hc = int(config["hyper_connection_count"])
        cls.n_embd = int(config["hidden_size"])
        cls.iterations = int(config["sinkhorn_iterations"])
        # The reference ran on the single token "The".
        tokens = list(cls.model.tokenize("The"))
        assert len(tokens) == 1, tokens
        # token_embd is Q6_K in this build, so take the row through the runtime
        # rather than assuming it is f32.
        cls.embedding = np.asarray(
            cls.model.qwen_embedding(tokens[0], cls.n_embd), dtype=np.float32
        )
        cls.fn = _tensor(cls.model, "blk.0.hc_attn_fn.weight")
        cls.scale = _tensor(cls.model, "blk.0.hc_attn_scale.weight")
        cls.base = _tensor(cls.model, "blk.0.hc_attn_base.weight")

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def _streams(self) -> np.ndarray:
        # hc_init replicates the embedding into every stream.
        return np.repeat(self.embedding[None, :], self.hc, axis=0)

    def test_the_embedding_row_matches(self):
        self.assertAlmostEqual(float(self.embedding.sum()), REFERENCE["embd"], places=4)

    def test_streams_start_as_copies_of_the_embedding(self):
        self.assertAlmostEqual(
            float(self._streams().sum()), REFERENCE["hc_init"], places=4
        )

    def test_mixes_pre_and_collapse_match_the_reference(self):
        result = hyper_connection(
            self._streams(),
            self.fn,
            self.scale,
            self.base,
            sinkhorn_iterations=self.iterations,
            rms_epsilon=RMS_EPSILON,
            hc_epsilon=HC_EPSILON,
        )
        self.assertAlmostEqual(
            float(result.mixes.sum()), REFERENCE["hc_mixes-0"], delta=0.05
        )
        self.assertAlmostEqual(
            float(result.mixes[: self.hc].sum()), REFERENCE["pre_view"], delta=0.01
        )
        self.assertAlmostEqual(
            float(result.pre.sum()), REFERENCE["hc_pre-0"], places=4
        )
        self.assertAlmostEqual(
            float(result.collapsed.sum()), REFERENCE["hc_attn_pre-0"], places=4
        )

    def test_the_mixing_matrix_is_balanced(self):
        result = hyper_connection(
            self._streams(),
            self.fn,
            self.scale,
            self.base,
            sinkhorn_iterations=self.iterations,
            rms_epsilon=RMS_EPSILON,
            hc_epsilon=HC_EPSILON,
        )
        # The schedule ends on a column normalization, so columns sum to one
        # exactly. Rows are only approximately balanced -- around 8% off here
        # after 20 iterations -- because each row normalization is undone by
        # the column pass that follows it.
        np.testing.assert_allclose(result.comb.sum(axis=0), 1.0, atol=1e-4)
        np.testing.assert_allclose(result.comb.sum(axis=1), 1.0, atol=0.1)


class HyperConnectionShapeTests(unittest.TestCase):
    """Behaviour that does not need the checkpoint."""

    def test_combine_writes_the_block_into_every_stream(self):
        hc, n_embd = 4, 8
        rng = np.random.default_rng(3)
        streams = rng.standard_normal((hc, n_embd)).astype(np.float32)
        fn = rng.standard_normal(((2 + hc) * hc, hc * n_embd)).astype(np.float32) * 0.1
        scale = np.ones(3, dtype=np.float32)
        base = np.zeros(2 * hc + hc * hc, dtype=np.float32)
        block = rng.standard_normal(n_embd).astype(np.float32)

        result = hyper_connection(streams, fn, scale, base, block=block)
        self.assertEqual(result.combined.shape, (hc, n_embd))
        # Reproduce the documented combine directly.
        for dst in range(hc):
            expected = block * result.post[dst]
            for src in range(hc):
                expected = expected + streams[src] * result.comb[src, dst]
            np.testing.assert_allclose(result.combined[dst], expected, atol=1e-5)

    def test_collapse_is_the_pre_weighted_sum(self):
        hc, n_embd = 4, 6
        rng = np.random.default_rng(5)
        streams = rng.standard_normal((hc, n_embd)).astype(np.float32)
        fn = rng.standard_normal(((2 + hc) * hc, hc * n_embd)).astype(np.float32) * 0.1
        result = hyper_connection(
            streams, fn, np.ones(3, dtype=np.float32),
            np.zeros(2 * hc + hc * hc, dtype=np.float32),
        )
        expected = (streams * result.pre[:, None]).sum(axis=0)
        np.testing.assert_allclose(result.collapsed, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
