"""Multi-head latent attention projections, against the reference.

DeepSeek-V4's attention is MLA in absorbed form: the query goes through a
low-rank path, the key and value share one 512-wide latent that every head
reads, and each of those splits into a 448-wide part that skips RoPE and a
64-wide part that takes it.

Expected sums are `llama-eval-callback` output for the published UD-IQ3_XXS
checkpoint on the prompt "The", layer 0 (a ratio-0 sliding-window block, the
simplest of the three kinds). See tests/test_v2_deepseek4_hc.py for the command.

Tolerances differ by path, and the reason matters. Where a weight is f32 the two
implementations agree to about 1e-5 relative. Where it is quantized they cannot:
ggml quantizes the *activations* too before the dot product -- Q8_0 weights use
`vec_dot_type = GGML_TYPE_Q8_0`, Q6_K uses Q8_K -- while this runtime dots f32
against dequantized weights. Ours is the more precise of the two, so agreement
is bounded by the reference's own activation-quantization noise, measured here
at around 0.3%. Bit-exactness is the wrong target for these paths; the binding
acceptance test for the forward pass is token-level agreement under greedy
decoding, not matching sums.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from colibri_next.deepseek4 import hyper_connection, matvec, rms_norm
from colibri_next.v2 import V2Model

CHECKPOINT = os.environ.get("DEEPSEEK4_GGUF")

REFERENCE = {
    "attn_norm-0": -0.441551,
    "qr-0": -1.086790,
    "qr_norm-0": -0.614109,
    "q_b": -3.649877,       # node_19, before the per-head norm
    "q_norm-0": -91.963799,
    "q_nope": 29.689964,    # first 448 of each head
    "q_pe": -121.654030,    # last 64 of each head
    "kv": 0.399849,         # node_26
    "kv_norm-0": 2.031337,  # node_28
    "kv_nope": -3.260659,
    "kv_pe": 5.291998,
}
EPSILON = 9.999999974752427e-07
# Headroom for the reference's activation quantization on quantized weights.
QUANTIZED = 0.01


def _f32(model: V2Model, name: str) -> np.ndarray:
    info = model.tensor(name)
    count = 1
    for dimension in info["shape"]:
        count *= int(dimension)
    raw = model.read_tensor_slice(name, 0, count * 4)
    return np.frombuffer(bytes(raw), dtype=np.float32, count=count)


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class LatentAttentionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        config = cls.model.config
        cls.n_embd = int(config["hidden_size"])
        cls.heads = int(config["attention_heads"])
        cls.head_dim = int(config["kv_lora_rank"])       # 512
        cls.rope_dim = int(config["rotary_dimension"])   # 64
        cls.q_lora = int(config["q_lora_rank"])
        hc = int(config["hyper_connection_count"])

        token = list(cls.model.tokenize("The"))[0]
        embedding = np.asarray(cls.model.qwen_embedding(token, cls.n_embd), dtype=np.float32)
        streams = np.repeat(embedding[None, :], hc, axis=0)
        collapsed = hyper_connection(
            streams,
            _f32(cls.model, "blk.0.hc_attn_fn.weight").reshape((2 + hc) * hc, hc * cls.n_embd),
            _f32(cls.model, "blk.0.hc_attn_scale.weight"),
            _f32(cls.model, "blk.0.hc_attn_base.weight"),
            sinkhorn_iterations=int(config["sinkhorn_iterations"]),
            rms_epsilon=EPSILON,
            hc_epsilon=EPSILON,
        ).collapsed
        # The block reads the collapsed streams through its own norm.
        cls.attn_norm = rms_norm(
            collapsed, _f32(cls.model, "blk.0.attn_norm.weight"), epsilon=EPSILON
        )

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def _close(self, actual: float, expected: float, label: str, scale: float = 0.0) -> None:
        """Agree within the reference's activation-quantization noise.

        `scale` sets what the noise is measured against. It defaults to the
        expected value, but a sum over values that largely cancel is much
        smaller than the terms feeding it, and the error does not shrink with
        it -- so those pass the magnitude of the quantity they were split from.
        """
        allowed = QUANTIZED * max(abs(expected), abs(scale), 1.0)
        self.assertLessEqual(
            abs(actual - expected), allowed,
            msg=f"{label}: {actual} vs reference {expected} "
                f"(allowed {allowed}, {abs(actual - expected) / max(abs(expected), 1e-9):.2%})",
        )

    def test_block_input_norm(self):
        self.assertAlmostEqual(float(self.attn_norm.sum()), REFERENCE["attn_norm-0"], places=4)

    def _query(self):
        low_rank = matvec(self.model, "blk.0.attn_q_a.weight", self.attn_norm, self.q_lora)
        normed = rms_norm(
            low_rank, _f32(self.model, "blk.0.attn_q_a_norm.weight"), epsilon=EPSILON
        )
        wide = matvec(
            self.model, "blk.0.attn_q_b.weight", normed, self.heads * self.head_dim
        )
        return low_rank, normed, wide.reshape(self.heads, self.head_dim)

    def test_query_low_rank_path(self):
        low_rank, normed, wide = self._query()
        for actual, key in (
            (low_rank.sum(), "qr-0"),
            (normed.sum(), "qr_norm-0"),
            (wide.sum(), "q_b"),
        ):
            self._close(float(actual), REFERENCE[key], key)

    def test_per_head_query_norm_has_no_gain(self):
        _, _, wide = self._query()
        # Each head's 512 values are normalized on their own, with no weight.
        normed = rms_norm(wide, None, epsilon=EPSILON)
        self._close(float(normed.sum()), REFERENCE["q_norm-0"], "q_norm-0")
        nope = normed[:, : self.head_dim - self.rope_dim]
        rope = normed[:, self.head_dim - self.rope_dim :]
        # The two halves nearly cancel, so both are measured against the
        # magnitude of the RoPE half they were split from.
        scale = abs(REFERENCE["q_pe"])
        self._close(float(nope.sum()), REFERENCE["q_nope"], "q_nope", scale)
        self._close(float(rope.sum()), REFERENCE["q_pe"], "q_pe", scale)
        # Whatever the tolerance, the split itself must be exact.
        self.assertEqual(nope.shape[1] + rope.shape[1], self.head_dim)
        self.assertAlmostEqual(
            float(nope.sum() + rope.sum()), float(normed.sum()), places=3
        )

    def test_the_kv_latent_is_shared_by_every_head(self):
        latent = matvec(self.model, "blk.0.attn_kv.weight", self.attn_norm, self.head_dim)
        self._close(float(latent.sum()), REFERENCE["kv"], "kv")
        normed = rms_norm(
            latent, _f32(self.model, "blk.0.attn_kv_a_norm.weight"), epsilon=EPSILON
        )
        self._close(float(normed.sum()), REFERENCE["kv_norm-0"], "kv_norm-0")
        # One latent, not one per head: 512 wide against 64 heads.
        self.assertEqual(normed.shape, (self.head_dim,))
        nope = normed[: self.head_dim - self.rope_dim]
        rope = normed[self.head_dim - self.rope_dim :]
        scale = abs(REFERENCE["kv_pe"])
        self._close(float(nope.sum()), REFERENCE["kv_nope"], "kv_nope", scale)
        self._close(float(rope.sum()), REFERENCE["kv_pe"], "kv_pe", scale)
        self.assertEqual(nope.size + rope.size, self.head_dim)


class RmsNormTests(unittest.TestCase):
    def test_rows_are_normalized_independently(self):
        rng = np.random.default_rng(9)
        values = rng.standard_normal((4, 16)).astype(np.float32) * 3.0
        result = rms_norm(values, epsilon=0.0)
        for row in range(values.shape[0]):
            expected = values[row] / np.sqrt((values[row].astype(np.float64) ** 2).mean())
            np.testing.assert_allclose(result[row], expected, rtol=1e-5)

    def test_the_gain_is_applied_after_normalizing(self):
        rng = np.random.default_rng(10)
        values = rng.standard_normal(32).astype(np.float32)
        weight = rng.standard_normal(32).astype(np.float32)
        np.testing.assert_allclose(
            rms_norm(values, weight, epsilon=0.0),
            rms_norm(values, epsilon=0.0) * weight,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
