"""Rotary embedding, over real positions.

Earlier components were checked at position 0, where RoPE is the identity and
says nothing. This runs the ten-token prompt

    "The quick brown fox jumps over the lazy dog today"

through layer 0's KV path at every position and compares the rotated result
against the reference, which is the first check here that a position actually
moves anything.

Which RoPE applies is per layer, not global, and that is easy to get wrong: a
block whose compress ratio is zero -- layers 0 and 1 -- rotates at the model's
own `rope.freq_base` with no scaling and no YaRN, while the compressed blocks
use `attention.compress_rope_freq_base` together with YaRN. Layers 0 and 1 are
the plain case, so that is what this covers; the YaRN path is exercised once
the compressed layers exist.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from colibri_next.deepseek4 import hyper_connection, matvec, rms_norm, rope
from colibri_next.v2 import V2Model

CHECKPOINT = os.environ.get("DEEPSEEK4_GGUF")
PROMPT = "The quick brown fox jumps over the lazy dog today"

# Sums over all ten positions, layer 0.
REFERENCE = {
    "kv_nope": 15.299391,
    "kv_pe_before": 69.256844,
    "kv_pe_after": 137.333115,
    "kv": 152.632584,
}
EPSILON = 9.999999974752427e-07


def _f32(model: V2Model, name: str) -> np.ndarray:
    info = model.tensor(name)
    count = 1
    for dimension in info["shape"]:
        count *= int(dimension)
    return np.frombuffer(
        bytes(model.read_tensor_slice(name, 0, count * 4)), dtype=np.float32, count=count
    )


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class RopeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        config = cls.model.config
        cls.n_embd = int(config["hidden_size"])
        cls.head_dim = int(config["kv_lora_rank"])
        cls.rope_dim = int(config["rotary_dimension"])
        cls.freq_base = float(config["rope_freq_base"])
        hc = int(config["hyper_connection_count"])
        cls.tokens = list(cls.model.tokenize(PROMPT))

        mixer = _f32(cls.model, "blk.0.hc_attn_fn.weight").reshape((2 + hc) * hc, hc * cls.n_embd)
        scale = _f32(cls.model, "blk.0.hc_attn_scale.weight")
        base = _f32(cls.model, "blk.0.hc_attn_base.weight")
        attn_norm_weight = _f32(cls.model, "blk.0.attn_norm.weight")
        kv_norm_weight = _f32(cls.model, "blk.0.attn_kv_a_norm.weight")

        # Layer 0 has no cross-token mixing before attention, so each position's
        # KV latent can be built on its own.
        cls.latents = []
        for token in cls.tokens:
            embedding = np.asarray(
                cls.model.qwen_embedding(token, cls.n_embd), dtype=np.float32
            )
            collapsed = hyper_connection(
                np.repeat(embedding[None, :], hc, axis=0), mixer, scale, base,
                sinkhorn_iterations=int(config["sinkhorn_iterations"]),
                rms_epsilon=EPSILON, hc_epsilon=EPSILON,
            ).collapsed
            attn_norm = rms_norm(collapsed, attn_norm_weight, epsilon=EPSILON)
            latent = matvec(cls.model, "blk.0.attn_kv.weight", attn_norm, cls.head_dim)
            cls.latents.append(rms_norm(latent, kv_norm_weight, epsilon=EPSILON))
        cls.latents = np.stack(cls.latents)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_the_prompt_is_ten_tokens(self):
        self.assertEqual(len(self.tokens), 10)

    def _close(self, actual: float, expected: float, label: str, scale: float = 0.0) -> None:
        # One quantized matmul feeds these, plus the reference's own noise.
        allowed = 0.02 * max(abs(expected), abs(scale), 1.0)
        self.assertLessEqual(
            abs(actual - expected), allowed,
            msg=f"{label}: {actual} vs reference {expected}",
        )

    def test_the_unrotated_halves_match(self):
        nope = self.latents[:, : self.head_dim - self.rope_dim]
        rotating = self.latents[:, self.head_dim - self.rope_dim :]
        self._close(float(nope.sum()), REFERENCE["kv_nope"], "kv_nope")
        self._close(float(rotating.sum()), REFERENCE["kv_pe_before"], "kv_pe_before")

    def test_rotation_at_each_position_matches(self):
        rotated = np.stack([
            rope(latent, position, self.rope_dim, freq_base=self.freq_base)
            for position, latent in enumerate(self.latents)
        ])
        tail = rotated[:, self.head_dim - self.rope_dim :]
        # The rotation changes the sum from 69.26 to 137.33, so this is the
        # first assertion here that positions do anything at all.
        self._close(float(tail.sum()), REFERENCE["kv_pe_after"], "kv_pe_after")
        self._close(float(rotated.sum()), REFERENCE["kv"], "kv")

    def test_position_zero_is_the_identity(self):
        latent = self.latents[0]
        np.testing.assert_allclose(
            rope(latent, 0, self.rope_dim, freq_base=self.freq_base), latent, atol=0
        )

    def test_the_inverse_undoes_the_rotation(self):
        for position in (1, 5, 9):
            latent = self.latents[position]
            forward = rope(latent, position, self.rope_dim, freq_base=self.freq_base)
            back = rope(
                forward, position, self.rope_dim, freq_base=self.freq_base, inverse=True
            )
            np.testing.assert_allclose(back, latent, atol=2e-5)

    def test_only_the_trailing_dimensions_rotate(self):
        latent = self.latents[3]
        rotated = rope(latent, 3, self.rope_dim, freq_base=self.freq_base)
        head = self.head_dim - self.rope_dim
        np.testing.assert_allclose(rotated[:head], latent[:head], atol=0)
        self.assertFalse(np.allclose(rotated[head:], latent[head:]))


if __name__ == "__main__":
    unittest.main()
