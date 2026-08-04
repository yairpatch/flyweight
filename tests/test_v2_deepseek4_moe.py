"""Expert routing and the clamped SwiGLU.

The routing shape is taken from the reference graph rather than from the
metadata, which points the wrong way. `expert_gating_func` is 4 and the
DeepSeek-V3 lineage uses a sigmoid, but the graph reads:

    ffn_moe_logits  = ffn_gate_inp @ ffn_norm
    ffn_moe_probs   = SQRT(SOFTPLUS(logits))
    probs_biased    = probs + exp_probs_b.bias      (routed blocks only)
    topk            = ARGSORT(probs_biased)[:6]
    weights         = GET_ROWS(probs, topk)         -- unbiased
    weights_norm    = weights / CLAMP(SUM_ROWS(weights))
    weights_scaled  = weights_norm * 1.5

Two details that are easy to get backwards: the bias steers selection but not
the weights, which are gathered from the unbiased probabilities; and the
normalization comes before the scale, so a token's weights sum to
`expert_weights_scale` rather than to one.

Blocks 0 through 2 are hash layers and skip selection entirely --
`ffn_moe_topk = GET_ROWS(ffn_gate_tid2eid, inp_tokens)` reads the expert ids
from an int32 table by token id -- but weight them the same way.

These are unit checks. End-to-end router parity needs a composed layer to
produce a real `ffn_norm`, which does not exist yet.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from colibri_next.deepseek4 import route, swiglu
from colibri_next.v2 import V2Model

CHECKPOINT = os.environ.get("DEEPSEEK4_GGUF")


def _probabilities(logits: np.ndarray) -> np.ndarray:
    return np.sqrt(np.log1p(np.exp(logits.astype(np.float64))))


class RouterTests(unittest.TestCase):
    def test_weights_sum_to_the_configured_scale(self):
        rng = np.random.default_rng(1)
        logits = rng.standard_normal(256).astype(np.float32)
        _, weights = route(logits, used=6, weight_scale=1.5)
        # The reference normalizes and then scales, so 1.5 rather than 1.
        self.assertAlmostEqual(float(weights.sum()), 1.5, places=5)

    def test_selection_takes_the_largest_probabilities(self):
        rng = np.random.default_rng(2)
        logits = rng.standard_normal(64).astype(np.float32)
        chosen, _ = route(logits, used=6, weight_scale=1.0)
        expected = np.argsort(-_probabilities(logits))[:6]
        self.assertEqual(sorted(chosen.tolist()), sorted(expected.tolist()))

    def test_the_bias_steers_selection_but_not_the_weights(self):
        rng = np.random.default_rng(3)
        logits = rng.standard_normal(32).astype(np.float32)
        bias = np.zeros(32, dtype=np.float32)
        # Push one expert that would not otherwise be chosen.
        plain, _ = route(logits, used=4, weight_scale=1.0)
        outsider = next(e for e in range(32) if e not in plain.tolist())
        bias[outsider] = 100.0
        chosen, weights = route(logits, bias, used=4, weight_scale=1.0)
        self.assertIn(outsider, chosen.tolist())
        # Its weight comes from the unbiased probability, so the huge bias does
        # not make it dominate.
        probabilities = _probabilities(logits)
        share = weights[chosen.tolist().index(outsider)]
        expected = probabilities[outsider] / sum(probabilities[e] for e in chosen)
        self.assertAlmostEqual(float(share), float(expected), places=5)
        self.assertLess(float(share), 1.0)

    def test_hash_layers_use_the_ids_they_are_given(self):
        rng = np.random.default_rng(4)
        logits = rng.standard_normal(256).astype(np.float32)
        table = np.array([7, 3, 200, 41, 5, 99], dtype=np.int32)
        chosen, weights = route(logits, used=6, weight_scale=1.5, experts=table)
        np.testing.assert_array_equal(chosen, table)
        self.assertAlmostEqual(float(weights.sum()), 1.5, places=5)
        probabilities = _probabilities(logits)
        expected = probabilities[table] / probabilities[table].sum() * 1.5
        np.testing.assert_allclose(weights, expected, rtol=1e-5)

    def test_an_out_of_range_expert_id_is_rejected(self):
        logits = np.zeros(16, dtype=np.float32)
        with self.assertRaises(Exception):
            route(logits, used=2, experts=np.array([0, 99], dtype=np.int32))

    def test_softplus_does_not_overflow_on_large_logits(self):
        logits = np.full(8, 200.0, dtype=np.float32)
        logits[0] = 400.0
        _, weights = route(logits, used=4, weight_scale=1.0)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)


class ClampedSwigluTests(unittest.TestCase):
    def test_both_halves_are_clamped_before_combining(self):
        gate = np.array([-50.0, 0.5, 50.0], dtype=np.float32)
        up = np.array([50.0, -0.5, -50.0], dtype=np.float32)
        limit = 10.0
        result = swiglu(gate, up, limit)
        clamped_gate = np.clip(gate, -limit, limit)
        clamped_up = np.clip(up, -limit, limit)
        expected = (clamped_gate / (1.0 + np.exp(-clamped_gate))) * clamped_up
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_within_the_limit_it_is_plain_swiglu(self):
        rng = np.random.default_rng(6)
        gate = rng.standard_normal(64).astype(np.float32)
        up = rng.standard_normal(64).astype(np.float32)
        expected = (gate / (1.0 + np.exp(-gate))) * up
        np.testing.assert_allclose(swiglu(gate, up, 10.0), expected, rtol=1e-5)


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class CheckpointRoutingTests(unittest.TestCase):
    def setUp(self):
        self.model = V2Model(CHECKPOINT)
        self.addCleanup(self.model.close)

    def test_the_hash_table_holds_one_expert_set_per_token(self):
        info = self.model.tensor("blk.0.ffn_gate_tid2eid.weight")
        used = int(self.model.config["expert_used_count"])
        self.assertEqual(
            tuple(int(d) for d in info["shape"]),
            (used, int(self.model.config["vocabulary_size"])),
        )

    def test_routed_blocks_carry_a_bias_and_hash_blocks_do_not(self):
        names = {str(tensor["name"]) for tensor in self.model.tensors()}
        hash_layers = int(self.model.config["hash_layer_count"])
        for layer in (0, hash_layers - 1):
            self.assertIn(f"blk.{layer}.ffn_gate_tid2eid.weight", names)
            self.assertNotIn(f"blk.{layer}.exp_probs_b.bias", names)
        for layer in (hash_layers, hash_layers + 1):
            self.assertNotIn(f"blk.{layer}.ffn_gate_tid2eid.weight", names)
            self.assertIn(f"blk.{layer}.exp_probs_b.bias", names)


if __name__ == "__main__":
    unittest.main()
