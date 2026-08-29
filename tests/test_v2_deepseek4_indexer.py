"""The lightning indexer: scoring compressed blocks, and keeping the best.

On a 4:1 layer the compressed cache grows with the context, and past
`indexer_top_k` blocks the model is not supposed to attend to all of them. A
separate small projection scores every block and the top 512 survive; the rest
are masked out of the attention entirely.

Two properties are worth pinning here, because neither is visible on a short
prompt -- below 2048 tokens there are fewer than 512 blocks and the selection
keeps everything:

- the score is a *rectified* per-head agreement, weighted per head. Dropping the
  rectifier turns it into an ordinary attention score and reverses the ranking
  wherever a head disagrees strongly, which is exactly the case the rectifier
  exists to ignore;
- selection is over the visible blocks only, and keeping everything when there
  are fewer than k is what makes running it early equivalent to skipping it.

The scoring kernel is exposed so this can be checked against arithmetic written
out longhand, rather than only through a 2000-token generation.
"""

from __future__ import annotations

import unittest

import numpy as np

from flyweight.deepseek4 import indexer_scores, top_k_select


def reference_scores(queries, keys, weights):
    """score[j] = sum_h relu(q[h] . k[j]) * w[h], written out."""
    scores = np.zeros(len(keys), dtype=np.float64)
    for j, key in enumerate(keys):
        for h, query in enumerate(queries):
            agreement = float(np.dot(query.astype(np.float64), key.astype(np.float64)))
            if agreement > 0:
                scores[j] += agreement * float(weights[h])
    return scores.astype(np.float32)


class IndexerScoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.queries = rng.standard_normal((8, 16)).astype(np.float32)
        self.keys = rng.standard_normal((20, 16)).astype(np.float32)
        self.weights = rng.standard_normal(8).astype(np.float32)

    def test_it_matches_the_arithmetic_written_out(self):
        actual = indexer_scores(self.queries, self.keys, self.weights)
        np.testing.assert_allclose(
            actual, reference_scores(self.queries, self.keys, self.weights),
            rtol=1e-5, atol=1e-5,
        )

    def test_a_head_that_disagrees_contributes_nothing(self):
        # The rectifier is the whole difference from an attention score: a head
        # pointing away from a block must not push that block down.
        queries = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        keys = np.array([[1.0, 0.0]], dtype=np.float32)
        weights = np.array([1.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(
            float(indexer_scores(queries, keys, weights)[0]), 1.0, places=5
        )

    def test_a_block_no_head_agrees_with_scores_zero(self):
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        keys = np.array([[-1.0, 0.0]], dtype=np.float32)
        weights = np.array([1.0], dtype=np.float32)
        self.assertEqual(float(indexer_scores(queries, keys, weights)[0]), 0.0)

    def test_the_weight_scales_the_head_after_rectifying(self):
        # Scaling before the rectifier with a negative weight would flip the
        # sign of the contribution instead of dropping it.
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        keys = np.array([[2.0, 0.0]], dtype=np.float32)
        self.assertAlmostEqual(
            float(indexer_scores(queries, keys, np.array([-3.0], np.float32))[0]),
            -6.0, places=5,
        )

    def test_it_is_shape_checked(self):
        with self.assertRaises(Exception):
            indexer_scores(self.queries, self.keys, self.weights[:4])


class TopKTests(unittest.TestCase):
    def test_everything_survives_when_there_are_fewer_than_k(self):
        scores = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(top_k_select(scores, 5), [1, 1, 1])
        np.testing.assert_array_equal(top_k_select(scores, 3), [1, 1, 1])

    def test_it_keeps_the_highest(self):
        scores = np.array([3.0, 1.0, 2.0, 5.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(top_k_select(scores, 2), [1, 0, 0, 1, 0])

    def test_ties_go_to_the_lower_index(self):
        scores = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(top_k_select(scores, 2), [1, 1, 0])

    def test_keeping_none_is_allowed(self):
        scores = np.array([1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(top_k_select(scores, 0), [0, 0])


if __name__ == "__main__":
    unittest.main()
