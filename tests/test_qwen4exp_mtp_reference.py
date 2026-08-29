"""Pins the qwen4exp MTP input fusion (`mtp_input_fusion`).

There is no transformers oracle for this block -- upstream loads the qwen4exp
weights with `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`, so the module
does not exist there. The reference is transcribed from llama.cpp's
`graph_mtp` (refs/qwen4_exp/llamacpp_qwen4exp.cpp, from sglang
qwen4_exp_mtp.py), and these tests stand in for the missing oracle by pinning
the two choices that are silent when wrong.

Silent, because a wrong fusion never corrupts text: verify re-scores every
drafted token with the target model, so a broken draft costs acceptance rate
and nothing else. That is exactly the failure the Ling MTP work warned about
(plans/ling-mtp.md: "eh_proj input order ... silently only degrades
acceptance -- pin it against the reference, not by eye").

Needs no torch and no transformers.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native" / "tools"))
import qwen4exp_reference as ref  # noqa: E402

HC, HIDDEN, ROWS = 4, 8, 3
EPS = 1e-6


def _weights(seed: int = 20260828):
    rng = np.random.default_rng(seed)
    return {
        "hyper": rng.standard_normal((ROWS, HC * HIDDEN), dtype=np.float32),
        "embedding": rng.standard_normal((ROWS, HIDDEN), dtype=np.float32),
        # Baked norm weights (the GGUF stores 1 + w), so centre them on 1.
        "enorm": 1.0 + 0.1 * rng.standard_normal(HIDDEN).astype(np.float32),
        "hnorm": 1.0 + 0.1 * rng.standard_normal(HC * HIDDEN).astype(np.float32),
        "eh_proj": rng.standard_normal((2 * HIDDEN, HIDDEN), dtype=np.float32),
    }


class Qwen4ExpMtpFusionTest(unittest.TestCase):
    def test_matches_the_split_projection_identity(self):
        """concat(e, h) @ [A; B] == e @ A + h @ B, with EMBEDDING FIRST.

        The checkpoint carries two separate projections (mtp.fc_embedding,
        mtp.fc_hidden); conversion fuses them into one eh_proj by stacking
        them in that order. Reproducing the split form is what pins the
        concat order -- swapping it still runs and still produces tokens.
        """
        w = _weights()
        out = ref.mtp_input_fusion(w["hyper"], w["embedding"], w["enorm"],
                                   w["hnorm"], w["eh_proj"], HC, HIDDEN, EPS)

        fc_embedding = w["eh_proj"][:HIDDEN]     # A: the first half is the embedding
        fc_hidden = w["eh_proj"][HIDDEN:]        # B: the second half is the stream
        h_norm = ref.grouped_rms(w["hyper"], w["hnorm"], HC * HIDDEN, EPS)
        h_norm = h_norm.reshape(ROWS, HC, HIDDEN)
        e_norm = ref.grouped_rms(w["embedding"], w["enorm"], HIDDEN, EPS)

        expected = np.empty((ROWS, HC, HIDDEN), dtype=np.float32)
        for stream in range(HC):
            expected[:, stream] = e_norm @ fc_embedding + h_norm[:, stream] @ fc_hidden

        np.testing.assert_allclose(out.reshape(ROWS, HC, HIDDEN), expected,
                                   rtol=0, atol=1e-5)

    def test_swapping_the_concat_order_would_change_the_result(self):
        """Guards the test above: the two orders must not coincide."""
        w = _weights()
        out = ref.mtp_input_fusion(w["hyper"], w["embedding"], w["enorm"],
                                   w["hnorm"], w["eh_proj"], HC, HIDDEN, EPS)
        swapped = np.concatenate([w["eh_proj"][HIDDEN:], w["eh_proj"][:HIDDEN]])
        other = ref.mtp_input_fusion(w["hyper"], w["embedding"], w["enorm"],
                                     w["hnorm"], swapped, HC, HIDDEN, EPS)
        self.assertGreater(np.abs(out - other).max(), 1e-3)

    def test_hnorm_spans_the_whole_stream_row(self):
        """RMS over all hc*hidden at once, THEN split -- not a per-stream norm.

        deepseek4's MTP norms each stream separately; qwen4exp does not, and
        the forms differ only in the denominator, so a mix-up is invisible
        except as a worse acceptance rate.
        """
        w = _weights()
        out = ref.mtp_input_fusion(w["hyper"], w["embedding"], w["enorm"],
                                   w["hnorm"], w["eh_proj"], HC, HIDDEN, EPS)

        # Hand-rolled whole-row RMS, no helper.
        scale = np.sqrt(np.mean(w["hyper"] ** 2, axis=-1, keepdims=True) + EPS)
        h_norm = (w["hyper"] / scale * w["hnorm"]).reshape(ROWS, HC, HIDDEN)
        e_norm = ref.grouped_rms(w["embedding"], w["enorm"], HIDDEN, EPS)
        e_norm = np.repeat(e_norm[:, None, :], HC, axis=1)
        expected = np.concatenate([e_norm, h_norm], axis=-1) @ w["eh_proj"]

        np.testing.assert_allclose(out, expected.reshape(ROWS, HC * HIDDEN),
                                   rtol=0, atol=1e-5)

        # And the per-stream form must be measurably different, or the test
        # above proves nothing.
        per_stream = ref.grouped_rms(w["hyper"], w["hnorm"], HIDDEN, EPS)
        self.assertGreater(np.abs(per_stream - h_norm.reshape(ROWS, -1)).max(), 1e-3)

    def test_the_embedding_term_is_shared_by_every_stream(self):
        """With the streams zeroed, all hc outputs collapse to the same vector.

        The embedding is projected once and broadcast; only the stream half
        varies per stream.
        """
        w = _weights()
        out = ref.mtp_input_fusion(np.zeros((ROWS, HC * HIDDEN), dtype=np.float32),
                                   w["embedding"], w["enorm"], w["hnorm"],
                                   w["eh_proj"], HC, HIDDEN, EPS)
        streams = out.reshape(ROWS, HC, HIDDEN)
        for stream in range(1, HC):
            np.testing.assert_allclose(streams[:, 0], streams[:, stream],
                                       rtol=0, atol=1e-6)

    def test_stream_space_in_stream_space_out(self):
        """The fusion does NOT collapse: the head collapse is an output-side
        step (output_hc_*), which is why the standalone MTP GGUF ships it."""
        w = _weights()
        out = ref.mtp_input_fusion(w["hyper"], w["embedding"], w["enorm"],
                                   w["hnorm"], w["eh_proj"], HC, HIDDEN, EPS)
        self.assertEqual(out.shape, (ROWS, HC * HIDDEN))


if __name__ == "__main__":
    unittest.main()
