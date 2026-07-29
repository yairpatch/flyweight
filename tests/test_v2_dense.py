"""Dense qwen35 support, on a synthetic checkpoint.

Dense Qwen3.5/3.6 builds replace the router, shared expert and stacked routed
experts with a single ffn_gate/ffn_up/ffn_down triple. Every published dense
checkpoint uses IQ codebook quantization the runtime does not implement, so
these tests run against the f32 fixture in dense_gguf_fixture instead.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from colibri_next.v2 import V2Model
from colibri_next.v2_qwen import QwenDenseMLPLayer, QwenV2Decoder, _decode_tensor

from dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf

# Below the fixture's weight footprint, so most dense blocks' feed-forward is
# pushed onto the host. Every test using it asserts the spill really happened:
# a budget that quietly fits would make the comparisons vacuous.
SPILL_BUDGET_BYTES = 32 * 1024 * 1024


def _model(**kwargs) -> tuple[V2Model, DenseQwenSpec, Path]:
    directory = Path(tempfile.mkdtemp(prefix="colibri-dense-"))
    path = directory / "dense.gguf"
    spec = build_dense_qwen35_gguf(path, **kwargs)
    return V2Model(path), spec, path


def _decode(model: V2Model, name: str, count: int):
    info = model.tensor(name)
    return _decode_tensor(model.view_tensor(name), info["ggml_type"], count)


def _logits(model: V2Model, hidden, spec: DenseQwenSpec):
    epsilon = float(model.config.get("rms_norm_epsilon") or 1e-6)
    vector = np.asarray(hidden, dtype=np.float32)
    weights = _decode(model, "output_norm.weight", spec.hidden)
    normalized = vector / np.sqrt(np.mean(vector * vector) + epsilon) * weights
    head = _decode(model, "output.weight", spec.hidden * spec.vocabulary)
    return head.reshape(spec.vocabulary, spec.hidden) @ normalized


def _native(model: V2Model, **options):
    try:
        runtime = model.native_runtime(context_limit=64, mtp_drafts=0, **options)
        runtime.prepare()
    except Exception as error:  # pragma: no cover - depends on the host GPU
        raise unittest.SkipTest(f"native CUDA runtime is unavailable: {error}")
    return runtime


class DensePlanTests(unittest.TestCase):
    def test_config_reports_a_dense_model_with_no_experts(self):
        model, spec, _ = _model()
        config = model.config
        self.assertEqual(config["architecture"], "qwen35")
        self.assertEqual(config["layer_count"], spec.layers)
        self.assertEqual(config["hidden_size"], spec.hidden)
        self.assertEqual(config["intermediate_size"], spec.intermediate)
        # A dense checkpoint carries no routing width at all; the runtime used
        # to reject exactly this as an incomplete config.
        self.assertEqual(config["expert_count"], 0)
        self.assertEqual(config["expert_used_count"], 0)

    def test_decoder_selects_dense_feed_forward_for_every_block(self):
        model, spec, _ = _model()
        decoder = QwenV2Decoder(model)
        self.assertTrue(decoder.dense_ffn)
        for layer in range(spec.layers):
            self.assertIsInstance(
                decoder._feed_forward(layer, cuda_only=False), QwenDenseMLPLayer
            )


class DenseReferenceTests(unittest.TestCase):
    def test_reference_decoder_runs_the_dense_stack_without_routing(self):
        model, _, _ = _model()
        decoder = QwenV2Decoder(model)
        state = decoder.new_state()
        hidden, routes = decoder.forward_token(5, state)
        self.assertEqual(routes, [])
        self.assertTrue(np.isfinite(np.asarray(hidden, dtype=np.float32)).all())

    def test_dense_block_matches_an_explicit_swiglu(self):
        model, spec, _ = _model()
        layer = QwenDenseMLPLayer(model, 0)
        rng = np.random.default_rng(3)
        vector = (rng.standard_normal(spec.hidden) * 0.1).astype(np.float32)

        epsilon = float(model.config.get("rms_norm_epsilon") or 1e-6)
        weights = _decode(model, "blk.0.post_attention_norm.weight", spec.hidden)
        normalized = vector / np.sqrt(np.mean(vector * vector) + epsilon) * weights
        gate = _decode(model, "blk.0.ffn_gate.weight", spec.hidden * spec.intermediate)
        up = _decode(model, "blk.0.ffn_up.weight", spec.hidden * spec.intermediate)
        down = _decode(model, "blk.0.ffn_down.weight", spec.intermediate * spec.hidden)
        projected_gate = gate.reshape(spec.intermediate, spec.hidden) @ normalized
        projected_up = up.reshape(spec.intermediate, spec.hidden) @ normalized
        activated = projected_gate / (1.0 + np.exp(-projected_gate)) * projected_up
        expected = vector + down.reshape(spec.hidden, spec.intermediate) @ activated

        actual = np.asarray(layer.forward_residual(vector.tolist()), dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class DenseNativeTests(unittest.TestCase):
    def test_native_decode_matches_the_reference_decoder(self):
        model, spec, _ = _model()
        for token in (5, 11, 23):
            decoder = QwenV2Decoder(model)
            hidden, _ = decoder.forward_token(token, decoder.new_state())
            expected = int(np.argmax(_logits(model, hidden, spec)))
            runtime = _native(model)
            try:
                self.assertEqual(runtime.decode(token), expected)
            finally:
                runtime.close()

    def test_host_spilled_feed_forward_matches_the_resident_one(self):
        """The offload has to be numerically invisible.

        With a budget below the weight footprint, most dense blocks' SwiGLU runs
        on the CPU straight from the mapping instead of the GPU arena.
        """
        model, _, _ = _model()
        for token in (5, 11, 23):
            resident = _native(model)
            spilled = _native(model, gpu_cache_bytes=SPILL_BUDGET_BYTES)
            try:
                self.assertEqual(resident.info["host_ffn_layers"], 0)
                self.assertGreater(spilled.info["host_ffn_layers"], 0)
                self.assertEqual(spilled.decode(token), resident.decode(token))
            finally:
                resident.close()
                spilled.close()

    def test_spilling_shrinks_the_resident_footprint(self):
        model, spec, _ = _model()
        resident = _native(model)
        spilled = _native(model, gpu_cache_bytes=SPILL_BUDGET_BYTES)
        try:
            info = spilled.info
            self.assertGreater(info["host_ffn_layers"], 0)
            self.assertLessEqual(info["host_ffn_layers"], spec.layers)
            self.assertGreater(info["host_ffn_bytes"], 0)
            self.assertLess(
                info["gpu_allocated_bytes"],
                resident.info["gpu_allocated_bytes"],
            )
        finally:
            resident.close()
            spilled.close()

    def test_dense_stack_matches_numpy_when_the_mixer_is_muted(self):
        """Isolates the feed-forward from attention and the recurrence.

        With the mixer output projections zeroed each block contributes only its
        SwiGLU, so the whole forward pass is reproducible in a few lines of
        numpy and a mismatch is attributable to the dense path alone.
        """
        model, spec, _ = _model(mute_mixer=True)
        for token in (5, 11, 23):
            hidden = _decode(
                model, "token_embd.weight", spec.hidden * spec.vocabulary
            ).reshape(spec.vocabulary, spec.hidden)[token].copy()
            epsilon = float(model.config.get("rms_norm_epsilon") or 1e-6)
            for layer in range(spec.layers):
                prefix = f"blk.{layer}."
                weights = _decode(model, prefix + "post_attention_norm.weight", spec.hidden)
                normalized = hidden / np.sqrt(np.mean(hidden * hidden) + epsilon) * weights
                gate = _decode(model, prefix + "ffn_gate.weight", spec.hidden * spec.intermediate)
                up = _decode(model, prefix + "ffn_up.weight", spec.hidden * spec.intermediate)
                down = _decode(model, prefix + "ffn_down.weight", spec.intermediate * spec.hidden)
                projected_gate = gate.reshape(spec.intermediate, spec.hidden) @ normalized
                projected_up = up.reshape(spec.intermediate, spec.hidden) @ normalized
                activated = projected_gate / (1.0 + np.exp(-projected_gate)) * projected_up
                hidden = hidden + down.reshape(spec.hidden, spec.intermediate) @ activated
            expected = int(np.argmax(_logits(model, hidden, spec)))
            for options in ({}, {"gpu_cache_bytes": SPILL_BUDGET_BYTES}):
                runtime = _native(model, **options)
                try:
                    if options:
                        self.assertGreater(runtime.info["host_ffn_layers"], 0)
                    self.assertEqual(runtime.decode(token), expected)
                finally:
                    runtime.close()

    def test_parallel_engine_uses_only_the_two_hidden_dma_buffers(self):
        """Spilled dense FFNs remain safe in the multi-sequence path.

        This path used to place the output after a nonexistent pinned
        activation buffer, beyond the allocation made by prepare().
        """
        model, _, _ = _model()
        runtime = _native(
            model,
            parallel_sequences=2,
            gpu_cache_bytes=36 * 1024 * 1024,
        )
        try:
            self.assertGreater(runtime.info["host_ffn_layers"], 0)
            task_ids = {
                runtime.task_submit([5], 2),
                runtime.task_submit([11], 2),
            }
            completed = set()
            for _ in range(12):
                for task_id, _, kind in runtime.engine_step():
                    if kind == 1:
                        completed.add(task_id)
                if completed == task_ids:
                    break
            self.assertEqual(completed, task_ids)
        finally:
            runtime.close()


class DenseLmHeadTests(unittest.TestCase):
    def test_f32_lm_head_is_not_decoded_as_q8(self):
        """An f32 head used to fall through to the Q8_0 argmax kernel.

        That decodes the head as noise, which pins the argmax to one token
        regardless of the input -- so distinct inputs must give distinct
        winners here.
        """
        model, spec, _ = _model()
        winners = set()
        for token in range(spec.vocabulary):
            decoder = QwenV2Decoder(model)
            hidden, _ = decoder.forward_token(token, decoder.new_state())
            winners.add(int(np.argmax(_logits(model, hidden, spec))))
        self.assertGreater(len(winners), 1)


if __name__ == "__main__":
    unittest.main()
