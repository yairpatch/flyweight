import unittest

from colibri_next.v2 import V2Error, V2Model
from colibri_next.v2_qwen import QwenDeltaLayer


class QwenV2ReferenceTests(unittest.TestCase):
    MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"

    def test_native_runtime_catalogs_real_decoder_stack(self):
        try:
            model = V2Model(self.MODEL)
        except Exception as error:
            raise unittest.SkipTest(str(error)) from error
        try:
            with model.native_qwen_runtime(
                context_limit=2048, gpu_cache_bytes=6 * 1024**3
            ) as runtime:
                info = runtime.info
                self.assertEqual(info["layers"], 40)
                self.assertEqual(
                    info["deltanet_layers"] + info["attention_layers"], 40
                )
                self.assertGreater(info["deltanet_layers"], 0)
                self.assertGreater(info["attention_layers"], 0)
                self.assertEqual(info["hidden_size"], 2048)
                self.assertEqual(info["expert_count"], 256)
                self.assertEqual(info["expert_used_count"], 8)
                self.assertEqual(info["context_limit"], 2048)
                self.assertGreater(info["static_tensor_bytes"], 0)
                self.assertGreater(info["expert_tensor_bytes"], 0)
                self.assertEqual(info["gpu_allocated_bytes"], 0)
                self.assertFalse(info["decode_ready"])
                with self.assertRaisesRegex(V2Error, "not prepared"):
                    runtime.decode(0)
                runtime.cancel()
                runtime.reset()
                self.assertEqual(runtime.info["position"], 0)
        finally:
            model.close()

    def test_native_runtime_accepts_cpu_moe_mode(self):
        try:
            model = V2Model(self.MODEL)
        except Exception as error:
            raise unittest.SkipTest(str(error)) from error
        try:
            with model.native_qwen_runtime(
                context_limit=2048,
                gpu_cache_bytes=4 * 1024**3,
                moe_device="cpu",
            ) as runtime:
                self.assertEqual(runtime.info["moe_device"], 1)
                self.assertFalse(runtime.info["decode_ready"])
        finally:
            model.close()

    def test_native_runtime_rejects_unknown_moe_mode(self):
        try:
            model = V2Model(self.MODEL)
        except Exception as error:
            raise unittest.SkipTest(str(error)) from error
        try:
            with self.assertRaisesRegex(ValueError, "moe_device"):
                model.native_qwen_runtime(moe_device="accelerator")
        finally:
            model.close()

    def test_native_runtime_accepts_hybrid_moe_mode(self):
        try:
            model = V2Model(self.MODEL)
        except Exception as error:
            raise unittest.SkipTest(str(error)) from error
        try:
            with model.native_qwen_runtime(
                context_limit=2048,
                gpu_cache_bytes=6 * 1024**3,
                moe_device="hybrid",
            ) as runtime:
                self.assertEqual(runtime.info["moe_device"], 2)
        finally:
            model.close()

    def test_native_runtime_catalogs_embedded_mtp_separately(self):
        try:
            model = V2Model(self.MODEL)
        except Exception as error:
            raise unittest.SkipTest(str(error)) from error
        try:
            with model.native_qwen_runtime(
                context_limit=2048,
                gpu_cache_bytes=4 * 1024**3,
                moe_device="hybrid",
                mtp_drafts=2,
                cache_type_k="f32",
                cache_type_v="f32",
            ) as runtime:
                info = runtime.info
                self.assertEqual(info["layers"], 40)
                self.assertTrue(info["mtp_available"])
                self.assertTrue(info["mtp_enabled"])
                self.assertEqual(info["mtp_layer"], 40)
                self.assertEqual(info["mtp_drafts"], 2)
                self.assertGreater(info["mtp_tensor_bytes"], 0)
        finally:
            model.close()

    def test_native_runtime_rejects_invalid_mtp_depth(self):
        try:
            model = V2Model(self.MODEL)
        except Exception as error:
            raise unittest.SkipTest(str(error)) from error
        try:
            with self.assertRaisesRegex(ValueError, "mtp_drafts"):
                model.native_qwen_runtime(mtp_drafts=9)
        finally:
            model.close()

    def test_native_runtime_rejects_invalid_prefill_cache_seed(self):
        try:
            model = V2Model(self.MODEL)
        except Exception as error:
            raise unittest.SkipTest(str(error)) from error
        try:
            with self.assertRaisesRegex(ValueError, "prefill_cache_seed"):
                model.native_qwen_runtime(prefill_cache_seed=257)
        finally:
            model.close()

    def test_native_runtime_rejects_invalid_expert_paging(self):
        try:
            model = V2Model(self.MODEL)
        except Exception as error:
            raise unittest.SkipTest(str(error)) from error
        try:
            with self.assertRaisesRegex(ValueError, "expert_paging"):
                model.native_qwen_runtime(expert_paging="pinned-everywhere")
        finally:
            model.close()

    def test_real_delta_layer_state_is_persistent_and_resettable(self):
        try:
            model = V2Model(self.MODEL)
        except Exception as error:
            raise unittest.SkipTest(str(error)) from error
        try:
            layer = QwenDeltaLayer(model, 0)
            state = layer.new_state()
            hidden = [0.01] * layer.hidden
            first = layer.forward_residual(hidden, state)
            second = layer.forward_residual(hidden, state)
            self.assertEqual(state.tokens, 2)
            self.assertNotEqual(first, second)
            state.reset()
            self.assertEqual(state.tokens, 0)
            replay = layer.forward_residual(hidden, state)
            for expected, actual in zip(first, replay):
                self.assertAlmostEqual(expected, actual, places=5)
        finally:
            model.close()


if __name__ == "__main__":
    unittest.main()
