"""The device path's foundation: same numbers, and fast enough to be worth it.

Stage C moves the dense half of this model -- 6.9 GiB against 90 GiB of routed
experts, and read in full every token -- onto the GPU. All of it goes through a
quantized matvec, so two things had to be true before any of that was built, and
this checks both against the real checkpoint.

**The same numbers.** The quantized matvec kernels were written for Qwen. That
they decode a deepseek4 tensor identically is an assumption worth testing rather
than reasoning about: Q8_0 comes back bit-identical, and Q6_K within float
rounding order.

**Fast enough.** The shared Q8_0 kernel measured 80 GiB/s where the CPU path
already does 46 across sixteen cores, which would have made the whole stage
worth about 1.3x. It walks one 32-value block per warp iteration, so only 32
bytes of weights are in flight behind each dependent load. Four blocks per
iteration measures 300 GiB/s on the same hardware and the same layout, which is
what puts Stage C back at the 2x its attribution promised. This test holds that
floor, well below what was measured, so it fails on a regression rather than on
a slow day.

Timings on this laptop GPU need the clock to ramp under load first, which is why
there is a warmup pass and why the measured pass is seconds long rather than
milliseconds.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from colibri_next.v2 import V2Model

_CHECKPOINT_PATH = os.environ.get("DEEPSEEK4_GGUF")
# A stale path is as good as no path: the variable often outlives the file it
# named, and treating that as "configured" turns a missing checkpoint into a
# wall of errors instead of a skip.
CHECKPOINT = _CHECKPOINT_PATH if _CHECKPOINT_PATH and os.path.exists(_CHECKPOINT_PATH) else None


def gpu_present() -> bool:
    try:
        return bool(V2Model.gpu_info()["available"])
    except Exception:
        return False


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
@unittest.skipUnless(gpu_present(), "no CUDA device available")
class DeviceMatvecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = V2Model(CHECKPOINT)
        cls.tensors = {
            tensor["name"]: tensor for tensor in cls.model.tensors()
        }

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def a_tensor_of_type(self, ggml_type: int) -> str:
        """Any 2-D tensor stored as `ggml_type`, or skip.

        Which tensor carries which type is a property of the quantization, not
        of the architecture -- a name that is Q6_K in one build is Q5_K in
        another -- so these tests look the type up instead of assuming it.
        """
        for name, tensor in self.tensors.items():
            if int(tensor["ggml_type"]) == ggml_type and len(tensor["shape"]) == 2:
                return name
        raise unittest.SkipTest(f"this checkpoint stores nothing as type {ggml_type}")

    def check(self, name: str, iterations: int = 0):
        from colibri_next.deepseek4 import gpu_matvec_check

        tensor = self.tensors[name]
        rows, columns = tensor["shape"][0], tensor["shape"][1]
        vector = np.random.default_rng(5).standard_normal(rows).astype(np.float32)
        on_gpu, on_cpu, seconds = gpu_matvec_check(
            self.model, name, vector, columns, iterations=iterations
        )
        return on_gpu, on_cpu, seconds, tensor["size"]

    def test_q8_is_bit_identical_to_the_cpu(self):
        # Not "close": the same decode, the same order of accumulation per row.
        names = [
            name for name, tensor in self.tensors.items()
            if int(tensor["ggml_type"]) == 8 and len(tensor["shape"]) == 2
        ][:3]
        self.assertTrue(names, "checkpoint stores nothing as Q8_0")
        for name in names:
            with self.subTest(name=name):
                on_gpu, on_cpu, _, _ = self.check(name)
                np.testing.assert_array_equal(on_gpu, on_cpu)

    def test_q6k_agrees_within_float_rounding(self):
        name = self.a_tensor_of_type(14)
        on_gpu, on_cpu, _, _ = self.check(name)
        np.testing.assert_allclose(on_gpu, on_cpu, rtol=1e-4, atol=1e-5)

    def test_iq1s_expert_slice_agrees_with_the_cpu(self):
        tensor = next(
            (item for item in self.tensors.values()
             if int(item["ggml_type"]) == 19 and len(item["shape"]) == 3),
            None,
        )
        if tensor is None:
            self.skipTest("this checkpoint stores no IQ1_S expert tensor")
        on_gpu, on_cpu, _, _ = self.check(tensor["name"])
        np.testing.assert_allclose(on_gpu, on_cpu, rtol=2e-4, atol=2e-4)

    def test_the_q8_kernel_clears_the_cpu_by_a_wide_margin(self):
        name = self.a_tensor_of_type(8)
        self.check(name, iterations=2000)          # let the clocks come up
        _, _, seconds, size = self.check(name, iterations=6000)
        rate = 6000 * size / 1024**3 / seconds
        # Measured 300 GiB/s; the floor is set well under that so this reports
        # a regression rather than a warm afternoon. The CPU path does 46.
        self.assertGreater(rate, 150.0, f"{rate:.0f} GiB/s")


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
@unittest.skipUnless(gpu_present(), "no CUDA device available")
class ResidentDenseWeightTests(unittest.TestCase):
    """The dense half resident on the device, the experts left where they are.

    Which half is the point. The routed experts are 90 GiB and can never be
    resident; everything else is under 7 and is read in full on every token.
    The activations that cross for each call are a few kilobytes against
    megabytes of weights that do not move at all, so crossing per call is
    affordable even before the fiddlier pieces have device kernels.

    Measured on the real checkpoint, warm: attention 102 -> 37 ms a token and
    3.90 -> 5.89 tok/s. What the test holds is the part that would be a defect
    rather than a slow day -- that the answer does not change.
    """

    @classmethod
    def setUpClass(cls):
        from colibri_next.deepseek4 import Deepseek4Runtime

        cls.model = V2Model(CHECKPOINT)
        cls.tokens = list(cls.model.tokenize("The capital city of France is"))
        cls.Runtime = Deepseek4Runtime

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def logits(self, gpu: bool):
        with self.Runtime(self.model, 256) as runtime:
            if gpu:
                runtime.use_gpu(0)
            for token in self.tokens[:-1]:
                runtime.forward(token, logits=False)
            return runtime.forward(self.tokens[-1]), dict(runtime.info)

    def test_the_device_agrees_with_the_cpu_on_the_next_token(self):
        # Not bit-identical: Q6_K accumulates in a different order on each, and
        # 43 layers amplify that. The bar is the one this port already uses --
        # the same answer, not the same bits.
        on_cpu, _ = self.logits(False)
        on_gpu, info = self.logits(True)
        self.assertEqual(int(on_gpu.argmax()), int(on_cpu.argmax()))
        top_cpu = set(int(index) for index in on_cpu.argsort()[::-1][:5])
        top_gpu = set(int(index) for index in on_gpu.argsort()[::-1][:5])
        self.assertGreaterEqual(len(top_cpu & top_gpu), 4)
        self.assertGreater(info["gpu_matvec_calls"], 0)

    def test_only_the_dense_half_is_uploaded(self):
        _, info = self.logits(True)
        resident = info["gpu_weight_bytes"] / 1024**3
        # Under 7 GiB fits the 12 GiB card with room for the caches; if the
        # routed experts ever crept in this would be ninety.
        self.assertGreater(resident, 4.0)
        self.assertLess(resident, 8.0)

    def test_layer_major_prefill_matches_sequential_forward(self):
        sequential, batched = [], []
        for use_prefill, target in ((False, sequential), (True, batched)):
            with self.Runtime(self.model, 256) as runtime:
                runtime.use_gpu(0)
                if use_prefill:
                    runtime.prefill(self.tokens[:-1])
                else:
                    for token in self.tokens[:-1]:
                        runtime.forward(token, logits=False)
                logits = runtime.forward(self.tokens[-1])
                target.extend(int(index) for index in logits.argsort()[::-1][:10])
                if use_prefill:
                    self.assertEqual(runtime.info["prefill_tokens"], len(self.tokens) - 1)
        self.assertEqual(batched[0], sequential[0])
        self.assertGreaterEqual(len(set(batched) & set(sequential)), 9)


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
@unittest.skipUnless(gpu_present(), "no CUDA device available")
class HybridServiceTests(unittest.TestCase):
    """Serving hybrid, which is where the context-per-thread rule bites.

    The driver retains the device's primary context and makes it current on the
    calling thread. The weights are uploaded from whichever thread builds the
    service, and the scheduler forwards from its own -- where nothing was
    current, so every launch failed and every request returned a 500. It is
    invisible from a single-threaded script, which is how it survived a whole
    round of measurement, so the test drives the engine rather than the runtime.
    """

    def test_a_generation_survives_the_scheduler_thread(self):
        from colibri_next.deepseek4_server import NativeDeepseek4InferenceService

        service = NativeDeepseek4InferenceService(
            CHECKPOINT, context_window=512, max_new_tokens=8, device=0
        )
        self.addCleanup(service.close)
        execution = service.health()["execution"]
        self.assertEqual(execution["backend"], "native-v2-deepseek4-hybrid")
        self.assertGreater(execution["gpu_weight_bytes"], 4 * 1024**3)
        reply = service.chat_completion({
            "model": service.model_name,
            "messages": [{"role": "user", "content": "Capital of France, one word."}],
            "max_tokens": 4,
        })["choices"][0]["message"]["content"]
        self.assertIn("Paris", reply)

    def test_multiple_slots_share_one_dense_weight_upload(self):
        from colibri_next.deepseek4_server import NativeDeepseek4InferenceService

        service = NativeDeepseek4InferenceService(
            CHECKPOINT, context_window=512, device=0, parallel_sequences=2
        )
        self.addCleanup(service.close)
        runtimes = [slot.runtime for slot in service.generator.engine._pool]
        self.assertEqual(runtimes[0].info["gpu_weight_bytes"],
                         runtimes[1].info["gpu_weight_bytes"])
        self.assertGreater(runtimes[0].info["gpu_weight_bytes"], 4 * 1024**3)


if __name__ == "__main__":
    unittest.main()
