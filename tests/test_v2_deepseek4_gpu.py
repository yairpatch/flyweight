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

CHECKPOINT = os.environ.get("DEEPSEEK4_GGUF")


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
        for name in ("blk.10.attn_q_b.weight", "blk.10.attn_output_b.weight",
                     "blk.10.attn_kv.weight"):
            with self.subTest(name=name):
                on_gpu, on_cpu, _, _ = self.check(name)
                self.assertEqual(self.tensors[name]["ggml_type"], 8)
                np.testing.assert_array_equal(on_gpu, on_cpu)

    def test_q6k_agrees_within_float_rounding(self):
        on_gpu, on_cpu, _, _ = self.check("blk.10.attn_q_a.weight")
        self.assertEqual(self.tensors["blk.10.attn_q_a.weight"]["ggml_type"], 14)
        np.testing.assert_allclose(on_gpu, on_cpu, rtol=1e-4, atol=1e-5)

    def test_the_q8_kernel_clears_the_cpu_by_a_wide_margin(self):
        name = "blk.10.attn_q_b.weight"
        self.check(name, iterations=2000)          # let the clocks come up
        _, _, seconds, size = self.check(name, iterations=6000)
        rate = 6000 * size / 1024**3 / seconds
        # Measured 300 GiB/s; the floor is set well under that so this reports
        # a regression rather than a warm afternoon. The CPU path does 46.
        self.assertGreater(rate, 150.0, f"{rate:.0f} GiB/s")


if __name__ == "__main__":
    unittest.main()
