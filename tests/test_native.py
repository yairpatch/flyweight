import math
import unittest

from colibri_next.native import active_native
from colibri_next.q4 import Q4BlockTensor

from tests.test_converter import bf16_bytes


class NativeKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = active_native()
        if cls.backend is None:
            raise unittest.SkipTest("native backend is not built")

    def test_native_q4_matvec_matches_portable_kernel(self) -> None:
        for shape in ((7, 37), (128, 512), (64, 2048)):
            values = [
                math.sin(index * 0.013) * 0.7
                for index in range(math.prod(shape))
            ]
            vector = [math.cos(index * 0.021) for index in range(shape[1])]
            tensor = Q4BlockTensor.from_bf16(
                bf16_bytes(values), shape, prefer_numpy=False
            )
            expected = tensor.matvec(vector, prefer_numpy=False)
            actual = self.backend.q4_matvec(tensor, vector)
            for expected_value, actual_value in zip(expected, actual):
                self.assertAlmostEqual(expected_value, actual_value, places=3)

    def test_runtime_reports_selected_simd_features(self) -> None:
        self.assertGreaterEqual(self.backend.version, 1)
        self.assertTrue(self.backend.features)


if __name__ == "__main__":
    unittest.main()
