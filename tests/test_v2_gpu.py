import unittest

from flyweight.v2 import V2Error, V2Model


class V2GpuFoundationTests(unittest.TestCase):
    def test_memory_plan_reserves_in_runtime_order(self):
        plan = V2Model.memory_plan(100, 40, 20, 10, 50, 20)
        self.assertEqual(plan["active_experts"], 30)
        self.assertEqual(plan["staging"], 0)
        self.assertEqual(plan["unused"], 0)

    def test_memory_plan_rejects_persistent_overcommit(self):
        with self.assertRaises(V2Error):
            V2Model.memory_plan(10, 8, 3, 0, 0, 0)

    def test_gpu_probe_is_safe_without_cuda(self):
        info = V2Model.gpu_info()
        self.assertIn(info["available"], (0, 1))


if __name__ == "__main__":
    unittest.main()
