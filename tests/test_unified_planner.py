import unittest

from colibri_next.hardware import (
    ComputeDevice,
    DeviceKind,
    HardwareTopology,
    MemoryTier,
    TierKind,
)
from colibri_next.models import QWEN36_35B_A3B
from colibri_next.planner import GIB, PlacementPlanner


class UnifiedMemoryPlannerTests(unittest.TestCase):
    def test_metal_backend_uses_shared_ram_without_double_counting(self) -> None:
        topology = HardwareTopology(
            system="Darwin",
            architecture="arm64",
            cpu_name="Apple Silicon",
            cpu_threads=12,
            tiers=(
                MemoryTier("storage:0", TierKind.STORAGE, 512 * GIB, 400 * GIB),
                MemoryTier("ram:0", TierKind.RAM, 32 * GIB, 32 * GIB),
            ),
            devices=(
                ComputeDevice("cpu:0", DeviceKind.CPU, "Apple Silicon", "ram:0"),
                ComputeDevice(
                    "metal:0",
                    DeviceKind.METAL,
                    "Apple Silicon GPU",
                    "ram:0",
                    ("unified_memory", "fp16"),
                ),
            ),
        )
        plan = PlacementPlanner(topology).plan(QWEN36_35B_A3B)
        decisions = {decision.group: decision for decision in plan.decisions}
        self.assertTrue(plan.supported)
        self.assertEqual(
            decisions["static_and_shared"].compute_backend_id, "metal:0:backend"
        )
        self.assertEqual(
            decisions["routed_experts"].compute_backend_id, "metal:0:backend"
        )
        self.assertEqual(decisions["routed_experts"].home_tier_id, "ram:0")


if __name__ == "__main__":
    unittest.main()
