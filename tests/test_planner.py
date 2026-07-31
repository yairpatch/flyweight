import json
import tempfile
import unittest
from pathlib import Path

from colibri_next.hardware import (
    ComputeDevice,
    DeviceKind,
    HardwareTopology,
    MemoryTier,
    TierKind,
)
from colibri_next.models import QWEN36_35B_A3B
from colibri_next.planner import GIB, PlacementPlanner


def topology(*, ram_gib: int, storage_gib: int, vram_gib: int = 0) -> HardwareTopology:
    tiers = [
        MemoryTier("storage:0", TierKind.STORAGE, storage_gib * GIB, storage_gib * GIB),
        MemoryTier("ram:0", TierKind.RAM, ram_gib * GIB, ram_gib * GIB),
    ]
    devices = [
        ComputeDevice("cpu:0", DeviceKind.CPU, "Test CPU", "ram:0", ("x86_64",))
    ]
    if vram_gib:
        tiers.append(
            MemoryTier(
                "cuda:0:vram",
                TierKind.VRAM,
                vram_gib * GIB,
                vram_gib * GIB,
                device_id="cuda:0",
            )
        )
        devices.append(
            ComputeDevice(
                "cuda:0",
                DeviceKind.CUDA,
                "Test GPU",
                "cuda:0:vram",
                ("fp16", "int8", "int4"),
            )
        )
    return HardwareTopology(
        system="TestOS",
        architecture="x86_64",
        cpu_name="Test CPU",
        cpu_threads=16,
        tiers=tuple(tiers),
        devices=tuple(devices),
    )


class HardwareTopologyTests(unittest.TestCase):
    def test_profile_round_trip(self) -> None:
        original = topology(ram_gib=64, storage_gib=1024, vram_gib=12)
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "hardware.json"
            profile_path.write_text(original.to_json(), encoding="utf-8")
            restored = HardwareTopology.from_json_file(profile_path)
        self.assertEqual(original, restored)
        json.loads(original.to_json())


class PlacementPlannerTests(unittest.TestCase):
    def test_mixed_plan_for_64gb_ram_and_12gb_vram(self) -> None:
        plan = PlacementPlanner(
            topology(ram_gib=64, storage_gib=1024, vram_gib=12)
        ).plan(QWEN36_35B_A3B)
        decisions = {decision.group: decision for decision in plan.decisions}

        self.assertTrue(plan.supported)
        self.assertEqual(decisions["static_and_shared"].home_tier_id, "cuda:0:vram")
        self.assertEqual(decisions["routed_experts"].home_tier_id, "ram:0")
        self.assertEqual(decisions["routed_experts"].cache_tier_id, "cuda:0:vram")
        self.assertGreater(decisions["routed_experts"].cache_bytes, 0)

    def test_cpu_only_plan_keeps_quantized_model_in_ram(self) -> None:
        plan = PlacementPlanner(topology(ram_gib=32, storage_gib=256)).plan(
            QWEN36_35B_A3B
        )
        decisions = {decision.group: decision for decision in plan.decisions}
        self.assertTrue(plan.supported)
        self.assertEqual(decisions["routed_experts"].home_tier_id, "ram:0")
        self.assertEqual(decisions["routed_experts"].compute_backend_id, "cpu:0:backend")

    def test_low_ram_plan_streams_experts_from_storage(self) -> None:
        plan = PlacementPlanner(topology(ram_gib=8, storage_gib=128)).plan(
            QWEN36_35B_A3B
        )
        decisions = {decision.group: decision for decision in plan.decisions}
        self.assertTrue(plan.supported)
        self.assertEqual(decisions["static_and_shared"].quantization, "int8")
        self.assertEqual(decisions["routed_experts"].home_tier_id, "storage:0")
        self.assertEqual(decisions["routed_experts"].cache_tier_id, "ram:0")

    def test_large_gpu_keeps_all_weights_in_vram(self) -> None:
        plan = PlacementPlanner(
            topology(ram_gib=64, storage_gib=256, vram_gib=48)
        ).plan(QWEN36_35B_A3B)
        decisions = {decision.group: decision for decision in plan.decisions}
        self.assertTrue(plan.supported)
        self.assertEqual(decisions["routed_experts"].home_tier_id, "cuda:0:vram")
        self.assertIsNone(decisions["routed_experts"].cache_tier_id)

    def test_plan_rejects_insufficient_storage(self) -> None:
        plan = PlacementPlanner(topology(ram_gib=64, storage_gib=10)).plan(
            QWEN36_35B_A3B
        )
        self.assertFalse(plan.supported)
        self.assertIn("Insufficient storage", plan.warnings[-1])


if __name__ == "__main__":
    unittest.main()
