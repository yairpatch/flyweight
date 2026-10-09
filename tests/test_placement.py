import unittest

from flyweight.placement import device_total, format_placement
from flyweight.v2 import _PlacementPlan

_MIB = 1024 * 1024


def _plan(**fields: int) -> dict[str, int]:
    plan = {name: 0 for name, _ in _PlacementPlan._fields_}
    plan.update(
        gpu_total_bytes=12_000 * _MIB, gpu_free_bytes=2_000 * _MIB,
        gpu_budget_bytes=9_000 * _MIB, gpu_budget_auto=1, layers=40,
        context_limit=32768, cache_type_k=1, cache_type_v=1, parallel_sequences=1,
        gpu_expert_layers=0xFFFFFFFF,
    )
    plan.update(fields)
    return plan


class PlacementReportTests(unittest.TestCase):
    def test_device_rows_sum_and_other_is_the_measured_remainder(self) -> None:
        plan = _plan(static_weights_bytes=2_000 * _MIB, kv_state_bytes=700 * _MIB,
                     expert_cache_bytes=5_000 * _MIB,
                     gpu_used_by_prepare_bytes=7_750 * _MIB)
        self.assertEqual(device_total(plan), 7_700 * _MIB)
        text = format_placement(plan)
        self.assertIn("runtime arenas", text)
        self.assertRegex(text, r"other \(measured\)\s+50 MiB")
        self.assertIn("auto-fit", text)

    def test_moe_cache_share_and_whole_layer_split(self) -> None:
        cached = format_placement(_plan(moe_layers=40, expert_mode=2,
                                        expert_cache_slots=2_560, expert_bundles=10_240))
        self.assertIn("cache holds 2,560 of 10,240 (25%)", cached)
        pinned = format_placement(_plan(moe_layers=40, expert_mode=2,
                                        expert_cache_slots=2_560, expert_bundles=10_240,
                                        gpu_expert_layers=10))
        self.assertIn("10 of 40 layers' experts pinned on GPU", pinned)
        cpu = format_placement(_plan(moe_layers=40, expert_mode=1))
        self.assertIn("all on CPU", cpu)

    def test_expert_preload_is_listed_under_ram(self) -> None:
        text = format_placement(_plan(moe_layers=40, expert_mode=2,
                                      preload_expert_bytes=46_800 * _MIB))
        self.assertRegex(text, r"expert preload \(bg\)\s+46,800 MiB")
        self.assertNotIn("expert preload", format_placement(_plan()))

    def test_dense_spill_and_cpu_backend(self) -> None:
        text = format_placement(_plan(dense_ffn_layers=64, host_ffn_layers=4,
                                      gpu_total_bytes=0))
        self.assertIn("60 of 64 blocks on GPU, 4 on CPU", text)
        self.assertIn("none (CPU backend)", text)


if __name__ == "__main__":
    unittest.main()
