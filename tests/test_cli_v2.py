import unittest
from unittest.mock import patch

from colibri_next.cli import (
    _benchmark_native_prefill,
    _steady_state_counters,
    _validate_mtp_cache_types,
)


class NativeV2BenchmarkTests(unittest.TestCase):
    def test_prefill_benchmark_uses_blocking_generate(self) -> None:
        class Runtime:
            def __init__(self):
                self.calls = []

            def generate(self, prompt, max_tokens, callback):
                self.calls.append((prompt, max_tokens))
                callback(42)

        runtime = Runtime()
        with patch(
            "colibri_next.cli.time.perf_counter", side_effect=(10.0, 10.25)
        ):
            token, elapsed = _benchmark_native_prefill(runtime, [1, 2, 3])
        self.assertEqual(token, 42)
        self.assertEqual(elapsed, 0.25)
        self.assertEqual(runtime.calls, [([1, 2, 3], 1)])

    def test_steady_state_includes_cpu_expert_compute(self) -> None:
        fields = (
            "decode_calls", "decode_nanoseconds", "route_wait_nanoseconds",
            "expert_page_nanoseconds", "tail_wait_nanoseconds",
            "expert_compute_nanoseconds", "expert_cache_hits",
            "expert_cache_misses", "expert_cache_evictions",
        )
        start = {field: 0 for field in fields}
        end = {field: 0 for field in fields}
        end.update({"decode_calls": 2, "expert_compute_nanoseconds": 600})
        result = _steady_state_counters(start, end)
        self.assertEqual(result["expert_compute_ns_per_token"], 300)

    def test_mtp_requires_f32_kv(self) -> None:
        _validate_mtp_cache_types(0, "f16", "f16")
        _validate_mtp_cache_types(1, "f32", "f32")
        with self.assertRaisesRegex(SystemExit, "currently requires"):
            _validate_mtp_cache_types(1, "f16", "f32")


if __name__ == "__main__":
    unittest.main()
