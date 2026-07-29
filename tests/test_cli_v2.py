import os
import unittest
from pathlib import Path
from unittest.mock import patch

from colibri_next.cli import (
    _benchmark_native_prefill,
    _drop_file_cache,
    _parser,
    _steady_state_counters,
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

    def test_cold_cache_advises_and_closes_model(self) -> None:
        dontneed = 4
        with (
            patch("colibri_next.cli.os.open", return_value=17) as open_file,
            patch(
                "colibri_next.cli.os.posix_fadvise", create=True
            ) as advise,
            patch(
                "colibri_next.cli.os.POSIX_FADV_DONTNEED",
                dontneed,
                create=True,
            ),
            patch("colibri_next.cli.os.close") as close_file,
        ):
            _drop_file_cache(Path("model.gguf"))
        open_file.assert_called_once_with(Path("model.gguf"), os.O_RDONLY)
        advise.assert_called_once_with(17, 0, 0, dontneed)
        close_file.assert_called_once_with(17)

    def test_mtp_accepts_quantized_kv(self) -> None:
        """The MTP path stores and reads its KV at the configured precision."""
        for command in ("benchmark-v2", "probe-native-v2", "serve-v2"):
            with self.subTest(command=command):
                args = _parser().parse_args(
                    [command, "model.gguf", "--mtp-drafts", "1",
                     "--cache-type-k", "q8_0", "--cache-type-v", "f16"]
                )
                self.assertEqual(args.mtp_drafts, 1)
                self.assertEqual(args.cache_type_k, "q8_0")
                self.assertEqual(args.cache_type_v, "f16")


if __name__ == "__main__":
    unittest.main()
