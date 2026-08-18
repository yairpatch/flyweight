import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from colibri_next.cli import (
    DEFAULT_QUANT,
    _benchmark_native_generate,
    _benchmark_bailing_generate,
    _architecture,
    _benchmark_native_prefill,
    _drop_file_cache,
    _parser,
    _prefill_cache_seed,
    _quant_from_answer,
    _stop_tokens,
    _quant_menu,
    _resolve_quant,
    _steady_state_counters,
)


class NativeV2BenchmarkTests(unittest.TestCase):
    def test_hf_architecture_detection_reads_config_without_opening_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({
                "model_type": "bailing_hybrid",
                "architectures": ["BailingMoeV3ForCausalLM"],
            }))
            with patch("colibri_next.cli.V2Model") as model:
                self.assertEqual(_architecture(path), "bailingmoe3")
            model.assert_not_called()

    def test_bailing_benchmark_uses_the_server_eval_sample_loop(self) -> None:
        class Runtime:
            def __init__(self):
                self.steps = []
                self.samples = iter((41, 42))
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1

            def eval_into(self, step):
                self.steps.append(list(step))

            def sample(self, config):
                return next(self.samples)

        runtime = Runtime()
        config = object()
        with patch(
            "colibri_next.cli.time.perf_counter",
            side_effect=(10.0, 10.1, 10.3, 10.4),
        ):
            generated, arrivals, elapsed = _benchmark_bailing_generate(
                runtime, [1, 2, 3], 2, config,
            )
        self.assertEqual(runtime.reset_count, 1)
        self.assertEqual(runtime.steps, [[1, 2, 3], [41]])
        self.assertEqual(generated, [41, 42])
        self.assertAlmostEqual(arrivals[0], 0.1)
        self.assertAlmostEqual(arrivals[1], 0.3)
        self.assertAlmostEqual(elapsed, 0.4)

    def test_prefill_cache_seed_accepts_auto_off_and_bounded_counts(self) -> None:
        self.assertEqual(_prefill_cache_seed("auto"), "auto")
        self.assertEqual(_prefill_cache_seed("OFF"), "off")
        self.assertEqual(_prefill_cache_seed("0"), 0)
        self.assertEqual(_prefill_cache_seed("4"), 4)
        with self.assertRaisesRegex(Exception, r"\[0, 256\]"):
            _prefill_cache_seed("257")

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

    def test_mtp_benchmark_times_generate_instead_of_decode(self) -> None:
        class Runtime:
            def __init__(self):
                self.calls = []

            def generate(self, prompt, max_tokens, callback):
                self.calls.append((prompt, max_tokens))
                callback(41)
                callback(42)

        runtime = Runtime()
        with patch(
            "colibri_next.cli.time.perf_counter",
            side_effect=(10.0, 10.1, 10.3, 10.4),
        ):
            generated, arrivals, elapsed = _benchmark_native_generate(
                runtime, [1, 2, 3], 2,
            )
        self.assertEqual(runtime.calls, [([1, 2, 3], 2)])
        self.assertEqual(generated, [41, 42])
        self.assertAlmostEqual(arrivals[0], 0.1)
        self.assertAlmostEqual(arrivals[1], 0.3)
        self.assertAlmostEqual(elapsed, 0.4)

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

    def test_mtp_sidecar_is_available_to_runtime_commands(self) -> None:
        for command in ("generate", "benchmark-v2", "probe-native-v2", "serve-v2"):
            with self.subTest(command=command):
                argv = [command, "model.gguf", "--mtp-model", "mtp.gguf"]
                if command == "generate":
                    argv.extend(("--prompt", "hello"))
                args = _parser().parse_args(argv)
                self.assertEqual(args.mtp_model, Path("mtp.gguf"))

    def test_embedded_mtp_does_not_require_a_sidecar(self) -> None:
        for command in ("generate", "benchmark-v2", "probe-native-v2", "serve-v2"):
            with self.subTest(command=command):
                argv = [command, "model.gguf", "--mtp-drafts", "4"]
                if command == "generate":
                    argv.extend(("--prompt", "hello"))
                args = _parser().parse_args(argv)
                self.assertEqual(args.mtp_drafts, 4)
                self.assertIsNone(args.mtp_model)

    def test_dense_requant_policy_is_available_to_runtime_commands(self) -> None:
        for command in ("generate", "benchmark-v2", "probe-native-v2", "serve-v2"):
            with self.subTest(command=command):
                argv = [command, "model.gguf", "--dense-requant", "off"]
                if command == "generate":
                    argv.extend(("--prompt", "hello"))
                args = _parser().parse_args(argv)
                self.assertEqual(args.dense_requant, "off")

    def test_public_command_names_and_short_limit_options(self) -> None:
        serve = _parser().parse_args(
            ["serve", "model.gguf", "--context", "65536", "--max-tokens", "8192",
             "--concurrency", "4"]
        )
        self.assertEqual(serve.context_window, 65536)
        self.assertEqual(serve.max_new_tokens, 8192)
        self.assertEqual(serve.max_concurrent_requests, 4)

        inspect = _parser().parse_args(["inspect", "model.gguf"])
        benchmark = _parser().parse_args(["benchmark", "model.gguf"])
        self.assertEqual(inspect.command, "inspect")
        self.assertEqual(benchmark.command, "benchmark")

    def test_legacy_serve_names_remain_compatible(self) -> None:
        args = _parser().parse_args(
            ["serve-v2", "model.gguf", "--context-window", "4096",
             "--max-new-tokens", "512"]
        )
        self.assertEqual(args.context_window, 4096)
        self.assertEqual(args.max_new_tokens, 512)


class ProbeStopTokenTests(unittest.TestCase):
    def test_probe_collects_the_ids_that_end_a_turn(self) -> None:
        """The raw generate loop has no stop set, so the caller needs one.

        Left running past the model's end token, decode continues into an
        invented next turn -- `<|im_end|><|im_start|>Human...` -- which is what
        `probe` was printing as though the model had written it.
        """
        model = SimpleNamespace(config={
            "eos_token_id": 248046,
            # UINT32_MAX is how the runtime spells "this model has none".
            "eot_token_id": 0xFFFFFFFF,
            "bos_token_id": 248044,
        })
        self.assertEqual(_stop_tokens(model), {248046})

        both = SimpleNamespace(config={"eos_token_id": 7, "eot_token_id": 9})
        self.assertEqual(_stop_tokens(both), {7, 9})


class QuantPromptTests(unittest.TestCase):
    """Choosing a quantization before a safetensors checkpoint is opened.

    The loader takes its answer from COLIBRI_HF_QUANT, so what these pin is
    which value ends up there -- and, as much, when nothing does: a pipe, a
    service manager and an explicit environment all have to keep loading the
    way they did before the prompt existed.
    """

    OPTIONS = [
        {"name": "Q4_K", "arena_bytes": 16 * 1024**3, "cache_bytes": 16 * 1024**3,
         "cache_path": "/models/m/colibri-1.cache"},
        {"name": "Q6_K", "arena_bytes": 22 * 1024**3, "cache_bytes": 0,
         "cache_path": "/models/m/colibri-2.cache"},
        {"name": "F32", "arena_bytes": 109 * 1024**3, "cache_bytes": 0,
         "cache_path": "/models/m/colibri-3.cache"},
    ]

    def setUp(self) -> None:
        self._previous = os.environ.pop("COLIBRI_HF_QUANT", None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        os.environ.pop("COLIBRI_HF_QUANT", None)
        if self._previous is not None:
            os.environ["COLIBRI_HF_QUANT"] = self._previous

    def _args(self, directory, quant=None):
        return SimpleNamespace(model=Path(directory), quant=quant)

    def test_an_answer_may_be_a_number_a_name_or_empty(self) -> None:
        for answer, expected in (
            ("1", "Q4_K"), ("3", "F32"),
            ("q4_k", "Q4_K"), ("Q4K", "Q4_K"), (" f32 ", "F32"),
            ("", DEFAULT_QUANT), ("   ", DEFAULT_QUANT),
        ):
            with self.subTest(answer=answer):
                self.assertEqual(
                    _quant_from_answer(answer, self.OPTIONS, DEFAULT_QUANT), expected)

    def test_an_answer_outside_the_menu_is_rejected_rather_than_guessed(self) -> None:
        for answer in ("0", "4", "-1", "q3_k", "yes"):
            with self.subTest(answer=answer):
                self.assertIsNone(
                    _quant_from_answer(answer, self.OPTIONS, DEFAULT_QUANT))

    def test_the_menu_separates_a_cached_arena_from_one_that_must_be_packed(self) -> None:
        menu = _quant_menu(self.OPTIONS, DEFAULT_QUANT)
        lines = menu.splitlines()
        self.assertIn("cached", lines[0])
        self.assertIn("16.0 GiB", lines[0])
        self.assertIn("packs on first open", lines[1])
        # The default is marked, and it is the loader's own default.
        self.assertIn("[default]", lines[1])
        self.assertIn(DEFAULT_QUANT, lines[1])

    def test_an_explicit_choice_needs_no_terminal_and_asks_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("colibri_next.cli.V2Model") as model:
                _resolve_quant(self._args(directory, "Q4_K"))
            model.hf_quant_options.assert_not_called()
        self.assertEqual(os.environ["COLIBRI_HF_QUANT"], "Q4_K")

    def test_a_pipe_is_left_with_the_loader_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.stdin.isatty", return_value=False), \
                 patch("colibri_next.cli.V2Model") as model:
                _resolve_quant(self._args(directory))
            model.hf_quant_options.assert_not_called()
        self.assertNotIn("COLIBRI_HF_QUANT", os.environ)

    def test_an_exported_variable_wins_over_the_prompt(self) -> None:
        os.environ["COLIBRI_HF_QUANT"] = "Q8_0"
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.stdin.isatty", return_value=True), \
                 patch("sys.stderr.isatty", return_value=True), \
                 patch("colibri_next.cli.V2Model") as model:
                _resolve_quant(self._args(directory))
            model.hf_quant_options.assert_not_called()
        self.assertEqual(os.environ["COLIBRI_HF_QUANT"], "Q8_0")

    def test_a_gguf_file_is_never_asked_about(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "model.gguf"
            model_file.write_bytes(b"GGUF")
            with patch("sys.stdin.isatty", return_value=True), \
                 patch("sys.stderr.isatty", return_value=True), \
                 patch("colibri_next.cli.V2Model") as model:
                _resolve_quant(SimpleNamespace(model=model_file, quant=None))
            model.hf_quant_options.assert_not_called()
        self.assertNotIn("COLIBRI_HF_QUANT", os.environ)

    def test_ask_prompts_and_publishes_the_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("colibri_next.cli.V2Model.hf_quant_options",
                       return_value=self.OPTIONS), \
                 patch("builtins.input", return_value="1"):
                _resolve_quant(self._args(directory, "ask"))
        self.assertEqual(os.environ["COLIBRI_HF_QUANT"], "Q4_K")

    def test_a_rejected_answer_is_asked_again(self) -> None:
        answers = iter(("banana", "2"))
        with tempfile.TemporaryDirectory() as directory:
            with patch("colibri_next.cli.V2Model.hf_quant_options",
                       return_value=self.OPTIONS), \
                 patch("builtins.input", side_effect=lambda: next(answers)):
                _resolve_quant(self._args(directory, "ask"))
        self.assertEqual(os.environ["COLIBRI_HF_QUANT"], "Q6_K")

    def test_interrupting_the_question_stops_rather_than_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("colibri_next.cli.V2Model.hf_quant_options",
                       return_value=self.OPTIONS), \
                 patch("builtins.input", side_effect=KeyboardInterrupt):
                with self.assertRaises(SystemExit) as caught:
                    _resolve_quant(self._args(directory, "ask"))
        self.assertEqual(caught.exception.code, 130)
        self.assertNotIn("COLIBRI_HF_QUANT", os.environ)

    def test_a_checkpoint_that_cannot_be_described_is_left_to_the_loader(self) -> None:
        from colibri_next.v2 import V2Error

        with tempfile.TemporaryDirectory() as directory:
            with patch("colibri_next.cli.V2Model.hf_quant_options",
                       side_effect=V2Error("unsupported HF architecture")), \
                 patch("builtins.input", side_effect=AssertionError("must not ask")):
                _resolve_quant(self._args(directory, "ask"))
        self.assertNotIn("COLIBRI_HF_QUANT", os.environ)

    def test_the_flag_is_available_to_every_command_that_takes_a_model(self) -> None:
        for command in ("generate", "serve", "benchmark", "inspect", "probe"):
            with self.subTest(command=command):
                argv = [command, "model.gguf", "--quant", "Q5_K"]
                if command == "generate":
                    argv.extend(("--prompt", "hello"))
                self.assertEqual(_parser().parse_args(argv).quant, "Q5_K")


if __name__ == "__main__":
    unittest.main()
