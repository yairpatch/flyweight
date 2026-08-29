import argparse
import io
import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flyweight.cli import (
    AUTO_PROMPT_CACHE_MIB,
    DEFAULT_QUANT,
    _benchmark_native_generate,
    _benchmark_bailing_generate,
    _architecture,
    _benchmark_native_prefill,
    _drop_file_cache,
    _generate,
    _parser,
    _prefill_cache_seed,
    _prompt_tokens,
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
            with patch("flyweight.cli.V2Model") as model:
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
            "flyweight.cli.time.perf_counter",
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
            "flyweight.cli.time.perf_counter", side_effect=(10.0, 10.25)
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
            "flyweight.cli.time.perf_counter",
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
            patch("flyweight.cli.os.open", return_value=17) as open_file,
            patch(
                "flyweight.cli.os.posix_fadvise", create=True
            ) as advise,
            patch(
                "flyweight.cli.os.POSIX_FADV_DONTNEED",
                dontneed,
                create=True,
            ),
            patch("flyweight.cli.os.close") as close_file,
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


class HelpTests(unittest.TestCase):
    """What `--help` shows.

    Every runtime flag used to be added with `help=SUPPRESS` on `serve`,
    `generate` and `imatrix`, so the options the README recommends for a model
    that does not fit in VRAM -- `--parallel`, `--expert-mode`,
    `--cache-type-k` -- could not be found from the command itself. The top
    level had the same problem in miniature: its command list was a hand-written
    string that two commands had been added without.
    """

    def _help(self, *argv: str) -> str:
        parser = _parser()
        if argv:
            # The subparser for a command, reached the way argparse reaches it.
            action = next(
                action for action in parser._actions
                if isinstance(action, argparse._SubParsersAction)
            )
            parser = action.choices[argv[0]]
        return parser.format_help()

    def test_every_command_appears_in_the_command_list(self) -> None:
        listing = self._help()
        for command in ("serve", "generate", "benchmark", "inspect",
                        "imatrix", "probe"):
            self.assertIn(command, listing)

    def test_every_option_a_command_accepts_is_documented(self) -> None:
        """Nothing is parsed but hidden, and nothing is shown but undescribed."""
        parser = _parser()
        action = next(
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        for command, subparser in action.choices.items():
            for option in subparser._actions:
                with self.subTest(command=command, option=option.dest):
                    self.assertNotEqual(
                        option.help, argparse.SUPPRESS,
                        f"{command} hides --{option.dest.replace('_', '-')}",
                    )
                    self.assertTrue(
                        option.help,
                        f"{command} does not describe "
                        f"--{option.dest.replace('_', '-')}",
                    )

    def test_legacy_spellings_stay_out_of_the_command_list(self) -> None:
        """They are still accepted; the epilog is where they are mentioned."""
        listing = self._help().split("examples:")[0]
        for alias in ("serve-v2", "inspect-gguf", "probe-qwen-native-v2"):
            self.assertNotIn(alias, listing)

    def test_an_unknown_command_is_answered_with_the_real_ones(self) -> None:
        with self.assertRaises(SystemExit), \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            _parser().parse_args(["srve", "model.gguf"])
        message = stderr.getvalue()
        self.assertIn("invalid command 'srve'", message)
        self.assertIn("serve, generate, benchmark", message)
        self.assertNotIn("serve-v2", message)

    def test_serve_shows_the_placement_flags_its_operator_needs(self) -> None:
        text = self._help("serve")
        for flag in ("--parallel", "--expert-mode", "--cache-type-k",
                     "--gpu-cache-mib", "--cpu-threads", "--mtp-drafts",
                     "--max-connections"):
            self.assertIn(flag, text)

    def test_a_budget_default_is_named_rather_than_printed_as_a_sentinel(self) -> None:
        """`--cache auto` is carried as a MiB count no one would recognize."""
        text = self._help("serve")
        self.assertIn("(default: auto)", text)
        self.assertNotIn(str(AUTO_PROMPT_CACHE_MIB), text)

    def test_imatrix_does_not_offer_what_the_gather_overrides(self) -> None:
        """It pins experts to the CPU and turns drafting off for the run."""
        imatrix = _parser().parse_args(
            ["imatrix", "model.gguf", "--text", "calibration.txt"]
        )
        self.assertFalse(hasattr(imatrix, "expert_mode"))
        self.assertFalse(hasattr(imatrix, "mtp_drafts"))
        with self.assertRaises(SystemExit):
            _parser().parse_args(
                ["imatrix", "model.gguf", "--text", "t.txt",
                 "--expert-mode", "gpu"]
            )


class BackendSelectionTests(unittest.TestCase):
    """--backend has to reach the runtime, not just the parser.

    Only `serve` selected a backend; `generate --backend cpu` parsed the flag
    and then ran on the GPU anyway. Every command that builds a runtime now
    settles it before the model is opened, since allocations belong to whichever
    backend was active when they were made.
    """

    def test_every_runtime_command_accepts_a_backend(self) -> None:
        for command in ("serve", "generate", "benchmark", "probe", "imatrix"):
            with self.subTest(command=command):
                argv = [command, "model.gguf", "--backend", "cpu"]
                if command == "generate":
                    argv.extend(("--prompt", "hi"))
                if command == "imatrix":
                    argv.extend(("--text", "calibration.txt"))
                self.assertEqual(_parser().parse_args(argv).backend, "cpu")

    def test_generate_settles_the_backend_before_it_loads(self) -> None:
        args = _parser().parse_args(
            ["generate", "model.gguf", "--prompt", "hi", "--backend", "cpu"]
        )
        service = SimpleNamespace(
            model_name="m", close=lambda: None,
            chat_completion=lambda body: {
                "choices": [{"message": {"content": "ok"}}]
            },
        )
        order: list[str] = []

        def select(backend: str) -> str:
            order.append(f"select:{backend}")
            return backend

        def build(*args_, **kwargs):
            order.append("load")
            return service

        with patch("flyweight.cli._architecture", return_value="qwen3"), \
                patch("flyweight.cli.V2Model") as model, \
                patch("flyweight.v2_server.NativeV2InferenceService", build):
            model.select_backend.side_effect = select
            _generate(args)
        # Selected first: the service is what opens the model and allocates,
        # and an allocation belongs to the backend that was active for it.
        self.assertEqual(order, ["select:cpu", "load"])


class PromptTokenTests(unittest.TestCase):
    def test_probe_probes_the_token_it_was_given(self) -> None:
        """--token-id was inert: probe read benchmark's --tokens list instead."""
        probe = _parser().parse_args(["probe", "model.gguf", "--token-id", "42"])
        self.assertEqual(_prompt_tokens(SimpleNamespace(), probe), [42])

    def test_benchmark_still_reads_a_list_of_ids(self) -> None:
        benchmark = _parser().parse_args(
            ["benchmark", "model.gguf", "--tokens", "1,2,3"]
        )
        self.assertEqual(_prompt_tokens(SimpleNamespace(), benchmark), [1, 2, 3])


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

    The loader takes its answer from FLYWEIGHT_HF_QUANT, so what these pin is
    which value ends up there -- and, as much, when nothing does: a pipe, a
    service manager and an explicit environment all have to keep loading the
    way they did before the prompt existed.
    """

    OPTIONS = [
        {"name": "Q4_K", "arena_bytes": 16 * 1024**3, "cache_bytes": 16 * 1024**3,
         "cache_path": "/models/m/flyweight-1.cache"},
        {"name": "Q6_K", "arena_bytes": 22 * 1024**3, "cache_bytes": 0,
         "cache_path": "/models/m/flyweight-2.cache"},
        {"name": "F32", "arena_bytes": 109 * 1024**3, "cache_bytes": 0,
         "cache_path": "/models/m/flyweight-3.cache"},
    ]

    def setUp(self) -> None:
        self._previous = os.environ.pop("FLYWEIGHT_HF_QUANT", None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        os.environ.pop("FLYWEIGHT_HF_QUANT", None)
        if self._previous is not None:
            os.environ["FLYWEIGHT_HF_QUANT"] = self._previous

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

    def test_choosing_an_unavailable_option_names_its_reason(self) -> None:
        # "not one of the options" against a menu that plainly shows the
        # number reads as a broken prompt; the refusal has to say why.
        from flyweight.cli import _unavailable_reason

        options = [
            {"name": "IQ2_XS", "arena_bytes": 0, "cache_bytes": 0,
             "unavailable": "needs an importance matrix"},
            *self.OPTIONS,
        ]
        for answer in ("1", "iq2_xs", "IQ2XS"):
            with self.subTest(answer=answer):
                self.assertIsNone(
                    _quant_from_answer(answer, options, DEFAULT_QUANT))
                reason = _unavailable_reason(answer, options)
                self.assertIn("IQ2_XS is unavailable here", reason)
                self.assertIn("importance matrix", reason)
        # Answers that are merely wrong keep the generic rejection.
        self.assertEqual(_unavailable_reason("0", options), "")
        self.assertEqual(_unavailable_reason("yes", options), "")
        self.assertEqual(_unavailable_reason("2", options), "")

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
            with patch("flyweight.cli.V2Model") as model:
                _resolve_quant(self._args(directory, "Q4_K"))
            model.hf_quant_options.assert_not_called()
        self.assertEqual(os.environ["FLYWEIGHT_HF_QUANT"], "Q4_K")

    def test_a_pipe_is_left_with_the_loader_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.stdin.isatty", return_value=False), \
                 patch("flyweight.cli.V2Model") as model:
                _resolve_quant(self._args(directory))
            model.hf_quant_options.assert_not_called()
        self.assertNotIn("FLYWEIGHT_HF_QUANT", os.environ)

    def test_an_exported_variable_wins_over_the_prompt(self) -> None:
        os.environ["FLYWEIGHT_HF_QUANT"] = "Q8_0"
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.stdin.isatty", return_value=True), \
                 patch("sys.stderr.isatty", return_value=True), \
                 patch("flyweight.cli.V2Model") as model:
                _resolve_quant(self._args(directory))
            model.hf_quant_options.assert_not_called()
        self.assertEqual(os.environ["FLYWEIGHT_HF_QUANT"], "Q8_0")

    def test_a_gguf_file_is_never_asked_about(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "model.gguf"
            model_file.write_bytes(b"GGUF")
            with patch("sys.stdin.isatty", return_value=True), \
                 patch("sys.stderr.isatty", return_value=True), \
                 patch("flyweight.cli.V2Model") as model:
                _resolve_quant(SimpleNamespace(model=model_file, quant=None))
            model.hf_quant_options.assert_not_called()
        self.assertNotIn("FLYWEIGHT_HF_QUANT", os.environ)

    def test_ask_prompts_and_publishes_the_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("flyweight.cli.V2Model.hf_quant_options",
                       return_value=self.OPTIONS), \
                 patch("builtins.input", return_value="1"):
                _resolve_quant(self._args(directory, "ask"))
        self.assertEqual(os.environ["FLYWEIGHT_HF_QUANT"], "Q4_K")

    def test_a_rejected_answer_is_asked_again(self) -> None:
        answers = iter(("banana", "2"))
        with tempfile.TemporaryDirectory() as directory:
            with patch("flyweight.cli.V2Model.hf_quant_options",
                       return_value=self.OPTIONS), \
                 patch("builtins.input", side_effect=lambda: next(answers)):
                _resolve_quant(self._args(directory, "ask"))
        self.assertEqual(os.environ["FLYWEIGHT_HF_QUANT"], "Q6_K")

    def test_interrupting_the_question_stops_rather_than_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("flyweight.cli.V2Model.hf_quant_options",
                       return_value=self.OPTIONS), \
                 patch("builtins.input", side_effect=KeyboardInterrupt):
                with self.assertRaises(SystemExit) as caught:
                    _resolve_quant(self._args(directory, "ask"))
        self.assertEqual(caught.exception.code, 130)
        self.assertNotIn("FLYWEIGHT_HF_QUANT", os.environ)

    def test_a_checkpoint_that_cannot_be_described_is_left_to_the_loader(self) -> None:
        from flyweight.v2 import V2Error

        with tempfile.TemporaryDirectory() as directory:
            with patch("flyweight.cli.V2Model.hf_quant_options",
                       side_effect=V2Error("unsupported HF architecture")), \
                 patch("builtins.input", side_effect=AssertionError("must not ask")):
                _resolve_quant(self._args(directory, "ask"))
        self.assertNotIn("FLYWEIGHT_HF_QUANT", os.environ)

    def test_the_flag_is_available_to_every_command_that_takes_a_model(self) -> None:
        for command in ("generate", "serve", "benchmark", "inspect", "probe"):
            with self.subTest(command=command):
                argv = [command, "model.gguf", "--quant", "Q5_K"]
                if command == "generate":
                    argv.extend(("--prompt", "hello"))
                self.assertEqual(_parser().parse_args(argv).quant, "Q5_K")


if __name__ == "__main__":
    unittest.main()
