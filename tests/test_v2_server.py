from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from colibri_next.cli import _parser
from colibri_next.sampling import SamplingConfig
from colibri_next.server import InferenceService
from colibri_next.v2_server import (
    NativeV2Generator,
    NativeV2InferenceService,
    NativeV2Tokenizer,
    _generation_config_for_model,
)
from colibri_next.v2 import TASK_EVENT_PREFILL


class StubV2Model:
    info = {"architecture": "qwen3moe"}
    pieces = {20: "Hello", 30: " world", 99: "<|im_end|>"}
    # Raw UTF-8 bytes per token; 40/41 split one character ("⚽" = e2 9a bd)
    # across two tokens, as byte-level BPE routinely does.
    piece_bytes = {
        20: b"Hello",
        30: b" world",
        40: b"\xe2\x9a",
        41: b"\xbd",
        99: b"<|im_end|>",
    }

    def tokenize(self, text: str) -> list[int]:
        return [1, 2]

    def decode_tokens(self, tokens: list[int]) -> str:
        return "".join(self.pieces.get(token, "") for token in tokens)

    def decode_token_bytes(self, tokens: list[int]) -> bytes:
        return b"".join(self.piece_bytes.get(token, b"") for token in tokens)

    def token_id(self, text: str) -> int:
        if text in ("<|im_end|>", "<|endoftext|>"):
            return 99
        raise KeyError(text)


class StubV2Runtime:
    def __init__(self, outputs: list[int]):
        self.outputs = iter(outputs)
        self.inputs: list[int] = []
        self.resets = 0
        self.cancels = 0

    @property
    def info(self) -> dict[str, int]:
        return {
            "position": len(self.inputs),
            "prefix_cache_hits": 3,
            "prefix_cache_misses": 1,
            "prefix_cache_reused_tokens": 42,
        }

    def reset(self) -> None:
        self.resets += 1

    def decode(self, token: int) -> int:
        self.inputs.append(token)
        return next(self.outputs)

    def generate(self, prompt: list[int], max_tokens: int, callback) -> None:
        self.reset()
        next_token = 0
        for token in prompt:
            next_token = self.decode(token)
        for index in range(max_tokens):
            if callback(next_token) is False:
                break
            if index + 1 < max_tokens:
                next_token = self.decode(next_token)

    def cancel(self) -> None:
        self.cancels += 1

    # Cooperative-engine API mirroring the native semantics (prefill the whole
    # prompt, then per step: emit token; stop on stop-token or max; else decode
    # the emitted token as the next input).
    def task_submit(
        self,
        prompt,
        max_tokens,
        stop_tokens=(),
        *,
        temperature=0.0,
        top_k=20,
        top_p=0.95,
        seed=None,
    ):
        self._tasks = getattr(self, "_tasks", {})
        self._next_task = getattr(self, "_next_task", 0) + 1
        self._last_sampling = (temperature, top_k, top_p, seed)
        self._tasks[self._next_task] = (list(prompt), max_tokens, tuple(stop_tokens))
        return self._next_task

    def engine_step(self, capacity: int = 256):
        events = []
        for task_id, (prompt, max_tokens, stops) in list(
            getattr(self, "_tasks", {}).items()
        ):
            self.reset()
            next_token = 0
            for token in prompt:
                next_token = self.decode(token)
            emitted = 0
            while True:
                events.append((task_id, next_token, 0))
                emitted += 1
                if next_token in stops or emitted >= max_tokens:
                    break
                next_token = self.decode(next_token)
            events.append((task_id, 0, 1))
            del self._tasks[task_id]
        return events

    def task_cancel(self, task_id: int) -> None:
        self.cancels += 1
        getattr(self, "_tasks", {}).pop(task_id, None)


class BlockingV2Runtime(StubV2Runtime):
    def __init__(self):
        super().__init__([10])
        self.entered = threading.Event()
        self.cancelled = threading.Event()

    def engine_step(self, capacity: int = 256):
        self.entered.set()
        self.cancelled.wait(2)
        return []

    def task_cancel(self, task_id: int) -> None:
        super().task_cancel(task_id)
        self.cancelled.set()


class ProgressV2Runtime(StubV2Runtime):
    def engine_step(self, capacity: int = 256):
        events = []
        for task_id, (prompt, _max_tokens, _stops) in list(self._tasks.items()):
            events.extend(
                (task_id, processed, TASK_EVENT_PREFILL)
                for processed in (0, 1, len(prompt))
            )
            events.extend(((task_id, 20, 0), (task_id, 0, 1)))
            del self._tasks[task_id]
        return events


class NativeV2ServerTests(unittest.TestCase):
    def make_generator(self, outputs: list[int]):
        model = StubV2Model()
        runtime = StubV2Runtime(outputs)
        tokenizer = NativeV2Tokenizer(model)  # type: ignore[arg-type]
        generator = NativeV2Generator(  # type: ignore[arg-type]
            model, runtime, tokenizer
        )
        return generator, runtime

    def test_generation_config_adjacent_to_model_supplies_sampling_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.touch()
            config = root / "generation_config.json"
            config.write_text(
                json.dumps(
                    {
                        "do_sample": True,
                        "temperature": 0.6,
                        "top_k": 20,
                        "top_p": 0.9,
                        "max_new_tokens": 512,
                    }
                ),
                encoding="utf-8",
            )

            defaults, source = _generation_config_for_model(model)

        self.assertEqual(
            defaults,
            {"temperature": 0.6, "top_k": 20, "top_p": 0.9, "max_new_tokens": 512},
        )
        self.assertEqual(source, str(config))

    def test_generation_config_disables_sampling_when_do_sample_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.touch()
            config = root / "model.generation_config.json"
            config.write_text(
                json.dumps({"do_sample": False, "temperature": 0.7}),
                encoding="utf-8",
            )

            defaults, _ = _generation_config_for_model(model)

        self.assertEqual(defaults["temperature"], 0.0)

    def test_health_exposes_resolved_expert_policy(self) -> None:
        class Runtime:
            info = {
                "expert_mode": "cpu",
                "requested_expert_mode": "auto",
                "expert_fallback_reason": "working set did not fit",
                "moe_device": 1,
                "position": 0,
                "prefix_cache_hits": 0,
                "prefix_cache_misses": 0,
                "prefix_cache_reused_tokens": 0,
            }

        service = object.__new__(NativeV2InferenceService)
        InferenceService.__init__(
            service,
            "native-test",
            self.make_generator([20])[0],
            context_window=128,
        )
        service.v2_runtime = Runtime()
        service.expert_mode = "cpu"
        service.requested_expert_mode = "auto"
        service.expert_fallback_reason = "working set did not fit"
        service.moe_device = "cpu"
        service.mtp_drafts = 0
        service.gpu_cache_mib = 0
        execution = service.health()["execution"]
        self.assertEqual(execution["expert_mode"], "cpu")
        self.assertEqual(execution["requested_expert_mode"], "auto")
        self.assertEqual(
            execution["expert_fallback_reason"], "working set did not fit"
        )

    def test_native_generator_prefills_and_greedily_decodes(self) -> None:
        generator, runtime = self.make_generator([10, 20, 30])
        result = generator.generate_messages(
            [{"role": "user", "content": "Hi"}],
            max_new_tokens=2,
            sampling=SamplingConfig(temperature=0),
        )
        self.assertEqual(result.generated_ids, (20, 30))
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(runtime.inputs, [1, 2, 20])
        self.assertEqual(runtime.resets, 1)

    def test_native_generator_reports_live_prefill_progress(self) -> None:
        model = StubV2Model()
        runtime = ProgressV2Runtime([20])
        generator = NativeV2Generator(  # type: ignore[arg-type]
            model, runtime, NativeV2Tokenizer(model)  # type: ignore[arg-type]
        )
        progress: list[tuple[int, int]] = []

        result = generator.generate_text(
            "Hi", max_new_tokens=1, progress=lambda done, total: progress.append(
                (done, total)
            )
        )

        self.assertEqual(result.generated_ids, (20,))
        self.assertEqual(progress, [(0, 2), (1, 2), (2, 2)])

    def test_stream_steps_use_stable_constant_time_token_snapshots(self) -> None:
        generator, _ = self.make_generator([10, 20, 30])
        steps = list(generator.stream_text("Hi", max_new_tokens=2))
        live = [step for step in steps if not step.finished]
        self.assertEqual(tuple(live[0].generated_ids), (20,))
        self.assertEqual(tuple(live[1].generated_ids), (20, 30))
        self.assertEqual(live[0].text, "")
        self.assertEqual(steps[-1].generated_ids, (20, 30))
        self.assertEqual(steps[-1].text, "Hello world")

    def test_gemma4_chat_format_uses_turn_and_channel_tokens(self) -> None:
        tokenizer = object.__new__(NativeV2Tokenizer)
        tokenizer.architecture = "gemma4"
        prompt = tokenizer.format_messages(
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ]
        )
        self.assertEqual(
            prompt,
            "<bos><|turn>system\nBe concise.<turn|>\n"
            "<|turn>user\nHello<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>",
        )

    def test_qwen_chat_format_controls_thinking_mode(self) -> None:
        tokenizer = object.__new__(NativeV2Tokenizer)
        tokenizer.architecture = "qwen3moe"
        messages = [{"role": "user", "content": "Explain MoE simply."}]

        thinking = tokenizer.format_messages(messages, enable_thinking=True)
        direct = tokenizer.format_messages(messages, enable_thinking=False)

        self.assertTrue(thinking.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertTrue(
            direct.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        )

    def test_gguf_chat_template_takes_precedence_over_architecture_fallback(self) -> None:
        class TemplateModel(StubV2Model):
            config = {}
            chat_template = (
                "{% for message in messages %}[{{ message.role }}]"
                "{{ message.content }}{% endfor %}"
                "{% if add_generation_prompt %}{% generation %}[assistant]"
                "{% if enable_thinking %}<think>{% endif %}"
                "{% endgeneration %}{% endif %}"
            )

        tokenizer = NativeV2Tokenizer(TemplateModel())  # type: ignore[arg-type]
        prompt = tokenizer.format_messages(
            [{"role": "user", "content": "Hello"}], enable_thinking=True
        )

        self.assertEqual(prompt, "[user]Hello[assistant]<think>")
        self.assertEqual(tokenizer.chat_template_source, "gguf")

    def test_gguf_chat_template_can_access_missing_tool_calls(self) -> None:
        class TemplateModel(StubV2Model):
            config = {}
            chat_template = (
                "{% for message in messages %}[{{ message.role }}]"
                "{{ message.content }}"
                "{% if message.tool_calls %}[tools]{% endif %}{% endfor %}"
                "{% if add_generation_prompt %}[assistant]{% endif %}"
            )

        tokenizer = NativeV2Tokenizer(TemplateModel())  # type: ignore[arg-type]

        prompt = tokenizer.format_messages(
            [{"role": "user", "content": "Hello"}]
        )

        self.assertEqual(prompt, "[user]Hello[assistant]")

    def test_multibyte_character_split_across_tokens_decodes_intact(self) -> None:
        # Byte-level BPE splits "⚽" (e2 9a bd) across tokens 40+41. Per-token
        # string decoding turned each half into U+FFFD, and that corruption was
        # written into files by tool calls. The incremental UTF-8 decoder must
        # reassemble the character.
        generator, _ = self.make_generator([0, 40, 41, 99])
        result = generator.generate_text("Hi", max_new_tokens=8)
        self.assertEqual(result.text, "⚽")
        self.assertNotIn("�", result.text)

    def test_native_generator_stops_on_eos(self) -> None:
        generator, runtime = self.make_generator([10, 99])
        result = generator.generate_text("Hi", max_new_tokens=4)
        self.assertEqual(result.generated_ids, (99,))
        self.assertEqual(result.text, "")
        self.assertTrue(result.stopped_on_eos)
        self.assertEqual(runtime.inputs, [1, 2])

    def test_native_generator_forwards_sampling_to_engine(self) -> None:
        generator, runtime = self.make_generator([10, 20])
        result = generator.generate_text(
            "Hi",
            max_new_tokens=1,
            sampling=SamplingConfig(
                temperature=0.5, top_k=7, top_p=0.8, seed=42
            ),
        )
        self.assertIsNotNone(result)
        self.assertEqual(runtime._last_sampling, (0.5, 7, 0.8, 42))

    def test_native_generator_reports_runtime_prefix_cache(self) -> None:
        generator, _ = self.make_generator([10])
        self.assertEqual(
            generator.prefix_cache_stats(),
            {
                "entries": 0,
                "capacity": 1,
                "hits": 3,
                "misses": 1,
                "evictions": 0,
                "reused_tokens": 42,
                "last_prompt_tokens": 0,
                "last_reused_tokens": 0,
                "last_lcp_live": 0,
                "last_lcp_snapshot": 0,
            },
        )

    def test_close_cancels_tasks_and_joins_engine_before_runtime_teardown(self) -> None:
        model = StubV2Model()
        runtime = BlockingV2Runtime()
        generator = NativeV2Generator(  # type: ignore[arg-type]
            model, runtime, NativeV2Tokenizer(model)  # type: ignore[arg-type]
        )
        _, queue = generator.engine.submit([1, 2], 8, ())
        self.assertTrue(runtime.entered.wait(1))

        generator.close()

        self.assertEqual(runtime.cancels, 1)
        self.assertEqual(queue.get(timeout=1)[0], "error")
        self.assertIsNotNone(generator.engine._thread)
        self.assertFalse(generator.engine._thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            generator.engine.submit([1, 2], 8, ())
        generator.close()  # idempotent

    def test_native_chat_continuation_preserves_generated_token_ids(self) -> None:
        generator, _ = self.make_generator([10, 20, 30])
        first_messages = (("user", "Hi"),)
        generator.generate_messages(
            [{"role": "user", "content": "Hi"}], max_new_tokens=2
        )
        continued = generator._continued_chat_prompt(
            (
                *first_messages,
                ("assistant", "Hello world"),
                ("user", "Again"),
            ),
            False,
        )
        self.assertEqual(continued, [1, 2, 20, 30, 1, 2, 1, 2])

    def test_native_tool_call_round_trip_preserves_generated_token_ids(self) -> None:
        generator, _ = self.make_generator([10])
        generator._chat_messages = (("user", "Write it"),)
        generator._chat_prompt_ids = (7, 8)
        generator._chat_generated_ids = (20, 30)
        generator._chat_text = (
            "Internal reasoning that the API may omit.\n"
            "<tool_call>\n<function=write_file>\n"
            "<parameter=path>\n/tmp/example\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        generator._chat_thinking = False

        continued = generator._continued_chat_prompt(
            (
                ("user", "Write it"),
                (
                    "assistant",
                    "<tool_call>\n<function=write_file>\n"
                    "<parameter=path>\n/tmp/example\n</parameter>\n"
                    "</function>\n</tool_call>",
                ),
                ("user", "<tool_response>\ndone\n</tool_response>"),
            ),
            False,
        )

        self.assertEqual(continued, [7, 8, 20, 30, 1, 2, 1, 2])

    def test_concurrent_side_chat_does_not_evict_main_continuation(self) -> None:
        generator, _ = self.make_generator([10])
        main = (("user", "Long main conversation"),)
        generator._chat_continuations[main] = (
            (7, 8),
            (20, 30),
            "Hello world",
            False,
        )
        # A later short request owns the legacy last-writer fields.
        generator._chat_messages = (("user", "Make a title"),)
        generator._chat_prompt_ids = (90,)
        generator._chat_generated_ids = (91,)
        generator._chat_text = "Title"
        generator._chat_thinking = False

        continued = generator._continued_chat_prompt(
            (
                *main,
                ("assistant", "Hello world"),
                ("user", "Continue"),
            ),
            False,
        )

        self.assertEqual(continued, [7, 8, 20, 30, 1, 2, 1, 2])

    def test_native_generator_works_through_server_contract(self) -> None:
        generator, _ = self.make_generator([10, 20, 30])
        service = InferenceService(
            "native-test", generator, max_new_tokens=8, context_window=32
        )
        response = service.chat_completion(
            {
                "model": "native-test",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 2,
                "temperature": 0,
            }
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(response["usage"]["completion_tokens"], 2)

    def test_serve_v2_cli_defaults_to_auto_expert_mode(self) -> None:
        args = _parser().parse_args(["serve-v2", "model.gguf"])
        self.assertEqual(args.expert_mode, "auto")
        # 0 = auto-fit the GPU expert cache to free VRAM (manual MiB still settable).
        self.assertEqual(args.gpu_cache_mib, 0)
        self.assertEqual(args.context_window, 32768)
        self.assertEqual(args.mtp_drafts, 0)
        self.assertIsNone(args.prefill_cache_seed)
        self.assertEqual(args.expert_paging, "auto")
        self.assertEqual(args.cpu_prefetch_mib, 0)
        self.assertFalse(args.cpu_prefetch_auto)
        self.assertEqual(args.next_layer_prefetch, 0)
        self.assertEqual(args.cpu_threads, 0)
        self.assertEqual(args.hybrid_prefill, "split")
        self.assertIsNone(args.expert_residency)
        self.assertEqual(args.dense_requant, "auto")

    def test_benchmark_v2_exposes_native_runtime_tuning_options(self) -> None:
        defaults = _parser().parse_args(["benchmark-v2", "model.gguf"])
        self.assertEqual(defaults.gpu_cache_mib, 0)

        args = _parser().parse_args([
            "benchmark-v2", "model.gguf",
            "--cache-type-k", "f32",
            "--cache-type-v", "q8_0",
            "--expert-top-k", "6",
            "--expert-top-p", "0.9",
            "--parallel", "2",
            "--prompt-cache-mib", "512",
            "--prefill-cache-seed", "8",
            "--expert-paging", "direct",
            "--cpu-prefetch-mib", "768",
            "--next-layer-prefetch", "6",
            "--hybrid-prefill", "cpu",
            "--expert-residency", "immutable",
            "--dense-requant", "off",
            '--cpu-threads', '12',
            "--cold-cache",
        ])
        self.assertEqual(args.cache_type_k, "f32")
        self.assertEqual(args.cache_type_v, "q8_0")
        self.assertEqual(args.expert_top_k, 6)
        self.assertAlmostEqual(args.expert_top_p, 0.9)
        self.assertEqual(args.parallel_sequences, 2)
        self.assertEqual(args.prompt_cache_mib, 512)
        self.assertEqual(args.prefill_cache_seed, 8)
        self.assertEqual(args.expert_paging, "direct")
        self.assertEqual(args.cpu_prefetch_mib, 768)
        self.assertEqual(args.next_layer_prefetch, 6)
        self.assertEqual(args.cpu_threads, 12)
        self.assertEqual(args.hybrid_prefill, "cpu")
        self.assertEqual(args.expert_residency, "immutable")
        self.assertEqual(args.dense_requant, "off")
        self.assertTrue(args.cold_cache)
        self.assertFalse(args.cpu_prefetch_auto)

    def test_benchmark_v2_exposes_auto_cpu_prefetch(self) -> None:
        args = _parser().parse_args([
            "benchmark-v2", "model.gguf", "--cpu-prefetch-auto",
        ])
        self.assertEqual(args.cpu_prefetch_mib, 0)
        self.assertTrue(args.cpu_prefetch_auto)

    def test_serve_v2_cli_accepts_legacy_gpu_alias(self) -> None:
        args = _parser().parse_args([
            "serve-v2", "model.gguf",
            "--moe-device", "gpu",
            "--expert-paging", "direct",
            "--next-layer-prefetch", "4",
        ])
        self.assertEqual(args.expert_mode, "gpu")
        self.assertEqual(args.expert_paging, "direct")
        self.assertEqual(args.next_layer_prefetch, 4)

    def test_serve_v2_cli_accepts_canonical_expert_modes(self) -> None:
        for mode in ("cpu", "auto", "resident"):
            with self.subTest(mode=mode):
                args = _parser().parse_args(
                    ["serve-v2", "model.gguf", "--expert-mode", mode]
                )
                self.assertEqual(args.expert_mode, mode)


if __name__ == "__main__":
    unittest.main()
