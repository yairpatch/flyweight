from __future__ import annotations

import unittest

from colibri_next.cli import _parser
from colibri_next.sampling import SamplingConfig
from colibri_next.server import APIError, InferenceService
from colibri_next.v2_server import NativeV2Generator, NativeV2Tokenizer


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
    def task_submit(self, prompt, max_tokens, stop_tokens=()):
        self._tasks = getattr(self, "_tasks", {})
        self._next_task = getattr(self, "_next_task", 0) + 1
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


class NativeV2ServerTests(unittest.TestCase):
    def make_generator(self, outputs: list[int]):
        model = StubV2Model()
        runtime = StubV2Runtime(outputs)
        tokenizer = NativeV2Tokenizer(model)  # type: ignore[arg-type]
        generator = NativeV2Generator(  # type: ignore[arg-type]
            model, runtime, tokenizer
        )
        return generator, runtime

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

    def test_native_generator_accepts_and_ignores_sampling(self) -> None:
        # The native path is greedy; sampling params are accepted and ignored
        # (clamped to greedy) rather than rejected, so agentic clients that
        # always send a temperature still work.
        generator, _ = self.make_generator([10, 20])
        result = generator.generate_text(
            "Hi",
            max_new_tokens=1,
            sampling=SamplingConfig(temperature=0.5),
        )
        self.assertIsNotNone(result)

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
            },
        )

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

    def test_serve_v2_cli_defaults_to_hybrid_autofit(self) -> None:
        args = _parser().parse_args(["serve-v2", "model.gguf"])
        self.assertEqual(args.moe_device, "hybrid")
        # 0 = auto-fit the GPU expert cache to free VRAM (manual MiB still settable).
        self.assertEqual(args.gpu_cache_mib, 0)
        self.assertEqual(args.context_window, 32768)
        self.assertEqual(args.mtp_drafts, 0)
        self.assertEqual(args.prefill_cache_seed, 0)
        self.assertEqual(args.expert_paging, "auto")
        self.assertEqual(args.cpu_prefetch_mib, 0)

    def test_benchmark_v2_exposes_native_runtime_tuning_options(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
