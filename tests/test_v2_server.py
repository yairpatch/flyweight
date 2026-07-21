from __future__ import annotations

import unittest

from colibri_next.cli import _parser
from colibri_next.sampling import SamplingConfig
from colibri_next.server import APIError, InferenceService
from colibri_next.v2_server import NativeV2Generator, NativeV2Tokenizer


class StubV2Model:
    pieces = {20: "Hello", 30: " world", 99: "<|im_end|>"}

    def tokenize(self, text: str) -> list[int]:
        return [1, 2]

    def decode_tokens(self, tokens: list[int]) -> str:
        return "".join(self.pieces.get(token, "") for token in tokens)

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


if __name__ == "__main__":
    unittest.main()
