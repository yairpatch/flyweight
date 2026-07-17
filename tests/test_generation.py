import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from colibri_next.causal_lm import QwenForCausalLM
from colibri_next.cli import main
from colibri_next.generation import TextGenerator
from colibri_next.model_io_converter import QwenModelIOConverter
from colibri_next.q4 import np
from colibri_next.sampling import LogitsSampler, SamplingConfig
from colibri_next.tokenizer import HuggingFaceTokenizer
from colibri_next.tokenizer_converter import TokenizerAssetsConverter

from tests.test_decoder import create_decoder_checkpoint, convert_decoder


class FakeEncoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class FakeBackend:
    @classmethod
    def from_file(cls, path: str) -> "FakeBackend":
        return cls()

    def encode(self, text: str, add_special_tokens: bool = False) -> FakeEncoding:
        return FakeEncoding([ord(character) % 4 for character in text])

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = True
    ) -> str:
        return "".join("ABCD"[token_id] for token_id in token_ids)

    def token_to_id(self, token: str) -> int | None:
        return 3 if token == "<eos>" else None


def create_tokenizer_assets(source: Path) -> None:
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<eos>"}), encoding="utf-8"
    )
    (source / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [99]}), encoding="utf-8"
    )
    (source / "chat_template.jinja").write_text("fake", encoding="utf-8")


def convert_generation_model(source: Path, output: Path) -> None:
    convert_decoder(source, output)
    QwenModelIOConverter(source).convert(output)
    TokenizerAssetsConverter(source).convert(output)


class SamplingTests(unittest.TestCase):
    def test_greedy_and_seeded_sampling_are_deterministic(self) -> None:
        logits = [0.0, 3.0, 2.0, 1.0]
        self.assertEqual(
            LogitsSampler(SamplingConfig(temperature=0.0)).sample(logits), 1
        )
        self.assertEqual(
            LogitsSampler(
                SamplingConfig(temperature=1.0, top_k=1, seed=7)
            ).sample(logits),
            1,
        )
        first = LogitsSampler(
            SamplingConfig(temperature=1.0, top_k=4, top_p=0.9, seed=42)
        ).sample(logits)
        second = LogitsSampler(
            SamplingConfig(temperature=1.0, top_k=4, top_p=0.9, seed=42)
        ).sample(logits)
        self.assertEqual(first, second)

    @unittest.skipIf(np is None, "NumPy is unavailable")
    def test_device_sampling_keeps_logits_off_host(self) -> None:
        accelerator = type("Accelerator", (), {"cp": np})()
        logits = np.asarray([0.0, 3.0, 2.0, 1.0], dtype=np.float32)
        greedy = LogitsSampler(SamplingConfig(temperature=0.0))
        sampled = LogitsSampler(
            SamplingConfig(temperature=1.0, top_k=1, seed=7)
        )
        self.assertEqual(greedy.sample_device(logits, accelerator), 1)
        self.assertEqual(sampled.sample_device(logits, accelerator), 1)


class TextGenerationTests(unittest.TestCase):
    def test_assets_copy_and_chat_formatting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            create_tokenizer_assets(source)
            storage = TokenizerAssetsConverter(source).convert(output)
            tokenizer = HuggingFaceTokenizer.from_model_directory(
                output, backend=FakeBackend()
            )
            formatted = tokenizer.format_chat(
                "Hello", system="Be concise", enable_thinking=False
            )
            self.assertEqual(storage["mode"], "huggingface-tokenizer-json")
            self.assertIn("<|im_start|>system\nBe concise", formatted)
            self.assertIn("<|im_start|>user\nHello", formatted)
            self.assertTrue(formatted.endswith("<think>\n\n</think>\n\n"))
            self.assertEqual(tokenizer.eos_token_ids, (99,))

    def test_complete_generation_loop_returns_decoded_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            create_tokenizer_assets(source)
            convert_generation_model(source, output)
            tokenizer = HuggingFaceTokenizer.from_model_directory(
                output, backend=FakeBackend()
            )
            model = QwenForCausalLM.from_model_directory(output, rows_per_chunk=2)
            result = TextGenerator(model, tokenizer).generate(
                "Hi",
                max_new_tokens=3,
                sampling=SamplingConfig(temperature=0.0),
            )
            self.assertEqual(len(result.generated_ids), 3)
            self.assertEqual(
                result.text,
                FakeBackend().decode(list(result.generated_ids)),
            )
            self.assertEqual(
                result.state_tokens,
                len(result.prompt_ids) + len(result.generated_ids) - 1,
            )

    def test_cli_copies_tokenizer_and_generates_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            create_tokenizer_assets(source)
            convert_decoder(source, output)
            QwenModelIOConverter(source).convert(output)
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    main(["convert-tokenizer", str(source), str(output)]), 0
                )
            capture = StringIO()
            with patch("colibri_next.tokenizer.BackendTokenizer", FakeBackend):
                with redirect_stdout(capture), redirect_stderr(StringIO()):
                    self.assertEqual(
                        main([
                            "generate-text",
                            str(output),
                            "--prompt", "Hi",
                            "--max-new-tokens", "2",
                            "--rows-per-chunk", "2",
                        ]),
                        0,
                    )
            report = json.loads(capture.getvalue())
            self.assertEqual(len(report["generated_tokens"]), 2)
            self.assertTrue(report["text"])


if __name__ == "__main__":
    unittest.main()
