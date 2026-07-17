import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from colibri_next.bf16 import BF16Tensor
from colibri_next.causal_lm import QwenForCausalLM
from colibri_next.cli import main
from colibri_next.decoder import QwenDecoderStack
from colibri_next.model_io import QwenModelIO
from colibri_next.model_io_converter import QwenModelIOConverter
from colibri_next.q4 import Q4BlockTensor
from colibri_next.tensor_container import ColiTensorFile

from tests.test_converter import bf16_bytes
from tests.test_decoder import create_decoder_checkpoint, convert_decoder


class BF16LargeMatrixTests(unittest.TestCase):
    def test_row_lookup_and_chunked_matvec_match_regular_execution(self) -> None:
        tensor = BF16Tensor(
            (4, 2),
            bf16_bytes([1.0, 0.0, 0.0, 1.0, 0.5, -0.5, -1.0, 0.25]),
        )
        vector = [0.75, -0.25]
        expected = tensor.matvec(vector)
        self.assertEqual(tensor.row(2), [0.5, -0.5])
        for chunk_size in (1, 2, 3, 8):
            actual = tensor.matvec_chunked(vector, rows_per_chunk=chunk_size)
            for expected_value, actual_value in zip(expected, actual):
                self.assertAlmostEqual(actual_value, expected_value, places=6)
        portable = tensor.matvec_chunked(vector, prefer_numpy=False)
        for expected_value, portable_value in zip(expected, portable):
            self.assertAlmostEqual(portable_value, expected_value, places=6)


    def test_q4_row_lookup_matches_full_dequantization(self) -> None:
        tensor = Q4BlockTensor.from_bf16(
            bf16_bytes([1.0, 0.0, 0.0, 1.0, 0.5, -0.5]),
            (3, 2),
        )
        values = tensor.dequantize()
        for row in range(3):
            expected = values[row * 2 : row * 2 + 2]
            self.assertEqual(tensor.row(row), expected)


class QwenCausalLMTests(unittest.TestCase):
    def test_converter_writes_model_io_container_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            convert_decoder(source, output)

            storage = QwenModelIOConverter(source).convert(output)

            self.assertEqual(storage["mode"], "bf16-model-io")
            container = ColiTensorFile(output / "model_io.coli")
            self.assertEqual(container.tensors["embed_tokens.weight"].shape, (4, 2))
            self.assertEqual(container.tensors["norm.weight"].shape, (2,))
            self.assertEqual(container.tensors["lm_head.weight"].shape, (4, 2))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["model_io_storage"]["path"], "model_io.coli")

    def test_q4_model_io_conversion_loads_mixed_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            convert_decoder(source, output)
            storage = QwenModelIOConverter(source).convert(
                output, quantization="q4"
            )
            model_io = QwenModelIO.from_model_directory(output)
            self.assertEqual(storage["mode"], "q4-model-io")
            self.assertIsInstance(model_io.embedding, Q4BlockTensor)
            self.assertIsInstance(model_io.lm_head, Q4BlockTensor)
            self.assertEqual(len(model_io.embed(2)), model_io.hidden_size)

    def test_token_ids_to_logits_matches_manual_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            convert_decoder(source, output)
            QwenModelIOConverter(source).convert(output)
            model = QwenForCausalLM.from_model_directory(output, rows_per_chunk=2)
            state = model.new_state()
            manual_decoder = QwenDecoderStack.from_model_directory(output)
            manual_state = manual_decoder.new_state()
            model_io = QwenModelIO.from_model_directory(output, rows_per_chunk=2)

            for token_id in (0, 1):
                decoded = manual_decoder.forward_token(
                    model_io.embed(token_id), manual_state
                )
                expected_hidden = model_io.normalize(decoded.output)
                expected_logits = model_io.lm_head.matvec_chunked(
                    expected_hidden, rows_per_chunk=2
                )
                result = model.forward_token(token_id, state)
                for expected, actual in zip(expected_hidden, result.hidden):
                    self.assertAlmostEqual(actual, expected, places=5)
                for expected, actual in zip(expected_logits, result.logits):
                    self.assertAlmostEqual(actual, expected, places=5)
                self.assertEqual(
                    result.greedy_token,
                    max(range(4), key=result.logits.__getitem__),
                )
            self.assertEqual(state.tokens, 2)

    def test_prefill_projects_only_final_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            convert_decoder(source, output)
            QwenModelIOConverter(source).convert(output)
            model = QwenForCausalLM.from_model_directory(output, rows_per_chunk=2)
            reference = QwenForCausalLM.from_model_directory(
                output, rows_per_chunk=2
            )

            state = model.new_state()
            calls = 0
            original_forward = model.forward_token

            def counted_forward(token_id, current_state):
                nonlocal calls
                calls += 1
                return original_forward(token_id, current_state)

            model.forward_token = counted_forward
            result = model.prefill([0, 1, 2], state)

            reference_state = reference.new_state()
            reference_result = reference.forward_ids([0, 1, 2], reference_state)[-1]
            self.assertEqual(calls, 1)
            self.assertEqual(state.tokens, 3)
            for expected, actual in zip(reference_result.logits, result.logits):
                self.assertAlmostEqual(actual, expected, places=5)

    def test_cli_converts_model_io_and_executes_logits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            convert_decoder(source, output)
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    main(["convert-model-io", str(source), str(output)]),
                    0,
                )
            capture = StringIO()
            with redirect_stdout(capture):
                self.assertEqual(
                    main([
                        "benchmark-logits",
                        str(output),
                        "--token-ids", "0,1",
                        "--rows-per-chunk", "2",
                    ]),
                    0,
                )
            report = json.loads(capture.getvalue())
            self.assertEqual(report["state_tokens"], 2)
            self.assertEqual(report["vocab_size"], 4)
            self.assertIn(report["top_token"], range(4))


if __name__ == "__main__":
    unittest.main()
