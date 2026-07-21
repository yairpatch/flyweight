import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from colibri_next.attention import AttentionKVCache, QwenFullAttentionLayer
from colibri_next.attention_converter import QwenAttentionConverter
from colibri_next.cli import main
from colibri_next.converter import QwenCheckpointConverter
from colibri_next.cuda import CudaUnavailableError, configure_cuda, disable_cuda
from colibri_next.q4 import Q4BlockTensor
from colibri_next.tensor_container import ColiTensorFile

from tests.test_converter import bf16_bytes, write_safetensors


def create_attention_checkpoint(root: Path) -> None:
    config = {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "num_experts": 1,
            "num_experts_per_tok": 1,
            "hidden_size": 2,
            "moe_intermediate_size": 2,
            "shared_expert_intermediate_size": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        }
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    prefix = "model.language_model.layers.0"
    query_rows = [
        1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
    ]
    tensors = {
        f"{prefix}.mlp.experts.gate_up_proj": (
            "BF16", (1, 4, 2), bf16_bytes([0.0] * 8)
        ),
        f"{prefix}.mlp.experts.down_proj": (
            "BF16", (1, 2, 2), bf16_bytes([0.0] * 4)
        ),
        f"{prefix}.input_layernorm.weight": (
            "BF16", (2,), bf16_bytes([0.0, 0.0])
        ),
        f"{prefix}.self_attn.q_proj.weight": (
            "BF16", (8, 2), bf16_bytes(query_rows)
        ),
        f"{prefix}.self_attn.k_proj.weight": (
            "BF16", (2, 2), bf16_bytes([1.0, 0.0, 0.0, 1.0])
        ),
        f"{prefix}.self_attn.v_proj.weight": (
            "BF16", (2, 2), bf16_bytes([1.0, 0.0, 0.0, 1.0])
        ),
        f"{prefix}.self_attn.o_proj.weight": (
            "BF16", (2, 4), bf16_bytes([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        ),
        f"{prefix}.self_attn.q_norm.weight": (
            "BF16", (2,), bf16_bytes([0.0, 0.0])
        ),
        f"{prefix}.self_attn.k_norm.weight": (
            "BF16", (2,), bf16_bytes([0.0, 0.0])
        ),
    }
    shard_name = "model-00001-of-00001.safetensors"
    write_safetensors(root / shard_name, tensors)
    index = {
        "metadata": {"total_size": sum(len(value[2]) for value in tensors.values())},
        "weight_map": {name: shard_name for name in tensors},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def rms_norm(vector: list[float], epsilon: float) -> list[float]:
    inverse = 1.0 / math.sqrt(
        sum(value * value for value in vector) / len(vector) + epsilon
    )
    return [value * inverse for value in vector]


class QwenAttentionTests(unittest.TestCase):
    def test_cache_clear_releases_device_storage_metadata(self) -> None:
        cache = AttentionKVCache(2, 4)
        cache.tokens = 3
        cache.cuda_keys = object()
        cache.cuda_values = object()
        cache.cuda_key_scales = object()
        cache.cuda_value_scales = object()
        cache.cuda_cache_type = "q8"
        cache.cuda_capacity = 256

        cache.clear()

        self.assertEqual(cache.length, 0)
        self.assertIsNone(cache.cuda_keys)
        self.assertIsNone(cache.cuda_values)
        self.assertIsNone(cache.cuda_key_scales)
        self.assertIsNone(cache.cuda_value_scales)
        self.assertIsNone(cache.cuda_cache_type)
        self.assertEqual(cache.cuda_capacity, 0)

    def test_converter_writes_attention_container_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_attention_checkpoint(source)
            QwenCheckpointConverter(source).convert(output)

            storage = QwenAttentionConverter(source).convert(output)

            self.assertEqual(storage["layers"], [0])
            container = ColiTensorFile(output / "attention_layers/layer-000.coli")
            self.assertEqual(container.metadata["num_attention_heads"], 2)
            self.assertEqual(container.metadata["num_key_value_heads"], 1)
            self.assertEqual(container.metadata["rotary_dim"], 2)
            self.assertEqual(container.tensors["q_proj.weight"].shape, (8, 2))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["attention_layer_storage"]["mode"],
                "bf16-full-attention",
            )

    def test_q4_attention_conversion_loads_projection_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_attention_checkpoint(source)
            storage = QwenAttentionConverter(source).convert(
                output, quantization="q4"
            )
            layer = QwenFullAttentionLayer.from_model_directory(output, 0)
            self.assertEqual(storage["mode"], "q4-full-attention")
            self.assertIsInstance(layer.q_projection, Q4BlockTensor)
            self.assertIsInstance(layer.o_projection, Q4BlockTensor)

    def test_incremental_attention_matches_reference_and_residual(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_attention_checkpoint(source)
            QwenAttentionConverter(source).convert(output)
            layer = QwenFullAttentionLayer.from_model_directory(output, 0)
            cache = layer.new_cache()

            first_hidden = [1.0, 0.0]
            first = layer.forward(first_hidden, 0, cache)
            first_value = rms_norm(first_hidden, 1e-6)
            self.assertEqual(cache.length, 1)
            self.assertEqual(first.attention_weights, ((1.0,), (1.0,)))
            self.assertAlmostEqual(first.output[0], first_value[0] * 0.5, places=5)
            self.assertAlmostEqual(first.output[1], 0.0, places=5)

            second_hidden = [0.0, 1.0]
            second = layer.forward_residual(second_hidden, 1, cache)
            normalized = rms_norm(second_hidden, 1e-6)
            query = rms_norm(normalized, 1e-6)
            old_key = rms_norm(first_value, 1e-6)
            current_key = query
            rotated_query = [
                -query[1] * math.sin(1.0),
                query[1] * math.cos(1.0),
            ]
            rotated_current_key = [
                -current_key[1] * math.sin(1.0),
                current_key[1] * math.cos(1.0),
            ]
            scores = [
                sum(left * right for left, right in zip(rotated_query, old_key))
                / math.sqrt(2.0),
                sum(
                    left * right
                    for left, right in zip(rotated_query, rotated_current_key)
                ) / math.sqrt(2.0),
            ]
            maximum = max(scores)
            exponentials = [math.exp(score - maximum) for score in scores]
            weights = [value / sum(exponentials) for value in exponentials]
            expected_branch = [
                0.5 * weights[0] * first_value[0],
                0.5 * weights[1] * normalized[1],
            ]
            self.assertEqual(cache.length, 2)
            self.assertAlmostEqual(second.attention_weights[0][0], weights[0], places=5)
            self.assertAlmostEqual(second.output[0], expected_branch[0], places=5)
            self.assertAlmostEqual(second.output[1], 1.0 + expected_branch[1], places=5)

    def test_cuda_attention_matches_incremental_cpu_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_attention_checkpoint(source)
            QwenAttentionConverter(source).convert(output)
            layer = QwenFullAttentionLayer.from_model_directory(output, 0)
            cpu_cache = layer.new_cache()
            inputs = ([1.0, 0.0], [0.0, 1.0])
            expected = [
                layer.forward_residual(list(hidden), position, cpu_cache)
                for position, hidden in enumerate(inputs)
            ]
            try:
                configure_cuda(cache_mib=64)
            except CudaUnavailableError as error:
                self.skipTest(str(error))
            try:
                cuda_cache = layer.new_cache()
                actual = [
                    layer.forward_residual(list(hidden), position, cuda_cache)
                    for position, hidden in enumerate(inputs)
                ]
            finally:
                disable_cuda()
            self.assertEqual(cuda_cache.length, cpu_cache.length)
            for expected_result, actual_result in zip(expected, actual):
                for expected_value, actual_value in zip(
                    expected_result.output, actual_result.output
                ):
                    self.assertAlmostEqual(expected_value, actual_value, places=4)
                for expected_head, actual_head in zip(
                    expected_result.attention_weights,
                    actual_result.attention_weights,
                ):
                    for expected_weight, actual_weight in zip(
                        expected_head, actual_head
                    ):
                        self.assertAlmostEqual(
                            expected_weight, actual_weight, places=4
                        )

    def test_cli_converts_and_benchmarks_attention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_attention_checkpoint(source)
            capture = StringIO()
            with redirect_stdout(capture):
                self.assertEqual(
                    main(["convert-attention-layers", str(source), str(output)]),
                    0,
                )
            capture = StringIO()
            with redirect_stdout(capture):
                self.assertEqual(
                    main([
                        "benchmark-attention-layer",
                        str(output),
                        "--layer", "0",
                        "--tokens", "2",
                    ]),
                    0,
                )
            report = json.loads(capture.getvalue())
            self.assertEqual(report["cache_length"], 2)


if __name__ == "__main__":
    unittest.main()
