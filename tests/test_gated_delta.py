import json
import math
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import colibri_next.gated_delta as gated_delta_module
from colibri_next.cli import main
from colibri_next.converter import QwenCheckpointConverter
from colibri_next.cuda import CudaUnavailableError, configure_cuda, disable_cuda
from colibri_next.gated_delta import QwenGatedDeltaLayer
from colibri_next.gated_delta_converter import QwenGatedDeltaConverter
from colibri_next.q4 import Q4BlockTensor
from colibri_next.tensor_container import ColiTensorFile

from tests.test_converter import bf16_bytes, write_safetensors


def create_gated_delta_checkpoint(root: Path) -> None:
    config = {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 1,
            "layer_types": ["linear_attention"],
            "num_experts": 1,
            "num_experts_per_tok": 1,
            "hidden_size": 2,
            "moe_intermediate_size": 2,
            "shared_expert_intermediate_size": 2,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 2,
            "rms_norm_eps": 1e-6,
        }
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    prefix = "model.language_model.layers.0"
    qkv_rows = [
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
        0.0, 1.0,
        1.0, 0.0,
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
        f"{prefix}.linear_attn.in_proj_qkv.weight": (
            "BF16", (8, 2), bf16_bytes(qkv_rows)
        ),
        f"{prefix}.linear_attn.in_proj_z.weight": (
            "BF16", (4, 2), bf16_bytes([1.0, 0.0, 0.0, 1.0] * 2)
        ),
        f"{prefix}.linear_attn.in_proj_b.weight": (
            "BF16", (2, 2), bf16_bytes([0.0] * 4)
        ),
        f"{prefix}.linear_attn.in_proj_a.weight": (
            "BF16", (2, 2), bf16_bytes([0.0] * 4)
        ),
        f"{prefix}.linear_attn.conv1d.weight": (
            "BF16", (8, 1, 2), bf16_bytes([0.25, 1.0] * 8)
        ),
        f"{prefix}.linear_attn.dt_bias": (
            "BF16", (2,), bf16_bytes([0.0, 0.0])
        ),
        f"{prefix}.linear_attn.A_log": (
            "F32", (2,), struct.pack("<2f", 0.0, 0.0)
        ),
        f"{prefix}.linear_attn.norm.weight": (
            "F32", (2,), struct.pack("<2f", 1.0, 1.0)
        ),
        f"{prefix}.linear_attn.out_proj.weight": (
            "BF16", (2, 4), bf16_bytes([1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5])
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


class ReferenceDeltaNet:
    def __init__(self) -> None:
        self.conv = [[0.0, 0.0] for _ in range(8)]
        self.recurrent = [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]

    def forward(self, hidden: list[float]) -> list[float]:
        inverse = 1.0 / math.sqrt(
            sum(value * value for value in hidden) / 2 + 1e-6
        )
        normalized = [value * inverse for value in hidden]
        mixed = normalized * 3 + [normalized[1], normalized[0]]
        convolved = []
        for channel, value in enumerate(mixed):
            self.conv[channel] = [self.conv[channel][1], value]
            total = self.conv[channel][0] * 0.25 + value
            convolved.append(total / (1.0 + math.exp(-total)))
        query = convolved[:2]
        key = convolved[2:4]
        values = [convolved[4:6], convolved[6:8]]
        query_norm = math.sqrt(sum(value * value for value in query) + 1e-6)
        key_norm = math.sqrt(sum(value * value for value in key) + 1e-6)
        query = [value / query_norm / math.sqrt(2.0) for value in query]
        key = [value / key_norm for value in key]
        cores = []
        for head in range(2):
            matrix = self.recurrent[head]
            for row in range(2):
                for column in range(2):
                    matrix[row][column] *= 0.5
            memory = [
                sum(matrix[row][column] * key[row] for row in range(2))
                for column in range(2)
            ]
            delta = [
                (value - remembered) * 0.5
                for value, remembered in zip(values[head], memory)
            ]
            for row in range(2):
                for column in range(2):
                    matrix[row][column] += key[row] * delta[column]
            core = [
                sum(matrix[row][column] * query[row] for row in range(2))
                for column in range(2)
            ]
            core_inverse = 1.0 / math.sqrt(
                sum(value * value for value in core) / 2 + 1e-6
            )
            gates = normalized
            cores.append([
                value * core_inverse * gate / (1.0 + math.exp(-gate))
                for value, gate in zip(core, gates)
            ])
        return [
            cores[0][0] + 0.5 * cores[1][0],
            cores[0][1] + 0.5 * cores[1][1],
        ]


class QwenGatedDeltaTests(unittest.TestCase):
    def test_converter_writes_linear_container_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_gated_delta_checkpoint(source)
            QwenCheckpointConverter(source).convert(output)

            storage = QwenGatedDeltaConverter(source).convert(output)

            self.assertEqual(storage["layers"], [0])
            container = ColiTensorFile(output / "linear_layers/layer-000.coli")
            self.assertEqual(container.metadata["num_key_heads"], 1)
            self.assertEqual(container.metadata["num_value_heads"], 2)
            self.assertEqual(container.tensors["conv1d.weight"].shape, (8, 1, 2))
            self.assertEqual(container.tensors["A_log"].dtype, "F32")
            self.assertEqual(container.tensors["norm.weight"].dtype, "F32")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["linear_layer_storage"]["mode"],
                "bf16-gated-deltanet",
            )

    def test_q4_linear_conversion_loads_projection_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_gated_delta_checkpoint(source)
            storage = QwenGatedDeltaConverter(source).convert(
                output, quantization="q4"
            )
            layer = QwenGatedDeltaLayer.from_model_directory(output, 0)
            self.assertEqual(storage["mode"], "q4-gated-deltanet")
            self.assertIsInstance(layer.in_proj_qkv, Q4BlockTensor)
            self.assertIsInstance(layer.out_proj, Q4BlockTensor)

    def test_incremental_execution_matches_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_gated_delta_checkpoint(source)
            QwenGatedDeltaConverter(source).convert(output)
            layer = QwenGatedDeltaLayer.from_model_directory(output, 0)
            state = layer.new_state()
            reference = ReferenceDeltaNet()

            for hidden in ([1.0, 0.5], [-0.25, 1.0], [0.75, -0.5]):
                expected = reference.forward(list(hidden))
                actual = layer.forward(list(hidden), state).output
                for expected_value, actual_value in zip(expected, actual):
                    self.assertAlmostEqual(actual_value, expected_value, places=5)
            self.assertEqual(state.tokens, 3)

    def test_cuda_execution_matches_numpy_state_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_gated_delta_checkpoint(source)
            QwenGatedDeltaConverter(source).convert(output)
            layer = QwenGatedDeltaLayer.from_model_directory(output, 0)
            inputs = [[1.0, 0.5], [-0.25, 1.0], [0.75, -0.5]]
            disable_cuda()
            numpy_state = layer.new_state()
            expected = [
                layer.forward(value, numpy_state).output for value in inputs
            ]
            try:
                configure_cuda(cache_mib=64)
            except CudaUnavailableError as error:
                self.skipTest(str(error))
            try:
                cuda_state = layer.new_state()
                actual = [
                    layer.forward(value, cuda_state).output for value in inputs
                ]
            finally:
                disable_cuda()
            for expected_output, actual_output in zip(expected, actual):
                for expected_value, actual_value in zip(
                    expected_output, actual_output
                ):
                    self.assertAlmostEqual(
                        expected_value, actual_value, places=4
                    )
            self.assertEqual(cuda_state.tokens, len(inputs))

    def test_numpy_and_portable_paths_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_gated_delta_checkpoint(source)
            QwenGatedDeltaConverter(source).convert(output)
            layer = QwenGatedDeltaLayer.from_model_directory(output, 0)
            inputs = [[1.0, 0.5], [-0.25, 1.0]]
            numpy_state = layer.new_state()
            numpy_outputs = [layer.forward(value, numpy_state).output for value in inputs]
            with patch.object(gated_delta_module, "np", None):
                python_state = layer.new_state()
                python_outputs = [
                    layer.forward(value, python_state).output for value in inputs
                ]
            for numpy_output, python_output in zip(numpy_outputs, python_outputs):
                for numpy_value, python_value in zip(numpy_output, python_output):
                    self.assertAlmostEqual(numpy_value, python_value, places=5)

    def test_cli_converts_and_benchmarks_linear_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_gated_delta_checkpoint(source)
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    main(["convert-linear-layers", str(source), str(output)]),
                    0,
                )
            capture = StringIO()
            with redirect_stdout(capture):
                self.assertEqual(
                    main([
                        "benchmark-linear-layer",
                        str(output),
                        "--layer", "0",
                        "--tokens", "3",
                    ]),
                    0,
                )
            report = json.loads(capture.getvalue())
            self.assertEqual(report["state_tokens"], 3)


if __name__ == "__main__":
    unittest.main()

