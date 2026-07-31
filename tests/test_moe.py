import json
import math
import tempfile
import unittest
from pathlib import Path

from colibri_next.converter import QwenCheckpointConverter
from colibri_next.kernels import Q4SwiGLUExpert
from colibri_next.moe import QwenMoELayer
from colibri_next.moe_converter import QwenMoELayerConverter
from test_converter import create_checkpoint


class QwenMoELayerTests(unittest.TestCase):
    def test_converted_layer_executes_reference_routing_and_shared_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "converted"
            source.mkdir()
            create_checkpoint(source)
            QwenCheckpointConverter(source).convert(
                output, extract_experts=True, quantization="q4"
            )
            storage = QwenMoELayerConverter(source).convert(output)
            layer = QwenMoELayer.from_model_directory(output, 0)
            hidden = [0.5, -0.25]

            result = layer.forward(hidden)

            self.assertEqual(storage["mode"], "q4-shared-bf16-router")
            self.assertEqual(result.selected_experts, (0,))
            self.assertAlmostEqual(sum(result.routing_weights), 1.0)
            routed = Q4SwiGLUExpert.from_file(
                output / "experts/layer-000/expert-0000.coli"
            ).forward(hidden)
            shared = layer.shared_expert.forward(hidden)
            shared_logit = layer.shared_gate.matvec(hidden)[0]
            shared_weight = 1.0 / (1.0 + math.exp(-shared_logit))
            expected = [
                routed_value + shared_weight * shared_value
                for routed_value, shared_value in zip(routed, shared)
            ]
            for actual, expected_value in zip(result.output, expected):
                self.assertAlmostEqual(actual, expected_value, places=6)

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["moe_layer_storage"], storage)

    def test_residual_path_applies_qwen_rmsnorm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "converted"
            source.mkdir()
            create_checkpoint(source)
            QwenCheckpointConverter(source).convert(
                output, extract_experts=True, quantization="q4"
            )
            QwenMoELayerConverter(source).convert(output)
            layer = QwenMoELayer.from_model_directory(output, 0)
            hidden = [0.5, -0.25]

            normalized = layer.normalize(hidden)
            residual = layer.forward_residual(hidden)
            direct = layer.forward(normalized)

            variance = sum(value * value for value in hidden) / len(hidden)
            scale = 1.0 / math.sqrt(variance + 1e-6)
            for actual, value in zip(normalized, hidden):
                self.assertAlmostEqual(actual, value * scale, places=6)
            for actual, base, update in zip(residual.output, hidden, direct.output):
                self.assertAlmostEqual(actual, base + update, places=6)


if __name__ == "__main__":
    unittest.main()
