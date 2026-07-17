import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from colibri_next.attention import AttentionKVCache
from colibri_next.attention_converter import QwenAttentionConverter
from colibri_next.cli import main
from colibri_next.converter import QwenCheckpointConverter
from colibri_next.decoder import QwenDecoderLayer, QwenDecoderStack
from colibri_next.gated_delta import GatedDeltaState
from colibri_next.gated_delta_converter import QwenGatedDeltaConverter
from colibri_next.moe_converter import QwenMoELayerConverter

from tests.test_converter import bf16_bytes, write_safetensors


def create_decoder_checkpoint(root: Path) -> None:
    config = {
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_experts": 1,
            "num_experts_per_tok": 1,
            "hidden_size": 2,
            "moe_intermediate_size": 2,
            "shared_expert_intermediate_size": 2,
            "vocab_size": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000.0,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 2,
            "rms_norm_eps": 1e-6,
        }
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {}
    for layer in range(2):
        prefix = f"model.language_model.layers.{layer}"
        tensors[f"{prefix}.mlp.experts.gate_up_proj"] = (
            "BF16", (1, 4, 2), bf16_bytes([0.0] * 8)
        )
        tensors[f"{prefix}.mlp.experts.down_proj"] = (
            "BF16", (1, 2, 2), bf16_bytes([0.0] * 4)
        )
        tensors[f"{prefix}.mlp.gate.weight"] = (
            "BF16", (1, 2), bf16_bytes([0.0, 0.0])
        )
        tensors[f"{prefix}.mlp.shared_expert.gate_proj.weight"] = (
            "BF16", (2, 2), bf16_bytes([0.0] * 4)
        )
        tensors[f"{prefix}.mlp.shared_expert.up_proj.weight"] = (
            "BF16", (2, 2), bf16_bytes([0.0] * 4)
        )
        tensors[f"{prefix}.mlp.shared_expert.down_proj.weight"] = (
            "BF16", (2, 2), bf16_bytes([0.0] * 4)
        )
        tensors[f"{prefix}.mlp.shared_expert_gate.weight"] = (
            "BF16", (1, 2), bf16_bytes([0.0, 0.0])
        )
        tensors[f"{prefix}.post_attention_layernorm.weight"] = (
            "BF16", (2,), bf16_bytes([0.0, 0.0])
        )
        tensors[f"{prefix}.input_layernorm.weight"] = (
            "BF16", (2,), bf16_bytes([0.0, 0.0])
        )

    tensors["model.language_model.embed_tokens.weight"] = (
        "BF16", (4, 2), bf16_bytes([1.0, 0.0, 0.0, 1.0, 0.5, -0.5, -1.0, 0.25])
    )
    tensors["model.language_model.norm.weight"] = (
        "BF16", (2,), bf16_bytes([0.0, 0.0])
    )
    tensors["lm_head.weight"] = (
        "BF16", (4, 2), bf16_bytes([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -1.0, 0.25])
    )

    linear = "model.language_model.layers.0.linear_attn"
    identity = [1.0, 0.0, 0.0, 1.0]
    tensors[f"{linear}.in_proj_qkv.weight"] = (
        "BF16", (6, 2), bf16_bytes(identity * 3)
    )
    tensors[f"{linear}.in_proj_z.weight"] = (
        "BF16", (2, 2), bf16_bytes(identity)
    )
    tensors[f"{linear}.in_proj_b.weight"] = (
        "BF16", (1, 2), bf16_bytes([0.0, 0.0])
    )
    tensors[f"{linear}.in_proj_a.weight"] = (
        "BF16", (1, 2), bf16_bytes([0.0, 0.0])
    )
    tensors[f"{linear}.conv1d.weight"] = (
        "BF16", (6, 1, 2), bf16_bytes([0.25, 1.0] * 6)
    )
    tensors[f"{linear}.dt_bias"] = ("BF16", (1,), bf16_bytes([0.0]))
    tensors[f"{linear}.A_log"] = ("BF16", (1,), bf16_bytes([0.0]))
    tensors[f"{linear}.norm.weight"] = (
        "BF16", (2,), bf16_bytes([1.0, 1.0])
    )
    tensors[f"{linear}.out_proj.weight"] = (
        "BF16", (2, 2), bf16_bytes(identity)
    )

    attention = "model.language_model.layers.1.self_attn"
    tensors[f"{attention}.q_proj.weight"] = (
        "BF16", (4, 2), bf16_bytes(identity + [0.0] * 4)
    )
    tensors[f"{attention}.k_proj.weight"] = (
        "BF16", (2, 2), bf16_bytes(identity)
    )
    tensors[f"{attention}.v_proj.weight"] = (
        "BF16", (2, 2), bf16_bytes(identity)
    )
    tensors[f"{attention}.o_proj.weight"] = (
        "BF16", (2, 2), bf16_bytes(identity)
    )
    tensors[f"{attention}.q_norm.weight"] = (
        "BF16", (2,), bf16_bytes([0.0, 0.0])
    )
    tensors[f"{attention}.k_norm.weight"] = (
        "BF16", (2,), bf16_bytes([0.0, 0.0])
    )

    shard_name = "model-00001-of-00001.safetensors"
    write_safetensors(root / shard_name, tensors)
    index = {
        "metadata": {"total_size": sum(len(value[2]) for value in tensors.values())},
        "weight_map": {name: shard_name for name in tensors},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def convert_decoder(source: Path, output: Path) -> None:
    QwenCheckpointConverter(source).convert(
        output, extract_experts=True, quantization="q4"
    )
    QwenMoELayerConverter(source).convert(output)
    QwenGatedDeltaConverter(source).convert(output)
    QwenAttentionConverter(source).convert(output)


class QwenDecoderTests(unittest.TestCase):
    def test_manifest_preserves_layer_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            manifest = QwenCheckpointConverter(source).convert(output)
            self.assertEqual(
                manifest["model"]["layer_types"],
                ["linear_attention", "full_attention"],
            )

    def test_stack_matches_manual_layer_composition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            convert_decoder(source, output)
            decoder = QwenDecoderStack.from_model_directory(output)
            state = decoder.new_state()
            manual_layers = [
                QwenDecoderLayer.from_model_directory(
                    output, 0, "linear_attention"
                ),
                QwenDecoderLayer.from_model_directory(
                    output, 1, "full_attention"
                ),
            ]
            manual_states = [layer.new_state() for layer in manual_layers]

            for hidden in ([1.0, 0.5], [-0.25, 1.0]):
                expected = list(hidden)
                for layer, layer_state in zip(manual_layers, manual_states):
                    expected = layer.forward(expected, layer_state).output
                result = decoder.forward_token(list(hidden), state)
                for expected_value, actual_value in zip(expected, result.output):
                    self.assertAlmostEqual(actual_value, expected_value, places=5)
                self.assertEqual(
                    [layer.selected_experts for layer in result.layer_results],
                    [(0,), (0,)],
                )
            self.assertEqual(state.tokens, 2)
            self.assertIsInstance(
                state.layer_states[0].token_mixer_state, GatedDeltaState
            )
            self.assertIsInstance(
                state.layer_states[1].token_mixer_state, AttentionKVCache
            )
            state.clear()
            self.assertEqual(state.tokens, 0)

    def test_moe_layers_can_be_placed_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            convert_decoder(source, output)
            decoder = QwenDecoderStack.from_model_directory(output)
            decoder.configure_moe_placement(1)
            self.assertEqual(decoder.cpu_moe_layers, 1)
            self.assertEqual(decoder.layers[0].moe.expert_device, "cpu")
            self.assertEqual(decoder.layers[1].moe.expert_device, "cuda")
            with self.assertRaises(ValueError):
                decoder.configure_moe_placement(3)

    def test_experts_can_be_preloaded_with_storage_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            convert_decoder(source, output)
            decoder = QwenDecoderStack.from_model_directory(output)
            progress = []
            loaded = decoder.preload_experts(
                lambda completed, total: progress.append((completed, total))
            )
            self.assertEqual(loaded, 2)
            self.assertGreater(decoder.estimated_expert_storage_bytes, 0)
            self.assertEqual(progress[-1], (2, 2))
            self.assertTrue(all(len(layer.moe._experts) == 1 for layer in decoder.layers))

    def test_cli_benchmarks_complete_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            convert_decoder(source, output)
            stdout = StringIO()
            with redirect_stdout(stdout), redirect_stderr(StringIO()):
                self.assertEqual(
                    main(["benchmark-decoder", str(output), "--tokens", "2"]),
                    0,
                )
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["layers"], 2)
            self.assertEqual(report["state_tokens"], 2)
            self.assertTrue(math.isfinite(report["output_checksum"]))


if __name__ == "__main__":
    unittest.main()

