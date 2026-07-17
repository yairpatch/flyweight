import json
import tempfile
import unittest
from pathlib import Path

from colibri_next.mtp_converter import QwenMtpConverter
from colibri_next.tensor_container import ColiTensorFile

from tests.test_converter import bf16_bytes, write_safetensors


def create_mtp_checkpoint(root: Path) -> None:
    config = {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "num_experts": 1,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 2,
        }
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    vec = ("BF16", (2,), bf16_bytes([0.0, 0.0]))
    mat = ("BF16", (2, 2), bf16_bytes([0.0] * 4))
    tensors = {
        "mtp.fc.weight": ("BF16", (2, 4), bf16_bytes([0.0] * 8)),
        "mtp.pre_fc_norm_embedding.weight": vec,
        "mtp.pre_fc_norm_hidden.weight": vec,
        "mtp.norm.weight": vec,
        "mtp.layers.0.input_layernorm.weight": vec,
        "mtp.layers.0.post_attention_layernorm.weight": vec,
        "mtp.layers.0.self_attn.q_proj.weight": (
            "BF16", (4, 2), bf16_bytes([0.0] * 8)
        ),
        "mtp.layers.0.self_attn.k_proj.weight": mat,
        "mtp.layers.0.self_attn.v_proj.weight": mat,
        "mtp.layers.0.self_attn.o_proj.weight": mat,
        "mtp.layers.0.self_attn.q_norm.weight": vec,
        "mtp.layers.0.self_attn.k_norm.weight": vec,
        "mtp.layers.0.mlp.gate.weight": ("BF16", (1, 2), bf16_bytes([0.0] * 2)),
        "mtp.layers.0.mlp.experts.gate_up_proj": (
            "BF16", (1, 4, 2), bf16_bytes([0.0] * 8)
        ),
        "mtp.layers.0.mlp.experts.down_proj": (
            "BF16", (1, 2, 2), bf16_bytes([0.0] * 4)
        ),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": mat,
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": mat,
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": mat,
        "mtp.layers.0.mlp.shared_expert_gate.weight": (
            "BF16", (1, 2), bf16_bytes([0.0] * 2)
        ),
    }
    shard = "model-00001-of-00001.safetensors"
    write_safetensors(root / shard, tensors)
    index = {
        "metadata": {
            "total_size": sum(len(value[2]) for value in tensors.values())
        },
        "weight_map": {name: shard for name in tensors},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


class MtpConverterTests(unittest.TestCase):
    def test_converts_all_tensors_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            create_mtp_checkpoint(source)
            (output / "manifest.json").write_text(
                json.dumps({"format": "colibri-model"}), encoding="utf-8"
            )
            storage = QwenMtpConverter(source).convert(output)
            self.assertEqual(storage["tensor_count"], 19)
            container = ColiTensorFile(output / "mtp.coli")
            self.assertEqual(len(container.tensors), 19)
            self.assertIn("fc.weight", container.tensors)
            self.assertIn("layers.0.mlp.experts.gate_up_proj", container.tensors)
            self.assertEqual(container.metadata["kind"], "qwen-mtp-head")
            self.assertEqual(container.metadata["dtype"], "bf16")
            self.assertEqual(container.metadata["top_k"], 1)
            self.assertEqual(container.metadata["rotary_dim"], 2)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["mtp_storage"], storage)

    def test_refuses_overwrite_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            create_mtp_checkpoint(source)
            QwenMtpConverter(source).convert(output)
            with self.assertRaises(FileExistsError):
                QwenMtpConverter(source).convert(output)
            QwenMtpConverter(source).convert(output, overwrite=True)

    def test_rejects_checkpoint_without_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"other.weight": "x"}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                QwenMtpConverter(root)


if __name__ == "__main__":
    unittest.main()
