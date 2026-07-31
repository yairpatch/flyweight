import json
import struct
import tempfile
import unittest
from pathlib import Path

from colibri_next.converter import QwenCheckpointConverter, QwenSafetensorCheckpoint
from colibri_next.q4 import Q4BlockTensor
from colibri_next.safetensors import SafeTensorFile
from colibri_next.tensor_container import ColiTensorFile


def bf16_bytes(values: list[float]) -> bytes:
    output = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        output.extend(struct.pack("<H", bits >> 16))
    return bytes(output)


def write_safetensors(
    path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]
) -> None:
    offset = 0
    header = {}
    payloads = []
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payloads.append(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for payload in payloads:
            handle.write(payload)


def create_checkpoint(root: Path) -> dict[str, bytes]:
    config = {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "hidden_size": 2,
            "moe_intermediate_size": 2,
            "shared_expert_intermediate_size": 2,
            "vocab_size": 4,
        }
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    expected: dict[str, bytes] = {}
    for layer in range(2):
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        gate_name = f"{prefix}.gate_up_proj"
        down_name = f"{prefix}.down_proj"
        gate_data = bf16_bytes([0.01 * (layer + value - 8) for value in range(16)])
        down_data = bf16_bytes([0.02 * (layer + value - 4) for value in range(8)])
        tensors[gate_name] = ("BF16", (2, 4, 2), gate_data)
        tensors[down_name] = ("BF16", (2, 2, 2), down_data)
        expected[f"{layer}:gate"] = gate_data
        expected[f"{layer}:down"] = down_data
        layer_prefix = f"model.language_model.layers.{layer}"
        tensors[f"{layer_prefix}.mlp.gate.weight"] = (
            "BF16", (2, 2), bf16_bytes([1.0, 0.0, -1.0, 0.0])
        )
        tensors[f"{layer_prefix}.mlp.shared_expert.gate_proj.weight"] = (
            "BF16", (2, 2), bf16_bytes([0.2, -0.1, 0.1, 0.3])
        )
        tensors[f"{layer_prefix}.mlp.shared_expert.up_proj.weight"] = (
            "BF16", (2, 2), bf16_bytes([0.4, 0.2, -0.2, 0.1])
        )
        tensors[f"{layer_prefix}.mlp.shared_expert.down_proj.weight"] = (
            "BF16", (2, 2), bf16_bytes([0.3, -0.2, 0.2, 0.5])
        )
        tensors[f"{layer_prefix}.mlp.shared_expert_gate.weight"] = (
            "BF16", (1, 2), bf16_bytes([0.5, -0.25])
        )
        tensors[f"{layer_prefix}.post_attention_layernorm.weight"] = (
            "BF16", (2,), bf16_bytes([0.0, 0.0])
        )
    embedding_name = "model.language_model.embed_tokens.weight"
    tensors[embedding_name] = ("BF16", (4, 2), bytes(range(16)))
    shard_name = "model-00001-of-00001.safetensors"
    write_safetensors(root / shard_name, tensors)
    index = {
        "metadata": {"total_size": sum(len(value[2]) for value in tensors.values())},
        "weight_map": {name: shard_name for name in tensors},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    return expected


class SafeTensorReaderTests(unittest.TestCase):
    def test_reads_axis_zero_without_decoding_tensor_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = create_checkpoint(root)
            safe_file = SafeTensorFile(root / "model-00001-of-00001.safetensors")
            name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
            data, shape = safe_file.read_axis0_slice(name, 1)
            self.assertEqual(shape, (4, 2))
            self.assertEqual(data, expected["0:gate"][16:])


class QwenConverterTests(unittest.TestCase):
    def test_inspection_classifies_stacked_experts_and_static_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_checkpoint(root)
            report = QwenSafetensorCheckpoint(root).inspect()
            self.assertTrue(report["weights_available"])
            self.assertEqual(report["stacked_expert_tensor_count"], 4)
            self.assertEqual(report["expected_expert_file_count"], 4)
            self.assertEqual(report["static_tensor_count"], 13)

    def test_converter_splits_byte_exact_expert_containers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "converted"
            source.mkdir()
            expected = create_checkpoint(source)

            manifest = QwenCheckpointConverter(source).convert(
                output, extract_experts=True
            )

            self.assertEqual(manifest["expert_storage"]["mode"], "split-bf16")
            self.assertIn(
                "model.language_model.embed_tokens.weight",
                manifest["static_storage"]["tensors"],
            )
            expert = ColiTensorFile(output / "experts/layer-000/expert-0001.coli")
            self.assertEqual(expert.metadata["layer"], 0)
            self.assertEqual(expert.metadata["expert"], 1)
            self.assertEqual(expert.tensors["gate_up_proj"].shape, (4, 2))
            self.assertEqual(expert.tensors["down_proj"].shape, (2, 2))
            self.assertEqual(expert.read("gate_up_proj"), expected["0:gate"][16:])
            self.assertEqual(expert.read("down_proj"), expected["0:down"][8:])

    def test_converter_can_emit_q4_expert_containers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "converted"
            source.mkdir()
            create_checkpoint(source)

            manifest = QwenCheckpointConverter(source).convert(
                output, extract_experts=True, quantization="q4"
            )
            expert = ColiTensorFile(output / "experts/layer-000/expert-0000.coli")
            gate = Q4BlockTensor.from_container(expert, "gate_up_proj")

            self.assertEqual(manifest["expert_storage"]["mode"], "split-q4")
            self.assertEqual(expert.metadata["quantization"]["scheme"], "q4_symmetric")
            self.assertEqual(gate.shape, (4, 2))
            self.assertEqual(len(gate.dequantize()), 8)

    def test_overwrite_replaces_split_experts_without_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "converted"
            source.mkdir()
            create_checkpoint(source)
            converter = QwenCheckpointConverter(source)
            converter.convert(output, extract_experts=True)

            stale = output / "experts" / "layer-999" / "expert-9999.coli"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")

            converter.convert(output, extract_experts=True, overwrite=True)

            self.assertFalse(stale.exists())
            self.assertTrue(
                (output / "experts" / "layer-001" / "expert-0001.coli").exists()
            )

    def test_manifest_only_does_not_require_weight_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "converted"
            source.mkdir()
            create_checkpoint(source)
            (source / "model-00001-of-00001.safetensors").unlink()

            manifest = QwenCheckpointConverter(source).convert(output)
            source_tensor = manifest["expert_storage"]["source_tensors"]["0"]
            self.assertIsNone(source_tensor["gate_up_proj"]["dtype"])
            self.assertEqual(manifest["expert_storage"]["mode"], "source-stacked")


if __name__ == "__main__":
    unittest.main()
