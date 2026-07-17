from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .q4 import BLOCK_SIZE, Q4BlockTensor
from .safetensors import SafeTensorFile
from .tensor_container import TensorPayload, write_coli_tensor_file


_EXPERT_PATTERN = re.compile(
    r"^(?:model\.language_model|model)\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)


@dataclass(frozen=True, slots=True)
class SourceTensor:
    name: str
    shard: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    byte_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QwenSafetensorCheckpoint:
    """Inspects Qwen3.5/Qwen3.6-style sharded safetensor checkpoints."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        config_path = self.root / "config.json"
        index_path = self.root / "model.safetensors.index.json"
        if not config_path.exists():
            raise FileNotFoundError(f"missing Qwen config: {config_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"missing safetensors index: {index_path}")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.text_config = self.config.get("text_config", self.config)
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = dict(self.index["weight_map"])
        self.layers = int(self.text_config["num_hidden_layers"])
        self.experts_per_layer = int(self.text_config["num_experts"])
        self.hidden_size = int(self.text_config["hidden_size"])
        self.expert_intermediate_size = int(self.text_config["moe_intermediate_size"])
        self.expert_sources: dict[int, dict[str, str]] = {
            layer: {} for layer in range(self.layers)
        }
        for name in self.weight_map:
            match = _EXPERT_PATTERN.match(name)
            if match:
                layer = int(match.group(1))
                if layer in self.expert_sources:
                    self.expert_sources[layer][match.group(2)] = name
        self._validate_expert_index()
        self._shard_headers: dict[str, SafeTensorFile] = {}

    @property
    def shard_names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.weight_map.values())))

    @property
    def static_tensor_names(self) -> tuple[str, ...]:
        expert_names = {
            name for sources in self.expert_sources.values() for name in sources.values()
        }
        return tuple(name for name in self.weight_map if name not in expert_names)

    def inspect(self) -> dict[str, Any]:
        missing_shards = [
            shard for shard in self.shard_names if not (self.root / shard).exists()
        ]
        return {
            "model_type": self.text_config.get("model_type"),
            "layers": self.layers,
            "experts_per_layer": self.experts_per_layer,
            "routed_experts_per_token": self.text_config.get("num_experts_per_tok"),
            "hidden_size": self.hidden_size,
            "expert_intermediate_size": self.expert_intermediate_size,
            "tensor_count": len(self.weight_map),
            "static_tensor_count": len(self.static_tensor_names),
            "stacked_expert_tensor_count": self.layers * 2,
            "expected_expert_file_count": self.layers * self.experts_per_layer,
            "shard_count": len(self.shard_names),
            "source_bytes": int(self.index.get("metadata", {}).get("total_size", 0)),
            "missing_shards": missing_shards,
            "weights_available": not missing_shards,
        }

    def source_tensor(self, name: str, *, require_weights: bool = False) -> SourceTensor:
        shard = self.weight_map[name]
        shard_path = self.root / shard
        if not shard_path.exists():
            if require_weights:
                raise FileNotFoundError(f"missing checkpoint shard: {shard_path}")
            return SourceTensor(name=name, shard=shard)
        safe_file = self._shard_headers.get(shard)
        if safe_file is None:
            safe_file = SafeTensorFile(shard_path)
            self._shard_headers[shard] = safe_file
        info = safe_file.tensor(name)
        return SourceTensor(
            name=name,
            shard=shard,
            dtype=info.dtype,
            shape=info.shape,
            byte_size=info.byte_size,
        )

    def safe_file_for(self, tensor_name: str) -> SafeTensorFile:
        source = self.source_tensor(tensor_name, require_weights=True)
        return self._shard_headers[source.shard]

    def _validate_expert_index(self) -> None:
        missing = {
            layer: sorted({"gate_up_proj", "down_proj"} - set(sources))
            for layer, sources in self.expert_sources.items()
            if set(sources) != {"gate_up_proj", "down_proj"}
        }
        if missing:
            summary = ", ".join(
                f"layer {layer}: {roles}" for layer, roles in list(missing.items())[:5]
            )
            raise ValueError(f"checkpoint has incomplete routed expert tensors: {summary}")


class QwenCheckpointConverter:
    def __init__(self, source: Path | str):
        self.checkpoint = QwenSafetensorCheckpoint(source)

    def convert(
        self,
        output: Path | str,
        *,
        extract_experts: bool = False,
        quantization: str = "bf16",
        overwrite: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        if quantization not in {"bf16", "q4"}:
            raise ValueError("quantization must be 'bf16' or 'q4'")
        output_path = Path(output)
        if output_path.exists() and any(output_path.iterdir()) and not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_path}; pass overwrite=True"
            )
        output_path.mkdir(parents=True, exist_ok=True)

        manifest = self._base_manifest(
            extract_experts=extract_experts, quantization=quantization
        )
        if extract_experts:
            expected_bytes = self._estimated_expert_bytes(quantization)
            available_bytes = shutil.disk_usage(output_path).free
            safety_margin = 1024**3
            if available_bytes < expected_bytes + safety_margin:
                raise OSError(
                    f"expert extraction requires {expected_bytes + safety_margin} free bytes "
                    f"including safety margin, but only {available_bytes} are available"
                )
            self._extract_experts(
                output_path,
                quantization=quantization,
                overwrite=overwrite,
                progress=progress,
            )
        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    def _base_manifest(
        self, *, extract_experts: bool, quantization: str
    ) -> dict[str, Any]:
        checkpoint = self.checkpoint
        expert_sources = {
            str(layer): {
                role: checkpoint.source_tensor(name).to_dict()
                for role, name in sources.items()
            }
            for layer, sources in checkpoint.expert_sources.items()
        }
        static_tensors = {
            name: checkpoint.source_tensor(name).to_dict()
            for name in checkpoint.static_tensor_names
        }
        return {
            "format": "colibri-model",
            "version": 1,
            "architecture": checkpoint.text_config.get("model_type"),
            "source": {
                "path": str(checkpoint.root.resolve()),
                "format": "huggingface-safetensors",
                "source_bytes": int(
                    checkpoint.index.get("metadata", {}).get("total_size", 0)
                ),
                "shards": list(checkpoint.shard_names),
            },
            "model": {
                "layers": checkpoint.layers,
                "hidden_size": checkpoint.hidden_size,
                "layer_types": checkpoint.text_config.get("layer_types"),
                "experts_per_layer": checkpoint.experts_per_layer,
                "routed_experts_per_token": checkpoint.text_config.get(
                    "num_experts_per_tok"
                ),
                "expert_intermediate_size": checkpoint.expert_intermediate_size,
                "shared_expert_intermediate_size": checkpoint.text_config.get(
                    "shared_expert_intermediate_size"
                ),
                "vocab_size": checkpoint.text_config.get("vocab_size"),
            },
            "expert_storage": {
                "mode": f"split-{quantization}" if extract_experts else "source-stacked",
                "path_pattern": "experts/layer-{layer:03d}/expert-{expert:04d}.coli"
                if extract_experts
                else None,
                "container_format": "coli-tensor-v1" if extract_experts else None,
                "source_tensors": expert_sources,
            },
            "static_storage": {
                "mode": "source-safetensors",
                "tensors": static_tensors,
            },
        }

    def _estimated_expert_bytes(self, quantization: str) -> int:
        checkpoint = self.checkpoint
        sources = [
            checkpoint.source_tensor(name, require_weights=True)
            for layer_sources in checkpoint.expert_sources.values()
            for name in layer_sources.values()
        ]
        if quantization == "bf16":
            return sum(source.byte_size or 0 for source in sources)
        total = 0
        for source in sources:
            elements_per_expert = (source.byte_size or 0) // 2 // checkpoint.experts_per_layer
            blocks_per_expert = (elements_per_expert + BLOCK_SIZE - 1) // BLOCK_SIZE
            total += blocks_per_expert * 18 * checkpoint.experts_per_layer
        return total + checkpoint.layers * checkpoint.experts_per_layer * 1024

    def _extract_experts(
        self,
        output: Path,
        *,
        quantization: str,
        overwrite: bool,
        progress: Callable[[int, int], None] | None,
    ) -> None:
        checkpoint = self.checkpoint
        total = checkpoint.layers * checkpoint.experts_per_layer
        completed = 0
        for layer in range(checkpoint.layers):
            sources = checkpoint.expert_sources[layer]
            gate_name = sources["gate_up_proj"]
            down_name = sources["down_proj"]
            gate_file = checkpoint.safe_file_for(gate_name)
            down_file = checkpoint.safe_file_for(down_name)
            gate_info = gate_file.tensor(gate_name)
            down_info = down_file.tensor(down_name)
            self._validate_shapes(layer, gate_info.shape, down_info.shape)

            for expert in range(checkpoint.experts_per_layer):
                destination = (
                    output
                    / "experts"
                    / f"layer-{layer:03d}"
                    / f"expert-{expert:04d}.coli"
                )
                if destination.exists() and not overwrite:
                    raise FileExistsError(f"expert file already exists: {destination}")
                gate_data, gate_shape = gate_file.read_axis0_slice(gate_name, expert)
                down_data, down_shape = down_file.read_axis0_slice(down_name, expert)
                if quantization == "q4":
                    gate_tensor = Q4BlockTensor.from_bf16(gate_data, gate_shape)
                    down_tensor = Q4BlockTensor.from_bf16(down_data, down_shape)
                    payloads = [
                        *gate_tensor.payloads("gate_up_proj"),
                        *down_tensor.payloads("down_proj"),
                    ]
                    quantization_metadata: dict[str, Any] = {
                        "scheme": "q4_symmetric",
                        "block_size": BLOCK_SIZE,
                        "tensors": {
                            "gate_up_proj": gate_tensor.metadata(),
                            "down_proj": down_tensor.metadata(),
                        },
                    }
                else:
                    payloads = [
                        TensorPayload("gate_up_proj", gate_info.dtype, gate_shape, gate_data),
                        TensorPayload("down_proj", down_info.dtype, down_shape, down_data),
                    ]
                    quantization_metadata = {"scheme": "none"}
                write_coli_tensor_file(
                    destination,
                    payloads,
                    metadata={
                        "architecture": checkpoint.text_config.get("model_type"),
                        "layer": layer,
                        "expert": expert,
                        "quantization": quantization_metadata,
                    },
                )
                completed += 1
                if progress is not None:
                    progress(completed, total)

    def _validate_shapes(
        self,
        layer: int,
        gate_shape: tuple[int, ...],
        down_shape: tuple[int, ...],
    ) -> None:
        checkpoint = self.checkpoint
        expected_gate = (
            checkpoint.experts_per_layer,
            checkpoint.expert_intermediate_size * 2,
            checkpoint.hidden_size,
        )
        expected_down = (
            checkpoint.experts_per_layer,
            checkpoint.hidden_size,
            checkpoint.expert_intermediate_size,
        )
        if gate_shape != expected_gate:
            raise ValueError(
                f"layer {layer} gate_up_proj shape {gate_shape} != {expected_gate}"
            )
        if down_shape != expected_down:
            raise ValueError(f"layer {layer} down_proj shape {down_shape} != {expected_down}")

