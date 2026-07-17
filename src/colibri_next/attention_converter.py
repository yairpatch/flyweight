from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from .converter import QwenSafetensorCheckpoint
from .matrix import encode_matrix, quantization_metadata
from .tensor_container import TensorPayload, write_coli_tensor_file


_ATTENTION_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"
)


class QwenAttentionConverter:
    """Converts Qwen full-attention layers into executable BF16 containers."""

    def __init__(self, source: Path | str):
        self.checkpoint = QwenSafetensorCheckpoint(source)

    @property
    def attention_layers(self) -> tuple[int, ...]:
        layer_types = self.checkpoint.text_config.get("layer_types")
        if layer_types is not None:
            if len(layer_types) != self.checkpoint.layers:
                raise ValueError("layer_types length does not match num_hidden_layers")
            return tuple(
                layer
                for layer, layer_type in enumerate(layer_types)
                if layer_type == "full_attention"
            )
        return tuple(
            sorted(
                int(match.group(1))
                for name in self.checkpoint.weight_map
                if (match := _ATTENTION_PATTERN.match(name)) is not None
            )
        )

    def convert(
        self,
        output: Path | str,
        *,
        overwrite: bool = False,
        quantization: str = "bf16",
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, object]:
        if quantization not in {"bf16", "q4"}:
            raise ValueError(f"unsupported quantization: {quantization}")
        output_root = Path(output)
        layer_directory = output_root / "attention_layers"
        layer_directory.mkdir(parents=True, exist_ok=True)
        layers = self.attention_layers
        for completed, layer in enumerate(layers, start=1):
            destination = layer_directory / f"layer-{layer:03d}.coli"
            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"attention layer file already exists: {destination}"
                )
            self._convert_layer(layer, destination, quantization)
            if progress is not None:
                progress(completed, len(layers))

        storage = {
            "mode": f"{quantization}-full-attention",
            "path_pattern": "attention_layers/layer-{layer:03d}.coli",
            "layer_count": len(layers),
            "layers": list(layers),
        }
        manifest_path = output_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["attention_layer_storage"] = storage
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        return storage

    def _convert_layer(
        self, layer: int, destination: Path, quantization: str
    ) -> None:
        prefix = f"model.language_model.layers.{layer}"
        tensors = {
            "input_layernorm.weight": self._read(
                f"{prefix}.input_layernorm.weight"
            ),
            "q_proj.weight": self._read(f"{prefix}.self_attn.q_proj.weight"),
            "k_proj.weight": self._read(f"{prefix}.self_attn.k_proj.weight"),
            "v_proj.weight": self._read(f"{prefix}.self_attn.v_proj.weight"),
            "o_proj.weight": self._read(f"{prefix}.self_attn.o_proj.weight"),
            "q_norm.weight": self._read(f"{prefix}.self_attn.q_norm.weight"),
            "k_norm.weight": self._read(f"{prefix}.self_attn.k_norm.weight"),
        }
        geometry = self._geometry()
        self._validate_shapes(layer, tensors, geometry)
        payloads = []
        quantized = {}
        matrix_names = {
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
        }
        for name, tensor in tensors.items():
            if name in matrix_names:
                encoded, metadata = encode_matrix(name, tensor, quantization)
                payloads.extend(encoded)
                if metadata is not None:
                    quantized[name] = metadata
            else:
                dtype, shape, data = tensor
                payloads.append(TensorPayload(name, dtype, shape, data))
        metadata = {
            "architecture": self.checkpoint.text_config.get("model_type"),
            "layer": layer,
            "rms_norm_eps": float(
                self.checkpoint.text_config.get("rms_norm_eps", 1e-6)
            ),
            **geometry,
        }
        if quantized:
            metadata["quantization"] = quantization_metadata(quantized)
        write_coli_tensor_file(destination, payloads, metadata=metadata)

    def _geometry(self) -> dict[str, int | float]:
        config = self.checkpoint.text_config
        num_heads = int(config["num_attention_heads"])
        num_kv_heads = int(config.get("num_key_value_heads", num_heads))
        head_dim = int(config.get("head_dim", self.checkpoint.hidden_size // num_heads))
        rope = config.get("rope_parameters", {})
        partial_factor = float(
            rope.get(
                "partial_rotary_factor",
                config.get("partial_rotary_factor", 1.0),
            )
        )
        rotary_dim = int(head_dim * partial_factor)
        rope_theta = float(rope.get("rope_theta", config.get("rope_theta", 10000.0)))
        return {
            "hidden_size": self.checkpoint.hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
            "rope_theta": rope_theta,
        }

    def _read(self, name: str) -> tuple[str, tuple[int, ...], bytes]:
        source = self.checkpoint.source_tensor(name, require_weights=True)
        safe_file = self.checkpoint.safe_file_for(name)
        info = safe_file.tensor(name)
        if info.dtype != "BF16":
            raise ValueError(f"tensor {name} must be BF16, got {info.dtype}")
        return source.dtype or info.dtype, info.shape, safe_file.read(name)

    def _validate_shapes(
        self,
        layer: int,
        tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
        geometry: dict[str, int | float],
    ) -> None:
        hidden = int(geometry["hidden_size"])
        heads = int(geometry["num_attention_heads"])
        kv_heads = int(geometry["num_key_value_heads"])
        head_dim = int(geometry["head_dim"])
        rotary_dim = int(geometry["rotary_dim"])
        if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
            raise ValueError("attention heads must be divisible by key/value heads")
        if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise ValueError("rotary dimension must be positive, even, and <= head_dim")
        expected = {
            "input_layernorm.weight": (hidden,),
            "q_proj.weight": (heads * head_dim * 2, hidden),
            "k_proj.weight": (kv_heads * head_dim, hidden),
            "v_proj.weight": (kv_heads * head_dim, hidden),
            "o_proj.weight": (hidden, heads * head_dim),
            "q_norm.weight": (head_dim,),
            "k_norm.weight": (head_dim,),
        }
        mismatches = [
            f"{name} {tensors[name][1]} != {shape}"
            for name, shape in expected.items()
            if tensors[name][1] != shape
        ]
        if mismatches:
            raise ValueError(
                f"layer {layer} invalid attention tensors: {', '.join(mismatches)}"
            )
