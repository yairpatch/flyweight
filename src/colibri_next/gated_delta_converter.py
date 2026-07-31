from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from .converter import QwenSafetensorCheckpoint
from .matrix import encode_matrix, quantization_metadata
from .tensor_container import TensorPayload, write_coli_tensor_file


_LINEAR_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$"
)


class QwenGatedDeltaConverter:
    """Converts Qwen linear-attention layers into BF16 Gated DeltaNet containers."""

    def __init__(self, source: Path | str):
        self.checkpoint = QwenSafetensorCheckpoint(source)

    @property
    def linear_layers(self) -> tuple[int, ...]:
        layer_types = self.checkpoint.text_config.get("layer_types")
        if layer_types is not None:
            if len(layer_types) != self.checkpoint.layers:
                raise ValueError("layer_types length does not match num_hidden_layers")
            return tuple(
                layer
                for layer, layer_type in enumerate(layer_types)
                if layer_type == "linear_attention"
            )
        return tuple(
            sorted(
                int(match.group(1))
                for name in self.checkpoint.weight_map
                if (match := _LINEAR_PATTERN.match(name)) is not None
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
        layer_directory = output_root / "linear_layers"
        layer_directory.mkdir(parents=True, exist_ok=True)
        layers = self.linear_layers
        for completed, layer in enumerate(layers, start=1):
            destination = layer_directory / f"layer-{layer:03d}.coli"
            if destination.exists() and not overwrite:
                raise FileExistsError(f"linear layer file already exists: {destination}")
            self._convert_layer(layer, destination, quantization)
            if progress is not None:
                progress(completed, len(layers))

        storage = {
            "mode": f"{quantization}-gated-deltanet",
            "path_pattern": "linear_layers/layer-{layer:03d}.coli",
            "layer_count": len(layers),
            "layers": list(layers),
        }
        manifest_path = output_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["linear_layer_storage"] = storage
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
            "in_proj_qkv.weight": self._read(
                f"{prefix}.linear_attn.in_proj_qkv.weight"
            ),
            "in_proj_z.weight": self._read(
                f"{prefix}.linear_attn.in_proj_z.weight"
            ),
            "in_proj_b.weight": self._read(
                f"{prefix}.linear_attn.in_proj_b.weight"
            ),
            "in_proj_a.weight": self._read(
                f"{prefix}.linear_attn.in_proj_a.weight"
            ),
            "conv1d.weight": self._read(f"{prefix}.linear_attn.conv1d.weight"),
            "dt_bias": self._read(f"{prefix}.linear_attn.dt_bias"),
            "A_log": self._read(
                f"{prefix}.linear_attn.A_log", allowed_dtypes=("BF16", "F32")
            ),
            "norm.weight": self._read(
                f"{prefix}.linear_attn.norm.weight",
                allowed_dtypes=("BF16", "F32"),
            ),
            "out_proj.weight": self._read(
                f"{prefix}.linear_attn.out_proj.weight"
            ),
        }
        geometry = self._geometry()
        self._validate_shapes(layer, tensors, geometry)
        payloads = []
        quantized = {}
        matrix_names = {
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "out_proj.weight",
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

    def _geometry(self) -> dict[str, int]:
        config = self.checkpoint.text_config
        return {
            "hidden_size": self.checkpoint.hidden_size,
            "num_key_heads": int(config["linear_num_key_heads"]),
            "num_value_heads": int(config["linear_num_value_heads"]),
            "key_head_dim": int(config["linear_key_head_dim"]),
            "value_head_dim": int(config["linear_value_head_dim"]),
            "conv_kernel_size": int(config["linear_conv_kernel_dim"]),
        }

    def _read(
        self, name: str, *, allowed_dtypes: tuple[str, ...] = ("BF16",)
    ) -> tuple[str, tuple[int, ...], bytes]:
        self.checkpoint.source_tensor(name, require_weights=True)
        safe_file = self.checkpoint.safe_file_for(name)
        info = safe_file.tensor(name)
        if info.dtype not in allowed_dtypes:
            expected = " or ".join(allowed_dtypes)
            raise ValueError(f"tensor {name} must be {expected}, got {info.dtype}")
        return info.dtype, info.shape, safe_file.read(name)

    def _validate_shapes(
        self,
        layer: int,
        tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
        geometry: dict[str, int],
    ) -> None:
        hidden = geometry["hidden_size"]
        key_heads = geometry["num_key_heads"]
        value_heads = geometry["num_value_heads"]
        key_head_dim = geometry["key_head_dim"]
        value_head_dim = geometry["value_head_dim"]
        kernel = geometry["conv_kernel_size"]
        key_dim = key_heads * key_head_dim
        value_dim = value_heads * value_head_dim
        conv_dim = key_dim * 2 + value_dim
        if key_heads <= 0 or value_heads <= 0 or value_heads % key_heads:
            raise ValueError("value heads must be divisible by key heads")
        if min(key_head_dim, value_head_dim, kernel) <= 0:
            raise ValueError("Gated DeltaNet dimensions must be positive")
        expected = {
            "input_layernorm.weight": (hidden,),
            "in_proj_qkv.weight": (conv_dim, hidden),
            "in_proj_z.weight": (value_dim, hidden),
            "in_proj_b.weight": (value_heads, hidden),
            "in_proj_a.weight": (value_heads, hidden),
            "conv1d.weight": (conv_dim, 1, kernel),
            "dt_bias": (value_heads,),
            "A_log": (value_heads,),
            "norm.weight": (value_head_dim,),
            "out_proj.weight": (hidden, value_dim),
        }
        mismatches = [
            f"{name} {tensors[name][1]} != {shape}"
            for name, shape in expected.items()
            if tensors[name][1] != shape
        ]
        if mismatches:
            raise ValueError(
                f"layer {layer} invalid Gated DeltaNet tensors: "
                + ", ".join(mismatches)
            )

