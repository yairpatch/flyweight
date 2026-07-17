from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .converter import QwenSafetensorCheckpoint
from .q4 import BLOCK_SIZE, Q4BlockTensor
from .tensor_container import TensorPayload, write_coli_tensor_file


class QwenMoELayerConverter:
    """Converts per-layer router, shared expert, gate, and RMSNorm tensors."""

    def __init__(self, source: Path | str):
        self.checkpoint = QwenSafetensorCheckpoint(source)

    def convert(
        self,
        output: Path | str,
        *,
        overwrite: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, object]:
        output_root = Path(output)
        layer_directory = output_root / "moe_layers"
        layer_directory.mkdir(parents=True, exist_ok=True)
        for layer in range(self.checkpoint.layers):
            destination = layer_directory / f"layer-{layer:03d}.coli"
            if destination.exists() and not overwrite:
                raise FileExistsError(f"MoE layer file already exists: {destination}")
            self._convert_layer(layer, destination)
            if progress is not None:
                progress(layer + 1, self.checkpoint.layers)

        storage = {
            "mode": "q4-shared-bf16-router",
            "path_pattern": "moe_layers/layer-{layer:03d}.coli",
            "layer_count": self.checkpoint.layers,
        }
        manifest_path = output_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["moe_layer_storage"] = storage
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        return storage

    def _convert_layer(self, layer: int, destination: Path) -> None:
        prefix = f"model.language_model.layers.{layer}"
        router = self._read(f"{prefix}.mlp.gate.weight")
        shared_gate_projection = self._read(
            f"{prefix}.mlp.shared_expert.gate_proj.weight"
        )
        shared_up_projection = self._read(
            f"{prefix}.mlp.shared_expert.up_proj.weight"
        )
        shared_down_projection = self._read(
            f"{prefix}.mlp.shared_expert.down_proj.weight"
        )
        shared_gate = self._read(f"{prefix}.mlp.shared_expert_gate.weight")
        post_attention_norm = self._read(
            f"{prefix}.post_attention_layernorm.weight"
        )
        self._validate_shapes(
            layer,
            router[1],
            shared_gate_projection[1],
            shared_up_projection[1],
            shared_down_projection[1],
            shared_gate[1],
            post_attention_norm[1],
        )

        fused_gate_up_shape = (
            shared_gate_projection[1][0] + shared_up_projection[1][0],
            shared_gate_projection[1][1],
        )
        fused_gate_up = Q4BlockTensor.from_bf16(
            shared_gate_projection[2] + shared_up_projection[2], fused_gate_up_shape
        )
        down = Q4BlockTensor.from_bf16(
            shared_down_projection[2], shared_down_projection[1]
        )
        quantization = {
            "scheme": "q4_symmetric",
            "block_size": BLOCK_SIZE,
            "tensors": {
                "shared_expert.gate_up_proj": fused_gate_up.metadata(),
                "shared_expert.down_proj": down.metadata(),
            },
        }
        write_coli_tensor_file(
            destination,
            [
                TensorPayload("router.weight", router[0], router[1], router[2]),
                TensorPayload(
                    "shared_expert_gate.weight",
                    shared_gate[0],
                    shared_gate[1],
                    shared_gate[2],
                ),
                TensorPayload(
                    "post_attention_layernorm.weight",
                    post_attention_norm[0],
                    post_attention_norm[1],
                    post_attention_norm[2],
                ),
                *fused_gate_up.payloads("shared_expert.gate_up_proj"),
                *down.payloads("shared_expert.down_proj"),
            ],
            metadata={
                "architecture": self.checkpoint.text_config.get("model_type"),
                "layer": layer,
                "top_k": int(self.checkpoint.text_config["num_experts_per_tok"]),
                "rms_norm_eps": float(
                    self.checkpoint.text_config.get("rms_norm_eps", 1e-6)
                ),
                "quantization": quantization,
            },
        )

    def _read(self, name: str) -> tuple[str, tuple[int, ...], bytes]:
        source = self.checkpoint.source_tensor(name, require_weights=True)
        safe_file = self.checkpoint.safe_file_for(name)
        info = safe_file.tensor(name)
        if info.dtype != "BF16":
            raise ValueError(f"tensor {name} must be BF16, got {info.dtype}")
        return info.dtype, info.shape, safe_file.read(name)

    def _validate_shapes(
        self,
        layer: int,
        router: tuple[int, ...],
        shared_gate_projection: tuple[int, ...],
        shared_up_projection: tuple[int, ...],
        shared_down_projection: tuple[int, ...],
        shared_gate: tuple[int, ...],
        post_attention_norm: tuple[int, ...],
    ) -> None:
        checkpoint = self.checkpoint
        intermediate = int(
            checkpoint.text_config["shared_expert_intermediate_size"]
        )
        hidden = checkpoint.hidden_size
        expected = {
            "router": (checkpoint.experts_per_layer, hidden),
            "shared_gate_projection": (intermediate, hidden),
            "shared_up_projection": (intermediate, hidden),
            "shared_down_projection": (hidden, intermediate),
            "shared_gate": (1, hidden),
            "post_attention_norm": (hidden,),
        }
        actual = {
            "router": router,
            "shared_gate_projection": shared_gate_projection,
            "shared_up_projection": shared_up_projection,
            "shared_down_projection": shared_down_projection,
            "shared_gate": shared_gate,
            "post_attention_norm": post_attention_norm,
        }
        mismatches = [
            f"{name} {actual[name]} != {shape}"
            for name, shape in expected.items()
            if actual[name] != shape
        ]
        if mismatches:
            raise ValueError(f"layer {layer} invalid MoE tensors: {', '.join(mismatches)}")
