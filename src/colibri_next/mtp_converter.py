from __future__ import annotations

import json
from pathlib import Path

from .safetensors import SafeTensorFile
from .tensor_container import TensorPayload, write_coli_tensor_file


MTP_PREFIX = "mtp."


class QwenMtpConverter:
    """Extract the Qwen3.5/3.6 multi-token-prediction head to one container.

    The MTP head is one full MoE decoder layer plus a projection that fuses
    the draft token's embedding with the main model's last hidden state. All
    tensors are kept at BF16: quantizing the draft path measurably destroys
    draft acceptance, and at ~1.7 GiB the head fits RAM comfortably.
    """

    def __init__(self, source: Path | str):
        self.source = Path(source)
        index_path = self.source / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing safetensors index: {index_path}")
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))[
            "weight_map"
        ]
        self.tensor_shards = {
            name: shard
            for name, shard in weight_map.items()
            if name.startswith(MTP_PREFIX)
        }
        if not self.tensor_shards:
            raise ValueError(
                f"checkpoint has no {MTP_PREFIX}* tensors: {self.source}"
            )
        config = json.loads(
            (self.source / "config.json").read_text(encoding="utf-8")
        )
        self.text_config = config.get("text_config", config)

    def convert(
        self, output: Path | str, *, overwrite: bool = False
    ) -> dict[str, object]:
        output_root = Path(output)
        destination = output_root / "mtp.coli"
        if destination.exists() and not overwrite:
            raise FileExistsError(f"MTP file already exists: {destination}")
        payloads: list[TensorPayload] = []
        by_shard: dict[str, list[str]] = {}
        for name, shard in sorted(self.tensor_shards.items()):
            by_shard.setdefault(shard, []).append(name)
        for shard, names in sorted(by_shard.items()):
            container = SafeTensorFile(self.source / shard)
            for name in names:
                info = container.tensor(name)
                if info.dtype != "BF16":
                    raise ValueError(
                        f"MTP tensor {name} must be BF16, got {info.dtype}"
                    )
                payloads.append(
                    TensorPayload(
                        name[len(MTP_PREFIX) :],
                        info.dtype,
                        info.shape,
                        container.read(name),
                    )
                )
        config = self.text_config
        rope = config.get("rope_parameters", {})
        head_dim = int(config["head_dim"])
        rotary_factor = float(
            rope.get(
                "partial_rotary_factor",
                config.get("partial_rotary_factor", 1.0),
            )
        )
        metadata = {
            "kind": "qwen-mtp-head",
            "dtype": "bf16",
            "hidden_size": int(config["hidden_size"]),
            "num_attention_heads": int(config["num_attention_heads"]),
            "num_key_value_heads": int(config["num_key_value_heads"]),
            "head_dim": head_dim,
            "rotary_dim": int(head_dim * rotary_factor),
            "rope_theta": float(
                rope.get("rope_theta", config.get("rope_theta", 10000.0))
            ),
            "rms_norm_eps": float(config["rms_norm_eps"]),
            "num_experts": int(config["num_experts"]),
            "top_k": int(config["num_experts_per_tok"]),
            "moe_intermediate_size": int(config["moe_intermediate_size"]),
        }
        write_coli_tensor_file(destination, payloads, metadata=metadata)
        storage = {
            "mode": "bf16",
            "path": "mtp.coli",
            "tensor_count": len(payloads),
            "byte_size": destination.stat().st_size,
        }
        manifest_path = output_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["mtp_storage"] = storage
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        return storage
