from __future__ import annotations

import json
from pathlib import Path

from .converter import QwenSafetensorCheckpoint
from .matrix import encode_matrix, quantization_metadata
from .tensor_container import TensorPayload, write_coli_tensor_file


class QwenModelIOConverter:
    """Converts token embeddings, final RMSNorm, and the language-model head."""

    def __init__(self, source: Path | str):
        self.checkpoint = QwenSafetensorCheckpoint(source)

    def convert(
        self,
        output: Path | str,
        *,
        overwrite: bool = False,
        quantization: str = "bf16",
    ) -> dict[str, object]:
        if quantization not in {"bf16", "q4"}:
            raise ValueError(f"unsupported quantization: {quantization}")
        output_root = Path(output)
        output_root.mkdir(parents=True, exist_ok=True)
        destination = output_root / "model_io.coli"
        if destination.exists() and not overwrite:
            raise FileExistsError(f"model I/O file already exists: {destination}")

        embedding = self._read("model.language_model.embed_tokens.weight")
        final_norm = self._read("model.language_model.norm.weight")
        tied = bool(self.checkpoint.config.get("tie_word_embeddings", False))
        lm_head = None if tied else self._read("lm_head.weight")
        self._validate_shapes(embedding[1], final_norm[1], None if tied else lm_head[1])
        payloads = []
        quantized = {}
        encoded, metadata = encode_matrix(
            "embed_tokens.weight", embedding, quantization
        )
        payloads.extend(encoded)
        if metadata is not None:
            quantized["embed_tokens.weight"] = metadata
        payloads.append(TensorPayload("norm.weight", *final_norm))
        if lm_head is not None:
            encoded, metadata = encode_matrix(
                "lm_head.weight", lm_head, quantization
            )
            payloads.extend(encoded)
            if metadata is not None:
                quantized["lm_head.weight"] = metadata
        container_metadata = {
            "architecture": self.checkpoint.text_config.get("model_type"),
            "hidden_size": self.checkpoint.hidden_size,
            "vocab_size": int(self.checkpoint.text_config["vocab_size"]),
            "rms_norm_eps": float(
                self.checkpoint.text_config.get("rms_norm_eps", 1e-6)
            ),
            "tie_word_embeddings": tied,
        }
        if quantized:
            container_metadata["quantization"] = quantization_metadata(quantized)
        write_coli_tensor_file(
            destination, payloads, metadata=container_metadata
        )
        storage = {
            "mode": f"{quantization}-model-io",
            "path": "model_io.coli",
            "tie_word_embeddings": tied,
        }
        manifest_path = output_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["model_io_storage"] = storage
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        return storage

    def _read(self, name: str) -> tuple[str, tuple[int, ...], bytes]:
        self.checkpoint.source_tensor(name, require_weights=True)
        safe_file = self.checkpoint.safe_file_for(name)
        info = safe_file.tensor(name)
        if info.dtype != "BF16":
            raise ValueError(f"tensor {name} must be BF16, got {info.dtype}")
        return info.dtype, info.shape, safe_file.read(name)

    def _validate_shapes(
        self,
        embedding: tuple[int, ...],
        final_norm: tuple[int, ...],
        lm_head: tuple[int, ...] | None,
    ) -> None:
        hidden = self.checkpoint.hidden_size
        vocab = int(self.checkpoint.text_config["vocab_size"])
        expected_matrix = (vocab, hidden)
        mismatches = []
        if embedding != expected_matrix:
            mismatches.append(f"embedding {embedding} != {expected_matrix}")
        if final_norm != (hidden,):
            mismatches.append(f"final norm {final_norm} != {(hidden,)}")
        if lm_head is not None and lm_head != expected_matrix:
            mismatches.append(f"LM head {lm_head} != {expected_matrix}")
        if mismatches:
            raise ValueError("invalid model I/O tensors: " + ", ".join(mismatches))
