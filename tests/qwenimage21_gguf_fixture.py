"""A tiny Qwen-Image-2.1 GGUF DiT fixture, for testing GGUF transformer loading and inference."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from tests import qwenimage21_hf_fixture as hf

GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
GGUF_STRING = 8
GGML_F32 = 0
ALIGNMENT = 32


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, kind: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", kind) + payload


def _uint(value: int) -> bytes:
    return struct.pack("<I", value)


def build_gguf_transformer(path: Path) -> Path:
    """Writes a GGUF DiT file from the transformer tensors in qwenimage21_hf_fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hf_tensors = hf._transformer_tensors()

    # Convert HF tensor dictionary to numpy arrays
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, raw) in hf_tensors.items():
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32).reshape(shape)
        arrays[name] = arr

    # Map to GGUF DiT tensors:
    # 1. Attention: to_q, to_k, to_v remain separate
    # 2. MLP: fuse gate_layer and proj into gate_up
    tensors: dict[str, np.ndarray] = {
        "img_in.weight": arrays["img_in.weight"],
        "txt_in.text_norm.weight": arrays["txt_in.text_norm.weight"],
        "txt_in.in_layer.weight": arrays["txt_in.in_layer.weight"],
        "txt_in.out_layer.weight": arrays["txt_in.out_layer.weight"],
        "time_text_embed.timestep_embedder.linear_1.weight": arrays[
            "time_text_embed.timestep_embedder.linear_1.weight"
        ],
        "time_text_embed.timestep_embedder.linear_2.weight": arrays[
            "time_text_embed.timestep_embedder.linear_2.weight"
        ],
        "modulation.1.weight": arrays["modulation.1.weight"],
        "norm_out.linear.weight": arrays["norm_out.linear.weight"],
        "proj_out.weight": arrays["proj_out.weight"],
    }

    for index in range(hf.DIT_LAYERS):
        prefix = f"transformer_blocks.{index}."
        tensors[prefix + "attn.norm_q.weight"] = arrays[prefix + "attn.norm_q.weight"]
        tensors[prefix + "attn.norm_k.weight"] = arrays[prefix + "attn.norm_k.weight"]
        tensors[prefix + "attn.to_q.weight"] = arrays[prefix + "attn.to_q.weight"]
        tensors[prefix + "attn.to_k.weight"] = arrays[prefix + "attn.to_k.weight"]
        tensors[prefix + "attn.to_v.weight"] = arrays[prefix + "attn.to_v.weight"]
        tensors[prefix + "attn.to_out.0.weight"] = arrays[prefix + "attn.to_out.0.weight"]

        # Concatenate gate_layer and proj into gate_up along axis 0
        gate = arrays[prefix + "img_mlp.gate_layer.weight"]
        proj = arrays[prefix + "img_mlp.proj.weight"]
        gate_up = np.concatenate([gate, proj], axis=0)
        tensors[prefix + "img_mlp.gate_up.weight"] = gate_up
        tensors[prefix + "img_mlp.out.weight"] = arrays[prefix + "img_mlp.out.weight"]

    metadata = [
        _kv("general.architecture", GGUF_STRING, _string("qwen_image")),
        _kv("general.name", GGUF_STRING, _string("qwen_image_2.1_test")),
    ]

    body = b"".join(metadata)
    descriptors = bytearray()
    payload = bytearray()
    for name, array in tensors.items():
        values = np.ascontiguousarray(array, dtype=np.float32)
        dimensions = list(values.shape)[::-1]
        descriptors += _string(name)
        descriptors += struct.pack("<I", len(dimensions))
        descriptors += b"".join(struct.pack("<Q", extent) for extent in dimensions)
        descriptors += struct.pack("<IQ", GGML_F32, len(payload))
        payload += values.tobytes()
        pad = (ALIGNMENT - len(payload) % ALIGNMENT) % ALIGNMENT
        payload += b"\0" * pad

    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata))
    prefix_length = len(header) + len(body) + len(descriptors)
    pad = (ALIGNMENT - prefix_length % ALIGNMENT) % ALIGNMENT

    with path.open("wb") as handle:
        handle.write(header)
        handle.write(body)
        handle.write(descriptors)
        handle.write(b"\0" * pad)
        handle.write(payload)

    return path
