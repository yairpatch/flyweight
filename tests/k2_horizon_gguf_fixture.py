"""Builds a tiny but complete k2-horizon GGUF fixture."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGML_F32 = 0

ALIGNMENT = 32


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, kind: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", kind) + payload


def _uint_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_UINT32, len(values)) + b"".join(
        struct.pack("<I", value) for value in values
    )


def _string_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_STRING, len(values)) + b"".join(
        _string(value) for value in values
    )


class K2HorizonSpec:
    """Geometry of the synthetic K2-Horizon model."""

    def __init__(
        self,
        *,
        hidden: int = 128,
        layers: int = 2,
        intermediate: int = 256,
        vocabulary: int = 96,
        heads: int = 4,
        kv_heads: int = 2,
        head_dim: int = 32,
        norm_groups: int = 4,
        rope_freq_base: float = 10_000_000.0,
        rotary_dim: int = 32,
    ):
        self.hidden = hidden
        self.layers = layers
        self.intermediate = intermediate
        self.vocabulary = vocabulary
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.norm_groups = norm_groups
        self.rope_freq_base = rope_freq_base
        self.rotary_dim = rotary_dim


def build_k2_horizon_gguf(
    path: Path | str, spec: K2HorizonSpec | None = None, seed: int = 42
) -> K2HorizonSpec:
    """Write the synthetic K2-Horizon fixture and return its geometry."""
    spec = spec or K2HorizonSpec()
    rng = np.random.default_rng(seed)
    tensors: list[tuple[str, tuple[int, ...], np.ndarray]] = []

    def projection(name: str, inputs: int, outputs: int, scale: float = 0.25) -> None:
        # GGUF reports [inputs, outputs] while the byte stream is [outputs, inputs].
        data = (rng.standard_normal((outputs, inputs)) * scale).astype(np.float32)
        tensors.append((name, (inputs, outputs), data))

    def vector(name: str, size: int, value: float | None = None, scale: float = 0.05) -> None:
        data = (
            np.full(size, value, dtype=np.float32)
            if value is not None
            else (rng.standard_normal(size) * scale).astype(np.float32)
        )
        tensors.append((name, (size,), data.astype(np.float32)))

    projection("token_embd.weight", spec.hidden, spec.vocabulary)
    vector("output_norm.weight", spec.hidden, value=1.0)
    projection("output.weight", spec.hidden, spec.vocabulary)

    for layer in range(spec.layers):
        prefix = f"blk.{layer}."
        vector(prefix + "attn_norm.weight", spec.hidden, value=1.0)
        projection(prefix + "attn_q.weight", spec.hidden, spec.heads * spec.head_dim)
        projection(prefix + "attn_k.weight", spec.hidden, spec.kv_heads * spec.head_dim)
        projection(prefix + "attn_v.weight", spec.hidden, spec.kv_heads * spec.head_dim)
        projection(prefix + "attn_output.weight", spec.heads * spec.head_dim, spec.hidden)
        vector(prefix + "ffn_norm.weight", spec.hidden, value=1.0)
        projection(prefix + "ffn_gate.weight", spec.hidden, spec.intermediate)
        projection(prefix + "ffn_up.weight", spec.hidden, spec.intermediate)
        projection(prefix + "ffn_down.weight", spec.intermediate, spec.hidden)

    vocabulary = [chr(index) for index in range(32, 32 + spec.vocabulary - 4)]
    control = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|unk|>"]
    vocabulary = control + vocabulary
    vocabulary = vocabulary[: spec.vocabulary]
    token_types = [3] * len(control) + [1] * (len(vocabulary) - len(control))

    metadata = [
        _kv("general.architecture", GGUF_STRING, _string("k2-horizon")),
        _kv("general.name", GGUF_STRING, _string("k2-horizon-fixture")),
        _kv("k2-horizon.block_count", GGUF_UINT32, struct.pack("<I", spec.layers)),
        _kv("k2-horizon.embedding_length", GGUF_UINT32, struct.pack("<I", spec.hidden)),
        _kv("k2-horizon.feed_forward_length", GGUF_UINT32, struct.pack("<I", spec.intermediate)),
        _kv("k2-horizon.context_length", GGUF_UINT32, struct.pack("<I", 512)),
        _kv("k2-horizon.attention.head_count", GGUF_UINT32, struct.pack("<I", spec.heads)),
        _kv("k2-horizon.attention.head_count_kv", GGUF_UINT32, struct.pack("<I", spec.kv_heads)),
        _kv("k2-horizon.attention.key_length", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("k2-horizon.attention.value_length", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("k2-horizon.attention.layer_norm_rms_epsilon", GGUF_FLOAT32, struct.pack("<f", 1e-6)),
        _kv("k2-horizon.attention.group_norm_groups", GGUF_UINT32, struct.pack("<I", spec.norm_groups)),
        _kv("k2-horizon.rope.dimension_count", GGUF_UINT32, struct.pack("<I", spec.rotary_dim)),
        _kv("k2-horizon.rope.freq_base", GGUF_FLOAT32, struct.pack("<f", spec.rope_freq_base)),
        _kv("tokenizer.ggml.model", GGUF_STRING, _string("gpt2")),
        _kv("tokenizer.ggml.pre", GGUF_STRING, _string("k2-horizon")),
        _kv("tokenizer.ggml.tokens", GGUF_ARRAY, _string_array(vocabulary)),
        _kv("tokenizer.ggml.token_type", GGUF_ARRAY, _uint_array(token_types)),
        _kv("tokenizer.ggml.merges", GGUF_ARRAY, _string_array([])),
        _kv("tokenizer.ggml.eos_token_id", GGUF_UINT32, struct.pack("<I", 0)),
        _kv("tokenizer.ggml.bos_token_id", GGUF_UINT32, struct.pack("<I", 0)),
        _kv("general.alignment", GGUF_UINT32, struct.pack("<I", ALIGNMENT)),
    ]

    infos = bytearray()
    payloads = bytearray()
    for name, shape, data in tensors:
        offset = len(payloads)
        infos += _string(name)
        infos += struct.pack("<I", len(shape))
        infos += b"".join(struct.pack("<Q", dim) for dim in shape)
        infos += struct.pack("<IQ", GGML_F32, offset)
        raw = np.ascontiguousarray(data, dtype=np.float32).tobytes()
        assert len(raw) == int(np.prod(shape)) * 4, name
        payloads += raw
        payloads += b"\0" * ((-len(payloads)) % ALIGNMENT)

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    body = bytearray(header + b"".join(metadata) + bytes(infos))
    body += b"\0" * ((-len(body)) % ALIGNMENT)
    Path(path).write_bytes(bytes(body) + bytes(payloads))
    return spec
