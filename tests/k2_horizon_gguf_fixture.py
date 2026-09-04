"""Builds a tiny but complete k2-horizon GGUF fixture."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from tests.dense_gguf_fixture import _quantize_q8_0

GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGML_F32 = 0
GGML_Q8_0 = 8

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
        # MoE and MoVA, as the 36B-A4B member carries them. Zero experts keeps
        # the dense layout the smaller members ship, which is the default.
        experts: int = 0,
        experts_used: int = 2,
        expert_intermediate: int = 64,
        shared_intermediate: int = 64,
        leading_dense: int = 1,
        expert_weights_scale: float = 2.5,
        expert_weights_norm: bool = True,
        value_experts: int = 0,
        value_experts_used: int = 2,
        attention_gate: bool = False,
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
        self.experts = experts
        self.experts_used = experts_used
        self.expert_intermediate = expert_intermediate
        self.shared_intermediate = shared_intermediate
        self.leading_dense = leading_dense
        self.expert_weights_scale = expert_weights_scale
        self.expert_weights_norm = expert_weights_norm
        self.value_experts = value_experts
        self.value_experts_used = value_experts_used
        self.attention_gate = attention_gate

    def is_moe_layer(self, layer: int) -> bool:
        return self.experts > 0 and layer >= self.leading_dense

    def is_mova_layer(self, layer: int) -> bool:
        return self.is_moe_layer(layer) and self.value_experts > 0


def build_k2_horizon_gguf(
    path: Path | str, spec: K2HorizonSpec | None = None, seed: int = 42
) -> K2HorizonSpec:
    """Write the synthetic K2-Horizon fixture and return its geometry."""
    spec = spec or K2HorizonSpec()
    rng = np.random.default_rng(seed)
    # (name, gguf shape, ggml type, payload bytes)
    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []

    def projection(name: str, inputs: int, outputs: int, scale: float = 0.25) -> None:
        # GGUF reports [inputs, outputs] while the byte stream is [outputs, inputs].
        data = (rng.standard_normal((outputs, inputs)) * scale).astype(np.float32)
        tensors.append((name, (inputs, outputs), GGML_F32, data.tobytes()))

    def vector(name: str, size: int, value: float | None = None, scale: float = 0.05) -> None:
        data = (
            np.full(size, value, dtype=np.float32)
            if value is not None
            else (rng.standard_normal(size) * scale).astype(np.float32)
        )
        tensors.append((name, (size,), GGML_F32, data.astype(np.float32).tobytes()))

    projection("token_embd.weight", spec.hidden, spec.vocabulary)
    vector("output_norm.weight", spec.hidden, value=1.0)
    projection("output.weight", spec.hidden, spec.vocabulary)

    def stack(name: str, inputs: int, outputs: int, count: int, scale: float = 0.25,
              q8: bool = False) -> None:
        """A 3-D expert stack, reported [inputs, outputs, count].

        The value-expert stack is written as Q8_0 rather than f32: the device
        value-expert kernels exist for the quantized types the published
        checkpoints ship, and an f32 stack would keep the cache path untested.
        """
        data = (rng.standard_normal((count, outputs, inputs)) * scale).astype(np.float32)
        if q8:
            payload = _quantize_q8_0(data.reshape(count * outputs, inputs))
            tensors.append((name, (inputs, outputs, count), GGML_Q8_0, payload))
        else:
            tensors.append((name, (inputs, outputs, count), GGML_F32, data.tobytes()))

    kv_width = spec.kv_heads * spec.head_dim
    for layer in range(spec.layers):
        prefix = f"blk.{layer}."
        vector(prefix + "attn_norm.weight", spec.hidden, value=1.0)
        projection(prefix + "attn_q.weight", spec.hidden, spec.heads * spec.head_dim)
        projection(prefix + "attn_k.weight", spec.hidden, spec.kv_heads * spec.head_dim)
        if spec.is_mova_layer(layer):
            # MoVA replaces attn_v with a router, its selection bias and the
            # per-expert value stack.
            projection(prefix + "attn_v_gate.weight", spec.hidden, spec.value_experts)
            vector(prefix + "attn_v_gate.bias", spec.value_experts)
            stack(prefix + "attn_v_exps.weight", spec.hidden, kv_width, spec.value_experts, q8=True)
        else:
            projection(prefix + "attn_v.weight", spec.hidden, kv_width)
        projection(prefix + "attn_output.weight", spec.heads * spec.head_dim, spec.hidden)
        if spec.attention_gate:
            projection(prefix + "attn_gate.weight", spec.hidden, spec.heads * spec.head_dim)
        vector(prefix + "ffn_norm.weight", spec.hidden, value=1.0)
        if spec.is_moe_layer(layer):
            projection(prefix + "ffn_gate_inp.weight", spec.hidden, spec.experts)
            vector(prefix + "exp_probs_b.bias", spec.experts)
            stack(prefix + "ffn_gate_exps.weight", spec.hidden, spec.expert_intermediate, spec.experts)
            stack(prefix + "ffn_up_exps.weight", spec.hidden, spec.expert_intermediate, spec.experts)
            stack(prefix + "ffn_down_exps.weight", spec.expert_intermediate, spec.hidden, spec.experts)
            projection(prefix + "ffn_gate_shexp.weight", spec.hidden, spec.shared_intermediate)
            projection(prefix + "ffn_up_shexp.weight", spec.hidden, spec.shared_intermediate)
            projection(prefix + "ffn_down_shexp.weight", spec.shared_intermediate, spec.hidden)
        else:
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

    if spec.experts:
        metadata += [
            _kv("k2-horizon.expert_count", GGUF_UINT32, struct.pack("<I", spec.experts)),
            _kv("k2-horizon.expert_used_count", GGUF_UINT32, struct.pack("<I", spec.experts_used)),
            _kv("k2-horizon.expert_feed_forward_length", GGUF_UINT32, struct.pack("<I", spec.expert_intermediate)),
            _kv("k2-horizon.expert_shared_count", GGUF_UINT32, struct.pack("<I", 1)),
            _kv("k2-horizon.expert_shared_feed_forward_length", GGUF_UINT32, struct.pack("<I", spec.shared_intermediate)),
            _kv("k2-horizon.leading_dense_block_count", GGUF_UINT32, struct.pack("<I", spec.leading_dense)),
            _kv("k2-horizon.moe_every_n_layers", GGUF_UINT32, struct.pack("<I", 1)),
            # 2 is sigmoid, the only gating the published checkpoints use.
            _kv("k2-horizon.expert_gating_func", GGUF_UINT32, struct.pack("<I", 2)),
            _kv("k2-horizon.expert_weights_scale", GGUF_FLOAT32, struct.pack("<f", spec.expert_weights_scale)),
            _kv("k2-horizon.expert_weights_norm", GGUF_BOOL, struct.pack("<B", 1 if spec.expert_weights_norm else 0)),
        ]
    if spec.value_experts:
        metadata += [
            _kv("k2-horizon.attention.value_expert_count", GGUF_UINT32, struct.pack("<I", spec.value_experts)),
            _kv("k2-horizon.attention.value_expert_used_count", GGUF_UINT32, struct.pack("<I", spec.value_experts_used)),
        ]

    infos = bytearray()
    payloads = bytearray()
    for name, shape, ggml_type, raw in tensors:
        offset = len(payloads)
        infos += _string(name)
        infos += struct.pack("<I", len(shape))
        infos += b"".join(struct.pack("<Q", dim) for dim in shape)
        infos += struct.pack("<IQ", ggml_type, offset)
        elements = int(np.prod(shape))
        expected = elements * 4 if ggml_type == GGML_F32 else elements // 32 * 34
        assert len(raw) == expected, name
        payloads += raw
        payloads += b"\0" * ((-len(payloads)) % ALIGNMENT)

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    body = bytearray(header + b"".join(metadata) + bytes(infos))
    body += b"\0" * ((-len(body)) % ALIGNMENT)
    Path(path).write_bytes(bytes(body) + bytes(payloads))
    return spec
