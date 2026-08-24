"""Builds a tiny but complete laguna GGUF.

The published Laguna checkpoints are hundreds of gigabytes and quantized with
codebook formats, so there is no small real model that exercises the laguna
plan. This writes a synthetic one -- same tensor names, same layout conventions,
f32 weights -- covering the parts of the architecture that differ from Qwen: the
per-layer head count, the period-4 full/sliding attention cycle with its two
RoPE configurations, the softplus per-head attention gate, the leading dense
block ahead of the MoE blocks, and sigmoid routing with a score-correction bias.
"""
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


class LagunaSpec:
    """Geometry of the synthetic model.

    Defaults mirror the real checkpoint's structure at toy widths: full
    attention on every fourth block with fewer heads and a partial rotary span,
    sliding-window attention elsewhere with the whole head rotated.
    """

    def __init__(
        self,
        *,
        hidden: int = 128,
        layers: int = 8,
        dense_intermediate: int = 256,
        expert_intermediate: int = 64,
        vocabulary: int = 96,
        full_heads: int = 4,
        swa_heads: int = 6,
        kv_heads: int = 2,
        head_dim: int = 32,
        experts: int = 8,
        experts_used: int = 2,
        sliding_window: int = 16,
        leading_dense: int = 1,
    ):
        self.hidden = hidden
        self.layers = layers
        self.dense_intermediate = dense_intermediate
        self.expert_intermediate = expert_intermediate
        self.vocabulary = vocabulary
        self.full_heads = full_heads
        self.swa_heads = swa_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.experts = experts
        self.experts_used = experts_used
        self.sliding_window = sliding_window
        self.leading_dense = leading_dense

    def is_sliding(self, layer: int) -> bool:
        # Period 4 starting with a full-attention layer, as the architecture
        # implies; the GGUF carries no explicit pattern.
        return layer % 4 != 0

    def heads(self, layer: int) -> int:
        return self.swa_heads if self.is_sliding(layer) else self.full_heads

    def is_moe(self, layer: int) -> bool:
        return layer >= self.leading_dense


def build_laguna_gguf(
    path: Path | str, spec: LagunaSpec | None = None, seed: int = 11
) -> LagunaSpec:
    """Write the fixture and return its geometry."""
    spec = spec or LagunaSpec()
    rng = np.random.default_rng(seed)
    tensors: list[tuple[str, tuple[int, ...], np.ndarray]] = []

    def projection(name: str, inputs: int, outputs: int, scale: float = 0.25) -> None:
        # GGUF reports [input, output] while the byte stream is [output, input].
        data = (rng.standard_normal((outputs, inputs)) * scale).astype(np.float32)
        tensors.append((name, (inputs, outputs), data))

    def stacked(name: str, inputs: int, outputs: int, scale: float = 0.25) -> None:
        """A stacked per-expert tensor, [input, output, expert]."""
        data = (
            rng.standard_normal((spec.experts, outputs, inputs)) * scale
        ).astype(np.float32)
        tensors.append((name, (inputs, outputs, spec.experts), data))

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
        heads = spec.heads(layer)
        vector(prefix + "attn_norm.weight", spec.hidden, value=1.0)
        projection(prefix + "attn_q.weight", spec.hidden, heads * spec.head_dim)
        projection(prefix + "attn_k.weight", spec.hidden, spec.kv_heads * spec.head_dim)
        projection(prefix + "attn_v.weight", spec.hidden, spec.kv_heads * spec.head_dim)
        projection(prefix + "attn_output.weight", heads * spec.head_dim, spec.hidden)
        vector(prefix + "attn_q_norm.weight", spec.head_dim, value=1.0)
        vector(prefix + "attn_k_norm.weight", spec.head_dim, value=1.0)
        # One gate scalar per head, so the projection is n_head wide.
        projection(prefix + "attn_gate.weight", spec.hidden, heads)
        vector(prefix + "ffn_norm.weight", spec.hidden, value=1.0)
        if spec.is_moe(layer):
            projection(prefix + "ffn_gate_inp.weight", spec.hidden, spec.experts)
            vector(prefix + "exp_probs_b.bias", spec.experts)
            stacked(prefix + "ffn_gate_exps.weight", spec.hidden, spec.expert_intermediate)
            stacked(prefix + "ffn_up_exps.weight", spec.hidden, spec.expert_intermediate)
            stacked(prefix + "ffn_down_exps.weight", spec.expert_intermediate, spec.hidden)
            projection(prefix + "ffn_gate_shexp.weight", spec.hidden, spec.expert_intermediate)
            projection(prefix + "ffn_up_shexp.weight", spec.hidden, spec.expert_intermediate)
            projection(prefix + "ffn_down_shexp.weight", spec.expert_intermediate, spec.hidden)
        else:
            projection(prefix + "ffn_gate.weight", spec.hidden, spec.dense_intermediate)
            projection(prefix + "ffn_up.weight", spec.hidden, spec.dense_intermediate)
            projection(prefix + "ffn_down.weight", spec.dense_intermediate, spec.hidden)

    # A byte-level vocabulary plus the control tokens the chat template uses, so
    # tokenization round-trips without needing a real merge table.
    vocabulary = [chr(index) for index in range(32, 32 + spec.vocabulary - 4)]
    control = ["〈|EOS|〉", "〈|CODE_START|〉", "〈|CODE_END|〉", "〈|UNK|〉"]
    vocabulary = control + vocabulary
    vocabulary = vocabulary[: spec.vocabulary]
    token_types = [3] * len(control) + [1] * (len(vocabulary) - len(control))

    metadata = [
        _kv("general.architecture", GGUF_STRING, _string("laguna")),
        _kv("general.name", GGUF_STRING, _string("laguna-fixture")),
        _kv("laguna.block_count", GGUF_UINT32, struct.pack("<I", spec.layers)),
        _kv("laguna.embedding_length", GGUF_UINT32, struct.pack("<I", spec.hidden)),
        _kv("laguna.feed_forward_length", GGUF_UINT32, struct.pack("<I", spec.dense_intermediate)),
        _kv("laguna.expert_feed_forward_length", GGUF_UINT32, struct.pack("<I", spec.expert_intermediate)),
        _kv("laguna.expert_shared_feed_forward_length", GGUF_UINT32, struct.pack("<I", spec.expert_intermediate)),
        _kv("laguna.context_length", GGUF_UINT32, struct.pack("<I", 512)),
        _kv(
            "laguna.attention.head_count",
            GGUF_ARRAY,
            _uint_array([spec.heads(layer) for layer in range(spec.layers)]),
        ),
        _kv("laguna.attention.head_count_kv", GGUF_UINT32, struct.pack("<I", spec.kv_heads)),
        _kv("laguna.attention.key_length", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("laguna.attention.value_length", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("laguna.attention.sliding_window", GGUF_UINT32, struct.pack("<I", spec.sliding_window)),
        _kv("laguna.attention.layer_norm_rms_epsilon", GGUF_FLOAT32, struct.pack("<f", 1e-6)),
        # Full-attention layers rotate half the head with the long-context base
        # and YaRN; sliding-window layers rotate the whole head with the short one.
        _kv("laguna.rope.dimension_count", GGUF_UINT32, struct.pack("<I", spec.head_dim // 2)),
        _kv("laguna.rope.dimension_count_swa", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("laguna.rope.freq_base", GGUF_FLOAT32, struct.pack("<f", 500_000.0)),
        _kv("laguna.rope.freq_base_swa", GGUF_FLOAT32, struct.pack("<f", 10_000.0)),
        _kv("laguna.rope.scaling.type", GGUF_STRING, _string("yarn")),
        _kv("laguna.rope.scaling.factor", GGUF_FLOAT32, struct.pack("<f", 32.0)),
        _kv("laguna.rope.scaling.original_context_length", GGUF_UINT32, struct.pack("<I", 128)),
        _kv("laguna.rope.scaling.yarn_attn_factor", GGUF_FLOAT32, struct.pack("<f", 1.0)),
        _kv("laguna.rope.scaling.yarn_beta_fast", GGUF_FLOAT32, struct.pack("<f", 32.0)),
        _kv("laguna.rope.scaling.yarn_beta_slow", GGUF_FLOAT32, struct.pack("<f", 1.0)),
        _kv("laguna.expert_count", GGUF_UINT32, struct.pack("<I", spec.experts)),
        _kv("laguna.expert_used_count", GGUF_UINT32, struct.pack("<I", spec.experts_used)),
        _kv("laguna.expert_gating_func", GGUF_UINT32, struct.pack("<I", 2)),
        _kv("laguna.expert_weights_norm", GGUF_BOOL, struct.pack("<B", 1)),
        _kv("laguna.expert_weights_scale", GGUF_FLOAT32, struct.pack("<f", 2.5)),
        _kv("laguna.leading_dense_block_count", GGUF_UINT32, struct.pack("<I", spec.leading_dense)),
        _kv("tokenizer.ggml.model", GGUF_STRING, _string("gpt2")),
        _kv("tokenizer.ggml.pre", GGUF_STRING, _string("laguna")),
        _kv("tokenizer.ggml.tokens", GGUF_ARRAY, _string_array(vocabulary)),
        _kv("tokenizer.ggml.token_type", GGUF_ARRAY, _uint_array(token_types)),
        _kv("tokenizer.ggml.merges", GGUF_ARRAY, _string_array([])),
        # eos ends generation; eot ends one chat turn. Laguna closes an
        # assistant turn with </assistant> and never emits eos in conversation.
        _kv("tokenizer.ggml.eos_token_id", GGUF_UINT32, struct.pack("<I", 0)),
        _kv("tokenizer.ggml.eot_token_id", GGUF_UINT32, struct.pack("<I", 2)),
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
