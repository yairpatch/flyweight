"""Build a tiny synthetic ``inkling2`` GGUF.

The published Inkling-2-Mini checkpoints are ~8.5 GB with codebook quantizers,
so there is no small real file that exercises the metadata path. This fixture
writes a toy model with the real key names and a looped layer layout so config
parsing, tokenizer handling and the desktop support gate can be tested without
downloading anything.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

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


def _bool_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_BOOL, len(values)) + b"".join(
        struct.pack("<B", 1 if value else 0) for value in values
    )


def _string_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_STRING, len(values)) + b"".join(
        _string(value) for value in values
    )


class Inkling2Spec:
    """Geometry of the synthetic looped model."""

    def __init__(
        self,
        *,
        hidden: int = 32,
        applications: int = 12,
        experts: int = 8,
        experts_used: int = 2,
        vocabulary: int = 48,
        num_loops: int = 3,
        core_start: int = 2,
        core_end: int = 4,
        block_size: int = 2,
        head_dim: int = 8,
        heads: int = 4,
        kv_heads: int = 2,
    ):
        self.hidden = hidden
        self.applications = applications
        self.experts = experts
        self.experts_used = experts_used
        self.vocabulary = vocabulary
        self.num_loops = num_loops
        self.core_start = core_start
        self.core_end = core_end
        self.block_size = block_size
        self.head_dim = head_dim
        self.heads = heads
        self.kv_heads = kv_heads

    @property
    def sliding(self) -> tuple[bool, ...]:
        # The real checkpoint alternates five sliding layers with one full one.
        return tuple(index % self.block_size != self.block_size - 1
                     for index in range(self.applications))


def build_inkling2_gguf(
    path: Path | str,
    spec: Inkling2Spec | None = None,
    *,
    seed: int | None = None,
) -> Inkling2Spec:
    spec = spec or Inkling2Spec()

    # The vocabulary is deliberately tiny, but its tokens include a control
    # token and a leading space so the GPT-4o pre-tokenizer has something to do.
    tokens = ["<|end_message|>", " Hello", " world", "!", "123"] + [
        chr(0x100 + index) for index in range(spec.vocabulary - 5)
    ]
    tokens = tokens[: spec.vocabulary]
    token_types = [3] + [1] * (len(tokens) - 1)

    metadata = [
        _kv("general.architecture", GGUF_STRING, _string("inkling2")),
        _kv("general.name", GGUF_STRING, _string("Inkling-2-Mini-fixture")),
        _kv("general.type", GGUF_STRING, _string("model")),
        _kv("inkling2.block_count", GGUF_UINT32, struct.pack("<I", spec.applications)),
        _kv("inkling2.context_length", GGUF_UINT32, struct.pack("<I", 4096)),
        _kv("inkling2.embedding_length", GGUF_UINT32, struct.pack("<I", spec.hidden)),
        _kv("inkling2.feed_forward_length", GGUF_UINT32, struct.pack("<I", spec.hidden * 2)),
        _kv("inkling2.expert_feed_forward_length", GGUF_UINT32, struct.pack("<I", spec.hidden)),
        _kv("inkling2.attention.head_count", GGUF_ARRAY,
            _uint_array([spec.heads] * spec.applications)),
        _kv("inkling2.attention.head_count_kv", GGUF_ARRAY,
            _uint_array([spec.kv_heads] * spec.applications)),
        _kv("inkling2.attention.key_length", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("inkling2.attention.value_length", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("inkling2.attention.key_length_swa", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("inkling2.attention.value_length_swa", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("inkling2.attention.layer_norm_rms_epsilon", GGUF_FLOAT32, struct.pack("<f", 1e-6)),
        _kv("inkling2.expert_count", GGUF_UINT32, struct.pack("<I", spec.experts)),
        _kv("inkling2.expert_used_count", GGUF_UINT32, struct.pack("<I", spec.experts_used)),
        _kv("inkling2.num_loops", GGUF_UINT32, struct.pack("<I", spec.num_loops)),
        _kv("inkling2.loop.core_start", GGUF_UINT32, struct.pack("<I", spec.core_start)),
        _kv("inkling2.loop.core_end", GGUF_UINT32, struct.pack("<I", spec.core_end)),
        _kv("inkling2.loop.gate_rank", GGUF_UINT32, struct.pack("<I", 8)),
        _kv("inkling2.attn_res.block_size", GGUF_UINT32, struct.pack("<I", spec.block_size)),
        _kv("inkling2.vocab_size", GGUF_UINT32, struct.pack("<I", spec.vocabulary)),
        _kv("inkling2.vocab_size_unpadded", GGUF_UINT32, struct.pack("<I", spec.vocabulary)),
        _kv("inkling2.attention.sliding_window", GGUF_UINT32, struct.pack("<I", 16)),
        _kv("inkling2.attention.sliding_window_pattern", GGUF_ARRAY,
            _bool_array(spec.sliding)),
        _kv("inkling2.rope.dimension_count", GGUF_UINT32, struct.pack("<I", 0)),
        _kv("inkling2.shortconv.l_cache", GGUF_UINT32, struct.pack("<I", 4)),
        _kv("inkling2.expert_weights_scale", GGUF_FLOAT32, struct.pack("<f", 8.0)),
        _kv("inkling2.activation.situ_beta", GGUF_FLOAT32, struct.pack("<f", 4.0)),
        _kv("inkling2.activation.situ_linear_beta", GGUF_FLOAT32, struct.pack("<f", 25.0)),
        _kv("inkling2.final_logit_softcapping", GGUF_FLOAT32, struct.pack("<f", 30.0)),
        _kv("inkling2.attention.relative_length", GGUF_UINT32, struct.pack("<I", 8)),
        _kv("inkling2.attention.relative_extent", GGUF_UINT32, struct.pack("<I", 16)),
        _kv("inkling2.attention.qk_clip", GGUF_FLOAT32, struct.pack("<f", 22.0)),
        _kv("inkling2.attention.relative_clip", GGUF_FLOAT32, struct.pack("<f", 8.0)),
        _kv("inkling2.attention.log_scale_floor", GGUF_UINT32, struct.pack("<I", 8192)),
        _kv("inkling2.attention.log_scale_alpha", GGUF_FLOAT32, struct.pack("<f", 0.1)),
        _kv("tokenizer.ggml.model", GGUF_STRING, _string("gpt2")),
        _kv("tokenizer.ggml.pre", GGUF_STRING, _string("inkling2")),
        _kv("tokenizer.ggml.tokens", GGUF_ARRAY, _string_array(tokens)),
        _kv("tokenizer.ggml.token_type", GGUF_ARRAY, _uint_array(token_types)),
        _kv("tokenizer.ggml.merges", GGUF_ARRAY, _string_array([])),
        _kv("tokenizer.ggml.eos_token_id", GGUF_UINT32, struct.pack("<I", 0)),
        _kv("tokenizer.ggml.eot_token_id", GGUF_UINT32, struct.pack("<I", 0)),
        _kv("tokenizer.chat_template", GGUF_STRING, _string(
            "{% for m in messages %}{{ m.content }}{% endfor %}")),
        _kv("general.alignment", GGUF_UINT32, struct.pack("<I", ALIGNMENT)),
    ]

    # Tiny f32 weights for every tensor the native runtime resolves. Values are
    # irrelevant here -- this exercises tensor lookup, state sizing and one
    # forward step, not numerics against a reference.
    tensors: list[tuple[str, tuple[int, ...], bytes]] = []

    def add(name: str, *shape: int) -> None:
        count = 1
        for dim in shape:
            count *= dim
        if seed is None:
            raw = b"\0" * (count * 4)
        else:
            # Deterministic small non-zero weights so GPU and CPU parity tests
            # exercise real matmuls, routing, SiTU activations, and softcap.
            offset = (seed * 1315423911 + sum(ord(ch) * (idx + 1) for idx, ch in enumerate(name))) & 0xFFFF
            if name.endswith("_norm.weight") or name.endswith("_clip.weight"):
                values = [1.0 + 0.05 * math.sin(offset + i * 0.37) for i in range(count)]
            else:
                values = [0.05 * math.sin(offset + i * 0.13) for i in range(count)]
            raw = struct.pack(f"<{count}f", *values)
        tensors.append((name, tuple(shape), raw))

    add("token_embd.weight", spec.hidden, spec.vocabulary)
    add("token_embd_norm.weight", spec.hidden)
    add("output.weight", spec.hidden, spec.vocabulary)
    add("output_norm.weight", spec.hidden)
    add("output_res_query.weight", spec.hidden)
    add("loop_res_query.weight", spec.hidden, 24)
    add("loop_kv_res_query.weight", spec.hidden, 24)

    physical = spec.core_end + (spec.applications -
                                (spec.core_start + (spec.core_end - spec.core_start) * spec.num_loops))
    q_dim = spec.heads * spec.head_dim
    kv_dim = spec.kv_heads * spec.head_dim
    rel = 8
    extent = 16
    for layer in range(physical):
        pre = f"blk.{layer}."
        add(pre + "attn_q.weight", spec.hidden, q_dim)
        add(pre + "attn_k.weight", spec.hidden, kv_dim)
        add(pre + "attn_v.weight", spec.hidden, kv_dim)
        add(pre + "attn_output.weight", q_dim, spec.hidden)
        add(pre + "attn_gate.weight", spec.hidden, q_dim)
        add(pre + "attn_norm.weight", spec.hidden)
        add(pre + "ffn_norm.weight", spec.hidden)
        add(pre + "attn_k_conv.weight", 4, kv_dim)
        add(pre + "attn_v_conv.weight", 4, kv_dim)
        add(pre + "attn_post_conv.weight", 4, spec.hidden)
        add(pre + "ffn_post_conv.weight", 4, spec.hidden)
        add(pre + "attn_q_clip.weight", spec.head_dim, spec.heads)
        add(pre + "attn_rel_clip.weight", rel, spec.heads)
        add(pre + "attn_sink_proj.weight", spec.hidden, spec.heads)
        add(pre + "attn_rel.weight", spec.hidden, rel * spec.heads)
        add(pre + "attn_rel_bias.weight", rel, extent)
        add(pre + "ffn_gate_inp.weight", spec.hidden, spec.experts)
        add(pre + "exp_probs_b.bias", spec.experts)
        add(pre + "ffn_gate_up_exps.weight", spec.hidden, 2 * spec.hidden, spec.experts)
        add(pre + "ffn_down_exps.weight", spec.hidden, spec.hidden, spec.experts)
        add(pre + "ffn_res_query.weight", spec.hidden)
        if layer > 0:
            add(pre + "attn_res_query.weight", spec.hidden)
        if spec.core_start <= layer < spec.core_end:
            for name in ("attn", "k", "v", "mlp"):
                add(pre + f"loop_{name}_down.weight", 2 * spec.hidden, 8)
                add(pre + f"loop_{name}_gain.weight", 8, spec.hidden)
                add(pre + f"loop_{name}_forget.weight", 8, spec.hidden)
    infos = bytearray()
    payloads = bytearray()
    for name, shape, raw in tensors:
        offset = len(payloads)
        infos += _string(name)
        infos += struct.pack("<I", len(shape))
        infos += b"".join(struct.pack("<Q", dim) for dim in shape)
        infos += struct.pack("<IQ", GGML_F32, offset)
        payloads += raw
        payloads += b"\0" * ((-len(payloads)) % ALIGNMENT)

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    body = bytearray(header + b"".join(metadata) + bytes(infos))
    body += b"\0" * ((-len(body)) % ALIGNMENT)
    Path(path).write_bytes(bytes(body) + bytes(payloads))
    return spec
