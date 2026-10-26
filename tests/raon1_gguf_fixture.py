"""Synthetic GGUF fixture for Raon-1 architecture tests.

Builds a minimal valid 2-layer Raon-1 model:
- Layer 0: RDA recurrent delta-rule linear attention + dense SiTU-GLU FFN
- Layer 1: MLA multi-head latent attention (NoPE) + top-2 MoE SiTU-GLU FFN + shared expert
- Raon-mHC 4-stream Helmert orthonormal mixing on both layers
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

VOCAB = 32
HIDDEN = 64
HEADS = 4
HEAD_DIM = 16
CHANNELS = HEADS * HEAD_DIM  # 64
CONV_WIDTH = 4
SSM_DT_RANK = 16
Q_LORA = 32
KV_LORA = 32
QK_ROPE = 16
QK_NOPE = 16
V_HEAD = 16
DENSE_FFN = 128
EXPERTS = 4
EXPERT_USED = 2
SHARED_EXPERTS = 1
EXPERT_FFN = 64
STREAMS = 4


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, kind: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", kind) + payload


def _uint(value: int) -> bytes:
    return struct.pack("<I", value)


def _uint_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_UINT32, len(values)) + b"".join(
        struct.pack("<I", int(value)) for value in values
    )


def _float_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_FLOAT32, len(values)) + b"".join(
        struct.pack("<f", float(value)) for value in values
    )


def _string_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_STRING, len(values)) + b"".join(
        _string(value) for value in values
    )


def _make_tensors(seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return (rng.standard_normal(shape).astype(np.float32) * 0.05)

    def ones(*shape):
        return np.ones(shape, dtype=np.float32)

    def zeros(*shape):
        return np.zeros(shape, dtype=np.float32)

    tensors: dict[str, np.ndarray] = {
        "token_embd.weight": rand(VOCAB, HIDDEN),
        "output_norm.weight": ones(HIDDEN),
        "output.weight": rand(VOCAB, HIDDEN),
    }

    # Layer 0: RDA + Dense FFN
    tensors["blk.0.attn_norm.weight"] = ones(HIDDEN)
    tensors["blk.0.ffn_norm.weight"] = ones(HIDDEN)

    tensors["blk.0.hc_attn_fn.weight"] = rand(17, STREAMS * HIDDEN)
    tensors["blk.0.hc_attn_base.weight"] = zeros(17)
    tensors["blk.0.hc_attn_scale.weight"] = ones(4)
    tensors["blk.0.hc_ffn_fn.weight"] = rand(17, STREAMS * HIDDEN)
    tensors["blk.0.hc_ffn_base.weight"] = zeros(17)
    tensors["blk.0.hc_ffn_scale.weight"] = ones(4)

    tensors["blk.0.attn_q.weight"] = rand(CHANNELS, HIDDEN)
    tensors["blk.0.attn_k.weight"] = rand(CHANNELS, HIDDEN)
    tensors["blk.0.attn_v.weight"] = rand(CHANNELS, HIDDEN)
    tensors["blk.0.ssm_conv1d_q.weight"] = rand(CHANNELS, 1, CONV_WIDTH)
    tensors["blk.0.ssm_conv1d_k.weight"] = rand(CHANNELS, 1, CONV_WIDTH)
    tensors["blk.0.ssm_conv1d_v.weight"] = rand(CHANNELS, 1, CONV_WIDTH)
    tensors["blk.0.ssm_f_a.weight"] = rand(SSM_DT_RANK, HIDDEN)
    tensors["blk.0.ssm_f_b.weight"] = rand(CHANNELS, SSM_DT_RANK)
    tensors["blk.0.ssm_beta.weight"] = rand(HEADS, HIDDEN)
    tensors["blk.0.ssm_rda.weight"] = rand(HEADS, HIDDEN)
    # ssm_a is stored as -exp(A_log), negative like the released checkpoint
    tensors["blk.0.ssm_a"] = ones(HEADS) * -0.9
    tensors["blk.0.ssm_dt.bias"] = zeros(CHANNELS)
    tensors["blk.0.ssm_g.weight"] = rand(CHANNELS, HIDDEN)
    tensors["blk.0.ssm_norm.weight"] = ones(HEAD_DIM)
    tensors["blk.0.attn_output.weight"] = rand(HIDDEN, CHANNELS)

    tensors["blk.0.ffn_gate.weight"] = rand(DENSE_FFN, HIDDEN)
    tensors["blk.0.ffn_up.weight"] = rand(DENSE_FFN, HIDDEN)
    tensors["blk.0.ffn_down.weight"] = rand(HIDDEN, DENSE_FFN)

    # Layer 1: MLA + Sparse MoE
    tensors["blk.1.attn_norm.weight"] = ones(HIDDEN)
    tensors["blk.1.ffn_norm.weight"] = ones(HIDDEN)

    tensors["blk.1.hc_attn_fn.weight"] = rand(17, STREAMS * HIDDEN)
    tensors["blk.1.hc_attn_base.weight"] = zeros(17)
    tensors["blk.1.hc_attn_scale.weight"] = ones(4)
    tensors["blk.1.hc_ffn_fn.weight"] = rand(17, STREAMS * HIDDEN)
    tensors["blk.1.hc_ffn_base.weight"] = zeros(17)
    tensors["blk.1.hc_ffn_scale.weight"] = ones(4)

    tensors["blk.1.attn_q_a.weight"] = rand(Q_LORA, HIDDEN)
    tensors["blk.1.attn_q_a_norm.weight"] = ones(Q_LORA)
    tensors["blk.1.attn_q_b.weight"] = rand(HEADS * (QK_NOPE + QK_ROPE), Q_LORA)
    tensors["blk.1.attn_kv_a_mqa.weight"] = rand(KV_LORA + QK_ROPE, HIDDEN)
    tensors["blk.1.attn_kv_a_norm.weight"] = ones(KV_LORA)
    tensors["blk.1.attn_k_b.weight"] = rand(HEADS, KV_LORA, QK_NOPE)
    tensors["blk.1.attn_v_b.weight"] = rand(HEADS, V_HEAD, KV_LORA)
    tensors["blk.1.attn_gate.weight"] = rand(HEADS * V_HEAD, HIDDEN)
    tensors["blk.1.attn_output.weight"] = rand(HIDDEN, HEADS * V_HEAD)

    tensors["blk.1.ffn_gate_inp.weight"] = rand(EXPERTS, HIDDEN)
    tensors["blk.1.exp_probs_b.bias"] = zeros(EXPERTS)
    tensors["blk.1.ffn_gate_exps.weight"] = rand(EXPERTS, EXPERT_FFN, HIDDEN)
    tensors["blk.1.ffn_up_exps.weight"] = rand(EXPERTS, EXPERT_FFN, HIDDEN)
    tensors["blk.1.ffn_down_exps.weight"] = rand(EXPERTS, HIDDEN, EXPERT_FFN)
    tensors["blk.1.ffn_gate_shexp.weight"] = rand(EXPERT_FFN, HIDDEN)
    tensors["blk.1.ffn_up_shexp.weight"] = rand(EXPERT_FFN, HIDDEN)
    tensors["blk.1.ffn_down_shexp.weight"] = rand(HIDDEN, EXPERT_FFN)

    return tensors


def build(directory: Path, name: str = "raon1.gguf") -> Path:
    """Build and write a synthetic Raon-1 GGUF model."""
    directory.mkdir(parents=True, exist_ok=True)
    tensors = _make_tensors()

    tokens = [f"<|token_{i}|>" for i in range(VOCAB)]
    tokens[0] = "<|pad|>"
    tokens[1] = "<|bos|>"
    tokens[2] = "<|eos|>"
    tokens[3] = "<|im_start|>"
    tokens[4] = "<|im_end|>"
    tokens[5] = "hello"
    tokens[6] = "world"
    tokens[7] = " "
    tokens[8] = "123"
    types = [1] * VOCAB
    types[0] = 3
    types[1] = 3
    types[2] = 3
    types[3] = 3
    types[4] = 3

    metadata = [
        _kv("general.architecture", GGUF_STRING, _string("raon1")),
        _kv("general.name", GGUF_STRING, _string("raon1-synthetic-test")),
        _kv("raon1.block_count", GGUF_UINT32, _uint(2)),
        _kv("raon1.context_length", GGUF_UINT32, _uint(512)),
        _kv("raon1.embedding_length", GGUF_UINT32, _uint(HIDDEN)),
        _kv("raon1.feed_forward_length", GGUF_UINT32, _uint(DENSE_FFN)),
        _kv("raon1.attention.head_count", GGUF_UINT32, _uint(HEADS)),
        _kv("raon1.attention.head_count_kv", GGUF_ARRAY, _uint_array([0, 1])),
        _kv("raon1.attention.head_dim", GGUF_UINT32, _uint(HEAD_DIM)),
        _kv("raon1.rope.freq_base", GGUF_FLOAT32, struct.pack("<f", 10000.0)),
        _kv("raon1.attention.layer_norm_rms_epsilon", GGUF_FLOAT32, struct.pack("<f", 1e-6)),
        _kv("raon1.expert_count", GGUF_UINT32, _uint(EXPERTS)),
        _kv("raon1.expert_used_count", GGUF_UINT32, _uint(EXPERT_USED)),
        _kv("raon1.expert_shared_count", GGUF_UINT32, _uint(SHARED_EXPERTS)),
        _kv("raon1.expert_feed_forward_length", GGUF_UINT32, _uint(EXPERT_FFN)),
        _kv("raon1.expert_shared_feed_forward_length", GGUF_UINT32, _uint(EXPERT_FFN)),
        _kv("raon1.leading_dense_block_count", GGUF_UINT32, _uint(1)),
        _kv("raon1.vocab_size", GGUF_UINT32, _uint(VOCAB)),
        _kv("raon1.ssm.conv_kernel", GGUF_UINT32, _uint(CONV_WIDTH)),
        _kv("raon1.kda.head_dim", GGUF_UINT32, _uint(HEAD_DIM)),
        _kv("raon1.ssm.time_step_rank", GGUF_UINT32, _uint(SSM_DT_RANK)),
        _kv("raon1.attention.kv_lora_rank", GGUF_UINT32, _uint(KV_LORA)),
        _kv("raon1.attention.q_lora_rank", GGUF_UINT32, _uint(Q_LORA)),
        _kv("raon1.rope.dimension_count", GGUF_UINT32, _uint(QK_ROPE)),
        _kv("raon1.attention.key_length_mla", GGUF_UINT32, _uint(QK_NOPE + QK_ROPE)),
        _kv("raon1.attention.value_length_mla", GGUF_UINT32, _uint(V_HEAD)),
        _kv("raon1.hyper_connection.count", GGUF_UINT32, _uint(STREAMS)),
        _kv("raon1.hyper_connection.magnitude", GGUF_FLOAT32, struct.pack("<f", 2.0)),
        _kv("raon1.activation.situ_beta", GGUF_FLOAT32, struct.pack("<f", 4.0)),
        _kv("raon1.activation.situ_linear_beta", GGUF_FLOAT32, struct.pack("<f", 25.0)),
        _kv("raon1.kda.gate_lower_bound", GGUF_FLOAT32, struct.pack("<f", -5.0)),
        _kv("raon1.attention.latent_norm_rms_eps", GGUF_FLOAT32, struct.pack("<f", 1e-6)),
        _kv("tokenizer.ggml.model", GGUF_STRING, _string("gpt2")),
        _kv("tokenizer.ggml.pre", GGUF_STRING, _string("raon")),
        _kv("tokenizer.ggml.tokens", GGUF_ARRAY, _string_array(tokens)),
        _kv("tokenizer.ggml.token_type", GGUF_ARRAY, _uint_array(types)),
        _kv("tokenizer.ggml.merges", GGUF_ARRAY, _string_array([])),
        _kv("tokenizer.ggml.bos_token_id", GGUF_UINT32, _uint(1)),
        _kv("tokenizer.ggml.eos_token_id", GGUF_UINT32, _uint(2)),
        _kv("tokenizer.chat_template", GGUF_STRING, _string(
            "{%- for message in messages %}{{ message.content }}{%- endfor %}"
        )),
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
    path = directory / name
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(body)
        handle.write(descriptors)
        handle.write(b"\0" * pad)
        handle.write(payload)
    return path
