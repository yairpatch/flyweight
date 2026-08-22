"""Builds a tiny but complete dense qwen35 GGUF.

Every published dense Qwen3.5/3.6 checkpoint is quantized with IQ codebook
formats the runtime does not implement, so there is no small real model that
exercises the dense feed-forward path. This writes a synthetic one -- same
tensor names, same layout conventions, f32 weights -- so the dense plan, the
dense SwiGLU and the host-side feed-forward spill can be tested for real.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
GGUF_STRING = 8
GGUF_UINT64 = 10
GGML_F32 = 0
GGML_Q8_0 = 8

ALIGNMENT = 32


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _quantize_q8_0(data: np.ndarray) -> bytes:
    """Pack a row-major matrix as GGML Q8_0.

    Block layout is 32 values per block: an fp16 scale followed by 32 int8
    weights, 34 bytes total. Blocks run along the contiguous dimension, which
    for these tensors is the input width.
    """
    rows, columns = data.shape
    assert columns % 32 == 0, "Q8_0 needs a multiple of 32 along the row"
    blocks = data.reshape(rows, columns // 32, 32).astype(np.float32)
    absmax = np.abs(blocks).max(axis=2)
    scale = (absmax / 127.0).astype(np.float32)
    inverse = np.where(scale > 0.0, 1.0 / np.where(scale > 0.0, scale, 1.0), 0.0)
    quantized = np.rint(blocks * inverse[:, :, None]).clip(-127, 127).astype(np.int8)

    out = bytearray()
    scale_f16 = scale.astype(np.float16)
    for row in range(rows):
        for block in range(columns // 32):
            out += scale_f16[row, block].tobytes()
            out += quantized[row, block].tobytes()
    return bytes(out)


def _kv(key: str, kind: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", kind) + payload


class DenseQwenSpec:
    """Geometry of the synthetic model.

    The DeltaNet widths follow the real checkpoints: the fused qkv projection is
    ``2 * key_dim + value_dim`` wide, value heads outnumber key heads, and the
    attention blocks carry a per-head gate alongside the query.
    """

    def __init__(
        self,
        *,
        hidden: int = 256,
        layers: int = 8,
        intermediate: int = 512,
        vocabulary: int = 64,
        heads: int = 8,
        kv_heads: int = 4,
        head_dim: int = 32,
        value_heads: int = 8,
        ssm_head_dim: int = 32,
        key_heads: int = 4,
        conv_kernel: int = 4,
        attention_every: int = 4,
        experts: int = 0,
        experts_used: int = 0,
        expert_intermediate: int = 64,
    ):
        self.hidden = hidden
        self.layers = layers
        self.intermediate = intermediate
        self.vocabulary = vocabulary
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.value_heads = value_heads
        self.ssm_head_dim = ssm_head_dim
        self.key_heads = key_heads
        self.conv_kernel = conv_kernel
        self.attention_every = attention_every
        # experts > 0 swaps every block's dense SwiGLU for the MoE layout:
        # router, sigmoid-gated shared expert, and stacked routed experts.
        self.experts = experts
        self.experts_used = experts_used
        self.expert_intermediate = expert_intermediate

    @property
    def value_dim(self) -> int:
        return self.value_heads * self.ssm_head_dim

    @property
    def key_dim(self) -> int:
        return self.key_heads * self.ssm_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    def is_attention(self, layer: int) -> bool:
        # Match the real interleaving: every Nth block is full attention.
        return (layer + 1) % self.attention_every == 0


def build_dense_qwen35_gguf(
    path: Path | str,
    spec: DenseQwenSpec | None = None,
    seed: int = 7,
    *,
    mute_mixer: bool = False,
    quantize: str | None = None,
    tied_lm_head: bool = False,
    mtp: bool = False,
    mtp_head_norm: bool = True,
) -> DenseQwenSpec:
    """Write the fixture.

    ``mute_mixer`` zeroes the attention and DeltaNet output projections so each
    block contributes only its feed-forward. That isolates the dense SwiGLU from
    the recurrence and the KV cache, which is what makes a mismatch attributable.
    """
    spec = spec or DenseQwenSpec()
    rng = np.random.default_rng(seed)
    tensors: list[tuple[str, tuple[int, ...], np.ndarray]] = []
    muted = {"attn_output.weight", "ssm_out.weight"} if mute_mixer else set()

    def projection(name: str, inputs: int, outputs: int, scale: float = 0.25) -> None:
        # GGUF reports [input, output] while the byte stream is [output, input],
        # so the payload is the logical matrix in output-major order.
        data = (rng.standard_normal((outputs, inputs)) * scale).astype(np.float32)
        if any(name.endswith(suffix) for suffix in muted):
            data = np.zeros_like(data)
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
    if not tied_lm_head:
        projection("output.weight", spec.hidden, spec.vocabulary)

    for layer in range(spec.layers):
        prefix = f"blk.{layer}."
        vector(prefix + "attn_norm.weight", spec.hidden, value=1.0)
        if spec.is_attention(layer):
            # The query projection carries a per-head gate after each head's
            # query, hence the factor of two.
            projection(prefix + "attn_q.weight", spec.hidden, spec.heads * spec.head_dim * 2)
            projection(prefix + "attn_k.weight", spec.hidden, spec.kv_heads * spec.head_dim)
            projection(prefix + "attn_v.weight", spec.hidden, spec.kv_heads * spec.head_dim)
            projection(prefix + "attn_output.weight", spec.heads * spec.head_dim, spec.hidden)
            vector(prefix + "attn_q_norm.weight", spec.head_dim, value=1.0)
            vector(prefix + "attn_k_norm.weight", spec.head_dim, value=1.0)
        else:
            projection(prefix + "attn_qkv.weight", spec.hidden, spec.conv_dim)
            projection(prefix + "attn_gate.weight", spec.hidden, spec.value_dim)
            projection(prefix + "ssm_out.weight", spec.value_dim, spec.hidden)
            projection(prefix + "ssm_alpha.weight", spec.hidden, spec.value_heads)
            projection(prefix + "ssm_beta.weight", spec.hidden, spec.value_heads)
            # GGML stores the tap dimension contiguously, so the payload is
            # [channel][tap] and the reported shape is [tap, channel].
            conv = (rng.standard_normal((spec.conv_dim, spec.conv_kernel)) * 0.3).astype(np.float32)
            tensors.append((prefix + "ssm_conv1d.weight", (spec.conv_kernel, spec.conv_dim), conv))
            vector(prefix + "ssm_dt.bias", spec.value_heads)
            # ssm_a ships already transformed to -exp(A_log), so it is negative.
            tensors.append((
                prefix + "ssm_a",
                (spec.value_heads,),
                -np.exp(rng.standard_normal(spec.value_heads) * 0.2).astype(np.float32),
            ))
            vector(prefix + "ssm_norm.weight", spec.ssm_head_dim, value=1.0)
        vector(prefix + "post_attention_norm.weight", spec.hidden, value=1.0)
        if spec.experts:
            # The Qwen3.5 MoE feed-forward: softmax top-k router, a
            # sigmoid-gated shared expert, and one stacked tensor per routed
            # role. Stacked payloads are [expert][output][input]; GGUF reports
            # the reversed shape.
            projection(prefix + "ffn_gate_inp.weight", spec.hidden, spec.experts)
            projection(prefix + "ffn_gate_shexp.weight", spec.hidden, spec.expert_intermediate)
            projection(prefix + "ffn_up_shexp.weight", spec.hidden, spec.expert_intermediate)
            projection(prefix + "ffn_down_shexp.weight", spec.expert_intermediate, spec.hidden)
            vector(prefix + "ffn_gate_inp_shexp.weight", spec.hidden)
            for name, inputs, outputs in (
                ("ffn_gate_exps.weight", spec.hidden, spec.expert_intermediate),
                ("ffn_up_exps.weight", spec.hidden, spec.expert_intermediate),
                ("ffn_down_exps.weight", spec.expert_intermediate, spec.hidden),
            ):
                data = (
                    rng.standard_normal((spec.experts, outputs, inputs)) * 0.25
                ).astype(np.float32)
                tensors.append((prefix + name, (inputs, outputs, spec.experts), data))
        else:
            projection(prefix + "ffn_gate.weight", spec.hidden, spec.intermediate)
            projection(prefix + "ffn_up.weight", spec.hidden, spec.intermediate)
            projection(prefix + "ffn_down.weight", spec.intermediate, spec.hidden)

    if mtp:
        prefix = f"blk.{spec.layers}."
        vector(prefix + "attn_norm.weight", spec.hidden, value=1.0)
        projection(prefix + "attn_q.weight", spec.hidden, spec.heads * spec.head_dim * 2)
        projection(prefix + "attn_k.weight", spec.hidden, spec.kv_heads * spec.head_dim)
        projection(prefix + "attn_v.weight", spec.hidden, spec.kv_heads * spec.head_dim)
        projection(prefix + "attn_output.weight", spec.heads * spec.head_dim, spec.hidden)
        vector(prefix + "attn_q_norm.weight", spec.head_dim, value=1.0)
        vector(prefix + "attn_k_norm.weight", spec.head_dim, value=1.0)
        vector(prefix + "post_attention_norm.weight", spec.hidden, value=1.0)
        projection(prefix + "ffn_gate.weight", spec.hidden, spec.intermediate)
        projection(prefix + "ffn_up.weight", spec.hidden, spec.intermediate)
        projection(prefix + "ffn_down.weight", spec.intermediate, spec.hidden)
        projection(prefix + "nextn.eh_proj.weight", 2 * spec.hidden, spec.hidden)
        vector(prefix + "nextn.enorm.weight", spec.hidden, value=1.0)
        vector(prefix + "nextn.hnorm.weight", spec.hidden, value=1.0)
        if mtp_head_norm:
            vector(prefix + "nextn.shared_head_norm.weight", spec.hidden, value=1.0)

    metadata = [
        _kv("general.architecture", GGUF_STRING, _string("qwen35")),
        _kv("general.name", GGUF_STRING, _string("dense-qwen35-fixture")),
        _kv("qwen35.block_count", GGUF_UINT32, struct.pack("<I", spec.layers)),
        _kv("qwen35.embedding_length", GGUF_UINT32, struct.pack("<I", spec.hidden)),
        _kv("qwen35.feed_forward_length", GGUF_UINT32, struct.pack("<I", spec.intermediate)),
        _kv("qwen35.context_length", GGUF_UINT32, struct.pack("<I", 512)),
        _kv("qwen35.attention.head_count", GGUF_UINT32, struct.pack("<I", spec.heads)),
        _kv("qwen35.attention.head_count_kv", GGUF_UINT32, struct.pack("<I", spec.kv_heads)),
        _kv("qwen35.attention.key_length", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("qwen35.rope.dimension_count", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("qwen35.rope.freq_base", GGUF_FLOAT32, struct.pack("<f", 10_000_000.0)),
        _kv("qwen35.attention.layer_norm_rms_epsilon", GGUF_FLOAT32, struct.pack("<f", 1e-6)),
        _kv("qwen35.full_attention_interval", GGUF_UINT32, struct.pack("<I", spec.attention_every)),
        _kv("qwen35.ssm.conv_kernel", GGUF_UINT32, struct.pack("<I", spec.conv_kernel)),
        _kv("qwen35.ssm.inner_size", GGUF_UINT32, struct.pack("<I", spec.value_dim)),
        _kv("qwen35.ssm.time_step_rank", GGUF_UINT32, struct.pack("<I", spec.value_heads)),
        _kv("tokenizer.ggml.vocab_size", GGUF_UINT32, struct.pack("<I", spec.vocabulary)),
        _kv("general.alignment", GGUF_UINT32, struct.pack("<I", ALIGNMENT)),
    ]
    if spec.experts:
        metadata += [
            _kv("qwen35.expert_count", GGUF_UINT32, struct.pack("<I", spec.experts)),
            _kv("qwen35.expert_used_count", GGUF_UINT32, struct.pack("<I", spec.experts_used)),
            _kv(
                "qwen35.expert_feed_forward_length",
                GGUF_UINT32,
                struct.pack("<I", spec.expert_intermediate),
            ),
        ]

    infos = bytearray()
    payloads = bytearray()
    for name, shape, data in tensors:
        offset = len(payloads)
        # Quantize only the wide 2D projections. Norms and other vectors stay
        # f32 exactly as a real checkpoint leaves them.
        # The stacked expert tensors are 3-D [expert][output][input]; their
        # contiguous rows are shape[0] long exactly like a 2-D projection's,
        # so the same per-row packing applies.
        use_q8 = (
            quantize == "q8_0"
            and len(shape) in (2, 3)
            and shape[0] % 32 == 0
            and name.endswith(".weight")
            and "norm" not in name
        )
        infos += _string(name)
        infos += struct.pack("<I", len(shape))
        infos += b"".join(struct.pack("<Q", dim) for dim in shape)
        infos += struct.pack("<IQ", GGML_Q8_0 if use_q8 else GGML_F32, offset)
        if use_q8:
            raw = _quantize_q8_0(np.ascontiguousarray(
                data, dtype=np.float32).reshape(-1, shape[0]))
            assert len(raw) == int(np.prod(shape)) // 32 * 34, name
        else:
            raw = np.ascontiguousarray(data, dtype=np.float32).tobytes()
            assert len(raw) == int(np.prod(shape)) * 4, name
        payloads += raw
        padding = (-len(payloads)) % ALIGNMENT
        payloads += b"\0" * padding

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    body = bytearray(header + b"".join(metadata) + bytes(infos))
    body += b"\0" * ((-len(body)) % ALIGNMENT)
    Path(path).write_bytes(bytes(body) + bytes(payloads))
    return spec
