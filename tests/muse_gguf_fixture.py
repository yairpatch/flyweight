"""Builds a tiny but complete muse-glimmer GGUF.

The published Muse Glimmer checkpoint is ~28B parameters, so this writes a
synthetic one at toy widths with the same tensor names and metadata keys,
covering the parts that differ from every architecture already supported:

  * the sliding-window pattern written as a scalar *period* rather than a
    per-layer boolean array, with the full-attention layer closing each cycle,
  * RoPE on the sliding-window layers and NoPE on the full-attention ones,
  * a per-channel sigmoid attention gate ahead of the output projection,
  * post-attention and post-feed-forward norms, which sit between each block
    and the residual join rather than after it,
  * the output multiplier and tanh logit softcap on the head.

The norm weights are written the way the conversion script leaves them, with
the `+1` already folded in, so a weight of 1.0 here is an identity scale.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
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


class MuseSpec:
    """Geometry of the synthetic model."""

    def __init__(
        self,
        *,
        hidden: int = 128,
        layers: int = 8,
        intermediate: int = 256,
        vocabulary: int = 96,
        heads: int = 8,
        kv_heads: int = 2,
        # A multiple of 32: the batched-prefill attention kernels require it,
        # and a narrower head silently routes the rows test onto the per-token
        # fallback instead, leaving the fused path untested.
        head_dim: int = 32,
        sliding_window: int = 16,
        window_period: int = 4,
        logit_scale: float = 0.196_116_13,
        logit_softcap: float = 20.0,
    ):
        self.hidden = hidden
        self.layers = layers
        self.intermediate = intermediate
        self.vocabulary = vocabulary
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.window_period = window_period
        self.logit_scale = logit_scale
        self.logit_softcap = logit_softcap

    def is_sliding(self, layer: int) -> bool:
        # The cycle ends on the full-attention layer: for period 4 that is
        # layers 3, 7, 11 ... and the other three slide.
        return layer % self.window_period + 1 < self.window_period


def build_muse_gguf(
    path: Path | str, spec: MuseSpec | None = None, seed: int = 17
) -> MuseSpec:
    """Write the fixture and return its geometry."""
    spec = spec or MuseSpec()
    rng = np.random.default_rng(seed)
    tensors: list[tuple[str, tuple[int, ...], np.ndarray]] = []

    def projection(name: str, inputs: int, outputs: int, scale: float = 0.25) -> None:
        # GGUF reports [input, output] while the byte stream is [output, input].
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

    query_width = spec.heads * spec.head_dim
    for layer in range(spec.layers):
        prefix = f"blk.{layer}."
        vector(prefix + "attn_norm.weight", spec.hidden, value=1.0)
        projection(prefix + "attn_q.weight", spec.hidden, query_width)
        projection(prefix + "attn_k.weight", spec.hidden, spec.kv_heads * spec.head_dim)
        projection(prefix + "attn_v.weight", spec.hidden, spec.kv_heads * spec.head_dim)
        projection(prefix + "attn_output.weight", query_width, spec.hidden)
        # The q_norm weights carry qk_scale_factor broadcast over the head; the
        # k_norm weights are ones. That is a conversion-time decision, not a
        # runtime one, so the fixture reproduces it.
        vector(prefix + "attn_q_norm.weight", spec.head_dim, value=3.87)
        vector(prefix + "attn_k_norm.weight", spec.head_dim, value=1.0)
        # One gate value per query channel, not per head.
        projection(prefix + "attn_gate.weight", spec.hidden, query_width)
        vector(prefix + "post_attention_norm.weight", spec.hidden, value=1.0)
        vector(prefix + "ffn_norm.weight", spec.hidden, value=1.0)
        projection(prefix + "ffn_gate.weight", spec.hidden, spec.intermediate)
        projection(prefix + "ffn_up.weight", spec.hidden, spec.intermediate)
        projection(prefix + "ffn_down.weight", spec.intermediate, spec.hidden)
        vector(prefix + "post_ffw_norm.weight", spec.hidden, value=1.0)

    # A byte-level vocabulary plus the control tokens the chat format uses, so
    # tokenization round-trips without needing a real merge table.
    control = ["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"]
    printable = [chr(index) for index in range(32, 32 + spec.vocabulary - len(control))]
    vocabulary = (control + printable)[: spec.vocabulary]
    token_types = [3] * len(control) + [1] * (len(vocabulary) - len(control))

    metadata = [
        _kv("general.architecture", GGUF_STRING, _string("muse-glimmer")),
        _kv("general.name", GGUF_STRING, _string("muse-glimmer-fixture")),
        _kv("muse-glimmer.block_count", GGUF_UINT32, struct.pack("<I", spec.layers)),
        _kv("muse-glimmer.embedding_length", GGUF_UINT32, struct.pack("<I", spec.hidden)),
        _kv("muse-glimmer.feed_forward_length", GGUF_UINT32, struct.pack("<I", spec.intermediate)),
        _kv("muse-glimmer.context_length", GGUF_UINT32, struct.pack("<I", 512)),
        _kv("muse-glimmer.attention.head_count", GGUF_UINT32, struct.pack("<I", spec.heads)),
        _kv("muse-glimmer.attention.head_count_kv", GGUF_UINT32, struct.pack("<I", spec.kv_heads)),
        _kv("muse-glimmer.attention.key_length", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("muse-glimmer.attention.value_length", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv(
            "muse-glimmer.attention.sliding_window",
            GGUF_UINT32,
            struct.pack("<I", spec.sliding_window),
        ),
        # The scalar form of the key: a period, not a per-layer array.
        _kv(
            "muse-glimmer.attention.sliding_window_pattern",
            GGUF_UINT32,
            struct.pack("<I", spec.window_period),
        ),
        _kv(
            "muse-glimmer.attention.layer_norm_rms_epsilon",
            GGUF_FLOAT32,
            struct.pack("<f", 1e-5),
        ),
        _kv("muse-glimmer.rope.freq_base", GGUF_FLOAT32, struct.pack("<f", 500_000.0)),
        _kv("muse-glimmer.logit_scale", GGUF_FLOAT32, struct.pack("<f", spec.logit_scale)),
        _kv(
            "muse-glimmer.final_logit_softcapping",
            GGUF_FLOAT32,
            struct.pack("<f", spec.logit_softcap),
        ),
        _kv("tokenizer.ggml.model", GGUF_STRING, _string("gpt2")),
        _kv("tokenizer.ggml.pre", GGUF_STRING, _string("llama4")),
        _kv("tokenizer.ggml.tokens", GGUF_ARRAY, _string_array(vocabulary)),
        _kv("tokenizer.ggml.token_type", GGUF_ARRAY, _uint_array(token_types)),
        _kv("tokenizer.ggml.merges", GGUF_ARRAY, _string_array([])),
        # <|eot|> closes an assistant turn; <|eom|> only closes one message
        # within a turn, so it is not a terminator.
        _kv("tokenizer.ggml.eos_token_id", GGUF_UINT32, struct.pack("<I", 3)),
        _kv("tokenizer.ggml.eot_token_id", GGUF_UINT32, struct.pack("<I", 3)),
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
