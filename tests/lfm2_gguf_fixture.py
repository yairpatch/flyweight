"""Builds a tiny but complete lfm2moe GGUF, plus a numpy forward over it.

The real LFM2-MoE checkpoints are tens of gigabytes, so this writes a synthetic
one with the same tensor names and layout conventions at toy widths, f32
throughout. It covers what the architecture does differently from the other
families on the qwen plan: gated short-convolution blocks between the GQA
blocks (marked by a zero in the per-layer KV-head array), per-head QK norm
with no output gate, a leading dense block, sigmoid routing with a
selection-only bias and no shared expert, `token_embd_norm` as the final norm
and a head tied to the embedding table.

`reference_greedy` is an independent float64 implementation of the same
forward, written from llama.cpp's src/models/lfm2.cpp, so a test can pin the
native runtime's semantics rather than only its self-consistency.
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

CONTROL_TOKENS = ["<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|tool_call_start|>",
                  "<|tool_call_end|>"]
# Type 4 (user-defined) in the real vocabulary, and emitted by the template
# as text, so they must still be split out whole.
USER_DEFINED_TOKENS = ["<think>", "</think>"]
# Multi-character entries so the llama3 digit grouping is observable.
EXTRA_TOKENS = ["123", "12"]


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


class Lfm2Spec:
    """Geometry of the synthetic model.

    The default cadence is the real one's at a sixth of the depth: two conv
    blocks, then attention, repeated.
    """

    def __init__(
        self,
        *,
        hidden: int = 128,
        layers: int = 6,
        dense_intermediate: int = 256,
        expert_intermediate: int = 64,
        heads: int = 4,
        kv_heads: int = 2,
        experts: int = 8,
        experts_used: int = 2,
        conv_kernel: int = 3,
        leading_dense: int = 1,
        # Far below the real 1e7, so a dozen positions rotate enough for the
        # rope pairing and frequencies to decide the continuation.
        rope_theta: float = 100.0,
        architecture: str = "lfm2moe",
    ):
        # Dense LFM2 is the same block stack with every feed-forward dense.
        self.architecture = architecture
        if architecture == "lfm2":
            leading_dense = layers
        self.hidden = hidden
        self.layers = layers
        self.dense_intermediate = dense_intermediate
        self.expert_intermediate = expert_intermediate
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = hidden // heads
        self.experts = experts
        self.experts_used = experts_used
        self.conv_kernel = conv_kernel
        self.leading_dense = leading_dense
        self.rope_theta = rope_theta
        self.vocabulary = 96 + len(CONTROL_TOKENS) + len(USER_DEFINED_TOKENS) + len(EXTRA_TOKENS)

    def is_attention(self, layer: int) -> bool:
        return layer % 3 == 2

    def is_moe(self, layer: int) -> bool:
        return layer >= self.leading_dense


def build_lfm2_gguf(
    path: Path | str, spec: Lfm2Spec | None = None, seed: int = 5
) -> tuple[Lfm2Spec, dict[str, np.ndarray]]:
    """Write the fixture; return its geometry and the weights by name.

    Every array is returned in the [output][input] orientation the byte stream
    stores (GGUF reports the reverse), which is also what a matmul wants.
    """
    spec = spec or Lfm2Spec()
    rng = np.random.default_rng(seed)
    tensors: list[tuple[str, tuple[int, ...], np.ndarray]] = []
    weights: dict[str, np.ndarray] = {}

    def add(name: str, shape: tuple[int, ...], data: np.ndarray) -> None:
        data = np.ascontiguousarray(data, dtype=np.float32)
        tensors.append((name, shape, data))
        weights[name] = data

    def projection(name: str, inputs: int, outputs: int, scale: float = 0.2) -> None:
        add(name, (inputs, outputs), rng.standard_normal((outputs, inputs)) * scale)

    def stacked(name: str, inputs: int, outputs: int, scale: float = 0.2) -> None:
        add(name, (inputs, outputs, spec.experts),
            rng.standard_normal((spec.experts, outputs, inputs)) * scale)

    def vector(name: str, size: int, base: float = 1.0, scale: float = 0.1) -> None:
        add(name, (size,), base + rng.standard_normal(size) * scale)

    projection("token_embd.weight", spec.hidden, spec.vocabulary, scale=1.0)
    vector("token_embd_norm.weight", spec.hidden)
    for layer in range(spec.layers):
        prefix = f"blk.{layer}."
        vector(prefix + "attn_norm.weight", spec.hidden)
        if spec.is_attention(layer):
            projection(prefix + "attn_q.weight", spec.hidden, spec.heads * spec.head_dim)
            projection(prefix + "attn_k.weight", spec.hidden, spec.kv_heads * spec.head_dim)
            projection(prefix + "attn_v.weight", spec.hidden, spec.kv_heads * spec.head_dim)
            projection(prefix + "attn_output.weight", spec.heads * spec.head_dim, spec.hidden)
            vector(prefix + "attn_q_norm.weight", spec.head_dim)
            vector(prefix + "attn_k_norm.weight", spec.head_dim)
        else:
            projection(prefix + "shortconv.in_proj.weight", spec.hidden, 3 * spec.hidden)
            projection(prefix + "shortconv.out_proj.weight", spec.hidden, spec.hidden)
            # {l_cache, hidden}: the taps of one channel are contiguous.
            add(prefix + "shortconv.conv.weight", (spec.conv_kernel, spec.hidden),
                rng.standard_normal((spec.hidden, spec.conv_kernel)) * 0.5)
        vector(prefix + "ffn_norm.weight", spec.hidden)
        if spec.is_moe(layer):
            projection(prefix + "ffn_gate_inp.weight", spec.hidden, spec.experts, scale=0.5)
            vector(prefix + "exp_probs_b.bias", spec.experts, base=0.0, scale=0.3)
            stacked(prefix + "ffn_gate_exps.weight", spec.hidden, spec.expert_intermediate)
            stacked(prefix + "ffn_up_exps.weight", spec.hidden, spec.expert_intermediate)
            stacked(prefix + "ffn_down_exps.weight", spec.expert_intermediate, spec.hidden)
        else:
            projection(prefix + "ffn_gate.weight", spec.hidden, spec.dense_intermediate)
            projection(prefix + "ffn_up.weight", spec.hidden, spec.dense_intermediate)
            projection(prefix + "ffn_down.weight", spec.dense_intermediate, spec.hidden)

    # Printable ASCII as single-character tokens, then the specials, so text
    # tokenizes without a merge table.
    plain = [chr(index) for index in range(32, 32 + 96)]
    vocabulary = plain + CONTROL_TOKENS + USER_DEFINED_TOKENS + EXTRA_TOKENS
    token_types = (
        [1] * len(plain) + [3] * len(CONTROL_TOKENS)
        + [4] * len(USER_DEFINED_TOKENS) + [1] * len(EXTRA_TOKENS)
    )
    assert len(vocabulary) == spec.vocabulary
    bos = vocabulary.index("<|startoftext|>")
    eos = vocabulary.index("<|im_end|>")

    kv_heads = [spec.kv_heads if spec.is_attention(layer) else 0 for layer in range(spec.layers)]
    arch = spec.architecture
    metadata = [
        _kv("general.architecture", GGUF_STRING, _string(arch)),
        _kv("general.name", GGUF_STRING, _string(arch + "-fixture")),
        _kv(arch + ".block_count", GGUF_UINT32, struct.pack("<I", spec.layers)),
        _kv(arch + ".context_length", GGUF_UINT32, struct.pack("<I", 512)),
        _kv(arch + ".embedding_length", GGUF_UINT32, struct.pack("<I", spec.hidden)),
        _kv(arch + ".feed_forward_length", GGUF_UINT32, struct.pack("<I", spec.dense_intermediate)),
        _kv(arch + ".attention.head_count", GGUF_UINT32, struct.pack("<I", spec.heads)),
        _kv(arch + ".attention.head_count_kv", GGUF_ARRAY, _uint_array(kv_heads)),
        _kv(arch + ".rope.freq_base", GGUF_FLOAT32, struct.pack("<f", spec.rope_theta)),
        _kv(arch + ".attention.layer_norm_rms_epsilon", GGUF_FLOAT32, struct.pack("<f", 1e-5)),
        _kv(arch + ".vocab_size", GGUF_UINT32, struct.pack("<I", spec.vocabulary)),
        _kv(arch + ".shortconv.l_cache", GGUF_UINT32, struct.pack("<I", spec.conv_kernel)),
    ]
    if arch == "lfm2moe":
        metadata += [
            _kv(arch + ".expert_count", GGUF_UINT32, struct.pack("<I", spec.experts)),
            _kv(arch + ".expert_used_count", GGUF_UINT32, struct.pack("<I", spec.experts_used)),
            _kv(arch + ".expert_feed_forward_length", GGUF_UINT32,
                struct.pack("<I", spec.expert_intermediate)),
            _kv(arch + ".leading_dense_block_count", GGUF_UINT32,
                struct.pack("<I", spec.leading_dense)),
            _kv(arch + ".expert_gating_func", GGUF_UINT32, struct.pack("<I", 2)),
        ]
    metadata += [
        _kv("tokenizer.ggml.model", GGUF_STRING, _string("gpt2")),
        _kv("tokenizer.ggml.pre", GGUF_STRING, _string("lfm2")),
        _kv("tokenizer.ggml.tokens", GGUF_ARRAY, _string_array(vocabulary)),
        _kv("tokenizer.ggml.token_type", GGUF_ARRAY, _uint_array(token_types)),
        _kv("tokenizer.ggml.merges", GGUF_ARRAY, _string_array([])),
        _kv("tokenizer.ggml.bos_token_id", GGUF_UINT32, struct.pack("<I", bos)),
        _kv("tokenizer.ggml.eos_token_id", GGUF_UINT32, struct.pack("<I", eos)),
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
        raw = data.tobytes()
        assert len(raw) == int(np.prod(shape)) * 4, name
        payloads += raw
        payloads += b"\0" * ((-len(payloads)) % ALIGNMENT)

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    body = bytearray(header + b"".join(metadata) + bytes(infos))
    body += b"\0" * ((-len(body)) % ALIGNMENT)
    Path(path).write_bytes(bytes(body) + bytes(payloads))
    return spec, weights


def reference_greedy(
    spec: Lfm2Spec, weights: dict[str, np.ndarray], prompt: list[int], count: int
) -> list[int]:
    """Greedy continuation of `prompt` by a float64 forward pass."""
    w = {name: value.astype(np.float64) for name, value in weights.items()}
    eps = 1e-5
    hd = spec.head_dim
    half = hd // 2
    inv_freq = spec.rope_theta ** (-np.arange(half, dtype=np.float64) * 2 / hd)
    group = spec.heads // spec.kv_heads

    def rms(x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight

    def rope(x, position):
        # NEOX: channel i pairs with channel i + head_dim / 2.
        angle = position * inv_freq
        cos, sin = np.cos(angle), np.sin(angle)
        a, b = x[..., :half], x[..., half:]
        return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)

    def silu(x):
        return x / (1 + np.exp(-x))

    keys: dict[int, list[np.ndarray]] = {}
    values: dict[int, list[np.ndarray]] = {}
    history = {
        layer: np.zeros((spec.hidden, spec.conv_kernel - 1))
        for layer in range(spec.layers) if not spec.is_attention(layer)
    }

    def step(token: int, position: int) -> int:
        x = w["token_embd.weight"][token].copy()
        for layer in range(spec.layers):
            p = f"blk.{layer}."
            h = rms(x, w[p + "attn_norm.weight"])
            if spec.is_attention(layer):
                q = (w[p + "attn_q.weight"] @ h).reshape(spec.heads, hd)
                k = (w[p + "attn_k.weight"] @ h).reshape(spec.kv_heads, hd)
                v = (w[p + "attn_v.weight"] @ h).reshape(spec.kv_heads, hd)
                q = rope(rms(q, w[p + "attn_q_norm.weight"]), position)
                k = rope(rms(k, w[p + "attn_k_norm.weight"]), position)
                keys.setdefault(layer, []).append(k)
                values.setdefault(layer, []).append(v)
                cached_k, cached_v = np.stack(keys[layer]), np.stack(values[layer])
                out = np.empty((spec.heads, hd))
                for head in range(spec.heads):
                    scores = cached_k[:, head // group] @ q[head] / np.sqrt(hd)
                    scores = np.exp(scores - scores.max())
                    out[head] = (scores / scores.sum()) @ cached_v[:, head // group]
                y = w[p + "attn_output.weight"] @ out.reshape(-1)
            else:
                bcx = w[p + "shortconv.in_proj.weight"] @ h
                b, c, xs = np.split(bcx, 3)
                bx = b * xs
                taps = w[p + "shortconv.conv.weight"]  # [hidden][kernel], oldest first
                window = np.concatenate([history[layer], bx[:, None]], axis=1)
                history[layer] = window[:, 1:]
                y = w[p + "shortconv.out_proj.weight"] @ (c * np.sum(taps * window, axis=1))
            x = x + y
            h = rms(x, w[p + "ffn_norm.weight"])
            if spec.is_moe(layer):
                probabilities = 1 / (1 + np.exp(-(w[p + "ffn_gate_inp.weight"] @ h)))
                chosen = np.argsort(-(probabilities + w[p + "exp_probs_b.bias"]),
                                    kind="stable")[: spec.experts_used]
                routing = probabilities[chosen] / probabilities[chosen].sum()
                f = np.zeros(spec.hidden)
                for expert, weight in zip(chosen, routing):
                    gate = w[p + "ffn_gate_exps.weight"][expert] @ h
                    up = w[p + "ffn_up_exps.weight"][expert] @ h
                    f += weight * (w[p + "ffn_down_exps.weight"][expert] @ (silu(gate) * up))
            else:
                f = w[p + "ffn_down.weight"] @ (
                    silu(w[p + "ffn_gate.weight"] @ h) * (w[p + "ffn_up.weight"] @ h))
            x = x + f
        logits = w["token_embd.weight"] @ rms(x, w["token_embd_norm.weight"])
        return int(np.argmax(logits))

    produced: list[int] = []
    token = None
    for position, prompt_token in enumerate(prompt):
        token = step(prompt_token, position)
    for offset in range(count):
        produced.append(token)
        if offset + 1 < count:
            token = step(token, len(prompt) + offset)
    return produced
