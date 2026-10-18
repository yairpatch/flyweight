"""A tiny DFlash + LiLiCorr drafter GGUF beside the Nemotron H fixture, and a
float64 reference draft over it.

The released drafter for Nemotron 3.5 Bolt is llama.cpp's `dflash` arch with
the LiLiCorr re-ranker (general.name Li5_SWA). This writes the same tensor
names, metadata keys and dtypes at toy widths: NVFP4 `fc` and trunk FFN with
their out-of-line `.scale`, bf16 attention and conv projections, f32 conv
bases, sinks, norms and LiLiCorr heads. A small sliding window (16) puts the
context-window edge and the ring wrap inside a short prompt.

`reference_draft` is written from llama.cpp cd0a5b6 (src/models/dflash.cpp,
the draft-dflash path of common/speculative.cpp); plans/nemotron-dflash.md
derives it. Two things are arranged so the comparison measures the drafter
and not its storage:

* The bf16 and f32 matrices the runtime uploads as Q8_0 are generated
  Q8_0-exact: per 32 values an int8 code under a power-of-two scale, with one
  code at +-127, so the conversion reproduces them bit for bit.
* NVFP4 matmuls take Q8 activations in the runtime, as in llama.cpp; the
  reference quantizes those activations the same way.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from tests.nemotron_gguf_fixture import (
    ALIGNMENT, GGML_F32, GGML_NVFP4, GGUF_ARRAY, GGUF_BOOL, GGUF_FLOAT32,
    GGUF_STRING, GGUF_UINT32, NVFP4_TENSOR_SCALE, VOCABULARY, _kv, _pack_nvfp4,
    _string, _string_array, _uint_array,
)

GGML_BF16 = 30
_BLOCK_BYTES = {GGML_F32: (1, 4), GGML_BF16: (1, 2), GGML_NVFP4: (64, 36)}


class DflashSpec:
    def __init__(self, hidden: int, *, target_layers=(1, 3, 6), window: int = 16,
                 layers: int = 2, mask_token: int | None = None):
        self.hidden = hidden
        self.target_layers = list(target_layers)
        self.features = len(self.target_layers) * hidden
        self.layers = layers
        self.heads, self.kv_heads, self.head_dim = 4, 2, hidden // 4
        self.ffn = 2 * hidden
        self.block = 8
        self.conv_group = 16
        self.window = window
        self.mask_token = mask_token if mask_token is not None else VOCABULARY.index("<unk>")
        self.width = hidden          # LiLiCorr width (NVFP4 inputs need 64s)
        self.corrector_heads = 2
        self.corrector_layers = 2
        self.topk = 8
        self.mlp = 2 * hidden
        self.factor = hidden
        self.epsilon = 1e-6
        self.freq_base = 10000.0
        self.vector_eps = 1e-4
        self.logit_scale = 8.0

    @property
    def slots(self) -> int:
        return self.block - 1

    @property
    def nodes(self) -> int:
        return self.slots * self.topk


def build_dflash_gguf(path: Path | str, spec: DflashSpec, seed: int = 11):
    """Write the drafter; return the weights by name in [out][in] orientation
    (vectors and the conv bases as stored)."""
    rng = np.random.default_rng(seed)
    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
    weights: dict[str, np.ndarray] = {}

    def put(name, shape, kind, raw, value):
        tensors.append((name, shape, kind, raw))
        weights[name] = value

    def exact(outputs: int, inputs: int, magnitude: float) -> np.ndarray:
        # Q8_0-exact: per 32 inputs one power-of-two scale and int8 codes,
        # one of them at +-127 so amax/127 is the scale itself.
        codes = rng.integers(-127, 128, size=(outputs, inputs // 32, 32))
        codes[..., 0] = np.where(rng.random((outputs, inputs // 32)) < 0.5, -127, 127)
        exponent = np.round(np.log2(magnitude / 127.0)) + rng.integers(-1, 2, size=(outputs, inputs // 32))
        values = codes * np.exp2(exponent)[..., None]
        return values.reshape(outputs, inputs).astype(np.float32)

    def matrix(name, inputs, outputs, kind, magnitude=0.5):
        value = exact(outputs, inputs, magnitude)
        if kind == GGML_BF16:
            raw = (value.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        else:
            raw = value.tobytes()
        put(name, (inputs, outputs), kind, raw, value)

    def nvfp4(name, inputs, outputs, magnitude=0.15):
        data = (rng.standard_normal((outputs, inputs)) * magnitude).astype(np.float32)
        raw, decoded = _pack_nvfp4(data, rng)
        put(name, (inputs, outputs), GGML_NVFP4, raw,
            decoded.reshape(outputs, inputs) * NVFP4_TENSOR_SCALE)
        scale = np.array([NVFP4_TENSOR_SCALE], dtype=np.float32)
        tensors.append((name[:-len(".weight")] + ".scale", (1,), GGML_F32, scale.tobytes()))

    def f32(name, shape, value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        put(name, shape, GGML_F32, value.tobytes(), value)

    def vector(name, size, base=1.0, scale=0.1):
        f32(name, (size,), base + rng.standard_normal(size) * scale)

    H, s = spec.hidden, spec
    groups = H // s.conv_group
    nvfp4("fc.weight", s.features, H, magnitude=0.08)
    vector("enc.output_norm.weight", H)
    for layer in range(s.layers):
        p = f"blk.{layer}."
        vector(p + "attn_norm.weight", H)
        matrix(p + "attn_conv_proj.weight", H, 4 * groups, GGML_BF16, magnitude=0.05)
        # Tap 0 near identity and tap 1 near zero, as the released bases are.
        base = np.zeros((2, 2, H), np.float32)   # [side][tap][channel]
        base[:, 0] = 1.0 + rng.standard_normal((2, H)) * 0.05
        base[:, 1] = rng.standard_normal((2, H)) * 0.2
        f32(p + "attn_conv_base", (H, 2, 2), base)
        matrix(p + "attn_q.weight", H, s.heads * s.head_dim, GGML_BF16)
        matrix(p + "attn_k.weight", H, s.kv_heads * s.head_dim, GGML_BF16)
        matrix(p + "attn_v.weight", H, s.kv_heads * s.head_dim, GGML_BF16)
        vector(p + "attn_q_norm.weight", s.head_dim)
        vector(p + "attn_k_norm.weight", s.head_dim)
        f32(p + "attn_sinks.weight", (s.heads,), rng.standard_normal(s.heads))
        matrix(p + "attn_output.weight", s.heads * s.head_dim, H, GGML_BF16, magnitude=0.3)
        vector(p + "ffn_norm.weight", H)
        matrix(p + "ffn_conv_proj.weight", H, 4 * groups, GGML_BF16, magnitude=0.05)
        base = np.zeros((2, 2, H), np.float32)
        base[:, 0] = 1.0 + rng.standard_normal((2, H)) * 0.05
        base[:, 1] = rng.standard_normal((2, H)) * 0.2
        f32(p + "ffn_conv_base", (H, 2, 2), base)
        nvfp4(p + "ffn_gate.weight", H, s.ffn)
        nvfp4(p + "ffn_up.weight", H, s.ffn)
        nvfp4(p + "ffn_down.weight", s.ffn, H)
    vector("output_norm.weight", H)
    W = s.width
    for name in ("token_proj", "pass_hidden_proj", "context_proj"):
        matrix(f"lilicorr.{name}.weight", H, W, GGML_BF16, magnitude=0.3)
        vector(f"lilicorr.{name}.bias", W, base=0.0, scale=0.05)
    for i in range(s.corrector_layers):
        p = f"blk.{s.layers + i}."
        vector(p + "attn_norm.weight", W)
        matrix(f"lilicorr.layers.{i}.attn.in_proj.weight", W, 3 * W, GGML_F32, magnitude=0.4)
        vector(f"lilicorr.layers.{i}.attn.in_proj.bias", 3 * W, base=0.0, scale=0.05)
        matrix(p + "attn_output.weight", W, W, GGML_BF16, magnitude=0.3)
        vector(p + "attn_output.bias", W, base=0.0, scale=0.05)
        vector(p + "ffn_norm.weight", W)
        matrix(p + "ffn_up.weight", W, s.mlp, GGML_BF16, magnitude=0.3)
        vector(p + "ffn_up.bias", s.mlp, base=0.0, scale=0.05)
        matrix(p + "ffn_down.weight", s.mlp, W, GGML_BF16, magnitude=0.3)
        vector(p + "ffn_down.bias", W, base=0.0, scale=0.05)
    feature = f"blk.{s.layers + s.corrector_layers}."
    vector(feature + "ffn_norm.weight", 5)
    vector(feature + "ffn_norm.bias", 5, base=0.0)
    f32(feature + "ffn_up.weight", (5, W), rng.standard_normal((W, 5)) * 0.5)
    vector(feature + "ffn_up.bias", W, base=0.0, scale=0.05)
    matrix(feature + "ffn_down.weight", W, W, GGML_F32, magnitude=0.3)
    vector(feature + "ffn_down.bias", W, base=0.0, scale=0.05)
    f32("lilicorr.slot_embedding.weight", (W, s.slots), rng.standard_normal((s.slots, W)) * 0.3)
    f32("lilicorr.rank_embedding.weight", (W, s.topk), rng.standard_normal((s.topk, W)) * 0.3)
    f32("lilicorr.attention_bias.bias", (s.nodes, s.nodes, s.corrector_heads),
        rng.standard_normal((s.corrector_heads, s.nodes, s.nodes)) * 0.5)
    vector("lilicorr.output_norm.weight", W)
    vector("lilicorr.anchor_norm.weight", W)
    matrix("lilicorr.factor_input_proj.weight", 3 * W, s.factor, GGML_F32, magnitude=0.3)
    vector("lilicorr.factor_input_proj.bias", s.factor, base=0.0, scale=0.05)
    for name in ("out_head", "in_head"):
        matrix(f"lilicorr.{name}.weight", s.factor, s.factor, GGML_F32, magnitude=0.3)
        vector(f"lilicorr.{name}.bias", s.factor, base=0.0, scale=0.05)
    matrix("lilicorr.anchor_out_head.weight", W, s.factor, GGML_F32, magnitude=0.3)
    vector("lilicorr.anchor_out_head.bias", s.factor, base=0.0, scale=0.05)

    arch = "dflash"
    bools = struct.pack("<IQ", GGUF_BOOL, s.layers) + b"\x01" * s.layers
    metadata = [
        _kv("general.architecture", GGUF_STRING, _string(arch)),
        _kv("general.name", GGUF_STRING, _string("Li5_SWA-fixture")),
        _kv(arch + ".block_count", GGUF_UINT32, struct.pack("<I", s.layers)),
        _kv(arch + ".context_length", GGUF_UINT32, struct.pack("<I", 512)),
        _kv(arch + ".embedding_length", GGUF_UINT32, struct.pack("<I", H)),
        _kv(arch + ".feed_forward_length", GGUF_UINT32, struct.pack("<I", s.ffn)),
        _kv(arch + ".attention.head_count", GGUF_UINT32, struct.pack("<I", s.heads)),
        _kv(arch + ".attention.head_count_kv", GGUF_UINT32, struct.pack("<I", s.kv_heads)),
        _kv(arch + ".attention.key_length", GGUF_UINT32, struct.pack("<I", s.head_dim)),
        _kv(arch + ".attention.value_length", GGUF_UINT32, struct.pack("<I", s.head_dim)),
        _kv(arch + ".rope.freq_base", GGUF_FLOAT32, struct.pack("<f", s.freq_base)),
        _kv(arch + ".attention.layer_norm_rms_epsilon", GGUF_FLOAT32,
            struct.pack("<f", s.epsilon)),
        _kv(arch + ".block_size", GGUF_UINT32, struct.pack("<I", s.block)),
        _kv(arch + ".conv_kernel_size", GGUF_UINT32, struct.pack("<I", 2)),
        _kv(arch + ".conv_group_size", GGUF_UINT32, struct.pack("<I", s.conv_group)),
        _kv(arch + ".target_layers", GGUF_ARRAY, _uint_array(s.target_layers)),
        _kv(arch + ".attention.sliding_window", GGUF_UINT32, struct.pack("<I", s.window)),
        _kv(arch + ".attention.sliding_window_pattern", GGUF_ARRAY, bools),
        _kv(arch + ".attention.causal", GGUF_BOOL, b"\x00"),
        _kv("lilicorr.hidden_size", GGUF_UINT32, struct.pack("<I", W)),
        _kv("lilicorr.candidate_topk", GGUF_UINT32, struct.pack("<I", s.topk)),
        _kv("lilicorr.num_heads", GGUF_UINT32, struct.pack("<I", s.corrector_heads)),
        _kv("lilicorr.num_layers", GGUF_UINT32, struct.pack("<I", s.corrector_layers)),
        _kv("lilicorr.factor_dim", GGUF_UINT32, struct.pack("<I", s.factor)),
        _kv("lilicorr.mlp_hidden_size", GGUF_UINT32, struct.pack("<I", s.mlp)),
        _kv("lilicorr.vector_eps", GGUF_FLOAT32, struct.pack("<f", s.vector_eps)),
        _kv("lilicorr.logit_scale", GGUF_FLOAT32, struct.pack("<f", s.logit_scale)),
        _kv("tokenizer.ggml.model", GGUF_STRING, _string("gpt2")),
        _kv("tokenizer.ggml.pre", GGUF_STRING, _string("pixtral")),
        _kv("tokenizer.ggml.tokens", GGUF_ARRAY, _string_array(VOCABULARY)),
        _kv("tokenizer.ggml.merges", GGUF_ARRAY, _string_array([])),
        _kv("tokenizer.ggml.mask_token_id", GGUF_UINT32, struct.pack("<I", s.mask_token)),
        _kv("general.alignment", GGUF_UINT32, struct.pack("<I", ALIGNMENT)),
    ]
    infos = bytearray()
    payloads = bytearray()
    for name, shape, kind, raw in tensors:
        infos += _string(name)
        infos += struct.pack("<I", len(shape))
        infos += b"".join(struct.pack("<Q", dim) for dim in shape)
        infos += struct.pack("<IQ", kind, len(payloads))
        block_elements, block_bytes = _BLOCK_BYTES[kind]
        assert len(raw) == int(np.prod(shape)) // block_elements * block_bytes, name
        payloads += raw
        payloads += b"\0" * ((-len(payloads)) % ALIGNMENT)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    body = bytearray(header + b"".join(metadata) + bytes(infos))
    body += b"\0" * ((-len(body)) % ALIGNMENT)
    Path(path).write_bytes(bytes(body) + bytes(payloads))
    return weights


def _rms(x, weight, epsilon):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * weight


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _rope(x, positions, base):
    # x [rows, heads, d]; NEOX pairs (j, j + d/2), theta_j = p * base^(-2j/d).
    d = x.shape[-1]
    j = np.arange(d // 2)
    theta = positions[:, None, None] * base ** (-2.0 * j / d)
    a, b = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([a * np.cos(theta) - b * np.sin(theta),
                           a * np.sin(theta) + b * np.cos(theta)], axis=-1)


def _q8_activations(x):
    """The runtime's quantize_q8_blocks_rows, dequantized: per 32 values
    scale = amax/127 in f32, codes round(x/scale), the scale stored as f16."""
    x32 = np.asarray(x, np.float32)
    blocks = x32.reshape(*x32.shape[:-1], -1, 32)
    amax = np.abs(blocks).max(axis=-1, keepdims=True)
    scale = np.where(amax > 0, amax / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    codes = np.clip(np.rint(blocks / scale), -127, 127)
    return (codes * scale.astype(np.float16).astype(np.float64)).reshape(x32.shape)


class ReferenceDrafter:
    """float64 drafter state: the context ring as llama.cpp keeps it (every
    injected row at its position) and the injected rows' fc+norm outputs."""

    def __init__(self, spec: DflashSpec, weights, target_weights):
        self.s, self.w, self.t = spec, weights, target_weights
        self.k: dict[int, list[np.ndarray]] = {}
        self.v: dict[int, list[np.ndarray]] = {}
        self.g: dict[int, np.ndarray] = {}

    def _nvfp4(self, name, x):
        # NVFP4 against Q8 activations, the runtime's kernels and llama.cpp's.
        return _q8_activations(np.asarray(x, np.float64)) @ self.w[name].T

    def inject(self, features, positions):
        s, w = self.s, self.w
        g = _rms(self._nvfp4("fc.weight", features), w["enc.output_norm.weight"], s.epsilon)
        for row, position in enumerate(positions):
            self.g[int(position)] = g[row]
        for layer in range(s.layers):
            p = f"blk.{layer}."
            k = (g @ w[p + "attn_k.weight"].T).reshape(-1, s.kv_heads, s.head_dim)
            v = (g @ w[p + "attn_v.weight"].T).reshape(-1, s.kv_heads, s.head_dim)
            k = _rope(_rms(k, w[p + "attn_k_norm.weight"], s.epsilon),
                      np.asarray(positions, np.float64), s.freq_base)
            for row, position in enumerate(positions):
                self.k.setdefault(int(position), [None] * s.layers)[layer] = k[row]
                self.v.setdefault(int(position), [None] * s.layers)[layer] = v[row]

    def _conv(self, x, coeff, base, side):
        s = self.s
        groups = s.hidden // s.conv_group
        g = np.arange(s.hidden) // s.conv_group
        w0 = coeff[:, g + 2 * groups * side] + base[side, 0][None, :]
        w1 = coeff[:, g + groups + 2 * groups * side] + base[side, 1][None, :]
        previous = np.zeros_like(x)
        previous[1:] = x[:-1]
        return w0 * x + w1 * previous

    def draft(self, id_last: int, pos0: int, n_max: int):
        s, w, t = self.s, self.w, self.t
        rows = s.block
        positions = pos0 + np.arange(rows)
        tokens = [id_last] + [s.mask_token] * (rows - 1)
        h = t["token_embd.weight"][tokens].astype(np.float64)
        context = sorted(p for p in self.k if p < pos0)
        for layer in range(s.layers):
            p = f"blk.{layer}."
            a = _rms(h, w[p + "attn_norm.weight"], s.epsilon)
            ca = a @ w[p + "attn_conv_proj.weight"].T
            a2 = self._conv(a, ca, w[p + "attn_conv_base"], 0)
            q = (a2 @ w[p + "attn_q.weight"].T).reshape(rows, s.heads, s.head_dim)
            k = (a2 @ w[p + "attn_k.weight"].T).reshape(rows, s.kv_heads, s.head_dim)
            v = (a2 @ w[p + "attn_v.weight"].T).reshape(rows, s.kv_heads, s.head_dim)
            q = _rope(_rms(q, w[p + "attn_q_norm.weight"], s.epsilon), positions.astype(np.float64), s.freq_base)
            k = _rope(_rms(k, w[p + "attn_k_norm.weight"], s.epsilon), positions.astype(np.float64), s.freq_base)
            sinks = w[p + "attn_sinks.weight"]
            out = np.zeros((rows, s.heads, s.head_dim))
            for i in range(rows):
                visible = [c for c in context if positions[i] - c < s.window]
                keys = np.stack([self.k[c][layer] for c in visible] + list(k))
                values = np.stack([self.v[c][layer] for c in visible] + list(v))
                for head in range(s.heads):
                    kv = head // (s.heads // s.kv_heads)
                    scores = keys[:, kv] @ q[i, head] / np.sqrt(s.head_dim)
                    m = max(scores.max(), sinks[head])
                    e = np.exp(scores - m)
                    out[i, head] = e @ values[:, kv] / (e.sum() + np.exp(sinks[head] - m))
            attn = out.reshape(rows, -1) @ w[p + "attn_output.weight"].T
            r = self._conv(attn, ca, w[p + "attn_conv_base"], 1) + h
            f = _rms(r, w[p + "ffn_norm.weight"], s.epsilon)
            cf = f @ w[p + "ffn_conv_proj.weight"].T
            f2 = self._conv(f, cf, w[p + "ffn_conv_base"], 0)
            gate = self._nvfp4(p + "ffn_gate.weight", f2)
            up = self._nvfp4(p + "ffn_up.weight", f2)
            y = self._nvfp4(p + "ffn_down.weight", _silu(gate) * up)
            h = self._conv(y, cf, w[p + "ffn_conv_base"], 1) + r
        z = _rms(h, w["output_norm.weight"], s.epsilon)
        logits = z[1:] @ t["output.weight"].T
        candidates, scores = self._lilicorr(logits, z, self.g[pos0 - 1])
        previous, drafts = 0, []
        for i in range(n_max):
            row = scores[i][0 if i == 0 else previous]
            previous = int(np.argmax(row))
            drafts.append(int(candidates[i][previous]))
        return {"logits": logits, "candidates": candidates, "scores": scores, "drafts": drafts}

    def _lilicorr(self, logits, z, anchor):
        s, w, t = self.s, self.w, self.t
        S, K = s.slots, s.topk
        L = s.layers
        candidates = np.zeros((S, K), np.int64)
        feature = np.zeros((S, K, 5))
        for slot in range(S):
            row = logits[slot]
            order = np.lexsort((np.arange(len(row)), -row))[:K]
            candidates[slot] = order
            m = row.max()
            lp = row[order] - (m + np.log(np.exp(row - m).sum()))
            feature[slot, :, 0] = lp
            feature[slot, :, 1] = np.exp(lp)
            feature[slot, :, 2] = lp - lp[0]
            feature[slot, :, 3] = np.arange(K) / (K - 1)
            feature[slot, :, 4] = np.arange(K) == 0
        fp = f"blk.{L + s.corrector_layers}."
        f = feature.reshape(-1, 5)
        mean = f.mean(-1, keepdims=True)
        variance = ((f - mean) ** 2).mean(-1, keepdims=True)
        f = (f - mean) / np.sqrt(variance + 1e-5) * w[fp + "ffn_norm.weight"] + w[fp + "ffn_norm.bias"]
        f = _silu(f @ w[fp + "ffn_up.weight"].T + w[fp + "ffn_up.bias"])
        fs = f @ w[fp + "ffn_down.weight"].T + w[fp + "ffn_down.bias"]

        def proj(name, x):
            return x @ w[f"lilicorr.{name}.weight"].T + w[f"lilicorr.{name}.bias"]
        tok = proj("token_proj", t["token_embd.weight"][candidates.reshape(-1)].astype(np.float64))
        passed = np.repeat(proj("pass_hidden_proj", z[1:]), K, axis=0)
        position = (w["lilicorr.slot_embedding.weight"][:, None, :] +
                    w["lilicorr.rank_embedding.weight"][None, :, :]).reshape(-1, s.width)
        x = ((tok + passed) + fs) + position
        anc = proj("context_proj", anchor[None, :])[0]
        heads = s.corrector_heads
        hd = s.width // heads
        bias = w["lilicorr.attention_bias.bias"]
        for i in range(s.corrector_layers):
            p = f"blk.{L + i}."
            y = _rms(x, w[p + "attn_norm.weight"], s.epsilon)
            qkv = y @ w[f"lilicorr.layers.{i}.attn.in_proj.weight"].T + w[f"lilicorr.layers.{i}.attn.in_proj.bias"]
            q = qkv[:, : s.width].reshape(-1, heads, hd)
            k = qkv[:, s.width: 2 * s.width].reshape(-1, heads, hd)
            v = qkv[:, 2 * s.width:].reshape(-1, heads, hd)
            o = np.zeros_like(q)
            for head in range(heads):
                a = q[:, head] @ k[:, head].T / np.sqrt(hd) + bias[head]
                a = np.exp(a - a.max(-1, keepdims=True))
                a /= a.sum(-1, keepdims=True)
                o[:, head] = a @ v[:, head]
            x = x + o.reshape(-1, s.width) @ w[p + "attn_output.weight"].T + w[p + "attn_output.bias"]
            y = _rms(x, w[p + "ffn_norm.weight"], s.epsilon)
            hidden = _silu(y @ w[p + "ffn_up.weight"].T + w[p + "ffn_up.bias"])
            x = x + hidden @ w[p + "ffn_down.weight"].T + w[p + "ffn_down.bias"]
        x = _rms(x, w["lilicorr.output_norm.weight"], s.epsilon)
        anc = _rms(anc, w["lilicorr.anchor_norm.weight"], s.epsilon)
        phi = _silu(np.concatenate([x, np.broadcast_to(anc, x.shape), x * anc], -1)
                    @ w["lilicorr.factor_input_proj.weight"].T + w["lilicorr.factor_input_proj.bias"])

        def l2n(v):
            return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), s.vector_eps)
        out = l2n(phi @ w["lilicorr.out_head.weight"].T + w["lilicorr.out_head.bias"]).reshape(S, K, -1)
        inn = l2n(phi @ w["lilicorr.in_head.weight"].T + w["lilicorr.in_head.bias"]).reshape(S, K, -1)
        anchor_factor = l2n(anc @ w["lilicorr.anchor_out_head.weight"].T + w["lilicorr.anchor_out_head.bias"])
        scores = np.zeros((S, K, K))
        scores[0, 0] = s.logit_scale * (inn[0] @ anchor_factor)
        for slot in range(1, S):
            scores[slot] = s.logit_scale * (out[slot - 1] @ inn[slot].T)
        return candidates, scores
