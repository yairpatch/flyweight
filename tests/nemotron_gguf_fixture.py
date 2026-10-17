"""Builds a tiny but complete nemotron_h_moe GGUF, plus a numpy forward over it.

The real Nemotron 3.5 Bolt checkpoints are tens of gigabytes, so this writes a
synthetic one with the same tensor names, metadata keys and layout conventions
at toy widths, f32 throughout. It covers what the architecture does differently
from the other families: Mamba-2 blocks whose depthwise causal convolution
covers only the x|B|C segment (z and dt bypass it, and the `ssm_dt` bias joins
inside the scan), GQA blocks with no RoPE and no per-head norms anywhere, and
sigmoid-routed relu² experts running from a `moe_latent_size` reduction with a
bias that affects selection only, plus an always-on unweighted shared expert.
The extreme per-layer heterogeneity the Puzzle checkpoints ship is exercised by
three MoE layers with different expert counts, widths and top-k, one of them
without latent projections at all.

`reference_greedy` is an independent float64 implementation of the same
forward, written from llama.cpp's src/models/nemotron-h.cpp,
src/models/mamba-base.cpp (`build_mamba2_layer`) and the f32
`ggml_ssm_conv` / `ggml_ssm_scan` kernels, so a test can pin the native
runtime's semantics rather than only its self-consistency.
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

CONTROL_TOKENS = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"]
# Type 4 (user-defined) in the real vocabulary and emitted by the template as
# text, so the tokenizer must still split them out whole.
USER_DEFINED_TOKENS = ["<think>", "</think>", "<tool_call>", "</tool_call>"]
# Multi-character entries: "12"/"123" reached through the merges below, so the
# llama3 digit grouping is observable, and three spaces / " a" in the GGUF
# byte spelling, which only the llama3 whitespace split produces whole.
EXTRA_TOKENS = ["12", "123", "\u0120\u0120\u0120", "\u0120a"]
MERGES = ["1 2", "12 3"]
# Printable ASCII first, so a prompt id is its character's offset from " ".
PLAIN_TOKENS = [chr(index) for index in range(32, 32 + 96)]
VOCABULARY = PLAIN_TOKENS + CONTROL_TOKENS + USER_DEFINED_TOKENS + EXTRA_TOKENS

# Mamba-2 channel splits, shared by writer and reference.
D_INNER = 128
D_STATE = 8
N_GROUP = 2
N_HEAD = 8
CONV_KERNEL = 4
HEAD_DIM = D_INNER // N_HEAD
CONV_CHANNELS = D_INNER + 2 * N_GROUP * D_STATE
D_IN_PROJ = 2 * D_INNER + 2 * N_GROUP * D_STATE + N_HEAD

ATTN_HEADS = 4
ATTN_KV_HEADS = 2
ATTN_HEAD_DIM = 32


class NemotronSpec:
    """Geometry of the synthetic model.

    The cadence runs mamba, attention, mamba, then three MoE layers at rows
    3, 5 and 7 whose expert count, width and top-k all disagree, the last one
    without latent projections so the full-width expert stack is covered too.
    """

    def __init__(
        self,
        *,
        hidden: int = 64,
        layers: int = 8,
        moe_latent: int = 32,
        shared_intermediate: int = 48,
    ):
        self.hidden = hidden
        self.layers = layers
        self.moe_latent = moe_latent
        self.shared_intermediate = shared_intermediate
        self.expert_ff = {3: 24, 5: 32, 7: 28}
        self.expert_count = {3: 10, 5: 12, 7: 8}
        self.expert_used = {3: 2, 5: 3, 7: 2}
        self.experts_scale = 2.5

    def is_attention(self, layer: int) -> bool:
        return layer in (1, 6)

    def is_mamba(self, layer: int) -> bool:
        return layer in (0, 2, 4)

    def is_moe(self, layer: int) -> bool:
        return layer in self.expert_count

    def has_latent(self, layer: int) -> bool:
        return self.is_moe(layer) and layer != 7


def build_nemotron_gguf(
    path: Path | str, spec: NemotronSpec | None = None, seed: int = 7
) -> tuple[NemotronSpec, dict[str, np.ndarray]]:
    """Write the fixture; return its geometry and the weights by name.

    Every array is returned in the [output][input] orientation the byte stream
    stores (GGUF reports the reverse), which is also what a matmul wants.
    """
    spec = spec or NemotronSpec()
    rng = np.random.default_rng(seed)
    tensors: list[tuple[str, tuple[int, ...], np.ndarray]] = []
    weights: dict[str, np.ndarray] = {}

    def add(name: str, shape: tuple[int, ...], data: np.ndarray) -> None:
        data = np.ascontiguousarray(data, dtype=np.float32)
        tensors.append((name, shape, data))
        weights[name] = data

    def projection(name: str, inputs: int, outputs: int, scale: float = 0.2) -> None:
        add(name, (inputs, outputs), rng.standard_normal((outputs, inputs)) * scale)

    def stacked(name: str, shape: tuple[int, ...]) -> None:
        # permuted: the GGUF dims are (in, out, experts), so the natural
        # (experts, out, in) write stores exactly that.
        data = rng.standard_normal((shape[2], shape[1], shape[0])) * 0.2
        add(name, shape, data)

    def vector(name: str, size: int, base: float = 1.0, scale: float = 0.1) -> None:
        add(name, (size,), base + rng.standard_normal(size) * scale)

    def relu2_init(name: str, inputs: int, outputs: int) -> None:
        # relu² carries no saturation: keep magnitudes small so the routed
        # sums stay near the linear regime.
        add(name, (inputs, outputs), rng.standard_normal((outputs, inputs)) * 0.08)

    projection("token_embd.weight", spec.hidden, len(VOCABULARY), scale=1.0)
    vector("output_norm.weight", spec.hidden)
    projection("output.weight", spec.hidden, len(VOCABULARY), scale=0.7)
    for layer in range(spec.layers):
        prefix = f"blk.{layer}."
        vector(prefix + "attn_norm.weight", spec.hidden)
        if spec.is_mamba(layer):
            projection(prefix + "ssm_in.weight", spec.hidden, D_IN_PROJ)
            # {taps, channels}: the taps of one channel are contiguous, the
            # GGUF dims the same (kernel, inner + 2*n_group*d_state).
            add(prefix + "ssm_conv1d.weight", (CONV_KERNEL, CONV_CHANNELS),
                rng.standard_normal((CONV_CHANNELS, CONV_KERNEL)) * 0.5)
            add(prefix + "ssm_conv1d.bias", (CONV_CHANNELS,),
                rng.standard_normal(CONV_CHANNELS) * 0.1)
            add(prefix + "ssm_dt.bias", (N_HEAD,),
                rng.standard_normal(N_HEAD) * 0.5 + 0.1)
            # ssm_a = -expm(A_log) in the conversion: negative downdates keep
            # the state decaying. ssm_d is the skip a plain D.
            add(prefix + "ssm_a", (N_HEAD,), -np.abs(
                rng.standard_normal(N_HEAD) * 0.5 + 0.5))
            add(prefix + "ssm_d", (N_HEAD,), rng.standard_normal(N_HEAD) * 0.1)
            # {group-major}: group 0 channels, then group 1, each chunk
            # d_inner/n_group wide. The GGUF dims (chunk, n_group) report the
            # reverse; the flat order is (group, chunk) per the norm reshape.
            chunk = D_INNER // N_GROUP
            add(prefix + "ssm_norm.weight", (chunk, N_GROUP),
                (1.0 + rng.standard_normal((N_GROUP, chunk)) * 0.1))
            projection(prefix + "ssm_out.weight", D_INNER, spec.hidden)
        elif spec.is_attention(layer):
            projection(prefix + "attn_q.weight", spec.hidden,
                       ATTN_HEADS * ATTN_HEAD_DIM)
            projection(prefix + "attn_k.weight", spec.hidden,
                       ATTN_KV_HEADS * ATTN_HEAD_DIM)
            projection(prefix + "attn_v.weight", spec.hidden,
                       ATTN_KV_HEADS * ATTN_HEAD_DIM)
            projection(prefix + "attn_output.weight",
                       ATTN_HEADS * ATTN_HEAD_DIM, spec.hidden)
        else:
            experts = spec.expert_count[layer]
            ff = spec.expert_ff[layer]
            relu2_init(prefix + "ffn_gate_inp.weight", spec.hidden, experts)
            add(prefix + "exp_probs_b.bias", (experts,),
                rng.standard_normal(experts) * 0.3)
            relu2_init(prefix + "ffn_up_shexp.weight", spec.hidden,
                       spec.shared_intermediate)
            relu2_init(prefix + "ffn_down_shexp.weight",
                       spec.shared_intermediate, spec.hidden)
            if spec.has_latent(layer):
                relu2_init(prefix + "ffn_latent_down.weight", spec.hidden,
                           spec.moe_latent)
                relu2_init(prefix + "ffn_latent_up.weight", spec.moe_latent,
                           spec.hidden)
                stacked(prefix + "ffn_up_exps.weight",
                        (spec.moe_latent, ff, experts))
                stacked(prefix + "ffn_down_exps.weight",
                        (ff, spec.moe_latent, experts))
            else:
                stacked(prefix + "ffn_up_exps.weight",
                        (spec.hidden, ff, experts))
                stacked(prefix + "ffn_down_exps.weight",
                        (ff, spec.hidden, experts))

    # Printable ASCII as single-character tokens, then the specials, so text
    # tokenizes with a two-entry merge table.
    vocabulary = VOCABULARY
    token_types = (
        [1] * len(PLAIN_TOKENS) + [3] * len(CONTROL_TOKENS)
        + [4] * len(USER_DEFINED_TOKENS) + [1] * len(EXTRA_TOKENS)
    )
    bos = vocabulary.index("<s>")
    eos = vocabulary.index("<|im_end|>")

    kv_heads = [
        ATTN_KV_HEADS if spec.is_attention(layer) else 0
        for layer in range(spec.layers)
    ]
    expert_counts = [spec.expert_count.get(layer, 0) for layer in range(spec.layers)]
    expert_ff = [spec.expert_ff.get(layer, 0) for layer in range(spec.layers)]
    expert_used = [spec.expert_used.get(layer, 0) for layer in range(spec.layers)]
    arch = "nemotron_h_moe"
    metadata = [
        _kv("general.architecture", GGUF_STRING, _string(arch)),
        _kv("general.name", GGUF_STRING, _string(arch + "-fixture")),
        _kv(arch + ".block_count", GGUF_UINT32, struct.pack("<I", spec.layers)),
        _kv(arch + ".context_length", GGUF_UINT32, struct.pack("<I", 512)),
        _kv(arch + ".embedding_length", GGUF_UINT32, struct.pack("<I", spec.hidden)),
        # A scalar maximum, as the released Bolt GGUF stores it: the per-layer
        # counts live only in the router widths.
        _kv(arch + ".expert_count", GGUF_UINT32,
            struct.pack("<I", max(expert_counts))),
        _kv(arch + ".expert_used_count", GGUF_ARRAY, _uint_array(expert_used)),
        _kv(arch + ".expert_feed_forward_length", GGUF_ARRAY, _uint_array(expert_ff)),
        _kv(arch + ".expert_shared_feed_forward_length", GGUF_UINT32,
            struct.pack("<I", spec.shared_intermediate)),
        _kv(arch + ".expert_shared_count", GGUF_UINT32, struct.pack("<I", 1)),
        _kv(arch + ".expert_gating_func", GGUF_UINT32, struct.pack("<I", 2)),
        _kv(arch + ".expert_weights_norm", GGUF_BOOL, b"\x01"),
        _kv(arch + ".expert_weights_scale", GGUF_FLOAT32,
            struct.pack("<f", spec.experts_scale)),
        _kv(arch + ".moe_latent_size", GGUF_UINT32,
            struct.pack("<I", spec.moe_latent)),
        _kv(arch + ".attention.head_count", GGUF_UINT32,
            struct.pack("<I", ATTN_HEADS)),
        _kv(arch + ".attention.head_count_kv", GGUF_ARRAY, _uint_array(kv_heads)),
        _kv(arch + ".attention.layer_norm_rms_epsilon", GGUF_FLOAT32,
            struct.pack("<f", 1e-5)),
        _kv(arch + ".ssm.conv_kernel", GGUF_UINT32,
            struct.pack("<I", CONV_KERNEL)),
        _kv(arch + ".ssm.inner_size", GGUF_UINT32, struct.pack("<I", D_INNER)),
        _kv(arch + ".ssm.state_size", GGUF_UINT32, struct.pack("<I", D_STATE)),
        _kv(arch + ".ssm.time_step_rank", GGUF_UINT32, struct.pack("<I", N_HEAD)),
        _kv(arch + ".ssm.group_count", GGUF_UINT32, struct.pack("<I", N_GROUP)),
        _kv("tokenizer.ggml.model", GGUF_STRING, _string("gpt2")),
        _kv("tokenizer.ggml.pre", GGUF_STRING, _string("pixtral")),
        _kv("tokenizer.ggml.tokens", GGUF_ARRAY, _string_array(vocabulary)),
        _kv("tokenizer.ggml.token_type", GGUF_ARRAY, _uint_array(token_types)),
        _kv("tokenizer.ggml.merges", GGUF_ARRAY, _string_array(MERGES)),
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


def reference_greedy(
    spec: NemotronSpec, weights: dict[str, np.ndarray], prompt: list[int], count: int
) -> list[int]:
    """Greedy continuation of `prompt` by a float64 forward pass."""
    w = {name: value.astype(np.float64) for name, value in weights.items()}
    eps = 1e-5

    def rms(x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight

    def silu(x):
        return x / (1 + np.exp(-x))

    def softplus(x):
        # matches ggml_compute_softplus_f32 including its overflow clamp.
        return np.where(x > 20.0, x, np.log1p(np.exp(x)))

    def relu2(x):
        return np.maximum(x, 0.0) ** 2

    # Ring and scan state per mamba layer, zero at position 0 like the
    # runtime's freshly prepared slots; attention caches grow per layer.
    conv_history = {
        layer: np.zeros((CONV_CHANNELS, CONV_KERNEL - 1))
        for layer in range(spec.layers) if spec.is_mamba(layer)
    }
    ssm_states = {
        layer: np.zeros((N_HEAD, HEAD_DIM, D_STATE))
        for layer in range(spec.layers) if spec.is_mamba(layer)
    }
    keys: dict[int, list[np.ndarray]] = {}
    values: dict[int, list[np.ndarray]] = {}

    def mamba(layer: int, h: np.ndarray) -> np.ndarray:
        p = f"blk.{layer}."
        proj = w[p + "ssm_in.weight"] @ h
        z, x_bc_raw, dt_raw = np.split(proj, [D_INNER, D_INNER + CONV_CHANNELS])
        # Depthwise causal conv over x|B|C alone; the ring holds the previous
        # kernel-1 raw values per channel, oldest first, and the last tap
        # multiplies the current value.
        taps = w[p + "ssm_conv1d.weight"]
        bias = w[p + "ssm_conv1d.bias"]
        window = np.concatenate([conv_history[layer], x_bc_raw[:, None]], axis=1)
        conv_history[layer] = window[:, 1:]
        x_bc = silu(np.sum(taps * window, axis=1) + bias)
        x = x_bc[:D_INNER].reshape(N_HEAD, HEAD_DIM)
        b = x_bc[D_INNER:D_INNER + N_GROUP * D_STATE].reshape(N_GROUP, D_STATE)
        c = x_bc[D_INNER + N_GROUP * D_STATE:].reshape(N_GROUP, D_STATE)
        # dt joins ssm_dt.bias before softplus, inside the scan.
        dt_f = softplus(dt_raw + w[p + "ssm_dt.bias"])
        y = np.zeros((N_HEAD, HEAD_DIM))
        for head in range(N_HEAD):
            group = head // (N_HEAD // N_GROUP)
            decay = np.exp(dt_f[head] * w[p + "ssm_a"][head])
            state = ssm_states[layer][head]
            for i1 in range(HEAD_DIM):
                x_dt = x[head, i1] * dt_f[head]
                next_state = state[i1] * decay + b[group] * x_dt
                state[i1] = next_state
                y[head, i1] = next_state @ c[group]
            y[head] += x[head] * w[p + "ssm_d"][head]
        y = silu(z.reshape(N_HEAD, HEAD_DIM)) * y
        # Grouped RMS: each n_group chunk gets its own norm with the stored
        # weight, whose flat order indexes the chunks by global channel.
        chunk = D_INNER // N_GROUP
        flat = y.reshape(-1)
        norm_weight = w[p + "ssm_norm.weight"].reshape(-1)
        normed = np.empty(flat.shape)
        for group in range(N_GROUP):
            piece = flat[group * chunk:(group + 1) * chunk]
            normed[group * chunk:(group + 1) * chunk] = piece / np.sqrt(
                np.mean(piece * piece) + eps
            ) * norm_weight[group * chunk:(group + 1) * chunk]
        return w[p + "ssm_out.weight"] @ normed

    def attention(layer: int, h: np.ndarray, position: int) -> np.ndarray:
        p = f"blk.{layer}."
        q = w[p + "attn_q.weight"] @ h
        k = w[p + "attn_k.weight"] @ h
        v = w[p + "attn_v.weight"] @ h
        keys.setdefault(layer, []).append(k)
        values.setdefault(layer, []).append(v)
        k_stack = np.stack(keys[layer]).reshape(-1, ATTN_KV_HEADS, ATTN_HEAD_DIM)
        v_stack = np.stack(values[layer]).reshape(-1, ATTN_KV_HEADS, ATTN_HEAD_DIM)
        group = ATTN_HEADS // ATTN_KV_HEADS
        out = np.empty((ATTN_HEADS, ATTN_HEAD_DIM))
        for head in range(ATTN_HEADS):
            kv = head // group
            scores = k_stack[:, kv] @ q[head * ATTN_HEAD_DIM:(
                head + 1) * ATTN_HEAD_DIM] / np.sqrt(ATTN_HEAD_DIM)
            scores = np.exp(scores - scores.max())
            out[head] = (scores / scores.sum()) @ v_stack[:, kv]
        return w[p + "attn_output.weight"] @ out.reshape(-1)

    def moe(layer: int, h: np.ndarray) -> np.ndarray:
        p = f"blk.{layer}."
        used = spec.expert_used[layer]
        logits = w[p + "ffn_gate_inp.weight"] @ h
        scores = 1.0 / (1.0 + np.exp(-logits))
        ranking = scores + w[p + "exp_probs_b.bias"]
        # Strict-greater argmax per rank, ties to the lowest expert: exactly
        # the serial selection route_topk_sigmoid_bias reproduces.
        winning = []
        pool = ranking.tolist()
        for _ in range(used):
            winner = pool.index(max(pool))
            winning.append(winner)
            pool[winner] = -np.inf
        total = scores[winning].sum()
        routing = scores[winning] / total if total > 0 else np.ones(used) / used
        routing = routing * spec.experts_scale
        # The experts run in the latent space when the layer carries the
        # ffn_latent_down/up bridge, else at full width.
        expert_in = spec.moe_latent if spec.has_latent(layer) else spec.hidden
        stream = w[p + "ffn_latent_down.weight"] @ h if spec.has_latent(layer) else h
        latent = np.zeros(expert_in)
        for expert, weight in zip(winning, routing):
            latent += weight * (w[p + "ffn_down_exps.weight"][expert] @ relu2(
                w[p + "ffn_up_exps.weight"][expert] @ stream))
        if spec.has_latent(layer):
            latent = w[p + "ffn_latent_up.weight"] @ latent
        shared = w[p + "ffn_down_shexp.weight"] @ relu2(
            w[p + "ffn_up_shexp.weight"] @ h)
        return latent + shared

    def step(token: int, position: int) -> int:
        x = w["token_embd.weight"][token].copy()
        for layer in range(spec.layers):
            p = f"blk.{layer}."
            h = rms(x, w[p + "attn_norm.weight"])
            if spec.is_mamba(layer):
                y = mamba(layer, h)
            elif spec.is_attention(layer):
                y = attention(layer, h, position)
            else:
                y = moe(layer, h)
            x = x + y
        logits = w["output.weight"] @ rms(x, w["output_norm.weight"])
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
