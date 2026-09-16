"""Builds a tiny but complete qwen4exp (Qwen3.8-Flash-Next) GGUF.

The only published qwen4exp checkpoint is 72.5 GB of IQ1_S, so every runtime
path is developed and pinned against this synthetic model instead: same tensor
names, same metadata keys, same layout conventions as the unsloth conversion
(llama.cpp PR #27739), f32 weights unless asked to quantize.

qwen4exp on top of the qwen35 hybrid layout (see dense_gguf_fixture.py):

  - NO attn_norm / post_attention_norm / output_norm. Every block boundary is
    a gated-residual (hyper-connection) set: hc_{attn,ffn}_{norm,down,up,
    inject} per layer, output_hc_{norm,down,up} at the head. Norm weights are
    written BAKED (the conversion adds +1 to the zero-centred HF weights), so
    value=1.0 here means "untrained identity" exactly like the other fixtures.
  - PLE: per_layer_token_embd table plus ple_{conv1d,key,value,norm_conv,
    norm_key,norm_query} on the blocks named by ple.layers.
  - QSA indexer projections on every full-attention block (dense fallback
    ignores them; they must still load).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

try:
    from tests.dense_gguf_fixture import (
        ALIGNMENT,
        GGML_F32,
        GGUF_FLOAT32,
        GGUF_STRING,
        GGUF_UINT32,
        GGUF_UINT64,
        _kv,
        _string,
    )
except ImportError:  # direct execution from the tests directory
    from dense_gguf_fixture import (
        ALIGNMENT,
        GGML_F32,
        GGUF_FLOAT32,
        GGUF_STRING,
        GGUF_UINT32,
        GGUF_UINT64,
        _kv,
        _string,
    )

GGUF_ARRAY = 9
GGML_Q2_0 = 42


def pack_q2_0(matrix: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Pack rows (the last axis) as GGML Q2_0 and return what they decode to.

    Q2_0 (ggml type 42) is an f16 scale d = max|x| over 64 values, then 16
    bytes of two-bit codes with element j at bits 2*(j%4) of byte j/4. Code q
    stands for (q-1)*d, so the levels are {-d, 0, d, 2d}; the quantizer is
    round(x/d) clamped to [-1, 2], as in ggml's quantize_row_q2_0_ref.
    """
    rows = np.ascontiguousarray(matrix, dtype=np.float32).reshape(-1, matrix.shape[-1])
    assert rows.shape[1] % 64 == 0, rows.shape
    blocks = rows.reshape(rows.shape[0], -1, 64)
    d = np.abs(blocks).max(axis=-1, keepdims=True).astype(np.float16)
    d32 = d.astype(np.float32)
    inverse = np.where(d32 > 0, 1.0 / np.where(d32 > 0, d32, 1.0), 0.0)
    codes = np.clip(np.rint(blocks * inverse).astype(np.int32) + 1, 0, 3)
    packed = (
        codes[..., 0::4] | (codes[..., 1::4] << 2)
        | (codes[..., 2::4] << 4) | (codes[..., 3::4] << 6)
    ).astype(np.uint8)
    raw = np.concatenate(
        [d.view(np.uint8).reshape(*packed.shape[:2], 2), packed], axis=-1
    ).tobytes()
    decoded = (d32 * (codes - 1).astype(np.float32)).reshape(matrix.shape)
    return raw, decoded


GGML_IQ4_NL = 20
IQ4_NL_VALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)


def pack_iq4_nl(matrix: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Pack rows as GGML IQ4_NL: f16 scale then 16 bytes of nibbles per 32
    values, byte j holding element j low and element j+16 high, each nibble
    indexing the 16-entry codebook. The scale is simply max|x|/113 with a
    nearest-codebook search; ggml's quantizer refines the scale, but any scale
    yields a valid block, and what matters here is that the runtime decodes
    the bytes it is given."""
    rows = np.ascontiguousarray(matrix, dtype=np.float32).reshape(-1, matrix.shape[-1])
    assert rows.shape[1] % 32 == 0, rows.shape
    blocks = rows.reshape(rows.shape[0], -1, 32)
    d = (np.abs(blocks).max(axis=-1, keepdims=True) / 113.0).astype(np.float16)
    d32 = d.astype(np.float32)
    scaled = np.where(d32 > 0, blocks / np.where(d32 > 0, d32, 1.0), 0.0)
    codes = np.abs(scaled[..., None] - IQ4_NL_VALUES).argmin(axis=-1).astype(np.uint8)
    packed = codes[..., :16] | (codes[..., 16:] << 4)
    raw = np.concatenate(
        [d.view(np.uint8).reshape(*packed.shape[:2], 2), packed], axis=-1
    ).tobytes()
    decoded = (d32 * IQ4_NL_VALUES[codes]).reshape(matrix.shape)
    return raw, decoded


GGML_BF16 = 30


def pack_bf16(matrix: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Round to bf16 (truncation, which is what the runtime's decoder inverts
    exactly) and return the rounded values."""
    bits = np.ascontiguousarray(matrix, dtype=np.float32).view(np.uint32) >> 16
    decoded = (bits.astype(np.uint32) << 16).view(np.float32)
    return bits.astype("<u2").tobytes(), decoded


GGML_IQ2_S = 22
_IQ2S_GRID: np.ndarray | None = None


def _iq2s_grid() -> np.ndarray:
    """The runtime's own 1024x8 IQ2_S grid, parsed from its header."""
    global _IQ2S_GRID
    if _IQ2S_GRID is None:
        import re
        header = (Path(__file__).resolve().parents[1] / "native" / "src"
                  / "qwen_iq_tables.h").read_text()
        body = header.split("kIq2sGrid[1024] = {", 1)[1].split("};", 1)[0]
        words = np.array([int(w) for w in re.findall(r"(\d+)ull", body)], dtype=np.uint64)
        assert words.size == 1024, words.size
        _IQ2S_GRID = np.stack(
            [((words >> np.uint64(8 * e)) & np.uint64(0xFF)).astype(np.float32)
             for e in range(8)], axis=1)
    return _IQ2S_GRID


def pack_iq2_s(matrix: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Pack rows as GGML IQ2_S: per 256 values d(f16), qs[32] grid indices
    (low 8 bits), signs[32] (bit e negates value e of the octet), qh[8]
    (two high index bits per octet), scales[8] (a nibble per 16 values).
    Value = d * (0.5 + scale) * 0.25 * grid[e] * sign. A nearest-octet
    search over the grid; ggml's quantizer is smarter about the scale, but
    any well-formed block serves, since the twin holds the decoded values."""
    grid = _iq2s_grid()
    rows = np.ascontiguousarray(matrix, dtype=np.float32).reshape(-1, matrix.shape[-1])
    assert rows.shape[1] % 256 == 0, rows.shape
    blocks = rows.reshape(-1, 256)
    out = bytearray()
    decoded = np.zeros_like(blocks)
    for b, block in enumerate(blocks):
        amax = float(np.abs(block).max())
        d = np.float16(amax / (43.0 * 15.5 * 0.25)) if amax > 0 else np.float16(0)
        d32 = float(d)
        qs = np.zeros(32, np.uint8); signs = np.zeros(32, np.uint8)
        qh = np.zeros(8, np.uint8); scales = np.zeros(8, np.uint8)
        for group in range(16):
            values = block[group * 16:(group + 1) * 16]
            gmax = float(np.abs(values).max())
            scale = 0 if d32 == 0 else int(np.clip(round(gmax / (d32 * 0.25 * 43.0) - 0.5), 0, 15))
            scales[group >> 1] |= scale << (4 * (group & 1))
            db = d32 * (0.5 + scale) * 0.25
            for half in range(2):
                index = group * 2 + half
                octet = values[half * 8:(half + 1) * 8]
                target = np.abs(octet) / db if db > 0 else np.zeros(8, np.float32)
                entry = int(np.argmin(((grid - target) ** 2).sum(axis=1)))
                qs[index] = entry & 0xFF
                qh[index >> 2] |= (entry >> 8) << (2 * (index & 3))
                negative = octet < 0
                signs[index] = int(sum(1 << e for e in range(8) if negative[e]))
                decoded[b, group * 16 + half * 8:group * 16 + half * 8 + 8] = (
                    db * grid[entry] * np.where(negative, -1.0, 1.0))
        out += np.array([d], dtype=np.float16).tobytes() + qs.tobytes() + signs.tobytes() + qh.tobytes() + scales.tobytes()
    return bytes(out), decoded.reshape(matrix.shape)


_PACKERS = {GGML_Q2_0: pack_q2_0, GGML_IQ4_NL: pack_iq4_nl, GGML_BF16: pack_bf16,
            GGML_IQ2_S: pack_iq2_s}


def _u32_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_UINT32, len(values)) + b"".join(
        struct.pack("<I", int(v)) for v in values
    )


def _u64_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_UINT64, len(values)) + b"".join(
        struct.pack("<Q", int(v)) for v in values
    )


def _primes_after(start: int, count: int) -> list[int]:
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        return all(n % d for d in range(3, int(n**0.5) + 1, 2))

    primes, n = [], start
    while len(primes) < count:
        n += 1
        while not is_prime(n):
            n += 1
        primes.append(n)
    return primes


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _hf_layer_multipliers(vocab: int, ngram_size: int, seed: int = 1234,
                          ple_layer_index: int = 0) -> list[int]:
    """The exact derivation transformers' Qwen4ExpTextNGramEmbedding uses, so
    the fixture's metadata equals the reference module's buffers and a parity
    test can run both from the same file."""
    multiplier_max = ((1 << 63) - 1) // max(vocab, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + 10007 * ple_layer_index
    return [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64)
             % half_bound) + 1
        for index in range(ngram_size)
    ]


class Qwen4ExpSpec:
    """Geometry of the synthetic model.

    Same proportions as the release checkpoint, shrunk: 3-of-4 DeltaNet blocks,
    hc_count residual streams with a low-rank mixer, one PLE block, MoE with a
    sigmoid-gated shared expert on every block.
    """

    def __init__(
        self,
        *,
        hidden: int = 64,
        layers: int = 8,
        vocabulary: int = 96,
        heads: int = 4,
        kv_heads: int = 2,
        head_dim: int = 32,
        rotary_dim: int = 16,
        value_heads: int = 4,
        key_heads: int = 2,
        ssm_head_dim: int = 16,
        conv_kernel: int = 4,
        attention_every: int = 4,
        hc_count: int = 4,
        hc_low_rank: int = 16,
        experts: int = 8,
        experts_used: int = 2,
        expert_intermediate: int = 32,
        ple_layers: tuple[int, ...] = (1,),
        ngram_size: int = 3,
        heads_per_ngram: int = 2,
        ple_head_dim: int = 16,
        ple_vocab_base: int = 96,
        ple_conv_kernel: int = 4,
        ple_eos_token_id: int = 5,
        indexer_heads: int = 2,
        indexer_key_dim: int = 16,
        indexer_top_k: int = 2048,
    ):
        self.hidden = hidden
        self.layers = layers
        self.vocabulary = vocabulary
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.value_heads = value_heads
        self.key_heads = key_heads
        self.ssm_head_dim = ssm_head_dim
        self.conv_kernel = conv_kernel
        self.attention_every = attention_every
        self.hc_count = hc_count
        self.hc_low_rank = hc_low_rank
        self.experts = experts
        self.experts_used = experts_used
        self.expert_intermediate = expert_intermediate
        self.ple_layers = tuple(ple_layers)
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ple_head_dim = ple_head_dim
        self.ple_conv_kernel = ple_conv_kernel
        self.ple_eos_token_id = ple_eos_token_id
        self.indexer_heads = indexer_heads
        self.indexer_key_dim = indexer_key_dim
        self.indexer_top_k = indexer_top_k
        # PLE hash constants, derived once here the way the conversion exports
        # them: one prime-sized vocab per head, concatenated by running offset.
        # The multipliers just need to be odd and < 2**63 / vocabulary so the
        # int64 products cannot overflow (matching the HF bound).
        head_count = (ngram_size - 1) * heads_per_ngram
        self.ple_head_vocab_sizes = _primes_after(ple_vocab_base - 1, head_count)
        self.ple_head_offsets = list(
            np.concatenate([[0], np.cumsum(self.ple_head_vocab_sizes)[:-1]])
        )
        self.ple_table_rows = int(sum(self.ple_head_vocab_sizes))
        self.ple_multipliers = _hf_layer_multipliers(vocabulary, ngram_size)
        assert self.ple_head_dim * head_count == self.hidden, (
            "ple embedding must concatenate to hidden width"
        )

    @property
    def wide(self) -> int:
        return self.hc_count * self.hidden

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
        return (layer + 1) % self.attention_every == 0


def build_qwen4exp_gguf(
    path: Path | str,
    spec: Qwen4ExpSpec | None = None,
    seed: int = 9,
    *,
    mute_mixer: bool = False,
    mtp: bool = False,
    quantize: dict[str, int] | None = None,
    quantize_in_f32: bool = False,
) -> Qwen4ExpSpec:
    """Write the fixture.

    ``quantize`` maps a tensor-name suffix (``"ffn_down_exps.weight"``) to the
    GGML type it should be stored in; ``spec.tensors`` then holds what those
    bytes decode to, so a reference built from it sees the same weights the
    runtime does. ``quantize_in_f32`` rounds through the same quantizer but
    stores the decoded values as f32: a twin file whose every weight equals
    the quantized one, so the two runs must agree to float-order noise.

    ``mute_mixer`` zeroes the attention and DeltaNet output projections so each
    block contributes only its feed-forward through the gated residual --
    the same attribution trick the dense fixture uses.

    ``mtp`` appends the nextn draft block at index ``spec.layers`` and reports
    ``block_count = spec.layers + 1``, matching the release Q4_K_XL layout
    (the UD-IQ1_S carries no draft block at all). See the MTP section of
    plans/qwen4exp-semantics.md: the draft block is always FULL ATTENTION
    regardless of what ``attention_every`` would say for its index, it carries
    indexer tensors that go unused because the draft runs dense, and it has no
    PLE.
    """
    spec = spec or Qwen4ExpSpec()
    rng = np.random.default_rng(seed)
    tensors: list[tuple[str, tuple[int, ...], np.ndarray]] = []
    muted = {"attn_output.weight", "ssm_out.weight"} if mute_mixer else set()

    def projection(name: str, inputs: int, outputs: int, scale: float = 0.25) -> None:
        # GGUF reports [input, output]; the byte stream is [output, input].
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

    def conv1d(name: str, channels: int, kernel: int, scale: float = 0.3) -> None:
        # GGML stores taps contiguously: payload [channel][tap], shape [tap, channel].
        data = (rng.standard_normal((channels, kernel)) * scale).astype(np.float32)
        tensors.append((name, (kernel, channels), data))

    def hyper_connection(prefix: str, kind: str) -> None:
        # Baked norms: 1.0 == untrained. Down/up/inject small so the sigmoid
        # gates sit near 0.5 and the mean-mix keeps activations bounded.
        vector(f"{prefix}hc_{kind}_norm.weight", spec.wide, value=1.0)
        projection(f"{prefix}hc_{kind}_down.weight", spec.wide, spec.hc_low_rank, scale=0.2)
        projection(f"{prefix}hc_{kind}_up.weight", spec.hc_low_rank, spec.wide, scale=0.2)
        projection(f"{prefix}hc_{kind}_inject.weight", spec.wide, spec.hc_count, scale=0.2)

    projection("token_embd.weight", spec.hidden, spec.vocabulary)
    projection("output.weight", spec.hidden, spec.vocabulary)
    vector("output_hc_norm.weight", spec.wide, value=1.0)
    projection("output_hc_down.weight", spec.wide, spec.hc_low_rank, scale=0.2)
    projection("output_hc_up.weight", spec.hc_low_rank, spec.wide, scale=0.2)
    # The n-gram table: [rows][head_dim] payload, GGUF shape [head_dim, rows].
    table = (rng.standard_normal((spec.ple_table_rows, spec.ple_head_dim)) * 0.1)
    tensors.append((
        "per_layer_token_embd.weight",
        (spec.ple_head_dim, spec.ple_table_rows),
        table.astype(np.float32),
    ))

    for layer in range(spec.layers + (1 if mtp else 0)):
        prefix = f"blk.{layer}."
        # The draft block sits one past the executed stack and is full
        # attention whatever the interval pattern says for its index.
        draft = mtp and layer == spec.layers
        hyper_connection(prefix, "attn")
        hyper_connection(prefix, "ffn")
        if draft or spec.is_attention(layer):
            projection(prefix + "attn_q.weight", spec.hidden, spec.heads * spec.head_dim * 2)
            projection(prefix + "attn_k.weight", spec.hidden, spec.kv_heads * spec.head_dim)
            projection(prefix + "attn_v.weight", spec.hidden, spec.kv_heads * spec.head_dim)
            projection(prefix + "attn_output.weight", spec.heads * spec.head_dim, spec.hidden)
            vector(prefix + "attn_q_norm.weight", spec.head_dim, value=1.0)
            vector(prefix + "attn_k_norm.weight", spec.head_dim, value=1.0)
            # The indexer q/k projections share a base component so the head
            # dots are mostly positive. Independent noise would zero every
            # head's relu for ~25% of blocks, and blocks tied at exactly 0.0
            # make the top-k depend on torch.topk's internal tie order --
            # which no other engine can (or should) replicate. A trained
            # indexer has correlated q/k by construction; iid noise does not.
            base = (rng.standard_normal((spec.indexer_key_dim, spec.hidden))
                    * 0.25).astype(np.float32)
            q_data = np.concatenate([
                base + (rng.standard_normal(base.shape) * 0.08).astype(np.float32)
                for _ in range(spec.indexer_heads)
            ], axis=0)
            tensors.append((prefix + "indexer.q_proj.weight",
                            (spec.hidden, spec.indexer_heads * spec.indexer_key_dim),
                            q_data))
            k_data = base + (rng.standard_normal(base.shape) * 0.08).astype(np.float32)
            tensors.append((prefix + "indexer.k_proj.weight",
                            (spec.hidden, spec.indexer_key_dim), k_data))
            vector(prefix + "indexer.q_norm.weight", spec.indexer_key_dim, value=1.0)
            vector(prefix + "indexer.k_norm.weight", spec.indexer_key_dim, value=1.0)
        else:
            projection(prefix + "attn_qkv.weight", spec.hidden, spec.conv_dim)
            projection(prefix + "attn_gate.weight", spec.hidden, spec.value_dim)
            projection(prefix + "ssm_out.weight", spec.value_dim, spec.hidden)
            projection(prefix + "ssm_alpha.weight", spec.hidden, spec.value_heads)
            projection(prefix + "ssm_beta.weight", spec.hidden, spec.value_heads)
            conv1d(prefix + "ssm_conv1d.weight", spec.conv_dim, spec.conv_kernel)
            vector(prefix + "ssm_dt.bias", spec.value_heads)
            tensors.append((
                prefix + "ssm_a",
                (spec.value_heads,),
                -np.exp(rng.standard_normal(spec.value_heads) * 0.2).astype(np.float32),
            ))
            vector(prefix + "ssm_norm.weight", spec.ssm_head_dim, value=1.0)
        if not draft and layer in spec.ple_layers:
            conv1d(prefix + "ple_conv1d.weight", spec.wide, spec.ple_conv_kernel, scale=0.2)
            projection(prefix + "ple_key.weight", spec.hidden, spec.wide, scale=0.2)
            projection(prefix + "ple_value.weight", spec.hidden, spec.hidden, scale=0.2)
            vector(prefix + "ple_norm_conv.weight", spec.wide, value=1.0)
            vector(prefix + "ple_norm_key.weight", spec.wide, value=1.0)
            vector(prefix + "ple_norm_query.weight", spec.wide, value=1.0)
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
        if draft:
            # hnorm spans the whole stream row (hc*hidden); enorm is one hidden
            # vector broadcast across the streams; eh_proj is [2*hidden ->
            # hidden] applied per stream, embedding half first.
            projection(prefix + "nextn.eh_proj.weight", 2 * spec.hidden, spec.hidden)
            vector(prefix + "nextn.enorm.weight", spec.hidden, value=1.0)
            vector(prefix + "nextn.hnorm.weight", spec.wide, value=1.0)

    compress = [
        4 if spec.is_attention(layer) else 0 for layer in range(spec.layers)
    ]
    metadata = [
        _kv("general.architecture", GGUF_STRING, _string("qwen4exp")),
        _kv("general.name", GGUF_STRING, _string("qwen4exp-fixture")),
        # Like the release files, block_count INCLUDES the draft block.
        _kv("qwen4exp.block_count", GGUF_UINT32,
            struct.pack("<I", spec.layers + (1 if mtp else 0))),
        _kv("qwen4exp.embedding_length", GGUF_UINT32, struct.pack("<I", spec.hidden)),
        _kv("qwen4exp.context_length", GGUF_UINT32, struct.pack("<I", 512)),
        _kv("qwen4exp.attention.head_count", GGUF_UINT32, struct.pack("<I", spec.heads)),
        _kv("qwen4exp.attention.head_count_kv", GGUF_UINT32, struct.pack("<I", spec.kv_heads)),
        _kv("qwen4exp.attention.key_length", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("qwen4exp.attention.value_length", GGUF_UINT32, struct.pack("<I", spec.head_dim)),
        _kv("qwen4exp.rope.dimension_count", GGUF_UINT32, struct.pack("<I", spec.rotary_dim)),
        _kv("qwen4exp.rope.freq_base", GGUF_FLOAT32, struct.pack("<f", 10_000_000.0)),
        _kv("qwen4exp.attention.layer_norm_rms_epsilon", GGUF_FLOAT32, struct.pack("<f", 1e-6)),
        _kv("qwen4exp.full_attention_interval", GGUF_UINT32, struct.pack("<I", spec.attention_every)),
        _kv("qwen4exp.ssm.conv_kernel", GGUF_UINT32, struct.pack("<I", spec.conv_kernel)),
        _kv("qwen4exp.ssm.state_size", GGUF_UINT32, struct.pack("<I", spec.ssm_head_dim)),
        _kv("qwen4exp.ssm.group_count", GGUF_UINT32, struct.pack("<I", spec.key_heads)),
        _kv("qwen4exp.ssm.time_step_rank", GGUF_UINT32, struct.pack("<I", spec.value_heads)),
        _kv("qwen4exp.ssm.inner_size", GGUF_UINT32, struct.pack("<I", spec.value_dim)),
        _kv("qwen4exp.hyper_connection.count", GGUF_UINT32, struct.pack("<I", spec.hc_count)),
        _kv("qwen4exp.hyper_connection.low_rank", GGUF_UINT32, struct.pack("<I", spec.hc_low_rank)),
        _kv("qwen4exp.expert_count", GGUF_UINT32, struct.pack("<I", spec.experts)),
        _kv("qwen4exp.expert_used_count", GGUF_UINT32, struct.pack("<I", spec.experts_used)),
        _kv("qwen4exp.expert_feed_forward_length", GGUF_UINT32,
            struct.pack("<I", spec.expert_intermediate)),
        _kv("qwen4exp.expert_shared_feed_forward_length", GGUF_UINT32,
            struct.pack("<I", spec.expert_intermediate)),
        _kv("qwen4exp.attention.indexer.head_count", GGUF_UINT32,
            struct.pack("<I", spec.indexer_heads)),
        _kv("qwen4exp.attention.indexer.key_length", GGUF_UINT32,
            struct.pack("<I", spec.indexer_key_dim)),
        _kv("qwen4exp.attention.indexer.top_k", GGUF_UINT32,
            struct.pack("<I", spec.indexer_top_k)),
        _kv("qwen4exp.attention.compress_ratios", GGUF_ARRAY, _u32_array(compress)),
        _kv("qwen4exp.ple.layers", GGUF_ARRAY, _u32_array(spec.ple_layers)),
        _kv("qwen4exp.ple.ngram_size", GGUF_UINT32, struct.pack("<I", spec.ngram_size)),
        _kv("qwen4exp.ple.heads_per_ngram", GGUF_UINT32, struct.pack("<I", spec.heads_per_ngram)),
        _kv("qwen4exp.ple.conv_kernel", GGUF_UINT32, struct.pack("<I", spec.ple_conv_kernel)),
        _kv("qwen4exp.ple.eos_token_id", GGUF_UINT32, struct.pack("<I", spec.ple_eos_token_id)),
        _kv("qwen4exp.embedding_length_per_layer_input", GGUF_UINT32,
            struct.pack("<I", spec.ple_head_dim)),
        _kv("qwen4exp.ple.layer_multipliers", GGUF_ARRAY, _u64_array(spec.ple_multipliers)),
        _kv("qwen4exp.ple.head_offsets", GGUF_ARRAY, _u64_array(spec.ple_head_offsets)),
        _kv("qwen4exp.ple.head_vocab_sizes", GGUF_ARRAY, _u64_array(spec.ple_head_vocab_sizes)),
        _kv("tokenizer.ggml.vocab_size", GGUF_UINT32, struct.pack("<I", spec.vocabulary)),
        _kv("general.alignment", GGUF_UINT32, struct.pack("<I", ALIGNMENT)),
    ]
    if mtp:
        # The release MTP files carry this; nothing in the runtime reads it yet
        # (the draft block is found by tensor name), but the fixture should not
        # be the only qwen4exp GGUF in the world that omits it.
        metadata.append(
            _kv("qwen4exp.nextn_predict_layers", GGUF_UINT32, struct.pack("<I", 1))
        )

    infos = bytearray()
    payloads = bytearray()
    decoded_tensors: list[tuple[str, tuple[int, ...], np.ndarray]] = []
    for name, shape, data in tensors:
        offset = len(payloads)
        ggml_type = GGML_F32
        for suffix, wanted in (quantize or {}).items():
            if name.endswith(suffix):
                ggml_type = wanted
        if ggml_type != GGML_F32:
            raw, data = _PACKERS[ggml_type](np.ascontiguousarray(data, dtype=np.float32))
            if quantize_in_f32:
                ggml_type = GGML_F32
                raw = np.ascontiguousarray(data, dtype=np.float32).tobytes()
        else:
            raw = np.ascontiguousarray(data, dtype=np.float32).tobytes()
            assert len(raw) == int(np.prod(shape)) * 4, name
        decoded_tensors.append((name, shape, data))
        infos += _string(name)
        infos += struct.pack("<I", len(shape))
        infos += b"".join(struct.pack("<Q", dim) for dim in shape)
        infos += struct.pack("<IQ", ggml_type, offset)
        payloads += raw
        payloads += b"\0" * ((-len(payloads)) % ALIGNMENT)

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    body = bytearray(header + b"".join(metadata) + bytes(infos))
    body += b"\0" * ((-len(body)) % ALIGNMENT)
    Path(path).write_bytes(bytes(body) + bytes(payloads))
    # The raw payload arrays, for tests that rebuild the reference model from
    # the same weights.
    spec.tensors = {
        name: np.ascontiguousarray(data, dtype=np.float32)
        for name, shape, data in decoded_tensors
    }
    return spec
