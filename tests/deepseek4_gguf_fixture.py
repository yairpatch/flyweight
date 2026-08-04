"""Builds a miniature `deepseek4` GGUF with the real metadata vocabulary.

The published DeepSeek-V4-Flash checkpoint is 104 GB across four shards, so the
metadata contract is pinned here instead: same architecture id, same key names,
same value types and the same per-layer array conventions as the real header,
at a geometry small enough to keep in a test.

Values mirror `unsloth/DeepSeek-V4-Flash-0731-GGUF` UD-IQ3_XXS where they are
structural (256 experts becomes 16, but 6 routed and 1 shared, gating func 4,
weight scale 1.5 and the 0/4/128 compress-ratio cycle are verbatim). Tensor
names follow the reference port's table so a plan built against this fixture
sees the names a real checkpoint carries.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

GGUF_INT32 = 5
GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGML_F32 = 0
GGML_I32 = 26

ALIGNMENT = 32


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, kind: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", kind) + payload


def _uint32_array(values) -> bytes:
    body = b"".join(struct.pack("<I", int(value)) for value in values)
    return struct.pack("<IQ", GGUF_UINT32, len(values)) + body


def _float_array(values) -> bytes:
    body = b"".join(struct.pack("<f", float(value)) for value in values)
    return struct.pack("<IQ", GGUF_FLOAT32, len(values)) + body


def _int32_array(values) -> bytes:
    body = b"".join(struct.pack("<i", int(value)) for value in values)
    return struct.pack("<IQ", GGUF_INT32, len(values)) + body


def _string_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_STRING, len(values)) + b"".join(
        _string(value) for value in values
    )


def _byte_vocabulary(size: int) -> list[str]:
    """GPT-2's byte-to-unicode alphabet, truncated to `size` tokens.

    Bytes that are already printable ASCII map to themselves; the rest are
    lifted into a private range so every byte has a single-character spelling.
    """
    printable = set(
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAD))
        + list(range(0xAE, 0x100))
    )
    alphabet, spare = [], 0
    for byte in range(256):
        if byte in printable:
            alphabet.append(chr(byte))
        else:
            alphabet.append(chr(0x100 + spare))
            spare += 1
    return alphabet[:size]


class DeepSeek4Spec:
    """Geometry of the miniature model, shaped like the real checkpoint."""

    def __init__(
        self,
        *,
        # Proportions are the published model's, divided down. The reference
        # graph relies on some of them -- notably (heads/groups)*head_dim ==
        # hidden, and heads*head_dim == 8*hidden -- so they are preserved
        # rather than picked for convenience.
        hidden: int = 256,
        layers: int = 6,
        vocabulary: int = 256,
        heads: int = 16,
        kv_heads: int = 1,
        kv_lora_rank: int = 128,
        q_lora_rank: int = 128,
        rope_dimension: int = 16,
        experts: int = 16,
        experts_used: int = 6,
        expert_intermediate: int = 64,
        indexer_heads: int = 16,
        indexer_key_length: int = 32,
        indexer_top_k: int = 512,
        output_lora_rank: int = 128,
        output_groups: int = 8,
        hyper_connections: int = 4,
        sinkhorn_iterations: int = 20,
        hash_layers: int = 3,
        # Three trailing entries beyond `layers`, as the real header carries
        # (46 for 43 blocks); the reference reads the first `layers` and
        # ignores the rest.
        extra_compress_ratios: int = 3,
    ):
        self.hidden = hidden
        self.layers = layers
        self.vocabulary = vocabulary
        self.heads = heads
        self.kv_heads = kv_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_dimension = rope_dimension
        self.experts = experts
        self.experts_used = experts_used
        self.expert_intermediate = expert_intermediate
        self.indexer_heads = indexer_heads
        self.indexer_key_length = indexer_key_length
        self.indexer_top_k = indexer_top_k
        self.output_lora_rank = output_lora_rank
        self.output_groups = output_groups
        self.hyper_connections = hyper_connections
        self.sinkhorn_iterations = sinkhorn_iterations
        self.hash_layers = hash_layers
        self.extra_compress_ratios = extra_compress_ratios

    @property
    def compress_ratios(self) -> tuple[int, ...]:
        """Per-layer attention kind: 0 sliding window, 4 CSA, 128 HCA.

        The real checkpoint opens with two sliding-window blocks and then
        alternates 4 and 128.
        """
        cycle = [0, 0] + [4 if index % 2 == 0 else 128 for index in range(self.layers - 2)]
        ratios = tuple(cycle[: self.layers])
        # A negative count truncates instead, for the too-short-array case.
        if self.extra_compress_ratios < 0:
            return ratios[: self.extra_compress_ratios]
        return ratios + (0,) * self.extra_compress_ratios


def build_deepseek4_gguf(
    path: Path | str,
    spec: DeepSeek4Spec | None = None,
    seed: int = 11,
) -> DeepSeek4Spec:
    spec = spec or DeepSeek4Spec()
    rng = np.random.default_rng(seed)
    tensors: list[tuple[str, tuple[int, ...], np.ndarray]] = []

    def projection(name: str, inputs: int, outputs: int) -> None:
        data = (rng.standard_normal((outputs, inputs)) * 0.25).astype(np.float32)
        tensors.append((name, (inputs, outputs), data))

    def vector(name: str, size: int, value: float | None = None) -> None:
        data = (
            np.full(size, value, dtype=np.float32)
            if value is not None
            else (rng.standard_normal(size) * 0.05).astype(np.float32)
        )
        tensors.append((name, (size,), data))

    projection("token_embd.weight", spec.hidden, spec.vocabulary)
    vector("output_norm.weight", spec.hidden, value=1.0)
    projection("output.weight", spec.hidden, spec.vocabulary)
    # The head collapses the streams with a pre-weight only, so its mixer is
    # [hc*hidden, hc] with a single scale.
    vector("output_hc_base.weight", spec.hyper_connections)
    projection("output_hc_fn.weight", spec.hyper_connections * spec.hidden, spec.hyper_connections)
    vector("output_hc_scale.weight", 1)

    head_dim = spec.hidden // spec.heads
    for layer in range(spec.layers):
        prefix = f"blk.{layer}."
        ratios = spec.compress_ratios
        # A deliberately short array still has to produce a well-formed file:
        # the loader, not the fixture, is what should reject it.
        ratio = ratios[layer] if layer < len(ratios) else 0
        vector(prefix + "attn_norm.weight", spec.hidden, value=1.0)
        # Multi-head latent attention, with the names and shape relationships
        # the published checkpoint uses: a low-rank query path, a single KV
        # latent projection shared by every head, a grouped low-rank output
        # projection, and one attention sink per head.
        projection(prefix + "attn_q_a.weight", spec.hidden, spec.q_lora_rank)
        vector(prefix + "attn_q_a_norm.weight", spec.q_lora_rank, value=1.0)
        projection(prefix + "attn_q_b.weight", spec.q_lora_rank, spec.heads * spec.kv_lora_rank)
        projection(prefix + "attn_kv.weight", spec.hidden, spec.kv_lora_rank)
        vector(prefix + "attn_kv_a_norm.weight", spec.kv_lora_rank, value=1.0)
        # The file carries wo_a as [n_head*head_dim/groups, lora_rank*groups],
        # which the loader reshapes to three dimensions. In the published model
        # the first dimension happens to equal n_embd; that is a coincidence of
        # its geometry, not the rule.
        projection(
            prefix + "attn_output_a.weight",
            spec.heads * spec.kv_lora_rank // spec.output_groups,
            spec.output_lora_rank * spec.output_groups,
        )
        projection(
            prefix + "attn_output_b.weight",
            spec.output_groups * spec.output_lora_rank,
            spec.hidden,
        )
        vector(prefix + "attn_sinks.weight", spec.heads)
        # Hyper-connection mixers, one pair per block.
        hc = spec.hyper_connections
        for role in ("hc_attn", "hc_ffn"):
            vector(prefix + f"{role}_base.weight", 2 * hc + hc * hc)
            projection(prefix + f"{role}_fn.weight", hc * spec.hidden, (2 + hc) * hc)
            vector(prefix + f"{role}_scale.weight", 3)
        if ratio:
            # Compressed attention layers carry a token compressor. A 4:1 layer
            # keeps two rows per compressed token, a 128:1 layer one, which is
            # what sets the compressor width.
            coff = 2 if ratio == 4 else 1
            width = coff * spec.kv_lora_rank
            tensors.append((
                prefix + "attn_compressor_ape.weight",
                (width, ratio),
                (rng.standard_normal((ratio, width)) * 0.25).astype(np.float32),
            ))
            projection(prefix + "attn_compressor_kv.weight", spec.hidden, width)
            projection(prefix + "attn_compressor_gate.weight", spec.hidden, width)
            vector(prefix + "attn_compressor_norm.weight", spec.kv_lora_rank, value=1.0)
        if ratio == 4:
            # Only 4:1 layers run the lightning indexer.
            projection(prefix + "indexer.proj.weight", spec.hidden, spec.indexer_heads)
            projection(
                prefix + "indexer.attn_q_b.weight",
                spec.q_lora_rank,
                spec.indexer_heads * spec.indexer_key_length,
            )
            index_width = 2 * spec.indexer_key_length
            tensors.append((
                prefix + "indexer_compressor_ape.weight",
                (index_width, ratio),
                (rng.standard_normal((ratio, index_width)) * 0.25).astype(np.float32),
            ))
            projection(prefix + "indexer_compressor_kv.weight", spec.hidden, index_width)
            projection(prefix + "indexer_compressor_gate.weight", spec.hidden, index_width)
            vector(prefix + "indexer_compressor_norm.weight", spec.indexer_key_length, value=1.0)
        # Sparse MoE with a shared expert, stacked expert-major.
        vector(prefix + "ffn_norm.weight", spec.hidden, value=1.0)
        projection(prefix + "ffn_gate_inp.weight", spec.hidden, spec.experts)
        if layer < spec.hash_layers:
            # Hash layers route by token id through an int32 table instead of
            # the learned router bias the other blocks carry.
            table = rng.integers(
                0, spec.experts, size=(spec.vocabulary, spec.experts_used), dtype=np.int32
            )
            tensors.append((
                prefix + "ffn_gate_tid2eid.weight",
                (spec.experts_used, spec.vocabulary),
                table,
            ))
        else:
            vector(prefix + "exp_probs_b.bias", spec.experts)
        for role, inputs, outputs in (
            ("ffn_gate_exps", spec.hidden, spec.expert_intermediate),
            ("ffn_up_exps", spec.hidden, spec.expert_intermediate),
            ("ffn_down_exps", spec.expert_intermediate, spec.hidden),
        ):
            data = (rng.standard_normal((spec.experts, outputs, inputs)) * 0.25).astype(np.float32)
            tensors.append((prefix + role + ".weight", (inputs, outputs, spec.experts), data))
        for role, inputs, outputs in (
            ("ffn_gate_shexp", spec.hidden, spec.expert_intermediate),
            ("ffn_up_shexp", spec.hidden, spec.expert_intermediate),
            ("ffn_down_shexp", spec.expert_intermediate, spec.hidden),
        ):
            projection(prefix + role + ".weight", inputs, outputs)

    metadata = [
        _kv("general.architecture", GGUF_STRING, _string("deepseek4")),
        _kv("general.name", GGUF_STRING, _string("deepseek4-fixture")),
        _kv("general.alignment", GGUF_UINT32, struct.pack("<I", ALIGNMENT)),
        _kv("deepseek4.block_count", GGUF_UINT32, struct.pack("<I", spec.layers)),
        _kv("deepseek4.embedding_length", GGUF_UINT32, struct.pack("<I", spec.hidden)),
        _kv("deepseek4.context_length", GGUF_UINT32, struct.pack("<I", 1_048_576)),
        _kv("deepseek4.attention.head_count", GGUF_UINT32, struct.pack("<I", spec.heads)),
        _kv("deepseek4.attention.head_count_kv", GGUF_UINT32, struct.pack("<I", spec.kv_heads)),
        _kv("deepseek4.attention.key_length", GGUF_UINT32, struct.pack("<I", spec.kv_lora_rank)),
        _kv("deepseek4.attention.value_length", GGUF_UINT32, struct.pack("<I", spec.kv_lora_rank)),
        _kv("deepseek4.attention.q_lora_rank", GGUF_UINT32, struct.pack("<I", spec.q_lora_rank)),
        _kv(
            "deepseek4.attention.output_lora_rank",
            GGUF_UINT32,
            struct.pack("<I", spec.output_lora_rank),
        ),
        _kv(
            "deepseek4.attention.output_group_count",
            GGUF_UINT32,
            struct.pack("<I", spec.output_groups),
        ),
        _kv("deepseek4.attention.sliding_window", GGUF_UINT32, struct.pack("<I", 128)),
        _kv("deepseek4.rope.dimension_count", GGUF_UINT32, struct.pack("<I", spec.rope_dimension)),
        _kv("deepseek4.rope.freq_base", GGUF_FLOAT32, struct.pack("<f", 10_000.0)),
        _kv("deepseek4.rope.scaling.type", GGUF_STRING, _string("yarn")),
        _kv("deepseek4.rope.scaling.factor", GGUF_FLOAT32, struct.pack("<f", 16.0)),
        _kv(
            "deepseek4.rope.scaling.original_context_length",
            GGUF_UINT32,
            struct.pack("<I", 65_536),
        ),
        _kv("deepseek4.rope.scaling.yarn_beta_fast", GGUF_FLOAT32, struct.pack("<f", 32.0)),
        _kv("deepseek4.rope.scaling.yarn_beta_slow", GGUF_FLOAT32, struct.pack("<f", 1.0)),
        _kv("deepseek4.attention.layer_norm_rms_epsilon", GGUF_FLOAT32, struct.pack("<f", 1e-6)),
        _kv("deepseek4.expert_count", GGUF_UINT32, struct.pack("<I", spec.experts)),
        _kv("deepseek4.expert_used_count", GGUF_UINT32, struct.pack("<I", spec.experts_used)),
        _kv("deepseek4.expert_shared_count", GGUF_UINT32, struct.pack("<I", 1)),
        _kv(
            "deepseek4.expert_feed_forward_length",
            GGUF_UINT32,
            struct.pack("<I", spec.expert_intermediate),
        ),
        _kv("deepseek4.expert_gating_func", GGUF_UINT32, struct.pack("<I", 4)),
        _kv("deepseek4.expert_weights_scale", GGUF_FLOAT32, struct.pack("<f", 1.5)),
        _kv("deepseek4.expert_weights_norm", GGUF_BOOL, struct.pack("<B", 1)),
        _kv(
            "deepseek4.swiglu_clamp_exp",
            GGUF_ARRAY,
            _float_array([10.0] * spec.layers),
        ),
        _kv(
            "deepseek4.swiglu_clamp_shexp",
            GGUF_ARRAY,
            _float_array([10.0] * spec.layers),
        ),
        _kv(
            "deepseek4.attention.indexer.head_count",
            GGUF_UINT32,
            struct.pack("<I", spec.indexer_heads),
        ),
        _kv(
            "deepseek4.attention.indexer.key_length",
            GGUF_UINT32,
            struct.pack("<I", spec.indexer_key_length),
        ),
        _kv(
            "deepseek4.attention.indexer.top_k",
            GGUF_UINT32,
            struct.pack("<I", spec.indexer_top_k),
        ),
        _kv(
            "deepseek4.attention.compress_ratios",
            GGUF_ARRAY,
            _uint32_array(spec.compress_ratios),
        ),
        _kv(
            "deepseek4.attention.compress_rope_freq_base",
            GGUF_FLOAT32,
            struct.pack("<f", 160_000.0),
        ),
        _kv(
            "deepseek4.hyper_connection.count",
            GGUF_UINT32,
            struct.pack("<I", spec.hyper_connections),
        ),
        _kv(
            "deepseek4.hyper_connection.sinkhorn_iterations",
            GGUF_UINT32,
            struct.pack("<I", spec.sinkhorn_iterations),
        ),
        _kv("deepseek4.hyper_connection.epsilon", GGUF_FLOAT32, struct.pack("<f", 1e-6)),
        _kv("deepseek4.hash_layer_count", GGUF_UINT32, struct.pack("<I", spec.hash_layers)),
        _kv("tokenizer.ggml.model", GGUF_STRING, _string("gpt2")),
        _kv("tokenizer.ggml.pre", GGUF_STRING, _string("joyai-llm")),
        _kv("tokenizer.ggml.vocab_size", GGUF_UINT32, struct.pack("<I", spec.vocabulary)),
        # A byte-level vocabulary, so the fixture is a complete model the
        # reference implementation can load and run rather than a metadata
        # skeleton. Every token is one byte in GPT-2's byte-to-unicode
        # encoding, which makes tokenization trivially reversible.
        _kv("tokenizer.ggml.tokens", GGUF_ARRAY, _string_array(_byte_vocabulary(spec.vocabulary))),
        _kv(
            "tokenizer.ggml.token_type",
            GGUF_ARRAY,
            _int32_array([1] * spec.vocabulary),
        ),
        _kv("tokenizer.ggml.merges", GGUF_ARRAY, _string_array([])),
        _kv("tokenizer.ggml.bos_token_id", GGUF_UINT32, struct.pack("<I", 0)),
        _kv("tokenizer.ggml.eos_token_id", GGUF_UINT32, struct.pack("<I", 1)),
        _kv("tokenizer.ggml.add_bos_token", GGUF_BOOL, struct.pack("<B", 0)),
        _kv("tokenizer.ggml.add_eos_token", GGUF_BOOL, struct.pack("<B", 0)),
    ]

    infos = bytearray()
    payloads = bytearray()
    for name, shape, data in tensors:
        infos += _string(name)
        infos += struct.pack("<I", len(shape))
        infos += b"".join(struct.pack("<Q", dimension) for dimension in shape)
        # The hash-layer routing tables are int32 tables, not weights.
        integral = data.dtype == np.int32
        infos += struct.pack("<IQ", GGML_I32 if integral else GGML_F32, len(payloads))
        payloads += np.ascontiguousarray(
            data, dtype=np.int32 if integral else np.float32
        ).tobytes()
        payloads += b"\0" * ((-len(payloads)) % ALIGNMENT)

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    body = bytearray(header + b"".join(metadata) + bytes(infos))
    body += b"\0" * ((-len(body)) % ALIGNMENT)
    Path(path).write_bytes(bytes(body) + bytes(payloads))
    return spec
